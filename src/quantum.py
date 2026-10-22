"""A small variational quantum classifier (VQC), simulated with NumPy.

No quantum SDK is required: the state-vector simulator below is vectorised
over a batch of samples, so it is fast enough for a few thousand URLs.

Circuit (n qubits, L layers)
----------------------------
1. Angle encoding: each feature x_i in [0, pi] is loaded with RY(x_i) on qubit i.
2. L variational layers: RY(theta) and RZ(phi) on every qubit followed by a
   ring of CNOTs (entanglement).
3. Readout: <Z> on qubit 0, mapped to P(phishing) = (1 + <Z>) / 2.

Training uses the parameter-shift rule (exact gradients, hardware-compatible)
with the Adam optimiser and binary cross-entropy loss.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-7


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------
def _ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rz(theta: float) -> np.ndarray:
    return np.array(
        [[np.exp(-0.5j * theta), 0], [0, np.exp(0.5j * theta)]], dtype=complex
    )


def _ry_batch(thetas: np.ndarray) -> np.ndarray:
    c, s = np.cos(thetas / 2), np.sin(thetas / 2)
    g = np.zeros((len(thetas), 2, 2), dtype=complex)
    g[:, 0, 0] = c
    g[:, 0, 1] = -s
    g[:, 1, 0] = s
    g[:, 1, 1] = c
    return g


def _apply_gate(state: np.ndarray, gate: np.ndarray, qubit: int, n: int) -> np.ndarray:
    """Apply a 2x2 gate (shared) or a (B,2,2) gate stack to one qubit."""
    b = state.shape[0]
    s = state.reshape(b, 2**qubit, 2, 2 ** (n - qubit - 1))
    if gate.ndim == 2:
        out = np.einsum("ij,bljr->blir", gate, s)
    else:
        out = np.einsum("bij,bljr->blir", gate, s)
    return out.reshape(b, -1)


def _cnot_perm(control: int, target: int, n: int) -> np.ndarray:
    """Index permutation implementing CNOT(control -> target)."""
    idx = np.arange(2**n)
    cbit = (idx >> (n - 1 - control)) & 1
    return np.where(cbit == 1, idx ^ (1 << (n - 1 - target)), idx)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------
class VariationalQuantumClassifier:
    """Binary classifier backed by a simulated parameterised quantum circuit.

    Inputs must already be scaled to [0, pi].
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 3,
        steps: int = 120,
        batch_size: int = 64,
        lr: float = 0.1,
        seed: int = 42,
    ) -> None:
        if n_qubits < 1:
            raise ValueError("n_qubits must be >= 1")
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.steps = steps
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        self.params_: np.ndarray | None = None
        self.loss_history_: list[float] = []

        n = n_qubits
        pairs = [(q, (q + 1) % n) for q in range(n)] if n > 2 else (
            [(0, 1)] if n == 2 else []
        )
        self._perms = [_cnot_perm(c, t, n) for c, t in pairs]
        idx = np.arange(2**n)
        self._z0 = 1.0 - 2.0 * ((idx >> (n - 1)) & 1)  # +1 / -1 for qubit 0

    # -- circuit ----------------------------------------------------------
    def _encode(self, X: np.ndarray) -> np.ndarray:
        n = self.n_qubits
        state = np.zeros((X.shape[0], 2**n), dtype=complex)
        state[:, 0] = 1.0
        for q in range(n):
            state = _apply_gate(state, _ry_batch(X[:, q]), q, n)
        return state

    def _expectation(self, state: np.ndarray, params: np.ndarray) -> np.ndarray:
        """<Z_0> for an already-encoded batch under *params*."""
        n = self.n_qubits
        for layer in range(self.n_layers):
            for q in range(n):
                state = _apply_gate(state, _ry(params[layer, q, 0]), q, n)
                state = _apply_gate(state, _rz(params[layer, q, 1]), q, n)
            for perm in self._perms:
                state = state[:, perm]
        probs = state.real**2 + state.imag**2
        return probs @ self._z0

    # -- training ---------------------------------------------------------
    def _loss_and_grad(self, enc: np.ndarray, y: np.ndarray, params: np.ndarray):
        f = self._expectation(enc, params)
        p = np.clip((1.0 + f) / 2.0, _EPS, 1 - _EPS)
        loss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        dl_dp = (p - y) / (p * (1 - p))

        grad = np.zeros_like(params)
        shift = np.pi / 2
        for idx in np.ndindex(*params.shape):
            plus, minus = params.copy(), params.copy()
            plus[idx] += shift
            minus[idx] -= shift
            df = (self._expectation(enc, plus) - self._expectation(enc, minus)) / 2.0
            grad[idx] = np.mean(dl_dp * 0.5 * df)
        return loss, grad

    def fit(self, X: np.ndarray, y: np.ndarray) -> "VariationalQuantumClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.shape[1] != self.n_qubits:
            raise ValueError(
                f"Expected {self.n_qubits} features (one per qubit), got {X.shape[1]}."
            )
        rng = np.random.default_rng(self.seed)
        params = rng.uniform(-np.pi, np.pi, size=(self.n_layers, self.n_qubits, 2))
        m, v = np.zeros_like(params), np.zeros_like(params)
        b1, b2 = 0.9, 0.999
        self.loss_history_ = []

        for t in range(1, self.steps + 1):
            idx = rng.choice(len(X), size=min(self.batch_size, len(X)), replace=False)
            enc = self._encode(X[idx])
            loss, grad = self._loss_and_grad(enc, y[idx], params)
            m = b1 * m + (1 - b1) * grad
            v = b2 * v + (1 - b2) * grad**2
            params = params - self.lr * (m / (1 - b1**t)) / (
                np.sqrt(v / (1 - b2**t)) + 1e-8
            )
            self.loss_history_.append(loss)

        self.params_ = params
        return self

    # -- inference --------------------------------------------------------
    def predict_proba(self, X: np.ndarray, chunk: int = 20000) -> np.ndarray:
        """Return P(phishing) for each row of *X*."""
        if self.params_ is None:
            raise RuntimeError("Classifier is not fitted yet.")
        X = np.asarray(X, dtype=float)
        out = []
        for i in range(0, len(X), chunk):
            f = self._expectation(self._encode(X[i : i + chunk]), self.params_)
            out.append((1.0 + f) / 2.0)
        return np.clip(np.concatenate(out) if out else np.array([]), 0.0, 1.0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)
