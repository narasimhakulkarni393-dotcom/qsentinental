"""Q-Sentinel — ⚛️ Quantum Lab (Streamlit multipage; shows up in the sidebar automatically)."""

import json
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from src.model import load_model, predict_url
from src.quantum_model import load_quantum_model, predict_url_quantum

st.set_page_config(page_title="Q-Sentinel · Quantum Lab", page_icon="⚛️", layout="wide")
st.markdown("# ⚛️ Quantum Lab")
st.markdown("*Variational quantum classifier vs. classical RandomForest*")
st.markdown("---")

q_metrics_path = Path("results") / "quantum_metrics.json"
c_metrics_path = Path("results") / "metrics.json"
q_model_path = Path("models") / "quantum_model.joblib"

if not q_metrics_path.is_file() or not q_model_path.is_file():
    st.warning(
        "⚠️ No quantum model found. Run "
        "`python train_quantum.py --data <your_dataset.csv>` first."
    )
    st.stop()

q = json.loads(q_metrics_path.read_text())
c = json.loads(c_metrics_path.read_text()) if c_metrics_path.is_file() else None

st.info(
    f"The circuit uses **{q['n_qubits']} qubits** and **{q['n_layers']} variational layers**, "
    f"trained on {q['num_quantum_train']} samples for {q['steps']} steps. "
    f"Qubit inputs: `{', '.join(q['features_used'])}`.  The model is simulated "
    "classically with NumPy, so no quantum hardware is required."
)

# --- Metric comparison ------------------------------------------------------
keys = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"),
        ("false_positive_rate", "FPR")]
fig = go.Figure()
fig.add_bar(name="Quantum (VQC)", x=[k[1] for k in keys],
            y=[q[k[0]] for k in keys], marker_color="#b388ff")
if c:
    fig.add_bar(name="Classical (RF)", x=[k[1] for k in keys],
                y=[c[k[0]] for k in keys], marker_color="#00e5ff")
fig.update_layout(barmode="group", height=380, paper_bgcolor="#0e1117",
                  plot_bgcolor="#0e1117", font_color="#e0e0e0")
st.plotly_chart(fig, use_container_width=True)
if not c:
    st.caption("Run `python train.py ...` to add the classical baseline to this chart.")

# --- Training loss ----------------------------------------------------------
st.markdown("### Training loss")
loss = go.Figure(go.Scatter(y=q["loss_history"], mode="lines", line_color="#b388ff"))
loss.update_layout(height=300, xaxis_title="Optimiser step", yaxis_title="BCE loss",
                   paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font_color="#e0e0e0")
st.plotly_chart(loss, use_container_width=True)

# --- Side-by-side scanner ---------------------------------------------------
st.markdown("### Try a URL")
url = st.text_input("URL", placeholder="http://paypa1-verify.com/signin?id=12345")
if st.button("Compare models", type="primary") and url.strip():
    left, right = st.columns(2)
    qr = predict_url_quantum(url, load_quantum_model())
    left.markdown("#### ⚛️ Quantum")
    left.metric(qr["label_text"], f"{qr['probability_phishing']:.1%} phishing")
    if (Path("models") / "model.joblib").is_file():
        clf, scaler = load_model()
        cr = predict_url(url, clf=clf, scaler=scaler)
        right.markdown("#### 🌲 Classical")
        right.metric(cr["label_text"], f"{cr['probability_phishing']:.1%} phishing")
    else:
        right.info("Train the classical model to compare.")
    st.caption("⚠️ Statistical estimates only — not proof of safety.")
