import sys, pathlib

# Ensure the package can locate sibling modules when imported from the repository root.
_project_root = pathlib.Path(__file__).resolve().parent.parent
_src_path = _project_root / 'src'
if _src_path.is_dir():
    sys.path.insert(0, str(_src_path))

"""Q-Sentinel: Hybrid Quantum-Classical AI for Phishing URL Detection."""
