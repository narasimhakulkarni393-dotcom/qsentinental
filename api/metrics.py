import json
from pathlib import Path
from http.server import BaseHTTPRequestHandler
from api.index import ROOT_DIR, FEATURE_NAMES

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        curr_dir = Path(__file__).resolve().parent
        m_paths = [curr_dir / "results" / "metrics.json", ROOT_DIR / "results" / "metrics.json"]
        q_paths = [curr_dir / "results" / "quantum_metrics.json", ROOT_DIR / "results" / "quantum_metrics.json"]
        b_paths = [curr_dir / "results" / "benchmark_comparison.json", ROOT_DIR / "results" / "benchmark_comparison.json"]
        
        c_metrics = {}
        q_metrics = {}
        b_metrics = {}

        for p in m_paths:
            if p.is_file():
                try:
                    c_metrics = json.loads(p.read_text())
                    break
                except Exception:
                    pass

        for p in q_paths:
            if p.is_file():
                try:
                    q_metrics = json.loads(p.read_text())
                    break
                except Exception:
                    pass

        for p in b_paths:
            if p.is_file():
                try:
                    b_metrics = json.loads(p.read_text())
                    break
                except Exception:
                    pass

        res_body = json.dumps({
            "classical": c_metrics,
            "quantum": q_metrics,
            "benchmark": b_metrics,
            "features": FEATURE_NAMES
        }, indent=2).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(res_body)))
        self.end_headers()
        self.wfile.write(res_body)
