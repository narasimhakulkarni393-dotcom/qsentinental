import json
from pathlib import Path
from http.server import BaseHTTPRequestHandler
from api.index import ROOT_DIR, FEATURE_NAMES

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        metrics_path = ROOT_DIR / "results" / "metrics.json"
        q_metrics_path = ROOT_DIR / "results" / "quantum_metrics.json"
        
        c_metrics = {}
        q_metrics = {}
        if metrics_path.is_file():
            try:
                c_metrics = json.loads(metrics_path.read_text())
            except Exception:
                pass
        if q_metrics_path.is_file():
            try:
                q_metrics = json.loads(q_metrics_path.read_text())
            except Exception:
                pass

        res_body = json.dumps({
            "classical": c_metrics,
            "quantum": q_metrics,
            "features": FEATURE_NAMES
        }, indent=2).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(res_body)))
        self.end_headers()
        self.wfile.write(res_body)
