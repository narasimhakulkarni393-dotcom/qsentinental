import json
from http.server import BaseHTTPRequestHandler
from api.index import get_models, IMPORT_ERROR

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        clf, q_b = get_models()
        data = {
            "status": "healthy",
            "service": "Q-Sentinel Threat Engine",
            "version": "1.2.0",
            "import_error": IMPORT_ERROR,
            "models": {
                "classical_rf": clf is not None and clf is not False,
                "quantum_vqc": q_b is not None and q_b is not False
            }
        }
        res_body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(res_body)))
        self.end_headers()
        self.wfile.write(res_body)
