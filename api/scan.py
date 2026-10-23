import json
from http.server import BaseHTTPRequestHandler
from api.index import perform_full_scan

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"
        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception:
            payload = {}
        
        url = payload.get("url", "")
        if not url:
            res_body = json.dumps({"error": "Missing 'url' parameter"}).encode("utf-8")
            self.send_response(400)
        else:
            result = perform_full_scan(url)
            res_body = json.dumps(result, indent=2).encode("utf-8")
            self.send_response(200)

        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(res_body)))
        self.end_headers()
        self.wfile.write(res_body)
