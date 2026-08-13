"""
eval_server.py — per-machine evaluation server

每台 worker 機器啟動一個 eval_server，GA 透過 HTTP 把 sequences 派給它，
它跑完 MSA + AF3 + GNN 後把 probabilities 回傳。

啟動：
  python3 -m ga.eval_server [--port 8765] [--msa-backend mmseqs2]

依賴：標準庫 only（http.server + json）。不需要 fastapi / ray。
"""

import argparse
import json
import sys
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# ── Handler ───────────────────────────────────────────────────────────────────

class EvalHandler(BaseHTTPRequestHandler):
    oracle = None   # set by main() before serving

    def do_GET(self):
        if self.path == "/healthz":
            self._respond(200, {"status": "ok"})
        else:
            self._respond(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/evaluate":
            self._respond(404, {"error": "not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            body   = json.loads(self.rfile.read(length))
            seqs   = body["sequences"]
            print(f"[Server] evaluate {len(seqs)} sequences", flush=True)
            result = EvalHandler.oracle(seqs)
            self._respond(200, result)
        except Exception:
            tb = traceback.format_exc()
            print(f"[Server] ERROR:\n{tb}", file=sys.stderr, flush=True)
            self._respond(500, {"error": tb})

    def _respond(self, code: int, payload: dict):
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        # Suppress default access log; keep our own print()s
        pass


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",        type=int, default=8765)
    parser.add_argument("--host",        default="0.0.0.0")
    parser.add_argument("--msa-backend", default="mmseqs2",
                        choices=["jackhmmer", "mmseqs2"])
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from ga.worker import PeptideWorker

    print(f"[Server] Starting worker (MSA backend: {args.msa_backend})...", flush=True)
    worker = PeptideWorker(msa_backend=args.msa_backend)
    EvalHandler.oracle = worker.evaluate   # callable: list[str] → dict
    print(f"[Server] Ready on {args.host}:{args.port}", flush=True)

    server = HTTPServer((args.host, args.port), EvalHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[Server] Shutting down.")


if __name__ == "__main__":
    main()
