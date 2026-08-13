"""
remote_oracle.py — GA oracle that fans out to eval_server instances

Usage:
  oracle = RemoteOracle(["http://192.168.1.10:8765", "http://192.168.1.11:8765"])
  scores = oracle(["KCNTATCATQ...", ...])
  # → {seq: {receptor: prob}}

Sequences are split round-robin across workers; all calls are parallel.
If a worker fails, its sequences fall back to 0.0 with a warning.
"""

import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


class RemoteOracle:
    """
    Fan-out oracle: distributes sequences across eval_server workers.

    Args:
        worker_urls: list of "http://host:port" strings
        timeout:     per-request timeout in seconds (default 2 h)
    """

    def __init__(self, worker_urls: list[str], timeout: int = 7200):
        if not worker_urls:
            raise ValueError("RemoteOracle requires at least one worker URL.")
        self.workers = worker_urls
        self.timeout = timeout
        print(f"[RemoteOracle] {len(self.workers)} worker(s): {self.workers}")

    # ── health check ──────────────────────────────────────────────────────────

    def check_workers(self) -> dict[str, bool]:
        """Ping each worker /healthz. Returns {url: alive}."""
        results = {}
        for url in self.workers:
            try:
                with urllib.request.urlopen(f"{url}/healthz", timeout=5) as r:
                    results[url] = r.status == 200
            except Exception:
                results[url] = False
        return results

    # ── main callable ─────────────────────────────────────────────────────────

    def __call__(self, sequences: list[str]) -> dict[str, dict[str, float]]:
        if not sequences:
            return {}

        n = len(self.workers)
        # Round-robin split: worker i gets sequences[i], sequences[i+n], ...
        chunks = [sequences[i::n] for i in range(n)]

        t0 = time.time()
        results: dict[str, dict[str, float]] = {}

        with ThreadPoolExecutor(max_workers=n) as ex:
            futures = {
                ex.submit(self._call_worker, url, chunk): (url, chunk)
                for url, chunk in zip(self.workers, chunks)
                if chunk
            }
            for fut in as_completed(futures):
                url, chunk = futures[fut]
                try:
                    results.update(fut.result())
                except Exception as e:
                    print(f"[RemoteOracle] WARN worker {url} failed: {e}")
                    # fallback: all-zeros for this worker's sequences
                    for seq in chunk:
                        results[seq] = {"AMY1R": 0.0, "AMY2R": 0.0,
                                        "AMY3R": 0.0, "CTR": 0.0}

        print(f"[RemoteOracle] {len(sequences)} sequences done in "
              f"{(time.time()-t0)/60:.1f} min")
        return results

    # ── internals ─────────────────────────────────────────────────────────────

    def _call_worker(self, url: str, sequences: list[str]) -> dict:
        payload = json.dumps({"sequences": sequences}).encode()
        req = urllib.request.Request(
            f"{url}/evaluate", data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())
