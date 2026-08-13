"""
worker.py — unified peptide evaluation worker

Pipeline:
  MSA threads (CPU)  →  threading.Queue  →  AF3 consumer (GPU)
                                                   ↓
                                             GNN scoring (GPU)

MSA and AF3 run on different resources and overlap in time:
  - While AF3 is running inference on batch k, MSA is already working on batch k+1.
  - AF3 consumer flushes the queue when it reaches `infer_batch_size` JSONs
    or after `infer_flush_sec` seconds with no new arrivals (prevents stalling
    when the last MSA group is smaller than infer_batch_size).

Input:  list[str]  — ligand sequences (1-letter AA or HELM)
Output: dict[str, dict[str, float]]  — {seq: {receptor: P(agonist)}}
"""

import queue
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from ga.oracle import (
    RECEPTORS, OUT_DIR,
    PEPTIDE_PREFIX,
    helm_to_sequence,
    _msa_mmseqs2, _clean_a3m,
    run_msa_batch,
    build_complex_json,
    run_inference_batch,
    gnn_predict_folder,
    _load_gnn,
    _DAEMON_CONTAINER,
)


# ── Module-level MSA helper (top-level for ThreadPool pickling) ───────────────

def _msa_one(seq: str, af3_seq: str | None = None) -> str:
    """Single mmseqs2 search in its own tempdir. Thread-safe.
    seq:     sequence used for mmseqs2 search (plain AA)
    af3_seq: sequence written into the a3m query line (modified residues as X)
    """
    query = af3_seq if af3_seq is not None else seq
    try:
        raw = _msa_mmseqs2(seq)
        return _clean_a3m(raw, query)
    except Exception:
        return f">query\n{query}\n"


# ── Worker ────────────────────────────────────────────────────────────────────

class PeptideWorker:
    """
    Self-contained evaluation worker with MSA–AF3 pipeline parallelism.

    Args:
        msa_backend:      "mmseqs2" (default) or "jackhmmer"
        msa_parallel:     concurrent mmseqs2 searches  (rule of thumb: cpu_count // 4)
        infer_batch_size: complex JSONs per AF3 invocation
                          4 = run AF3 as soon as one sequence's MSA finishes
                          16 = wait for 4 sequences (more GPU-efficient per launch)
        infer_flush_sec:  max seconds the AF3 consumer waits before flushing a
                          partial batch (prevents stalling at end of MSA phase)
    """

    def __init__(
        self,
        msa_backend:      str   = "mmseqs2",
        msa_parallel:     int   = 4,
        infer_batch_size: int   = 16,   # 4 sequences × 4 receptors
        infer_flush_sec:  float = 30.0,
    ):
        self.msa_backend      = msa_backend
        self.msa_parallel     = msa_parallel
        self.infer_batch_size = infer_batch_size
        self.infer_flush_sec  = infer_flush_sec

        print("[Worker] Loading GNN models …", flush=True)
        _load_gnn()
        print(
            f"[Worker] Ready  backend={msa_backend}  "
            f"msa_parallel={msa_parallel}  "
            f"infer_batch={infer_batch_size}  "
            f"flush={infer_flush_sec}s",
            flush=True,
        )

    # ── public API ────────────────────────────────────────────────────────────

    def evaluate(self, sequences: list[str]) -> dict[str, dict[str, float]]:
        """
        Evaluate a batch of ligand sequences with MSA–AF3 pipeline overlap.

        Returns {sequence: {"AMY1R": p, "AMY2R": p, "AMY3R": p, "CTR": p}}.
        Keys match the original input strings (HELM preserved if given).
        """
        if not sequences:
            return {}

        seqs  = [helm_to_sequence(s) if "PEPTIDE" in s else s for s in sequences]
        ts    = int(time.time())
        names = [f"W_{ts}_{i:04d}" for i in range(len(seqs))]

        # shared queue between MSA producer and AF3 consumer
        # sentinel value None signals "producer finished"
        json_q: queue.Queue = queue.Queue()

        # ── MSA producer thread ───────────────────────────────────────────────
        producer_error: list[Exception] = []

        def msa_producer():
            try:
                full_seqs = [PEPTIDE_PREFIX + s for s in seqs]
                # NH2 C-terminal amide → AF3 represents modified position as X
                af3_seqs  = [fs[:-1] + 'X' for fs in full_seqs]
                if self.msa_backend == "jackhmmer":
                    # jackhmmer: batch run, then push all at once
                    msa_map = run_msa_batch(names, full_seqs, backend="jackhmmer")
                    for name, seq in zip(names, seqs):
                        for receptor in RECEPTORS:
                            json_q.put(build_complex_json(
                                name, seq, msa_map[name], receptor))
                else:
                    # mmseqs2: parallel searches; push 4 JSONs per completed sequence
                    with ThreadPoolExecutor(max_workers=self.msa_parallel) as ex:
                        fut_map = {
                            ex.submit(_msa_one, full_seq, af3_seq): (name, seq, af3_seq)
                            for name, seq, full_seq, af3_seq in zip(names, seqs, full_seqs, af3_seqs)
                        }
                        for fut in as_completed(fut_map):
                            name, seq, af3_seq = fut_map[fut]
                            try:
                                msa = fut.result()
                            except Exception as e:
                                print(f"[Worker] MSA failed {name}: {e}", flush=True)
                                msa = f">query\n{af3_seq}\n"
                            print(f"[Worker] MSA done {name} "
                                  f"({msa.count(chr(62))} seqs)", flush=True)
                            for receptor in RECEPTORS:
                                json_q.put(build_complex_json(
                                    name, seq, msa, receptor))
            except Exception as e:
                producer_error.append(e)
            finally:
                json_q.put(None)   # sentinel: producer finished

        # ── AF3 consumer thread ───────────────────────────────────────────────
        consumer_error: list[Exception] = []

        def af3_consumer():
            buf: list[dict] = []
            try:
                while True:
                    try:
                        item = json_q.get(timeout=self.infer_flush_sec)
                    except queue.Empty:
                        # flush partial batch on timeout
                        if buf:
                            print(f"[Worker] AF3 flush (timeout)  "
                                  f"{len(buf)} JSONs", flush=True)
                            run_inference_batch(buf)
                            buf = []
                        continue

                    if item is None:           # sentinel: MSA finished
                        if buf:
                            print(f"[Worker] AF3 flush (final)  "
                                  f"{len(buf)} JSONs", flush=True)
                            run_inference_batch(buf)
                        break

                    buf.append(item)
                    if len(buf) >= self.infer_batch_size:
                        print(f"[Worker] AF3 batch  "
                              f"{len(buf)} JSONs", flush=True)
                        run_inference_batch(buf)
                        buf = []
            except Exception as e:
                consumer_error.append(e)

        # ── Run both threads ──────────────────────────────────────────────────
        t0       = time.time()
        t_prod   = threading.Thread(target=msa_producer, name="msa-producer")
        t_cons   = threading.Thread(target=af3_consumer, name="af3-consumer")
        t_prod.start()
        t_cons.start()
        t_prod.join()
        t_cons.join()

        if producer_error:
            raise RuntimeError(f"MSA producer failed: {producer_error[0]}") from producer_error[0]
        if consumer_error:
            raise RuntimeError(f"AF3 consumer failed: {consumer_error[0]}") from consumer_error[0]

        print(f"[Worker] MSA+AF3 pipeline done in "
              f"{(time.time()-t0)/60:.1f} min", flush=True)

        # ── GNN scoring ───────────────────────────────────────────────────────
        results: dict[str, dict[str, float]] = {}
        for orig, name, seq in zip(sequences, names, seqs):
            scores: dict[str, float] = {}
            for receptor in RECEPTORS:
                folder = OUT_DIR / f"{name}_{receptor}"
                if folder.exists():
                    scores[receptor] = gnn_predict_folder(folder)
                    # Files are owned by Docker root; delete via container
                    docker_path = "/root/outputs/" + folder.name
                    subprocess.run(
                        ["docker", "exec", _DAEMON_CONTAINER, "rm", "-rf", docker_path],
                        check=False,
                    )
                else:
                    scores[receptor] = 0.0
                    print(f"[Worker] WARN missing output: {name}_{receptor}",
                          flush=True)
            results[orig] = scores

        return results
