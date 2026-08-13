"""
queue_worker.py — Worker process: Redis queue → PeptideWorker → Redis result

Run on each worker machine:
  python3 -m ga.queue_worker [options]

The worker:
  1. Connects to Redis
  2. BLPOP ga:queue  (sleeps in kernel until a job arrives — zero busy loop)
  3. Runs PeptideWorker.evaluate([sequence])
  4. Writes scores back to Redis
  5. Repeats forever

Multiple workers on different machines can point at the same Redis.
Each job_id is claimed atomically by exactly one worker (BLPOP is atomic).

Environment variables (override CLI defaults):
  REDIS_HOST, REDIS_PORT, MMSEQS_BIN, MMSEQS_DB, MMSEQS_DB_IDX, MMSEQS_THREADS
"""

import argparse
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ga.queue  import JobQueue
from ga.worker import PeptideWorker


def main():
    parser = argparse.ArgumentParser(description="GA evaluation queue worker")
    parser.add_argument("--redis-host",   default="localhost",
                        help="Redis host (default: localhost)")
    parser.add_argument("--redis-port",   type=int, default=6379)
    parser.add_argument("--redis-db",     type=int, default=0)
    parser.add_argument("--msa-backend",  default="mmseqs2",
                        choices=["jackhmmer", "mmseqs2"])
    parser.add_argument("--msa-parallel", type=int, default=4,
                        help="Parallel mmseqs2 searches (mmseqs2 only)")
    args = parser.parse_args()

    worker_id = socket.gethostname()
    print(f"[QueueWorker] {worker_id} starting …", flush=True)

    queue  = JobQueue(host=args.redis_host, port=args.redis_port, db=args.redis_db)
    worker = PeptideWorker(msa_backend=args.msa_backend,
                           msa_parallel=args.msa_parallel)

    print(f"[QueueWorker] Ready — listening on "
          f"redis://{args.redis_host}:{args.redis_port}", flush=True)

    while True:
        # BLPOP: process sleeps here until a job arrives (no CPU spin)
        job = queue.claim_job(timeout=30)
        if job is None:
            continue   # 30-s timeout expired, loop and BLPOP again

        job_id, gen_id, sequence = job
        print(f"[QueueWorker] {worker_id} claimed {job_id}  "
              f"gen={gen_id}  seq={sequence[:16]}…", flush=True)

        t0 = time.time()
        try:
            results = worker.evaluate([sequence])
            scores  = results.get(sequence, {})
            elapsed = time.time() - t0
            queue.complete_job(job_id, gen_id, scores, worker_id, elapsed)
            amy = (scores.get("AMY1R", 0) + scores.get("AMY3R", 0)) / 2
            print(f"[QueueWorker] Done  {job_id}  {elapsed:.0f}s  "
                  f"AMY13={amy:.3f}  CTR={scores.get('CTR', 0):.3f}", flush=True)
        except Exception as e:
            elapsed = time.time() - t0
            queue.fail_job(job_id, gen_id, str(e))
            print(f"[QueueWorker] FAIL  {job_id}  {elapsed:.0f}s  {e}", flush=True)


if __name__ == "__main__":
    main()
