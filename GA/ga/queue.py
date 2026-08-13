"""
queue.py — Redis-backed job queue for GA ↔ Worker communication

Redis schema:
  ga:queue                 LIST    pending job_ids  (RPUSH / BLPOP)
  ga:queue:running         ZSET    in-flight job_ids, score = deadline timestamp
  ga:job:{job_id}          HASH    sequence, generation, submitted_at
  ga:result:{job_id}       STRING  JSON {scores, worker_id, elapsed_sec, completed_at}
  ga:gen:{gen_id}:pending  SET     all job_ids for this generation
  ga:gen:{gen_id}:done     SET     completed (or failed) job_ids

Crash recovery:
  Worker crash  — job stays in ga:queue:running past its deadline.
                  wait_all() calls requeue_stale_jobs() every poll cycle:
                  expired entries are moved back to ga:queue for any worker to retry.

  GA crash      — caller must checkpoint population + library.json after each
                  generation; see run_ga.py --resume for the GA side.

Key TTL: 3 days (auto-cleaned).
Failed jobs count as done so GA never blocks forever.
"""

import hashlib
import json
import time
from typing import Optional

try:
    import redis as _redis_lib
except ImportError:
    _redis_lib = None

QUEUE_KEY   = "ga:queue"
RUNNING_KEY = "ga:queue:running"   # ZSET: score = deadline unix timestamp
JOB_TTL     = 86400 * 3            # 3 days
JOB_TIMEOUT = 7200                  # seconds before a claimed job is considered stale


class JobQueue:
    """
    Redis job queue.

    GA side:     submit() → wait_all()
    Worker side: claim_job() → complete_job() / fail_job()
    """

    def __init__(self, host: str = "localhost", port: int = 6379,
                 db: int = 0, password: str | None = None):
        if _redis_lib is None:
            raise ImportError("pip install redis")
        self.r = _redis_lib.Redis(
            host=host, port=port, db=db, password=password,
            decode_responses=True,
        )
        self.r.ping()

    # ── GA side ───────────────────────────────────────────────────────────────

    def submit(self, gen_id: str, sequences: list[str]) -> list[str]:
        """
        Enqueue one job per sequence.
        Returns job_ids in the same order as sequences.
        """
        job_ids: list[str] = []
        pipe = self.r.pipeline()
        now  = time.time()

        for seq in sequences:
            h      = hashlib.md5(f"{seq}{now}".encode()).hexdigest()[:8]
            job_id = f"{gen_id}_{h}"
            job_ids.append(job_id)

            pipe.hset(f"ga:job:{job_id}", mapping={
                "sequence":     seq,
                "generation":   gen_id,
                "submitted_at": str(now),
            })
            pipe.expire(f"ga:job:{job_id}", JOB_TTL)
            pipe.sadd(f"ga:gen:{gen_id}:pending", job_id)
            pipe.expire(f"ga:gen:{gen_id}:pending", JOB_TTL)
            pipe.rpush(QUEUE_KEY, job_id)

        pipe.execute()
        print(f"[Queue] Submitted {len(job_ids)} jobs for {gen_id}", flush=True)
        return job_ids

    def wait_all(
        self,
        gen_id:      str,
        job_ids:     list[str],
        poll_sec:    float = 10.0,
        timeout_sec: float = 86400.0,
    ) -> dict[str, dict[str, float]]:
        """
        Block until every job_id has a result.
        Returns {sequence: {receptor: P(agonist)}}.

        Every poll cycle also calls requeue_stale_jobs() so crashed workers
        don't permanently stall the run.
        """
        pending = set(job_ids)
        t0      = time.time()

        while pending:
            if time.time() - t0 > timeout_sec:
                raise TimeoutError(
                    f"[Queue] Timeout after {timeout_sec/3600:.1f} h: "
                    f"{len(pending)} jobs still pending"
                )

            # recover any jobs whose workers died
            self.requeue_stale_jobs()

            done_set   = self.r.smembers(f"ga:gen:{gen_id}:done")
            newly_done = pending & done_set
            pending   -= newly_done

            if pending:
                print(f"[Queue] {gen_id}: "
                      f"{len(job_ids)-len(pending)}/{len(job_ids)} done, "
                      f"queue depth={self.queue_length()} …", flush=True)
                time.sleep(poll_sec)

        # Collect results
        results: dict[str, dict[str, float]] = {}
        pipe = self.r.pipeline()
        for job_id in job_ids:
            pipe.hget(f"ga:job:{job_id}", "sequence")
            pipe.get(f"ga:result:{job_id}")
        rows = pipe.execute()

        for i, job_id in enumerate(job_ids):
            seq = rows[2 * i] or ""
            raw = rows[2 * i + 1]
            if raw:
                payload = json.loads(raw)
                scores  = payload.get("scores", {})
            else:
                scores = {}
                print(f"[Queue] WARN no result for {job_id}", flush=True)
            results[seq] = scores

        return results

    def requeue_stale_jobs(self) -> int:
        """
        Find jobs in ga:queue:running whose deadline has passed (worker died),
        move them back to ga:queue so another worker can retry.
        Returns number of jobs requeued.
        """
        stale = self.r.zrangebyscore(RUNNING_KEY, 0, time.time())
        if not stale:
            return 0
        pipe = self.r.pipeline()
        pipe.zrem(RUNNING_KEY, *stale)
        for job_id in stale:
            pipe.rpush(QUEUE_KEY, job_id)
        pipe.execute()
        print(f"[Queue] Requeued {len(stale)} stale jobs: {stale}", flush=True)
        return len(stale)

    def queue_length(self) -> int:
        return self.r.llen(QUEUE_KEY)

    def gen_status(self, gen_id: str) -> dict:
        total    = self.r.scard(f"ga:gen:{gen_id}:pending")
        done     = self.r.scard(f"ga:gen:{gen_id}:done")
        in_flight = self.r.zcard(RUNNING_KEY)
        return {"total": total, "done": done, "pending": total - done,
                "in_flight": in_flight}

    # ── Worker side ───────────────────────────────────────────────────────────

    def claim_job(self, timeout: int = 30,
                  job_timeout: int = JOB_TIMEOUT) -> Optional[tuple[str, str, str]]:
        """
        Blocking pop. Returns (job_id, gen_id, sequence) or None on timeout.

        Registers the job in ga:queue:running with a deadline so that if
        this worker dies, wait_all() will eventually requeue it.
        """
        item = self.r.blpop(QUEUE_KEY, timeout=timeout)
        if item is None:
            return None
        _, job_id = item

        # Register in-flight with deadline
        deadline = time.time() + job_timeout
        self.r.zadd(RUNNING_KEY, {job_id: deadline})

        job = self.r.hgetall(f"ga:job:{job_id}")
        return job_id, job.get("generation", ""), job.get("sequence", "")

    def complete_job(self, job_id: str, gen_id: str,
                     scores: dict[str, float],
                     worker_id: str, elapsed_sec: float):
        """Write scores and mark done."""
        payload = json.dumps({
            "scores":       scores,
            "worker_id":    worker_id,
            "elapsed_sec":  round(elapsed_sec, 1),
            "completed_at": time.time(),
        })
        pipe = self.r.pipeline()
        pipe.set(f"ga:result:{job_id}", payload, ex=JOB_TTL)
        pipe.sadd(f"ga:gen:{gen_id}:done", job_id)
        pipe.expire(f"ga:gen:{gen_id}:done", JOB_TTL)
        pipe.zrem(RUNNING_KEY, job_id)   # remove from in-flight tracking
        pipe.execute()

    def fail_job(self, job_id: str, gen_id: str, error: str):
        """Mark failed. Counted as done so GA is not permanently blocked."""
        payload = json.dumps({
            "scores":    {},
            "error":     error,
            "failed_at": time.time(),
        })
        pipe = self.r.pipeline()
        pipe.set(f"ga:result:{job_id}", payload, ex=JOB_TTL)
        pipe.sadd(f"ga:gen:{gen_id}:done", job_id)
        pipe.expire(f"ga:gen:{gen_id}:done", JOB_TTL)
        pipe.zrem(RUNNING_KEY, job_id)
        pipe.execute()


# ── GA oracle wrapper ─────────────────────────────────────────────────────────

class QueueOracle:
    """
    Drop-in oracle_fn for FitnessEvaluator that dispatches through Redis.

    start_gen allows resuming the generation counter after a crash.
    """

    def __init__(self, queue: JobQueue, start_gen: int = 0):
        self._queue = queue
        self._gen   = start_gen

    @property
    def current_gen(self) -> int:
        return self._gen

    def __call__(self, sequences: list[str]) -> dict[str, dict[str, float]]:
        gen_id      = f"g{self._gen:03d}"
        self._gen  += 1
        job_ids     = self._queue.submit(gen_id, sequences)
        return self._queue.wait_all(gen_id, job_ids)
