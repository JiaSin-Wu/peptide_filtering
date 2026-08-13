"""
run_ga.py

Minimax peptide optimisation loop.

Constraint: min(AMY1R, AMY2R, AMY3R) >= epsilon  (feasibility)
Objective:  minimize max(CTR, CGRP, AM1R, AM2R)  (among feasible)

Usage:
  python3 -m ga.run_ga --seed_seqs seeds.txt --generations 50 --pop_size 50
  python3 -m ga.run_ga --seed_seqs seeds.txt --msa-backend mmseqs2
"""

import argparse
import csv
import json
import os
import random
import time
import numpy as np
from pathlib import Path

from ga.peptide import Peptide, mutate, crossover, insert, delete
from ga.fitness import FitnessEvaluator, FitnessResult, SequenceLibrary

OUT_DIR = Path(__file__).resolve().parents[1] / "runs"


# ── ε-constraint selection helpers ───────────────────────────────────────────

def epsilon_tournament(results: list[FitnessResult], epsilon: float) -> FitnessResult:
    """Binary tournament under ε-constraint (minimax).
    Feasible (min AMY ≥ ε) always beats infeasible.
    Among feasible: lower p_off_max wins.
    Among infeasible: higher p_amy_min wins (guides toward feasibility).
    """
    if len(results) < 2:
        return results[0]
    a, b = random.sample(results, 2)
    a_ok = a.p_amy_min >= epsilon
    b_ok = b.p_amy_min >= epsilon
    if a_ok and not b_ok:
        return a
    if b_ok and not a_ok:
        return b
    if a_ok and b_ok:
        return a if a.p_off_max <= b.p_off_max else b
    return a if a.p_amy_min >= b.p_amy_min else b


def epsilon_select(combined: list[FitnessResult],
                   pop_size: int, epsilon: float) -> list[FitnessResult]:
    """Fill population: feasible sorted by p_off_max asc, then infeasible by p_amy_min desc."""
    feasible   = sorted([r for r in combined if r.p_amy_min >= epsilon], key=lambda r:  r.p_off_max)
    infeasible = sorted([r for r in combined if r.p_amy_min <  epsilon], key=lambda r: -r.p_amy_min)
    return (feasible + infeasible)[:pop_size]


def make_offspring(cur_results: list[FitnessResult],
                   peptide_map: dict[str, Peptide],
                   pop_size: int,
                   epsilon: float,
                   known_seqs: set[str] | None = None) -> list[Peptide]:
    """Generate pop_size novel offspring via ε-constraint tournament + operators.

    Skips sequences already in known_seqs (library cache) to avoid wasting
    oracle calls on already-evaluated sequences.  Mutation strength escalates
    automatically when the search struggles to find new sequences.
    """
    valid = [r for r in cur_results if r.sequence in peptide_map]
    if not valid:
        return []

    known     = known_seqs or set()
    offspring: list[Peptide] = []
    # insert/delete add length diversity; previously had weight 0
    op_weights = {"mutate": 0.55, "crossover": 0.2, "insert": 0.15, "delete": 0.1}
    max_tries  = pop_size * 60
    tries      = 0

    while len(offspring) < pop_size and tries < max_tries:
        tries += 1
        # Escalate mutation strength when struggling to find novel sequences
        n_mut = 1 + (tries // (pop_size * 10))   # 1 → 2 → 3 → …
        n_mut = min(n_mut, 5)

        op = random.choices(list(op_weights), weights=list(op_weights.values()))[0]

        candidates: list[Peptide] = []
        if op == "crossover":
            r1 = epsilon_tournament(valid, epsilon)
            r2 = epsilon_tournament(valid, epsilon)
            c1, c2 = crossover(peptide_map[r1.sequence], peptide_map[r2.sequence])
            candidates = [c1, c2]
        else:
            r      = epsilon_tournament(valid, epsilon)
            parent = peptide_map[r.sequence]
            if op == "mutate":
                child = mutate(parent, n_mutations=n_mut)
            elif op == "insert":
                child = insert(parent)
            else:
                child = delete(parent)
            candidates = [child]

        for c in candidates:
            seq = c.to_string()
            if c.is_valid() and seq not in known and len(offspring) < pop_size:
                offspring.append(c)
                known.add(seq)   # prevent same novel seq appearing twice in one batch

    return offspring


# ── Checkpoint helpers ────────────────────────────────────────────────────────

LOG_FIELDS = ["gen", "n_feasible",
              "best_p_amy_min", "best_p_off_max",
              "best_amy1r", "best_amy2r", "best_amy3r",
              "best_ctr", "best_cgrp", "best_am1r", "best_am2r",
              "mean_p_amy_min", "mean_p_off_max",
              "oracle_calls", "best_seq"]


def _save_checkpoint(path: Path, next_gen: int, cur_results: list[FitnessResult],
                     peptide_map: dict[str, Peptide],
                     oracle_calls: int, local_seen: set[str], queue_gen: int = 0):
    """Atomically write checkpoint (rename-on-close, crash-safe)."""
    data = {
        "version":         2,
        "generation":      next_gen,        # next generation to run
        "population_helm": [
            peptide_map[r.sequence].to_helm()
            for r in cur_results
            if r.sequence in peptide_map
        ],
        "oracle_calls":    oracle_calls,
        # Every sequence freshly evaluated (non-cache) by this run, not just
        # the current population — lets offspring dedup survive --resume.
        "local_seen":      sorted(local_seen),
        "queue_gen":       queue_gen,
        "saved_at":        time.time(),
    }
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)   # POSIX atomic rename


def _load_checkpoint(path: Path, library: SequenceLibrary):
    """
    Load checkpoint and reconstruct population.
    Scores are read from library (evaluated sequences survive crashes).
    Returns (start_gen, queue_gen, peptide_map, cur_results, oracle_calls, local_seen).
    """
    with open(path) as f:
        chk = json.load(f)
    population = [Peptide.from_helm(h) for h in chk["population_helm"]]
    pmap       = {p.to_string(): p for p in population}
    results    = []
    for p in population:
        seq    = p.to_string()
        scores = library.lookup(seq) or {}
        results.append(FitnessResult(seq, scores, from_cache=bool(scores)))
    # Fallback for checkpoints saved before "local_seen" existed (version 1):
    # best effort is just the current population's non-cache sequences.
    local_seen = set(chk["local_seen"]) if "local_seen" in chk else {
        r.sequence for r in results if not r.from_cache
    }
    return (
        chk["generation"],
        chk.get("queue_gen", chk["generation"]),
        pmap,
        results,
        chk.get("oracle_calls", 0),
        local_seen,
    )


# ── Seed loading ──────────────────────────────────────────────────────────────

def load_seed_sequences(path: Path) -> list[Peptide]:
    seqs = []
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#"):
            p = Peptide.from_string(line)
            if p.is_valid():
                seqs.append(p)
            else:
                print(f"  [SKIP] invalid seed: {line}")
    return seqs


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed_seqs",    required=True, help="Seed sequences (one per line)")
    parser.add_argument("--global-library", default=None,
                        help="Shared library across all runs (read at start, updated each generation)")
    parser.add_argument("--generations",  type=int,   default=50)
    parser.add_argument("--pop_size",     type=int,   default=50)
    parser.add_argument("--epsilon",      type=float, default=0.9,
                        help="min(AMY1R,AMY2R,AMY3R) lower bound constraint (default 0.9)")
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--run_name",     default="ga_run")
    parser.add_argument("--msa-backend",  default="jackhmmer",
                        choices=["jackhmmer", "mmseqs2"],
                        help="MSA backend: jackhmmer (default) or mmseqs2")
    parser.add_argument("--workers", nargs="*", default=None,
                        metavar="URL",
                        help="HTTP worker URLs, e.g. http://host1:8765 http://host2:8765")
    parser.add_argument("--queue-host", default=None,
                        help="Redis host for queue-based distributed evaluation")
    parser.add_argument("--queue-port", type=int, default=6379)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint.json in run_dir")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = OUT_DIR / args.run_name
    run_dir.mkdir(exist_ok=True)

    epsilon = args.epsilon
    library = SequenceLibrary()

    global_library_path = Path(args.global_library) if args.global_library else None

    # Load order: global library (cross-run score cache) → run-specific
    # library (resume; already a superset containing the global entries as
    # of when this run started, since library.save() writes the full db)
    if global_library_path and global_library_path.exists():
        library.load(global_library_path)
        print(f"Loaded global library: {len(library)} sequences")

    library_path = run_dir / "library.json"
    if args.resume and library_path.exists():
        library.load(library_path)
        print(f"Loaded run library: {len(library)} sequences")

    # ── Oracle ────────────────────────────────────────────────────────────────
    msa_backend = args.msa_backend
    checkpoint_path = run_dir / "checkpoint.json"
    queue_gen_start = 0

    if args.resume and checkpoint_path.exists():
        # peek at checkpoint to know queue_gen_start before building oracle
        with open(checkpoint_path) as f:
            _chk_peek = json.load(f)
        queue_gen_start = _chk_peek.get("queue_gen", _chk_peek.get("generation", 0))

    if args.queue_host:
        from ga.queue import JobQueue, QueueOracle
        _queue    = JobQueue(host=args.queue_host, port=args.queue_port)
        oracle_fn = QueueOracle(_queue, start_gen=queue_gen_start)
        print(f"[Oracle] Queue mode → redis://{args.queue_host}:{args.queue_port}")
    elif args.workers:
        from ga.remote_oracle import RemoteOracle
        oracle_fn = RemoteOracle(args.workers)
        print(f"[Oracle] HTTP workers: {len(args.workers)}")
    else:
        try:
            from ga.worker import PeptideWorker
            _worker   = PeptideWorker(msa_backend=msa_backend)
            oracle_fn = _worker.evaluate
            print(f"[Oracle] Local worker (MSA: {msa_backend})")
        except Exception as e:
            oracle_fn = None
            print(f"[WARN] Worker not available ({e}) — cached sequences only.")

    evaluator = FitnessEvaluator(library, oracle_fn=oracle_fn)

    # ── Population init (fresh or resume) ─────────────────────────────────────
    resumed_local_seen: set[str] | None = None
    if args.resume and checkpoint_path.exists():
        start_gen, queue_gen_start, peptide_map, cur_results, prev_oracle_calls, resumed_local_seen = \
            _load_checkpoint(checkpoint_path, library)
        evaluator.oracle_calls = prev_oracle_calls
        print(f"[Resume] Resuming from generation {start_gen} "
              f"({len(cur_results)} peptides, {len(library)} library entries, "
              f"{len(resumed_local_seen)} locally-seen sequences)")
    else:
        seeds = load_seed_sequences(Path(args.seed_seqs))
        print(f"Loaded {len(seeds)} valid seed sequences")
        if not seeds:
            raise ValueError("No valid seed sequences found.")

        population: list[Peptide] = []
        max_tries = args.pop_size * 60
        tries     = 0
        while len(population) < args.pop_size and tries < max_tries:
            tries += 1
            p = random.choice(seeds)
            child = mutate(p, n_mutations=random.randint(1, 3))
            if child.is_valid():
                population.append(child)
        if len(population) < args.pop_size:
            print(f"[WARN] Only generated {len(population)}/{args.pop_size} "
                  f"initial peptides after {max_tries} tries")
        seen: set[str] = set()
        unique_pop = []
        for p in population:
            s = p.to_string()
            if s not in seen:
                seen.add(s)
                unique_pop.append(p)
        population = unique_pop[:args.pop_size]

        peptide_map: dict[str, Peptide] = {p.to_string(): p for p in population}
        print(f"Evaluating initial population ({len(population)} peptides)…")
        cur_results = evaluator.evaluate_batch(population)
        start_gen   = 0

    # Tracks sequences generated fresh in this run (not global library).
    # Used to avoid re-generating locally-seen sequences while still allowing
    # global library sequences to appear as offspring (cache hit, no oracle cost).
    # On resume this is restored from the checkpoint so history from generations
    # already discarded from the population is not forgotten (see _save_checkpoint).
    run_local_seen: set[str] = (
        resumed_local_seen if resumed_local_seen is not None
        else {r.sequence for r in cur_results if not r.from_cache}
    )

    # ── Log CSV (append-safe on resume) ───────────────────────────────────────
    log_path   = run_dir / "log.csv"
    log_exists = log_path.exists() and args.resume
    if not log_exists:
        with open(log_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=LOG_FIELDS).writeheader()

    # ── Main loop ─────────────────────────────────────────────────────────────
    for gen in range(start_gen, start_gen + args.generations):
        feasible = [r for r in cur_results if r.p_amy_min >= epsilon]
        best_feasible = sorted(feasible, key=lambda r: r.p_off_max)
        best = (best_feasible[0] if best_feasible
                else max(cur_results, key=lambda r: r.p_amy_min))
        mean_p_amy_min = float(np.mean([r.p_amy_min for r in feasible])) if feasible else 0.0
        mean_p_off_max = float(np.mean([r.p_off_max for r in feasible])) if feasible else 0.0

        print(f"Gen {gen:3d} | feasible={len(feasible):3d} "
              f"best_amy_min={best.p_amy_min:.3f} best_off_max={best.p_off_max:.3f} "
              f"[CTR={best.p_ctr:.3f} CGRP={best.scores.get('CGRP',0):.3f} "
              f"AM1R={best.scores.get('AM1R',0):.3f} AM2R={best.scores.get('AM2R',0):.3f}] "
              f"oracle={evaluator.oracle_calls} seq={best.sequence[:18]}…")

        row = {
            "gen":            gen,
            "n_feasible":     len(feasible),
            "best_p_amy_min": best.p_amy_min,
            "best_p_off_max": best.p_off_max,
            "best_amy1r":     best.scores.get("AMY1R", 0.0),
            "best_amy2r":     best.scores.get("AMY2R", 0.0),
            "best_amy3r":     best.scores.get("AMY3R", 0.0),
            "best_ctr":       best.scores.get("CTR",   0.0),
            "best_cgrp":      best.scores.get("CGRP",  0.0),
            "best_am1r":      best.scores.get("AM1R",  0.0),
            "best_am2r":      best.scores.get("AM2R",  0.0),
            "mean_p_amy_min": mean_p_amy_min,
            "mean_p_off_max": mean_p_off_max,
            "oracle_calls":   evaluator.oracle_calls,
            "best_seq":       best.sequence,
        }
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=LOG_FIELDS).writerow(row)

        # Evolve — only exclude sequences generated fresh in this run;
        # global library sequences are allowed (score is cached, no oracle cost)
        offspring   = make_offspring(cur_results, peptide_map, args.pop_size,
                                     epsilon, known_seqs=run_local_seen)
        off_results = evaluator.evaluate_batch(offspring)
        for p, r in zip(offspring, off_results):
            peptide_map[p.to_string()] = p
            if not r.from_cache:
                run_local_seen.add(r.sequence)

        combined    = list({r.sequence: r for r in cur_results + off_results}.values())
        cur_results = epsilon_select(combined, args.pop_size, epsilon)

        # ── Checkpoint (atomic, crash-safe) ───────────────────────────────────
        q_gen = oracle_fn.current_gen if hasattr(oracle_fn, "current_gen") else 0
        _save_checkpoint(checkpoint_path, gen + 1, cur_results,
                         peptide_map, evaluator.oracle_calls, run_local_seen, queue_gen=q_gen)

        # Library: atomic write so crash here doesn't corrupt it
        tmp_lib = library_path.with_suffix(".tmp")
        library.save(tmp_lib)
        os.replace(tmp_lib, library_path)

        if global_library_path:
            tmp_glob = global_library_path.with_suffix(".tmp")
            library.save(tmp_glob)
            os.replace(tmp_glob, global_library_path)

    # ── Final output ──────────────────────────────────────────────────────────
    # Global Pareto front from all evaluated sequences (library)
    all_results = []
    for seq, scores in library._db.items():
        if not scores:
            continue
        all_results.append(FitnessResult(seq, scores, from_cache=True))

    def _dominated(a, others):
        for b in others:
            if b is a:
                continue
            if b.p_amy_min >= a.p_amy_min and b.p_off_max <= a.p_off_max:
                if b.p_amy_min > a.p_amy_min or b.p_off_max < a.p_off_max:
                    return True
        return False

    final_pareto = sorted(
        [r for r in all_results if not _dominated(r, all_results)],
        key=lambda r: (-r.p_amy_min, r.p_off_max),
    )

    with open(run_dir / "pareto.json", "w") as f:
        json.dump(
            [{"sequence":    r.sequence,
              "p_amy_min":   r.p_amy_min,
              "p_off_max":   r.p_off_max,
              "scores":      r.scores,
              "helm":        peptide_map[r.sequence].to_helm()}
             for r in final_pareto if r.sequence in peptide_map],
            f, indent=2,
        )
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"\nFinal Pareto front ({len(final_pareto)} solutions):")
    for r in final_pareto[:10]:
        print(f"  amy_min={r.p_amy_min:.3f}  off_max={r.p_off_max:.3f}  "
              f"[CTR={r.p_ctr:.3f} CGRP={r.scores.get('CGRP',0):.3f} "
              f"AM1R={r.scores.get('AM1R',0):.3f} AM2R={r.scores.get('AM2R',0):.3f}]  "
              f"{r.sequence[:24]}…")
    print(f"Results saved to {run_dir}/")


if __name__ == "__main__":
    main()
