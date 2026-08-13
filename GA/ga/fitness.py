"""
fitness.py

Fitness evaluation for GA candidates (ε-constraint minimax).

Feasibility: min(AMY1R, AMY2R, AMY3R) >= epsilon
Objective:   minimize max(CTR, CGRP, AM1R, AM2R) among feasible solutions
"""

import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ga.peptide import Peptide

RECEPTORS     = ["AMY1R", "AMY2R", "AMY3R", "CTR", "CGRP", "AM1R", "AM2R"]
AMY_RECEPTORS = ["AMY1R", "AMY2R", "AMY3R"]
OFF_RECEPTORS = ["CTR", "CGRP", "AM1R", "AM2R"]


@dataclass
class FitnessResult:
    sequence:  str
    scores:    dict[str, float]   # {receptor: P(agonist)}
    from_cache: bool  = False

    @property
    def p_amy_min(self) -> float:
        """Bottleneck AMY score: min(AMY1R, AMY2R, AMY3R). Used for feasibility."""
        vals = [self.scores[r] for r in AMY_RECEPTORS if r in self.scores]
        return float(min(vals)) if vals else 0.0

    @property
    def p_amy(self) -> float:
        """Mean P(agonist) across AMY1R, AMY2R, AMY3R. For logging only."""
        vals = [self.scores[r] for r in AMY_RECEPTORS if r in self.scores]
        return float(np.mean(vals)) if vals else 0.0

    @property
    def p_off_max(self) -> float:
        """Worst off-target: max(CTR, CGRP, AM1R, AM2R). Primary objective (minimize)."""
        vals = [self.scores[r] for r in OFF_RECEPTORS if r in self.scores]
        return float(max(vals)) if vals else 0.0

    @property
    def p_ctr(self) -> float:
        return self.scores.get("CTR", 0.0)


class SequenceLibrary:
    """In-memory cache of (sequence → scores) from evaluated peptides."""

    def __init__(self):
        self._db: dict[str, dict[str, float]] = {}

    def add(self, seq: str, scores: dict[str, float]):
        if not all(r in scores for r in RECEPTORS):
            return  # incomplete receptor coverage — would be invalidated on lookup anyway
        self._db[seq] = scores

    def lookup(self, seq: str) -> dict[str, float] | None:
        return self._db.get(seq)

    def __len__(self):
        return len(self._db)

    def save(self, path: Path):
        with open(path, "w") as f:
            json.dump(self._db, f, indent=2)

    def load(self, path: Path):
        with open(path) as f:
            self._db = json.load(f)


# ── Evaluator ─────────────────────────────────────────────────────────────────

class FitnessEvaluator:
    """
    Evaluate fitness for a batch of peptides.
    Cache-first; cache misses go to oracle (AF3+GNN).
    """

    def __init__(
        self,
        library:   SequenceLibrary,
        oracle_fn: Callable[[list[str]], dict[str, dict[str, float]]] | None = None,
    ):
        self.library      = library
        self.oracle_fn    = oracle_fn
        self.oracle_calls = 0

    def evaluate_batch(self, peptides: list[Peptide]) -> list[FitnessResult]:
        results: list[tuple[int, FitnessResult | None] | None] = []
        novel:   list[tuple[int, str]] = []

        for i, p in enumerate(peptides):
            seq = p.to_string()

            scores = self.library.lookup(seq)
            if scores is not None and not all(r in scores for r in RECEPTORS):
                scores = None  # invalidate cache entries missing new receptors
            if scores is not None:
                results.append((i, FitnessResult(seq, scores, from_cache=True)))
                continue

            novel.append((i, seq))
            results.append((i, None))

        if novel and self.oracle_fn is not None:
            novel_seqs    = [seq for _, seq in novel]
            oracle_scores = self.oracle_fn(novel_seqs)
            self.oracle_calls += len(novel_seqs)
            for idx, seq in novel:
                scores = oracle_scores.get(seq, {r: 0.0 for r in RECEPTORS})
                self.library.add(seq, scores)
                results[idx] = (idx, FitnessResult(seq, scores, from_cache=False))

        final = []
        for item in results:
            if item is None or item[1] is None:
                final.append(FitnessResult("", {}, from_cache=False))
            else:
                final.append(item[1])
        return final
