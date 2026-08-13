"""
peptide.py

HELM-format peptide representation and genetic operators.

Monomer representation:
  - Standard: single uppercase letter  (A, C, D, ...)
  - Terminal mods: [am] (C-term amide), [ac] (N-term acetyl) — NOT mutatable

Restrictions for GA:
  - Core length (excluding [am]/[ac]): LEN_MIN–LEN_MAX
"""

import re
import random
from dataclasses import dataclass, field

# ── Monomer alphabet ──────────────────────────────────────────────────────────

STD_AA = list("ACDEFGHIKLMNPQRSTVWY")

ALL_MONOMERS = STD_AA
MONOMER_SET  = set(ALL_MONOMERS)

# Terminal modifiers — fixed, never touched by operators
TERMINAL_MODS = {"[am]", "[ac]", "[Glp]"}

LEN_MIN, LEN_MAX = 24, 24


# ── HELM utilities ────────────────────────────────────────────────────────────

def parse_helm_monomers(helm: str) -> list[str]:
    """
    Tokenise a HELM string into an ordered monomer list.
    Handles both standard (single-letter) and non-standard ([XYZ]) monomers.
    Returns the raw tokens including any [am]/[ac] at the termini.
    """
    m = re.search(r'PEPTIDE\d+\{([^}]+)\}', helm)
    if not m:
        raise ValueError(f"Cannot parse HELM: {helm[:80]}")
    return [t.strip() for t in m.group(1).split(".") if t.strip()]


# ── Peptide dataclass ─────────────────────────────────────────────────────────

@dataclass
class Peptide:
    """
    Core monomer list (without terminal [am]/[ac]).
    Terminal amide is always added in to_helm(); N-terminal acetyl is optional.
    """
    sequence:  list[str]       # ordered monomers, no terminal mods
    n_acetyl:  bool = False    # add [ac] at N-terminus in HELM output

    def __post_init__(self):
        self.sequence = list(self.sequence)

    # ── constructors ─────────────────────────────────────────────────────────

    @classmethod
    def from_helm(cls, helm: str) -> "Peptide":
        """Parse from HELM string; strips [am]/[ac] from stored sequence."""
        tokens = parse_helm_monomers(helm)
        n_ac = bool(tokens and tokens[0] == "[ac]")
        if n_ac:
            tokens = tokens[1:]
        if tokens and tokens[-1] == "[am]":
            tokens = tokens[:-1]
        return cls(tokens, n_acetyl=n_ac)

    @classmethod
    def from_string(cls, seq: str) -> "Peptide":
        """Plain uppercase letter sequence (standard AA only)."""
        return cls(list(seq.upper()))

    # ── serialisation ─────────────────────────────────────────────────────────

    def to_string(self) -> str:
        """Join monomers with no separator; [XYZ] shown as-is."""
        return "".join(self.sequence)

    def to_helm(self) -> str:
        """
        Emit canonical HELM with C-terminal [am] monomer,
        matching the format used in ml_label_final.csv.
        """
        monomers = list(self.sequence)
        if self.n_acetyl:
            monomers = ["[ac]"] + monomers
        monomers.append("[am]")
        inner = ".".join(monomers)
        return f"PEPTIDE1{{{inner}}}$$$$"

    # ── validation ────────────────────────────────────────────────────────────

    def is_valid(self) -> bool:
        n = len(self.sequence)
        if not (LEN_MIN <= n <= LEN_MAX):
            return False
        if not all(m in MONOMER_SET for m in self.sequence):
            return False
        return True

    # ── dunder ───────────────────────────────────────────────────────────────

    def __len__(self):
        return len(self.sequence)

    def __hash__(self):
        return hash(self.to_string())

    def __eq__(self, other):
        return isinstance(other, Peptide) and self.sequence == other.sequence


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pick_monomer(exclude: str | None = None) -> str:
    """Random monomer from ALL_MONOMERS, optionally excluding current value."""
    pool = [m for m in ALL_MONOMERS if m != exclude]
    return random.choice(pool)


# ── Genetic operators ─────────────────────────────────────────────────────────

def mutate(p: Peptide, n_mutations: int = 1) -> Peptide:
    """
    Point substitution at n_mutations free positions.
    Replacement is drawn from ALL_MONOMERS.
    """
    seq  = p.sequence.copy()
    free = list(range(len(seq)))
    if not free:
        return Peptide(seq, p.n_acetyl)
    for pos in random.sample(free, min(n_mutations, len(free))):
        seq[pos] = _pick_monomer(exclude=seq[pos])
    return Peptide(seq, p.n_acetyl)


def crossover(p1: Peptide, p2: Peptide) -> tuple["Peptide", "Peptide"]:
    """
    Two-point crossover when parents have equal length;
    single-point crossover when lengths differ (children swap the suffix).
    """
    n1, n2 = len(p1), len(p2)

    if n1 == n2:
        if n1 < 4:
            return p1, p2
        i, j = sorted(random.sample(range(1, n1), 2))
        s1 = p1.sequence[:i] + p2.sequence[i:j] + p1.sequence[j:]
        s2 = p2.sequence[:i] + p1.sequence[i:j] + p2.sequence[j:]
    else:
        # single-point: cut at a position valid for both parents
        cut = random.randint(1, min(n1, n2) - 1)
        s1 = p1.sequence[:cut] + p2.sequence[cut:]
        s2 = p2.sequence[:cut] + p1.sequence[cut:]

    return Peptide(s1, p1.n_acetyl), Peptide(s2, p2.n_acetyl)


def insert(p: Peptide) -> Peptide:
    """Insert a random monomer at a free position (if length < LEN_MAX)."""
    if len(p) >= LEN_MAX:
        return p
    seq  = p.sequence.copy()
    free = list(range(len(seq)))
    pos  = random.choice(free) if free else random.randint(0, len(seq))
    seq.insert(pos, _pick_monomer())
    result = Peptide(seq, p.n_acetyl)
    return result if result.is_valid() else p


def delete(p: Peptide) -> Peptide:
    """Delete a random non-fixed monomer (if length > LEN_MIN)."""
    if len(p) <= LEN_MIN:
        return p
    free = list(range(len(p.sequence)))
    if not free:
        return p
    seq = p.sequence.copy()
    seq.pop(random.choice(free))
    result = Peptide(seq, p.n_acetyl)
    return result if result.is_valid() else p


def apply_operator(p: Peptide, weights: dict | None = None) -> Peptide:
    """Apply one randomly chosen genetic operator."""
    if weights is None:
        weights = {"mutate": 0.6, "insert": 0.2, "delete": 0.2}
    op = random.choices(list(weights.keys()), weights=list(weights.values()))[0]
    if op == "mutate":
        return mutate(p)
    if op == "insert":
        return insert(p)
    if op == "delete":
        return delete(p)
    return p
