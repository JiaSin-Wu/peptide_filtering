"""
filter_lib.py

Shared library for the 3_Filter pipeline: candidate-id-list I/O, the raw
per-stage output loaders, and the per-stage gate functions. Used by
run_pipeline.py (a real sequential funnel, in whatever --order is given).

Gate-split rationale: the final rollup is
    final_pass = amy_pose_pass AND safety_pass
where amy_pose_pass only needs 01_pose_check's data and safety_pass is
tango_pass(03) AND aller_pass(04) AND tox3_pass(05). Boolean AND is
order-independent, so applying the four per-stage gates (gate_amy_pose=01,
gate_tango=03, gate_aller=04, gate_tox3=05) one at a time as a funnel is
equivalent to the single-shot AND, regardless of the order the stages run in.
01 still scores all 7 receptors and records the off-target poses, but only the
3 AMY receptors gate. See README.md for the current workflow and caveats.

(An off-target Rosetta energy stage, 02, existed earlier and was removed: it was
never run for the panel and played no part in the funnel. It lives in git
history. The rollup that excluded it, once called final_pass_no_energy, is now
simply final_pass.)

2026-08-29 -- GA_006 was re-folded with AF3 for all 7 receptors (real
peptide MSA this time -- 37aa query, jackhmmer against the full local
databases, padded to the 38aa folding sequence -- not the empty-MSA
placeholder used in an earlier single-receptor test), producing new
structures at structures/GA_006/<receptor>/GA_006_<receptor>_model.cif.
The canonical on-target pose result is now the sequence-verified 21-row
AMY1R/AMY2R/AMY3R table in 01_pose_check/outputs/results_from_md_af3.csv.
"""

import csv
import glob
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent

POSE_CSV    = HERE / "01_pose_check" / "outputs" / "results_from_md_af3.csv"
TANGO_CSV   = HERE / "03_aggregation" / "outputs" / "tango_results.csv"
TOX3_CSV    = HERE / "05_toxicity" / "outputs" / "toxinpred3_raw_ga.csv"
ALLER_GLOB  = str(HERE / "04_allergenicity" / "outputs" / "AllerCatPro2_prediction_*.csv")
SEQ_CSV     = HERE / "inputs" / "sequences_ga.csv"
IMM_CSV     = HERE / "06_immunogenicity" / "outputs" / "netmhciipan_sb_summary.csv"
CD4_CSV     = HERE / "06_immunogenicity" / "outputs" / "cd4episcore_summary.csv"

RECEPTORS = ["AMY1R", "AMY2R", "AMY3R", "CTR", "CGRP", "AM1R", "AM2R"]
AMY_RECS  = ["AMY1R", "AMY2R", "AMY3R"]
AMYLIN_ID = "amylin"   # reference peptide, pose-checked alongside the candidates (stage 01)


# ── id-list I/O (plain text, one id per line) ──────────────────────────────

def read_ids(path: Path) -> list[str]:
    return [l.strip() for l in Path(path).read_text().splitlines() if l.strip()]


def write_ids(path: Path, ids: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(sorted(ids)) + "\n")


def resolve_ids(all_ids: list[str] | None, seq: str | None, seq_file: str | None,
                 extra_id: str | None = None) -> list[str]:
    """Common CLI id-resolution shared by every stage script.

    - If `all_ids` is given, --seq/--seq-file *filter* that known universe
      down to the requested subset (01/03/05's case: there's a natural
      "everything" default when neither flag is passed).
    - If `all_ids` is None, --seq/--seq-file *are* the id list directly, no
      filtering (no cheap "everything" default, an explicit subset is
      required).
    - --extra-id always appends ids (e.g. amylin) regardless of the above.
    """
    if seq_file:
        wanted = read_ids(Path(seq_file))
    elif seq:
        wanted = [s.strip() for s in seq.split(",") if s.strip()]
    else:
        wanted = None

    if wanted is not None:
        ids = [i for i in all_ids if i in set(wanted)] if all_ids is not None else list(wanted)
    else:
        ids = list(all_ids) if all_ids is not None else []

    if extra_id:
        for e in extra_id.split(","):
            e = e.strip()
            if e and e not in ids:
                ids.append(e)
    return ids


def apply_gate(alive_ids: list[str], gate_result: dict[str, bool]) -> tuple[list[str], list[str]]:
    """Split alive_ids into (survivors, eliminated). An id missing from
    gate_result (no data yet for it) is a conservative fail, matching the
    `.get(id, False)` convention every loader below already uses.
    """
    survivors  = [i for i in alive_ids if gate_result.get(i, False)]
    eliminated = [i for i in alive_ids if not gate_result.get(i, False)]
    return survivors, eliminated


# ── raw-output loaders ──────────────────────────────────────────────────────

def load_pose() -> dict:
    d = defaultdict(dict)
    for r in csv.DictReader(open(POSE_CSV)):
        d[r["id"]][r["receptor"]] = int(r["pose_pass"])
    return d


def load_tox3() -> dict:
    """ToxinPred3: Non-Toxin = PASS."""
    d = {}
    for r in csv.DictReader(open(TOX3_CSV)):
        sid = r.get("Subject") or r.get("id", "")
        d[sid] = r.get("Prediction", "") == "Non-Toxin"
    return d


def load_allercatpro() -> dict:
    """AllerCatPro2: 'no evidence' = PASS. Globs every batch file, so a new
    subset batch (see run_pipeline.py's 04 pause/resume) merges in for free
    without touching this loader.
    """
    d = {}
    for path in glob.glob(ALLER_GLOB):
        for r in csv.DictReader(open(path)):
            sid = r.get("Protein", "")
            if sid:
                d[sid] = r.get("Result", "") == "no evidence"
    return d


def load_immunogenicity() -> dict:
    d = {}
    if not IMM_CSV.exists():
        return d
    for r in csv.DictReader(open(IMM_CSV)):
        d[r["seq_id"]] = r
    return d


def load_cd4episcore() -> dict:
    """CD4episcore Combined Score summary from
    06_immunogenicity/integrate_cd4episcore.py. A candidate absent from this
    file has zero rank<10% MHC-II binder peptides -- trivially cd4_pass=1,
    nothing for CD4episcore to flag. A candidate present with status
    "incomplete" has binder peptides that haven't been submitted to
    CD4episcore yet -- reference/ranking only, not gating, so
    build_report_rows() reports it as an empty cd4_pass rather than
    guessing.
    """
    d = {}
    if not CD4_CSV.exists():
        return d
    for r in csv.DictReader(open(CD4_CSV)):
        d[r["id"]] = r
    return d


# ── gate functions (one boolean per candidate) ──────────────────────────────

def load_tango() -> dict:
    """03 aggregation, straight from run_tango.py's canonical output."""
    return {r["id"]: r["TANGO_pass"] == "PASS" for r in csv.DictReader(open(TANGO_CSV))}


def gate_tango(ids: list[str], tango: dict | None = None) -> dict[str, bool]:
    """03 aggregation. Reads 03_aggregation/outputs/tango_results.csv."""
    tango = tango if tango is not None else load_tango()
    return {sid: tango.get(sid, False) for sid in ids}


def gate_aller(ids: list[str], aller: dict | None = None) -> dict[str, bool]:
    aller = aller if aller is not None else load_allercatpro()
    return {sid: aller.get(sid, False) for sid in ids}


def gate_tox3(ids: list[str], tox3: dict | None = None) -> dict[str, bool]:
    tox3 = tox3 if tox3 is not None else load_tox3()
    return {sid: tox3.get(sid, False) for sid in ids}


def gate_amy_pose(ids: list[str], pose: dict | None = None) -> dict[str, bool]:
    """01 pose_check. Only gates on the 3 AMY receptors -- off-target
    pose_pass values are still computed and recorded by run_pose_check.py
    for all 7 receptors, just not used to eliminate here. A candidate that
    fails to dock an off-target receptor is *safer* there, not worse, so it
    can't be a 01-stage rejection reason.
    """
    pose = pose if pose is not None else load_pose()
    return {sid: all(pose.get(sid, {}).get(r, 0) == 1 for r in AMY_RECS) for sid in ids}


GATES = {
    "01": gate_amy_pose,
    "03": gate_tango,
    "04": gate_aller,
    "05": gate_tox3,
}


# ── final report assembly (used by run_pipeline.py) ──

def build_report_rows(sequences: list[dict], eliminated_at: dict[str, str] | None = None) -> list[dict]:
    """Assemble the final_results.csv wide table for `sequences` (rows from
    sequences_ga.csv). `eliminated_at` (id -> stage name, "" if survived) is
    optional -- run_pipeline.py's funnel audit trail supplies it; the
    canonical outputs/final_results.csv is written without it.
    """
    pose    = load_pose()
    tango   = load_tango()
    tox3    = load_tox3()
    aller   = load_allercatpro()
    cd4     = load_cd4episcore()

    ids = [s["id"] for s in sequences]
    tango_g    = gate_tango(ids, tango)
    aller_g    = gate_aller(ids, aller)
    tox3_g     = gate_tox3(ids, tox3)
    amy_pose_g = gate_amy_pose(ids, pose)

    rows = []
    for seq in sequences:
        sid = seq["id"]
        safety_pass = tango_g[sid] and tox3_g[sid] and aller_g[sid]
        final_pass  = amy_pose_g[sid] and safety_pass
        row = {
            "id": sid, "seq_24aa": seq["seq_24aa"], "full_37aa": seq["full_37aa"],
            "pose_AMY1R": pose[sid].get("AMY1R", 0),
            "pose_AMY2R": pose[sid].get("AMY2R", 0),
            "pose_AMY3R": pose[sid].get("AMY3R", 0),
            "pose_CTR":   pose[sid].get("CTR",   0),
            "pose_CGRP":  pose[sid].get("CGRP",  0),
            "pose_AM1R":  pose[sid].get("AM1R",  0),
            "pose_AM2R":  pose[sid].get("AM2R",  0),
            "tango_pass": int(tango_g[sid]),
            "tox3_pass":  int(tox3_g[sid]),
            "aller_pass": int(aller_g[sid]),
            "cd4_pass": cd4.get(sid, {}).get("cd4_pass", 1),  # absent from cd4 summary = zero WB peptides = trivially safe
            "amy_pose_pass": int(amy_pose_g[sid]),
            "safety_pass": int(safety_pass),
            "final_pass":  int(final_pass),
        }
        if eliminated_at is not None:
            row["eliminated_at_stage"] = eliminated_at.get(sid, "")
        rows.append(row)
    return rows


def write_report(rows: list[dict], out_csv: Path) -> None:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
