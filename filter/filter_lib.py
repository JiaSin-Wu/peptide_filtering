"""
filter_lib.py

Shared library for the 3_Filter pipeline: candidate-id-list I/O, the raw
per-stage output loaders, and the per-stage gate functions. Used by both
run_final_filter.py (one-shot AND merge over every candidate at once) and
run_pipeline.py (a real sequential funnel, in whatever --order is given).

Gate-split rationale: the original final_pass formula is
    final_pass = amy_pose_pass AND offt_pass AND safety_pass
where amy_pose_pass only needs 01_pose_check's data, offt_pass needs BOTH
01's off-target pose_pass AND 02's dG_AB_REU, and safety_pass is
tango_pass(03) AND aller_pass(04) AND tox3_pass(05). Boolean AND is
order-independent, so splitting this into five separate per-stage gates
(gate_amy_pose=01, gate_offtarget=02, gate_tango=03, gate_aller=04,
gate_tox3=05) and applying them one at a time as a funnel is mathematically
equivalent to the original single-shot AND, *regardless of what order the
five stages run in* -- as long as 01 keeps scoring all 7 receptors (not
just the 3 AMY ones) so gate_offtarget still has off-target pose data to
read whenever 02 runs. See filter.md's Verification section for the
--replay-only regression check that confirms this across several --order
permutations.

2026-08-28 -- stage 02 (binding_energy / Rosetta dG_AB) is being deferred:
it is NOT run for the current batch. `final_pass` still requires
`offt_pass`, so with 02 not run every candidate missing from
rosetta_results.csv is conservatively `offt_pass = 0` and `final_pass = 0`
-- that is a "not yet evaluated for energy" state, NOT a rejection. To keep
final_results.csv from reading as if those candidates were filtered out,
build_report_rows() also emits
    final_pass_no_energy = amy_pose_pass AND safety_pass
i.e. off-target selectivity dropped entirely (it is meaningless without
dG_AB). As of this date 6 candidates have final_pass_no_energy = 1
(GA_005/006/025/078/079/090); only GA_006 also has final_pass = 1. Re-run
stage 02 and this column becomes redundant with final_pass again.

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
ROSETTA_CSV = HERE / "02_binding_energy" / "rosetta" / "rosetta_results.csv"
TANGO_CSV   = HERE / "03_aggregation" / "outputs" / "tango_results.csv"
TOX3_CSV    = HERE / "05_toxicity" / "outputs" / "toxinpred3_raw_ga.csv"
ALLER_GLOB  = str(HERE / "04_allergenicity" / "outputs" / "AllerCatPro2_prediction_*.csv")
SEQ_CSV     = HERE / "inputs" / "sequences_ga.csv"
IMM_CSV     = HERE / "06_immunogenicity" / "outputs" / "netmhciipan_sb_summary.csv"
CD4_CSV     = HERE / "06_immunogenicity" / "outputs" / "cd4episcore_summary.csv"

RECEPTORS = ["AMY1R", "AMY2R", "AMY3R", "CTR", "CGRP", "AM1R", "AM2R"]
AMY_RECS  = ["AMY1R", "AMY2R", "AMY3R"]
OFFT_RECS = ["CTR", "CGRP", "AM1R", "AM2R"]
AMYLIN_ID = "amylin"


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
      filtering (02's case: no cheap "everything" default, an explicit
      subset is required).
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


def load_rosetta() -> dict:
    d = defaultdict(dict)
    for r in csv.DictReader(open(ROSETTA_CSV)):
        d[r["id"]][r["receptor"]] = float(r["dG_AB_REU"])
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
    can't be a 01-stage rejection reason -- that's gate_offtarget's job.
    """
    pose = pose if pose is not None else load_pose()
    return {sid: all(pose.get(sid, {}).get(r, 0) == 1 for r in AMY_RECS) for sid in ids}


def gate_offtarget(ids: list[str], pose: dict | None = None, ros: dict | None = None) -> dict[str, bool]:
    """02 binding_energy. Needs 01's off-target pose_pass (already on disk
    by the time 02 runs -- 01 always precedes 02 whenever both are in the
    funnel, since offt_pass is only defined once both exist) plus 02's own
    dG_AB_REU, compared against the amylin baseline.
    """
    pose = pose if pose is not None else load_pose()
    ros  = ros if ros is not None else load_rosetta()
    if AMYLIN_ID not in ros:
        raise RuntimeError(f"amylin not found in Rosetta results — check {ROSETTA_CSV}")
    amylin_ros = ros[AMYLIN_ID]
    out = {}
    for sid in ids:
        out[sid] = all(
            pose.get(sid, {}).get(r, 0) == 0 or ros.get(sid, {}).get(r, 0.0) > amylin_ros[r]
            for r in OFFT_RECS
        )
    return out


GATES = {
    "01": gate_amy_pose,
    "02": gate_offtarget,
    "03": gate_tango,
    "04": gate_aller,
    "05": gate_tox3,
}


def report_amy_scores(ids: list[str], pose: dict | None = None, ros: dict | None = None) -> dict[str, bool]:
    """amy_pass: reported only, never gating -- AMY dG better than amylin's
    for all 3 AMY receptors. Docking/Rosetta affinity isn't a reliable
    enough proxy for agonism; dynamic assessment is reported separately in
    the thesis MD analysis.
    """
    pose = pose if pose is not None else load_pose()
    ros  = ros if ros is not None else load_rosetta()
    amylin_ros = ros[AMYLIN_ID]
    return {
        sid: all(
            pose.get(sid, {}).get(r, 0) == 1 and ros.get(sid, {}).get(r, 0.0) < amylin_ros[r]
            for r in AMY_RECS
        )
        for sid in ids
    }


# ── final report assembly (shared by run_final_filter.py and run_pipeline.py) ──

def build_report_rows(sequences: list[dict], eliminated_at: dict[str, str] | None = None) -> list[dict]:
    """Assemble the final_results.csv wide table for `sequences` (rows from
    sequences_ga.csv). `eliminated_at` (id -> stage name, "" if survived) is
    optional -- run_final_filter.py's one-shot AND doesn't have one,
    run_pipeline.py's funnel audit trail supplies it.
    """
    pose    = load_pose()
    ros     = load_rosetta()
    tango   = load_tango()
    tox3    = load_tox3()
    aller   = load_allercatpro()
    cd4     = load_cd4episcore()

    if AMYLIN_ID not in ros:
        raise RuntimeError(f"amylin not found in Rosetta results — check {ROSETTA_CSV}")

    ids = [s["id"] for s in sequences]
    tango_g    = gate_tango(ids, tango)
    aller_g    = gate_aller(ids, aller)
    tox3_g     = gate_tox3(ids, tox3)
    amy_pose_g = gate_amy_pose(ids, pose)
    offt_g     = gate_offtarget(ids, pose, ros)
    amy_score_g = report_amy_scores(ids, pose, ros)

    rows = []
    for seq in sequences:
        sid = seq["id"]
        safety_pass = tango_g[sid] and tox3_g[sid] and aller_g[sid]
        final_pass  = amy_pose_g[sid] and offt_g[sid] and safety_pass
        # 2026-08-28: stage 02 deferred -- see module docstring. This column
        # is the intended pass set while offt_pass/energy is not evaluated;
        # it drops off-target selectivity (meaningless without dG_AB).
        final_pass_no_energy = amy_pose_g[sid] and safety_pass
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
            "amy_pass":    int(amy_score_g[sid]),
            "offt_pass":   int(offt_g[sid]),
            "safety_pass": int(safety_pass),
            "final_pass":  int(final_pass),
            "final_pass_no_energy": int(final_pass_no_energy),
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
