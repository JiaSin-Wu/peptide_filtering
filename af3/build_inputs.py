"""
build_inputs.py — build AlphaFold3 complex JSONs (receptor + G protein + peptide).

Rescued from git history (fc709dc:GA/ga/oracle.py build_complex_json) and adapted to
the current data/af3_ref templates.

Differences from the historic builder:
  * The peptide is appended as chain "L" (what filter/01_pose_check expects).
    The old code overwrote chain "B", which is the RAMP in the current templates.
  * Templates live in data/af3_ref/<RECEPTOR>/<ACC>_ag/<ACC>_ag_data.json.
  * The peptide MSA comes from AF3's own data pipeline (jackhmmer against
    /opt/af3_databases), run once on the peptide alone, instead of mmseqs2.

Peptide = PEPTIDE_PREFIX (13 aa) + seq_24aa (37 aa), C-terminal NH2 amidation.
The peptide MSA query line has its last residue replaced by X (modified residue),
as in the historic pipeline. The MSA's templates are discarded (templates: []).
"""

import copy
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

ROOT        = Path(__file__).resolve().parents[1]
REF_DIR     = ROOT / "data" / "af3_ref"
SEQ_CSV     = ROOT / "filter" / "inputs" / "sequences_ga.csv"
LIGAND_DIR  = REF_DIR / "ligand"                  # persistent peptide MSAs: <id>.a3m
MSA_WORK    = ROOT / "af3" / "output" / "msa_search"   # scratch: peptide-only data-pipeline runs

PEPTIDE_PREFIX = "KCNTATCATQRLA"   # fixed N-terminal 13 aa (amylin ring + helix)
PEPTIDE_CHAIN  = "L"

IMAGE    = config.AF3_IMAGE
DB_DIR   = config.AF3_DB_DIR
MSA_CPUS = config.AF3_MSA_CPUS


def load_template(receptor: str) -> dict:
    hits = sorted(REF_DIR.glob(f"{receptor}/*_ag/*_data.json"))
    if len(hits) != 1:
        raise FileNotFoundError(
            f"expected exactly one template for {receptor} under {REF_DIR}, got {hits}")
    with open(hits[0]) as f:
        return json.load(f)


def load_peptides(ids: list[str]) -> dict[str, str]:
    """Return {id: seq_24aa} from sequences_ga.csv; raise if any id is missing."""
    with open(SEQ_CSV) as f:
        rows = {r["id"]: r for r in csv.DictReader(f)}
    missing = [i for i in ids if i not in rows]
    if missing:
        raise KeyError(f"ids not in {SEQ_CSV.name}: {missing}")
    out = {}
    for i in ids:
        r = rows[i]
        if PEPTIDE_PREFIX + r["seq_24aa"] != r["full_37aa"]:
            raise ValueError(f"{i}: prefix + seq_24aa != full_37aa")
        out[i] = r["seq_24aa"]
    return out


# ── peptide MSA (AF3 data pipeline, peptide alone, CPU only) ──────────────────

def _query_only(seq_24: str) -> str:
    full = PEPTIDE_PREFIX + seq_24
    return f">query\n{full[:-1]}X\n"


def _set_query_x(a3m: str, seq_24: str) -> str:
    """Make the first (query) record's sequence end in X for the NH2-modified residue."""
    lines = a3m.splitlines()
    full = PEPTIDE_PREFIX + seq_24
    if len(lines) < 2 or lines[1].strip() != full:
        raise ValueError("MSA query line does not match the peptide sequence")
    lines[1] = full[:-1] + "X"
    return "\n".join(lines) + "\n"


def compute_msas(peptides: dict[str, str], no_msa: bool = False) -> dict[str, str]:
    """
    Return {peptide_id: unpairedMsa}. The raw MSA (query line unmodified) is kept in
    data/af3_ref/ligand/<id>.a3m and shared by all receptors; only peptides without
    that file are searched, with one run_alphafold.py --run_inference=false call.
    """
    if no_msa:
        return {i: _query_only(s) for i, s in peptides.items()}

    todo = [i for i in peptides if not (LIGAND_DIR / f"{i}.a3m").exists()]
    if todo:
        in_dir = MSA_WORK / "_in"
        if in_dir.exists():
            shutil.rmtree(in_dir)
        in_dir.mkdir(parents=True)
        for i in todo:
            (in_dir / f"{i}.json").write_text(json.dumps({
                "dialect": "alphafold3", "version": 1, "name": i, "modelSeeds": [1],
                "sequences": [{"protein": {"id": "A",
                                           "sequence": PEPTIDE_PREFIX + peptides[i]}}],
            }))
        cmd = [
            "docker", "run", "--rm",                      # no --gpus: search only
            "--user", f"{os.getuid()}:{os.getgid()}",
            "-v", f"{in_dir}:/root/inputs:ro",
            "-v", f"{MSA_WORK}:/root/outputs",
            "-v", f"{DB_DIR}:/root/public_databases:ro",
            "-e", "PYTHONPATH=/app/alphafold/src",
            IMAGE,
            "python3", "/app/alphafold/run_alphafold.py",
            "--input_dir=/root/inputs",
            "--output_dir=/root/outputs",
            "--db_dir=/root/public_databases",
            "--run_data_pipeline=true",
            "--run_inference=false",
            f"--jackhmmer_n_cpu={MSA_CPUS}",
            "--force_output_dir",
        ]
        print("$ " + " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

        LIGAND_DIR.mkdir(parents=True, exist_ok=True)
        for i in todo:
            d = json.loads((MSA_WORK / i / f"{i}_data.json").read_text())
            (LIGAND_DIR / f"{i}.a3m").write_text(
                d["sequences"][0]["protein"]["unpairedMsa"])
            print(f"  saved {(LIGAND_DIR / f'{i}.a3m').relative_to(ROOT)}", flush=True)

    return {i: _set_query_x((LIGAND_DIR / f"{i}.a3m").read_text(), seq_24)
            for i, seq_24 in peptides.items()}


# ── complex JSON ──────────────────────────────────────────────────────────────

def build_complex_json(peptide_id: str, seq_24: str, peptide_msa: str,
                       receptor: str, seed: int | None = None) -> dict:
    d = copy.deepcopy(load_template(receptor))
    d["name"] = f"{peptide_id}_{receptor}"
    if seed is not None:
        d["modelSeeds"] = [seed]

    if any(list(s.values())[0]["id"] == PEPTIDE_CHAIN for s in d["sequences"]):
        raise ValueError(f"template for {receptor} already has chain {PEPTIDE_CHAIN}")

    full = PEPTIDE_PREFIX + seq_24
    d["sequences"].append({"protein": {
        "id": PEPTIDE_CHAIN,
        "sequence": full,
        "modifications": [{"ptmType": "NH2", "ptmPosition": len(full)}],
        "unpairedMsa": peptide_msa,
        "pairedMsa": "",
        "templates": [],
    }})
    return d


def write_json(d: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{d['name']}.json"
    with open(p, "w") as f:
        json.dump(d, f)
    return p
