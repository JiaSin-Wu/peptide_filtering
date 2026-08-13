"""
oracle.py

AF3 + GNN oracle for GA fitness evaluation.

Pipeline for each batch of novel peptides:
  Step 1: MSA  — write peptide-only JSONs → run MSA → read MSA back
            backend="jackhmmer" (default): run_msa.sh → AF3 data pipeline
            backend="mmseqs2":             msa_worker.py → mmseqs2 UniRef90
  Step 2: Build — inject MSA into 4 complex JSONs (AMY1R/2R/3R/CTR)
  Step 3: Infer — run_script.sh on all complex JSONs
  Step 4: GNN  — load embeddings → GATv2 ensemble → P(agonist)
"""

import copy
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

_GA_DIR     = Path(__file__).resolve().parents[1]   # AMY123R_agonist_Design/GA/
_ROOT       = _GA_DIR.parent                         # AMY123R_agonist_Design/

MSA_DIR     = _GA_DIR / "msa"
MSA_OUT_DIR = _GA_DIR / "msa_outputs"
INF_DIR     = _GA_DIR / "input"
OUT_DIR     = _GA_DIR / "outputs"
MODEL_DIR   = _ROOT   / "model/models"

RUN_MSA_SH  = _GA_DIR / "run_msa.sh"
RUN_INF_SH  = _GA_DIR / "run_script.sh"

# mmseqs2 backend settings (system binaries via env var)
MMSEQS_BIN    = Path(os.environ.get("MMSEQS_BIN",    "mmseqs"))
MMSEQS_DB     = Path(os.environ.get("MMSEQS_DB",     "/data/mmseqs2_db/uniref90"))
MMSEQS_DB_IDX = Path(os.environ.get("MMSEQS_DB_IDX", "/data/mmseqs2_db/uniref90.idx"))
MMSEQS_THREADS = int(os.environ.get("MMSEQS_THREADS", "20"))
_VALID_AA = set("ACDEFGHIKLMNPQRSTVWYacdefghiklmnpqrstvwy-")

AF3_REF = _ROOT / "data/af3_ref"
TEMPLATE_FOLDERS = {
    "AMY1R": AF3_REF / "AMY1R/P30988_O60894_P63092_ctail100",
    "AMY2R": AF3_REF / "AMY2R/P30988_O60895_P63092_ctail100",
    "AMY3R": AF3_REF / "AMY3R/P30988_O60896_P63092_ctail100",
    "CTR":   AF3_REF / "CTR/P30988_P63092_ctail100",
    "CGRP":  AF3_REF / "CGRP/Q16602_O60894_P63092_ctail100",
    "AM1R":  AF3_REF / "AM1R/Q16602_O60895_P63092_ctail100",
    "AM2R":  AF3_REF / "AM2R/Q16602_O60896_P63092_ctail100",
}
RECEPTORS = list(TEMPLATE_FOLDERS.keys())


# ── HELM parser ───────────────────────────────────────────────────────────────

def helm_to_sequence(helm: str) -> str:
    m = re.search(r'PEPTIDE\d+\{([^}]+)\}', helm)
    if not m:
        raise ValueError(f"Cannot parse HELM: {helm[:80]}")
    return "".join(m.group(1).split("."))


# ── Template cache ────────────────────────────────────────────────────────────

_TEMPLATES: dict[str, dict] = {}

def _get_template(receptor: str) -> dict:
    if receptor not in _TEMPLATES:
        folder = TEMPLATE_FOLDERS[receptor]
        djson  = list(folder.glob("*_data.json"))
        if not djson:
            raise FileNotFoundError(f"No data.json in {folder}")
        with open(djson[0]) as f:
            _TEMPLATES[receptor] = json.load(f)
    return _TEMPLATES[receptor]


# ── Step 1: MSA ───────────────────────────────────────────────────────────────

def _write_msa_json(name: str, sequence: str, modifications: list | None = None):
    """Write peptide-only JSON to MSA_DIR for MSA pipeline."""
    d = {
        "name": name,
        "modelSeeds": [1],
        "sequences": [{
            "protein": {
                "sequence": sequence,
                "modifications": modifications or [{"ptmType": "NH2", "ptmPosition": len(sequence)}],
                "id": "A",
            }
        }],
        "dialect": "alphafold3",
        "version": 4,
    }
    with open(MSA_DIR / f"{name}.json", "w") as f:
        json.dump(d, f, indent=2)


def _read_msa_result(name: str) -> str | None:
    """Read unpairedMsa from MSA pipeline output. Returns None if missing."""
    p = MSA_OUT_DIR / name / f"{name}_data.json"
    if not p.exists():
        return None
    with open(p) as f:
        d = json.load(f)
    info = d["sequences"][0][list(d["sequences"][0].keys())[0]]
    return info.get("unpairedMsa", "")


def _msa_mmseqs2(sequence: str) -> str:
    """Run mmseqs2 search + result2msa for one sequence. Returns A3M string."""
    def mm(*args):
        subprocess.run([str(MMSEQS_BIN), *[str(a) for a in args]],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    with tempfile.TemporaryDirectory() as tmpdir:
        d = Path(tmpdir)
        (d / "query.fasta").write_text(f">query\n{sequence}\n")
        mm("createdb", d / "query.fasta", d / "query")
        mm("search", d / "query", MMSEQS_DB_IDX, d / "result", d / "tmp",
           "--threads", MMSEQS_THREADS, "-s", "7.5", "--db-load-mode", "2")
        mm("result2msa", d / "query", MMSEQS_DB, d / "result", d / "query.a3m",
           "--msa-format-mode", "6", "--db-load-mode", "2")
        return (d / "query.a3m").read_text()


def _clean_a3m(a3m: str, af3_seq: str) -> str:
    """Remove non-standard residues and fix query line with af3_seq (may have X)."""
    lines = a3m.splitlines()
    cleaned, i = [], 0
    while i < len(lines):
        hdr = lines[i]
        if hdr.startswith(">") and i + 1 < len(lines):
            seq = lines[i + 1].strip()
            if seq and all(c in _VALID_AA for c in seq):
                cleaned.append(hdr + "\n")
                cleaned.append(seq + "\n")
            i += 2
        else:
            i += 1
    if len(cleaned) >= 2:
        cleaned[1] = af3_seq + "\n"
    return "".join(cleaned)


def run_msa_batch(names: list[str], sequences: list[str],
                  modifications: list[list] | None = None,
                  timeout: int = 3600,
                  backend: str = "jackhmmer") -> dict[str, str]:
    """
    Run MSA for a batch of peptides.
    Returns {name: unpairedMsa_string}.

    backend="jackhmmer"  (default) — AF3 data pipeline via run_msa.sh
    backend="mmseqs2"              — mmseqs2 UniRef90, --db-load-mode 2
    """
    mods = modifications or [None] * len(names)

    if backend == "mmseqs2":
        print(f"[Oracle] Running mmseqs2 MSA for {len(names)} peptides...")
        t0 = time.time()
        result = {}
        for name, seq, mod in zip(names, sequences, mods):
            af3_seq = seq
            if mod:
                s = list(seq)
                for m in mod:
                    pos = m.get("ptmPosition", 0) - 1
                    # NH2 (C-term amide) → X; extend map as needed
                    s[pos] = "X"
                af3_seq = "".join(s)
            try:
                raw   = _msa_mmseqs2(seq)
                a3m   = _clean_a3m(raw, af3_seq)
                result[name] = a3m
                print(f"  {name}: {a3m.count(chr(62))} seqs")
            except Exception as e:
                print(f"  [WARN] mmseqs2 failed for {name}: {e}, using placeholder")
                result[name] = f">query\n{af3_seq}\n"
        print(f"[Oracle] mmseqs2 MSA done in {(time.time()-t0)/60:.1f} min")
        return result

    # ── jackhmmer (default) ────────────────────────────────────────────────────
    MSA_DIR.mkdir(exist_ok=True)
    for name, seq, mod in zip(names, sequences, mods):
        _write_msa_json(name, seq, mod)

    print(f"[Oracle] Running jackhmmer MSA for {len(names)} peptides...")
    t0     = time.time()
    result = subprocess.run(["bash", str(RUN_MSA_SH)],
                            capture_output=True, text=True, timeout=timeout)
    elapsed = time.time() - t0
    print(f"[Oracle] jackhmmer MSA done in {elapsed/60:.1f} min")

    if result.returncode != 0:
        raise RuntimeError(f"run_msa.sh failed:\n{result.stderr[-2000:]}")

    return {name: (_read_msa_result(name) or f">query\n{seq}\n")
            for name, seq in zip(names, sequences)}


# ── Step 2: Build complex JSONs ───────────────────────────────────────────────

PEPTIDE_PREFIX = "KCNTATCATQRLA"   # fixed N-terminal 13 AA (amylin ring + helix)


def build_complex_json(peptide_name: str, peptide_seq: str,
                       peptide_msa: str, receptor: str) -> dict:
    """
    Build full complex JSON for one receptor by:
    - Copying chains A/C/D from template (MSA already included)
    - Substituting chain B with new peptide + its MSA
    """
    template = _get_template(receptor)
    d        = copy.deepcopy(template)
    job_name = f"{peptide_name}_{receptor}"
    d["name"] = job_name

    full_seq = PEPTIDE_PREFIX + peptide_seq
    chain_b  = {
        "id":           "B",
        "sequence":     full_seq,
        "unpairedMsa":  peptide_msa,
        "pairedMsa":    "",
        "templates":    [],
        "modifications": [{"ptmType": "NH2", "ptmPosition": len(full_seq)}],
    }

    replaced = False
    for s in d["sequences"]:
        k    = list(s.keys())[0]
        info = s[k]
        if info["id"] == "B":
            info.update(chain_b)
            info.pop("pairedMsaPath", None)
            replaced = True
            break

    if not replaced:
        # Template has no chain B — insert peptide after chain A
        d["sequences"].insert(1, {"protein": chain_b})

    return d


# ── Step 3: Inference ─────────────────────────────────────────────────────────

_TRIGGER  = INF_DIR / ".trigger"
_DONE     = OUT_DIR / ".done"
_ERROR    = OUT_DIR / ".error"
_DAEMON_CONTAINER = os.environ.get("AF3_DAEMON_CONTAINER", "af3_daemon")


def _daemon_running() -> bool:
    """Return True if the af3_daemon Docker container is up."""
    r = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", _DAEMON_CONTAINER],
        capture_output=True, text=True,
    )
    return r.returncode == 0 and r.stdout.strip() == "true"


def run_inference_batch(complex_jsons: list[dict], timeout: int = 3600):
    """
    Write complex JSONs to INF_DIR and trigger AF3 inference.

    If the af3_daemon container is running (persistent mode):
      - write JSONs, atomic-rename .trigger, poll for .done
      - model stays in GPU memory between calls → no reload overhead

    Falls back to `docker run --rm` (one-shot mode) if daemon is not running.
    """
    INF_DIR.mkdir(exist_ok=True)

    # clear previous inputs
    for f in INF_DIR.glob("*.json"):
        f.unlink()

    for d in complex_jsons:
        with open(INF_DIR / f"{d['name']}.json", "w") as f:
            json.dump(d, f, indent=2)

    print(f"[Oracle] Running inference for {len(complex_jsons)} complexes...")
    t0 = time.time()

    if _daemon_running():
        # ── Persistent daemon mode ────────────────────────────────────────────
        _DONE.unlink(missing_ok=True)

        # Atomic trigger: write to .trigger.tmp first, then rename
        tmp_trigger = _TRIGGER.with_suffix(".tmp")
        tmp_trigger.touch()
        os.replace(tmp_trigger, _TRIGGER)

        deadline = time.time() + timeout
        while not _DONE.exists():
            if time.time() > deadline:
                raise TimeoutError(
                    f"[Oracle] AF3 daemon did not write .done within {timeout}s"
                )
            time.sleep(0.5)

        _DONE.unlink(missing_ok=True)
        if _ERROR.exists():
            msg = _ERROR.read_text().strip()
            _ERROR.unlink(missing_ok=True)
            raise RuntimeError(f"[Oracle] AF3 daemon reported error: {msg}")

    else:
        # ── One-shot fallback (docker run --rm) ───────────────────────────────
        print(f"[Oracle] Daemon '{_DAEMON_CONTAINER}' not running — "
              "falling back to docker run", flush=True)
        result = subprocess.run(["bash", str(RUN_INF_SH)],
                                capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            raise RuntimeError(f"run_script.sh failed:\n{result.stderr[-2000:]}")

    elapsed = time.time() - t0
    print(f"[Oracle] Inference done in {elapsed/60:.1f} min")


# ── Step 4: GNN inference ─────────────────────────────────────────────────────

_GNN_MODELS = None
_GNN_DEVICE = None
_MODEL_PATH = str(_ROOT / "model")


def _ensure_model_on_path():
    import sys
    if _MODEL_PATH not in sys.path:
        sys.path.insert(0, _MODEL_PATH)


def _load_gnn(model_dir: Path = MODEL_DIR / "gnn_gat_gat_h8_drop01",
              device_str: str = "cpu"):
    global _GNN_MODELS, _GNN_DEVICE
    if _GNN_MODELS is not None:
        return _GNN_MODELS, _GNN_DEVICE

    import torch
    _ensure_model_on_path()
    from gnn.model import AgonismGNN

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    with open(model_dir / "config.json") as f:
        cfg = json.load(f)
    n_mods = cfg.get("n_models", 5)

    models = []
    for i in range(n_mods):
        pt = model_dir / f"model_{i}.pt"
        if not pt.exists():
            continue
        m = AgonismGNN(
            hidden_dim=cfg["hidden_dim"], n_heads=cfg["n_heads"],
            n_layers=cfg["n_layers"],   dropout=cfg["dropout"],
        ).to(device)
        m.load_state_dict(torch.load(pt, map_location=device, weights_only=True))
        m.eval()
        models.append(m)

    _GNN_MODELS, _GNN_DEVICE = models, device
    print(f"[Oracle] Loaded {len(models)} GNN models on {device}")
    return models, device


def gnn_predict_folder(af3_folder: Path) -> float:
    """Run GNN ensemble on one AF3 output folder. Returns P(agonist)."""
    import torch
    _ensure_model_on_path()
    from gnn.graph import build_graph
    from torch_geometric.data import Batch

    models, device = _load_gnn()
    graph = build_graph(af3_folder, cutoff=8.0)
    if graph is None:
        return 0.0   # no contact edge → non-agonist prediction

    batch = Batch.from_data_list([graph.to(device)])
    probs = []
    with torch.no_grad():
        for m in models:
            probs.append(torch.sigmoid(m(batch)).item())
    return float(np.mean(probs))

