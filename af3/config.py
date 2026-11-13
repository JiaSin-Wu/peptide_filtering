"""
Settings for the AF3 scripts (build_inputs.py, run_batch.py).

Precedence: real environment variable > .env file at the repo root > default (optional
settings only). The image and the weights/database paths have no default: copy
.env.example to .env and edit it. .env is local to each machine and is gitignored.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv(path: Path) -> dict[str, str]:
    """Minimal KEY=VALUE parser (no python-dotenv dependency). Missing file is fine."""
    out = {}
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip("'\"")
    return out


_DOTENV = _load_dotenv(ROOT / ".env")


def _get(key: str, default: str | None = None) -> str:
    val = os.environ.get(key) or _DOTENV.get(key) or default
    if val is None:
        raise SystemExit(
            f"{key} is not set. Copy .env.example to .env at the repo root and edit it "
            f"(see README 'Structure prediction').")
    return val


# Required (machine-specific, no default): set them in .env
AF3_IMAGE = _get("AF3_IMAGE")          # Docker image with AlphaFold 3 (/app/alphafold/run_alphafold.py)
AF3_MODEL_DIR = _get("AF3_MODEL_DIR")  # directory with the AF3 weights (af3.bin)
AF3_DB_DIR = _get("AF3_DB_DIR")        # directory with the AF3 genetic databases

# Optional tuning, defaults are fine
AF3_MSA_CPUS = _get("AF3_MSA_CPUS", "8")              # jackhmmer threads per search
AF3_MEM_FRACTION = _get("AF3_MEM_FRACTION", "0.90")   # GPU memory fraction; lower on a shared GPU
