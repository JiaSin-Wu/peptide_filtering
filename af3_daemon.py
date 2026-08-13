"""
af3_daemon.py — Persistent AF3 inference daemon (runs inside Docker).

Loads ModelRunner ONCE at startup → model stays in GPU memory between batches.

File-based protocol (no networking needed):
  Host:   write JSONs → /root/inputs/
          atomic-rename /root/inputs/.trigger.tmp → /root/inputs/.trigger
  Daemon: detects .trigger
          deletes .trigger  (ack — host may now prepare next batch)
          runs process_fold_input for every *.json in /root/inputs/
          writes /root/outputs/.done
  Host:   detects .done → reads outputs → deletes .done

Start with (see start_af3_daemon.sh):
  docker run -d --name af3_daemon \
    --gpus all \
    -v GA/input:/root/inputs \
    -v GA/outputs:/root/outputs \
    -v /opt/model_parameter:/root/af3_params \
    -v af3_daemon.py:/app/alphafold/af3_daemon.py \
    -e PYTHONPATH=/app/alphafold/src \
    jiasin/alphafold3:latest \
    python3 /app/alphafold/af3_daemon.py
"""

import datetime
import os
import pathlib
import sys
import time
import typing

sys.path.insert(0, "/app/alphafold/src")

import jax
from alphafold3.common import folding_input
import tokamax

# Import from the co-located run_alphafold module
sys.path.insert(0, "/app/alphafold")
from run_alphafold import ModelRunner, make_model_config, process_fold_input

INPUT_DIR   = pathlib.Path("/root/inputs")
OUTPUT_DIR  = pathlib.Path("/root/outputs")
MODEL_DIR   = pathlib.Path("/root/af3_params")
TRIGGER     = INPUT_DIR  / ".trigger"
DONE_FLAG   = OUTPUT_DIR / ".done"
ERROR_FLAG  = OUTPUT_DIR / ".error"

POLL_SEC    = 0.5          # seconds between trigger polls
BUCKETS     = (256, 512, 768, 1024, 1280, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120)
MAX_TEMPLATE_DATE = datetime.date(2021, 9, 30)


def build_runner() -> ModelRunner:
    gpu_devices = jax.local_devices(backend="gpu")
    if not gpu_devices:
        raise RuntimeError("No GPU found — af3_daemon requires a GPU.")
    device = gpu_devices[0]
    print(f"[Daemon] GPU: {device}", flush=True)

    runner = ModelRunner(
        config=make_model_config(
            flash_attention_implementation=typing.cast(
                tokamax.DotProductAttentionImplementation, "triton"
            ),
            num_diffusion_samples=1,
            num_recycles=10,
            return_embeddings=True,
            return_distogram=False,
        ),
        device=device,
        model_dir=MODEL_DIR,
    )
    # Eagerly load params so the first batch doesn't pay the load cost.
    print("[Daemon] Loading model parameters …", flush=True)
    _ = runner.model_params
    print("[Daemon] Model ready — waiting for jobs.", flush=True)
    return runner


def run_batch(runner: ModelRunner) -> int:
    """Process every *.json in INPUT_DIR. Returns number of jobs run."""
    json_files = sorted(INPUT_DIR.glob("*.json"))
    if not json_files:
        print("[Daemon] WARN: .trigger present but no JSON files found.", flush=True)
        return 0

    fold_inputs = folding_input.load_fold_inputs_from_dir(INPUT_DIR)
    n = 0
    for fi in fold_inputs:
        t0 = time.time()
        out = OUTPUT_DIR / fi.sanitised_name()
        process_fold_input(
            fold_input=fi,
            data_pipeline_config=None,   # MSA already baked into JSON
            model_runner=runner,
            output_dir=out,
            buckets=BUCKETS,
            ref_max_modified_date=MAX_TEMPLATE_DATE,
            force_output_dir=True,
        )
        print(f"[Daemon] {fi.name} done in {time.time()-t0:.1f}s", flush=True)
        n += 1
    return n


def main():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    runner = build_runner()

    while True:
        if not TRIGGER.exists():
            time.sleep(POLL_SEC)
            continue

        # Ack trigger immediately so host can start writing next batch.
        try:
            TRIGGER.unlink()
        except FileNotFoundError:
            continue   # another process removed it (shouldn't happen, but safe)

        DONE_FLAG.unlink(missing_ok=True)
        ERROR_FLAG.unlink(missing_ok=True)

        t_batch = time.time()
        try:
            n = run_batch(runner)
            elapsed = time.time() - t_batch
            print(f"[Daemon] Batch complete: {n} jobs in {elapsed:.1f}s", flush=True)
            DONE_FLAG.touch()
        except Exception as e:
            msg = str(e)
            print(f"[Daemon] ERROR during batch: {msg}", flush=True)
            ERROR_FLAG.write_text(msg)
            DONE_FLAG.touch()   # wake up host so it doesn't stall


if __name__ == "__main__":
    main()
