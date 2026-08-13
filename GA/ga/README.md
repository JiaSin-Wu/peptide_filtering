# GA Peptide Optimisation

ε-constraint minimax optimisation for CALCR-family peptide ligands.

## Objectives

- Feasibility constraint: `min(P_AMY1R, P_AMY2R, P_AMY3R) >= epsilon`
- Objective (among feasible): minimise `max(P_CTR, P_CGRP, P_AM1R, P_AM2R)`

## Peptide design

- Total length sent to AF3: **37 AA**
- Fixed N-terminal prefix (amylin ring): `KCNTATCATQRLA` (13 AA)
- GA optimises the remaining **24 AA** C-terminal segment
- Alphabet: 20 standard amino acids + `[am]` C-terminal amide (added automatically)

## Quick start

```bash
# 1. Start persistent AF3 daemon (load model once, stays on GPU)
bash ~/start_af3_daemon.sh
# Wait for "Model ready" in the logs before proceeding

# 2. Run GA
conda run -n af3_ml python3 -m ga.run_ga \
  --seed_seqs ~/seeds.txt \
  --run_name run_001 \
  --generations 50 \
  --pop_size 50 \
  --msa-backend mmseqs2
```

## Output

Results are saved to `ga/runs/{run_name}/`:

| File | Content |
|---|---|
| `log.csv` | Per-generation stats (p_amy, p_ctr, oracle_calls, …) |
| `library.json` | All evaluated sequences and scores |
| `checkpoint.json` | Latest checkpoint for crash recovery |
| `config.json` | Run parameters |
| `pareto.json` | Final Pareto front |

## Resume after crash

```bash
conda run -n af3_ml python3 -m ga.run_ga \
  --seed_seqs ~/seeds.txt \
  --run_name run_001 \
  --generations 50 \
  --pop_size 50 \
  --msa-backend mmseqs2 \
  --resume
```

## Seeds

`~/seeds.txt` — last 24 AA of CALCR agonists from training data (natural AA, agonist label, CALCR uniprot):

```
NFLVHSSNNFGAILSSTNVGSNTY   # human amylin
NFLVRSSNNLGPVLPPTNVGSNTY   # rat amylin variant
```

## AF3 daemon

The persistent daemon keeps AF3 model weights in GPU memory across all generations, avoiding repeated model loading (~1–2 min saved per batch).

```bash
bash ~/start_af3_daemon.sh          # start
docker logs -f af3_daemon           # monitor
docker stop af3_daemon              # stop after GA finishes
```

If the daemon is not running, `run_inference_batch` falls back to `docker run --rm` automatically.

## Distributed evaluation (remote worker machine)

For running inference on a separate GPU machine, use the Redis queue mode.

### GA machine

```bash
# Start Redis
docker run -d --name redis -p 6379:6379 redis:7

# Run GA with queue mode
conda run -n af3_ml python3 -m ga.run_ga \
  --seed_seqs ~/seeds.txt \
  --run_name run_001 \
  --generations 50 \
  --pop_size 50 \
  --queue-host <GA_MACHINE_IP>
```

### Worker machine (Docker only)

```bash
# 1. Start AF3 daemon
bash ~/start_af3_daemon.sh

# 2. Build worker image (once)
cd ~/af3_ml
docker build -f Dockerfile.worker -t ga_worker .

# 3. Start worker
docker run -d --name ga_worker --gpus all \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v ~/AMY123R_GA_v2/GA/input:/root/input \
  -v ~/AMY123R_GA_v2/GA/outputs:/root/outputs \
  -v /data/mmseqs2_db:/data/mmseqs2_db \
  -v ~/af3_ml/models:/app/models \
  ga_worker --redis-host <GA_MACHINE_IP>
```

Multiple worker machines can point at the same Redis — jobs are distributed automatically.

The worker container needs three shared resources with `af3_daemon`:
- `GA/input` and `GA/outputs` — trigger/done protocol for AF3 inference
- `/var/run/docker.sock` — to detect whether `af3_daemon` is running
