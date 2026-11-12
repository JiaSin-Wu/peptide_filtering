# peptide_filtering

Design and triage of selective AMY receptor (AMY1R/AMY2R/AMY3R) agonist peptides.

A genetic algorithm proposes 37-residue amylin analogues, an AF3 + GNN oracle
scores each one against seven CALCR-family receptors, and a sequence/structure
filter narrows the resulting roster to a small panel that is then simulated.

This repository holds the **middle** of that workflow: the AF3 daemon, the
docking driver and the filter pipeline. The GA (with its scoring model) and the
MD stages live in their own repositories (see below).

## The workflow end to end

| # | Stage | Lives in | Status |
|---|---|---|---|
| 1 | GA peptide optimisation | [`JiaSin-Wu/GA`](https://github.com/JiaSin-Wu/GA) — `lab_193:~/2_GA` | separate repo |
| 2 | AF3 + GNN oracle (scores the GA's candidates) | the GA repo: `grader/inference/` | separate repo |
| 3 | Structure prediction / docking | this repo: `af3_daemon.py`, `start_af3_daemon.sh`, `run_docking.py` | here |
| 4 | Candidate filtering | this repo: [`filter/`](filter/README.md) | here |
| 5 | Molecular dynamics | [`JiaSin-Wu/GPCR_MD`](https://github.com/JiaSin-Wu/GPCR_MD) — `lab_193:~/4_MD` | separate repo |

The funnel this produced: **3,042 → 92 → 7 → 6**. The six retained candidates
are `GA_005`, `GA_006`, `GA_025`, `GA_078`, `GA_079`, `GA_090`.
[`filter/README.md`](filter/README.md) is the authority on the filter stages and
their evidence files; this file does not restate them.

## Repository layout

```
af3_daemon.py          persistent AF3 inference daemon (runs inside Docker)
start_af3_daemon.sh    launches that daemon
run_docking.py         AF3 docking driver, candidates x receptors  (see caveat)
data/af3_ref/          per-receptor AF3 template JSONs, one dir per receptor
filter/                the filtering pipeline  -> filter/README.md
Sequence_1             FASTA of the six retained candidates + GA_004
seq.aac, seq.dpc       AAC / DPC composition tables  (provenance unknown)
structures/            AF3 docking output, gitignored (100+ GB)
```

## Stage 1 — GA

The GA is an ε-constraint minimax search. The peptide is 37 residues: a fixed
13-residue amylin ring `KCNTATCATQRLA`, then 24 residues the GA optimises, then
a C-terminal amide.

- **Feasibility:** `min(P_AMY1R, P_AMY2R, P_AMY3R) >= epsilon`
- **Objective, among feasible candidates:** minimise `max(P_CTR, P_CGRP, P_AM1R, P_AM2R)`

Run it from the GA repo, not from here:

```bash
# on each worker machine: AF3 daemon first, wait for "Model ready"
bash start_af3_daemon.sh

# on the GA machine
docker run -d --name redis -p 6379:6379 redis:7
conda run -n af3_ml python3 -m ga.run_ga \
    --seed_seqs seeds.txt --run_name run_001 \
    --generations 60 --pop_size 50 --queue-host "$GA_HOST_IP"
```

A run writes `runs/<run_name>/` with `library.json` (every sequence evaluated),
`log.csv`, `checkpoint.json`, `config.json` and `pareto.json`. `--resume` picks
up from the checkpoint. See `WORKER_DEPLOYMENT.md` in the GA repo for the
current single-image worker setup (`WORKER_ROLE=infer|msa|both`).

The production run is `run_001`: 60 generations, population 50, elite 5,
seed 42, **3,042 sequences evaluated**. The 92-candidate roster in
`filter/inputs/sequences_ga.csv` is a verbatim subset of its `library.json` —
every row matches on all seven receptor probabilities.

> **Gap:** the 3,042 → 92 reduction is described as "three on-target floors and
> four off-target ceilings", but the threshold values are not recorded in any
> script, config or document in either repository, and no script that writes
> `sequences_ga.csv` could be found. The 92 rows are bounded below by
> AMY1R 0.4415 / AMY2R 0.4185 / AMY3R 0.3972 and above by CTR 0.2977 /
> CGRP 0.6730 / AM1R 0.4983 / AM2R 0.4990; applying exactly that box to the
> 3,042 reproduces the 92, but those are observed bounds, not the intended
> thresholds. **TODO: recover and document the real cut-offs.**

## Stage 2 — the AF3 + GNN oracle

Each candidate is scored for all seven receptors by this four-step oracle, which
the GA's worker drives:

1. **MSA** on the peptide alone (jackhmmer via the AF3 data pipeline, or MMseqs2
   against UniRef90)
2. **Build** the receptor complex JSONs from the templates in `data/af3_ref/`
3. **Infer** with AlphaFold 3, saving embeddings
4. **Score** the embeddings with a GATv2 ensemble → P(agonist)

**None of this lives here any more.** The scorer is part of the GA repo, at
`grader/inference/`, which resolves its weights relative to its own location:

```python
MODEL_DIR = Path(__file__).resolve().parent / "model/models"
_load_gnn(model_dir=MODEL_DIR / "gnn_gat_h128_d01")
```

The current ensemble is **`gnn_gat_h128_d01`** (5 folds, plus an `emb_val.npz`
the adapted heads need). The GA repo's `Dockerfile.worker` copies
`grader/inference/` into the worker image, so the worker is self-contained.

This repository used to carry a copy of those weights under `model/`. It was
removed once it was confirmed that nothing here reads it: no file in this repo,
including the whole of `filter/`, imports or references it. The copy was also
stale — it lacked `emb_val.npz` and so could not have run the current scoring
path. Earlier checkpoints `gnn_gat_drop01` and `gnn_gat_gat_h8_drop01` were
superseded when the model was switched, and `lgbm_default.pkl` lost its last
reader when `predict.py` was deleted.

The weights remain byte-identical in `~/2_GA/grader/inference/model/models/` and
`~/gpcr-agonism-gnn/checkpoints/`. Note that **both of those are gitignored**, so
this repository's history is the only version-controlled copy:

```bash
git checkout 127f1e0 -- model/        # weights + the gnn package
git checkout fc709dc  -- model/       # also the training/inference drivers
```

Those older drivers hardcode `/home/jiasin/AMY123R_agonist_Design/model` as their
root and need their paths adjusted before they will run.

## Stage 3 — structures

`af3_daemon.py` keeps AlphaFold 3 resident in GPU memory so successive batches
do not pay the model-load cost. It talks to the host through files, not
sockets: the host writes JSONs into the input directory and renames
`.trigger.tmp` → `.trigger`; the daemon consumes them and writes `.done` into
the output directory.

```bash
bash start_af3_daemon.sh                              # default container af3_daemon
MEM_FRACTION=0.45 bash start_af3_daemon.sh            # share a busy GPU
MODEL_DIR=/data1/jiasin/af3/af3_model_parameter bash start_af3_daemon.sh
```

It runs `jiasin/alphafold3:latest` with `--gpus all`, expects AF3 weights at
`/opt/model_parameter` and waits up to 300 s for `Model ready` in the logs.
Inference uses one diffusion sample and ten recycles.

`data/af3_ref/<RECEPTOR>/<uniprots>_ag/` holds one AF3 template per receptor,
named `<receptor>_<RAMP>_<Gαs>_ag`. Each has chain A = receptor (CALCR P30988 or
CLR Q16602), chain B = RAMP where the receptor needs one, and chain G =
full-length Gαs (P63092). These replaced the earlier `*_ctail100` templates,
which carried only the 100-residue C-terminal fragment of Gαs. (What the `_ag`
suffix abbreviates is not documented anywhere.)

`run_docking.py` folds candidates against all seven receptors, pipelining the
CPU-side MSA of one sequence against the GPU-side inference of the previous one,
and writes to `structures/<seq_id>/<receptor>/`.

> **Caveat:** `run_docking.py` does not run against the current `HEAD`. It
> imports `worker` and `oracle` from a `GA/` directory that no longer exists
> here, reads `filter/ga_final_ranked.csv` which is not in the repo, hardcodes
> `/home/jiasin/...` paths, and its help text still says "all 154" from the
> pre-rename roster. Use the GA repo's equivalents, or fix these up first.

## Stage 4 — filtering

Full documentation, including the funnel table and every evidence file, is in
**[`filter/README.md`](filter/README.md)**. In brief:

```bash
cd filter
python3 run_pipeline.py                 # default order: 03,04,05,01
python3 run_pipeline.py --replay-only   # re-apply gates to existing results
```

The six stages are pose check (`01`), Rosetta off-target binding energy (`02`),
TANGO aggregation (`03`), AllerCatPro 2.0 allergenicity (`04`), ToxinPred3
toxicity (`05`) and NetMHCIIpan + CD4episcore immunogenicity (`06`).

Two of them are not automated. Stage 04 is a manual upload to
<https://allercatpro.bioinf.nl/> — the pipeline writes the FASTA batches, exits
with code 75, and resumes once you drop the result CSV into
`04_allergenicity/outputs/`. Stage 06 is driven by its own scripts and is not
orchestrated by `run_pipeline.py` at all.

**Stage 02 (Rosetta) is deferred** and is not in the default order. It is slow,
needs a working `pyrosetta` env and the full `structures/` tree, and the
selectivity gate it implements is therefore not part of the current panel. Read
the pass set from `final_pass_no_energy`, not `final_pass`.

Two external tools are gitignored and must be installed before the pipeline can
run end to end:

| Expected path | Tool |
|---|---|
| `filter/05_toxicity/tools/toxinpred3/` | ToxinPred3 (required by stage 05) |
| `filter/06_immunogenicity/self_similarity/human_proteome*` | MMseqs2 human proteome DB (orphan — no tracked script reads it) |

The TANGO binary at `filter/03_aggregation/tools/tango_x86_64_release` **is**
tracked, so that stage needs no extra install. A FoldX binary and the AlgPred2 /
ToxinPred2 trees may also be present on disk; nothing in the current pipeline
reads them.

> **Known inconsistency, also flagged in `filter/README.md`:** the canonical six
> candidates in `filter/outputs/final_candidates.csv` cannot be reproduced by
> re-running `run_final_filter.py`. That script's `final_pass` still requires the
> deferred stage 02 and so passes only `GA_006`; `final_pass_no_energy` passes
> six; neither column includes the CD4episcore gate that actually excluded
> `GA_004`.

## Stage 5 — molecular dynamics

MD lives in [`JiaSin-Wu/GPCR_MD`](https://github.com/JiaSin-Wu/GPCR_MD)
(`lab_193:~/4_MD`, which is a plain working copy with no `.git`). AMBER,
ff19SB / lipid21 / TIP3P, POPC:cholesterol 7:3 bilayer, 310 K, 1 bar,
0.15 M NaCl. Chains are A = receptor, B = RAMP, L = peptide; Gα is excluded
from the simulation.

```bash
cd 0_preprocessing
python run_prep.py --inputs inputs --outputs outputs/structures  # AF3 cif -> AMBER pdb
python build_system.py --inputs outputs/structures --outputs outputs/systems

python ../validation/check_system.py \
    outputs/systems/GA_025_AMY1R_model/bilayer_GA_025_AMY1R_model_lipid.top \
    --setting inputs/AMY1R/setting.json

cd ../1_md
python run_amber.py --systems GA_025_AMY1R \
    --stages 00_min 01_min 02_heat1 03_heat2 04_eq 05_prod 06_Long
```

Stages run `00_min` (sander, CPU) then `01_min` → `02_heat1` → `03_heat2` →
`04_eq` ×10 → `05_prod` → `06_Long` on `pmemd.cuda` in the `amber-pmemd`
container, reaching 105,105 ps cumulative. `TIME(PS)` in the `.out` files is
cumulative, so `06_Long` progress is `(TIME − 15105) / 90000`.

Everything in `validation/` exists because the corresponding mistake was made at
least once — missing disulfides, uncapped termini, histidines all set to HIE,
salt silently dropped, an upside-down complex. Run them.

Two operating rules from that repo worth repeating: clear or rename
`1_md/outputs/<system>_model/` before restarting a system, because
`run_amber.py` only copies a file if the destination is absent and will silently
reuse stale input; and packmol is stochastic, so a rebuilt system is a
*different* system, not a reproduction of the old one.

## Environment

There is **no dependency manifest** in this repository — no `environment.yml`,
no `requirements.txt`, no `pyproject.toml`. The only build file is
`model/Dockerfile.worker`. What the code needs, as evidenced by the code itself:

- **Python ≥ 3.10** (PEP 604/585 syntax throughout); lab_193 runs 3.12.3. No
  version is pinned anywhere.
- **Conda environments, referenced by name:** `af3_ml` (stage 01, docking, GA),
  `pyrosetta` (stage 02), `raghava_tools` (stage 05). Stage 03 runs with no env.
- **Packages:** numpy, torch, torch-geometric, joblib, biopython, pyrosetta,
  redis. `model/Dockerfile.worker` pins `redis==7.4.0` and
  `torch-geometric==2.7.0` on `pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime`.
- **Docker** with access to `/var/run/docker.sock`; image
  `jiasin/alphafold3:latest`; AF3 weights at `/opt/model_parameter` and
  databases at `/opt/af3_databases`.
- **An NVIDIA GPU.** `af3_daemon.py` exits immediately without one.
- **MMseqs2** with UniRef90 at `/data/mmseqs2_db/uniref90` (override with
  `MMSEQS_BIN`, `MMSEQS_DB`, `MMSEQS_DB_IDX`, `MMSEQS_THREADS`).
- **Redis** (`redis:7`, port 6379) for the GA's queue mode.

**TODO:** pin this properly in an `environment.yml`.

## Open questions

Things that are genuinely not known, rather than merely undocumented:

1. The exact 3,042 → 92 thresholds, and what wrote `filter/inputs/sequences_ga.csv`.
2. The provenance of `seq.aac`, `seq.dpc` and `Sequence_1` — their contents are
   clear (AAC and DPC composition tables; a FASTA of seven candidates) but no
   script in the repository reads or writes them.
3. Which `structures/` layout the pose check actually consumed.
   `run_docking.py` writes `<seq_id>/<receptor>/`, but the tree on lab_193 is
   organised by receptor.
4. Why `filter/06_immunogenicity/self_similarity/` holds an MMseqs2 human
   proteome database that nothing in the pipeline reads.

## History

This repository was once the whole project. It has since been narrowed to the
AF3 and filtering stages as the GA and MD work moved into repositories of their
own.

| Commit | What it removed |
|---|---|
| `127f1e0` | the `GA/` tree, most of `model/`, the `*_ctail100` AF3 templates — recording the state of the `lab_193` working copy |
| this one | the rest of `model/`, superseded by the GA repo's `grader/inference/` |

Nothing was lost. Everything above is reachable from history:

```bash
git checkout fc709dc  -- GA/ model/   # GA code and the full model tree
git checkout 127f1e0  -- model/       # weights and the gnn package
```
