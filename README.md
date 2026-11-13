# peptide_filtering

Triage of GA-designed AMY receptor (AMY1R/AMY2R/AMY3R) agonist peptides.

A genetic algorithm proposes 37-residue amylin analogues and scores them against
seven CALCR-family receptors; this repository takes the resulting roster and
filters it down to a panel worth simulating.

**The funnel: 3,042 → 92 → 7 → 6.** The six retained candidates are `GA_005`,
`GA_006`, `GA_025`, `GA_078`, `GA_079`, `GA_090`.

Upstream is [`JiaSin-Wu/GA`](https://github.com/JiaSin-Wu/GA); downstream is
[`JiaSin-Wu/GPCR_MD`](https://github.com/JiaSin-Wu/GPCR_MD).

## Filtering

**[`filter/README.md`](filter/README.md) is the documentation** — the funnel
table, every stage, every evidence file. Start there. In brief:

```bash
cd filter
python3 run_pipeline.py                 # default order: 03,04,05,01
python3 run_pipeline.py --replay-only   # re-apply gates to existing results
```

| Stage | Gate | Automated? |
|---|---|---|
| `01_pose_check` | AMY1R/2R/3R pose geometry | yes (`af3_ml` env) |
| `02_binding_energy` | Rosetta off-target ΔG | **deferred — not in the default order** |
| `03_aggregation` | TANGO, no APR | yes |
| `04_allergenicity` | AllerCatPro 2.0 | **no — manual web upload** |
| `05_toxicity` | ToxinPred3 | yes (`raghava_tools` env) |
| `06_immunogenicity` | NetMHCIIpan + CD4episcore | separate scripts, not orchestrated |

Three things that will bite you:

- **Stage 04 pauses the pipeline.** It writes FASTA batches, exits with code 75,
  and waits for you to upload them to <https://allercatpro.bioinf.nl/> and drop
  the result CSV into `04_allergenicity/outputs/`.
- **Read the pass set from `final_pass_no_energy`, not `final_pass`.** The latter
  still requires the deferred stage 02 and so passes only `GA_006`. Neither
  column includes CD4episcore — the gate that actually excluded `GA_004`.
- **The canonical six in `filter/outputs/final_candidates.csv` cannot be
  reproduced by re-running `run_final_filter.py`**, for that reason.

ToxinPred3 must be installed separately at `filter/05_toxicity/tools/toxinpred3/`
(gitignored). The TANGO binary is tracked, so stage 03 needs no extra install.

## Structure prediction

`af3/` folds the candidates with AlphaFold 3 and puts the cifs where
`filter/01_pose_check` reads them: `structures/<seq_id>/<receptor>/<seq_id>_<receptor>_model.cif`
(gitignored, 100+ GB). No daemon: one `docker run` over all inputs.

```bash
python3 af3/run_batch.py --msa-only           # 1. peptide MSAs -> data/af3_ref/ligand/<id>.a3m
python3 af3/run_batch.py --dry-run            # 2. write the complex JSONs to af3/input/
python3 af3/run_batch.py                      # 3. fold, copy cifs into structures/
python3 af3/run_batch.py --seq GA_005 --receptor CTR --skip-done
```

Default scope is the six candidates (GA_005, 006, 025, 078, 079, 090) against CTR.

- Complex = receptor (chain A) + G protein (chain G) + peptide (chain `L`, 37 aa with
  C-terminal NH2), assembled from the template in `data/af3_ref/<RECEPTOR>/` by
  `af3/build_inputs.py`. Receptor and G-protein MSAs are already in the template.
- The peptide MSA is searched once with AF3's own data pipeline (jackhmmer, CPU only)
  and cached per peptide, shared by all receptors. Folding runs with the data
  pipeline off.
- **Setup: create `.env` first.** The scripts need a Docker image with AlphaFold 3, the
  weights and the genetic databases, and there are no built-in paths. Copy the template
  and edit it for your machine (`.env` is gitignored, so it is never committed):

  ```bash
  cp .env.example .env     # then set AF3_IMAGE, AF3_MODEL_DIR, AF3_DB_DIR
  ```

  `AF3_MSA_CPUS` and `AF3_MEM_FRACTION` are optional. A missing required variable stops
  the script with a message naming it.

`data/af3_ref/` holds one AF3 template per receptor (219 MB), read by
`af3/build_inputs.py`. They are also used by the GA repo's oracle, which gitignores
its own byte-identical copy, so this history is the only version-controlled one.
Before dropping them, commit them there. `data/af3_ref/ligand/` caches the peptide MSAs.

## Environment

No dependency manifest exists — no `environment.yml`, no `requirements.txt`.
What the code needs, as evidenced by the code:

- Python ≥ 3.10 (lab_193 runs 3.12.3; nothing is pinned)
- conda envs by name: `af3_ml` (stage 01), `pyrosetta` (stage 02),
  `raghava_tools` (stage 05); stage 03 needs none
- numpy, biopython, pyrosetta
- Docker and an NVIDIA GPU for AF3 (the peptide MSA step is CPU only)

## Open questions

1. **The 3,042 → 92 thresholds.** Described as "three on-target floors and four
   off-target ceilings", but the values are not recorded in any script or config
   here or in the GA repo, and nothing that writes `filter/inputs/sequences_ga.csv`
   could be found. The 92 rows are bounded below by AMY1R 0.4415 / AMY2R 0.4185 /
   AMY3R 0.3972 and above by CTR 0.2977 / CGRP 0.6730 / AM1R 0.4983 /
   AM2R 0.4990 — observed bounds, not the intended cut-offs.
2. Which `structures/` layout the pose check consumed historically. The pose check
   reads `<seq_id>/<receptor>/` (what `af3/run_batch.py` writes); the tree on lab_193
   is organised by receptor.

## History

This repository was once the whole project, and has been narrowed as the GA and
MD work moved into repositories of their own. Nothing was lost — `fc709dc` still
has the `GA/` tree and the full `model/` tree, and `127f1e0` has the GNN weights
and the `gnn` package:

```bash
git checkout fc709dc -- GA/ model/
```

Those weights are worth knowing about: the copies in
`~/2_GA/grader/inference/model/models/` and `~/gpcr-agonism-gnn/checkpoints/` are
byte-identical but **gitignored**, so this repository's history is the only
version-controlled copy.
