# Peptide filtering pipeline

This directory contains the post-GA filtering workflow for the 92 candidates
in `inputs/sequences_ga.csv`. The current thesis panel contains six peptides:

`GA_005`, `GA_006`, `GA_025`, `GA_078`, `GA_079`, and `GA_090`.

`GA_004` passed aggregation, allergenicity, toxicity, and the AMY1R/AMY2R/
AMY3R pose check, but failed the prespecified CD4episcore rule. Seven of its
eight returned windows had Combined Score < 43 (minimum 31.04), so it is not
in the final panel. Its already-completed MD results are retained separately
in the thesis for transparent reporting.

## Start here

The consolidated, human-readable results are in `outputs/`:

| File | Meaning |
|---|---|
| `outputs/filtering_summary.csv` | Funnel counts: 3,042 → 92 → 7 → 6. |
| `outputs/final_candidates.csv` | Canonical six-candidate final panel. |
| `outputs/final_results.csv` | Full 92-candidate integrated table. |
| `outputs/pose_results_amy123r.csv` | Link to the 21-row, sequence-verified AMY1R/2R/3R pose result. |

Use `outputs/final_candidates.csv` when the question is "which candidates
remain?" The historical `final_pass` and `final_pass_no_energy` columns in
`outputs/final_results.csv` do not include CD4episcore as a hard gate and
must not be used alone to recover the current six-candidate panel.

## Current funnel

| Stage | Result | Canonical evidence |
|---|---:|---|
| Receptor-probability preselection | 92 / 3,042 | `inputs/sequences_ga.csv` |
| TANGO aggregation | 7 / 92 | `03_aggregation/outputs/tango_results.csv` |
| AllerCatPro 2.0 | 7 / 7 | `04_allergenicity/outputs/AllerCatPro2_prediction_2231213.csv` |
| ToxinPred3 hybrid model | 7 / 7 | `05_toxicity/outputs/toxinpred3_raw_ga.csv` |
| NetMHCIIpan + CD4episcore | 6 / 7 | `06_immunogenicity/outputs/cd4episcore_summary.csv` |
| AMY1R/2R/3R pose | 7 / 7 candidates; 21 / 21 structures | `01_pose_check/outputs/results_from_md_af3.csv` |
| Final panel | 6 / 7 | `outputs/final_candidates.csv` |

The pose check is a geometry sanity check, not evidence of affinity or
agonism. Off-target Rosetta energy screening is deferred and is not part of
the current six-candidate panel. Consequently, the retained candidates are
computationally prioritized candidates, not experimentally confirmed
selective agonists.

## Directory layout

```text
filter/
├── README.md
├── filter_lib.py
├── run_pipeline.py
├── run_final_filter.py
├── inputs/                 # canonical 92-candidate roster
├── metadata/               # ID provenance
├── outputs/                # consolidated results; start here
├── 01_pose_check/          # pose script and verified results
├── 02_binding_energy/      # deferred Rosetta workflow
├── 03_aggregation/         # TANGO
├── 04_allergenicity/       # AllerCatPro manual input/output
├── 05_toxicity/            # ToxinPred3
└── 06_immunogenicity/      # NetMHCIIpan + CD4episcore
```

Large third-party executables and databases are ignored by Git. They are
runtime dependencies, not thesis results.

Some ignored legacy tool installations (for example AlgPred2 or ToxinPred2)
may remain on a particular machine as environment assets. They are not read
by the current pipeline and are not canonical inputs or results.

## The two runners are different

### `run_pipeline.py`

This is the stage orchestrator. It executes a sequential funnel so that only
survivors enter the next stage. Its actual default order is `03,04,05,01`:
TANGO → AllerCatPro → ToxinPred3 → pose. AllerCatPro is manual, so the runner
can pause and emit FASTA files for web submission. Stage 02 (Rosetta) must be
requested explicitly and is currently deferred. Stage 06 (immunogenicity) is
not orchestrated by this runner.

Each run writes checkpoints and a report under
`pipeline_state/<stage-order>/`; it does not write the canonical
`outputs/final_results.csv`.

### `run_final_filter.py`

This is a one-shot integrator. It does not call prediction tools; it reads
the results already present on disk and writes
`outputs/final_results.csv`. Its legacy rollup remains
`amy_pose_pass AND offt_pass AND safety_pass`, where `safety_pass` contains
TANGO, ToxinPred3, and AllerCatPro but not CD4episcore. Therefore rerunning it
does not replace `outputs/final_candidates.csv` as the canonical final-panel
decision.

## Current stage commands

Run these from `filter/` unless noted otherwise:

```bash
# Orchestrate the default sequence/pose funnel
python3 run_pipeline.py

# Replay existing results without external calls
python3 run_pipeline.py --replay-only

# TANGO
python3 03_aggregation/run_tango.py

# ToxinPred3
conda run -n raghava_tools python3 05_toxicity/run_toxinpred3.py

# NetMHCIIpan
python3 06_immunogenicity/run_netmhciipan_sb.py

# Integrate a manually downloaded CD4episcore export
python3 06_immunogenicity/integrate_cd4episcore.py

# Rebuild the legacy 92-row integrated table from existing stage results
python3 run_final_filter.py
```

Do not rerun `run_final_filter.py` expecting it to make the six-candidate
decision until CD4episcore is intentionally added to the rollup logic.

## Manual stages

- AllerCatPro 2.0 has no API in this workflow. Upload FASTA manually and save
  the downloaded CSV under `04_allergenicity/outputs/`.
- CD4episcore is also a manual web submission. NetMHCIIpan binder windows are
  submitted, then the downloaded result is integrated with
  `integrate_cd4episcore.py`.

## Reproducibility notes

- Candidate IDs and sequences are authoritative in
  `inputs/sequences_ga.csv`.
- The old GA_004 AllerCatPro sequence was replaced; the current GA_004 full
  sequence is `KCNTATCATQRLAEFLVIHSYNNMATCWFTNVGSKTY`.
- All seven AMY2R rerun CIF peptide chains were checked directly against the
  canonical sequences before the pose result was accepted.
- `metadata/id_rename_map_20260826.csv` preserves the 2026-08-26 ID mapping.
