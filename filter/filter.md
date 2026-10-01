# GA v2 — Filter Pipeline

> **2026-08-26: roster reset, re-expanded, then renamed.** The original 154
> candidates (`GA_001`–`GA_154`) were removed from every file in this
> pipeline — their source GA run (referenced elsewhere as "run_003") isn't
> present anywhere on this machine, so their provenance couldn't be
> verified, and they were judged stale. The roster was rebuilt from
> `../2_GA/runs/run_001/` (60 generations, pop_size 50, 3042 unique
> sequences evaluated), keeping every unique sequence that satisfies the
> AMY/off-target probability thresholds originally used to pick the first
> 10 candidates from that run:
> `P_AMY1R>0.438`, `P_AMY2R>0.3965`, `P_AMY3R>0.355`, `P_CTR<0.330`,
> `P_CGRP<0.684`, `P_AM1R<0.501`, `P_AM2R<0.604` (the GA's own fitness
> scores in `inputs/sequences_ga.csv`, not this filter pipeline's pose/Rosetta
> scores) — **92 candidates total**. All 92 (old and new) were then
> **renamed** `GA_001`–`GA_092`, sorted by descending GA fitness — the old
> `GA_155`–`GA_246` numbering (a mix of an original hand-picked batch and
> newly-assigned IDs) no longer means anything and doesn't appear anywhere
> in this repo any more. If you have an old ID from a conversation or
> external note, the full old→new mapping is in
> `metadata/id_rename_map_20260826.csv` (old_id,new_id, 92 rows).
> **The candidate that currently passes every gate was `GA_160`, now
> `GA_006`.**
>
> A full backup of the pre-reset state (the old 164-candidate roster, every
> stage's outputs) lives at
> `../3_Filter_backup_before_154_removal_20260826/`. The old 3-filter-scheme
> history that only covered the removed 154 (`ga_final_ranked.csv` and the
> "Superseded content" section that used to sit at the bottom of this file)
> was deleted along with them — it's still in that backup if needed.
>
> **01/02 data gap**: only 6 of the 92 (`GA_005`, `GA_006`, `GA_025`,
> `GA_078`, `GA_079`, `GA_090` — the old `GA_159`–`GA_164`) have any
> pose-check/Rosetta data at all — `structures/` (AF3 docking output) isn't
> on this machine for anyone else. The other 86 are conservatively
> `amy_pose_pass=0` / `final_pass=0` until docking is run for them
> elsewhere and the results copied in.

## Overview

**Goal**: from 92 GA-evolved amylin-analog sequences (`GA_001`–`GA_092`,
from `../2_GA/runs/run_001/`, ranked by descending GA fitness), find
candidates that (a) dock correctly on
the three amylin receptors (AMY1R/2R/3R), (b) are *less* likely to bind the
four related off-target receptors (CTR, CGRP, AM1R, AM2R) than native
amylin, and (c) are safe (non-aggregating, non-toxic, non-allergenic).

All sequences are the **full 37aa construct**: fixed prefix `KCNTATCATQRLA`
(13aa) + 24aa GA-evolved variable region. `amylin` (native `KCNTATCATQRLA` +
`NFLVHSSNNFGAILSSTNVGSNTY`) is carried through 01/02 as the structural/
binding-energy reference (see the roster-reset note above for why 06 no
longer queries it).

Docking structures (AlphaFold3 complexes, receptor + peptide ± RAMP ± Gα) live
in `../structures/<seq_id>/<receptor>/` — one folder per sequence, one
subfolder per receptor.

## Pipeline

The pipeline is now a **modular, order-configurable funnel** run by
`run_pipeline.py`, not five independent stages merged by AND at the end.
Each of the five gates below is a small function in `filter_lib.py`
(`gate_amy_pose`, `gate_offtarget`, `gate_tango`, `gate_aller`,
`gate_tox3`); `run_pipeline.py --order <comma list>` runs them in whatever
sequence you name, only handing survivors of stage N to stage N+1, and
records which stage eliminated each candidate. Because boolean AND is
order-independent, any `--order` permutation converges on the same final
surviving set — verified with `--replay-only` (see Re-running below).

**Default order: `03,04,05,01`** (`02` deliberately left out of the
default, see the note under [Final integration](#final-integration--run_final_filterpy)
— it still gates `final_pass`, just isn't auto-run) — the three cheap,
sequence-only checks (aggregation/allergenicity/toxicity) run first to
eliminate weak candidates before spending time on structure-dependent
checks:

```
92 candidates (sequences_ga.csv) + amylin reference, docked in ../structures/
  │
  ├─ 03 Aggregation     (TANGO)              → gate_tango      → tango_pass
  ├─ 04 Allergenicity   (AllerCatPro 2.0)     → gate_aller      → aller_pass
  ├─ 05 Toxicity        (ToxinPred3)          → gate_tox3       → tox3_pass
  ├─ 01 Pose check      (geometry, AMY only)  → gate_amy_pose   → amy_pose_pass
  └─ 02 Binding energy  (Rosetta ΔG_AB)       → gate_offtarget  → offt_pass
             │  (each stage only processes survivors of the previous one)
     run_pipeline.py writes pipeline_state/<order>/final_results.csv;
     run_final_filter.py writes outputs/final_results.csv
             │
       outputs/final_results.csv  (92 rows, one per GA sequence)
```

`01`'s gate only eliminates on the 3 AMY receptors — a candidate that fails
to dock an off-target receptor is *safer* there, not worse, so that check
belongs to `02`'s gate (`offt_pass`), which reads 01's already-recorded
off-target `pose_pass` values together with its own Rosetta `dG_AB_REU`.
01 still scores and records all 7 receptors regardless of gate scope.

`06_immunogenicity` is a separate, non-gating step run afterward only on
whoever survives all five gates (`final_pass=1`) — see
[06 Immunogenicity](#06-immunogenicity--ranking-metric-not-a-hard-gate).

### Current numbers (`final_results.csv`, 92 GA sequences)

| Gate | Pass |
|---|---|
| AMY pose (all 3 amylin receptors dock correctly) | 7 / 92 |
| Off-target selectivity (all 4 off-targets) | 87 / 92 |
| Safety (TANGO + ToxinPred3 + AllerCatPro2) | 6 / 92 |
| **FINAL PASS (all of the above)** | **1 / 92** |
| **FINAL PASS (no energy)** — `final_pass_no_energy`, AMY pose + safety only | **6 / 92** |

Final candidate (`final_pass=1`): `GA_006` (old `GA_160`).

**2026-08-28 — stage 02 (binding energy) deferred, not run for this batch.**
`final_pass` still requires `offt_pass`, so it currently passes only
`GA_006`; the other 5 that clear AMY pose + safety are **not evaluated for
binding energy yet, not rejected**. `final_results.csv` carries a
`final_pass_no_energy` column (`= amy_pose_pass AND safety_pass`, off-target
selectivity dropped since it is meaningless without ΔG_AB) recording this
set: `GA_005, GA_006, GA_025, GA_078, GA_079, GA_090` (old `GA_159`–`GA_164`,
). Re-run stage 02 and the column becomes redundant with `final_pass` again.

Immunogenicity (n_sb_lt2pct, NetMHCIIpan 4.1 BA Strong Binders, run
2026-08-26 for `GA_006`, old `GA_160`): **1** strong binder, `n_wb_lt10pct = 8` — see
[06 Immunogenicity](#06-immunogenicity--ranking-metric-not-a-hard-gate).

---

## Stages

### 01 Pose check — `01_pose_check/run_pose_check.py`

Geometric sanity check on the AF3 docking pose, per `(seq_id, receptor)`.
Reads Cα coordinates straight out of the `.cif` model (chain A = receptor,
chain B = peptide).

- **N-term check**: peptide residues 1–13 (chain B) must sit within **8 Å** of
  the receptor TMD (chain A, residues after the ECD — TM1 starts at 132 for
  CTR-like chains of length 474, 130 for CLR-like chains of length 461).
- **C-term check**: peptide residues 26–37 (chain B) must sit within **8 Å** of
  the receptor's deep ECD groove (chain A, residues 80–131 for CTR / 80–129 for
  CLR — ECD boundaries from GPCRdb + UniProt signal peptide annotation).
- **`pose_pass`** = both checks pass.
- Output: `01_pose_check/outputs/results.csv` (up to `n_candidates × 7` rows — one
  per `(seq_id, receptor)` pair that has a docked structure).
- Run: `conda run -n af3_ml python3 filter/01_pose_check/run_pose_check.py`

### 02 Binding energy — `02_binding_energy/run_foldx.py`

> **This section still describes the original FoldX engine.** The pipeline
> switched to Rosetta's `InterfaceAnalyzerMover` (dG_AB in REU, not FoldX's
> kcal/mol) some time ago — see
> `02_binding_energy/rosetta/run_rosetta_iface_parallel.py`, which is what
> `run_pipeline.py`'s `02` gate and `filter_lib.gate_offtarget()` actually
> call/read (`rosetta_results.csv`, not `foldx_results.csv`). FoldX is kept
> only as a reference/cross-check engine; the amylin baseline table below is
> FoldX's, not the one currently used for gating (see
> [Current numbers](#current-numbers-final_resultscsv-92-ga-sequences) for
> the live Rosetta-based pass counts). Updating this whole section to
> describe Rosetta as primary is a separate documentation task, not part of
> the `run_pipeline.py` order/module refactor.

FoldX interaction energy on the same AF3 structures.

- Per structure: CIF→PDB (biopython, chains A/B/C only, Gα chain D dropped) →
  `FoldX RepairPDB` → `FoldX AnalyseComplex --analyseComplexChains=A,B,C`.
- Reports pairwise ΔG (kcal/mol) for **A↔B** (receptor↔peptide, the one that
  matters), A↔C (receptor↔RAMP), B↔C (peptide↔RAMP).
- Output: `02_binding_energy/outputs/foldx_results.csv` (`dG_AB`, `dG_AC`, `dG_BC` per
  seq × receptor). `amylin` row is the reference baseline for every receptor.
- Run: `conda run -n PDBFixer python3 filter/02_binding_energy/run_foldx.py
  [--seq ID1,ID2] [--skip-done] [--workers N]` (~143s/structure, parallelizable).

Amylin reference ΔG_AB (kcal/mol), used as the cutoff everywhere below:

| AMY1R | AMY2R | AMY3R | CTR | CGRP | AM1R | AM2R |
|---|---|---|---|---|---|---|
| -37.74 | -43.59 | -35.24 | -38.14 | -34.10 | -30.70 | -36.87 |

### 03 Aggregation — TANGO

- **Tool**: `tango_x86_64_release` binary. `run_tango.py` calls it directly
  (no conda env needed); if the binary's executable bit isn't preserved on
  your filesystem (e.g. a 9p/exFAT mount), copy it somewhere local and
  point `TANGO_BIN=<path>` at the copy — `run_tango.py` respects that env
  var instead of the default `tools/tango_x86_64_release`.
- **Conditions**: pH 7.4, 310 K, ionic strength 0.1 M, C-term amidated.
- **Rule**: FAIL if any ≥5 consecutive residues score >5% aggregation
  (TANGO's own APR — Aggregation-Prone Region — definition).
- Output: `03_aggregation/outputs/tango_results.csv` and
  `03_aggregation/outputs/tango_passed.csv`, read directly by
  `filter_lib.gate_tango()`.
- Current pass rate (92 candidates, re-run 2026-08-26): 7/92.

### 04 Allergenicity — AllerCatPro 2.0

- **Tool**: AllerCatPro 2.0 **web server** (manual — not scriptable). Inputs
  are ≤50-sequence FASTA batches in `04_allergenicity/inputs/`; currently
  just `ga_sequences_part5.fa` (the 6 old-`GA_159`–`GA_164` candidates,
  now `GA_005`/`GA_006`/`GA_025`/`GA_078`/`GA_079`/`GA_090`) — the other 86
  (including old `GA_155`–`GA_158`) have never been submitted, so they're
  conservatively `aller_pass=0`
  until someone uploads them.
- **Pass**: server verdict is "no evidence of being allergenic".
- Outputs manually downloaded to
  `04_allergenicity/outputs/AllerCatPro2_prediction_*.csv` — `run_final_filter.py`
  globs all of them, so a new batch's output file just needs to land in that
  folder, whatever it's named.
- Current pass rate: 6/6 submitted (4 not yet submitted).

### 05 Toxicity — ToxinPred3 — `05_toxicity/run_toxinpred3.py`

- **Tool**: `toxinpred3.py`, conda env `raghava_tools`, **Model 2 (Hybrid
  Score)**.
- **Pass**: Hybrid Score < 0.38 → "Non-Toxin". Native amylin baseline = 0.35
  (passes).
- Output: `05_toxicity/outputs/toxinpred3_raw_ga.csv` (`Subject`, `Prediction`
  columns consumed directly by `run_final_filter.py`).
- Run: `conda run -n raghava_tools python3 filter/05_toxicity/run_toxinpred3.py`
- Current pass rate (92 candidates, re-run 2026-08-26): 92/92.
- **Note**: ToxinPred2 (Model 1, threshold 0.5) was tried first and abandoned
  on the original (since-removed) candidate batch — it scored nearly all of
  them as Toxin (0.80–0.95), a false-positive artifact of W/F/R-rich motifs
  being out-of-distribution for its training data. The leftover
  `toxinpred2_*.csv` outputs and AlgPred2 (superseded by AllerCatPro2) were
  deleted along with that batch on 2026-08-26 — see the roster-reset note
  at the top of this file.

### 06 Immunogenicity — ranking metric, not a hard gate

**Current / finalized method — `run_netmhciipan_sb.py`**

Queries **NetMHCIIpan 4.1 BA** (`method=netmhciipan_ba`, IEDB's own
recommended binding predictor, 2023.09) via the **new** IEDB Next-Generation
Tools API (`https://api-nextgen-tools.iedb.org/api/v1/pipeline`) — no SSH
tunnel needed, this host is directly reachable. 7-allele DRB reference panel
(`HLA-DRB1*03:01/07:01/15:01`, `HLA-DRB3*01:01/02:02`, `HLA-DRB4*01:01`,
`HLA-DRB5*01:01`), 15-mer sliding window across the full 37aa sequence, run
on the 7 `final_pass` candidates + amylin reference in one call.

**Step 1 metric: `n_wb_lt10pct`** — count of (peptide × allele) pairs with
**percentile rank < 10%** (IEDB's general binder-flagging recommendation,
Paul et al. 2013, doi:10.1155/2013/467852). Lower rank = stronger predicted
binding; this just identifies which 15-mer windows are plausible MHC-II
binders, it says nothing yet about whether they're actually immunogenic —
that's Step 2 below.

2026-08-26: dropped the previously-used `n_sb_lt2pct` ("Strong Binder",
rank<2%) metric entirely. It was added earlier because `n_wb_lt10pct` alone
made several candidates in an older, smaller batch indistinguishable. But
n_SB turned out to be uninformative on its own terms: with ~161 (peptide ×
allele) combinations per candidate, a rank<2% cutoff has a ~2% background
hit rate by construction (~3 expected by pure chance), so the 0-1 values
every candidate showed were consistent with noise, not signal. `n_wb_lt10pct`
already shows real spread (0 to 23 across current candidates) and is what
the cited paper actually recommends using — see `run_netmhciipan_sb.py`'s
docstring for the full reasoning.

Note: this script no longer submits `amylin` as a baseline comparison —
it only queries whatever's in `final_pass` (or `--seq`), and the summary
no longer has a `sb_vs_amylin` column. Use it *comparatively* (rank
candidates against each other by `n_wb_lt10pct`), not against a fixed
cutoff — Paul et al. 2013 explicitly doesn't give one; see its methodology.

Outputs: `outputs/netmhciipan_sb_raw.csv` (full per-window × allele table),
`outputs/netmhciipan_sb_summary.csv` (`n_wb_lt10pct` per sequence) — read by
`integrate_cd4episcore.py` (Step 2 below) and available for manual
comparison, but **not merged into `final_results.csv`** any more: as of
2026-08-26 that file only carries pass/fail columns (see
[Final integration](#final-integration--run_final_filterpy)), no raw
scores/counts, so `n_wb_lt10pct` has to be read from this summary CSV
directly if you want it.

Run: `python3 06_immunogenicity/run_netmhciipan_sb.py` (also fixed a
`ROOT/"filter"` path bug — same issue `run_final_filter.py` had — that made
it unable to find `final_results.csv` at all before 2026-08-26.)

**Method note, 2026-08-26**: briefly switched `METHOD` to IEDB's
**Consensus** (median percentile rank across NN-align/SMM-align/Tepitope/
Comblib) on the strength of Paul et al. 2013, which found it beats each
2013-era constituent method individually. Reverted the same day after
reading Reynisson et al. 2020 (Nucleic Acids Res 48:W449,
doi:10.1093/nar/gkaa379) — NetMHCIIpan 4.0/4.1 is a single method but
trained on 4.1M data points across 116 MHC-II molecules, a full generation
newer than Consensus's four constituent methods, and IEDB's current default
recommendation. **Stick with `netmhciipan_ba`.**

**Step 2 — CD4episcore Combined Score, `integrate_cd4episcore.py`**

Takes the `n_wb_lt10pct` binder peptides from Step 1 and judges which ones
carry a real immunogenicity risk, using CD4episcore
(https://tools.iedb.org/CD4episcore/).

Status as of 2026-08-26: **CD4episcore works**, confirmed via the Legacy
web form (manual — no API access, see below). This directly contradicts
the previous finding on this page (both the new API and the Legacy form
were confirmed broken 2026-07-04, "Completion flag file missing" /
silent no-op) — whatever the server-side issue was, it's since been fixed
or worked around. The old incomplete `cd4episcore_results.csv` /
`cd4episcore_results_v2.csv` (23/42 and 8/17 peptide coverage) from that
broken period are stale artifacts, superseded by
`outputs/cd4episcore_results_v3.csv`.

**Rule** (Dhanda et al. 2018, Frontiers in Immunology,
doi:10.3389/fimmu.2018.01369, Table 2): Combined Score is a percentile-style
score, like MHC percentile rank — **lower values mean higher predicted
immunogenicity risk**, confirmed directly from the paper's methods
(`Imm_score = (1 − NN_output) × 100`, inverting the NN's own "high = more
immunogenic" output onto the same "low = risky" scale as the HLA
percentile-rank component it's averaged with). This is the *opposite* of
what an earlier analysis pass in this project assumed (flagging Combined
Score ≥ 43 as high-risk) — that was backwards; get the direction wrong and
every conclusion about which candidates look safe inverts.

  | Threshold | Sensitivity | Specificity |
  |---|---|---|
  | 8  | 20% | 91% |
  | 18 | 31% | 85% |
  | 36 | 51% | 65% |
  | **43** | **59%** | **59%** |
  | 66 | 75% | 37% |

  **43 is the paper's own recommended cutoff** — the balanced-accuracy
  point where sensitivity equals specificity — not an arbitrary choice.

  Decision rule: `Combined Score < 43` → that peptide is a predicted
  immunogenicity risk; `>= 43` → acceptable. Per candidate: `cd4_pass = 0`
  if **any** of its `n_wb_lt10pct` peptides scores `< 43`, else `1`. A
  candidate with zero binder peptides is trivially `cd4_pass = 1`,
  and one whose binder peptides haven't been submitted to CD4episcore yet
  gets `status=incomplete` with `cd4_pass` left blank (not a false pass).

Current results (the 6 candidates surviving 03→04→05 as of 2026-08-26,
i.e. before 01/02 — see [Current numbers](#current-numbers-final_resultscsv-92-ga-sequences)):
all 5 with any binder peptides (`GA_078` has none) score well above 43
(min 47.23-58.43) — **`cd4_pass=1` for all of them**. No candidate is
currently flagged as an immunogenicity risk by this rule.

**CD4episcore still can't be called programmatically** — the new API's
`immunogenicity` predictor was not retested this session, and there's no
scripted way to submit to the Legacy web form, so `integrate_cd4episcore.py`
only *consumes* a manually-downloaded CD4episcore export
(`outputs/cd4episcore_results_v3.csv`); it doesn't fetch one. To add more
candidates: get their `n_wb_lt10pct` peptides from
`outputs/netmhciipan_sb_raw.csv` (rank<10%), paste the unique set into
https://tools.iedb.org/CD4episcore/, save the CSV export to
`outputs/cd4episcore_results_v3.csv` (or edit `CD4_CSV` in the script), and
re-run `python3 06_immunogenicity/integrate_cd4episcore.py`.

**Status: reference/ranking only, not wired into `final_pass`** — coverage
is manual and currently only exists for candidates that already had binder
peptides submitted, not the full roster. Promote `cd4_pass` to a hard gate
only once coverage is decided to be complete enough.

**Other superseded paths — do not use**:
- `run_iedb.py`/`run_consensus_epitopes.py` targeted the legacy
  `tools-cluster-interface.iedb.org` API host, which is unreachable (dead
  server, confirmed independently from two separate networks).
- A from-scratch human-proteome self-similarity check (JanusMatrix-style:
  MHC-binding 9aa core vs UniProt human reviewed proteome, local
  brute-force Hamming-distance scan, `06_immunogenicity/self_similarity/`)
  was also tried as an alternative risk-refinement layer, but gave an
  inconclusive result (best matches ~67% identity — not similar enough to
  call "tolerized", not dissimilar enough to rule out) and was not adopted.

This stage is **not applied as a pass/fail filter** — `run_final_filter.py`
only carries `imm_*` / `cd4_pass` columns through to `final_results.csv`
for reference / ranking among the `final_pass` candidates.

---

## Final integration — `run_final_filter.py`

Reads all of the above and writes `final_results.csv` (92 rows, one per
current `sequences_ga.csv` candidate). Per sequence:

- **`amy_pose_pass`** — `pose_pass = 1` for **all three** of AMY1R/AMY2R/AMY3R.
- **`offt_pass`** — for **each** off-target receptor (CTR, CGRP, AM1R, AM2R)
  individually: `pose_pass = 0` **OR** `dG_AB` weaker (higher, less negative)
  than amylin's `dG_AB` for that receptor. Must hold for all 4.
  (i.e. "either it doesn't dock properly on the off-target, or it binds it
  more weakly than native amylin does" = safer than amylin there.)
- **`safety_pass`** — TANGO pass AND ToxinPred3 pass AND AllerCatPro2 pass
  (all three).
- **`amy_pass`** — *reported only, not gating*: pose pass **and** `dG_AB`
  stronger than amylin's, for all three AMY receptors. Would be the "is it a
  better AMY agonist than native amylin" criterion, deliberately not enforced
  at this stage (docking/FoldX affinity isn't a reliable enough proxy for
  agonism; dynamic assessment is reported separately in the thesis MD analysis).
- **`final_pass`** = `amy_pose_pass` AND `offt_pass` AND `safety_pass`.
- **`final_pass_no_energy`** = `amy_pose_pass` AND `safety_pass` (off-target
  selectivity dropped). Added 2026-08-28 while stage 02 is deferred — see
  the note under [Current numbers](#current-numbers-final_resultscsv-92-ga-sequences)
  and `filter_lib.py`'s module docstring. Records the intended pass set
  while `offt_pass`/energy is unevaluated; becomes redundant with
  `final_pass` once 02 is run.

**2026-08-26: `final_results.csv` was stripped down to pass/fail columns
only** — `id`/`seq_24aa`/`full_37aa` plus `pose_*` (per-receptor 0/1),
`tango_pass`/`tox3_pass`/`aller_pass`/`cd4_pass`, and the four gate/rollup
booleans above. The raw score columns that used to sit alongside them
(`ros_AMY1R`...`ros_AM2R`, `ros_amy_avg`, `imm_n_wb_le10pct`) were removed —
decision to just look at who passes, not rank by degree. Those numbers
still exist in their own source files if you need them (Rosetta:
`02_binding_energy/rosetta/rosetta_results.csv`; immunogenicity:
`06_immunogenicity/outputs/netmhciipan_sb_summary.csv` /
`cd4episcore_summary.csv`) — `filter_lib.build_report_rows()` just doesn't
merge them into the wide report any more.

**Also 2026-08-26: `02` (Rosetta) dropped from `run_pipeline.py`'s default
`--order`** (now `03,04,05,01`, was `03,04,05,01,02`) — it's the slowest
stage and needs `structures/` + a working pyrosetta env, neither reliably
available. This does **not** change the `final_pass` formula above, which
still requires `offt_pass` regardless of `--order`; a candidate 02 hasn't
been run for is just conservatively `offt_pass=0`. Run `02` explicitly
(`--order 03,04,05,01,02` or any permutation containing `02`) when you have
both prerequisites. **2026-08-28: 02 is being deferred for the current
batch** — read the pass set off `final_pass_no_energy`, not `final_pass`,
until 02 is run (see [Current numbers](#current-numbers-final_resultscsv-92-ga-sequences)).

This exact formula is implemented once, in `filter_lib.py`'s five `gate_*`
functions (see [Pipeline](#pipeline) above), and shared by both:
- `python3 run_final_filter.py` — one-shot AND over all candidates in
  `sequences_ga.csv`, assuming 01-05 are already fully computed; also the
  equivalence baseline for the funnel (see Verification below).
- `python3 run_pipeline.py [--order ...]` — the sequential funnel, only
  computing each stage for survivors of the previous one.

Both write `final_results.csv` with the same schema (`run_pipeline.py`
additionally writes it under `pipeline_state/<order>/` and adds an
`eliminated_at_stage` column).

---

## Files

| File | Description |
|---|---|
| `inputs/sequences_ga.csv` | 92 GA candidate sequences (`GA_001`–`GA_092`, ranked by descending GA fitness) + per-receptor GA fitness scores (AMY1R–AM2R). |
| `metadata/id_rename_map_20260826.csv` | old_id → new_id mapping from the 2026-08-26 rename (92 rows) |
| `outputs/` | Consolidated human-readable results: funnel summary, six-candidate final panel, full 92-row table, and the AMY123R pose result link. |
| `filter_lib.py` | **Shared library** — id-list I/O, raw-output loaders, and the five `gate_*` functions that define `final_pass`. Both scripts below import it instead of duplicating the logic. |
| `run_pipeline.py` | **Current** — order-configurable sequential funnel (`--order 03,04,05,01` default, `02` excluded by default — see [Final integration](#final-integration--run_final_filterpy)). Runs 01/02/03/05 via `conda run -n <env>`, pauses for the manual 04 (AllerCatPro2) step, writes `pipeline_state/<order>/{alive_after_*.txt,audit.csv,final_results.csv}`. `--replay-only` applies the gates to existing output files with no external tool calls (used for the order-equivalence regression check). |
| `run_final_filter.py` | One-shot integration script → `final_results.csv`, assumes 01-05 already computed for everyone. Also the regression baseline `run_pipeline.py` is checked against. |
| `01_pose_check/run_pose_check.py` | Docking pose geometry check. `--seq`/`--seq-file` narrow to a subset, `--extra-id` always adds extra ids (e.g. `amylin`) regardless of that filtering. |
| `02_binding_energy/rosetta/run_rosetta_iface_parallel.py` | **Current** binding-energy engine — Rosetta `InterfaceAnalyzerMover` dG_AB (REU). One of `--seq`/`--seq-file` is required (no cheap "everything" default, given the per-job cost). `run_rosetta_iface.py` (sequential) and `02_binding_energy/run_foldx.py` (FoldX) are superseded/reference-only — `run_final_filter.py`/`filter_lib.py` only read `rosetta_results.csv`. |
| `03_aggregation/run_tango.py` | TANGO aggregation. Reads `inputs/sequences_ga.csv`; stage results are written under `03_aggregation/outputs/`. |
| `05_toxicity/run_toxinpred3.py` | ToxinPred3 hybrid prediction. Reads `inputs/sequences_ga.csv`; stage results are written under `05_toxicity/outputs/`. |
| `06_immunogenicity/run_netmhciipan_sb.py` | **Current, Step 1** — NetMHCIIpan 4.1 BA, 7-allele, `n_wb_lt10pct` → `netmhciipan_sb_summary.csv` |
| `06_immunogenicity/integrate_cd4episcore.py` | **Current, Step 2** — applies the Combined Score<43 rule to Step 1's binder peptides → `cd4episcore_summary.csv`, reference only (not gating). Consumes a manually-downloaded CD4episcore export; CD4episcore itself works again as of 2026-08-26 (was confirmed broken 2026-07-04) but still isn't scriptable. |
| `06_immunogenicity/outputs/cd4episcore_results_v3.csv` | Manually-downloaded CD4episcore Legacy-form export (2026-08-26) — real, complete data, unlike the stale/incomplete `cd4episcore_results.csv`/`_v2.csv` from the broken-CD4episcore period |
| `06_immunogenicity/run_iedb.py` | Superseded — legacy whole-154 screen, targets dead `tools-cluster-interface.iedb.org` host |
| `06_immunogenicity/run_consensus_epitopes.py` | Superseded — same dead host |

## Re-running

**Preferred — the funnel, via `run_pipeline.py`** (see [Pipeline](#pipeline)):

```bash
# Default order: 03 (aggregation) -> 04 (allergenicity) -> 05 (toxicity)
# -> 01 (pose check). Only survivors of each stage reach the next one.
# 02 (Rosetta binding energy) is NOT in the default -- see the note under
# "Final integration" -- add it explicitly when structures/ + pyrosetta
# are both available:
python3 run_pipeline.py
python3 run_pipeline.py --order 03,04,05,01,02

# Any other order -- just a comma-separated string, no code changes needed:
python3 run_pipeline.py --order 01,02,03,04,05

# 04 (AllerCatPro2) has no API -- when the funnel reaches it and finds
# candidates with no existing outputs/AllerCatPro2_prediction_*.csv
# coverage, it writes a subset FASTA batch under 04_allergenicity/inputs/,
# prints upload instructions, and exits. After uploading to the web tool
# and saving the downloaded CSV into 04_allergenicity/outputs/, continue
# with the *same* --order:
python3 run_pipeline.py --resume

# Regression check: apply the gates to whatever's already on disk, no
# subprocess/external-tool calls at all -- confirms a given --order
# converges on the same final_pass set as run_final_filter.py's one-shot AND.
python3 run_pipeline.py --replay-only --order 03,04,05,01,02
```

**Individual stages** (each also runs standalone, e.g. for debugging one
stage in isolation; `--seq`/`--seq-file` narrow to a subset, default is
everyone in `sequences_ga.csv` except 02 which requires an explicit list):

```bash
# Pose check (requires ../structures/<id>/<receptor>/*.cif from docking)
conda run -n af3_ml python3 filter/01_pose_check/run_pose_check.py

# Rosetta binding energy (current engine; --seq or --seq-file required)
conda run -n pyrosetta python3 filter/02_binding_energy/rosetta/run_rosetta_iface_parallel.py --seq GA_006,amylin --workers 8

# TANGO aggregation
python3 filter/03_aggregation/run_tango.py

# AllerCatPro2 — manual web upload, filter/04_allergenicity/inputs/*.fa → outputs/*.csv

# ToxinPred3
conda run -n raghava_tools python3 filter/05_toxicity/run_toxinpred3.py

# IEDB immunogenicity (new API, no SSH tunnel needed) -- only meaningful
# after the funnel above has produced a final_pass set
python3 filter/06_immunogenicity/run_netmhciipan_sb.py

# One-shot integration (assumes 01-05 already fully computed for everyone)
python3 run_final_filter.py
```

---

## Note on removed history

This file used to end with a "Superseded content (old 3-filter version)"
section documenting the original TANGO+AllerCatPro2+ToxinPred3-only scheme
applied to the since-removed 154 candidates (`ga_final_ranked.csv`). Both
were deleted 2026-08-26 along with that candidate batch — see the
roster-reset note at the top of this file if you need to recover them from
`../3_Filter_backup_before_154_removal_20260826/`.
