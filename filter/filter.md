# GA v2 — Filter Pipeline

> Supersedes the earlier 3-filter version of this document (TANGO + AllerCatPro2 +
> ToxinPred3 only, 38/154 pass). The pipeline has since been extended with pose
> checking, FoldX binding energy, and off-target selectivity gating. See
> [Superseded content](#superseded-content-old-3-filter-version) at the bottom for
> the historical record.

## Overview

**Goal**: from 154 GA-evolved amylin-analog sequences (run_003, gen 0–65), find
candidates that (a) dock correctly on the three amylin receptors (AMY1R/2R/3R),
(b) are *less* likely to bind the four related off-target receptors (CTR, CGRP,
AM1R, AM2R) than native amylin, and (c) are safe (non-aggregating, non-toxic,
non-allergenic).

All sequences are the **full 37aa construct**: fixed prefix `KCNTATCATQRLA`
(13aa) + 24aa GA-evolved variable region. `amylin` (native `KCNTATCATQRLA` +
`NFLVHSSNNFGAILSSTNVGSNTY`) is carried through every stage as the reference.

Docking structures (AlphaFold3 complexes, receptor + peptide ± RAMP ± Gα) live
in `../structures/<seq_id>/<receptor>/` — one folder per sequence, one
subfolder per receptor, 7 receptors × 155 sequences (154 GA + amylin).

## Pipeline

```
155 structures (154 GA sequences + amylin reference) × 7 receptors, docked in ../structures/
  │
  ├─ 01 Pose check     (geometry)         → pose_pass per (seq, receptor)
  ├─ 02 Binding energy  (FoldX ΔG_AB)      → dG_AB per (seq, receptor)
  ├─ 03 Aggregation     (TANGO)            → tango_pass
  ├─ 04 Allergenicity   (AllerCatPro 2.0)  → aller_pass
  ├─ 05 Toxicity        (ToxinPred3)       → tox3_pass
  └─ 06 Immunogenicity  (IEDB NetMHCIIpan 4.1 BA, n_sb_lt2pct) → ranking metric only, not gating
             │
     run_final_filter.py integrates 01+02+03+04+05 (06 reported for reference)
             │
       final_results.csv  (154 rows, one per GA sequence)
```

`07_md` (molecular dynamics) exists as an empty stub — planned next validation
step for whatever survives the filter above, not yet started.

### Current numbers (`final_results.csv`, 154 GA sequences)

| Gate | Pass |
|---|---|
| AMY pose (all 3 amylin receptors dock correctly) | 78 / 154 |
| Off-target selectivity (all 4 off-targets) | 79 / 154 |
| Safety (TANGO + ToxinPred3 + AllerCatPro2) | 38 / 154 |
| **FINAL PASS (all of the above)** | **7 / 154** |

Final 7: `GA_051, GA_074, GA_085, GA_086, GA_087, GA_088, GA_104`

Immunogenicity (n_sb_lt2pct, NetMHCIIpan 4.1 BA Strong Binders, vs amylin=4):
all 7 final candidates score **0** — see [06 Immunogenicity](#06-immunogenicity--ranking-metric-not-a-hard-gate).

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
- Output: `01_pose_check/results.csv` (1085 rows = 155 seqs × 7 receptors).
- Run: `conda run -n af3_ml python3 filter/01_pose_check/run_pose_check.py`

### 02 Binding energy — `02_binding_energy/run_foldx.py`

FoldX interaction energy on the same AF3 structures.

- Per structure: CIF→PDB (biopython, chains A/B/C only, Gα chain D dropped) →
  `FoldX RepairPDB` → `FoldX AnalyseComplex --analyseComplexChains=A,B,C`.
- Reports pairwise ΔG (kcal/mol) for **A↔B** (receptor↔peptide, the one that
  matters), A↔C (receptor↔RAMP), B↔C (peptide↔RAMP).
- Output: `02_binding_energy/foldx_results.csv` (`dG_AB`, `dG_AC`, `dG_BC` per
  seq × receptor). `amylin` row is the reference baseline for every receptor.
- Run: `conda run -n PDBFixer python3 filter/02_binding_energy/run_foldx.py
  [--seq ID1,ID2] [--skip-done] [--workers N]` (~143s/structure, parallelizable).

Amylin reference ΔG_AB (kcal/mol), used as the cutoff everywhere below:

| AMY1R | AMY2R | AMY3R | CTR | CGRP | AM1R | AM2R |
|---|---|---|---|---|---|---|
| -37.74 | -43.59 | -35.24 | -38.14 | -34.10 | -30.70 | -36.87 |

### 03 Aggregation — TANGO

- **Tool**: `tango_x86_64_release` binary.
- **Conditions**: pH 7.4, 310 K, ionic strength 0.1 M, C-term amidated.
- **Rule**: FAIL if any ≥5 consecutive residues score >5% aggregation
  (TANGO's own APR — Aggregation-Prone Region — definition).
- Result lives in the `aggregation` column of `filter_tracker_ga.csv` (1 = PASS).
- Key finding from the original run: sequences containing `WFCFR` (64% of the
  154) FAIL — `LAWFCF` at positions 12–17 forms a 15–17% APR. Sequences with
  `WFSFR` (Ser instead of Cys, 23%) PASS — the serine breaks the hydrophobic
  stretch. Native amylin's max residue score is 2.6% (PASS).

### 04 Allergenicity — AllerCatPro 2.0

- **Tool**: AllerCatPro 2.0 **web server** (manual — not scriptable). Inputs
  split into 4 FASTA files of ≤50 sequences each in `04_allergenicity/inputs/`.
- **Pass**: server verdict is "no evidence of being allergenic".
- Outputs manually downloaded to
  `04_allergenicity/outputs/AllerCatPro2_prediction_*.csv` (4 files, one per
  batch) — `run_final_filter.py` globs all of them.
- Historical pass rate on the original 154: 154/154.

### 05 Toxicity — ToxinPred3 — `05_toxicity/run_toxinpred3.py`

- **Tool**: `toxinpred3.py`, conda env `raghava_tools`, **Model 2 (Hybrid
  Score)**.
- **Pass**: Hybrid Score < 0.38 → "Non-Toxin". Native amylin baseline = 0.35
  (passes).
- Output: `05_toxicity/outputs/toxinpred3_raw_ga.csv` (`Subject`, `Prediction`
  columns consumed directly by `run_final_filter.py`).
- Run: `conda run -n raghava_tools python3 filter/05_toxicity/run_toxinpred3.py`
- **Note**: ToxinPred2 (Model 1, threshold 0.5) was tried first and abandoned —
  it scored all 154 GA sequences as Toxin (0.80–0.95), a false-positive
  artifact of W/F/R-rich motifs being out-of-distribution for its training
  data. Leftover ToxinPred2 outputs (`toxinpred2_*.csv`,
  `filter_tracker_ga.csv` columns `toxicity`/`tox_score`) are **not** used by
  the final filter — kept only for reference. Same story for AlgPred2
  (`allergenicity`/`alg_score` columns) — superseded by AllerCatPro2.

### 06 Immunogenicity — ranking metric, not a hard gate

**Current / finalized method — `run_netmhciipan_sb.py`**

Queries **NetMHCIIpan 4.1 BA** (`method=netmhciipan_ba`, IEDB's own
recommended binding predictor, 2023.09) via the **new** IEDB Next-Generation
Tools API (`https://api-nextgen-tools.iedb.org/api/v1/pipeline`) — no SSH
tunnel needed, this host is directly reachable. 7-allele DRB reference panel
(`HLA-DRB1*03:01/07:01/15:01`, `HLA-DRB3*01:01/02:02`, `HLA-DRB4*01:01`,
`HLA-DRB5*01:01`), 15-mer sliding window across the full 37aa sequence, run
on the 7 `final_pass` candidates + amylin reference in one call.

**Metric: `n_sb_lt2pct`** — count of (peptide × allele) pairs with
**percentile rank < 2%** (Strong Binder, IEDB/NetMHCpan convention — stricter
than the "top 10%" Weak-Binder cutoff IEDB generally recommends for flagging
epitopes). SB was chosen over WB because at rank ≤10% several final
candidates are indistinguishable from each other: their sequence differences
sit outside every scored 15-mer window (e.g. GA_087/GA_088/GA_104 differ
only in the last 1-2 residues, C-terminal to every binder hit, so they share
an *identical* WB binder set), so WB only gives a fuzzy "all candidates below
amylin" signal. At the SB<2% cutoff the result is clean and decisive: native
amylin has 4 strong binders, **all 7 GA candidates have 0**.

Outputs: `outputs/netmhciipan_sb_raw.csv` (full per-window × allele table),
`outputs/netmhciipan_sb_summary.csv` (`n_sb_lt2pct`, `n_wb_le10pct`,
`sb_vs_amylin` per sequence). `run_final_filter.py` reads the summary and
carries `imm_n_sb_lt2pct` / `imm_n_wb_le10pct` / `imm_sb_vs_amylin` through
to `final_results.csv` — reference/ranking only, not a pass/fail gate.

Run: `python3 filter/06_immunogenicity/run_netmhciipan_sb.py`

**Superseded — do not use**: the previous two-step pipeline
(`run_iedb.py`/`run_consensus_epitopes.py` for binding, then
`integrate_cd4episcore.py` joining **CD4episcore** Combined Scores) is
abandoned:
- `run_iedb.py`/`run_consensus_epitopes.py` targeted the legacy
  `tools-cluster-interface.iedb.org` API host, which is now unreachable
  (dead server, confirmed independently from two separate networks).
- CD4episcore itself — both the new API's `immunogenicity` predictor and the
  Legacy `tools.iedb.org/CD4episcore/` web form — is currently broken:
  the new API reproducibly fails server-side ("Completion flag file
  missing", not a request-format issue — confirmed by testing both
  malformed and fully-correct requests), and the Legacy form silently
  no-ops (HTTP 200, form just re-renders, no job ever created). Also: the
  `Combined Score ≥ 43` cutoff used in the old `integrate_cd4episcore.py`
  was mislabeled — Dhanda et al. 2018 Table 2 actually gives 43 → 59%
  sensitivity, 66 → 75% sensitivity (the number the code comment claimed for
  43). The 23/42-peptide incomplete `cd4episcore_results.csv` and the
  `immunogenicity_report.csv` it produced are stale artifacts of this
  abandoned path.
- A from-scratch human-proteome self-similarity check (JanusMatrix-style:
  MHC-binding 9aa core vs UniProt human reviewed proteome, local
  brute-force Hamming-distance scan, `06_immunogenicity/self_similarity/`)
  was also tried as an alternative risk-refinement layer, but gave an
  inconclusive result (best matches ~67% identity — not similar enough to
  call "tolerized", not dissimilar enough to rule out) and was not adopted.

This stage is **not applied as a pass/fail filter** — `run_final_filter.py`
only carries `imm_*` columns through to `final_results.csv` for reference /
ranking among the 7 final candidates.

---

## Final integration — `run_final_filter.py`

Reads all of the above and writes `final_results.csv` (154 rows). Per
sequence:

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
  agonism yet — deferred to `07_md`).
- **`final_pass`** = `amy_pose_pass` AND `offt_pass` AND `safety_pass`.

Run: `python3 filter/run_final_filter.py`

---

## Files

| File | Description |
|---|---|
| `sequences_ga.csv` | 154 GA candidate sequences + per-receptor GA fitness scores (AMY1R–AM2R) |
| `filter_tracker_ga.csv` | Mixed legacy + current filter columns (see caveat below) |
| `final_results.csv` | **Current** integrated result — all 154 sequences, all gates, `final_pass` column |
| `ga_final_ranked.csv` | Output of the old 3-filter scheme (38 sequences) — superseded, kept for history |
| `run_final_filter.py` | Current integration script → `final_results.csv` |
| `run_ga_filters.py` | **Deprecated** — references old folder layout (`01_aggregation/`, `05_allergenicity/`, `06_toxicity/`) that no longer exists after the `01_pose_check`…`07_md` restructure. Do not run; kept for reference only. |
| `01_pose_check/run_pose_check.py` | Docking pose geometry check |
| `02_binding_energy/run_foldx.py` | FoldX ΔG_AB/AC/BC |
| `03_aggregation/run_tango.py` | TANGO aggregation (legacy per-run script; current pass/fail lives in `filter_tracker_ga.csv`'s `aggregation` column) |
| `05_toxicity/run_toxinpred3.py` | ToxinPred3 Hybrid Score |
| `06_immunogenicity/run_netmhciipan_sb.py` | **Current** — NetMHCIIpan 4.1 BA, 7-allele, n_sb_lt2pct → `netmhciipan_sb_summary.csv` |
| `06_immunogenicity/run_iedb.py` | Superseded — legacy whole-154 screen, targets dead `tools-cluster-interface.iedb.org` host |
| `06_immunogenicity/run_consensus_epitopes.py` | Superseded — same dead host; also feeds the abandoned CD4episcore path |
| `06_immunogenicity/integrate_cd4episcore.py` | Superseded — CD4episcore is broken (new API server bug + Legacy form silent no-op), see [06 Immunogenicity](#06-immunogenicity--ranking-metric-not-a-hard-gate) |

**Caveat on `filter_tracker_ga.csv`**: it accumulated columns across tool
iterations and now mixes stale and current data — `toxicity`/`tox_score` are
ToxinPred2 (abandoned), `allergenicity`/`alg_score` are AlgPred2 (superseded
by AllerCatPro2), `all_pass` reflects the old 3-filter scheme. Only the
`aggregation` column (TANGO) is still read by `run_final_filter.py`; toxicity
and allergenicity are read fresh from `05_toxicity/outputs/toxinpred3_raw_ga.csv`
and `04_allergenicity/outputs/AllerCatPro2_prediction_*.csv` instead. Treat
`tox3_score`/`toxicity3`/`n_rank_iedb`/`best_rank_iedb` as the current
columns.

## Re-running

```bash
# Pose check (requires ../structures/<id>/<receptor>/*.cif from docking)
conda run -n af3_ml python3 filter/01_pose_check/run_pose_check.py

# FoldX binding energy
conda run -n PDBFixer python3 filter/02_binding_energy/run_foldx.py --workers 8

# TANGO + ToxinPred3 — no unified current script; TANGO via 03_aggregation/run_tango.py,
# ToxinPred3 via:
conda run -n raghava_tools python3 filter/05_toxicity/run_toxinpred3.py

# AllerCatPro2 — manual web upload, filter/04_allergenicity/inputs/*.fa → outputs/*.csv

# IEDB immunogenicity (new API, no SSH tunnel needed)
python3 filter/06_immunogenicity/run_netmhciipan_sb.py

# Final integration
python3 filter/run_final_filter.py
```

---

## Superseded content (old 3-filter version)

The original version of this pipeline (before pose check / FoldX / off-target
gating were added) applied only 3 filters directly to the 154 sequences:

| Filter | Tool | Pass / Total |
|---|---|---|
| Aggregation | TANGO | 39 / 154 |
| Allergenicity | AllerCatPro 2.0 | 154 / 154 |
| Toxicity | ToxinPred3 | 151 / 154 |
| **All pass** | | **38 / 154** |

Ranked by `n_rank_iedb` (IEDB percentile-rank binder count) ascending, top 5
were:

| Rank | ID | seq_24aa | n_rank | best_rank | off_max |
|---|---|---|---|---|---|
| 1 | GA_074 | WFSFRSQQNWGNILGRSNSCSKAV | 13 | 1.3% | 0.166 |
| 2 | GA_122 | WFPFRSNPNAEAGLVRMASGKNTY | 13 | 0.2% | 0.217 |
| 3 | GA_072 | WFSFRSQYNFGNILGRSNSCSKAR | 14 | 1.4% | 0.165 |
| 4 | GA_073 | WFSFRSQHNFGNILGRSNSCSKAR | 14 | 0.9% | 0.166 |
| 5 | GA_085 | WFSFRSQQNFGNIWGRSNSCSKAR | 14 | 0.7% | 0.170 |

Full list: `ga_final_ranked.csv`. This scheme did not check docking pose or
binding energy against off-target receptors at all, which is why it was
extended into the current 7-stage pipeline above.
