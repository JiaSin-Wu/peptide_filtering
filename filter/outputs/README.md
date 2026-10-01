# Filtering outputs

This directory is the single entry point for consolidated filtering results.
Stage-specific raw outputs remain in each stage's own `outputs/` directory.

| File | Purpose |
|---|---|
| `final_results.csv` | Full 92-candidate integrated table. The legacy path `filter/final_results.csv` is a compatibility symlink to this file. |
| `final_candidates.csv` | The six candidates retained after all sequence filters and the AMY1R/AMY2R/AMY3R pose check. |
| `filtering_summary.csv` | Funnel counts and concise interpretation of each stage. |
| `pose_results_amy123r.csv` | Symlink to `01_pose_check/outputs/results_from_md_af3.csv`, the 21-row sequence-verified AMY1R/AMY2R/AMY3R pose result. |

`GA_004` is not a final candidate. It passed aggregation, allergenicity,
toxicity, and all three AMY-receptor pose checks, but failed CD4episcore
because 7 of 8 returned windows scored below 43 (minimum 31.04). Its existing
MD results are retained only for transparent reporting.

The legacy `final_pass` and `final_pass_no_energy` columns preserve the current
pipeline's historical rollup definitions and do not include CD4episcore as a
hard gate. Use `final_candidates.csv` for the current final panel.
