Run outputs go in `<run_name>/`; actual inspected V1–V5 go in `<run_name>/vis/`.
Weights, assets, generated plots and numerical logs are excluded from Git.
No experiment or visualization is implied by this directory's existence.

`smoke_seed17_v2j/`: completed 1200 real VGGT updates after 38 strict gsplat
tests. All three scenes' V1–V5, training curves and match/path diagnostics were
generated and opened. Brown normal erasure still has global pose effects;
Step 1.5 is not fully passed and Step 2 has not started. See
`../docs/step1_5_v2j_report.md` for measured results and remaining causes.
