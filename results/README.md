Run outputs go in `<run_name>/`; actual inspected V1–V5 go in `<run_name>/vis/`.
Weights, assets, generated plots and numerical logs are excluded from Git.
No experiment or visualization is implied by this directory's existence.

`smoke_seed17_v2j/`: completed 1200 real VGGT updates after 38 strict gsplat
tests. All three scenes' V1–V5, training curves and match/path diagnostics were
generated and opened. Brown normal erasure still has global pose effects;
Step 1.5 is not fully passed and Step 2 has not started. See
`../docs/step1_5_v2j_report.md` for measured results and remaining causes.

`smoke_seed17_v2k/`: completed 1200 real VGGT updates after 43 strict tests.
All V1–V5, two training plots and three diagnostic plots were opened. Normal
V5 inside/outside EPE-increase ratios are C5/Brown/C6 = 3.27/1.01/11.47;
Brown T0 still changes 7.20 cm. Step 1.5 remains incomplete; see
`../docs/step1_5_v2k_report.md`. No project GPU jobs remain; Step 2 is not running.
