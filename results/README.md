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

`smoke_seed17_v2l/`: completed 1200 real VGGT updates after 49 strict tests.
All 15 V1–V5 images, three training plots and three diagnostic plots were
opened. Anisotropic LM is energy-monotone and stable through 12 iterations;
Brown normal T0 change is 2.01 cm, but inside/outside EPE-increase ratio is
1.55 and C5 local degradation is weak. Heldout T1 calibration also remains
incomplete. See `../docs/step1_5_v2l_report.md`. Step 1.5 is not fully passed;
no project GPU jobs remain and Step 2 has not started.

The user subsequently accepted Step 1.5 and authorized Step 2.
`step2_full_seed17/`: fresh official VGGT initialization, 53 strict tests,
3000 real updates, 331 distinct training queries in 14 eligible scenes.
Scene-disjoint val contains C11 and robotics; all 126 heldout/val queries
were evaluated with same-weight full/empty/mismatched/erased memory.
All four unseen-query V1–V5 sets and curves were generated and opened.
Cross-scene EPE is 0.365 m; C11 is 0.250 m, while robotics is 0.723 m
with 1.09 m mean translation error. Training stopped at the authorized
midpoint due to this generalization bottleneck. The 18k continuation and
independently trained No-Memory/Local baseline have not run. See
`../docs/step2_report.md`. No project GPU jobs remain.
