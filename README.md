# VGGT Token-Scale Spatial Memory

Independent exploratory project: official frozen DINOv2 patch tokens → metric
token Gaussians → spatial moment fusion → O5 gsplat rendering + O6 geometric
cross attention → official VGGT reception block / DPT / bootstrap pose heads.

Shared assets are referenced read-only in `configs/runtime_paths.yaml`. DINOv2
weights are embedded in the full VGGT checkpoint; no separate file is assumed.
The complete pretrained checkpoint is never copied into this repository.
`AGENTS.md` defines the working rules; `docs/master_plan.md` describes the full
method and Step 1 → Step 2 → Step 3 sequence. Main train/val is scene-disjoint
in `configs/splits.yaml`; smoke held-out queries are labeled within-scene.

From this project directory, initialize project-local caches and run:

```bash
source scripts/project_env.sh
/home/liwt/miniColmap/.venv/bin/python -B -m pytest tests/test_geometry_conventions.py -q -rs
```

For actual CUDA tests, dynamically check/select a GPU (inside Codex this needs
host execution because the filesystem sandbox hides GPU devices):

```bash
REQUIRE_GSPLAT=1 /home/liwt/miniColmap/.venv/bin/python -B scripts/run_on_available_gpu.py -- \
  /home/liwt/miniColmap/.venv/bin/python -B -m pytest tests/test_geometry_conventions.py -q
```

The smoke runner performs strict real geometry tests before training. Use a new
run name, since existing outputs are preserved:

```bash
/home/liwt/miniColmap/.venv/bin/python -B scripts/run_on_available_gpu.py \
  --min-free-mib 24576 --memory-fraction .42 -- \
  /home/liwt/miniColmap/.venv/bin/python -B scripts/train_smoke.py --run smoke_seed17_new
```

Default: 1000 total updates, three real scenes round-robin, 112×140 images,
14-pixel patches. Trainable: Writer/fusion, dual Reader, native receiving global
block 4, native DPT and two pose heads. Other native blocks have frozen weights
but pass gradients to the receiving layer. Frozen DINOv2 stays in eval mode.
The query API receives only `QueryInput(rgb,K)` and GaussianMemory. Mapping
RGB-D and mapping poses are explicitly allowed inputs; query truth is used only
by the loss/evaluator. Memory contains no historical image or frame KV fields.

Run outputs include `metrics.jsonl`, a compact trainable-only checkpoint,
`evaluation.json`, training curves and actual V1–V5 PNGs for every scene.
Open the images before reporting a milestone. V6 is reserved for streaming
Step 3. To regenerate plots from a completed run without retraining:

```bash
/home/liwt/miniColmap/.venv/bin/python -B scripts/run_on_available_gpu.py \
  --min-free-mib 24576 --memory-fraction .42 -- \
  /home/liwt/miniColmap/.venv/bin/python -B scripts/visualize_run.py smoke_seed17_new
```

No prediction alignment, ICP, history import, automatic large asset download,
or writes into shared assets are used. All caches and outputs stay here.
