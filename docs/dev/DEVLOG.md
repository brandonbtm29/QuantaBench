# Development log

Newest entries at the bottom. One entry per working session or nightly run.

## 2026-10-22: 0.1.0 scaffold
- Geometry builder pipeline, PySide6 GUI (OVITO-style docks), CLI (`doctor`, `build`, `slab`,
  `settings`), resource manager, graphite surface builder. 35 tests.

## 2026-10-23: 0.2.0.dev0 (multi-agent build)
- Six parallel tracks: 3D viewer, 2D viewer, resumable batches + CompTox/Excel input, frequencies +
  thermochemistry + IR, batched conformer refinement, LAMMPS rigid-adsorbate steered-MD bridge.
- Integrated into the GUI (2D|3D split with shared selection, Rotate/Select/Move modes, surface builder
  move mode + LAMMPS export) and CLI (`build --run-dir/--resume/--chunk/--freq/--batched`, `status`,
  `merge`, `freq`, `md smd`, `md parse`).
- Three independent reviews (science, GUI/threading, robustness): 24 findings (0 critical, 4 high),
  all confirmed and fixed. 176 tests, ruff clean (Linux, CPU only).
- Not verified: anything on a GPU, Windows, or with real AIMNet2 weights (see ROADMAP 0.2 "Still open").
- Planning: `docs/UI_PLAN.md` (task-oriented workspaces, job model) and `docs/dev/BACKLOG.md`.
- Next: UI-1 app shell.
