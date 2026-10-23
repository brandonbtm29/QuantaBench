# Nightly agent runbook

QuantaBench is built partly by scheduled agent runs. Each run starts in a fresh cloud session with no
memory of earlier runs; this file, `BACKLOG.md` and `DEVLOG.md` are the hand-off. Brandon reviews and
merges in the morning.

## Ground rules

- **One backlog item per run.** Finish it properly (tests, docs, screenshots) instead of starting two.
- **Never push to `main`, never merge, never force-push someone else's branch.** Work on
  `nightly/<YYYY-MM-DD>-<item-id>-<slug>` and open a pull request against `main`.
- **Don't duplicate open work.** Before choosing, run `gh pr list --state open` and skip items that
  already have an open PR (the PR title starts with the item id, e.g. `[UI-1]`).
- **Signed commits** with the environment's default identity (`Claude <noreply@anthropic.com>`); never
  pass `-c commit.gpgsign=false`.
- **No GPU here.** Items marked `needs-Brandon` (hardware, Windows, ROCm) are never picked. If an item
  turns out to need the GPU or a decision from Brandon, stop, write it up in the PR/DEVLOG, and mark it
  `blocked`.
- **Respect the conventions in `CLAUDE.md`** (PySide6, lazy optional imports, pipeline stays UI-agnostic,
  QThread workers, Windows-first paths).
- **Stay in scope.** If you find unrelated bugs, add them to BACKLOG.md as new items instead of fixing
  them in the same PR (unless trivial and blocking).

## Steps

1. Clone: `git clone https://github.com/brandonbtm29/QuantaBench && cd QuantaBench`.
2. Read `CLAUDE.md`, `docs/ARCHITECTURE.md`, `docs/UI_PLAN.md`, `docs/dev/BACKLOG.md`, and the last few
   entries of `docs/dev/DEVLOG.md`.
3. Set up: `bash scripts/cloud_setup.sh && source ../.venv-qb/bin/activate`, then confirm the baseline:
   `ruff check .` and `xvfb-run -a python -m pytest -q` (note any failures on `main` in the DEVLOG).
4. Pick the first item in BACKLOG.md with status `ready` that has no open PR and whose dependencies are
   merged. Create the branch.
5. Build it. For anything larger than a small change, use a workflow (up to ~5 agents): design or
   build → independent review (correctness, GUI/threading, robustness) → fix confirmed findings.
   GUI work must be exercised under `xvfb-run` with a scripted smoke test and screenshots you actually
   look at.
6. Verify: ruff clean, full suite green, new tests for new behavior.
7. In the same branch: set the item's status to `in-PR`, add a DEVLOG entry (date, item, what changed,
   test count, screenshots, open questions, suggested next item), update README/ARCHITECTURE/ROADMAP/
   CHANGELOG if behavior changed.
8. Push the branch and open the PR: title `[<item-id>] <summary>`; body with what/why, how to test on
   Windows, screenshots (commit them under `docs/dev/screenshots/<item-id>/` if useful), risks, and
   questions for Brandon.
9. Stop. Do not start a second item.

## When things go wrong

- Baseline tests fail on `main`: if the fix is small and obvious, make that the run's item (`FIX-<n>`);
  otherwise document and pick an unaffected item.
- Merge conflicts with an open PR: rebase your branch on `main` only; never rewrite others' branches.
- Network blocks (PyPI is fine; pytorch.org and Google Cloud Storage may not be): use stand-ins in tests
  and note what needs real downloads.
