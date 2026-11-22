# Backlog

Ordered by priority. Nightly runs take the first `ready` item with no open PR and whose dependencies
are merged (see `RUNBOOK.md`). Status: `ready` · `in-PR` · `done` · `blocked` · `needs-Brandon`
(hardware, Windows, or a decision: never picked by nightly runs).

Design reference: `docs/UI_PLAN.md`. Keep QuantaBench its own app: borrow features, not layouts.

---

## CREST-RUN-1: CREST rebuild of the flexible neutral molecules · started 2026-11-17; paused 2026-11-18 (needs-Brandon to resume)
Brandon (2026-11-17): run CREST for the flexible neutral molecules, gas then water, without the 173 with 20 or more
rotatable bonds. 1,501 molecules with 9 to 19 rotatable bonds, `--conformers crest --crest-method gfnff
--crest-fallback rdkit --molecule-timeout 5400`, refine top 20, `--freq`, `--rank auto`, background priority; the
workstation around the clock and the MacBook from 2 to 8 am, both at a 66 % budget. Scripts in `scratch/batches/`
on the workstation (not in the repository).
- **State at the pause (gas phase):** 776 of 1,501 done. The workstation: 400 ok, 39 failed, all at the time limit
  (32 in the CREST search, 5 in refine, 2 in frequencies). The MacBook: 337 ok, none failed. 22 molecules got RDKit
  conformers because CREST failed (the fallback). Water has not started.
- **Median CREST search of the molecules that finished, by strict rotatable bonds** (the workstation: 7 lanes x 2
  threads through WSL; the MacBook: 2 lanes x 4 threads; the two machines did not run the same molecules):

  | rotatable bonds | workstation | MacBook | workstation, out of time |
  |---|---|---|---|
  | 9-11 | 319 s | 55 s | 0 of 83 |
  | 12-14 | 701 s | 176 s | 1 of 174 |
  | 15-16 | 1,982 s | 547 s | 1 of 94 |
  | 17-19 | 3,088 s | 1,991 s | 37 of 88 |

  The workstation used 230 lane-hours, 59 of them on molecules that then ran out of time; the MacBook 24.
- **The MacBook's window** worked on both transitions: a pause with drain at 08:00 moved its unstarted molecules to
  the workstation (DIST-7), and work sharing gave it molecules again after the resume at 02:00.
- **To resume** (needs-Brandon): `quantabench-cli worker resume` on the workstation, then the run script (it resumes
  only unfinished molecules) and the window script. Recommended first: SCHED-2's wider lanes, and TELEM-1 merged, so
  the rest of the run records full timings for killed attempts.
- **After the gas phase:** rerun the molecules that ran out of time with wider lanes and a 3 h limit; then water;
  then a decision on the 173 molecules left out.

## CREST-SCRATCH-1: CREST through WSL in a folder inside WSL · done 2026-11-20 (branch `fix/crest-wsl-scratch`)
On the workstation CREST runs as `wsl:.../bin/crest` and used to run in its Windows job folder through `/mnt/c`: every
file it wrote crossed WSL's 9P bridge to NTFS, about 20 MB/s of disk writes (some 70 GB an hour) just after a CREST
batch started.
- Done: `chem.wsl_scratch`. Each job runs in `/dev/shm/quantabench/<job>` when /dev/shm and WSL's memory have
  `engines.crest_wsl_shm_min_gb` (2 GB) plus twice the job's expected size free, else in
  `~/.cache/quantabench/scratch/<job>`; only the conformers and CREST's two summaries come back (small files on a
  failure, everything with keep_scratch); the folder is removed in a `finally`; killed runs' folders go through
  `sweep_scratch` (the Windows job folder names its twin) and at the next job (owner records). A note per molecule
  says which scratch was used. `engines.crest_wsl_scratch` = auto | shm | disk | off; Settings > Engines; `settings
  --crest-wsl-scratch`. Tests: `tests/test_wsl_scratch.py` (the real shell steps through a stand-in `wsl.exe`).
- Next (needs-Brandon): on the workstation, a CREST batch with `auto`. Check the disk writes in Task Manager or
  Resource Monitor against the 20 MB/s before (vmmem's writes should be near zero), the notes (/dev/shm or disk),
  `free -h` inside WSL while the lanes run, and that nothing is left in `/dev/shm/quantabench` afterwards. Then compare
  CREST times per molecule with CREST-RUN-1's.
- Later: telemetry columns for the scratch kind (today the note only), so the cost fit can tell the runs apart; an
  opt-in /dev/shm for native Linux CREST (off by default: it already runs on a local disk); the same folder inside WSL
  for other `wsl:` programs (xtb, if it is ever run that way).

## TELEM-1: Timings of killed attempts keep the molecule's features and finished stages · in-PR 2026-11-18 (PR #26)
Brandon (2026-11-18): "double check and see if we are recording the metadata for failed molecules, so we can continue
to improve the simulation load and thread allocation." A lane killed by the time limit sent no telemetry, so its
attempt had no features and only the stage it was killed in. The fix computes the features at dispatch and keeps
the stages the parent saw begin (`RunTelemetry.stage_seen`). Details in the PR and its CHANGELOG entry.

## SCHED-2: CREST lanes and the cost model by rotatable bonds · in progress (lanes and the fit: PR, 2026-11-18)
From CREST-RUN-1's timings: atoms and rotatable bonds explain the cost of a CREST search about equally (scatter x2.91
with atoms alone, x2.76 with log(1 + rotors)), rotatable bonds, not atom count, decide which searches run out of
time, and 2 threads per search are too few from about 17 rotatable bonds.
- **Done (2026-11-18, Brandon's item 6 of the Nov 18 review):** `--conformers crest` on a machine with no known
  performance cores plans lanes of at least 4 threads (the workstation 3 x 5 + 2 x 4; the run resumed that day by
  hand with 3 x 5); `timings --fit` fits a (1 + rotors)^r term, runs use it for the dynamic threads and the
  predicted seconds, and a CREST run with such a fit hands out its molecules by predicted cost. The built-in
  model keeps atoms^4 (the rebuild had no molecule under 9 rotors). Still open: routing, the time limit, and the
  replay below.
- **Lane plan:** for a batch of flexible molecules, fewer and wider CREST searches on a machine with many threads
  (for example 3 or 4 searches of 4 to 6 threads in place of 7 of 2); the launch-time rule of `dist/threads.py`
  should widen by predicted cost, not only when one molecule dominates the batch.
- **Cost model:** add strict rotatable bonds (and rings) to the CREST cost fit (`timings --fit`), which uses
  atoms^3.9 today; judge it by makespan and by time-limit failures, as SCHED-1 says. TELEM-1 supplies the features
  of the killed attempts.
- **Routing:** in a grid, send the molecules with the highest predicted cost to the machine with the widest lanes
  first (a first step toward DIST-4).
- **Time limit:** a per-molecule limit that scales with predicted cost, or a retry pass with a longer limit, so a
  fixed 90 minutes does not burn lane-hours on searches that cannot finish.
- Acceptance: a rerun of CREST-RUN-1's time-limit failures finishes, and a simulated replay of its gas phase
  shows fewer lane-hours lost.

## BATCH-1: The staged-batch scripts as features · in progress (time window: PR #30; batches: PR, 2026-11-18)
The QSAR runs depend on scripts outside the repository (`scratch/batches/` on the workstation): the tranche runner
with its deadline and resume, the budget setter, and the time window for one machine. They have no tests.
- **A time window per machine** in the settings (for example 02:00 to 08:00): the worker pauses with drain at the
  end of its window and resumes at its start; the Machines dock shows the window. Today a script on the client
  machine does this over the worker commands.
- **A deadline and resume for `grid build`:** stop at a given time (cancel the remote shares, stop the local share
  at its next molecule), and `--resume` for a phase folder that runs only what no part has finished.
- **Batches as a list:** run gas, then water, then `pair`, for a list of input files, each to its own run folder,
  with a short report per batch. `--include-paused` already exists in the script: a paused machine still joins.
- Acceptance: CREST-RUN-1's water phase runs from the CLI without the scratch scripts.
- 2026-11-18 (branch `feat/machine-window`): the time window is done. `machine_control` "window", `machines set
  --window`, the dock's "Time window"; the worker pauses with drain when it closes and resumes when it opens, only at
  the edges (`dist.control.parse_window`, `in_window`; `JobManager._check_window`). Left: the deadline and resume for
  `grid build`, and batch lists.
- 2026-11-18 (branch `feat/batch-runner`, on top of the time window): `quantabench-cli batch run` / `batch status`
  (`dist/batch.py`): phases, attempts over what no run folder of the phase has finished, the deadline (remote
  shares cancelled, the local share stopped at its next molecule, exit 5), snapshots of the remote shares from every
  unmerged `grid.json` of the phase, `--include-paused`, `--list` of input files, `paired.csv` and `report.md`. The
  deadline lives in `batch run`, not `grid build`. Not ported: the QSAR-specific `analyze_tranche.py` step, and the
  budget setter (`machines set --max-cpu` covers it). Left: run CREST-RUN-1's water phase with it (acceptance).

## FIX-6: A flaky macOS test of run priority · ready
`tests/test_machine_control.py::test_priority_applied_to_running_processes` fails now and then on macOS CI (a
process is left at nice -10); it passes on a rerun. Seen on #17 and #25. Find the race or make the test wait for
the priority change.

## REL-1: Tag 0.3.0 · done 2026-11-18 (Brandon: "complete item 8"; tag `v0.3.0`)
0.1, 0.2 and 0.3 are all unreleased and nothing is tagged. After TELEM-1 and this status update: set the version,
date the CHANGELOG's sections, tag, and say in the README what a release means for a project with a fork-based
install. Brandon decides when.

## CONF-AUTO: CREST for flexible neutral molecules · done 2026-11-16 (PR #17)
Brandon (2026-11-16), after the CREST gas and water tests: "Let's do it all". That means CREST (GFN-FF, quick) for
neutral molecules with 9 or more rotatable bonds, not for ions, optionally with the RDKit pool merged in. The evidence
is in docs/VALIDATION.md, "Conformer checks and the CREST test".
Done in the branch:
- `chem/search_rule.py`:
  - `decide` / `crest_for` / `may_use_crest`;
  - the charge class (zwitterions are not neutral; a nitro +/- pair is) and the strict rotatable-bond count, the
    same as the advisor's and the conformer checks';
  - `merge_pools` and `ConformerSet.refine_count` / `refine_ids`.
- `BuildConfig.conformer_search = "auto"`, `crest_min_rotors` (9), `crest_with_rdkit` (False):
  - in `jobs.SCIENCE_FIELDS`, idle where they cannot act (`_ONLY_WITH` takes several values now);
  - in the settings' BuildDefaults, the recipes' fields and the worker's choices.
- Pipeline:
  - the decision per molecule on the prepared structure, with a log line;
  - `conformer_search` in summary.csv is "crest", "rdkit" or "crest+rdkit";
  - the merged pool is not used with the refinement "mmff" (a note).
- CLI: `--conformers auto`, `--crest-min-rotors N`, `--crest-with-rdkit/--no-crest-with-rdkit`.
- GUI: Search > Auto, "CREST from N rotatable bonds" and "Add RDKit's conformers to CREST's pool" in the Pipeline dock
  and the job dialog; the review's Apply sets them.
- Advisor:
  - the per-molecule CREST cost (and the RDKit pool's with the merge);
  - a "search.flexible-neutrals" warning under the RDKit search;
  - the crest-ions finding only under the RDKit search;
  - the steps text;
  - the Careful preset and the Careful recipe use "auto" with the merge.
- Run-level checks use `may_use_crest`: lanes, dynamic threads, the cost model, telemetry and the time-limit note.
  A merged pool's conformer stage is left out of the CREST cost-model fit (`timings._is_crest`).
- Tests: `tests/test_search_rule.py`, `tests/test_crest.py` (auto with a fake CREST, the merged pool),
  `tests/test_gui_jobs.py::test_conformer_search_auto_in_the_pipeline_dock_and_the_dialog`; `test_advisor.py` and
  `test_recipes.py` updated for Careful.
Follow-ups (ready):
- Dynamic CREST threads per molecule: with "auto", idle lanes' threads may be lent to a molecule that turns out to get
  RDKit (harmless at the end of a batch, where lending happens; a prediction from the input would avoid it).
- The cost model and lane planning of a mixed "auto" batch: plan for the CREST share instead of all-CREST.
- Ions: a test of CREST in ALPB water for flexible anions (the gas test covered 7 ions; the DFT test suggested
  folded and extended pools in water). Until then "auto" keeps them on RDKit.
- The full run: started 2026-11-17 as CREST-RUN-1 (1,501 molecules with 9 to 19 rotatable bonds; CREST's pool
  alone, without the RDKit merge).

## FAIL-1: The failed molecules of the QSAR tranches · done 2026-11-16 (PR #20; rerun 2026-11-17: 14 of 15 build)
b01-b15 (13,206 molecules): 8 failed in gas and 13 in water (`Runs\failed-molecules-2026-11-16\report.md`).
Done in the branch:
- `conformers.embed`:
  - RDKit's timeout id -1 is not a conformer;
  - `FIRST_CONFORMER_S` (60 s) per ETKDG start, then the next start (without small-ring torsions, random
    coordinates);
  - the 5 glycosides ("Bad Conformer Id").
- `chem/cpcm_worker.py`: the rank step's GFN1-xTB gas/CPCM terms in a helper process. `rank.CpcmTerm` uses it
  (`in_process=`, `QUANTABENCH_CPCM_IN_PROCESS`). This fixes the 6 water crashes (tblite 0.8.0 ddX heap corruption at
  some geometries; reproduced on Windows and Linux with 2447-54-3's water minimum).
- `eng_tblite.TBLiteCalculator`: a failed SCF gets a last try with mixer damping 0.2 and up to 1,000 cycles
  (`n_damped`).
- `pipeline._refine`: a conformer whose optimization fails is dropped (noted); the molecule fails only if all do.
  This fixes the 5 SCF failures.
- Tests:
  - `test_conformers.py::test_an_etkdg_start_that_finds_no_conformer_gives_way`,
    `::test_no_way_of_embedding_finds_one_is_a_clear_error`;
  - `test_rank.py::test_the_cpcm_helper_survives_a_crash`, `::test_the_cpcm_helper_gives_the_in_process_terms`,
    `::test_a_crashed_cpcm_term_costs_that_conformer_only`;
  - `test_eng_tblite.py::test_an_scf_that_fails_gets_a_gentler_last_try`;
  - `test_pipeline.py::test_one_conformers_failure_drops_that_conformer`.
Follow-ups:
- Report the ddX crash upstream (tblite 0.8.0, CPCM, with 2447-54-3's geometry).
- Rerun the 15 molecules (21 molecule-phases) with `--retry-failed` once this is merged (the report lists the commands).
  - Done 2026-11-17 on main 79213ea: 14 of the 15 build in both phases, and the tranches are paired again.
  - Gitoxin (4562-36-1) failed again in both phases: "conformer embedding ran out of time (450 s)". Its first
    conformer took 9 s with small-ring torsions, so 39 of 50 were asked for in the 441 s left. RDKit's timeout
    stopped that batch before its first conformer, and no time was left for another way. Fixed in
    `fix/embed-batch-stopped`: the batch gets `EMBED_BATCH_SHARE` (80 %) of the time left. A stopped batch
    keeps the timed conformer, and the rest are embedded without small-ring torsions (about 0.5 s each).
- 1423-20-7 (C6F13-SF5) has no force-field parameters. Its refinements start from raw ETKDG geometries, and some end
  about 11 eV higher (bonds broken). A check that the refined bonds match the input would flag such conformers.

## FIX-5: Crashes in the GUI tests (Windows CI) · done 2026-11-17 (PR #23)
Main's Windows CI for 486e332 (#19's merge, with #21) aborted in
`test_gui_declutter.py::test_table_toolbar_menus_and_popout`, in the first `_wait` after a new window opened.
- **faulthandler:** "Fatal Python error: Aborted" in the main thread (`_wait` → `processEvents`). Two
  "quantabench-sensors" threads were running, one inside `ssl._load_windows_store_certs` (allocating a lot).
- **py-spy's native stack** of the main thread: `sendPostedEvents` → `QApplication::notify` → `QObject::event`
  (DeferredDelete) → a PySide wrapper's destructor → `QObject::~QObject` → `_purecall` → `abort`. A pure virtual call
  inside `~QObject` means something touched an object that was already being destroyed.
- **A second crash, on #22's Windows CI** (main + FIX-4): "Windows fatal exception: code 0xc0000374" (heap
  corruption), in the main thread while it collected garbage, in `test_gui_table.py`'s `_wait` after a new window
  opened. Again two sensors threads were running, one inside `_load_windows_store_certs`.
- **Likely cause: FIX-3's worker/runner cycle, collected in a sensors thread.**
  - The worker and its runner referred to each other, so once `_Keeper` let go of them, only the cyclic collector could
    free them. At that moment their `deleteLater` events are posted to the GUI thread.
  - The collector runs in whichever thread allocates. In a sensors thread (allocating hard while loading the
    certificate store) it freed the wrappers. PySide owns these objects, so it deleted the C++ objects right there,
    outside the GUI thread.
  - Meanwhile the GUI thread delivered the posted DeferredDelete to the same object: a pure virtual call in `~QObject`
    (crash 1), or a corrupted heap found later (crash 2).
  - Before FIX-3 no cycle held a worker. The app collects only in its GUI thread (`a717844`, `gui/app.py`), so this
    can happen only in the tests, which collected automatically in any thread, plus once per module.
- **Fix:**
  - `tests/conftest.py`: automatic collection off for the session; `gc.collect(1)` after every test and a full
    collection after each module, in the main thread.
  - `gui.workers.start`: the worker and its runner (FIX-3) referred to each other, so only the collector could free
    them. Now `_Keeper` holds both and lets go of them in the GUI thread, and the runner drops its worker when `run`
    ends.
- **The process then hung in Windows' process exit.** VTK's matplotlib text module, torn down with its DLL, called
  `PyGILState_Ensure`, and the GIL was held by a thread that `ExitProcess` had already ended. The watchdog printed
  "killing pid 496" (`taskkill /F /T`) at 7 min, but the job ran into its 30 min limit. On the workstation the old kill does end
  a stuck process that started the watchdog, so the exit state is the likely reason.
  - The watchdog now kills with psutil (the test process first, then its other children, never itself) and logs if
    the process is still there 15 s later.
  - Test: `test_hang_watchdog.py::test_the_watchdog_kills_the_process_it_is_a_child_of`.
- **Not reproduced:** Linux (xvfb), every GUI and project test with `gc.set_threshold(20, 1, 1)`: 166 passed; the
  two GUI modules 3 times: passed. Linux has no sensors thread loading certificates. The GUI tests were not run on
  the workstation: they open windows on Brandon's desktop.
- **Check:** the full suite with the change (cloud, Linux): 1,714 passed, 40 skipped, 7 min. Watch Windows CI for
  another "Fatal Python error: Aborted".

## FIX-4: A run failed by a status poll at its end (Windows) · done 2026-11-17 (PR #22)
Main's Windows CI after #18 failed once in `test_worker.py::test_partial_merge_and_per_share_errors`: the job ended
"failed" instead of "done" after its last molecule (the job's error is not in the test's output).
- **Likely cause** (the CI log has no job error; the test now prints it): `_RunLock.__exit__` removes `jobs.lock` at the end of every run, and status polls read that file
  (`open_run` → `_read_lock`: the worker's job status, polled every 0.1 s by the test; the GUI; `status`). On Windows
  a file cannot be deleted while it is open, so the removal raised `PermissionError` (WinError 32) out of
  `run_resumable`, and `JobManager._execute` marked the job failed. Antivirus opening the freshly written file can do
  the same.
- **Shown on the workstation** (`scratch/lock_race.py`): with one thread reading the lock without pause, 1,592 of 3,000 lock
  releases failed. 40 runs of the CI test itself on the workstation passed (the poll must land within the removal's moment).
- **Fix:** `_remove_lock_file` tries again for up to 2 s (100 x 20 ms). Should it still fail, the run does not fail:
  the file is emptied (`{}` names no process, so the next run takes it over without a word) and the failure logged.
- Tests (`tests/test_jobs.py`): three refused removals, then success; a removal that never succeeds (the run ends,
  the lock is emptied and taken over); 300 lock releases while a thread reads the lock (the race itself, with real
  files; it fails on Windows without the fix).
- Not changed: the worker's own files (`worker.json`, the state) are written with `_atomic_write`, which already
  retries a refused rename.

## FIX-3: The CI hang inside Qt · done 2026-11-16 (PR #21; the cause: a deadlock, fixed)
The hang seen since Nov 13 (Linux, before UI-8) now has two stacks, both from a project load in the tests' `_wait`
loop: a worker's `done` → `MainWindow._on_loaded` → `_after_load` → `_on_molecule_selected` → `_show_info` →
`collapsible.Notes.set_notes` → `self.setVisible(...)`, inside Qt.
- Windows CI of #15: `test_project.py::test_save_and_reopen_a_built_project`.
- macOS CI of #18: `test_project.py::test_resume_a_half_finished_project`.
Not reproduced: 65 runs of those tests on Linux (xvfb) passed. pytest-timeout (`--timeout=300`) never fired, so the
main thread holds the GIL inside Qt; the other Python threads (the machine monitor's, the sensors') were idle.
**Found (the watchdog's first catch, #17's Linux run) and fixed in the branch: a deadlock.**
- The GUI thread holds the GIL: `Notes.set_notes` → `QWidgetPrivate::setVisible` → `QLayout::activate` →
  `QObject::connect` → `QBasicMutex::lockInternal`.
- A worker's thread, ending, ran its `deleteLater`: `QObject::~QObject`, holding a signal-slot lock, called into PySide,
  which waits for the GIL to drop the Python callables connected to the worker.
- Fix: `gui.workers.start` runs the worker through a `_Runner` in the new thread. When `run` returns, the runner moves
  the worker and itself back to the GUI thread, so both are destroyed there; `_Keeper` releases the Python wrapper
  there too. Test: `tests/test_gui_workers.py` (where the worker is when it reports and where it is destroyed; 60
  workers ending while widgets are shown and hidden).
Done in the branch: `tests/hang_watchdog.py`, a separate process started by `tests/conftest.py` when
`QB_HANG_WATCHDOG=1` (CI). After 200 s it runs `py-spy dump --native` on the test process (every thread, native
frames, which thread holds the GIL; `sudo -n` on Linux and macOS; on macOS py-spy's Python frames plus `sample`), and
at 420 s it kills the process.
Tests: `tests/test_hang_watchdog.py` (a process stuck in the regex engine, holding the GIL).
Next: the watchdog stays in CI for whatever hangs next. Other QObjects created in Python and destroyed in a non-GUI
thread would deadlock the same way: `gui.workers.start` is the only place that moves one to another thread today.

## FIX-2: Flaky CI tests · done 2026-11-15 (PR #13)
- macOS (PR #11's first run): `tests/test_telemetry.py::test_the_load_sampler_records_sensors_or_skips_them`,
  KeyError `cpu_temp_c`. `LoadSampler.run` checked `_halt` before the sensor read and appended the sample anyway;
  it now drops a sample that `stop()` cut short. Test: `test_a_sample_cut_short_by_stop_is_dropped` (fails on main).
- Linux (PR #7's first run, 2026-11-13): a hang after 64 % (tests 1081-1152 in the collection order), killed at
  the 30 min job limit; the rerun passed. Not reproduced in the cloud (test_project.py 8 of 8). The Linux job now
  uses `--timeout-method=thread` and the suite `faulthandler_timeout = 240`, so the next hang names its test and
  shows every thread's stack.
Follow-up: when it shows up, fix the hanging test itself (a modal dialog at window close is the first suspect).
- 2026-11-16, the first stack (Windows CI of #15): `test_project.py::test_save_and_reopen_a_built_project` hung in
  `MainWindow._show_info` → `collapsible.Notes.set_notes` → `self.setVisible(...)`, inside Qt, during the test's
  `_wait` loop (a worker's `done` → `_on_loaded` → `_after_load` → `_on_molecule_selected`). 25 runs of the test on
  Linux (xvfb) did not hang. Next: a layout loop of the word-wrapped labels in the Molecules dock's Details is the
  suspect (the Nov 13 Linux hang, before UI-8, was in the same tests).
- 2026-11-16: the flaky `test_worker.py::test_archive_manifest_sync_temp_files_and_failed_streams` (Windows, a
  `jobs.sqlite-wal` left behind). First try (#16): `JobManager._checkpoint` retries until a status poll lets go; it
  failed again on #21's Windows run. The cause: a poll that came after the checkpoint and before "done" opened the
  table read-only, which recreates `-wal` and cannot remove it. Fixed (#21): from the final counts on, `Job.finishing`
  makes polls report the stored counts without opening the table.

## UI-8: Collapsible sections and pop-out panels · done 2026-11-15 (PR #12)
Brandon (2026-11-14): "At the moment it's kind of crowded. I think more collapsible menus / pop outs would be helpful";
he chose all four parts: the Pipeline dock, the molecule details, the Table toolbar, pop-out panels.
Done in the branch:
- `gui/collapsible.py` (`Section`, `Accordion`, `InfoLabel`, `InfoButton`, `with_info`, `Notes`) and
  `gui/popout.py` (View > Pop out panel, Dock all panels; `pop_out`, `dock_all`).
- The Pipeline dock's stage list and page stack became an accordion of stage sections. Each has a one-line summary
  (`MainWindow.stage_summary`), a status, and an ⓘ for the stage's description; the method guide's labels sit behind
  an ⓘ next to their field (⚠ when they say something is missing). `stage_items` maps a stage to its `Section`.
- The Molecules dock's Details section and folded notes; the Table toolbar's Columns / Features / Models / Export
  menus; the Plot's ⧉ pop-out button.
- Tests: `tests/test_gui_declutter.py`. Screenshots: `docs/dev/screenshots/UI-8/`.
Follow-ups: done in UI-8b (below).

## UI-8b: The UI-8 follow-ups · done 2026-11-16 (PR #19)
Brandon (2026-11-16): "Do the backlog items." The three UI-8 follow-ups:
- **Surfaces panel** (`gui/surface_panel.py`):
  - Surface / Color / Descriptors sections (`surface_section`, `color_section`, `descriptors_section`) with summaries;
    the colour bar outside them;
  - `legend_text` is an `InfoLabel` behind the ⓘ next to Color; `source_info` and `iso_info` hold the other ⓘ texts;
  - the HTML escaping of "Area V < 0".
- **Spectra panel** (`gui/spectra_panel.py`):
  - the Spectrum menu: `spectrum_btn`, with the actions `a_load`, `a_clear`, `a_export` (`load_btn`, `clear_btn`,
    `export_btn` name them);
  - `broadening_info`;
  - the Normal modes section (`modes_section`), which folds the splitter's right pane to its header.
- **Settings dialog** (`gui/settings_dialog.py`):
  - Compute in sections (`sections`: `compute_threads`, `compute_gpu`, `compute_lanes`, `compute_limits`,
    `compute_hardware`) with summaries;
  - the notes of the other tabs, and `priority_info`, behind ⓘ.
- **Dock title bars** (`gui/popout.py`): `DockTitleBar` (title, ⧉, ✕) on every dock at start (`add_title_bar` for
  later ones). It leaves drags and double-clicks to Qt.
- **Results fold** (`gui/results_fold.py`): `ResultsFolder` and View > Fold Results for one conformer.
- Tests: five more in `tests/test_gui_declutter.py`. Screenshots: `docs/dev/screenshots/UI-8b/`.
Follow-ups:
- An OVITO viewport dock opened later still has Qt's own title bar (`popout.add_title_bar` can give it the ⧉ bar).
- The Results fold stops at the dock's minimum height (about 120 px with the tab bar); a smaller minimum for the
  table would fold it further.

## RESUME-1: A resume keeps the run's memory, cube and time-limit options · done 2026-11-14 (PR #11)
Found in the b03 ESP pass (2026-11-13/14). The controller resumed the small part with a bare `build --resume`. That ran
at the settings' 4 GB per Psi4 job (the run was started with `--qm-memory 6`), so the lane plan gave 5 × 2 instead
of 3 × 4, and 14.5 GB of cubes were kept (`--no-keep-cubes` was lost). Of the 54 failures, 29 came before the resume
(the largest molecules run first) and 24 + 1 after it: the 1,800 s limit, more than the resume, caused them.
Done in the branch:
- `jobs.RUN_OPTION_FIELDS` (`qm_memory_gb`, `qm_keep_cubes`, `molecule_timeout_s`), `stored_run_options`,
  `run_options_note`.
- The CLI resume, `stored_config` (Build > Resume run folder, a new phase) and the note line.
- Tests: `tests/test_cli.py::test_resume_keeps_the_runs_options`,
  `tests/test_jobs.py::test_stored_config_keeps_the_run_options_and_old_runs_take_the_settings`.
Follow-ups:
- Done (RESUME-2, below): store the `--lanes` spec with the run.
- `--priority` and `--free-threads` stay per machine on purpose.

## RESUME-2: A resume keeps the run's `--lanes` on the machine that chose it · done 2026-11-16 (PR #15)
RESUME-1's follow-up. Before this, a resume without `--lanes` used the machine's lane setting, whatever the run had been
started with.
Done in the branch:
- `run.lanes` in the job table: `{"spec", "source": "option" | "settings", "machine"}` (`jobs.lanes_record`). The
  column is added to older tables when they are opened for writing.
  - `JobStore.set_lanes` never replaces a chosen spec with a machine's setting.
  - `run_resumable(lanes_spec=...)`, `RunStatus.lanes`.
- `jobs.resume_lanes`: `--lanes` given now wins. Otherwise the run's chosen spec is used on the machine that chose it
  (with a note). Otherwise this machine's setting is used, with a note when the run chose one elsewhere.
- The CLI `build --resume`, `status`, and the GUI's `BuildWorker` (a resume reads the stored spec; a new batch stores
  the machine's setting).
- Tests: `tests/test_jobs.py::test_the_run_stores_its_lanes_and_a_resume_on_that_machine_reuses_a_chosen_spec`,
  `::test_the_gui_resume_takes_the_runs_lanes`, `tests/test_cli.py::test_resume_keeps_the_runs_lanes_on_the_same_machine`.
Not stored: lanes of a grid worker's job (the worker plans its own) and of `grid build`'s local share.

## SCRATCH-1: Scratch cleanup and Psi4 memory · done 2026-11-13 (PR #5)
Brandon (2026-11-13): "The hard drive was nearly filled last night. Can you make it so that the scratch files are
deleted after we extract the information we need?"

What happened: killed lanes left 59 Psi4 job folders in `%TEMP%\QuantaBench`, holding 479 GB of DF integrals.

Done:
- owner records in scratch folders;
- failed jobs keep only their inputs and logs;
- `external.sweep_scratch` after lane kills, at pool close and at run start;
- `quantabench-cli scratch [--clean]`;
- `--qm-memory`, with QM lanes sized by memory and sharing the thread budget;
- the DF-on-disk notice from the driver and in `qm["df"]`.

Follow-ups:
- QM memory per molecule within one run (lanes with different memory, or a size-aware queue) instead of one value per run.
- Why the workstation's native Windows Psi4 looked slower than the Mac's (both ran DiskDF at 4 GB; disk speed or Defender scanning `%TEMP%`?). Measure one molecule on both with enough memory.

## UI-1: App shell with task workspaces · done 2026-11-12 (PR #1)
Done in the PR: the five workspaces (`gui.workspaces`), the Appearance dock, the playback bar (`gui.playback`),
the status line, a first Library (run folders under the output folder, `gui.library_panel`), per-workspace
layouts in QSettings with View > Reset layout, tests (`tests/test_gui_workspaces.py`) and screenshots
(`docs/dev/screenshots/UI-1/`). Left for later items: the Simulate workspace proper (SIM-1), the Library's
project database and search (UI-2), point group in the status bar, a properties inspector, the playback bar
for trajectory frames and molecules. Needs-Brandon: a look on Windows and macOS (HiDPI icons, the dock
layout at his window sizes).
Depends on: none (everything else in the UI plugs into this, so it goes first).
- Left activity bar switching workspaces: **Build, Calculate, Simulate, Analyze, Library**. Each is a
  page/dock set. Map the existing panels: molecule list + inputs → Build; pipeline stages → Calculate;
  surface builder / LAMMPS export → Simulate (stub page with what exists); conformer table, descriptors
  → Analyze; run folders/jobs → Library (stub).
- One **Appearance** panel (style, labels, color mode, hydrogens, show 2D) replacing the scattered view
  toolbar widgets; keep keyboard shortcuts.
- **Playback bar** under the viewport stepping through conformers now (frames/molecules later): first /
  prev / play / next / last + slider + label.
- **Status bar**: molecule count, current molecule/conformer, CPU threads, GPU name and, when on GPU,
  allocated VRAM (refresh ≤ 1 Hz, off the GUI thread if it costs anything).
- Persist workspace/dock layout per machine (QSettings), with "Reset layout" in View.
- Acceptance: all existing features reachable, existing tests pass, new tests for workspace switching
  and playback stepping, screenshots of each workspace committed in the PR.

## UI-2: Project document model · done 2026-11-13 (PR #9)
Brandon (2026-11-13): "I'd like you to also move onto the other UI-2 as well".
Done in the branch:
- `quantabench/project.py`: the `.qbproj` folder: `project.json` (format, version, name, times, the build settings,
  the table's state, workspace, selected molecule; keys of a newer version are kept), `molecules.jsonl` (light rows
  and inputs), `runs/` and `results/`. Relative paths with "/" inside the project, atomic writes, `ProjectError`
  for a newer version or a damaged file.
- `gui/project_actions.py`: File > New project (Ctrl+N), Open project... (Ctrl+Alt+O), Open recent project, Save
  project (Ctrl+S), Save project as... (Ctrl+Shift+S), Close project. It also handles:
  - the `[*]` title mark and the unsaved-changes question (New, Open, Close, and Quit with a project open);
  - pinned results written to `results/` on save;
  - builds into the project's `runs/` (every build through a run folder);
  - on open, statuses that changed in the job tables since the save, and the Analyze table's values read off the
    GUI thread (`TablePanel.read_tables`);
  - Resume (the newest run folder with unfinished molecules).
- The playback bar's Conformers / Molecules box: it steps (and plays) through the list in its order and filter.
- Tests: `tests/test_project.py` (the format, a moved project, Windows separators, versions; the window: save and
  reopen with results and the table, pinned results, a half-finished project resumed, the question, the playback
  Molecules mode). Screenshots: `docs/dev/screenshots/UI-2/`.
Left for later: Save as copies the run folders (now they stay where they are and are referenced); the Library
workspace listing projects; a project's own settings beyond the build settings (Display units, radii).
Depends on: UI-1.
- A project (`*.qbproj` folder: `project.json` + the existing per-run `jobs.sqlite` + result folders)
  holding many molecules, their inputs, results and the build settings used. New / Open / Save /
  Save As / recent projects. Unsaved-changes prompt.
- Opening a project restores the molecule list and built results (reads `summary.json` +
  `_conformers.sdf`); "Resume" continues unfinished molecules via `jobs.run_resumable`.
- The playback bar can step through molecules as well as conformers.
- Acceptance: round-trip save/open test with built results; resume of a half-finished project;
  Windows-safe paths (no absolute paths stored for files inside the project).

## UI-3: Spreadsheet, plots and formula columns · done 2026-11-12 (PR #2)
Done in the branch (stacked on `feat/app-shell`, UI-1's PR #1): the Table dock (`gui.table_panel`: one row per
molecule of the Molecules dock, row fields, posted fields, summary and descriptor values from results in memory or a
run folder's CSVs and job table read off the GUI thread, column chooser with each column's formula name, numeric
sort, filter, selection synced both ways, Ctrl+C / Ctrl+Shift+C as TSV, CSV and XLSX export), the Plot dock
(`gui.plot_panel`: two columns, colour by a third, fit line with R², click a point to select), formula columns and
the fit helper (`analysis.formula`: `ast` whitelist, NumPy tree walk, `least_squares` with standard errors, n, MAE,
RMSE, R²), the state (column choices, formulas, plot) in the window settings (`analyze/table`), Analyze's new
default (table at the bottom in front of Results/Spectra, plot on the right in front of Surfaces/Appearance;
`workspaces.STATE_VERSION` 2), tests (`tests/test_gui_table.py`) and screenshots (`docs/dev/screenshots/UI-3/`).
Left: move the state into the project document with UI-2 (`TablePanel.state()` / `restore()`); join columns from a
CSV by id (experimental targets that are not posted fields of the input, e.g. `pubchem targets` output); a line plot
and a parity view (fitted vs target with the 1:1 line); column order by drag; per-column number formats; plots of
several y columns. Needs-Brandon: a look on Windows and macOS (dock sizes at his window sizes, Excel paste of the
TSV with his locale's decimal separator).
Depends on: UI-1 (UI-2 nice to have).
- Analyze workspace: sortable/filterable table of molecules × properties (descriptors, energies,
  thermo), column chooser, copy/paste to Excel, CSV/XLSX export; selecting a row selects the molecule.
- Plot panel (pyqtgraph or matplotlib QtAgg): scatter/line of any two columns, color by a third,
  point click → select molecule, linear fit with R².
- **Formula columns**: safe expression evaluator (no `eval` of arbitrary Python; e.g. `asteval` or a
  small AST whitelist) over column names, plus a "fit coefficients" helper that fits parameters
  (scipy least squares) to a target column, e.g. the quantum parachor
  `γ = ((a·α + b·μ + c) / V)^4`. Saved with the project.
- Acceptance: tests for the evaluator (rejects attribute access/imports), fitting recovers known
  coefficients on synthetic data, GUI smoke screenshot.

## Structure input track (UI-4, UI-6, UI-7): plan 2026-11-07
Brandon asked (2026-11-07) for a ChemDraw-like editor, or at least text → 3D, and Spartan-style import
from many formats (his reference: Spartan's Model Kit and its SDF Import Options dialog). What exists
today: the Molecules box takes SMILES, names and CAS (names and CAS through PubChem) and builds 3D in
the pipeline; File > Open reads SDF/MOL, XYZ, PDB, SMILES lists and CSV/XLSX tables; File > Import
calculation output reads cclib outputs and `.spartan` documents; the 2D view (RDKit SVG) and the 3D view
(PyVista/VTK, select and move modes) display and pick but do not edit.
- **Order:** UI-4 first (small, no GUI-shell dependency, most of the day-to-day value), then the UI-6
  spike, then UI-6 and UI-7 in parallel. A full ChemDraw (publication drawing, reactions, text,
  name ↔ structure, NMR prediction) is not the goal: the goal is getting structures into calculations.
- **Facts checked 2026-11-07:** RDKit 2026.09 reads ChemDraw `.cdx` and `.cdxml` and writes CDXML
  (`rdkit.Chem.rdChemDraw`), reads MOL2 and Marvin MRV, and InChI. Ketcher (EPAM, Apache-2.0, v3.18.0,
  Sep 2026) has a standalone mode that runs in a browser with no server. Schrödinger's Sketcher
  (BSD-3-Clause, built on RDKit and Qt 6, release v2026-4 on 2026-10-30) has a WebAssembly build.
  Kekule.js is MIT. Open Babel is GPL-2.0, so it is only ever run as a separate program the user
  installed, never bundled or imported. The embedded editors need QtWebEngine, which is in
  `PySide6-Addons`, not the `PySide6-Essentials` QuantaBench depends on: an optional `[editor]` extra.

## UI-4: Import from many formats, with Spartan-style import options · in-PR (formats and options done; paste, drag-and-drop and Copy as in PR, branch `feat/ui-4-clipboard-dnd`)
Depends on: none (goes into today's File menu and Molecules box; UI-1 later moves it into Build).
- **Done 2026-11-07 (branch `feat/import-formats`):** the formats, the Open Babel fallback, the Import Options
  dialog with Settings > Import and `Settings.import_`, the text box (InChI, InChIKey), the CLI flags
  (`--keep-coords`, `--filter`, `--label-order`, `--same-structures`, `--import-options`, ...), posted fields as
  molecule-list and `summary.csv` columns, and *keep coordinates* in the pipeline (seeds RDKit's conformer
  search). Code: `chem/formats.py`, `chem/import_options.py`, `gui/import_options.py`; tests:
  `tests/test_import_formats.py`, `tests/test_import_options.py`, `tests/test_gui_import_options.py`.
  Screenshot: `docs/img/import_options_dialog.png`.
- **Paste, drag-and-drop and Copy as · in-PR 2026-11-22 (branch `feat/ui-4-clipboard-dnd`):** built on the
  2026-11-07 clipboard work (`feat/clipboard`, merged; bullet below).
  - Paste now goes through File > Open's path: read on a worker thread, then the Import Options (dialog when the
    paste has choices and Settings > Import says to ask; filters, 2D → 3D, posted fields as list columns). Text
    adds XYZ (frames or bare `element x y z` lines, bonds by `rdDetermineBonds`, stereo from 3D), tables with a
    SMILES column (CSV, `;`, or tab-separated cells copied from a spreadsheet; the Pipeline dock's SMILES/ID
    column and SMILES preference apply), and InChI / InChIKeys (already in `parse_text`). Nothing readable → a
    message naming the formats found and what QuantaBench reads.
  - Drops on the window and the molecule list: any number of files through `open_files` (every `read_file`
    format; unknown types get File > Open's message, or Open Babel), outputs through Import calculation output,
    a run folder, a `.qbproj` project, any other folder as the files directly in it.
  - Copy as (Edit menu and the molecule list's right-click menu, `MoleculeListView.menu_hooks`): SMILES as
    entered, canonical SMILES, InChI, InChIKey, MOL/SDF, XYZ (the conformer shown; best conformer of other built
    molecules; input 3D or the preview geometry otherwise), CDXML. SMILES of 14,000 rows copies at once (0.25 s
    with the selection, cloud VM); other formats above 200 molecules run on a worker thread (14,000: canonical
    SMILES 1.8 s, SDF 3.8 s, InChI 4.8 s; the window blocks ~0.2 s).
  - Tests: `tests/test_gui_clipboard.py`, `tests/test_interchange.py`. Screenshots: `docs/dev/screenshots/UI-4/`.
- **Left:** Needs-Brandon: open a few real
  ChemDraw `.cdx` files (the CDX reader is tested on hand-built files only; RDKit's own Python reader takes
  every file as CDXML). Follow-ups: *keep coordinates* is not used with CREST (it makes its own start
  geometry); posted fields show for molecules opened in this session, not for rows of an opened run folder;
  ORCA `%coords` and Z-matrix blocks and CIF disorder are not read; `.xyz` files (File > Open) do not take stereo
  from the 3D coordinates as pasted XYZ now does; Ctrl+C in the molecule list still copies the current cell's
  text (Qt's default), not the molecules; XYZ text of a charged structure needs a MOL block or SMILES (bonds are
  assigned for charge 0).
- **Formats** (all through `chem.inputs.read_file`, so the CLI gets them too): SDF/MOL V2000 and V3000,
  MOL2, PDB, XYZ (bonds by `rdDetermineBonds`), SMILES lists, CSV/XLSX, InChI lists (`.inchi`, one per
  line), ChemDraw `.cdx` / `.cdxml` (RDKit), Marvin `.mrv`, CML (small reader, or Open Babel), CIF of
  molecular crystals and Gaussian/ORCA inputs (`.gjf`, `.com`, `.inp`, through ASE, already a dependency;
  molecules split by connectivity). The cclib outputs and `.spartan` reader stay under Import calculation
  output, but File > Open should accept them too.
- **Open Babel fallback:** when `obabel` is on PATH (or set in Settings > Engines), any other format it
  reads is converted to SDF in a temporary file first. Report which reader was used.
- **Import Options dialog** (shown when a file has choices; "Don't ask again" stores the choices in a new
  `Settings.import_` section, editable in Settings):
  - Fixup: *Convert 2D structures to 3D* (all z = 0, or a 2D flag in the MOL header → embed);
    *Grow hydrogens* (add explicit H with coordinates); *Keep the file's 3D coordinates* as the first
    conformer instead of re-embedding.
  - Additional data: *Post SD / table fields to the molecule table* as columns (numbers parsed, units
    kept in the header); choose the ID and name fields.
  - Filters: remove water, remove HCl / HBr / small inorganic fragments, strip counter-ions (today's salt
    stripping, now visible), keep the largest fragment; each removal reported per molecule.
  - Labeling: atoms in read order (default, matches the file) or in a standard order (RDKit canonical
    ranks), so atom numbers line up with another program's when needed.
  - Several structures of one constitution in a file: separate molecules, or conformers of one molecule
    (Spartan's documents work this way).
- **Text box:** also accept InChI (`InChI=1S/...`, parsed locally) and InChIKey (PubChem).
- **Clipboard and drag-and-drop:** · done 2026-11-07 (`chem.interchange`, `gui.clipboard`), needs a hand
  check. Edit > Paste (Ctrl+V) reads, in order: ChemDraw CDX (Windows "ChemDraw Interchange Format", which
  Qt names `application/x-qt-windows-mime;value="ChemDraw Interchange Format"`; macOS
  `com.cambridgesoft.chemdraw.cdx`, read straight from NSPasteboard because Qt hides unknown pasteboard
  types; `chemical/x-cdx`), CDXML (`chemical/x-cdxml`, `com.cambridgesoft.chemdraw.cdxml`, a "CDXML"
  format), MDL molfile (`chemical/x-mdl-molfile` / `-sdfile`, Windows "MDLCT"), file URLs, text (MOL/SD,
  CDXML, MRV, SMILES, names, CAS; InChI once `parse_text` takes it). Names are compared ignoring case; the
  log names the format used, and on macOS the pasteboard types are logged. RDKit 2026.09 cannot read CDX
  bytes from Python, so `cdx_to_cdxml` translates CDX for its CDXML reader (checked on ChemDraw-written
  test files from RDKit; the fixtures here are written by hand). Edit > Copy as SMILES / MOL / CDXML goes
  the other way (text always; also `chemical/x-mdl-molfile`, Windows "MDLCT", `chemical/x-cdxml`). Drops:
  files open (File > Open path, outputs through Import calculation output, a folder as a run folder), text
  is parsed.
  - **Needs Brandon (hand check, Windows and macOS):** in ChemDraw draw PFOA, K-PFOS (salt) and GenX (one
    stereocenter), plus a label such as CF3 or CO2H; Ctrl+C; in QuantaBench click the molecule list and
    press Ctrl+V. Check the SMILES and stereo, and read the log line ("Paste: N molecule(s) from ..."). On
    the Mac the log also lists the pasteboard types: send them if the paste fails. Then Edit > Copy as
    MOL / CDXML / SMILES and paste into ChemDraw (Ctrl+V, or Edit > Paste Special > MOL Text / CDXML Text /
    SMILES). Also drag a `.sdf` file and a ChemDraw selection onto the window.
  - Not done: writing binary CDX for the clipboard (so a plain Ctrl+V in ChemDraw gets ChemDraw's own
    format), and macOS drags from ChemDraw that carry only the CDX pasteboard type (Qt hides it).
- Acceptance: one small fixture per format (made by RDKit or written by hand, never from Brandon's data)
  round-trips to the same canonical SMILES; the options each have a test (2D → 3D gives non-zero z,
  filters drop water and HCl, SD fields become columns, standard vs read order); clipboard parsing
  tested on recorded CDX/CDXML bytes; screenshot of the dialog. Mac and Windows clipboard checked by
  hand (needs-Brandon for the ChemDraw copy, since only he has ChemDraw).

## UI-6: 2D sketcher (embedded open-source editor) · done on Linux; Windows/macOS check needs-Brandon
Depends on: UI-4 for the MOL/CDXML plumbing; the spike can start any time.
- **Choice (Brandon, 2026-11-07): Schrödinger's Sketcher**, so no Ketcher spike. WebAssembly build of
  release `v2026-4` (BSD-3-Clause, built on RDKit and Qt 6), in a `QWebEngineView`.
- **Done (branch `feat/sketcher`, 2026-11-07):**
  - Files: `sketcher_assets.py` downloads `sketcher-2026-4-wasm.zip` (17.5 MB) on first use into the programs
    folder (`.../QuantaBench/programs/schrodinger-sketcher-2026-4-wasm/`), checks the pinned sha256, unpacks
    it and writes Schrödinger's licence next to it (copy in `src/quantabench/licenses/`). Offline: the error
    gives the URL and `quantabench-cli sketcher --install --from-zip FILE`. Overrides for a self-built
    version: `QUANTABENCH_SKETCHER_DIR`, or `quantabench-cli sketcher --use DIR` (this machine only).
  - Serving (`gui/sketcher_web.py`, `gui/sketcher_view.py`): a custom scheme `qb-sketcher://app/`
    registered before the QApplication (`gui/app.py`), not a local HTTP server. It needs the
    `FetchApiAllowed` flag (Emscripten fetches the wasm). No COOP/COEP: the build has no threads.
    `.wasm` goes out as `application/wasm`. The view uses Qt Quick's software renderer: with Qt's
    default GPU composition the web view and VTK's OpenGL 3D view spoiled each other under Linux/Mesa
    (black sketcher, clipped 3D view); `QUANTABENCH_SKETCHER_GPU=1` turns it back on to try.
    PySide6 6.12 returns JavaScript objects from `runJavaScript` as `''`, so the scripts answer in JSON.
  - Dock (`gui/sketcher_dock.py`, Build > Sketch a molecule..., View > Sketch): New, Edit selected (the
    input as MOL V3000 with 2D coordinates), Add to list, Replace selected (keeps id, name and place), and
    Insert fragment (CF3, (CF2)n, COOH, COO-, SO3H, SO3-, SO2NH2, CH2CH2, ether O, phosphate; added next to
    the drawing, joined by drawing a bond). The hand-off (`chem/sketch.py`) reads the MOL V3000 export
    with RDKit and reports valence, kekulization and query-atom problems in words before anything is
    added. Without Qt WebEngine the dock says `pip install "quantabench[editor]"`.
  - Tests: `tests/test_sketcher_assets.py`, `tests/test_sketch.py`, `tests/test_gui_sketcher.py`; with
    `QB_TEST_SKETCHER=1` the last one drives the real sketcher (PFOA, PFOS, GenX and 6:2 FTS round trip
    to the same canonical SMILES, an error message, a fragment, Edit selected, Add to list). It passed under
    xvfb on Linux (sketcher ready in about 3 s).
- **Left:**
  - Brandon: open the dock on Windows (the workstation, AMD) and the MacBook, with `pip install -e ".[editor]"` and
    `quantabench-cli sketcher --install`; check drawing, the 3D view next to it, and screenshots. Check
    whether `QUANTABENCH_SKETCHER_GPU=1` draws correctly there (faster, if it does).
  - Keep the coordinates of unchanged atoms on Replace selected (RDKit `coordMap` embedding), with UI-7's
    shared edit model.
  - Fragments joined at the selected atom, not added as a separate piece (needs the sketcher's selection,
    which its JavaScript API does not expose). Unify the list with `chem/fragments.py` (UI-7).
  - Copy as / paste from the sketcher's other formats (CDXML, MRV) once UI-4's clipboard plumbing is in.
  - The installers could run `quantabench-cli sketcher --install` with `--with-engines`.
- **Original spike plan (not needed after the choice):** load Ketcher (standalone) and Schrödinger's
  Sketcher (WASM) in a `QWebEngineView` dock and compare: MOL V3000 round trip through RDKit (charges,
  stereo, isotopes, explicit H), PFAS drawing speed (long CF2 chains, sulfonates), templates (custom PFAS
  fragments), start-up time, download size, offline use, Windows / macOS / Linux. Fallback if neither
  works: our own editor on the RDKit 2D view (QGraphicsScene: change element, add/delete atom, bond order
  cycle, charge ±, clean 2D).
- **Integration:** a Sketch dock in Build. "New" opens a blank canvas; "Edit" loads the selected
  molecule. *Add to list* and *Replace selected* hand a MOL V3000 block to `chem.inputs`, which builds 3D
  as usual; the 3D view updates, keeping the coordinates of unchanged atoms where it can (RDKit
  `coordMap` embedding). Structure problems (valence, unknown atoms) are reported before the hand-off.
- **PFAS fragment templates:** CF3, (CF2)n chain builder, ether O, carboxylic acid / carboxylate,
  sulfonic acid / sulfonate, sulfonamide, fluorotelomer CH2CH2 linker, phosphate, common rings.
- **Packaging:** the editor's files are downloaded on first use into the programs folder, pinned by
  sha256 (like the CREST and tblite builds), with their licence and NOTICE files; nothing large goes in
  the repository. `pip install quantabench[editor]` adds Qt WebEngine (the `PySide6` package: its WebEngine
  is in PySide6-Addons up to 6.11 and in PySide6-WebEngine from 6.12); without it the dock says what to
  install and everything else still works.
- Acceptance: tests that drive the editor through its JavaScript API (draw PFOA, PFOS, GenX and 6:2 FTS
  from SMILES and templates; the hand-off matches canonical SMILES); works offline; screenshots on
  Windows and macOS.

## UI-7: 3D model kit (Spartan-style builder in the 3D view) · core done 2026-11-07; build mode ready
Depends on: none to start (the 3D view exists); UI-1 places it in Build.
- **Done (core, branch `feat/model-kit-core`):** `chem/edit.py` (the edit model, commands, `History`) and
  `chem/fragments.py` (the fragment library). Every operation below is a plain function with tests;
  PFOA, PFOS, GenX and 6:2 FTS build with kit operations and match MMFF94 within 0.03 Å.
- **Left:** the GUI build mode (stubs, picking, palette dock, QUndoStack wrapper, Ctrl+Z / Ctrl+Y,
  screenshot). Start from `docs/dev/MODEL_KIT_GUI.md`. Also left: the InChI and Clipboard buttons (the
  SMILES path exists: `fragments.from_smiles`), the Peptide / Nucleotide tabs, and sharing the fragment
  list with the UI-6 sketcher.
- **Build mode** in the PyVista view (next to select and move): a palette of atom types with their
  hybridization, as in Spartan's Organic tab (C sp3 / sp2 / sp, N sp3 / sp2 / sp, O, S, P, Si, H, F, Cl,
  Br, I, =O, =S, ≡N). Free valences are drawn as short stubs; clicking a stub attaches the chosen atom at
  its ideal bond length and direction (VSEPR geometry from the hybridization). Clicking an atom with a
  type selected replaces it.
- **Groups and rings:** carboxylic acid / carboxylate, sulfonate, sulfonamide, CF3, ester, amide,
  nitro, phosphate; benzene, cyclohexane (chair), pyridine, five-membered rings; the PFAS fragments from
  UI-6 (shared definitions). Buttons for *SMILES*, *InChI* and *Clipboard* add a fragment from text.
- **Edit tools:** delete atom/fragment, change bond order, formal charge ±, make/break bond, measure and
  set distance / angle / dihedral (dragging a dihedral rotates the smaller side), and *Minimize* (MMFF94,
  UFF fallback, with chosen atoms frozen). A quick local relax runs after each edit.
- **One edit model** for 2D and 3D: `chem/edit.py`, an RDKit RWMol plus coordinates where every change
  is an undoable command; QUndoStack, Ctrl+Z / Ctrl+Y.
- Acceptance: tests that build PFOA, PFOS, GenX and 6:2 FTS with model-kit operations (no GUI clicks
  needed: the commands are plain functions) and match canonical SMILES; bond lengths within 0.05 Å of
  MMFF after Minimize; undo/redo round-trips; screenshot.

## ADV-1: Model advisor, a review of the molecules before a run · done 2026-11-09 (branch `feat/advisor`); follow-ups ready
Brandon (2026-11-09): "Maybe as part of the job submission process, there can also be a review checking what types
of atoms and groups are present in the molecules about to be submitted, and which makes recommendations of which
models to choose from. That's another reason I found Spartan often difficult to use. I didn't know how to choose
models or what they were intended to correct."
- Done:
  - `chem/advisor.py`: `review(inputs, cfg) -> Review`.
    - The profile, each count with its first 25 molecule ids: elements; charge class at the run's pH; net charges;
      open shells; PFAS by the OECD 2021 definition and the perfluoroalkyl chain length; 16 functional groups by
      SMARTS; pure hydrocarbons; heavy atoms and rotatable bonds; macrocycles; undefined stereo; fragments and what
      the run removes; isotopes; metals and unusual elements.
    - Findings (`level`, `topic`, ids, message, `changes` as BuildConfig fields, why, guide entries, evidence), each
      tied to the DFT conformer test, RANK-1, RANK-1b or a method's documentation. `rank_verdict` is the measured
      ranking table by method family, molecule class and phase.
    - Three presets (Fast screen, Recommended, Careful), each a valid BuildConfig with a rationale and a cost
      estimate. They never use a method outside a molecule's elements, charge or spin.
    - Speed: 14,000 molecules (median 34 heavy atoms) in about 7 s cold on a 2-vCPU cloud VM, 0.3 s warm. The
      fast path skips RDKit's normalization for neutral single-fragment records and gives the same species. The
      protonation rules' SMARTS are compiled once, and their distance matrix is built only near an N+.
  - `method_guide.py`:
    - `COVERAGE`: element, charge and spin coverage of every method, basis set and CREST level, with its source
      and how it was checked. MMFF94 was checked with RDKit, tblite's Z ≤ 86 by running it, and AIMNet2 and MOPAC
      against the engines.
    - New entries g-xTB, r2SCAN-3c and the -3c composites, BSSE and gCP, conformer ranking, protonation (54
      entries).
    - `corrects` / `use_for` / `avoid_for` on 35 entries.
    - The coverage table in docs/METHODS.md.
  - CLI: `quantabench-cli review` (build's inputs and options, `--phase gas|water`, `--json`). `build` and
    `grid build` print a short review header; `--no-review` skips it.
  - GUI: `gui/review_panel.py`: the Review window before a run (Run anyway / Cancel run) and on demand (Review...,
    F6), with the profile (click-through to the molecules), findings with icons, Apply, Explain and the presets.
    `MainWindow.request_run` is the Run buttons' entry; `run_build` never asks.
  - User guide: docs/CHOOSING_METHODS.md.
- Open:
  - Done 2026-11-11 (RANK-2): `rank_method` in the rules and the presets. The ranking rules judge the level that ranks
    each molecule (`advisor.rank_level`); in water the rank step (g-xTB + GFN1-xTB CPCM) replaces the DFT re-rank as
    the recommendation, in the gas phase AIMNet2 single points replace switching the refinement method; every preset
    sets `--rank auto`, and Careful checks the rank step's lowest 8 with DFT. Where no rank engine is installed the
    old DFT re-rank stays the recommendation.
  - r2SCAN-3c in the QM stage. Psi4 runs -3c methods with an empty basis (`energy('r2scan3c/')`); the stage takes
    'functional/basis'. Then the Careful preset uses it, the level RANK-1b recommends for DFT checks.
  - UI-5's job dialog: the review as its last page before Submit.
  - Cost: fit the refinement and QM stages from the run telemetry too (today only CREST has a fitted model; the
    rest is anchored on measured PFAS timings and scaled by size). Detect a GPU for the AIMNet2 estimate.
  - Measured rankings for transition-metal and main-group-element compounds: the review checks their coverage only.
  - Process-parallel profiling for very large lists (only needed if lists grow well past 14,000).

## UI-5b: Run on a worker from the job dialog · done 2026-11-16 (PR #18)
Brandon (2026-11-16): "Do the backlog items." UI-5's first follow-up, and the "Run on" part of DIST-3.
Done in the branch:
- `dist/remote_job.py`:
  - `RemoteJob` (one thread per job): submit as one share (`client.job_body`), follow `GET /v1/jobs/<id>`,
    cancel, then the run folder (`client.final_tables` of the synced copy, else `download_archive` into
    `from_workers/<worker>/`);
  - `choose_machine` ("auto": `client.resolve_weights` "auto" per waiting job);
  - `worker_choices` (reachable, same version), `label_for`.
- `gui/job_queue.py`: a job with `run_on` other than "local" goes to `make_follower` at once and never waits in the
  local queue; `_relay` brings the thread's updates to the GUI thread; `remote_rows` / `remote_marked` for the
  Molecules list; `stop_remote` when the window closes; "auto" that picks this computer becomes a local job.
- `gui/job_monitor.py`: the Machine column, Cancel for jobs on workers, `make_follower`, `apply_remote_rows`; a
  usage line that counts the jobs on workers.
- `gui/job_dialog.py`: Run on with Auto and the workers, a threaded `client.discover` that enables the reachable ones;
  threads, priority and GPU greyed out for a worker. `chem/recipes.py`: `JobSpec` takes a worker or "auto".
- Tests: `tests/test_gui_run_on.py` (localhost workers: synced and archive results, cancel, an unreachable worker,
  auto). Screenshots: `docs/dev/screenshots/UI-5b/`.
Follow-ups (ready):
- Split one job over several machines from the dialog (a grid build with its shares in the Jobs dock), with work
  sharing and DIST-7's moves while the window follows it.
- Follow a worker job again after the window was closed and reopened (the job's id and worker are not saved yet; its
  run folder can be opened by hand).
- The worker's lanes for the job (today the worker's own setting).

## UI-5: Job dialog (level of theory × task × resources) and job monitor · done 2026-11-16 (PR #14)
Depends on: UI-1.
Done in the branch (Brandon, 2026-11-15: "merge #12, then start UI-5 as the next UI piece"):
- `chem/recipes.py` (UI-agnostic):
  - `Task` and `TASKS`: conformer distribution, optimize, single point, frequencies; scan listed, not available;
  - `Recipe`, `BUILTIN` (Screen, Standard, Careful), the "gfn2" alias, `levels()` by family;
  - user recipes in `Settings.recipes` (`save_recipe`, `user_recipes`; bad entries skipped);
  - `JobSpec` (`config`, `problems`, `summary`) and `job_inputs` (charge and multiplicity overrides, a file's
    coordinates for the single-structure tasks).
- `gui/job_dialog.py`: the Job / Level of theory / Resources / Review pages, Save as recipe..., Submit
  (`submit_spec`).
- `gui/job_queue.py`: `Job`, `JobQueue`.
  - Every build is a job: `MainWindow.run_build`, `resume_run` and the dialog call `submit_job`.
  - The queue starts a job through `_start_job` and follows the worker's signals (`attach`).
  - `_on_finished` / `_on_failed` end the job and start the next.
- `gui/job_monitor.py`: the Jobs dock (Calculate, next to Results; workspace state version 3), Build > New job...
  (Ctrl+J), and the New job... button.
- Tests: `tests/test_recipes.py`, `tests/test_gui_jobs.py`. Screenshots: `docs/dev/screenshots/UI-5/`.
Follow-ups (ready):
- Done (UI-5b, below): Run on a worker (DIST-3), and "auto".
- Scan: constrained optimizations in the refinement (a bond, angle or dihedral), then the task.
- Single point on the file's geometry unchanged: today the structure is MMFF94s-relaxed first.
- Parallel lanes per job in Resources (today: Settings > Resources > Parallel lanes).
- The job monitor's MON-1 extras: per-job markers on the sparklines, the run's own CPU share, a throttling banner.
- Run all / Run selected while a build runs: queue it instead of the buttons being disabled.
- Recipes with more than one level: a refinement chain (MMFF → GFN2 → AIMNet2 as separate stages).

- Calculate workspace: one dialog to submit selected molecules. **Level of theory** grouped by family
  (force field / tight binding / ML potential / DFT (disabled until available)), each with charge,
  multiplicity, solvent (where supported); **task** (single point, optimize, conformer distribution,
  frequencies + thermo, scan (stub)); **resources** (threads, GPU on/off/device, memory cap; leave room
  for "Run on: machine", DIST-3).
- **Recipes**: named multi-level protocols saved in settings (e.g. "Screen: MMFF conformers → GFN2 →
  AIMNet2"), mapped onto `BuildConfig`. Ship 3 sensible presets.
- **Job monitor** dock: queued/running/done jobs, per-job progress and log, cancel, open output folder,
  CPU threads and GPU memory in use. The live machine metrics exist since MON-1 (the machine monitor strip in the
  Pipeline dock, `sensors.Monitor`): move or mirror them into this dock.
- Acceptance: tests that each preset produces the right `BuildConfig`; queue runs jobs in order with
  cancellation; screenshot.

## RES-1: MD front-end survey and Simulate workspace design · ready
Depends on: none (research only; can run any night).
- Survey how LAMMPS-GUI, VMD + QwikMD, CHARMM-GUI, OpenMM-Setup, Materials Studio Forcite/Amorphous
  Cell, MAPS, SAMSON, Packmol/moltemplate and OVITO handle: system building, force-field assignment,
  protocol setup, run control, live monitoring, trajectory analysis.
- Deliver `docs/SIMULATE_DESIGN.md`: proposed Simulate workspace flow, data model (System, ForceField,
  Protocol, Run), engines (OpenMM, LAMMPS), what to build in which order, open questions for Brandon.
  No code. Cite sources.

## SIM-1: Simulate workspace v1 · blocked (needs RES-1 design approved by Brandon)

## FIX-1: Windows CI green · done 2026-11-05
- 2026-11-05: first green CI on Windows, Linux and macOS (PLAT-2). The failures since then were separate bugs,
  each with its own item: FIX-2 to FIX-5.
- Make sure `.github/workflows/ci.yml` passes on `windows-latest` and `ubuntu-latest` with core deps
  (check the first CI runs after 0.2 is pushed; Mesa OpenGL on Windows).

---

## WIN-1 (see FORK-1): xTB dipole integrals on native Windows · done 2026-11-05 (FORK-1)
- Native Windows has no compatible dxtb `libcint` backend (tad-libcint 0.4 needs tad-mctc ≥ 0.11, dxtb 0.4
  pins 0.7), so GFN2-xTB, xTB polarizability and xTB IR intensities are unavailable there (they degrade
  with a note). Investigate: a newer dxtb release, building tad-libcint 0.3 for Windows (CI wheel), or
  a charge-response/finite-field fallback. Acceptance: GFN2 and IR intensities on Windows, or a
  documented decision why not.
- 2026-11-01: GFN2-xTB is now available on Windows through the xtb program (`xtb-gfn2`, finite-difference
  IR intensities from its dipoles). What remains is dxtb's GPU/batched GFN2 path; lower priority.
- 2026-11-05: done through FORK-1: the dxtb fork with tad-libcint 0.4 gives GFN2, polarizability and IR
  intensities natively on Windows (the workstation: matches Linux to ~1e-15 to 4e-9).

## MD-2: Liquid-slab surface tension · merged 2026-11-20 (PR #33); the pilot runs are running
- 2026-11-20, GPU: on the workstation's RX 9060 XT (OpenCL), LAMMPS' default AMD tuning gave empty neighbor lists, so there were no nonbonded forces (NPT expanded; `clear` hung). Fixed by `device_type generic` in `md.liquid.gpu_switches()`, plus a `QB_ERROR` guard in the inputs (PR with MD-BENCH-1). Other AMD/OpenCL cards untested.
Plan: the Claude Doc "Plan: first MD surface-tension tests in QuantaBench"; methods: docs/ENGINES.md, "Liquid-slab MD
and surface tension"; why: docs/SURFACE_TENSION.md, "Surface tension by molecular dynamics".
- Done (2026-11-20):
  - `md liquid-slab` and `md surface-tension`; `md.oplsaa`, `md.forcefield`, `md.liquid`, `md.surface_tension`,
    `md/ff_driver.py`; tests in `tests/test_md_liquid.py`, with a LAMMPS smoke test of both dispersion styles.
  - OPLS-AA with Watkins-Jorgensen perfluoroalkanes (85 checks against GROMACS and foyer, no difference). GAFF2
    and OpenFF Sage 2.3.0 with AM1-BCC in a conda env (`qbmd`: ambertools 23.3, openff-toolkit 0.19.0,
    interchange 0.5.5, packmol 21.2.1).
  - Pilot folders written on 2026-11-20, each at pilot size (4 x 4 nm lateral, 10 nm slab), 1 ns bulk, 2 ns settle
    and 10 ns production:
    - C6F14 OPLS-AA: ewald, cutoff12 and cutoff16 (489 molecules);
    - n-hexane OPLS-AA ewald (753);
    - C6F6 OPLS-AA ewald (803);
    - C6F14 GAFF2 ewald and C6F14 OpenFF ewald.
- Next:
  - Before the pilots: `quantabench-cli machines --bench md --md-size pilot` on the workstation (MD-BENCH-1) gives
    the ns/day of each variant and checks the GPU package's pressure at `run 0`.
  - Run the pilots on the workstation (installer `lmp.exe`, OpenMP; the cutoff16 one also on the GPU package).
    Compare σ and density with the measured values, and ewald against cutoff12/16.
  - Check that the slab plateau matches the bulk density, and that 10 ns gives an error bar under about
    0.5 mN/m. If not, use the full size (5 x 5 nm) and longer production.
  - Decide the default force field for PFAS after the pilots. OPLS-AA typing covers only alkanes,
    perfluoroalkanes and (fluoro)benzenes. Functional PFAS (acids, alcohols, ethers, sulfonyl groups) need GAFF2
    or OpenFF, or more OPLS-AA types (LigParGen-style 1.14·CM1A-LBCC charges).
  - The Träg & Zahn 2019 GAFF2 fluorocarbon refit through `--overrides`, once its parameters are taken from the
    paper (not typed from memory).
  - Semifluorinated chains need Pierce et al. 2008's modified OPLS-AA (extra H-F terms), which is not in the table.
  - A tail correction for the cutoff variants' σ (Blokhuis et al. 1995), so the GPU-friendly runs can be
    corrected rather than only compared.
  - The test-area method as a second σ estimator.
  - A GUI page (the Simulate workspace, SIM-1) and runs through the job queue.
- Open: the C6F14 reference. Freire et al. 2006 measured 12.33 mN/m at 298.15 K; Jasper's compilation (Stiles &
  Cady 1952) gives 11.44. The plan uses 12.33; both are recorded in the pilot READMEs.

## MD-BENCH-1: An MD benchmark of each machine (ns/day) · in-PR 2026-11-20 (branch `feat/md-bench`, after PR #33; the workstation run needs-Brandon)
Brandon (2026-11-20): "Maybe the worker benchmarking could include a short MD run?" Docs: docs/WORKERS.md, "The MD
benchmark"; code: `md/bench.py`.
- Done (2026-11-20):
  - `machines --bench md` and `worker --bench-methods ...,md` (never in the default list). Liquid C6F14 OPLS-AA:
    `small` has 3,000 atoms, `pilot` 9,780. The box is cached per QuantaBench version.
  - One LAMMPS process per variant: `cpu-ewald`, `cpu-cutoff12` and `cpu-cutoff16` (OpenMP), plus `gpu-cutoff12`
    and `gpu-cutoff16` (GPU package) where `lmp -h` lists it; `kokkos-*` when asked for.
  - The `run 0` CPU/GPU check of pressure and energy: `ok`, `mismatch` (unverified) or `skipped`.
  - Results in the registry key `md_throughput`; old files load. Shown in `machines` and in the Machines dock's
    "MD (ns/day)" column.
  - `md estimate` (hours from ns/day scaled by atoms), also printed by `md liquid-slab` and written into its
    README.txt. LAMMPS is now in `chem.external` (`lammps`).
  - Tests: `tests/test_md_bench.py`. A real 800-atom run checks the parsing and both outcomes of the check; it
    skips without LAMMPS.
  - Cloud container: 2 slow cores, the PyPI wheel (no GPU package), 1 thread by the default budget. The small box
    was built in 63 s with Packmol. cpu-ewald 5.5, cpu-cutoff12 6.9 and cpu-cutoff16 3.6 ns/day; 161 s in all at
    30 s per variant.
- Next:
  - Run it on the workstation with the installer's `lmp.exe`:
    `quantabench-cli machines --bench md --md-size pilot --md-threads 6`. cpu-ewald should come out near the
    6.5 ns/day of the pilot runs, and it shows whether the GPU package passes the pressure check (the half-density
    NPT). If the venv's PyPI wheel is found first (`machines` shows the LAMMPS version), set
    `quantabench-cli settings --engine-path lammps=<installer lmp.exe>`; the wheel has no GPU package.
  - The KOKKOS variants are written but have not run (no KOKKOS build here).
  - MPI ranks (`mpirun -np N lmp`) next to OpenMP threads; often faster for LAMMPS on a CPU.
  - If the slab turns out much slower than the bulk benchmark, add a slab-shaped variant (with vacuum).
  - A "Benchmark MD" button in the Machines dock. Use ns/day for splitting MD jobs across machines once MD runs as
    worker jobs (SIM-1).

## MD-1: Counter-ions in vacuum SMD (Windows LAMMPS non-determinism + ion-pair heating) · ready
- On the Windows workstation, PyPI's Windows LAMMPS wheel (2025.7.22.4) fails ~5-10% of *identical* runs of a
  TFA⁻ + Na⁺ on graphite setup with "Out of range atoms - cannot compute PPPM" (also with
  `OMP_NUM_THREADS=1`: 1/20); Linux is deterministic and passes 12/12 seeds. Reproducer:
  `scratch/ionrepro.py` (in git history of docs/dev; regenerate from tests/test_lammps.py setup).
- Physics side: a free Na⁺ falls onto the carboxylate in vacuum and releases tens of kcal/mol; even with
  Joung-Cheatham ion LJ and 20 fs ion friction the adsorbate runs at 1000+ K in short protocols.
  Options: start the counter-ion pre-paired at the head group, longer soft start, freeze spectator ions,
  or recommend no counter-ion (PPPM neutralizing background) for adsorption SMD in vacuum.
- Acceptance: a documented protocol that keeps T within ±20% of target after the soft start, and a
  minimal upstream LAMMPS issue (or a confirmed local cause) for the Windows failures.
- 2026-11-01: not only counter-ions: `test_smd_run_with_lammps_module` (TFA⁻ alone on graphite, PPPM)
  failed 2 of 6 identical runs on the Windows workstation with the same error; now also marked flaky on Windows.
  Compare with `lmp.exe` from the LAMMPS Windows installer to tell a wheel problem from an input problem.
- 2026-11-05: a third test fails the same way on the workstation, 1 of 6 identical runs: the LAMMPS run of the periodic-boundary test (TFA alone, 2-layer slab, now `test_molecule_across_periodic_boundary_runs`); marked flaky on Windows too.

## GPU-1 (= FORK-3): AIMNet2 on AMD and Apple GPUs (replace NVIDIA Warp D3) · done on ROCm 2026-11-05; MPS needs-Brandon
- 2026-11-04: Warp is CPU-only on macOS too, so the same adapter enables AIMNet2 on the MacBook's GPU (MPS,
  float32). Test the adapter on `mps` as well (needs-Brandon), and offer the change upstream to `aimnet`.
- `aimnet` 0.2 computes DFT-D3 (and neighbor lists) through `nvalchemi-toolkit-ops` / NVIDIA Warp, which
  rejects ROCm devices ("Invalid device identifier: cuda:0" on the Windows workstation). The NN itself is plain PyTorch.
- Load the model with dispersion disabled (`needs_dispersion=False`) and add D3(BJ) for ωB97M from
  `tad-dftd3` (pure PyTorch, already installed with dxtb) with AIMNet2's D3 parameters; keep the dense
  neighbor path (< `nb_threshold` atoms). Verify on CPU against the stock calculator to ~1e-5 eV and
  forces, then on GPU (needs-Brandon to run). Acceptance: AIMNet2 single + batched on a ROCm device.
- 2026-11-05: `chem/aimnet_torch.py` (Warp-free calculator: dense neighbor path, tad-dftd3 D3(BJ) with
  aimnet's parameters and reference tables). The workstation, real weights: CPU < 3.3e-6 eV and RX 9060 XT < 7.5e-6 eV
  from the stock calculator over 6 PFAS x 4 conformers; batched refinement of 8 PFOA conformers 5.7 s on
  the GPU vs 59.8 s on the CPU. Picked automatically on ROCm and MPS (`QUANTABENCH_AIMNET_BACKEND`).
  Found and fixed a stock aimnet bug on the way (`guard_neighbor_lists`, see DEVLOG). Left: a run on the
  Mac's MPS, and the upstream PR making Warp optional (text drafted; post only with Brandon's OK).

## GPU-2 (= FORK-4): dxtb on AMD and Apple GPUs · done 2026-11-05 (runs, but slower than the CPU)
- 2026-11-04: MPS has no float64 and unverified linalg coverage: the device-split wrapper (dense
  eigh/cholesky on the CPU) must cover `mps` too; then propose a configurable linalg device upstream.
- AMD's Windows ROCm PyTorch has no MAGMA/hipSOLVER linear algebra on the GPU
  (`torch.linalg.cholesky` / `eigh` raise). Options: route dxtb's small dense linear algebra to the CPU
  inside an otherwise-GPU calculation (tensors are ~100x100 for PFAS-size molecules), or document
  WSL2/Linux ROCm wheels (which include MAGMA). Measure whether GPU dxtb beats CPU at all for these sizes;
  batched conformers are where it should.
- 2026-11-05: `chem/torch_compat.py` routes Cholesky/solve (and everything on MPS) to the CPU inside dxtb
  calls; GFN1 on the RX 9060 XT matches CPU float64 to 8.6e-6 Eh. Measured 5-8x *slower* than the workstation's CPU at
  every size and batch size (host synchronizations in dxtb 0.4's overlap integrals, not the routed ops), so
  `dxtb_device()` keeps dxtb on the CPU there by default (Settings > Compute > dxtb on the GPU,
  `QUANTABENCH_DXTB_GPU`). Follow-up (later): profile/upstream the overlap-integral syncs in dxtb.

## PERF-1: Separate CPU and GPU environments on Windows · done 2026-11-06 (keep the CPU torch in `.venv`)
- Measured on the workstation (24 PFAS, 7-54 heavy atoms; `Runs/perf-2026-11-06/`, `timings.csv`). `.venv`: torch
  2.15.0+cpu with MKL (GEMM 912 GFLOP/s f32 at 12 threads). `.venv-rocm` (new, kept for GPU experiments):
  torch 2.9.1+rocm7.2.1 with OpenBLAS (49.5 GFLOP/s), and its threads oversubscribe (a 2-thread lane used
  ~2.8 cores), which likely explains the slow `doctor --bench` under ROCm.
- Wall times, CPU build vs ROCm build: MMFF 33.8 vs 39-42 s; dxtb GFN1 on the CPU 322 vs 1854 s (5.7x slower);
  AIMNet2 `auto` 76 vs 116 s (the GPU lane is 6.4 s per molecule, a CPU-build lane 14.3 s, so one GPU lane
  with 4 threads does not beat 4 CPU lanes once embedding, MMFF and the dxtb GFN1 polarizability are counted);
  tblite GFN2 126 vs 133 s; the validation protocol 241 vs 241 s (gas) and ~239 vs 257 s (water).
- So the "AIMNet2 ~10x on the GPU" of GPU-1 holds for the refinement step alone, not for the pipeline. FORK-11
  (exit hang) did not reproduce in 10 ROCm runs. Left: `doctor --bench` GFLOP/s with a warning (optional).

## ENG-1: External engines on native Windows · needs-Brandon
- Install `envs/tools-windows.yml` and `envs/psi4.yml` on the Windows workstation, run `quantabench-cli doctor`, then
  `pytest -m external` with `QUANTABENCH_XTB`, `QUANTABENCH_MOPAC`, `QUANTABENCH_PSI4_PYTHON` set (or the
  paths in Settings > Engines). Check: conda's `Library\bin` discovery, process-tree cancellation
  (`taskkill /T`), paths with spaces ("GitHub Projects"), Psi4's PCMSolver on win-64, GUI Engines tab.

## ENG-2: CREST through WSL · done 2026-11-16 for batch runs (the GUI checks below need-Brandon)
- 2026-11-20: CREST runs in a folder inside WSL (/dev/shm or WSL's disk), not in the Windows job folder through
  `/mnt/c` (CREST-SCRATCH-1).
- 2026-11-16: the workstation's path is set for that machine only, `doctor` shows it, and a 3-molecule gas test
  built: about 1.8x slower per molecule than the MacBook, conformers as low or lower. Since 2026-11-17 the
  workstation runs CREST this way in CREST-RUN-1.
- 2026-11-06: Brandon's decision: Windows runs CREST through WSL (conda-forge linux-64 3.0.2 in the workstation's Ubuntu ran
  the 20-molecule benchmark through QuantaBench). Set the workstation's path for that machine only, so the Mac keeps its own in the
  shared settings file: `quantabench-cli settings --engine-path crest=wsl:/home/<you>/micromamba/envs/qb-tools/bin/crest`
  on the workstation (or Settings > Engines there; `engines.machine_paths`). The CLI batch path is proven; the GUI checks
  below are still open.
- `wsl:/home/<user>/miniforge3/envs/qb-tools/bin/crest` from the Windows GUI on a PFAS: `--cd` path
  translation, `WSLENV` thread variables, cancellation of the WSL process, and gfn2//gfnff (needs `xtb` on
  WSL's default PATH). Only covered by mocked tests so far (a fake `wsl.exe` on Linux runs the real wrapper).
- The wrapper (`external.WSL_WRAPPER`) raises `ulimit -s` and writes the Linux PID to `.qb_pid`; cancel
  kills that PID through `wsl.exe` before `taskkill`. Children of CREST's legacy mode (xtb) may survive a
  cancel: if they do, start the program with `setsid -w` and kill the process group.

## DOC-1: Methods guide (what the methods mean) · done 2026-11-07 (branch `feat/method-guide`)
Brandon (2026-11-07): an explanation of the models in each pipeline (ωB97X / B3LYP / GFN2 ...), for a chemist
who is not a computational chemist; many of his Spartan runs had no dispersion correction, and QuantaBench should
not try to match Spartan exactly.
- Done: `chem/method_guide.py` (49 entries, references checked against Crossref), the info text and tooltips under
  each pipeline choice, Help > Methods guide... (F1), `quantabench-cli methods [NAME]`, `docs/METHODS.md` from
  `scripts/write_methods_doc.py`. Tests fail when a GUI choice or a registered engine has no entry, or the doc is
  stale.
- Follow-ups: an entry for each new engine or QM list item (the tests ask for it); entries for the frequency and
  thermochemistry choices (qRRHO, symmetry number) and the Properties page if those raise questions.

## ENG-3: Engine follow-ups · ready
- `vibrations.from_hessian(...)`: make the second half of `analyze()` public so `qm.vibrations_from_hessian`
  stops duplicating it (see the QM builder's suggested signature in DEVLOG 2026-11-01).
- QM charges (MBIS/Löwdin) as an option for the exported conformer charges (extxyz, LAMMPS data).
- Surfaces: a color bar in the 3D view; split A₊/A₋ at the V = 0 line instead of per vertex.
- `doctor`: probe the Psi4 environment for the pint/qcelemental break and missing dftd3/dftd4.
- tblite on Windows: document or support running QuantaBench inside the conda env that has tblite-python.
- GBSA as a user choice (factories accept `solvation_model`; BuildConfig always uses ALPB). The xtb program
  stops with "not parametrized" for GBSA ethanol/octanol/ethyl acetate (GFN1 also DMF, hexane): fall back to
  ALPB with a note, as tblite does.
- GIPF statistics at 0.001 au as well (the Murray-Politzer literature convention; `qm_surf_*` is at 0.002 au,
  Spartan's surface, recorded as `qm_surf_iso_au`). Water HF/6-31G*: σ²tot 478 (0.002) vs 345 (0.001).
- Psi4 log: write the MBIS free-atom and finite-field SCFs to a second output file, so cclib can re-import
  `<id>_qm_log.out` (today it stops at the free-atom UKS "Beta Occupied" block).
- Review 2026-11-01 leftovers (robustness): a Windows Job Object (kill-on-close) so a hard GUI crash cannot
  orphan xtb/CREST/Psi4; `subprocess.run(timeout=...)` in version probes can still block on Windows if a
  launcher's child keeps the pipe open (use Popen + kill); the conda scan takes the first env alphabetically,
  not the newest program version; failed-run scratch folders are kept by design but never pruned; the first
  GUI start still probes local program versions on the GUI thread (cached on disk afterwards).
- MOPAC COSMO: energies step as surface segments switch (CF3COO⁻ in water: finite-difference forces off by
  0.3 eV/Å at h = 1 mÅ); consider running MOPAC frequencies in the gas phase automatically.

## VAL-1: Validation tooling against Spartan · done 2026-11-05 (the 300-molecule run is done; three data questions need-Brandon)
Done (docs/VALIDATION.md):
- Spartan wide table: `inchikey`, `canonical_smiles` of the structure as Spartan drew it (formal charges from
  its bond orders), `structure_charge` + `charge_check`, `spartan_solvent` as QuantaBench's name
  (`spartan_solvent_raw` keeps `WASSER`), and one row per ambiguous structure (`label_resolved` empty,
  `label_source` = ambiguous, `cas_candidates`).
- `quantabench-cli sample`: reproducible validation set stratified by cohort x charge class (300, seed
  20261105), DFT-water first within a stratum, spread over size bins, AIMNet2's elements, <= 60 heavy atoms.
- `quantabench-cli validate`: joins by InChIKey / canonical SMILES / stereo-free InChIKey with a CAS
  cross-check; gas vs PM6, water vs DFT-water, solvation shifts flagged as mixed levels; N, MAE, RMSE, MSE, r,
  ρ, fit and outliers per pair, cohort and charge class; validation.csv, join.csv, summary.csv, report.md,
  parity plots.
Open (needs-Brandon):
- Re-import the 1,067 documents (`import --wide`) so every row has its keys, the ambiguous structures their
  rows and the charge check (the staged table predates them: 504 rows have no key from their `smiles`).
- Spartan's charge does not fit the drawn structure for some ions (C3A sample: 12 of 47, cations calculated as
  -1, phosphate dianions as -1); recalculate them or leave them out (the sampler does).
- The default sample has 237 molecules with PM6 and 64 with DFT-water but only 1 with both: DFT-water exists
  almost only for the charged cohorts, which have no PM6. Decide whether to add PM6 (or gas DFT) for those, or
  sample `--require dft` separately for the water comparison.
- Run the 300 in the gas phase and in ALPB water at GFN2-xTB (and AIMNet2), then `validate`. **Done
  2026-11-05:** grid workstation + MacBook, gas 18 min, water 21 min, 298/300 ok in both phases. Findings and fixes are
  in VAL-3.

## VAL-3: Follow-ups from the 300-molecule validation run · in progress
- Done:
  - Ion dipoles about the centre of mass in the Spartan import (Spartan's are about its frame origin; up to
    10.5 D off).
  - tblite's electronic dipole stored for every conformer.
  - At Spartan's own geometries, GFN2/ALPB matches B3LYP/C-PCM water dipoles to 0.41 D for anions
    (r 0.997) and 1.9 D for cations (`scratch/dipole_fixed_geom.py` on the workstation). The remaining 4.2 D MAE comes
    from the conformer each program picks.
  - 2026-11-06 (branch `feat/conformer-compare`; docs/VALIDATION.md, "VAL-3"): `quantabench-cli conformers compare`
    matches two programs' conformers. It reads Spartan documents or imports and QuantaBench run folders, uses a
    symmetric RMSD with the PFAS sibling-group factorization, torsions and stereo, and a GFN2 common level for
    search vs ranking. Results are in `Runs/conformers-compare-2026-11-06`.
    - C3A/C3B water ions: the conformer explains 88 % of the 4.35 D gap (GFN2 at Spartan's lowest geometry: 0.52 D).
      The cause is mostly ranking (38 of 55): GFN2/ALPB puts Spartan's B3LYP-lowest 11th of about 33.
    - QuantaBench's search is also short at its own level: Spartan's pool has a GFN2-lower conformer in 49 of 61
      ions, 14.8 kJ/mol lower on average. That conformer's dipole is 6.2 D from Spartan's.
    - Essential 76 (gas, 63; a new QuantaBench run with `--refine-top 5`): 8 have the same lowest conformer; the
      cause is mostly search (35 of 55). The 1.28 D gap is 37 % conformer; GFN2 is +0.76 D above ωB97X-D.
- Open:
  - Spartan's Est. DFT documents of TFA and PFPrA hold one conformer each, an anti acid (17-19 kJ/mol above syn at
    GFN2). Recompute them, or leave them out of the VAL-2 pairs; the VAL-2 pilot used these geometries.
  - Conformers, from `conformers compare`:
    - Refine more than 5 MMFF94s conformers for ions in water. The gas-phase force field ranks the pool, and the
      GFN2-lower conformers Spartan has are lost there. Try `--refine-top 20`, CREST in ALPB, and MMFF ranking with
      a solvent term, then rerun the comparison.
    - Done 2026-11-07: which level ranks the conformers for the QSAR descriptors. Brandon (2026-11-07): don't try to
      match Spartan exactly; many of his Spartan runs used methods without a dispersion correction (plain B3LYP,
      PM6), so Spartan's ranking is a reference, not the target.
      - The DFT conformer test (docs/VALIDATION.md, "DFT conformer test"; `Runs/dft-test-2026-11-07`):
        - Ions: the 10 worst. Pool: Spartan's, QuantaBench's and CREST's lowest 3 plus GFN2's best of Spartan's
          pool, 7-10 conformers per ion at common GFN2/ALPB geometries.
        - Levels: ωB97X-D, B3LYP-D3(BJ) and B3LYP with def2-SVP in PCM water; 168 + 22 Psi4 single points, 7.7 h
          on the workstation.
      - Dispersion does not explain it. D3(BJ) moves dE(Spartan's pick − GFN2's pick) by +10 kJ/mol of 68.
        ωB97X-D prefers Spartan's pick in 7 of 9 ions (B3LYP-D3(BJ) 6, plus 3 ties). No DFT level puts GFN2's pick
        lowest in any ion.
      - The GFN2 − DFT gap: Hamiltonian +39 kJ/mol, ALPB vs PCM +20. GFN2's picks pack anionic O against chain F
        (O···F < 2.6 Å) in 7 of 9 ions.
      - Decision: for ions in water, re-rank a diverse pool with DFT in PCM; GFN2's ranking and GFN2 Boltzmann
        averages are not defensible. DFT's lowest sits at GFN2 rank 2-6 (median 5), so the pool must not be
        GFN2's lowest few.
    - For the QSAR run's ions, build the DFT re-ranking:
      - A pool step that keeps folded and extended conformers from RDKit and CREST, 6-10 per ion. Then
        `--qm wb97x-d/def2-svp` (or `b3lyp-d3bj/def2-svp`, 30 % cheaper, same ranking to 2.6 kJ/mol) with
        `--qm-top N` in water.
      - Every descriptor from the QM-lowest conformer: today only `qm_*` follows it, and the xTB descriptors stay at
        GFN2's lowest.
      - Cost: 4-27 min per conformer on a 4-thread lane on the workstation (30-54 heavy atoms). The QSAR list has about 590
        molecules that are charged as drawn (largest fragment; median 26 heavy atoms, 90th percentile 46): at ~8
        conformers and ~5 min each that is roughly 400 lane-hours, a few days on the workstation + MacBook.
    - RANK-1 (done 2026-11-08, branch `feat/rank1`; docs/VALIDATION.md, "RANK-1"; `Runs/rank1-2026-11-08`): a
      cheap ranker against the DFT references, as single points at the references' geometries.
      - Sets: the DFT test's 9 ions (79 conformers), a holdout of 8 smaller ions (52 conformers, new ωB97X-D/PCM
        single points on the MacBook), and Spartan's 962 conformers of the Essential 76 neutrals (gas, ωB97X-D/6-31G*).
      - Candidates: GFN2 (gas, ALPB, CPCM, GBSA, CPCM-X), GFN1, GFN-FF, g-xTB (official v2.0.1 build), AIMNet2, PM7,
        PM6-D3H4X, MMFF94s, gas + ΔG_solv composites, HF-3c and r2SCAN-3c in PCM.
      - Ions: no cheap method reproduces ωB97X-D/PCM. The closest: g-xTB gas + a GFN1/GFN2 CPCM (ddX) solvation
        term, within 4 kJ/mol of DFT's pick in 13-14 of 17 ions (holdout 7/8), mean regret 2-3 kJ/mol, ~0.1 % of the
        DFT cost. GFN2 in any solvent model is no better than a random pick (4/17), and its 6 kcal/mol window drops
        DFT's lowest in 6 of 17 ions.
      - Neutrals (gas): AIMNet2 is the best cheap ranker (45/53 within 4 kJ/mol; GFN2 36, random 25); their dipoles
        barely depend on the conformer (0.3-0.4 D).
      - O···F: GFN2, PM7 and PM6-D3H4X put conformers with short anionic O···F contacts 15-21 kJ/mol per contact too
        low (Rg held fixed; holdout 10-16); g-xTB + CPCM and B3LYP-D3(BJ) do not.
      - Reference check: r2SCAN-3c (5 test ions) and ωB97X-D/def2-SVPD (2) in PCM agree with each other better
        than with ωB97X-D/def2-SVP. The def2-SVP levels favour compact conformers (+12 to +16 kJ/mol per Å of Rg
        relative to r2SCAN-3c). Against r2SCAN-3c, g-xTB + GFN1-xTB CPCM picks within 4 kJ/mol in 5 of 5 ions
        (ωB97X-D/def2-SVP 4 of 5, GFN2/ALPB 0 of 5). Recommended ranker for ions: g-xTB gas + GFN1-xTB CPCM ΔG_solv.
    - RANK-1b (done 2026-11-08, branch `feat/rank1b`; docs/VALIDATION.md, "RANK-1b"; `Runs/rank1b-2026-11-08`):
      RANK-1's rankers on Brandon's non-fluorinated handbook molecules and a matched PFAS control, one protocol
      (QuantaBench + Spartan pools, GFN2 re-optimized in the phase, 7 diverse conformers), r2SCAN-3c references in
      the gas phase and IEF-PCM water.
      - Sets: 12 hydrocarbons, 28 polar organics (C4A/C4B/C5A) and 20 matched C1A PFAS, gas and water; 2 usable
        C6A ions + 6 derived baseline ions (C4A carboxylates and ammoniums), water. RANK-1's PFAS ions (5 test + 8
        holdout, 52 new r2SCAN-3c points) for context. 1,079 Psi4 single points, ~31 lane-hours, 4.4 h on the workstation +
        MacBook; 10,218 cheap single points.
      - GFN2's failure is not fluorine-specific for neutrals but its size is for ions. Neutrals: GFN2/ALPB picks
        within 4 kJ/mol in 12/12 hydrocarbons, 16/28 polar organics and 14/20 PFAS in water (polar − PFAS: −0.13
        [−0.40, +0.15]). PFAS ions: 0/13, regret 36 kJ/mol, 1.6x a random pick's (O···F −17 ± 2 kJ/mol per contact
        at r2SCAN-3c). Baseline ions: 4/8, regret 4.5 kJ/mol, about a random pick's.
      - The rankers generalize to neutrals: gas, g-xTB 40/40 and AIMNet2 39/40 baseline (18/20 PFAS); water,
        g-xTB + CPCM(GFN1) 38/40 (19/20). On the 8 baseline ions g-xTB + CPCM(GFN1) picks 5/8 (PFAS ions 12/13):
        flat pools, and the electrostatics-only PCM reference favours extended chains (MMFF94s 7/8).
      - r2SCAN-3c agrees with ωB97X-D/def2-TZVP on 8 neutrals (ρ 0.98, MAE 1.1 kJ/mol). def2-SVP favours compact
        conformers without fluorine too: +7.8 ± 1.0 kJ/mol per Å of Rg for baseline ions against +15.0 ± 2.4 for PFAS
        ions (PCM), +3.7 ± 1.9 for neutrals in the gas phase (against def2-TZVP).
    - Follow-up of RANK-1b: a reference with non-electrostatic solvation (SMD, or IEF-PCM + cavitation/dispersion) for
      surfactant-like hydrocarbon ions, whose pools are flat; until then no cheap ranker is judged on them.
    - RANK-2 (done 2026-11-11, branch `feat/rank2`; docs/ENGINES.md "The rank step" and "g-xTB"; docs/VALIDATION.md,
      "RANK-2 integration check"): the rank step in the pipeline (`chem/rank.py`, `pipeline._rank_step`).
      - `rank_method` (BuildConfig, settings `build.rank_method`, `build --rank`, `grid build`, worker configs, the
        GUI's Conformer ranking page): `auto` (default), `refine`, `gxtb+cpcm`, `gxtb`, `aimnet2`, `aimnet2+cpcm`,
        `qm`. After refinement (RMSD pruning only), single points of every refined conformer at the rank level; the
        rank window (`rank_window_kj`, 30), order, Boltzmann weights and `kept[0]` use them, so every descriptor
        follows the rank level. `auto`: in water `gxtb+cpcm` > `aimnet2+cpcm` > `refine`; in the gas phase `aimnet2`
        (its elements, charge, closed shell) > `gxtb` > `refine`; explicit levels fall back to `refine`; all noted.
      - The default is `auto`; old run folders resume unchanged as `refine` (`jobs.LEGACY_DEFAULTS`), and a change of
        either rank field is a config mismatch.
      - g-xTB (`chem/eng_gxtb.py`): program entry `gxtb`, v2.0.1 builds pinned by SHA-256 (Windows, Linux, macOS
        arm64 and x86-64), the build recorded per molecule and in telemetry; the installers fetch it.
      - Checked: the rank code path reproduces RANK-1b's g-xTB + CPCM(GFN1), g-xTB and AIMNet2 picks in 196 of 196
        molecule-levels (relative energies to 2e-9 kJ/mol).
      - Open:
        - The macOS arm64 build of g-xTB v2.0.1 lacks `libgcc_s.1.1.dylib`: `install_macos.sh` now copies
          conda-forge's; report it upstream (Brandon) or wait for a fixed release. Until a machine has it, `auto`
          there ranks water runs with AIMNet2 + CPCM.
        - The CPCM term is most of the rank step's cost (about 0.5-0.9 s per conformer on one thread for 20-40
          atoms, about the GFN2-xTB/ALPB refinement's own time for PFAS ions). Options: fewer conformers into the
          rank step (the refinement window as a loose pre-filter, e.g. 60 kJ/mol), the gas-phase wavefunction as
          the CPCM SCF's guess, or more threads per lane for tblite.
        - A two-level rank: DFT (r2SCAN-3c in PCM, once the QM stage runs -3c methods) on the conformers within the
          cheap ranker's window only; `qm` today ranks every pooled conformer.
        - g-xTB as a refinement engine (v2.0.1 has analytic gradients); not needed for ranking.
        - Settle the cost estimate of the rank step in the review (`advisor.RANK_SP_S`) from run telemetry.
      - For the QSAR ions: g-xTB + CPCM ranks all. Where DFT quality is needed, DFT re-ranks the conformers within
        20-30 kJ/mol of its lowest, in PCM at r2SCAN-3c or with a diffuse basis, not at def2-SVP. First, a random
        ~10 % check subset (~80 lane-hours at r2SCAN-3c) measures the agreement on the real list.
    - Settle the conformer reference for PFAS ions (from RANK-1): ωB97X-D/def2-TZVPD or DLPNO-CCSD(T) PCM single
      points on 29811-19-6, 61119-62-8 and 75032-81-4, where ωB97X-D/def2-SVP and r2SCAN-3c disagree. That is about
      40-60 lane-hours; ask before starting.
    - Follow-ups of the DFT test:
      - Done in part 2026-11-08 (RANK-1): def2-SVPD on 2 test ions (3793-74-6, 29811-19-6). It moves ωB97X-D toward
        r2SCAN-3c (MAE 16 to 8 kJ/mol on 29811-19-6).
      - DFT optimizations of a few GFN2 picks, to see how much of their penalty is geometry.
      - Done 2026-11-08 (RANK-1): GFN2 overbinds anionic O···F-C contacts, −18 ± 2 kJ/mol per contact with
        compactness held fixed (test ions; holdout −10 ± 3). Open: DFT optimizations of a few contact-rich GFN2
        picks, to separate overbinding from the strain DFT single points see at GFN2 geometries.
      - 67940-02-7: Spartan computed the drawn sulfonamide-NH2⁺ tautomer, 183 kJ/mol above the amine-protonated one
        (ωB97X-D/PCM). Leave it out of the dipole comparison or re-protonate it; its 11.5 D gap is the proton, not
        the conformer.
    - Run the comparison on the CREST runs (`Runs/crest-2026-11-06`) once CREST works there. In the first MacBook
      runs, 18 of 20 molecules fell back to RDKit (`crest` exited with code 1).
  - `validate --boltzmann`: compare `dipole_debye_boltz` with a Boltzmann average over Spartan's DFT conformers
    (the `*_DFT_Top5` documents hold 5 per molecule).
  - The same fixed-geometry test for the gas phase against PM6 (r 0.62-0.67; C1B 0.18). It should show whether
    PM6 or the conformer is the cause.
  - GFN2's LUMO sits about 5.5 eV below PM6's and its HOMO 1.1 eV below, so its gap is 4.4 eV smaller
    (r 0.80-0.88). Document this as a systematic offset for QSAR: use the values as correlated descriptors,
    not as absolute values.
  - The Psi4 B3LYP subset on QuantaBench geometries (VAL-2) closes the loop on method vs geometry.
- Done 2026-11-06: the 300 rebuilt with 0.3.0.dev3 (`Runs/validation-2026-11-06`, the workstation OpenBLAS tblite 12x1 +
  MacBook, work sharing on): gas 7.1 min, water 7.2 min (Nov 5: 18 and 21); the machines finished within 1 s and
  12 s of each other; 298/300 ok in both phases. Validated against the v6 Spartan import (14,150 wide rows; the
  771 `MoleculeN` rows of v5 are gone):
  - CPK, like with like: gas vs PM6 area MAE 13.7 Å² (r 0.982), volume 6.7 Å³ (r 0.985); water vs B3LYP/C-PCM
    (MMFF geometry) area 7.9 Å² (r 0.996), volume 2.0 Å³ (r 1.000, slope 0.998). What is left is geometry.
  - `sasa_A2` (now the real solvent-accessible area) vs CPK area: correlation r 0.960 gas, 0.978 water (it is a
    larger surface, MAE ~200 Å² by design).
  - Dipoles, HOMO/LUMO/gap and polarizabilities are identical to Nov 5 to the reported digits (deterministic across
    lane layouts and the BLAS change); `dipole_debye_best` is now the GFN2 dipole: gas vs PM6 MAE 1.41 D (r 0.65),
    water vs B3LYP 4.35 D (r 0.84).
- Done 2026-11-06 (areas and volumes, from Brandon's screenshots of Spartan's Preferences):
  - Spartan's CPK area and volume are the union of spheres with its VDW Radii table (carbon 1.92 Å vs Bondi's
    1.70). `cpk_area_A2` / `cpk_volume_A3` reproduce CPKAREA to 0.5 % and CPKVOLUME to 0.1 % (60 C3A B3LYP
    documents); Bondi radii give −6.8 % volume, RDKit's `volume_A3` −4.1 %.
  - `sasa_A2` used FreeSASA's ProtOr radii, 0 for non-protein atoms: fixed (Bondi + 1.4 Å probe).

## VAL-2: ωB97X-D/def2-SVP vs Spartan for the 76 PFAS · in progress 2026-11-06 (branch `feat/val2-essential76`; the runs needs-Brandon)
- The QM stage reproduces Spartan's level (Spartan used 6-31G*: run `wb97x-d/6-31g*` too). Compare
  dipoles, polarizabilities, 0.002 au volumes/areas and ESP extrema with Spartan's exports; recommend a
  default level for the 4,700-PFAS descriptor run with timings (PFBA anion: ~4.5 min on 2 threads with
  PCM, cubes and finite-field α).
- Done 2026-11-06 (docs/VALIDATION.md, "VAL-2"):
  - The Essential 76 documents are a multi-step conformer search (MMFF -> DLFF03 -> Spartan's neural-network
    ωB97X-D/6-31G* geometry DLFFG1 -> ωB97X-D/6-31G* single points -> DLXD2XV ranking): 63 of 76 completed
    (1,006 conformers); 13 failed (I, Si, P, a Na salt: outside Spartan's neural networks; PFOS drawn as the
    anion, calculated as neutral). Spartan's ωB97X-D/6-31G* values are gas-phase energies and dipoles per
    conformer. HOMO/LUMO, ESP extrema, density-surface area/volume and EST_POLARIZ at that level exist for one
    molecule (perfluorooctane, a C-PCM water optimization), in all 1,067 documents.
  - Import fixes: `MoleculeN` labels resolved by structure (76 wide rows, not 847); the Est. DFT rows'
    ωB97X-D/6-31G* SCF dipole (`QM_DIPOLE`) as `dipole_debye` + `qm_dipole_level`; their best row by
    `POSTENERGY`; `validate` pairs `qm_dipole_debye` with that level (`gas_dipole_qm_vs_dlxd2xv`).
  - Pilot at Spartan's geometry (TFA, PFPrA, triflic acid, PFHpA): Psi4 = Spartan to 5e-5 Eh (exact integrals)
    or 4e-4 Eh (density fitting) and 0.001 D. Timings scale as nbf^2.5. In PCM water, the finite-field α is three
    quarters of the cost.
  - QM stage fixes found on the way: scratch cleaned after a failed free-atom SCF; `run_driver` stops a driver
    that hangs after writing its result.
- Open (needs-Brandon): run the stages in `scratch/val2/run_val2_<windows machine>.ps1` (and `run_val2_mac.sh`) when the workstation is
  free, then pick the default level from the report.
- Open (code): an option to compute the QM polarizability in the gas phase (analytic) when the SCF is PCM, or to
  skip it. Under PCM it is about 70 % of the QM stage, which takes the 4,700 run from about 1-2 weeks on the workstation in
  water to about the gas-phase cost.
- Open: the full import's wide table (`spartan-qsar-v5`) has 771 `MoleculeN` rows from the Essential 76
  conformer documents. Re-import with this branch. Spartan reference values for HOMO/LUMO, ESP and areas at
  a DFT level exist only at B3LYP/6-311+G** C-PCM (the charged cohorts); a Psi4 B3LYP/6-311+G** PCM run at
  Spartan's geometries would compare those (needs ESP/density-surface pairs in `validate`: QSAR_ELPOT_MIN/MAX,
  QSAR_ELPOT_AREA/VOLUME, polar areas vs `qm_surf_*`).

## SETTINGS-1: A settings dialog as detailed as Spartan's Preferences · ready (Surfaces & radii, Display done 2026-11-06)
Review of Spartan '24's eleven tabs: Brandon's Claude Doc "Spartan Preferences review and QuantaBench settings plan".
Rule: a setting that changes a computed value is a science setting (`jobs.SCIENCE_FIELDS`, sent to grid
workers by `dist.worker.validate_config`); display settings never rename or rescale exported columns.
- Done: **Surfaces & radii** (Spartan's Settings > Surface and VDW Radii): radii set, 111-element table with
  per-element values, SASA probe (`settings.surfaces`).
- Done: **Display** (Document), 2026-11-06: energy and relative-energy units (kJ/mol by default, kcal/mol, eV,
  hartree), significant digits and number style (`settings.display`, `quantabench.units`). Applied to the conformer
  table, the molecule list, the info and thermochemistry text, the energy windows (stored in kcal/mol, exact until
  edited: `gui.energy_widgets.EnergySpinBox`) and the Surfaces panel. Exports keep `_kcal`, `_eV` columns. Left:
  reaction-energy units (nothing computes reaction energies yet); the CLI's printouts and the Properties page's
  raw descriptor list keep the column units; orbital energies stay in eV.
- Next, one tab per change, each with a GUI test like `tests/test_gui_settings_surfaces.py`:
  - **Compute** (Spartan's Jobs): an Expert box hiding the lane spec, GPU linear algebra and dxtb-on-GPU; a
    "N cores, M logical" line.
  - **Build defaults** (Settings > Import Filters, Conformer Rules): strip water, HCl and named counter-ions
    before the largest-fragment rule (science settings); a ring-conformer option for the RDKit search.
  - **Lookups** (Databases, URLs): URL templates with `%inchikey`, `%inchi`, `%cas`, `%smiles`, `%name`
    (PubChem, NIST WebBook by CAS, nmrshiftdb, EPA CompTox), opened from the molecule list; PubChem cache folder
    and maximum age; a folder of reference tables with experimental values.
  - **Machines** (Available Servers, Embedded Server): the `dist` settings (workers, port, bind, token), now
    only in settings.json.
- Later: **Molecule display** (default 3D style, atom labels such as charges and exposed area, surface style; the
  radii set for space-filling and the charge-ESP surfaces, which use Bondi now), **Toolbar** (after UI-1),
  **Miscellaneous** (CSV decimal separator and line endings for Excel, an HTTP proxy for lookups).

## Platform and multi-machine track (plan: docs/PLATFORMS.md)

### WIN-2: xtb and MOPAC on Windows disagree with the Linux reference values · ready
- First full run of the external-engine tests on the workstation (2026-11-05, conda envs `qb-tools`/`qb-psi4` from
  `install_windows.ps1 -WithEngines`): 4 failures, all values the program prints, not QuantaBench's math:
  - xtb 6.7.1 (conda-forge win-64 build of 2026-06-10): water GFN2 dipole z -0.332 e·Å instead of -0.474
    (`test_eng_xtb.py::test_real_singlepoints_match_xtb_printout`); water in water `gfn2_dipole_debye` 2.032
    instead of 2.546 (`test_real_properties_water_in_water`). Energies and forces passed.
  - MOPAC 23.2.5: PM7 water `pm7_polarizability_iso_A3` 1.347 instead of 1.259, methanol 3.22 instead of
    3.0 ± 0.2 (`test_eng_mopac.py`); heat of formation, dipole, HOMO/LUMO and COSMO area/volume passed.
- xtb: **fixed 2026-11-05.** xtb writes `"program call"` with raw Windows backslashes, so `xtbout.json`
  was invalid JSON. Dipoles and gaps fell back to the charge-only values. `eng_xtb.read_xtbout` now escapes
  them, and the xtb tests pass on the workstation.
- MOPAC: open. On the workstation, MOPAC 23.2.5 prints `ISOTROPIC AVERAGE ALPHA = 9.09001 A.U.` for the test's water,
  where the reference output has 8.49573: a version difference in TDHF POLAR, not a parsing problem. Record
  the version that produced the reference and make the test version-aware. Until then, don't mix MOPAC
  polarizabilities from different versions in one QSAR table.

### PLAT-1: First run on the MacBook · done 2026-11-06 (the GUI by hand needs-Brandon)
- INSTALL.md's macOS steps on the MacBook: `doctor --bench`, full suite, `pytest -m external` with
  `envs/tools.yml` + `envs/psi4.yml`, GUI by hand (HiDPI picking, trackpad pinch/rotate, dark mode).
  Record chip, cores and RAM in PLATFORMS.md section 2, and anything that breaks as new items.
- 2026-11-06: full suite and `-m external` done over SSH (PLATFORMS.md section 5: 841 passed; the real failures
  were fixed on `fix/macos-checks`, except LAMMPS, which needs `pip install mpich` on the MacBook, and MOPAC's
  polarizability, WIN-2). AIMNet2 on `mps` checked against the CPU. Left for Brandon: the GUI by hand.

### PLAT-2: macOS in CI · done 2026-11-06 (green on all three OSes; MOPAC's polarizability is WIN-2)
- `macos-14` is in the CI matrix as non-blocking (2026-11-04). Make it green, then required. Add a second
  workflow job: micromamba `envs/tools.yml` + `envs/psi4.yml` on ubuntu-latest and macos-14, then
  `pytest -m external`.
- 2026-11-05: `.github/workflows/external-engines.yml` does the second part (weekly, manual, and on pushes
  touching engine code). Left: watch the first runs and make macos-14 required in `ci.yml`.
- 2026-11-05: first green CI on all three OSes (also fixed: Qt xcb libraries on Ubuntu, `Get-FileHash` under
  pwsh-launched Windows PowerShell, long runner hostnames, Psi4 needs `libxc-c<7`). External engines: Ubuntu
  green; on macos-14 two MOPAC checks fail: the PM7 polarizability is 1.347 Å³ for water (1.259 on Linux,
  from MOPAC's own .out) and 3.22 for methanol (3.0 ± 0.2), while heats of formation, dipoles and orbital
  energies match. Compare the osx-arm64 MOPAC build/version and its POLAR output before relaxing the tests.
- 2026-11-06: GitHub stopped running `macos-14` after 2026-11-05 19:15 UTC (jobs cancelled in 2 s,
  "configuration 'macos-14-arm64-us-default' is not supported"); both workflows now use `macos-15`. The GUI
  tests now run on macOS too (they had always been skipped there for want of `$DISPLAY`). MOPAC: the
  MacBook (OpenBLAS) and the workstation (MKL) both print 1.347, the cloud's Linux build (MKL) 1.259, all 23.2.5 build 0,
  so BLAS is not the cause: the Linux build differs from the macOS and Windows builds.

### DIST-1: Machine capability report and weighted chunks · done 2026-11-05
- `doctor --json` gains engines, external programs (versions), and `--bench` throughput per engine
  (molecules/hour for a fixed PFAS set). Write it to `machines/<hostname>.json` in the settings folder
  (synced, so every machine sees the others).
- `build --chunk-weights auto` (and the GUI) splits a list in proportion to those throughputs.
- Acceptance: tests for the report schema and the weighted split (deterministic, stable when the list grows).
- 2026-11-05: `dist.profile` (registry in `<settings>/machines/`, `quantabench-cli machines [--refresh]
  [--bench METHODS]`), `dist.plan` (`--chunk-weights 3,1 | name=3,... | auto`), weights stored with the run.
  Same-machine parallelism too: `dist.lanes` (`build --lanes auto | N | gpu:1x4,cpu:5x2`); the workstation: MMFF 3.7x,
  GFN1-xTB 3.45x with lanes.

### DIST-2: `quantabench-cli worker` over Tailscale · done 2026-11-05 (the workstation + MacBook run needs-Brandon)
- Small HTTP/JSON service (stdlib or one light dependency) bound to the Tailscale interface only, with a
  token from settings: submit, status, cancel, list. Each job runs through `jobs.run_resumable` into the
  worker's own run folder under the synced Runs folder (one writer per run folder). Version/commit check;
  environment fingerprint in `summary.json`.
- Windows firewall rule for the Tailscale interface; `caffeinate -i` on macOS while busy; pause switch.
- Acceptance: two workers on one Linux box in tests (localhost), submit/cancel/resume; a manual
  the workstation + MacBook run (needs-Brandon).
- 2026-11-05: `dist.worker` + `dist.client` (`grid machines|build|status|cancel|merge`), docs/WORKERS.md
  (firewall rule, Task Scheduler and LaunchAgent autostart). Left: the two-machine run over Tailscale.

### DIST-3: Machines dock and "Run on" routing · done 2026-11-16 (the dock; Run on: UI-5b, PR #18)
- Machines dock (status, load, engines). The job dialog's resources get "Run on: this machine / <worker> /
  auto"; auto picks by engine availability and throughput.
- 2026-11-05: minimal Machines dock (registry, worker status on demand, this computer's lane plan). Left:
  the "Run on" choice, which belongs in UI-5's job dialog.
- 2026-11-16: Run on in the job dialog (UI-5b): this computer, Auto, or a worker; see UI-5b.

### MON-1: Live machine metrics while jobs run · done 2026-11-09 (branch `feat/machine-monitor`; the Mac's temperatures need-Brandon: `brew install macmon`)
Brandon (2026-11-09): "during the submit job page, we could monitor other PC metrics like what the current actual
thread speed is and the temperature of the PC."
- Done: `quantabench/sensors.py` (`read()` never raises, about a second at most; `Monitor` samples in a thread),
  per OS: Windows performance counters through ctypes (Task Manager's effective clock, per logical CPU; GPU Engine
  and Adapter Memory) + LibreHardwareMonitor's web server (`/data.json`: CPU Tctl/Tdie and package power, GPU
  temperature, clock, power, VRAM); macOS macmon (a kept-running `macmon pipe`); Linux cpufreq, hwmon, RAPL, amdgpu
  sysfs, nvidia-smi; WSL2 says the sensors are on the Windows side. `quantabench-cli monitor` (`--per-cpu`, `--json`,
  `--worker`, `--set`), a Sensors section in `doctor`, the GUI's machine monitor strip in the Pipeline dock (this
  computer and the workers, sparklines, colours by threshold), `GET /v1/sensors` on workers (and `sensors` in
  `/v1/info`; older peers answer 404 and are left alone), `settings.sensors` with per-machine entries. Run timings:
  job table schema 5 adds the clock, CPU temperature and power, and GPU utilization, temperature and power to the
  60 s load samples; `timings` prints them per run with a likely-throttling flag (docs/ENGINES.md, Live machine
  metrics; UI_PLAN.md, job monitor).
- Checked on the workstation (2026-11-09, from a scratch copy, LHM's web server already on): every metric but the fans
  (LHM reads 0 RPM on all of the ASUS B650E's headers); effective clock 4.9-5.0 GHz at idle against a 4,701 MHz
  base, CPU 52-58 °C, 62-83 W, RX 9060 XT 36 °C, VRAM 2.0/15.9 GB; a read takes 25-50 ms (the first 0.3 s). The
  WMI namespace `root\LibreHardwareMonitor` is absent there, so only the web server is used.
- Left:
  - needs-Brandon: `brew install macmon` on the MacBook (not installed on 2026-11-09; the Mac reports utilization
    and RAM until then), then check the parser against its real JSON (`macmon pipe -s 1`; the fields the parser
    expects are in docs/ENGINES.md) and restart the worker so `/v1/sensors` includes temperatures.
  - Fans on the workstation: the board's fan headers read 0 RPM in LHM; HWiNFO may read them, LHM may need a newer version.
  - With UI-5's job monitor dock: per-job markers on the sparklines, the run's own CPU share next to the machine's,
    a throttling banner during a run (UI_PLAN.md).
  - Use the sensor columns in the CREST cost-model fit (SCHED-1): leave out or down-weight runs flagged as
    throttling or busy.
  - Intel hybrid CPUs (P- and E-cores) and multi-GPU machines: the strip shows the GPU with the most memory only.

### DIST-5: Speed-aware lanes at the end of a batch · done 2026-11-05 (the workstation measurement needs-Brandon)
- `lanes auto` for AIMNet2 on the workstation gives 1 GPU lane + 4 CPU lanes, and the GPU is ~10x faster per molecule.
  Lanes pull molecules first-come, so near the end of a batch a CPU lane can start a molecule the GPU lane
  would have finished long before; for a small batch the run can end up slower than the GPU lane alone.
  Use each lane's measured time per molecule: when the queue is shorter than (slow lane's time / fast lane's
  time), leave the rest to the fast lanes. Measure on the workstation with 8, 50 and 500 PFAS.
- 2026-11-05: `dist.speed` + `run_parallel`. A free lane leaves the next molecule when the lanes >= 1.5x
  faster, simulated taking the queue in order (each molecule to the one that finishes it first, from when
  each is free), would finish all of it before this lane finished that one. Speeds are seconds per atom per
  (device, threads), measured, else the registry benchmark scaled to the run, else unknown (no holding).
  Only ready lanes (or first starts, <= 60 s) are counted on. On by default (`resources.speed_aware_lanes`,
  `build --[no-]speed-aware-lanes`, Settings > Compute). Simulated (tests/test_speed.py, GPU 5 s and CPU 50 s
  per 20-atom molecule, lanes ready after 4-6 s, benchmark-seeded), makespan with / without / GPU lane alone:
  8 PFAS 46 / 54 / 46 s, 50 PFAS 196 / 204 / 256 s, 500 PFAS 1804 / 1804 / 2506 s. With 10-40-atom molecules
  and 10 % scatter it never ended later than either; with the benchmark ratio 10 % off and 20 % scatter it
  lost up to ~25 % against the GPU lane alone on single 8-molecule runs, never on average.
- Left (needs-Brandon): the real measurement on the workstation. In PowerShell, from the repo with this branch, with a
  PFAS CSV (`pfas.csv`, a SMILES column; add `--smiles-col` if needed) and an empty folder for the runs:
  ```
  quantabench-cli machines --refresh --bench aimnet2          # seeds: aimnet2@cpu and aimnet2@cuda:0
  quantabench-cli lanes auto --method aimnet2                 # expect gpu0 + 4 CPU lanes, speed-aware on
  foreach ($n in 8, 50, 500) {
    Get-Content pfas.csv -TotalCount ($n + 1) | Set-Content "pfas_$n.csv"
    foreach ($run in @(@("on", "--lanes", "auto", "--speed-aware-lanes"),
                       @("off", "--lanes", "auto", "--no-speed-aware-lanes"),
                       @("gpu", "--lanes", "gpu:1x4"))) {
      $t = Measure-Command { quantabench-cli build --file "pfas_$n.csv" --method aimnet2 --out dist5 $run[1..($run.Count - 1)] |
                             Tee-Object -FilePath "dist5_${n}_$($run[0]).log" | Out-Host }
      "$n $($run[0]) $([math]::Round($t.TotalSeconds, 1)) s" | Add-Content dist5_times.txt
    }
  }
  Get-Content dist5_times.txt
  Select-String -Path dist5_*.log -Pattern "Speed-aware lanes|Lane speeds|Molecules per lane"
  ```
  Record the 9 wall times here. If `on` is slower than `gpu` at 8 PFAS, compare the `Lane speeds` line
  (measured) with the benchmark ratio `aimnet2@cuda:0` / `aimnet2@cpu` in `quantabench-cli machines`: the
  8-molecule case relies on that ratio, since the CPU lanes never run a molecule there.

### SCALE-1: The GUI at 14,000 molecules · done 2026-11-05 (batch memory: one follow-up)
- From the PFAS QSAR project (2026-11-05): Spartan could not take 14,000 molecules in one document, so the
  work was cut into 200-molecule batches (1,067 `.spartan` files) that now have to be recombined by hand.
  QuantaBench's engine side already runs any number in one run folder (one job table, one `summary.csv` and
  `descriptors.csv`, per-molecule folders, resumable) and, since 2026-11-05, does not keep the ensembles in
  memory. The GUI still keeps every finished molecule's full result in memory and builds one list item per
  molecule.
- To do: a virtualized molecule list (QListView + model), results kept as summary rows with the full result
  loaded from the molecule's folder on selection (small LRU cache), the conformer table and descriptors from
  `summary.csv`/the job table, and "Open run folder..." (also the 0.2 item "resume a run folder from the GUI").
- Acceptance: open a 14,000-molecule run folder in under 5 s and under 500 MB, scroll, search and select
  any molecule; a GUI batch of 14,000 runs with flat memory.
- 2026-11-05: `gui.molecule_list` (table model of light rows, search/status filter, LRU `ResultStore` of 12
  read back with `export.read_molecule`/`jobs.load_result` off the GUI thread), File > Open run folder...,
  Build > Resume run folder (stored settings, `jobs.stored_config`). Measured under xvfb on 2 shared cores
  with a synthetic 14,000-molecule folder (`scratch/make_synthetic_run.py`: real MMFF templates written with
  the engine's own writers): listed in 0.56 s, first molecule shown at 0.98 s, **+31 MB** RSS (+41 MB after
  browsing 60 molecules); search 40 ms, sort 60 ms, reading one result ~5 ms. The window itself is ~1.1 GB
  RSS before any molecule: torch (~500 MB, imported at startup by the GFN2 availability check,
  `engines.libcint_status` -> dxtb) and VTK under software OpenGL. Selecting a molecule takes ~0.4 s
  there, almost all VTK renders (~65 ms each, 3-4 per selection) under llvmpipe.
- Follow-ups: (1) a 400-molecule MMFF batch from the GUI grew RSS by 8 MB (~20-29 KB/molecule) with the
  list keeping rows only; ~5 KB/molecule of that is the light results `run_resumable` still returns.
  Let a GUI batch keep no per-molecule results at all (touches `dist/lanes.py`, after DIST-5). (2) Do not
  import torch at startup just to label GFN2. (3) One VTK render per selection instead of 3-4.
- 2026-11-05 (branch `perf/gui-startup`), same synthetic folder and machine:
  - (2) done: no torch/dxtb/tad-libcint/aimnet/ovito in the GUI process at start-up. Labels use
    `engines.available(key, quick=True)` (versions and files, `libcint_status_from_versions`); GPU list and
    the full libcint check run in a child process (`resources.detect_isolated`) whose verdict relabels GFN2.
    Idle RSS with the 14,000-molecule folder open: **1,133 -> 634 MB** (window alone 1,102 -> 602 MB). VTK
    is ~355 MB of that (libraries ~155 MB, llvmpipe context and buffers ~200 MB); without VTK ~280 MB. So
    the 500 MB target holds only without VTK under software OpenGL; on a GPU the buffers leave RSS. Check
    the number on the workstation and the Mac before deciding on more.
  - (3) done: one render per selection (was 3-9: `Plotter.reset_camera` renders even with `render=False`,
    the selection clear rendered, and pyvistaqt's `auto_update` re-rendered 5x/s while idle). Selection
    median **557 -> 224 ms**; scrolling 14,000 rows in 200 steps 10.0 -> 2.5 s. What is left of a selection
    is one render with SSAA: ~150 ms under llvmpipe, ~40 ms with FXAA (RSS about the same). Possible next
    step: FXAA when the OpenGL renderer is software (llvmpipe, WSL without GPU).
  - (1) attributed: a Log line costs ~7 KB RSS (laid-out text block, shown or not) and a GUI MMFF batch logs
    ~3 lines per molecule (~21 KB/molecule), so the log grew until its 20,000-line cap (~140 MB). Cap now
    5,000 lines (~35 MB). Still open: the light results above, and an MMFF batch imports torch at its start
    (`jobs.run_resumable` and `pipeline.run` call `resources.select_device` before knowing whether the
    method needs torch: +~500 MB for the rest of the session). Select the device lazily, or only for
    torch engines (chem/jobs.py, chem/pipeline.py).
- 2026-11-05 (branch `fix/runtime-guards`), for the 14,223-molecule split across the workstation and the MacBook:
  - `lanes auto` plans a GPU lane only where this machine's benchmark (`<method>@<gpu>` vs `<method>@cpu`)
    says the GPU is the faster lane; MacBook AIMNet2 (6,387/h CPU, 2,978/h MPS): CPU lanes only (6 x 2 with
    the default 13 threads). Without a benchmark the GPU lane as before, and the reason says so. Still to check: a lanes benchmark on the Mac
    (`machines --bench aimnet2` with `--lanes`), since 1 MPS lane + 4 CPU lanes might still beat 6 CPU lanes
    if MPS and the CPU do not contend (the rule compares one lane with one lane).
  - Benchmarks in water (`machines --solvent water`: `<method>+water@<device>`) and `--bench-set mixed`;
    readers of the keys (speed-aware lanes, `lanes auto`, grid and chunk weights) use the run's solvent.
  - Settings > Compute: memory per lane and time limit per molecule.

### IMPORT-2: Read Spartan '24 `.spartan` documents · in review 2026-11-05 (branches `feat/spartan-import`, `feat/spartan-qsar`; acceptance pending)
- A `.spartan` file is an OLE compound document with one storage per molecule (`Molecules/M0001/...`):
  `Input.gz` holds the keywords, the label (CAS number in Brandon's files), charge and multiplicity and the
  Cartesian coordinates; `PropertyArchive.gz` holds about 50 scalar properties (`PROP VALUE NAME = value`:
  energies, dipole, HOMO/LUMO, CPK area/volume, ovality, QSAR ESP areas, polarizability) and per-atom blocks
  (Mulliken/NAO/ESP charges, atomic CPK areas). Checked on the PFAS QSAR files with `olefile` (pure Python).
- To do: `chem/io_spartan.py` + `quantabench-cli import *.spartan --table out.csv` (and the GUI import),
  property names mapped to QuantaBench's columns where they match; one row per molecule and file, keyed by
  label, so all 1,067 batch files become one table.
- Acceptance: one batch's values equal Spartan's own spreadsheet export; the 76-PFAS document imports whole.
- 2026-11-05: `chem/io_spartan.py`, `import *.spartan --table/--sdf/--conformers`, GUI routing. Checked on a
  100-molecule MMFF/DLFF03 conformer-search batch (labels match its input SDF), its extracted-conformers
  document and one small conformer document. Still open: (1) a Spartan spreadsheet export with values (the
  staged `.xlsx` was empty) to compare; (2) the 570 MB 76-PFAS document; (3) a document with the QSAR
  (quantum-chemical) properties, to confirm their Spartan names/units and fill `io_spartan.COLUMN_MAP`
  (HOMO/LUMO, dipole); (4) Spartan's own rule for the atoms its conformer libraries leave out (placed here
  by a rigid fit, typically ~0.5 Å from Spartan's placement).
- 2026-11-05 (`feat/spartan-qsar`): the 129,164 molecules a full run logged as "no Input.gz" (All_Conformers,
  SemiEmp_PM6 and base documents) come from Spartan for macOS, which writes the same streams uncompressed
  (`Input`, `PropertyArchive`); both are read now and anything still unreadable is counted by reason. Rows
  carry the level of theory (`Commands`/`Calculations`) and the job status (`JobLog`); PM6/DFT HOMO/LUMO
  (hartree -> eV), dipole, ZPE, H/G corrections and entropy fill `COLUMN_MAP` (units checked against the
  output, the manual and the archive's own identities). Labels Spartan made up (`M0003`, `M0018[1]`) are
  joined to the base document by structure (`label_source` input/structure/ambiguous/unresolved); `--wide`
  writes one row per label with each level's best result under a level prefix. Staged sets: 2,750 of 2,750
  molecules read; derived rows 2,480 resolved, 120 ambiguous (6 anions of C3A_P4 under two CAS numbers each),
  0 unresolved. Still open: (5) the optimized geometry of Spartan OPT jobs (Input keeps the starting one;
  probably in `Archive.carc`); (6) ambiguous structures (an anion under the CAS numbers of its salts): share
  the result with all its labels in the wide table, or leave it out (now); (7) a full-run timing on the workstation.

### QSAR-1: The PFAS QSAR cohorts in staged batches · all 15 batches run 2026-11-11 to 14; failed molecules rerun 2026-11-17 (FAIL-1); CREST rebuild: CREST-RUN-1
Decided 2026-11-11 (Brandon): no single run over all 14,223 molecules (`data_engineering/Cohort */*.sdf`: C4A 9,052,
C1A 2,492, C5A 930, C1B 409, C4B 376, C3A 341, C2A 331, C2B 193, C3B 80, C6A 19). Batches, each analysed before
the next. Inputs and selections in `Runs/batches/<batch>/` (input.csv, selection.csv, selection.md, design.json);
scripts in `scratch/batches/` on the workstation (not in the repository).
- **Protocol** (`run_batch.ps1`, per batch `run_b01.ps1`, `run_b02.ps1`): scratch/full_run.ps1's settings (RDKit
  ETKDG 50 + MMFF94s, the 5 lowest refined with GFN2-xTB/tblite, 6 kcal/mol window, `--freq`, `--no-protonate`)
  plus `--rank auto`; gas, then ALPB water, then `pair`; `grid build --wait --machines bench-mac,local
  --priority background` (work sharing on). Each run goes to `Runs/<date>_<batch>/` with timing.txt, run.json and
  the grid logs, and ends with the batch's analysis (`analysis/report.md`). The script stops before starting when
  the installed QuantaBench has no `--rank` (exit 3) or a remote worker is unreachable or on another commit (exit
  4); `-NoRank` runs without the rank step (smoke tests), `-First N -Phases gas -Machines local` a smoke test.
- **b01-essential76** (76): every Essential 76 PFAS, as Spartan drew them (its import's canonical SMILES and R/S;
  CompTox's salt SMILES for 2806-15-7, which has no Spartan structure). Charges drawn: 75 neutral, PFOS
  (45298-90-6) as the anion; QuantaBench computes 74 neutral and 2 anions (Na removed from 2806-15-7). At pH 7
  QuantaBench's rules give 27 others charged (`charge_ph7`, a record only; notes where the rules are off:
  phosphonic acids -1/-2, fluorinated beta-diketones are enolates, the amidine is likely neutral). Flags: 13
  without a Spartan ωB97X-D reference (6 iodides, 3 silanes, 2 phosphonic acids, the Na salt, PFOS), TFA and PFPrA
  (Spartan's anti acid), 10 with R/S from Spartan's geometry, 6 that hydrolyze in water.
  `analyze_b01.py`: `validate` against the E76 import and the v6 table, `conformers compare` against Spartan's
  ωB97X-D/6-31G* conformers (gas; the 2026-11-06 common-level cache, so only QuantaBench's conformers are
  computed), the rank step's pick changes, run timings and machine sensors, flagged molecules.
- **b02-st100** (100, 20 holdout): non-fluorinated, neutral, single-component organics (H C N O S P Cl Br I Si,
  <= 40 heavy atoms) with a pure-liquid σ at 20-25 °C from PubChem/HSDB (`pubchem targets --window 20,25`, moved
  to 25 °C), liquid at 1 atm at the record's temperature, within 30 % of the Sugden parachor estimate with a
  measured density and below 65 mN/m. 199 eligible; 16 functional classes by priority, equal quotas (water
  filling), compilation-tier values first, a seeded max-min spread over σ and size (seed 20261111); 65 values from
  compilations, 22 other, 13 CHRIS. Left out: 5 suspect values (water's range, or > 30 % from the Sugden
  estimate) and 5 compounds that are gases at 1 atm (selection.md lists them). `analyze_b02.py`: σ against α, V,
  μ, area, polar area (TPSA: the protocol has no QM surfaces) and ratios; Sugden parachor and volume parachor
  (measured density); the quantum parachor σ = ((a·α + b·μ + c)/V)^4 with and without a polar-area term; 5-fold
  and leave-class-out CV on the training set, the holdout once; parity plots.
- The `gamma_25` column of the data_engineering CRC extraction (`Cohort4_Merged_Extraction.csv`) is not usable as a
  target: a text scrape of the CRC surface-tension pages matched by name prefix, which mixes γ at other
  temperatures and of other compounds, water's σ and molecular weights (not parachors). `Runs/batches/
  gamma25_check.md` has the breakdown; b02 uses QuantaBench's PubChem pull instead.
- **b01 and b02 ran** 2026-11-11 (both machines, `--rank auto`): 76/76 and 100/100 ok in both phases, 4 and 3 min
  per phase. The E76 again with 20 refined conformers (`_rt20`, the workstation alone, 12 lanes): 3.7 min gas, 4.6 min water;
  the same lowest conformer as Spartan 11/63 (10/63 with 5), search misses 18 (37), conformer dipole MAE 1.21 D (1.39).
- **Decided 2026-11-11** (Brandon): "a stratified batch of 1000", batches scheduled for the evenings.
- **The partition** (`b03_select.py`; `Runs/batches/partition/`: records.csv with every record and where it went,
  excluded.csv, partition.md): all 14,223 SD records (C6A_P1.sdf as fixed on 2026-11-11), the structure as drawn
  (canonical isomeric SMILES, R/S and E/Z from the record's 3D geometry, charges as drawn: `--no-protonate`), CAS
  ids (the title in cohorts 1-3, CAS_RN in 4-6). Left out, first reason: more than one fragment 459, elements
  outside AIMNet2's (the gas rank level) 85, unpaired electrons 11, more than 60 heavy atoms 152, refused by
  `standardize` 3, unreadable or no InChIKey 4; the same structure as a kept record 266 (computed once); already in
  b01 86 and b02 100 records (by InChIKey). Eligible: 13,057. Could still run later (a decision, in no tranche): 405
  multi-fragment records as their largest fragment, 78 with elements outside AIMNet2 only, 152 above 60 heavy atoms
  only. Seed 20261111.
  - **b03-strat1000** (1,000): proportional to cohort x charge class x size bin (<= 10, 11-20, 21-35, > 35 heavy
    atoms) with a floor of 20 per cohort x charge class (C2B, C3A and C3B anions and cations, C6A at the floor or
    whole), the rest shared in proportion, then over the size bins; a seeded shuffle per stratum. 41 anions, 44
    cations; 102 above 35 heavy atoms (flag `large_gt35`; the floors of the long cohorts push it above the ~80 of a
    proportional sample), 64 above 60 atoms, 4 above 100 (155 at most); 833 with PM6 and 50 with DFT-water values
    in the v6 table. Input order: size bin ascending, the groups interleaved, so `-First 5` is a cheap mixed smoke
    test.
  - **b04-tranche1 ... b08-tranche5** (2,411-2,412 each): the other 12,057 dealt stratum by stratum (seeded), so
    every tranche has the same mix within one molecule per stratum (C3B 4-5 each; C6A is all in b03).
  - C6A: the v6 table has no C6A rows (no Spartan document), and none of the six records fixed on 2026-11-11 joins
    a v6 row by the old or the new structure, CAS or title; b03 has all six with the fixed structures.
  - 49 records of the PFAS cohorts have no fluorine (C3A cations 32, C1A 8, C2A 8, C3B 1): fixed 2026-11-12, below.
- **Evening runs** (`scratch/batches/run_tranche.ps1` + `run_tranche.py`, `run_b03.ps1`, `start_tranche.ps1`): the
  protocol above; before each attempt the MacBook is asked and joins when its worker answers, is idle and on this
  commit, else the workstation runs alone. A deadline (default 07:00) cancels the remote shares and stops the workstation's at its next
  molecule; `-Resume` (or the same command the same day) runs only the molecules no run folder of the phase has
  finished, as a new grid in the same phase folder. Exit 0 done, 5 stopped (deadline or Ctrl+C, resumable).
  `qb_grid.py` wraps `grid build` for the deadline and the watch below. `analyze_tranche.py` (generic) writes
  `analysis/report.md` and a short `summary.md`: coverage and failures per cohort x charge class (errors grouped),
  molecule and stage times per machine and size bin from the run telemetry, machine sensors and throttling, the
  rank step's pick changes by phase and charge class, `validate` against v6 by cohort, outliers (imaginary
  frequencies, dipoles > 30 D, CPK > 15 % off Spartan or between phases, solvation energies out of range), and
  drift against b01/b02 and the earlier tranches (KS tests within fluorinated/non-F x charge class). Tested on the
  b01, b02 and E76-rt20 runs (rt20 against b01: the descriptors agree, p ~ 1; the time per molecule differs).
- **`grid build --wait` hangs when a worker goes away** (main, 27a73bd): `follow` waits for an unreachable worker
  ("may come back"). Tested with a throwaway local worker frozen mid-share (SIGSTOP, like a sleeping MacBook):
  the local share finished, the frozen share stayed `running` at 0/17, "cannot reach ... (retrying)", no return;
  work sharing cannot help (the frozen worker cannot release its molecules, and it has no rate before 3 are done).
  DIST-7 (`feat/machine-control`) handles it in the client. Until that is on the workstation and the Mac, `qb_grid.py` stops
  following once the workstation has nothing left and the remote worker has not answered for 10 min, copies what the remote
  share finished from its synced run folder (a private copy of its job table), and the next attempt runs the
  rest; the orphaned job is cancelled when its worker answers again. Tested with the throwaway worker (frozen:
  rescue on the workstation; deadline: stop and resume; Ctrl+C: stop and resume), all 30 molecules paired each time.
  Related: `grid merge --partial` and `grid status` look for a remote share's synced copy beside the grid folder
  (`<grid folder>/../<run folder>`), so with `--out <run root>/<phase>` (the batch scripts) they miss the Mac's
  folder in `Runs/` (for DIST-7).
- **Lanes:** the workstation's `auto` (`dist.machine_lanes`) gives 12 CPU lanes x 1 thread for tblite-gfn2 at every priority
  on 27a73bd (not 6 x 2); the Mac runs 10 x 1.
- **Time estimate for b03** (per-stage power laws in atoms from the b01/b02 telemetry, the Mac at 10 lanes and
  refine top 5, and the E76-rt20 telemetry, the workstation at 12 lanes; refine and rank scaled by the conformers refined;
  the 64 molecules above 60 atoms extrapolated): gas + water about 0.6 h on both machines or 1.5-1.7 h on the workstation
  alone with refine top 5; 1.2 h or 3.1-3.3 h with refine top 20. A tranche: 1.2 h or 3.1-3.3 h (top 5), 2.4 h or
  6.4-6.6 h (top 20). The slowest molecule on a workstation lane: 14-27 min (top 5), 23-40 min (top 20, near the 1800 s
  molecule time limit). b03 measures it.
- **Smoke test** 2026-11-11 (the first 5 of b03, gas, the workstation alone, background, `--rank auto`): 5/5 ok in 0.5 min
  (12 lanes; the AIMNet2 load is about 14 s per lane), the report in 0.1 min.
- Open (needs-Brandon): TFA and PFPrA (recompute Spartan's conformers or leave them out, VAL-2); whether b02's σ
  should include values of gases above their boiling point (left out now; DESC-2: Sugden reproduces them within
  0-3 mN/m and adding them moves the CV error by < 0.1 mN/m, so leaving them out costs nothing); b02's holdout
  (looked at twice: freeze the model or draw a new one); whether the excluded salts (405), the elements outside
  AIMNet2 (78) and the molecules above 60 heavy atoms (152) get a batch of their own.
- **b03 ran** 2026-11-11 19:02-19:45 (`Runs/2026-11-11_b03-strat1000_rt20`; both machines on e29f93e, refine top 20,
  rank auto, as drawn, background): gas 998/1000 in 19 min, water 997/1000 in 22 min (the estimate was 1.2 h); the
  Mac did 726 gas and 438 water molecules. Failures: 1423-20-7 (tblite SCF not converged, both phases), 94200-46-1
  (SCF, gas), 447-41-6 and 2447-54-3 (water: the process crashed twice). Against PM6 (v6, 850 molecules): dipole MAE
  0.84 D (r 0.80), CPK volume 2.2 A3 (r 1.00), area 4.6 A2 (r 0.998), HOMO 1.05 eV below (r 0.91); water dipoles
  of 25 ions against B3LYP/C-PCM 5.94 D (origin-dependent). The rank step changed the pick for 44 % (gas, AIMNet2)
  and 43 % (water, g-xTB + CPCM), anions 65-76 %. Outliers: imaginary frequencies 75 per phase (7.5 %; 60 of them
  C4A), dipoles > 30 D 49 (all ions but one), solvation energies out of range 10, best conformer not converged 3
  (C3 anions). The drift flags against b01/b02 come from their different selections. No throttling (CPU peaks 94 °C
  Mac, 90 °C the workstation). Left before the tranches: the imaginary frequencies and the four failures.
- **Structure fixes 2026-11-12** (Brandon: "Draw the PFAS part"; `scratch/batches/pfas_part_fix.py`,
  `apply_structure_fixes.py`, `verify_fixes.py`; `Runs/batches/structure-fixes-2026-11-12/`): the 49 fluorine-free
  PFAS-cohort records were PFAS salts, adducts or blends drawn as the non-fluorinated partner only (a counter-ion,
  an amine, a phosphonite); PubChem's record for each CAS gives the whole compound. In the source SD files (a
  `.bak-2026-11-12` beside each of the 14 changed files, byte-identical to the original; a `Structure_Fix` field per
  record; every other record unchanged): 44 records now carry the PFAS part (the anion of a salt, the PFAS acid's
  anion where the record was drawn as an ion, C3F7I of an adduct; 25628-08-4: PubChem's structure joins the ions into
  an ester, so the nonaflate from its title), plus the three PubChem flagged earlier: 93894-66-7 (PubChem's
  perfluoro-isooctyl structure), 165951-17-7 (GenX with dipropylamine: the GenX acid), 218462-19-2 (HFE-7000 in a
  1:1 blend with trans-1,2-dichloroethylene: drawing unchanged, a note). Left out: four UVCB records with no PubChem
  structure (68187-25-7, 160336-21-0, 65530-63-4, 65530-74-7); the copper complex 29763-53-9 stays excluded. In the
  batches (no re-deal): 41 fixed records are now the same compound as a kept record (PFBS, PFOS, PFOA and GenX
  anions, C3F7I, ...; for a stereocentre PubChem leaves open, the same constitution and protonation) and are computed
  once there; 4 rows are rewritten in place (231301-26-1 b06, 2266-82-2 b05, 93894-66-7 b04, 332350-93-3 b07); the
  tranches now hold 2,410, 2,407, 2,403, 2,408 and 2,406. b03's results for 13 of its molecules describe the old
  drawings (`b03-strat1000/superseded.csv`): the QSAR table takes them from the kept records.
- **Decided 2026-11-12** (Brandon): ids of the baseline cohorts = CAS RN (b02 ran with Spartan labels; its
  selection maps each to its CAS); keep frequencies; the packing-density coefficients stay in the public code; a
  Psi4 pass for ESP polar areas on b02, keeping TPSA too; b02 keeps the 20 % holdout (5-fold CV on the 80), the
  5 gases stay out and the 13 CHRIS values in; the excluded salts, elements outside AIMNet2 and molecules above 60
  heavy atoms get batches of their own later; the imaginary frequencies first, before the tranches.
- **b02 ESP pass** 2026-11-11 (`Runs/2026-11-11_b02-st100_esp`, `scratch/batches/run_b02_esp.ps1`; gas, the b02
  protocol plus `--qm wb97x-d/6-31g* --cubes`, both machines, 31 min; the QM stage's CPSCF polarizability was most
  of it): `analyze_b02_esp.py`, 74 training molecules, 5-fold CV / leave-class-out MAE (mN/m): Sugden (measured ρ)
  1.66; quantum parachor (Bosque α, measured ρ) 1.65 without a polar area, 1.66 with TPSA, 1.63 / 1.63 / 1.68 with
  the ESP areas at 75 / 100 / 125 kJ/mol; with the packing density 2.90, 2.87 (TPSA), 2.83-3.03 (ESP); the
  density-free ln σ regression 3.24 with TPSA, 3.09-3.25 with ESP. A polar area adds nothing measurable for these
  non-fluorinated liquids. TPSA and ESP 75/100/125 correlate at r 0.73/0.77/0.68; 29 of 74 have TPSA 0, 7/20/27
  have ESP area 0. Brandon then asked for the whole ESP distribution instead of three thresholds (2026-11-12:
  `qm_surf_esp_area_*`, `<id>_esp_profile.csv`).
- **QM-ESP options** · done 2026-11-12 (PR #3): `--no-qm-polarizability`, `--qm-properties/--no-qm-properties`,
  `--keep-cubes/--no-keep-cubes`, and the imaginary-mode steps in the Psi4 frequency path. Then the staged ESP pass over b03 (gas, `--qm
  wb97x-d/6-31g* --cubes --no-qm-polarizability --no-keep-cubes`; `scratch/batches/run_b03_esp.ps1` on the workstation) for the ESP PCA substudy,
  appended to a night's chain once this is merged.
- **b03's imaginary frequencies** (`scratch/batches/imag_look.py`, `imag_test.py`, `imag_summary.py`;
  `analysis/imag_test_{gas,water}.csv`): recomputed at the stored geometries, all 150 come back (not noise). Gas:
  55 at -20..-50 cm-1, 12 at -50..-100, 8 below -100 (water 60/11/4). The small ones sit at a residual force
  (median 0.009 eV/Å at the Hessian, the pre-optimization's fmax 0.01): tightening to 0.002 fixes about half.
  The large ones are saddles (planar NH2 on aromatic rings: 1072-67-9 -431, 634-93-5 -281, 360-97-4, 131-14-6,
  134-58-7; an aldehyde and methyl rotors): tightening leaves them, one step along the mode fixes them. Both
  (tighten, then up to 3 steps): 73 of 75 per phase become minima; the energy falls by a median 6.7 meV (gas) /
  3.0 meV (water), more than 1 kcal/mol for 12 / 6 (those moved on to a lower nearby geometry, median largest atom
  shift 0.5 Å); about 2 s per affected molecule. Left: 1121-07-9, 2809-69-0 (gas), 2101-86-2, 3862-12-2 (water),
  -22 to -36 cm-1 (near-free rotors). Now the default of the frequency step (`freq_imag_retries` 3).
- **All batches done** (2026-11-14): the five tranches of about 2,400 were re-dealt on 2026-11-12 into twelve of
  about 1,003 (b04-b15), run one or more a night on both machines, 41 to 51 min each for both phases. The merged
  table has 13,206 molecules; after FAIL-1's rerun (2026-11-17) 13,205 are ok in both phases, and the last one
  (a glycoside, fixed in PR #25) waits for its rerun. Still in no batch: the multi-fragment records, the
  elements outside AIMNet2 and the molecules above 60 heavy atoms.
- Acceptance (unchanged): one merged table for all 14,223 molecules, no manual recombination, and the overlap with
  the Spartan results compared.

### QSAR-2: Follow-ups from the 325-molecule stress sample (2026-11-05) · ready (decisions: needs-Brandon)
Fixed on `fix/stress-sample`: metal complexes refused instead of computed without their metal, Hg/Sn bonds
kept, counter-ions/charge/multiplicity/force field in the notes and `summary.csv`, SDF record errors,
`--id-col` for SD files, placeholder atoms, unphysical xTB polarizabilities, the Warp CUDA-driver line.
Still open:
- **Protonation rules on polyamines and phosphonates** (chemistry): every aliphatic amine is protonated
  independently, so piperazines and 1,4-dimethylpiperazine come out +2 at pH 7 (pKa2 about 5.3 and 3.8:
  should be +1); every P-OH is deprotonated with the 1st-pKa rule (alkylphosphonic acids pKa2 about 7.5).
  Needs a site-interaction rule (no second protonation within 2-3 bonds of a charged site) or dimorphite-dl
  as default; 77 of 325 records change charge at pH 7, most of them correctly.
- **Al3+/Zn2+ salts** (needs-Brandon): monoatomic Al3+ or Zn2+ next to organic anions (fosetyl-Al) now
  fails as a metal complex. If Brandon treats them as counter-ions like Na+, add them to
  `standardize.COUNTERION_METALS` (one line; the anion is then computed).
- **Metal complexes that need a 3D builder**: Cr(III) azo dyes have no UFF parameters (method `mmff`
  fails; xTB methods refine unoptimized ETKDG geometries). Cobalamins, carbonyls and Rh/PPh3 complexes
  drawn as ions fail by design. A route: GFN-FF pre-optimization (xtb `--gfnff`) instead of UFF, or a
  coordination-geometry builder.
- **Per-molecule time limit**: done on `fix/memory-time-guards` (2026-11-05). `resources.molecule_timeout_s` /
  `build --molecule-timeout`, default 1800 s. Lanes are killed and restarted past it. One process stops the
  molecule at its next checkpoint (best effort). The molecule is failed `time limit (N s) in <stage>`, and
  `--retry-failed` redoes it. Still open: RDKit embedding has no checkpoint (ETKDG `EmbedParameters.timeout`
  since RDKit 2024.03 would bound it in one process too).
  Done on `fix/runtime-guards` (2026-11-05): the embedding gets `conformers.EMBED_TIME_FRACTION` (1/4) of the
  limit, in one process and in lanes. RDKit's timeout ends a call with a single conformer however many it had
  finished, so one conformer is timed first and fewer are asked for when they would not fit; the timeout is
  the hard stop. Notes say when it ran out and with how many conformers. Still open: the MMFF optimization of
  hundreds of conformers of such a molecule has no checkpoint either; the pipeline could pass
  `timeout_s=confs.embed_time_limit(cfg.molecule_timeout_s)` explicitly (today: `resources.set_molecule_limits`).
- **Ids of the baseline cohorts** (needs-Brandon): cohorts 1-3 are titled by CAS number, cohorts 4-6 by a
  name that is sometimes cut off after a word, with the CAS in `CAS_RN`. `--id-col CAS_RN`
  now gives CAS ids for those; decide which the merged QSAR table uses.
- **Transition-metal spin states**: an odd-electron complex without drawn radicals is computed as a
  doublet (Cr(III) is d3, usually a quartet), with a note. Tables need optional `charge`/`multiplicity`
  columns (MoleculeInput has the fields; the CSV/Excel readers do not read them) so such records can be
  given their spin state. dxtb on that complex from unoptimized ETKDG geometries (no UFF parameters)
  ran for many minutes.
- **Broken records with radicals**: an amine-borane record is stored with a bare B (2 unpaired electrons,
  +1); it is now computed as a triplet cation with a "check the record" note. A heuristic for
  hydrogen-deficient B/N/C could fail such records instead.
- **Memory of dxtb on large molecules** (highest priority for 14,000 molecules with lanes): four processes
  were killed by the memory cgroup at 2-3.9 GB resident: the MMFF run at a 90-heavy-atom cyclic peptide and
  a +2 alkaloid in the GFN1-xTB properties step (dxtb charges/polarizability, autograd field response),
  the gfn1-xtb run on a Cr(III) azo complex, and the tblite run in that properties step. The job table recovered (crash counted, molecule
  moved last), but on the workstation with many lanes this can kill lanes repeatedly. Measure memory vs atom count and
  skip (with a note) or chunk the polarizability above a size; or use tblite's finite-field polarizability.
  Done on `fix/memory-time-guards` (2026-11-05): `chem/memory_guard.py`. The cause is dxtb's `scf_mode="full"`
  (memory goes as SCF iterations x nao²). An estimate per step is checked against a per-lane budget
  (`--memory-budget`, default available / lanes, at least 1.5 GB). Over budget, the polarizability comes from
  tblite (the same value), or is skipped with a note; a refinement that does not fit fails with the reason, or
  `auto` moves on to its next engine. Measurements are in docs/ENGINES.md. Still open: the rule under-estimates
  hard SCF cases it cannot see (closed shell with near-degenerate levels: 5.8x the neutral run for a peptide
  with two electrons removed); calibrate on the workstation's ROCm torch and on the MacBook, whose allocators may differ.
  Done on `fix/runtime-guards` (2026-11-05): the frequencies stage (`memory_guard.frequencies_plan`, one call at
  the start of `vibrations.analyze`'s Hessian step). The analytic Hessian grows as atoms x nao² (TFA anion
  353 MB, PFBA anion 1.2 GB, PFOA anion > 6 GB); over the budget it becomes finite differences (one energy +
  forces of memory, about the same time), and finite differences or IR dipole derivatives that do not fit are
  skipped with a note. Measured only up to 2 GB resident here (other agents on an 8 GB VM); the GFN2 and the
  larger points of the rule are extrapolated: check a few on the workstation and the Mac. The pipeline could pass
  `memory_budget_gb=memory_budget(cfg)` to `vibrations.analyze` explicitly (today: the lane's share through
  `resources.set_molecule_limits`, auto in one process).
- **The xTB properties step on dications** (a piperazine, +2 at pH 7) took minutes of GFN1-xTB SCF in the
  MMFF run; measure and cap its SCF iterations.
- Settings isolation: during this work `~/.config/QuantaBench/settings.json` (and `QuantaBench.conf`) were
  rewritten with test-like values (pH 0, 1 conformer, 1 K) by another process on the machine; check
  that no test or GUI path writes there when `QUANTABENCH_SETTINGS_DIR` is set.

### QSAR-3: Surface tension model on QuantaBench descriptors · first models 2026-11-13; fluorinated σ data ready (needs-Brandon: the next target)
Plan: docs/SURFACE_TENSION.md. Two targets, never pooled: pure-liquid σ at 25 °C for compounds that are liquid
there, and γ_CMC/CMC of PFAS surfactants in water.
- Data: `build --phases gas,water` on the QSAR cohorts, then on the workstation (PubChem is blocked in the cloud
  environment) `pubchem enrich RUN --validate --experimental --records experimental.csv`. Add Jasper (1972)
  coefficients (via the MIT `chemicals` package) and NIST ThermoML (free bulk download) for pure liquids, and
  curated literature γ(c) data for PFAS (Brusseau 2019 Table S1 sources), each with the same provenance columns
  as the PubChem records.
  - Done 2026-11-11: the PubChem pull is in `Runs/pubchem-2026-11-05`; `pubchem targets` makes the σ(25 °C) table
    (step 3 of the data assembly). Its first use is batch b02 of QSAR-1 (100 non-fluorinated organics, 20
    holdout), whose analysis script fits the parachor baselines and the quantum parachor.
- Baselines: group-contribution parachor and a parachor fitted on QuantaBench volumes, σ = (Pρ/M)⁴; TEST
  predictions from CompTox.
- QSPR: α, α/V, vdW volume, SASA, μ, μ²/V, ESP (GIPF) descriptors, polar surface, F/C and fluorinated surface
  fraction, dG_solv. Ridge/PLS first, then GPR/boosting.
- Validation: leave-cohort-out, leave-class-out, leave-chain-length-out; applicability domain by leverage,
  kNN distance and F/C range; flag compounds solid at 25 °C.
- Acceptance: a training table with provenance per value, and a model report (MAE/RMSE vs the baselines per
  split).
- Needs Brandon: the target choice (pure-liquid σ vs γ_CMC first), 20 or 25 °C, and access to DIPPR/TRC for
  validation.
- Done 2026-11-11 (DESC-2, below): b02's analysis redone with estimated densities and better polarizabilities
  (`Runs/2026-11-11_b02-st100/analysis/report_v2.md`), and the target tiers compared: train on tiers 1 + 2 after
  the screens, keep CHRIS as a check set, keep the 20 → 25 °C correction (docs/SURFACE_TENSION.md, "Which surface
  tension values to train on"). Next: Jasper's coefficients (the `chemicals` package) as a second, independent
  value per molecule; ThermoML.
- 2026-11-13, on all 15 batches (377 non-fluorinated training liquids, 5-fold CV, MAE in mN/m): ridge on 26
  descriptors 2.03; with the dimensionless groups and the reduced target σ* (DESC-3) 1.81, and 2.40 leaving a
  functional class out. The learning curve is flat from about 200 molecules. The DFT ESP descriptors of b03
  (2026-11-15) did not improve it.
- **The gap is fluorinated data:** 9 fluorinated liquids have a usable σ, none with more than three fluorine
  atoms, and most fluorinated molecules of the list lie outside the training set's descriptor domain. Next: an
  importer for Jasper and ThermoML values of perfluoroalkanes, fluorotelomer alcohols, hydrofluoroethers and
  short perfluoro acids, then the leave-chain-length-out validation above. The holdout is still unused.

### QSAR-4: Property models in the Analyze table (Model...) · done 2026-11-14 (PR #10)
Brandon (2026-11-13 evening): "Could you continue further work on the UI?" The DESC-3 follow-up: the σ model of the
QSAR-3 scripts (dimensionless groups, the reduced target σ*, k-fold and leave-class-out CV) inside the table.
Done in the branch:
- `analysis/models.py`: ridge regression in NumPy (standardized features, α by efficient leave-one-out, as
  RidgeCV), elastic net / random forest / histogram gradient boosting through scikit-learn (the new `qsar` extra);
  median imputation inside each fold; k-fold out-of-fold predictions (seeded), leave one group out, a holdout
  column; the reduced target y/scale multiplied back; `write` (CSV for the join, JSON report).
- `gui/model_dialog.py`: Table > Model... (non-modal): target guess (a joined or imported σ), feature presets
  (Liquid groups, α μ V), σ*, folds, seed, group and holdout columns; the fit on a worker thread; Add to table
  writes `models/<name>.csv/.json` in the project (or the output folder) and joins `<name>_pred`, `<name>_oof`;
  Plot 1:1. `project.MODELS_DIR`.
- Tests: `tests/test_models.py` (ridge against the closed form, the scores, σ*, errors, scikit-learn models when
  installed, the files); `tests/test_gui_table.py::test_model_dialog_fits_joins_and_plots`. Screenshot:
  `docs/dev/screenshots/QSAR-4/`.
Follow-ups (ready):
- Applicability domain in the dialog: leverage and the kNN distance to the training rows as two more joined
  columns, and a flag beyond the 95th percentile (the PCA check of 2026-11-13 found most fluorinated molecules
  outside the non-fluorinated training domain).
- Gaussian-process regression (scikit-learn) with its predictive σ as an uncertainty column.
- Apply a saved model (its JSON) to another project's table: the JSON needs the coefficients and the
  standardization (ridge), or a pickled estimator (the others).

### DESC-3: Feature builder (dimensionless groups, joined targets, parity plot) · done 2026-11-13 (PR #8)
Brandon (2026-11-13): "Maybe certain combinations of those three terms (polarizability, dipole, and molecular
volume) or ratios would be useful for predictions? Maybe we could do some unit dimensionalization?" and "integrating
a feature building into QuantaBench would be a great idea".
Done in the branch:
- `chem/liquid.py`: the Lorentz-Lorenz function with Bosque's α and the refractive index it implies, the dipole
  orientation ratio, packing fraction, sphericity, the Lifshitz Hamaker constant and σ, and kT/v_m^⅔, the unit of the
  reduced surface tension. Written in every phase by the descriptors step.
- The Analyze table: **Features** (the same groups as formula columns for older runs, ratios of α, μ, V and A, the
  reduced target σ* of any column; `analysis/features.py`), **Join...** (columns of a CSV by id, kept in the
  table's state), and the Plot dock's **1:1** parity view (MAE, RMSE, bias, R² about y = x).
- The method guide entry `liquid-groups`, docs/SURFACE_TENSION.md "Dimensionless groups and the reduced target".
- Tests: `tests/test_liquid.py`, `tests/test_gui_table.py::test_feature_builder_join_and_parity_persist`; screenshots in
  `docs/dev/screenshots/DESC-3/` (`scratch/shot_desc3.py`, not committed).
Follow-ups (ready):
- A Model dock: ridge / elastic net / random forest with k-fold and leave-class-out CV, the reduced-target switch,
  coefficients and the parity plot, on the table's columns: QSAR-4 (below).
- Join on an imported field (CAS, InChIKey) as well as the id; a join by InChIKey would match across id schemes.
- An Onsager static dielectric in place of Debye's for polar liquids (the zero-frequency Hamaker term).

### DESC-2: Density and polarizability estimates · done 2026-11-11 (branch `feat/density-polarizability`)
Why: b02's quantum parachor failed because V_vdW is not the molar volume and the minimal-basis xTB polarizability
is half of DFT's. `chem/estimates.py` adds, for every molecule and phase: Girolami and packing-coefficient densities
and molar volumes, Bosque-Sales, Crippen and D4 polarizabilities, Sugden's parachor; the properties step adds
GFN2-xTB's polarizability where the refinement engine gave none. Validation in docs/VALIDATION.md (DESC-2); the
options in docs/SURFACE_TENSION.md. Scripts in `scratch/desc2/` and `scratch/batches/analyze_b02_v2.py` on the workstation.
Follow-ups (ready):
- **GCVOL** (Elbro 1991, Ihmels & Gmehling 2003): type in and check the group tables, test the CF2/CF3 groups on the
  fluorinated liquids of `Runs/desc2-2026-11-11/liquids.csv`, and compare with the packing model.
- **Miller's ahp table** (J. Am. Chem. Soc. 1990, 112, 8533): not checked in this session; add it only if it beats
  Bosque-Sales and Crippen on the DFT set.
- **D4 for PFAS:** 11 % above DFT for perfluorinated molecules (1.01 for the others); a per-element rescale fitted on
  the DFT set, or leave D4 for elements the additive tables lack.
- **More fluorinated densities:** the packing model saw 37 fluorinated liquids (29 in training). The ITRC table's
  densities (66 PFAS, with temperatures), CRC values for perfluoroalkanes and FTOHs, and Jasper's density column
  would test it on PFAS properly; refit with them if they move the F coefficient.
- **COSMO volume:** parse MOPAC's COSMO cavity volume and area into descriptors (a second volume for the packing fit).
- **MD densities** (OpenMM + OpenFF Sage/GAFF2): for a check set of PFAS liquids, with the Simulate track.
- **A DFT polarizability subset of each batch:** `--qm wb97x-d/def2-svpd` on about 50 molecules per batch gives the
  reference α at about 1-5 min per molecule (8 threads), anions included (diffuse basis; finite field in PCM).

### PUBCHEM-1: PubChem follow-ups · ready
- **`build --pubchem-validate`** (not done: it would touch the pipeline's summary row, which other branches are
  changing). Add `pubchem_cid`/`pubchem_match` to each molecule's summary row when online, never failing a
  molecule when PubChem is unreachable. Today: `pubchem enrich RUN --validate` after the build gives the same
  columns.
- **Annotation index:** `pug_view/annotations/heading/Surface%20Tension/JSON?page=N` lists every compound with a
  surface-tension annotation in a few requests (paged). Use it to fetch PUG View only for CIDs that have one,
  and to find surface-tension data for compounds outside the input list. Capture one page as a fixture first:
  the format is not covered by the current fixtures.
- **Refetch policy:** 404s are cached forever; add `--refresh-missing DAYS` for records PubChem adds later.
- **Parent CIDs:** `/compound/cid/X/cids/JSON?cids_type=parent` would tell a salt's CAS RN from its parent's
  without RDKit's largest-fragment rule.
- **Condition classes:** check the classifier on the full pull. List every `needs_review` surface-tension
  record, and extend the reference rules (pure-liquid tables vs surfactant literature) from what is there.
- **Trial findings (validation300, first 50, 2026-11-05):** 40 match, 1 stereo, 6 CAS not in PubChem (found
  by InChIKey), 3 different. Of the 3:
  - 93894-66-7: drawn with one F per chain carbon. The name says heptadecafluoro, so it is a structure error
    in the source.
  - 165951-17-7: PubChem's record is the dipropylammonium salt of HFPO-DA; the table has only the amine.
  - 218462-19-2: a mixture with dichloroethylene.

  The last two are now `component_of_mixture`. Feed `different_structure` and smaller-component cases back
  to the cohort curation before the QSAR fit.
- **Full pull (finished 2026-11-06 03:55, `Runs/pubchem-2026-11-05`, 4.7 h, 28,290 requests):** 14,921
  molecules: 14,044 found, 705 ambiguous, 106 not found, 66 errors; looked up by InChIKey 10,834, CAS RN 3,759,
  SMILES 185, name 77. Structure check (molecules with a CAS RN): 2,373 match, 836 stereo differs, 436 different
  structure, 123 CAS RN not in PubChem, 72 ambiguous CAS RN, 1 charge or protonation, 3 errors; 11,077 have no
  CAS RN. Experimental: 72,635 records for 6,116 molecules; surface tension for 369 (299 pure liquid, 8
  aqueous, 68 unclassified, to review).
  - Next: a list of the 436 `different_structure` molecules by cohort for Brandon (curation decision), the 68
    unclassified surface tensions, and a look at whether the 836 stereo differences matter (most PFAS
    descriptors here are stereo-blind).
- **Parser slips seen while building the b02 targets (2026-11-11; `pubchem targets` repairs or leaves out the
  surface-tension ones, the parser itself is unchanged):**
  - "X dynes/cm = Y N/m" and "X dynes/cm; Y Newtons/meter" strings read as Y with the dyn/cm unit (0.025 for 25);
    HSDB's CRC values printed as "N/m" for mN/m (x1000 too high); a few plain typos (322.4 for 32.24).
  - Density: CHRIS strings that list several quantities ("Liquid heat capacity = 0.520 BTU/lb-F ...; Saturated
    vapor density ...") give a "density" from the first number; "Relative density (water = 1): 0.99" is read as
    1.0; a refractive index (1.515) filed under Density. `pubchem_density_g_cm3` in the enriched table is
    therefore unreliable for those; b02 takes the median of plain density records instead (scratch).
  - Fix in `chem.pubchem`: pair each number with its own unit, skip multi-quantity CHRIS strings for Density, and
    read "Relative density (water = 1): x" as x.
- **CompTox (CTX API):** the same enrichment by DTXSID (batch POSTs of 1,000; free key): TEST's predicted
  surface tension and the experimental physchem values. Check the property names on the workstation first
  (api-ccte.epa.gov is blocked in the cloud environment).

### DIST-7: Machine control: split, budget, pause, a closing laptop · done 2026-11-11 (branch `feat/machine-control`; LaunchAgent on the Mac needs-Brandon)
Brandon (2026-11-11): "Can we make it a feature about how we balance the machine split and also how many processors /
much processing is used? ... I'm often opening and closing my laptop throughout the day. Is there a way we include a
pause button on a machine and the work is redistributed?"
- Done (docs/WORKERS.md, "Pausing a machine, a closing laptop"; `dist/control.py`, `jobs.BatchControl`,
  `worker.JobManager`, `client.MachineWatch`):
  - **Split:** per machine Auto (measured) / Manual / Off in the dock and `machines set --split`; `grid build
    --weights auto|measured|equal|benchmark|NAME=W`, `--exclude`; learned molecules/hour per protocol in
    `grid.json` `rates` (the DIST-6 "learned weights"); per-machine counts at the end of a grid and in `timings`.
  - **Budget:** priority, free threads, max CPU %, lanes, pause on battery per machine from the dock or
    `machines set`; the priority applied live to running lanes.
  - **Pause:** drain / now, persisted, CLI, HTTP, dock, this machine's GUI batch and local share; a grid moves the
    paused machine's unstarted molecules.
  - **Away and back:** 2 min away, 10 min for the running ones, `--wait` never hangs; sync on return, wake
    detection (clock jump, missed polls), the synced `grid.json` fallback; first-done merge and deduplicated timings.
  - **Settings:** three-way merge on save, no-op saves skipped, stamps and SyncThing conflict recovery; workers write
    their state only to `pause.json` and `machines/status/<name>.json`.
- Tested: tests/test_machine_control.py (batch hold/drain/now, sequential and lanes; worker pause, restart, sync,
  wake and lost-contact holds, battery; MachineWatch; end to end: a paused worker and a worker that goes dark mid-run,
  with duplicates merged once and counted once in the timings; weights; settings merge, conflict recovery; the CLI;
  live priority), tests/test_gui_machines.py (the dock). On the workstation (2026-11-11, scratch settings, two throwaway
  workers on localhost, 40 MMFF molecules): pausing tower-t1 after 4 molecules moved its 17 unstarted molecules to
  tower-t2 (40/40 merged in 57 s); freezing tower-t1's process for 55 s (what a sleep looks like from outside) made it
  away after 20 s (the test's setting), its 20 unfinished molecules went to tower-t2, it noticed the gap when thawed,
  the client told it 20 had moved (17 dropped there, 3 it had finished), and the merge kept each of the 3 duplicates
  once (40/40, 76 s).
- Left:
  - The Mac's LaunchAgent (docs/WORKERS.md, Setup on the MacBook, step 3) replaced the `nohup` start on 2026-11-12
    (Brandon asked for it): a restart or logout no longer ends the worker (found gone on 2026-11-09: a clean stop at
    20:10, the Mac rebooted at 20:14, nothing started it again). Left: close the lid during a grid and check the
    away/back log lines on both sides.
  - A real lid close: the freeze test stands in for it; check that macOS's monotonic clock stops during sleep on the
    M3 (the watcher works either way) and that Tailscale is back before the first poll.
  - A running lane pool is not resized when its lanes or free threads change; they apply with the next job. A
    "restart lanes now" (drain, then requeue the job) is a small step on top of the pause.
  - Raising a run priority live works on Windows only (POSIX cannot lower a nice value without privileges).
  - A machine named in `--machines` that is already asleep when `grid build` starts is still an error ("not
    reachable"); leaving it out with a note (as if it were away from the start) would be friendlier. Without
    `--machines` only the reachable workers are used, as before.
  - With UI-5: "Run on" picks machines from the dock's rows; a tray icon with Pause / Resume on the laptop.

### DIST-6: Grid weights from measured throughput · done 2026-11-06 (work sharing; learned weights optional)
- 2026-11-06: **work sharing** instead of learned weights: while `grid build` follows a grid, a machine that
  has finished takes molecules another has not started (`client.WorkSharing`, `POST /v1/jobs/<id>/release`,
  `jobs.Handoff`; docs/WORKERS.md), and every share runs its largest molecules first. In the 2026-11-05 water
  run the MacBook was idle for 8 of 21 minutes; with sharing that tail is about one molecule per lane. Left
  (optional): record molecules/hour per protocol from finished grids and use it for `--weights auto`, so
  fewer molecules move; measure the first real grid with sharing on the workstation + MacBook.
- Validation run (tblite GFN2, 50 conformers, top 5, frequencies), water phase: the MacBook finished its 169
  molecules in 12.5 min (811/h); the workstation took 20.8 min for its 131 (378/h). `--weights auto` had split them
  2680:2240 from the single-molecule benchmark. For the full run pass measured weights (about 2:1), and
  have `grid build` learn: record each machine's molecules/hour per protocol from finished grids, and
  prefer that over the benchmark in `--weights auto`. Work stealing (a machine that finishes early takes
  the other's pending molecules) would make this unnecessary.
- Lane layouts: both machines ran 6 lanes x 2 threads. Timed on 49 molecules (`scratch/lanes_test` on
  each machine, 2026-11-05):
  - Mac 6x2 181 s, 14x1 141 s.
  - The workstation 6x2 427 s, 12x1 363 s, 24x1 331 s, 12x2 318 s.

  `lanes auto` now gives tblite one lane per physical core with the threads shared out (c0427d9). On the same
  49 molecules the Mac is 2.3x the workstation, so pass `--weights bench-mac=2.3,local=1` until grids
  learn their weights.

### UPD-1: Update button gaps · in review 2026-11-06 (branch `feat/update-packages`; worker updates: design only)
Brandon asked whether Help > Check for updates pulls "the latest packages and GitHub version". It pulled
the code, but three things never arrived through it.

1. **Installed packages were never upgraded.** pip reinstalled only when `pyproject.toml` changed, and kept
   every package that still fit. Done: `quantabench-cli update --upgrade-packages [--upgrade-strategy
   eager|only-if-needed]` and, in the app, "Also upgrade the installed packages" or "Upgrade packages...".
   - It works with or without new commits, keeps the installed PyTorch with a constraints file and keeps the
     fork packages on pip's command line.
   - It lists `pip list` before/after, and the app asks before restarting.
2. **Windows fork pins lived only in `install_windows.ps1`**, so a new tblite build (win2, PERF-2) could
   never arrive through Update. Done: `scripts/fork-packages.json`.
   - Per platform: `packages`, `only_binary`, `find_links` as {package: URL}. All three installers read it,
     with the same pip lines and stamps as before (checked by dry runs, the stamp files, and the workstation's
     PowerShell 5.1 parser).
   - `update` reinstalls when this platform's entry changes in a pull. It then force-reinstalls
     (`--no-deps`) a git-tag or release-page package whose requirement or link changed.
   - `--upgrade-packages` force-reinstalls those packages too.
   - Tests: tests/test_update.py, tests/test_install_scripts.py.
3. **Workers keep the old code.** A running `quantabench-cli worker` imported the code it started with. An
   update on that machine changes the files under it: new modules can load next to old ones in the middle
   of a job. A grid then mixes commits; `grid build` already warns about that. Not changed yet:
   docs/WORKERS.md says to restart workers after an update. Options, roughly in order of preference:
   - **a. `quantabench-cli worker --update-when-idle` / `POST /v1/update`** (token, Tailscale bind as
     today). The worker pauses (like `POST /v1/pause`) and lets the running job finish. It runs
     `update.update()` (fast-forward only; `upgrade_packages` only if the request asks), then restarts
     itself. POSIX re-execs with the same argv. Windows starts a new detached process and exits, because the
     Task Scheduler restart is unverified (WORKERS.md). The queue resumes, since jobs are resumable already.
     It refuses while a job runs unless asked to wait, and reports the new commit in `/v1/info`.
   - **b. `quantabench-cli grid update [--machines ...] [--upgrade-packages]`** calls (a) on each worker
     and waits until each reports `origin/main`'s commit. `grid build` could offer it when commits differ,
     instead of only warning, and the Machines dock could show "N workers run another commit".
   - **c. Minimal, no API:** `quantabench-cli worker restart` for the local machine. It sends SIGINT
     (worker already stops after the current molecule), waits for the port to free, and starts the worker
     again with the arguments saved in its state folder. launchd's KeepAlive already restarts a stopped
     worker on the Mac.
   - Either way, `update.update()` should first check for a running worker from the same checkout and
     venv (via `psutil`, or a pid file in the worker's state folder) and say so. Upgrading packages under a
     running worker on Windows also hits the locked-file problem below.
- Left:
  - **Windows in-app upgrades** run pip while the GUI has PySide6, numpy, VTK and the rest loaded. pip
    usually manages by moving the files aside, leaving `~...` folders in site-packages, but it can fail on a
    locked DLL. The message then says to close the app and use the CLI. A sturdier route is to finish the
    upgrade in a helper process after the GUI exits: a console window shows pip, then the GUI starts again.
    Try `--upgrade-packages` from the app on the workstation first (needs-Brandon), and clean up `~...` leftovers on
    the next start if they appear.
  - `--check --upgrade-packages` could preview the upgrade with `pip install --dry-run --report -`.
  - The worker update (3).

### PERF-2: tblite Windows wheel with OpenBLAS · in progress 2026-11-06 (wheel built and verified; release needs a runner)
- 2026-11-06: built in CI on QuantaBench's branch `ci/tblite-wheels` (run 42181668274; Actions refuses to run
  on the fork, HTTP 422 "Actions has been disabled", though the permissions API says enabled). Fixes: dftd4,
  multicharge and ddX need their own `-D<sub>:lapack=openblas` (the top-level `python/` project has no lapack
  option to yield to). MSYS2's `-omp` is libgomp in UCRT64, so there is one OpenMP runtime; the workflow fails
  if libopenblas.dll ever imports LLVM's libomp. Fork branch `qb/windows-wheels-openblas` (c6887e8) matches.
- vs win1: 72 calculations (12 molecules; GFN2/GFN1; gas, ALPB, GBSA) agree to 1.4e-13 Eh, gradients 3.7e-15.
  Per energy+gradient at 31/62/110 atoms: 1 thread 1.41/1.70/2.20x faster; 12 processes x 1 thread
  1.43/1.63/1.88x. With OpenBLAS, 2 threads per process are slower than 1: keep tblite lanes at 1 thread.
  Validation protocol, 24 molecules, 12 lanes: gas 241 -> 150 s (1.61x), water ~239 -> 148 s; results
  identical but one conformer minimum (largest molecule, optimizer path).
- Left: a release with 3.11-3.13 wheels (enable Actions on the fork, or add the matrix and a release job on
  the CI branch), then the rollout below.
- One GFN2/ALPB energy+gradient (`scratch/tblite_scf_bench.py`, 1 thread): MacBook 13.5 / 32.5 / 81 ms
  for 31 / 62 / 110 atoms, the workstation 24.8 / 66.8 / 212 ms. The gap grows with size (1.8x to 2.6x). The macOS
  wheel links OpenBLAS; the fork's Windows wheel bundles reference BLAS/LAPACK.
- To rebuild with MSYS2's `mingw-w64-ucrt-x86_64-openblas` (`-Dtblite:lapack=openblas`), check first:
  - That package depends on LLVM's OpenMP (`-omp`), while tblite links libgomp, so two OpenMP runtimes
    would sit in one process.
  - It ships `openblas.pc` but no `lapack.pc`, which dftd4 and multicharge may look for.
  - Energies must match win1 to 1e-10 Eh.
- Expected gain: the workstation 1.5-2x on mid-size and large molecules, about 15 % off a two-machine full run.
- Rolling it out (UPD-1): point `windows.find_links.tblite` in `scripts/fork-packages.json` at the
  `qb-v0.8.0-win2` release, and change `chem/eng_tblite.py` `WINDOWS_WHEELS` to match
  (`tests/test_install_scripts.py` checks both). Help > Check for updates then reinstalls tblite on the workstation,
  even under the same version number. Give the wheel a new version (e.g. `0.8.0.post1`, pinned as
  `tblite==0.8.0.post1`) so rerunning the installer picks it up too. Its pip run does not replace a
  version that still fits.

### DIST-4: Per-stage routing · later
- Different stages of one molecule on different machines (CREST + Psi4 on the Mac, GPU refinement +
  LAMMPS on the workstation), handed over through run-folder artifacts.
- 2026-11-06: CREST on the Mac needs the fork build (FORK-5); conda-forge's fails from 19 atoms. The machines'
  CREST paths can differ now (`engines.machine_paths`), which routing will rely on.
- Shared-settings race, fixed for the per-machine tables (2026-11-06): the GUI saves the settings it loaded at
  start on every Run and on closing, so a GUI open on one machine wrote back its old copy of the other machines'
  `engines.machine_paths` and `dist.machine_lanes` entries. `settings.save` now re-reads the file and writes only
  the entries this process changed. Still open: every other setting is last-writer-wins, and a QuantaBench
  without `machine_paths` (0.3.0.dev3 and older) drops that table when it saves.

### FORK-1: dxtb with tad-libcint 0.4 (GFN2 on Windows and macOS) · done 2026-11-05 (upstream PR pending Brandon's OK)
- tad-libcint 0.4.0 (2026-10-20) has Windows and macOS-arm64 wheels but needs tad-mctc ≥ 0.11; dxtb 0.4
  pins 0.7. First: open an issue/PR at dxtb. If it stalls: fork `brandonbtm29/dxtb` from v0.4.0 (branch
  `qb/tad-mctc-0.11`), adapt to the tad-mctc API changes, run dxtb's tests on all three OSes, and document
  `pip install "dxtb @ git+https://github.com/brandonbtm29/dxtb@<tag>"`. Supersedes WIN-1 when done.
- 2026-11-05: fork `brandonbtm29/dxtb`, branch `qb/tad-mctc-0.11`, tag `qb-v0.4.0.post1`; its CI is green on
  Ubuntu, Windows and macOS. Matches PyPI's stack to ~1e-8 relative. Install line in INSTALL.md;
  `engines.libcint_status()` names it when tad-libcint is missing or incompatible.

### FORK-2: tblite Windows wheels · done 2026-11-05 (upstream PR pending Brandon's OK)
- Upstream PR adding Windows wheels (cibuildwheel, MinGW-w64 gfortran); interim builds in
  `brandonbtm29/quantabench-wheels` attached to GitHub releases (`pip install tblite --find-links ...`).
- 2026-11-05: built in the fork itself instead (`brandonbtm29/tblite`, branch `qb/windows-wheels`, release
  `qb-v0.8.0-win1`, CPython 3.11-3.13 win_amd64, MSYS2 gfortran). The workstation reproduces the Linux energies to
  1e-14 Eh. tblite's own CI, docs and wheel workflows are disabled in the fork (2026-11-05); only `qb-windows-wheels` runs.

### FORK-11: ROCm runtime hangs at process exit on Windows · ready (workaround in place)
- On the workstation (torch 2.9.1+rocm7.2.1), a Python process that used the GPU often never exits: `main()` returns
  and pytest prints its summary, then the process spins one core in the HIP runtime's teardown until it is
  killed (5 such processes on 2026-11-04/05). Workaround: `resources.hard_exit_if_needed` (`os._exit` after
  flushing) in `quantabench-cli`, the GUI, lane processes and pytest's `conftest.py`
  (`QUANTABENCH_HARD_EXIT=0|1` overrides). To do: a minimal repro (`import torch; torch.ones(1,
  device="cuda")`, then exit) and a report to ROCm/PyTorch; drop the workaround once fixed.

### FORK-10: ROCm SDK warning about paths with spaces · later
- On the workstation, AMD's Windows ROCm SDK prints "Unknown command line argument ...offload-arch.exe" when the venv
  path has a space ("GitHub Projects"): an unquoted path in its device-detection call. Harmless so far;
  reproduce outside QuantaBench and report to ROCm.

### FORK-5: CREST builds outside conda-forge · macOS done in CI 2026-11-06 (release needs-Brandon); Windows later
- **macOS arm64.** conda-forge's osx-arm64 crest 3.0.2 links Apple's Accelerate and fails every optimization of
  19+ atoms (docs/ENGINES.md, CREST). The fork builds it right: `brandonbtm29/crest` branch `qb/macos`
  (= `qb/windows` + workflow `qb-macos.yml`, docs/MACOS.md there): conda-forge's compilers, OpenBLAS instead of
  Accelerate, one OpenMP runtime, a folder that runs from anywhere. Run 42295799792 (commit `12eb61d`):
  artifact `crest-macos-arm64.tar.gz`, sha256 `47ba7883...4a33`; smoke cases match the Linux build of the
  same commit to 1e-10 Eh (single points, optimizations), and the MacBook's PFAS run modes match conda-forge's
  Linux 3.0.2 to 1e-8 to 2e-5 Eh.
  - Released 2026-11-07 (Brandon's OK): [`qb-v3.0.2-macos1`](https://github.com/brandonbtm29/crest/releases/tag/qb-v3.0.2-macos1),
    the run 42295799792 tarball unchanged (the downloaded asset has the pinned sha256). `scripts/fork-packages.json`
    points at it, so `install_macos.sh --with-engines` installs it and sets the Mac's CREST path. The MacBook
    already runs the same build, set as its path for that machine only (2026-11-07).
  - `quantabench-cli update` does not reinstall `programs` entries yet: add that when the first release exists.
  - Reported 2026-11-07 (Brandon's OK): conda-forge/crest-feedstock#22 (the fix is one CMake flag,
    `-DBLA_VENDOR=Generic`, plus the format commas or a libgfortran < 15 pin), and a comment on
    crest-lab/crest#412 with the diagnosis. Watch both; when a fixed conda-forge build ships, check it and
    consider going back to it.
  - The 20-molecule benchmark with the fork build (docs/ENGINES.md, CREST): GFN-FF gas and water 20/20 CREST,
    searches as close to the workstation's as two CREST runs are. Since 2026-11-07 the MacBook's own CREST path
    (`engines.machine_paths`, that machine only) is
    `~/Library/Application Support/QuantaBench/programs/crest-qb-v3.0.2-macos1/crest-macos-arm64/bin/crest`, so the
    GUI, the worker and the CLI use the fork build there (`doctor`: "settings (this machine)").
  - CREST lanes on the Mac (done 2026-11-07, branch `fix/crest-lanes`; re-time needs the Mac): the old
    `--lanes auto` gave 6 x 2 threads on the 14 cores (10 performance, 4 efficiency), and the batch waited for
    its slowest molecule on 2 threads. Lane timing on GFN-FF gas (2026-11-07, docs/ENGINES.md): 1 x 12 1,727 s,
    **3 x 4 773 s**, 5 x 2 1,457 s, auto 6 x 2 3,108 s.
    - Done: on Apple Silicon the CREST auto plan is 4 threads per lane, one lane per 4 performance cores
      (`hw.perflevel0.physicalcpu`), so 3 x 4 on the M3 Max. Other machines keep 2-thread lanes on all threads:
      on the workstation's 24 threads the code plans 11 x 2. The workstation's 2026-11-06 run was 11 x 2 too (its job table: 11
      molecules started at once), not "20 x 2" as the docs said; fixed. Every parallel run now starts its largest
      molecules first.
    - Done 2026-11-07 (branch `feat/dynamic-threads`): dynamic CREST threads. A running CREST cannot take more
      threads, but each CREST call's `-T` is set when it starts: a molecule that would set the end of the batch
      on its own gets whole lanes' worth of idle threads, and a run with fewer molecules than lanes lends the
      unused lanes' threads (docs/ENGINES.md, CREST). Simulated: no change for the Mac's 3 x 4 on the 20 PFAS or
      for large batches; the workstation's 11 x 2 on the 20 PFAS 1,244 -> 819 s. Measured on the Mac: the 20 PFAS 842 s
      (nothing widened); two large PFAS 585 and 555 s vs 1,309 and 1,096 s (part of it CREST's sampling).
    - Left: time it on the workstation (11 x 2, where the simulation predicts the gain; the workstation was busy with DFT on
      2026-11-07). Widening for the time limit (the GFN2 failures: 54950-05-9 needs about 3.7 h on 4 threads)
      needs the cost model in seconds before the first CREST call ends (a machine's `timings --fit` has that
      scale since 2026-11-08, SCHED-1; the policy does not use it for the limit yet). The xtb program and Psi4 could borrow
      threads the same way, but their thread count is set mid-molecule, so the lane would have to ask the parent.
    - Re-timed 2026-11-07 after the merge: `--lanes auto` gave 3 x 4 and started 54950-05-9 first; GFN-FF gas
      took 942 s (20/20), 3.3x faster than the old auto plan (the earlier hand-set 3 x 4 run: 773 s; sampling
      and the Mac's other load vary). Left: one GFN2 batch with the new plan.
  - GFN2 CREST on the Mac finished 2026-11-07: gas 16/20, water 15/20; the 9 failures are all the 3 h limit on
    the largest molecules. Compare their dipoles and conformers with the GFN-FF runs and Spartan (VAL-3).
- **Windows:** `qb/windows` has a native build (docs/WINDOWS.md there), but Brandon decided on WSL (ENG-2).
  Look at it again only if WSL becomes a problem.

### PRIO-1: Runs that keep the PC responsive · done 2026-11-09 (branch `feat/prio1`)
Brandon (2026-11-09): a 5-lane x 4-thread run on the workstation (below-normal CPU priority) made the PC hard to use, even
deleting files; Psi4's scratch I/O probably made it worse. QuantaBench had no priority or reserved-threads setting.
- Done: `quantabench/priority.py`; docs/ENGINES.md, "Run priority and free threads".
  - Two per-machine settings (`resources.machine_priority`, `resources.machine_free_threads`; the shared
    `resources.priority`/`free_threads` are only defaults):
    - run priority `normal` | `low` (default) | `background`: Windows below-normal or idle class plus low or very
      low I/O priority, POSIX nice 10/19 plus ionice best-effort 7 or idle, macOS disk I/O utility/throttle;
    - free threads (default 1 = the old "all but one", so every plan is unchanged; `background` leaves at least
      25 %, at least 2).
  - Applied to the lane processes, a CLI `build`/`grid build`/`worker` process, the GUI's build thread (in-process
    work only; never the GUI process), and every program `run_program` starts, through WSL too (`nice`/`ionice`
    in the wrapper). Never raised.
  - The auto thread budget, the lane plan and dynamic CREST threads leave the free threads alone.
  - Surfaces: `--priority`/`--free-threads` on `build`, `grid build`, `worker` and `lanes`; `settings --priority
    --free-threads [--machine]`; Settings > Compute "Run priority" and "Leave free"; `doctor`; run timings
    (`config_json`).
- Checked on the workstation (2026-11-09, a throwaway copy): a program started through `run_program` at `low` got
  BELOW_NORMAL_PRIORITY_CLASS and low I/O priority, at `background` IDLE_PRIORITY_CLASS and very low I/O, and the
  process it started inherited both.
- Left (needs-Brandon):
  - Time one CREST or Psi4 batch at `low` while using the workstation, against `normal`: is it fast enough when idle and
    light enough while working? If not, try `background` or more free threads.
  - WSL's VM (vmmem) runs at normal priority on Windows: niceness inside WSL does not make a CREST run through
    WSL give way to Windows programs; only fewer threads do.
  - Run the macOS path on the MacBook (`setiopolicy_np`, `taskpolicy -d`, the build thread's QoS class); it is
    written to Apple's documentation but not run yet.

### SCHED-1: Runtime-aware scheduling of conformer searches (study) · proposed 2026-11-08
Brandon (2026-11-08): choosing size-aware thread allocation could be a study in itself; also try moving small
molecules up the queue when only a few threads are free and the next molecule is large (it would do better on more
threads). This is moldable-job scheduling with backfilling, as HPC batch schedulers do it (EASY backfilling: a
small job may start in a gap if it does not delay the reserved start of the big one).
- What exists (2026-11-07, `dist/threads.py`, docs/ENGINES.md): a cost model (CREST time ~ atoms^3.9 with a x1.8
  scatter; speedup 2->4 threads 1.86x, 4->13 threads 1.74x), a discrete-event simulator (scratch, not in the repo),
  and a launch-time rule that widens a molecule only when it would dominate the batch.
- Study plan:
  1. Record every molecule's stage times, threads and CPU time in the job table, so each run adds training data.
     **Done 2026-11-08** (branch `feat/run-telemetry`; docs/ENGINES.md, Run timings): `telemetry_*` tables in
     every run's jobs.sqlite (run, load every 60 s, attempt, stage; partial rows for lost lanes),
     `quantabench-cli timings` (one CSV over run folders and grid parts) and `timings --fit` (per machine and CREST
     level, saved to `<settings>/machines/cost-models/<machine>.json` with >= 30 molecules over >= 2 widths; the
     dynamic threads use it). Overhead about 0.3 % (20 MMFF molecules). Left: run `timings --fit` on the workstation and the
     Mac once they have recorded runs (fixed lanes give one width: the fit needs dynamic-thread runs or other
     lane plans); leave out samples from a busy machine (own vs machine CPU percent) in the fit; the xtb program
     and Psi4 stages are recorded but have no cost model yet; Windows CPU time of `wsl:` CREST is not
     measurable from Windows (would need the Linux side's times).
  2. Move the simulator into the repo (`dist/simulate.py`) with policies as plug-ins: fixed lanes, largest first,
     launch-time width, reservations with backfilling (Brandon's idea), and online calibration of the cost model
     from the molecules already finished in the same run.
  3. Better runtime prediction: atoms, rotatable bonds, rings, charge and engine as features; a power law vs a
     small gradient-boosted model, judged by makespan, not by fit.
  4. Measure on the MacBook (performance + efficiency cores) and the workstation (SMT), batches of 20, 300 and 2,000, and on
     the GFN2 CREST tail, where widening could also avoid the 3 h time limit.
- Expectations: for thousands of molecules every policy ties (the queue empties evenly); the gains are in small
  batches, narrow lanes, a few dominant molecules and expensive stages (CREST GFN2, DFT re-ranking). Could become a
  short technical note.

### FORK-7: pyvistaqt 0.13 crash upstream · ready
- Minimal repro of the `initializeGL` access violation with the AMD driver, filed upstream; re-check the
  `<0.13` pin on each pyvistaqt release (and test 0.13 on macOS).

### FORK-8: LAMMPS reliability on Windows and a Mac GPU build · later
- With MD-1: the same SMD input 20 times with the Windows installer's `lmp.exe` vs the PyPI wheel. Then
  `scripts/build_lammps_macos.sh` for an OpenCL GPU-package build in mixed precision, with a measured
  speedup over the Mac's CPU.

### FORK-9: OpenMM on the Mac's GPU · later (0.4 MD)
- Benchmark the beta `openmm-metal` plugin against OpenCL; go/no-go note in PLATFORMS.md.

## Needs Brandon (hardware / Windows / decisions)

- **HW-1 · done 2026-10-28:** ROCm 7.2.1 PyTorch 2.9.1 on the Windows workstation sees the RX 9060 XT (matmul
  6.1 TFLOP/s fp32). But: no QuantaBench engine runs on it yet (see GPU-1, GPU-2), and AMD's Windows
  wheel ships a slow CPU BLAS (52 vs 881 GFLOP/s for PyPI's CPU torch), so CPU work got ~17x slower.
- **HW-2 · done 2026-11-05:** dxtb and AIMNet2 on the RX 9060 XT (single + batched), energies CPU vs GPU:
  see GPU-1 (AIMNet2 ~10x faster on the GPU) and GPU-2 (dxtb works but is 5-8x slower there).
- **HW-3 · needs-Brandon:** GUI on Windows 11. Done 2026-10-27: the full suite including all GUI tests
  passes natively on the Windows workstation (Radeon RX 9060 XT, OpenGL 4.6) with pyvistaqt 0.12; pyvistaqt 0.13
  crashed (now pinned `<0.13`). Still to check by hand: HiDPI picking feel, 20k-atom slab frame rate.
- **HW-4 · needs-Brandon:** LAMMPS `in.smd` with the GPU (OpenCL) build on Windows.
- **DEC-1 · needs-Brandon:** approve `docs/SIMULATE_DESIGN.md` (after RES-1).
- **VAL-1 · in review 2026-11-05:** tooling done on `feat/validate` (see VAL-1 above); the Spartan wide table of
  the PFAS QSAR run is the reference. Still needs-Brandon: a re-import for the new key columns, the charge
  mismatches, and the 300-molecule gas/water run. The Essential 76 ωB97X-D comparison is VAL-2.

## Later (from ROADMAP)
Orbital isosurfaces, NMR (PySCF), properties inspector,
align/overlay, constraints/scans/TS, isomer/tautomer enumeration (branched vs linear PFAS),
substructure search over results, force-field assignment (OpenFF/GAFF2/OPLS), cross-molecule GPU
batching, process-pool CPU stage.
