# Workers: one batch on several machines

`quantabench-cli worker` runs QuantaBench jobs for your other machines. `quantabench-cli grid build`
splits a molecule list across the workers (and, if you like, this machine), follows the progress, and
merges the result tables into one folder. Plan and background: [PLATFORMS.md](PLATFORMS.md), section 3;
code: `src/quantabench/dist/worker.py` and `client.py`.

```
Mac: quantabench-cli grid build --file pfas.csv --method gfn2-xtb --machines bench-pc,local --wait
        |  split by weight (dist.plan), the same BuildConfig for every share
        |--- HTTP/JSON over Tailscale ---> bench-pc worker -> ~/SyncThing/QuantaBench/Runs/<date>_<label>_bench-pc
        |--- in this process ------------> local share     -> ~/SyncThing/QuantaBench/Runs/<date>_<label>_<mac>
        '-- when all are done: summary.csv + descriptors.csv in Runs/<date>_grid_<label>/
            (a worker's run folder that has synced here completely is read in place; otherwise it is
             downloaded as a zip)
```

## What is shared, and what is not

| Where | What |
|---|---|
| `~/SyncThing/QuantaBench/Settings/settings.json` (synced) | `dist.worker_token` (the shared secret), `dist.worker_port`, `dist.worker_bind`, `dist.workers` (name -> URL overrides), `dist.machine_lanes` (each machine's own lane spec, by machine name) and `dist.lanes` (the default for a machine without one); since DIST-7 each machine's split choice, pause-on-battery and time window (`dist.machine_control`), when each per-machine value was set (`dist.machine_updated`) and the away times (`dist.away_after_s`, `away_running_after_s`, `rejoin_wait_s`) |
| `~/SyncThing/QuantaBench/Settings/machines/<name>.json` (synced) | each machine's profile; a running worker writes its URL into `address`, which is how `grid` finds it |
| `~/SyncThing/QuantaBench/Settings/machines/status/<name>.json` (synced) | DIST-7: what that machine's worker last said about itself (paused, last wake, battery); written only by that worker, so the Machines dock can show a closed laptop as paused |
| `~/SyncThing/QuantaBench/Runs/` (synced) | run folders. Each one has a single writer: the worker that runs it |
| Per-machine state folder (not synced) | the worker's job list and queued inputs: `%LOCALAPPDATA%\QuantaBench\worker` on Windows, `~/Library/Application Support/QuantaBench/worker` on macOS (`--state-dir` to change); since DIST-7 also `pause.json`, this machine's pause, which its worker, a `grid build`'s local share and the GUI's own batch follow |

The token is generated the first time a worker starts (`secrets.token_urlsafe(32)`) and saved in
`settings.json`. Start the first worker on one machine and let SyncThing copy the settings to the other
before starting the second, or both will generate a token and SyncThing will report a conflict.

## Commands

```
quantabench-cli worker [--bind tailscale|localhost|ADDR] [--port 8765] [--lanes SPEC] [--max-jobs 1]
                       [--name NAME] [--state-dir DIR] [--log-file FILE] [--tailscale-wait 120]
                       [--allow-any-address] [--allow-version-mismatch]
                       [--priority normal|low|background] [--free-threads N]
quantabench-cli worker pause [NAME] [--drain|--now]   (no NAME: this machine)   DIST-7
quantabench-cli worker resume [NAME]
quantabench-cli machines set [NAME] [--priority normal|low|background|default] [--free-threads N|default]
                       [--max-cpu PCT] [--lanes SPEC|default] [--split auto|manual|off] [--weight W]
                       [--[no-]pause-on-battery] [--window HH:MM-HH:MM|none]
quantabench-cli grid machines [--worker NAME=URL]
quantabench-cli grid build <same inputs and science options as build> --method M
                       [--machines bench-pc,bench-mac,local]
                       [--weights auto|measured|equal|benchmark|NAME=W,...|3,1] [--exclude NAME] [--wait]
                       [--no-share-work] [--label L] [--worker NAME=URL] [--sync-wait 60] [--away-after S]
quantabench-cli grid status [GRID_ID]          (no id: recent grid runs)
quantabench-cli grid cancel GRID_ID
quantabench-cli grid merge GRID_ID [--partial] (after a build without --wait)
quantabench-cli batch run <same inputs and science options as grid build> --method M [--phases gas,water]
                       [--run-dir DIR] [--deadline HH:MM|ISO|none] [--machines auto|local|NAME,...]
                       [--include-paused] [--max-attempts 4] [--no-pair]          BATCH-1
quantabench-cli batch run --list batches.txt [--root DIR] ...   (one input file per line, one batch each)
quantabench-cli batch status RUN_DIR
```

- **`--method` must be explicit.** `auto` picks the engine per machine (AIMNet2 on the workstation's GPU, something
  else on the Mac), which would mix levels of theory in one table; `grid build` refuses it unless
  `--allow-auto-method`.
- **Weights** (DIST-7; the split is deterministic, SHA-256 of the molecule id, so re-running a list puts every
  molecule on the same machine again):
  - `auto` (the default): each machine as its split setting says (the Machines dock's Split column, or
    `machines set NAME --split auto|manual|off --weight W`): *Auto (measured speed)*, *Manual weight* (on the scale
    where 1 is the mean of the Auto machines) or *Off* (left out of the grid). With no setting every machine is
    Auto, i.e. `measured`.
  - `measured`: each machine's molecules/hour in the newest grid of the same protocol that ran on all of them
    (method, solvent, conformer search, conformers, refine-top, frequencies, QM, rank; `grid.json` `rates`, written
    at the end of every followed grid), else its benchmark (`benchmark`). The log says which grid it used.
  - `benchmark`: the old `auto`: `throughput` in the profile its worker reports (else the registry copy), an
    estimate from its core count when it has no benchmark for that method, 0 where the method is unavailable; a
    machine with no profile gets the mean of the others.
  - `equal`; `3,1` (one weight per machine in the order of `--machines`); `bench-pc=3,local=1` (named).
  - `--exclude NAME` (repeatable or a comma list; `local` = this machine) leaves a machine out; with `3,1` each
    weight stays with its machine.
  - At the end of a grid, `grid build --wait` and `grid status` print the molecules each machine contributed to
    the merged tables, what moved and why, and each machine's measured rate.
- **Largest molecules first.** Each share's list is sorted by size (atoms with hydrogens) before it is
  submitted, so the end of every queue holds short, cheap molecules.
- **Work sharing** (on by default; `--no-share-work` turns it off). While `grid build` follows the grid
  (with `--wait`, or while its local share runs), a machine whose shares are all done takes molecules that
  another machine has not started:
  - The rates are measured in this grid: molecules finished per second of each machine's shares, once 3 have
    finished.
  - For an idle machine T and a running share of machine D with P molecules not started, it moves
    m = P r_T / (r_T + r_D), so that both would finish together. It only does so when that ends D's part at
    least 60 s sooner, after charging T 30 s to start a job.
  - D's batch hands them over between molecules, from the end of its queue (`POST /v1/jobs/<id>/release`).
    They become `moved` in its job table (never run there, also not on a resume), and T runs them as a new
    share of the grid (`parent` = D's job, `ids` = the molecules).
  - A machine can get several shares this way. Each has its own run folder, and a download goes to
    `parts/<machine>-<job id>`. `grid status` shows where a share's molecules came from, and the merge
    takes every share.
  - Machines whose share failed, was cancelled or was lost never take work. If T cannot start the molecules,
    they go back to D as another job; if that fails too, they are logged and end up in `grid.missing`.

  Weights still decide the first split, so good weights mean fewer moves.
- **Threads and GPU** come from each machine's own settings. `--threads`/`--gpu` on `grid build` override
  them everywhere.
- **Run priority and free threads** (docs/ENGINES.md) are each machine's too: a worker lowers its own process
  at start to its `--priority` (else that machine's setting, default `low`), and its jobs' lanes and programs run
  there. `grid build --priority`/`--free-threads` send the values to every machine, but a job never runs above its
  worker's own priority.
- `local` runs its share inside the `grid build` process, so that command stays open until the local share
  is done (with `--wait`, until everything is done and merged). Ctrl+C stops following; remote shares keep
  running (`grid status`, then `grid merge`).
- A worker runs one job at a time (`--max-jobs`) and each job through `jobs.run_resumable`, so `--lanes` (or
  this machine's lane spec from the settings) spreads one job over this machine's GPU and CPU lanes. The
  settings file is shared, so each machine keeps its own spec in `dist.machine_lanes` (Settings > Compute >
  Parallel lanes saves this machine's entry; `build`, the GUI, the worker and the local share of `grid build`
  all read it). A spec from the settings that does not fit this machine (a GPU lane on the Mac) is logged and
  the job runs in one lane; an explicit `--lanes` that does not fit is an error.
- **Molecule ids** that differ only in letter case (`c1ccccc1` and `C1CCCCC1`, ids made from SMILES) would
  share one folder on Windows and macOS: the list is renamed `..._2` before it is split, and a worker refuses
  such a pair.
- `grid build` warns when a worker runs another git commit than this machine (the version number does not
  change with every commit), and records each machine's commit in `grid.json`.
- **Restart a worker after an update.** A running worker keeps the code it started with. That applies to
  Help > Check for updates, `quantabench-cli update [--upgrade-packages]` or `git pull`, on its machine.
  The update changes the files under it, so restart the worker when it is idle (`grid machines` shows its
  queue): Ctrl+C or `pkill -INT -f "quantabench-cli worker"`, then start it again. On the Mac, launchd
  starts it again by itself. On Windows, run `Stop-ScheduledTask "QuantaBench worker"`, then
  `Start-ScheduledTask "QuantaBench worker"`; the stop ends the process at once, so wait until it is idle. `grid machines` then shows the
  new commit. On Windows, stop the worker before `--upgrade-packages`: Windows keeps the files of a running
  program locked, and pip cannot replace a package the worker has loaded. Help > Check for updates and
  `quantabench-cli update` now look at this computer's worker first (DIST-1): while it runs or is about to run a
  job from this checkout the update waits (the app) or is refused (the command line); afterwards they restart an
  idle worker kept by the LaunchAgent or the scheduled task (the app's "Restart this computer's worker too",
  `quantabench-cli update --restart-worker`). Updating workers on other machines from the client is planned
  (BACKLOG UPD-1).
- Stopping a worker with Ctrl+C, Ctrl+Break, SIGINT or SIGTERM stops its job after the current molecule. That
  includes a worker started in the background (`nohup … &`, over ssh), which `pkill -INT -f "quantabench-cli
  worker"` now stops too (it waits up
  to a minute, then exits); a killed or crashed worker loses only the molecule in progress. Either way the next
  start resumes its unfinished jobs, except those cancelled before it stopped. The job list is written with
  fsync, followed by a backup copy (`jobs.json.bak`), so a power cut cannot lose both.
- **Following and merging.** A worker that answers a status request with a 4xx (it does not know the job,
  or the token changed) marks that share `lost` and `grid build --wait` stops waiting for it; an unreachable
  worker is waited for. A share that cannot be collected is reported and left out of the merge
  (its molecules are listed as missing) instead of stopping it. `grid merge --partial` merges what is
  finished so far, reading unfinished shares from the synced copy of their run folder (from a private copy
  of its job table, so nothing is written into another machine's folder). A share run by `local` on the
  other machine is found in the synced output folder. Molecules that crashed their worker process twice have
  no summary row; `grid.json` lists them in `failed_without_result`.
- **Timings.** Each share's run folder records its own machine, lane plan and stage timings in its `jobs.sqlite`
  (docs/ENGINES.md, Run timings). `quantabench-cli timings <grid folder>` reads the downloaded parts
  (`parts/<machine>/...`), and `timings` on the output folder reads the synced ones; a run found in both is
  read once.

## Batches: phases, a deadline and resume (BATCH-1)

`batch run` runs one molecule list as phases (`--phases gas,water`) on the machines that answer, into one run folder,
and stops at a deadline. Running the same command again resumes it: only molecules without a result run. It is what
the QSAR tranches and the CREST rebuild ran through scripts on the workstation until 2026-11-18
(`src/quantabench/dist/batch.py` has the details).

- **Attempts.** Each phase (`<run dir>/gas`, `<run dir>/water`, ...) runs as attempts. An attempt is a
  `grid build --wait` over the molecules no run folder of the phase has finished. One that merges ends the phase,
  unless a machine went away with molecules: the next attempt runs those (at most `--max-attempts`, default 4).
- **Machines.** Before each attempt the workers are asked. `auto` takes every worker that answers, is not holding,
  and runs this machine's version and commit, plus this machine. A machine named in `--machines` that is not ready
  stops the batch (exit 4). `--include-paused` keeps a paused machine in the grid: its unstarted molecules move at
  once and it gets work again after its resume, so a machine with a time window (`machines set --window`) still
  joins a phase that starts outside its hours.
- **Deadline.** At `--deadline` (`07:00` = the next 07:00) the remote shares are cancelled (each stops after its
  current molecule), the local share stops at its next molecule, and the batch ends with exit 5.
- **Snapshots.** What the other machines finished in their synced run folders is copied into the phase folder
  (`partial_<run folder>`) after an attempt that did not merge and before the next one. The shares come from every
  unmerged `grid.json` of the phase, so even an attempt that was killed counts.
- **At the end:** the gas phase is paired with the others (`paired.csv`), and `report.md` lists each attempt:
  machines, molecules, minutes, what was left. `log.txt` has the batch's own lines; each attempt's grid output is in
  `grid_<phase>_rN.log`. `batch status RUN_DIR` prints the phases.
- **Exit codes:** 0 done; 5 stopped at the deadline or by Ctrl+C (resumable); 4 a named machine not ready; 1 an
  attempt failed or a phase used up its attempts (resumable); 2 bad input.
- **Taking over a script's run.** The phase folders have the same layout as `run_tranche.py`'s, so `batch run
  --run-dir <its run root>` with the same inputs and options continues it. Its earlier attempts count toward
  `--max-attempts`; give a higher number.

## Pausing a machine, a closing laptop (DIST-7)

Brandon (2026-11-11): "I'm often opening and closing my laptop throughout the day. Is there a way we include a pause
button on a machine and the work is redistributed?" Code: `dist/control.py`, `worker.JobManager` (pause, wake, sync),
`client.MachineWatch` (what the grid does), `jobs.BatchControl` (what a batch does).

### Pausing by hand

- **Where:** the Machines dock (a Pause / Resume button per machine; its arrow has *Pause now*),
  `quantabench-cli worker pause [NAME] [--drain|--now]` and `worker resume [NAME]`, or `POST /v1/pause`.
- **What it means:** the machine takes no new molecules.
  - `--drain` (the default) lets the molecules in progress finish.
  - `--now` stops them at their next checkpoint (between conformers, at an optimizer step). They are recorded as not
    started, so they can move too.
  - A worker's batch then ends as `paused`: its lanes exit and its memory and GPU are freed. The molecules it has
    not started stay in its job table. Resume queues the paused jobs again, first, and they go on from their job
    tables (new lanes, so new lane settings apply).
- **Which machine:**
  - No NAME means this machine. The pause is written to its pause file (`<worker state folder>/pause.json`), and to
    its worker if one runs here. The worker, the local share of a `grid build` running here and the GUI's own batch
    all follow that file. The GUI's batch holds and goes on at resume; it is not split anywhere else.
  - NAME means that machine's worker, over Tailscale.
- **It lasts:** a paused worker stays paused across restarts until it is resumed. `grid machines` shows
  `PAUSED (drain since ...)`, and the dock shows `paused (drain)`. For a worker that does not answer, the dock shows
  what its status file last said (`machines/status/<name>.json`).
- **What a grid does:**
  - A `grid build` that follows the grid (`--wait`, or while its local share runs) sees the pause in the next status
    poll (every 5 s).
  - It takes every molecule of that machine's shares that has not started: from the draining batch between
    molecules, or from the stopped job's table.
  - It splits them between the other machines by their rate in this grid (else the split's weights) and submits
    them as new shares (`reason: paused`).
  - With no other machine to take them, they wait for the resume.
  - This is DIST-6 work sharing with a paused machine's rate counted as zero. Work sharing itself stays on.
- **On battery:** `pause_on_battery` (on by default; `machines set NAME --no-pause-on-battery`, or the dock's budget
  box) pauses a laptop's worker while it runs on battery (`psutil.sensors_battery`, checked every 30 s) and resumes
  it on power. A pause by hand is never lifted by the power coming back. A desktop has no battery, so it never
  pauses.
- **A time window:** `machines set NAME --window 02:00-08:00` (or the dock's budget box, "Time window") lets a
  machine work only in those hours of its own local time; `22:00-06:00` wraps midnight, `none` removes it.
  - Its worker pauses itself with drain when the window closes (reason `window`, checked every 30 s), so a grid
    moves its unstarted molecules to the other machines as for any pause. When the window opens it resumes, and work
    sharing gives it molecules again.
  - It acts only when the window opens or closes. A pause by hand inside the window stays, and a resume by hand
    outside it holds until the window next closes. A worker that starts outside its window starts paused.
  - It replaces the scheduled script that did this for the MacBook in the CREST rebuild (2 to 8 am, Nov 17-18).
  - It pauses the machine, not one job: its worker and the local share of a `grid build` started on it follow the
    same pause file.

### A machine that stops answering (the lid closes)

When the lid closes, the Mac sleeps: its worker stops answering and ssh times out. Nothing has to be pressed:

1. **Away.** A machine whose worker has not answered a status poll for `dist.away_after_s` (default **120 s**;
   `grid build --away-after S`) is *away*.
   - Its molecules that had not started at the client's last look are given to the other machines (`reason:
     away`). The client fetches `GET /v1/jobs/<id>/molecules` each time a share's counts change, at most every 10 s.
   - After `dist.away_running_after_s` (default **600 s**), the molecules it was running are given away too.
   - Each share records what was given away in `grid.json` (`reassigned`).
2. **`--wait` never hangs on it.** A share whose machine is away and whose unfinished molecules have all gone
   elsewhere is not waited for. With no other machine to take them, it is not waited for after the longer grace
   either: its molecules are reported missing, and `grid merge <id> --partial` adds them later. The grid then
   merges what it has, from the synced copy of the away machine's run folder for what it finished before the lid
   closed. The end of the log lists what moved (`Moved: 17 molecule(s) bench-mac -> bench-pc (away)`).
3. **Back.**
   - The worker notices the sleep itself: a watcher thread checks every 2 s, and a wall-clock jump of more than
     30 s against its own sleep is a wake (`control.ClockWatch`). It works on every OS: macOS and Linux stop the
     monotonic clock while asleep, and elsewhere the check simply comes late.
   - It also notices when a grid client that followed a job stops asking for `away_after_s` (its network went
     away, or ours did).
   - Either way, a job that a client was following starts no new molecule (`rejoin`) and asks the client what is
     still its own. The client sees `rejoin: true` (or simply that the machine answers again) and sends
     `POST /v1/jobs/<id>/sync` with the molecules it gave away; the worker drops those it has not started and goes
     on with the rest.
   - If the client is gone (the grid has finished), the worker reads the grid's `grid.json` in the synced Runs
     folder instead and drops the molecules listed there. If neither answers within `dist.rejoin_wait_s` (default
     600 s), it goes on.
4. **No double counting.**
   - A molecule both machines finished (the away machine was still computing, or finished the one it was running
     when it woke) is merged once, from the machine that finished it first (an `ok` before a `failed`), and listed
     in `grid.json` `duplicates`.
   - `quantabench-cli timings` marks the dropped result's rows `duplicate=1`, leaves them out of the stage summary
     and prints the molecules per machine with each molecule counted once.

Tested in the cloud with fake workers (tests/test_machine_control.py) and on the workstation with two throwaway workers, one of
them frozen for 55 s (`psutil` suspend, which is what a sleep looks like to the other side); see the backlog
(DIST-7).

### Settings that change while a job runs

- **Who writes `settings.json`:** only a person does, through the Machines dock, `machines set` or
  Settings > Compute (and `ensure_token` once). Workers never write it: their own state goes to the per-machine
  pause file and the status file above. Every per-machine value is written with a fresh read, the change and a
  write (`settings.update_machine`), and stamped in `dist.machine_updated`.
- **Saves merge and skip:** every save merges with the file as it is now. Per-machine tables merge entry by entry,
  and other values key by key: what this process did not change comes from the file. A save that would not change
  the file does not write it (the GUI saves at every Run).
- **SyncThing conflicts:** if two machines still write within the same minute, SyncThing keeps one version and
  renames the other `settings.sync-conflict-*.json`. `load` then merges back every per-machine entry of the copy
  that is newer than the file's (by its stamp), and the next save writes it. The copy can be deleted afterwards.
- **A worker reads its machine's values at job start**, and re-reads the settings every 15 s:
  - The **run priority** of a job that takes it from the settings is applied to its running lanes at once
    (`priority.apply_to_pids`). Lowering works everywhere. Raising works on Windows only; macOS and Linux cannot
    raise a nice value without privileges, so there it applies to the next worker start.
  - **Lanes and free threads** apply with the machine's next job, or after a pause and resume, which restarts the
    job's lanes. A running pool is not resized.
- **Max CPU %** (dock, `machines set --max-cpu 50`) sets the free threads from the machine's thread count, e.g.
  12 of 24 on the workstation.

### Keeping the Mac's worker running (and not draining its battery)

Found on 2026-11-11 (read-only look over ssh):

- The worker had been started with `nohup` over ssh, and no LaunchAgent was installed (`~/Library/LaunchAgents`
  has none, `launchctl list` shows none).
- Its log shows a clean stop at 2026-11-09 20:10:21 ("Stopping: ..."): macOS sends SIGTERM at a restart or logout.
  The Mac then rebooted (boot time 20:14:57), and nothing started the worker again until it was started by hand on
  11-11 at 14:35.
- Sleep does not kill it: the worker started at 14:35 survived a 14-minute sleep (15:21-15:35) and kept its process.

So: install the LaunchAgent of the setup section below. `RunAtLoad` starts it at every login, and `KeepAlive` with
`SuccessfulExit false` restarts it after a crash but not after a clean stop. With it:

- **Lid closed:** the Mac sleeps (on battery always; on power unless it has an external display), the worker
  freezes with it, and a grid treats it as away (above). It does not keep the Mac awake: `caffeinate -i` is held only
  while a job runs, and only prevents *idle* sleep, never the lid's.
- **Battery:** `pause_on_battery` is on by default, so on battery the worker takes no work and holds no
  `caffeinate`. Plugged in, it resumes. To let it work on battery: `quantabench-cli machines set
  bench-mac --no-pause-on-battery`.
- **Wake:** launchd keeps the same process across sleep. The worker notices the wake and checks with its grid client
  before it starts another molecule.

The LaunchAgent is for the user to install; QuantaBench does not install services itself. On Brandon's MacBook it
was installed on 2026-11-12, at his request (`launchctl print gui/$(id -u)/io.github.brandonbtm29.quantabench-worker`
shows it; a test `kill -KILL` of the worker was followed by a new one within a second).

## The MD benchmark (MD-BENCH-1)

The throughput benchmark gives molecules/hour per engine. An MD job needs a different number: how many ns/day
this machine's LAMMPS runs a molecular liquid. `md` in a benchmark list measures it:

```
quantabench-cli machines --bench md [--md-size small|pilot] [--md-variants LIST] [--md-threads N] [--max-seconds 60]
quantabench-cli machines --bench default,md        (the engines as well)
quantabench-cli worker --bench-methods default,md  (at the worker's start; auto: only what is missing)
quantabench-cli md estimate (DIR | --atoms N --ns NS) [--dispersion ewald] [--include-unverified] [--all-variants]
```

It is never in the default list, so a worker starts as fast as before unless `md` is asked for.

- **The system.** Liquid C6F14, OPLS-AA with the Watkins-Jorgensen perfluoroalkane types (`md.forcefield`), at
  QuantaBench's density estimate. `small` is 150 molecules (3,000 atoms, a 37 Å cube). `pilot` is the 489
  molecules (9,780 atoms) of MD-2's pilot slab.
  - It is packed by Packmol, else as a lattice, then minimized, compressed to the estimated density in 1 ps of
    NVT and minimized again.
  - This is done once per QuantaBench version and size, in the machine's cache folder (`md_bench/`), not in the
    synced settings.
- **The runs.** One LAMMPS process per variant, each ending without `clear` (`in.run`, which chains bulk and slab
  with `clear`, hung on the GPU package).
  - Each run reads the cached box, draws velocities at 298.15 K from a fixed seed, does `run 0`, 100 warm-up
    steps, then NVT at 1 fs. The timed run stops at 10,000 steps or after `--max-seconds` (default 60; LAMMPS'
    `timer timeout`), whichever comes first.
  - ns/day is LAMMPS' own `Performance:` line for the timed run.
  - Variants are `<accelerator>-<dispersion>`, with the dispersion variants of `md liquid-slab` and its default
    long-range settings (since MD-LR, `ewald` with Coulomb 1e-6 and the slab-safe dispersion mesh; each record names
    them in `long_range`, and a record without it timed the older, coarser settings):
    - `cpu-*`: `-sf omp -pk omp N`, N = `--md-threads`, by default this machine's one-lane thread budget, at most
      its physical cores.
    - `gpu-*`: `-sf gpu -pk gpu 1`, cutoff variants only (LAMMPS has no GPU version of the `ewald` styles).
    - `kokkos-*`: `-k on g 1 -sf kk`, or `-k on t N -sf kk` for an OpenMP KOKKOS build.
  - The default is `cpu-ewald`, `cpu-cutoff12` and `cpu-cutoff16`, plus `gpu-cutoff12` and `gpu-cutoff16` where
    `lmp -h` lists the GPU package and a compatible GPU. KOKKOS only runs when listed in `--md-variants`.
  - A GPU run records the device the GPU package chose (`device`, from its `Device 0: ...` line), so a run that
    went to an integrated GPU shows.
- **The CPU/GPU check.** On the workstation the GPU package's NPT expanded a C6F14 box to half its density while
  the CPU held it, so a fast GPU number is not trusted on its own.
  - Every non-CPU variant's `run 0` is compared with the CPU's for the same dispersion: the same configuration and
    the same velocities. If the CPU variant is not in the list, its `run 0` alone is run.
  - The pressure must agree within 20 atm. The potential energy must agree within 0.5 kcal/mol + 1e-4 x the size
    of its terms. Mixed precision explains far less than either; the LJ tail correction alone is 58 atm at 16 Å.
  - The result is `check`: `ok`, `mismatch` (shown as *unverified*) or `skipped` (no CPU reference). The `run 0`
    numbers of both runs are stored, so a mismatch shows which term differs.
- **In the registry.** `md_throughput` in `machines/<name>.json` holds:
  - the LAMMPS version, its packages and accelerators, and the system;
  - per variant: ns/day, atoms, threads, steps, seconds, the `run 0` values and the check.
  - A file without the key (an older QuantaBench) loads as before.
  - Variants not measured again are kept. A failed re-measurement replaces the old number. A run without LAMMPS
    keeps the old numbers and notes the attempt.
  - `quantabench-cli machines` shows it as `md: cpu-ewald 6.5 ns/day (9,780 atoms, 6 thr), gpu-cutoff16 10.6
    ns/day (9,780 atoms, OpenCL) [pressure check: mismatch]`, and the Machines dock as its "MD (ns/day)" column
    (the full line in its tooltip).
- **Estimates.** `md estimate` and `md liquid-slab` scale a variant's ns/day by the benchmark's atoms / the job's
  atoms (PPPM MD costs about linearly in atoms at fixed density).
  - A cutoff that was not measured is scaled from the nearest one on the same accelerator by (rc_measured/rc)³,
    and the estimate says so.
  - Unverified variants are left out unless `--include-unverified`.
  - `quantabench.md.bench.estimate_hours(md_throughput, atoms, ns, dispersion=...)` is the same in Python.
- **Without LAMMPS** the benchmark is skipped with the reason (`md: not measured (LAMMPS not found (...))`).
  Nothing else in the profile changes.
- **Not yet.** MPI ranks (`mpirun -np N lmp`, often faster than OpenMP threads for LAMMPS on a CPU) are not
  benchmarked. The KOKKOS variants have not run on a KOKKOS build yet.

## API (version 1)

Every request needs `Authorization: Bearer <settings.dist.worker_token>`.

| Request | Answer |
|---|---|
| `GET /v1/info` | name, URL, QuantaBench version and commit, machine profile, queue (`running`, `queued`, `paused`), and since MON-1 the latest sensor sample (`sensors`, without the per-CPU lists; null before the first) |
| `GET /v1/sensors` | since MON-1: live machine metrics `{sensors: a sensors.Sample dict or null, why, interval_s, machine}` (CPU and GPU load, effective clock, temperatures, power, VRAM, RAM; docs/ENGINES.md, Live machine metrics). The worker samples while requests keep coming and stops after 2 minutes without one; older workers answer 404 |
| `GET /v1/jobs` | every job, newest first |
| `POST /v1/jobs` | body `{inputs: [jobs.input_to_json(...)], config: {BuildConfig fields}, label, chunk?, lanes?, client?, allow_version_mismatch?}`; header `X-QuantaBench-Version`. 201 `{id, job}` |
| `GET /v1/jobs/<id>?tail=20[&follow=1]` | state (`queued`, `running`, `paused`, `done`, `failed`, `cancelled`, `interrupted`), counts from the run folder's job table, `run_dir`, `run_folder`, progress, log tail; since DIST-7 `machine_paused` and `rejoin` (the job holds after a wake until its grid client syncs). `follow=1`: a grid client that follows the job (the worker then notices when its polls stop) |
| `POST /v1/jobs/<id>/cancel` | a queued job is dropped; a running one stops after the current molecule (its run folder stays resumable); a paused one (DIST-7: its batch stopped with molecules left) ends cancelled at once, so the machine's next resume does not run it (FIX-8) |
| `POST /v1/jobs/<id>/release` `{"max": N, "to": "machine"}` | work sharing: up to N molecules the job has not started, from the end of its queue, become `moved` there; 200 `{ids, job}` (`ids` empty when it has none left). A running batch answers between molecules: within 0.1 s with lanes, after the current molecule in one process (the request gives up after 30 s and then moves nothing); a paused or queued job (DIST-7) is changed at once, and a paused job left with nothing is finished |
| `GET /v1/jobs/<id>/archive` | zip stream of the run folder (finished jobs; 409 while running); its last member, `<run folder>/.qb-archive.json`, lists every other member and its size |
| `GET /v1/jobs/<id>/molecules` | DIST-7: `{"molecules": {id: status}}` (the job table, or the queued inputs of a job that has not run): which molecules it has not started |
| `POST /v1/jobs/<id>/sync` `{"revoke": [ids], "following": bool}` | DIST-7: the grid client gave these molecules to another machine while this one was away: those not started here become `moved`; ends a wake-up hold. `following: false`: the client stopped following (no lost-contact hold). 200 `{moved, done, running, job}` |
| `POST /v1/pause` `{"paused": bool, "mode": "drain"\|"now", "by"?}` | DIST-7: pause this machine (no new molecules; `drain`, the default, lets the running ones finish, `now` stops them at their next checkpoint) or resume it; saved, so a restart stays paused. 200 = the queue state (`paused`, `pause`, `paused_jobs`, `rejoin`, `last_wake`) |

`chunk` is `{"index": i}` for a share the client already split (recorded only), `{"index": i, "count": n}`
for a `jobs.chunk_index` chunk, or `{"index": i, "weights": [...]}` for a `plan.weighted_index` share of a full
list. Errors are `{"error": "..."}` with 400 (validation), 401 (token), 404, 405, 409 (version mismatch, job
still running), 411/413/415 (body).

## Security

- The worker listens on this machine's **Tailscale address** (from `tailscale ip -4`), or 127.0.0.1 with
  `--bind localhost`. It refuses 0.0.0.0 / `::` unless `--allow-any-address`. Tailscale (WireGuard) encrypts
  the traffic, so plain HTTP is fine there; the client refuses to send the token over plain HTTP to any
  address that is not loopback or Tailscale (`--allow-any-host` overrides).
- Token compared in constant time (`hmac.compare_digest`); every endpoint, including unknown ones, answers
  401 first. The token never appears in profiles, logs or API answers.
- Requests are limited in size (`--max-request-mb`, 64 MB) and validated field by field: only known
  `BuildConfig` fields, with types, ranges and choices checked (`method` must be an engine of the worker,
  `qm_method` and `solvent` go through their parsers, `formats` from the export list); `output_dir` is always
  the worker's own; labels and molecule ids must be single folder-safe names (no separators, `..`, or Windows
  device names); unknown fields anywhere are refused.
- The client checks **every connection** before it writes the Authorization header: the address it connects
  to must be loopback or Tailscale, and for a Tailscale address this machine's end of the connection must be a
  Tailscale address too. 100.64.0.0/10 is also carrier-grade NAT space: with Tailscale stopped or not yet
  reconnected on the Mac, a connection to `100.x` would leave through the café's Wi-Fi, and it is refused
  ("Tailscale is not connected on this machine"; `--allow-any-host` does not change that).
- The client never uses a proxy and never follows redirects (urllib would forward the Authorization header).
  Downloaded archives are extracted only after every member path is checked (no absolute paths, `..`, drive
  letters or links), the archive holds the job's own run folder, and it ends with the worker's manifest
  listing exactly the other members: a stream that failed or was cut off is refused, not merged short. A
  machine name a worker reports must be a single folder name (it names `parts/<machine>/`).
- A synced run folder is read in place only when the job has ended and its `worker.json` (written last, after
  the tables) lists the size and SHA-256 of `summary.csv` and `descriptors.csv` and the copies here match:
  SyncThing brings files in any order. SyncThing's temporary files never go into an archive.
- A settings file that exists but cannot be read stops the worker (`cannot read ... settings.json`) rather than
  being replaced by defaults and a new token.
- On Windows the socket uses `SO_EXCLUSIVEADDRUSE`, so no other program can bind the same port.

## Setup on the Windows workstation (Windows 11)

1. **Firewall rule** (PowerShell as administrator, once). It allows the worker port only on the Tailscale
   interface and only from tailnet addresses:

   ```powershell
   New-NetFirewallRule -DisplayName "QuantaBench worker (Tailscale)" -Direction Inbound -Action Allow `
     -Protocol TCP -LocalPort 8765 -InterfaceAlias "Tailscale" -RemoteAddress 100.64.0.0/10 -Profile Any
   ```

   Check the adapter name with `Get-NetAdapter | Where-Object InterfaceDescription -like "*Tailscale*"`.
   Create the rule *before* the first start. If Windows still asks whether Python may accept connections,
   choose **Allow**: pressing Cancel creates *block* rules for python.exe, and block rules beat allow rules
   (find them with `Get-NetFirewallRule -Direction Inbound -Action Block | Where-Object DisplayName -like "python*"`
   and remove them with `Remove-NetFirewallRule`). A port rule is used rather than a program rule because the
   venv's `python.exe` is a launcher that starts the base interpreter, which is the process that listens.

2. **First start, by hand** (generates the token and publishes `machines/bench-pc.json`):

   ```powershell
   cd "$HOME\GitHub Projects\QuantaBench"
   .venv\Scripts\quantabench-cli worker
   ```

   It should print `worker 'bench-pc' at http://100.64.0.10:8765`. Ctrl+C stops it.

   **A machine that joins is measured once.** On its first start (no throughput in its profile yet) the
   worker runs the throughput benchmark before it takes jobs: molecules/hour per installed engine on the CPU
   and each GPU, a few minutes, with submitted jobs held (`grid machines` shows it as benchmarking). The result
   goes into its profile, so `grid build --weights auto` gives it the right share. Later starts skip it;
   `--bench always` measures again (after a hardware or PyTorch change), `--bench never` never does. The GUI's
   Machines dock has the same as a button, "Benchmark this computer". `--bench-methods default,md` adds the MD
   benchmark (below; a few minutes more), and with `auto` a machine that has molecules/hour but no MD numbers
   then measures only the MD part.

3. **Start at login** (normal PowerShell; no console window, log in `%LOCALAPPDATA%\QuantaBench\worker.log`):

   ```powershell
   $repo = "$HOME\GitHub Projects\QuantaBench"
   $log = Join-Path $env:LOCALAPPDATA "QuantaBench\worker.log"
   $action = New-ScheduledTaskAction -Execute "$repo\.venv\Scripts\pythonw.exe" `
     -Argument "-m quantabench.cli worker --log-file `"$log`"" -WorkingDirectory $repo
   $trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
   $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
     -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew `
     -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
   Register-ScheduledTask -TaskName "QuantaBench worker" -Action $action -Trigger $trigger -Settings $settings `
     -Description "quantabench-cli worker on the Tailscale address (docs/WORKERS.md)"
   ```

   `-ExecutionTimeLimit 0` matters: the default (3 days) would end the worker. Stop and start it with
   `Stop-ScheduledTask "QuantaBench worker"` / `Start-ScheduledTask "QuantaBench worker"`; remove it with
   `Unregister-ScheduledTask "QuantaBench worker"`. Stopping the task ends the process at once; the molecule in
   progress is redone on the next start (the worker's lock records its process start time, so a stale lock
   whose process id now belongs to another program does not block the next start). While a job runs the
   worker asks Windows not to sleep.

   **Restarts are not verified.** Task Scheduler's restart settings (`-RestartCount`, `-RestartInterval`) are
   documented for a task that *fails*, and it is not clear (not tested on the workstation) that a worker exiting with an
   error code, or crashing, counts as that. The worker itself retries what is most likely at login: it waits
   `--tailscale-wait` (120 s) for Tailscale's address and then for the address to be usable. To check, after a
   reboot or a crash: `Get-ScheduledTaskInfo "QuantaBench worker"` (`LastTaskResult` 267009 = 0x41301 while it runs,
   2 for a start the worker refused; the reason is at the end of `%LOCALAPPDATA%\QuantaBench\worker.log`) and
   `quantabench-cli grid machines` from the Mac. If it does not come back by itself, start it with
   `Start-ScheduledTask "QuantaBench worker"`.

## Setup on the MacBook (macOS, Apple Silicon)

1. **Firewall:** if the macOS firewall is on, the first start asks whether Python may accept incoming
   connections: choose Allow. Tailscale's ACLs must allow your devices to reach each other on tcp:8765 (the
   default policy does).
2. **First start, by hand** (after SyncThing has brought `settings.json` with the token from the workstation):

   ```bash
   cd "$HOME/GitHub Projects/QuantaBench"
   .venv/bin/quantabench-cli worker
   ```

3. **Start at login** with a LaunchAgent (restarts it if it crashes; adjust the repo path if it differs):

   ```bash
   REPO="$HOME/GitHub Projects/QuantaBench"
   mkdir -p "$HOME/Library/Logs/QuantaBench"
   cat > "$HOME/Library/LaunchAgents/io.github.brandonbtm29.quantabench-worker.plist" <<EOF
   <?xml version="1.0" encoding="UTF-8"?>
   <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
   <plist version="1.0"><dict>
     <key>Label</key><string>io.github.brandonbtm29.quantabench-worker</string>
     <key>ProgramArguments</key><array>
       <string>$REPO/.venv/bin/python</string><string>-m</string><string>quantabench.cli</string><string>worker</string>
     </array>
     <key>WorkingDirectory</key><string>$REPO</string>
     <key>EnvironmentVariables</key><dict>
       <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
     </dict>
     <key>RunAtLoad</key><true/>
     <key>KeepAlive</key><dict><key>SuccessfulExit</key><false/></dict>
     <key>ThrottleInterval</key><integer>30</integer>
     <key>StandardOutPath</key><string>$HOME/Library/Logs/QuantaBench/worker.log</string>
     <key>StandardErrorPath</key><string>$HOME/Library/Logs/QuantaBench/worker.log</string>
   </dict></plist>
   EOF
   launchctl bootstrap gui/$(id -u) "$HOME/Library/LaunchAgents/io.github.brandonbtm29.quantabench-worker.plist"
   ```

   Stop it with `launchctl bootout gui/$(id -u)/io.github.brandonbtm29.quantabench-worker` (it finishes the
   current molecule and exits). At login Tailscale may not be up yet: the worker waits up to 120 s
   (`--tailscale-wait`), and launchd starts it again if it gives up.
4. **Sleep:** while a job runs the worker keeps `caffeinate -i` going, so the Mac does not idle-sleep. Closing
   the lid still sleeps it unless it is on power with an external display. To keep the worker reachable when
   idle, enable *Prevent automatic sleeping on power adapter when the display is off* (System Settings,
   Battery, Options). A grid treats a sleeping Mac as away and moves its work (Pausing a machine, a closing laptop,
   above); on battery the worker pauses itself (`pause_on_battery`, on by default). Prefer this LaunchAgent to a
   `nohup` start over ssh: a restart or logout ends a `nohup` worker (that is how it was found gone on
   2026-11-09), and launchd starts it again at login.

## From the app: Run on (UI-5b)

The job dialog (Build > New job..., Resources) has **Run on**: this computer, **Auto**, or a worker.
- **The worker list:** the dialog asks every known worker whether it answers. Those that answer, with the same
  QuantaBench version, can be chosen; the others are listed as unavailable, with the reason in their tooltip.
- **Sending:** a job on a worker is sent at once, as one share the way `grid build` sends one. The worker runs it with
  its own threads, GPU, priority and lanes.
- **Following:** the Jobs dock follows the job every few seconds (state, counts, the molecule running, the worker's
  log). Cancel sends the cancel to the worker.
- **The results:** when the job ends, its run folder comes back. That is the synced copy in this computer's output
  folder once its `worker.json` vouches for its tables, else the archive, downloaded into
  `<output folder>/from_workers/<worker>/`. The job's rows in the Molecules list then take the results.
- **Auto** picks the machine when the job is sent: the highest `grid build --weights auto` speed for the job's method
  per job the machine already has, this computer included. If this computer wins, the job runs in the local queue.
- **Closing the app** stops following; the job goes on on the worker, and its run folder can be opened later.

## Trying it

```bash
quantabench-cli grid machines                   # both workers, their versions and queues
quantabench-cli grid build --file pfas.csv --method gfn2-xtb --machines bench-pc,local --weights auto --wait
quantabench-cli grid status                     # recent grid runs
```

Both machines must run the same QuantaBench version (`git pull` on both, then restart the workers); a worker
refuses jobs from another version (`--allow-version-mismatch` on either side overrides).

## Limits

- One lane (the default) runs molecules in the worker process: a segfault in a native library ends the worker.
  launchd restarts it on the Mac; on Windows the Task Scheduler restart is unverified (see the setup section),
  so check `grid machines`. `jobs.py` marks a molecule failed after it has crashed the process twice. Several lanes (`--lanes`, see `dist.lanes`) run molecules in child processes.
- Work sharing, and moving a paused or away machine's molecules (DIST-7), need a `grid build` that follows the
  grid. Without `--wait` they only run while the local share does. A worker in one lane hands molecules over only between molecules, so a release can wait for
  the molecule in progress (up to 30 s, then it tries again at a later poll).
- The archive of a large run folder (cubes, trajectories) is a large download; prefer the synced folder.
