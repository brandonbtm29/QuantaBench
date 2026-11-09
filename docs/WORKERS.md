# Workers: one batch on several machines

`quantabench-cli worker` runs QuantaBench jobs for your other machines. `quantabench-cli grid build`
splits a molecule list across the workers (and, if you like, this machine), follows the progress, and
merges the result tables into one folder. Plan and background: [PLATFORMS.md](PLATFORMS.md), section 3;
code: `src/quantabench/dist/worker.py` and `client.py`.

```
Mac: quantabench-cli grid build --file pfas.csv --method gfn2-xtb --machines btm-nova,local --wait
        |  split by weight (dist.plan), the same BuildConfig for every share
        |--- HTTP/JSON over Tailscale ---> btm-nova worker -> ~/SyncThing/QuantaBench/Runs/<date>_<label>_btm-nova
        |--- in this process ------------> local share     -> ~/SyncThing/QuantaBench/Runs/<date>_<label>_<mac>
        '-- when all are done: summary.csv + descriptors.csv in Runs/<date>_grid_<label>/
            (a worker's run folder that has synced here completely is read in place; otherwise it is
             downloaded as a zip)
```

## What is shared, and what is not

| Where | What |
|---|---|
| `~/SyncThing/QuantaBench/Settings/settings.json` (synced) | `dist.worker_token` (the shared secret), `dist.worker_port`, `dist.worker_bind`, `dist.workers` (name -> URL overrides), `dist.machine_lanes` (each machine's own lane spec, by machine name) and `dist.lanes` (the default for a machine without one) |
| `~/SyncThing/QuantaBench/Settings/machines/<name>.json` (synced) | each machine's profile; a running worker writes its URL into `address`, which is how `grid` finds it |
| `~/SyncThing/QuantaBench/Runs/` (synced) | run folders. Each one has a single writer: the worker that runs it |
| Per-machine state folder (not synced) | the worker's job list and queued inputs: `%LOCALAPPDATA%\QuantaBench\worker` on Windows, `~/Library/Application Support/QuantaBench/worker` on macOS (`--state-dir` to change) |

The token is generated the first time a worker starts (`secrets.token_urlsafe(32)`) and saved in
`settings.json`. Start the first worker on one machine and let SyncThing copy the settings to the other
before starting the second, or both will generate a token and SyncThing will report a conflict.

## Commands

```
quantabench-cli worker [--bind tailscale|localhost|ADDR] [--port 8765] [--lanes SPEC] [--max-jobs 1]
                       [--name NAME] [--state-dir DIR] [--log-file FILE] [--tailscale-wait 120]
                       [--allow-any-address] [--allow-version-mismatch]
                       [--priority normal|low|background] [--free-threads N]
quantabench-cli grid machines [--worker NAME=URL]
quantabench-cli grid build <same inputs and science options as build> --method M
                       [--machines btm-nova,brandons-macbook-pro,local] [--weights auto|3,1] [--wait]
                       [--no-share-work] [--label L] [--worker NAME=URL] [--sync-wait 60]
quantabench-cli grid status [GRID_ID]          (no id: recent grid runs)
quantabench-cli grid cancel GRID_ID
quantabench-cli grid merge GRID_ID [--partial] (after a build without --wait)
```

- **`--method` must be explicit.** `auto` picks the engine per machine (AIMNet2 on NOVA's GPU, something
  else on the Mac), which would mix levels of theory in one table; `grid build` refuses it unless
  `--allow-auto-method`.
- **Weights** follow `build --chunk-weights` (`dist.plan`): `auto` uses each machine's measured throughput
  for the method (`throughput` in the profile its worker reports, else the registry copy), an estimate from
  its core count when it has no benchmark for that method, and 0 where the method is unavailable; a machine
  with no profile at all gets the mean of the others. `3,1` gives one weight per machine in the order of
  `--machines`; `btm-nova=3,local=1` names them. The split is deterministic (SHA-256 of the molecule id), so
  re-running a list puts every molecule on the same machine again.
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
  program locked, and pip cannot replace a package the worker has loaded. Updating workers from the client
  is planned (BACKLOG UPD-1).
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

## API (version 1)

Every request needs `Authorization: Bearer <settings.dist.worker_token>`.

| Request | Answer |
|---|---|
| `GET /v1/info` | name, URL, QuantaBench version and commit, machine profile, queue (`running`, `queued`, `paused`), and since MON-1 the latest sensor sample (`sensors`, without the per-CPU lists; null before the first) |
| `GET /v1/sensors` | since MON-1: live machine metrics `{sensors: a sensors.Sample dict or null, why, interval_s, machine}` (CPU and GPU load, effective clock, temperatures, power, VRAM, RAM; docs/ENGINES.md, Live machine metrics). The worker samples while requests keep coming and stops after 2 minutes without one; older workers answer 404 |
| `GET /v1/jobs` | every job, newest first |
| `POST /v1/jobs` | body `{inputs: [jobs.input_to_json(...)], config: {BuildConfig fields}, label, chunk?, lanes?, client?, allow_version_mismatch?}`; header `X-QuantaBench-Version`. 201 `{id, job}` |
| `GET /v1/jobs/<id>?tail=20` | state (`queued`, `running`, `done`, `failed`, `cancelled`, `interrupted`), counts from the run folder's job table, `run_dir`, `run_folder`, progress, log tail |
| `POST /v1/jobs/<id>/cancel` | a queued job is dropped; a running one stops after the current molecule (its run folder stays resumable) |
| `POST /v1/jobs/<id>/release` `{"max": N, "to": "machine"}` | work sharing: up to N molecules the running job has not started, from the end of its queue, become `moved` there; 200 `{ids, job}` (`ids` empty when the job is not running or has none left). The batch answers between molecules: within 0.1 s with lanes, after the current molecule in one process (the request gives up after 30 s and then moves nothing) |
| `GET /v1/jobs/<id>/archive` | zip stream of the run folder (finished jobs; 409 while running); its last member, `<run folder>/.qb-archive.json`, lists every other member and its size |
| `POST /v1/pause` `{"paused": true}` | running jobs finish, new ones wait (keeps the machine responsive) |

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

## Setup on BTM NOVA (Windows 11)

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

2. **First start, by hand** (generates the token and publishes `machines/btm-nova.json`):

   ```powershell
   cd "C:\Users\Brand\GitHub Projects\QuantaBench"
   .venv\Scripts\quantabench-cli worker
   ```

   It should print `worker 'btm-nova' at http://100.90.226.110:8765`. Ctrl+C stops it.

   **A machine that joins is measured once.** On its first start (no throughput in its profile yet) the
   worker runs the throughput benchmark before it takes jobs: molecules/hour per installed engine on the CPU
   and each GPU, a few minutes, with submitted jobs held (`grid machines` shows it as benchmarking). The result
   goes into its profile, so `grid build --weights auto` gives it the right share. Later starts skip it;
   `--bench always` measures again (after a hardware or PyTorch change), `--bench never` never does. The GUI's
   Machines dock has the same as a button, "Benchmark this computer".

3. **Start at login** (normal PowerShell; no console window, log in `%LOCALAPPDATA%\QuantaBench\worker.log`):

   ```powershell
   $repo = "C:\Users\Brand\GitHub Projects\QuantaBench"
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
   documented for a task that *fails*, and it is not clear (not tested on NOVA) that a worker exiting with an
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
2. **First start, by hand** (after SyncThing has brought `settings.json` with the token from NOVA):

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
   Battery, Options).

## Trying it

```bash
quantabench-cli grid machines                   # both workers, their versions and queues
quantabench-cli grid build --file pfas.csv --method gfn2-xtb --machines btm-nova,local --weights auto --wait
quantabench-cli grid status                     # recent grid runs
```

Both machines must run the same QuantaBench version (`git pull` on both, then restart the workers); a worker
refuses jobs from another version (`--allow-version-mismatch` on either side overrides).

## Limits

- One lane (the default) runs molecules in the worker process: a segfault in a native library ends the worker.
  launchd restarts it on the Mac; on Windows the Task Scheduler restart is unverified (see the setup section),
  so check `grid machines`. `jobs.py` marks a molecule failed after it has crashed the process twice. Several lanes (`--lanes`, see `dist.lanes`) run molecules in child processes.
- Work sharing needs a `grid build` that follows the grid. Without `--wait` it only runs while the local
  share does. A worker in one lane hands molecules over only between molecules, so a release can wait for
  the molecule in progress (up to 30 s, then it tries again at a later poll).
- The archive of a large run folder (cubes, trajectories) is a large download; prefer the synced folder.
