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
            (a worker's run folder that has synced here is read in place; otherwise it is downloaded as a zip)
```

## What is shared, and what is not

| Where | What |
|---|---|
| `~/SyncThing/QuantaBench/Settings/settings.json` (synced) | `dist.worker_token` (the shared secret), `dist.worker_port`, `dist.worker_bind`, `dist.workers` (name -> URL overrides) |
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
quantabench-cli grid machines [--worker NAME=URL]
quantabench-cli grid build <same inputs and science options as build> --method M
                       [--machines btm-nova,brandons-macbook-pro,local] [--weights auto|3,1] [--wait]
                       [--label L] [--worker NAME=URL] [--sync-wait 60]
quantabench-cli grid status [GRID_ID]          (no id: recent grid runs)
quantabench-cli grid cancel GRID_ID
quantabench-cli grid merge GRID_ID [--partial] (after a build without --wait)
```

- **`--method` must be explicit.** `auto` picks the engine per machine (AIMNet2 on NOVA's GPU, something
  else on the Mac), which would mix levels of theory in one table; `grid build` refuses it unless
  `--allow-auto-method`.
- **Weights:** `auto` uses each machine's measured throughput for the method (`throughput` in its profile,
  from the machine benchmark) and falls back to equal weights; `3,1` gives one weight per machine in the
  order of `--machines`. The split is deterministic (SHA-256 of the molecule id), so re-running a list puts
  every molecule on the same machine again.
- **Threads and GPU** come from each machine's own settings. `--threads`/`--gpu` on `grid build` override
  them everywhere.
- `local` runs its share inside the `grid build` process, so that command stays open until the local share
  is done (with `--wait`, until everything is done and merged). Ctrl+C stops following; remote shares keep
  running (`grid status`, then `grid merge`).
- A worker runs one job at a time (`--max-jobs`) and each job through `jobs.run_resumable`, so `--lanes` (or
  `settings.dist.lanes`) spreads one job over this machine's GPU and CPU lanes.
- Stopping a worker with Ctrl+C, Ctrl+Break or SIGTERM stops its job after the current molecule; a killed or
  crashed worker loses only the molecule in progress. Either way the next start resumes its unfinished jobs.

## API (version 1)

Every request needs `Authorization: Bearer <settings.dist.worker_token>`.

| Request | Answer |
|---|---|
| `GET /v1/info` | name, URL, QuantaBench version and commit, machine profile, queue (`running`, `queued`, `paused`) |
| `GET /v1/jobs` | every job, newest first |
| `POST /v1/jobs` | body `{inputs: [jobs.input_to_json(...)], config: {BuildConfig fields}, label, chunk?, lanes?, client?, allow_version_mismatch?}`; header `X-QuantaBench-Version`. 201 `{id, job}` |
| `GET /v1/jobs/<id>?tail=20` | state (`queued`, `running`, `done`, `failed`, `cancelled`, `interrupted`), counts from the run folder's job table, `run_dir`, `run_folder`, progress, log tail |
| `POST /v1/jobs/<id>/cancel` | a queued job is dropped; a running one stops after the current molecule (its run folder stays resumable) |
| `GET /v1/jobs/<id>/archive` | zip stream of the run folder (finished jobs; 409 while running) |
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
- The client never uses a proxy and never follows redirects (urllib would forward the Authorization header).
  Downloaded archives are extracted only after every member path is checked (no absolute paths, `..`, drive
  letters or links).
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
   progress is redone on the next start. While a job runs the worker asks Windows not to sleep.

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

Both machines must run the same QuantaBench version (`git pull` on both); a worker refuses jobs from another
version (`--allow-version-mismatch` on either side overrides).

## Limits

- One lane (the default) runs molecules in the worker process: a segfault in a native library ends the worker.
  The autostart above restarts it, and `jobs.py` marks a molecule failed after it has crashed the process
  twice. Several lanes (`--lanes`, see `dist.lanes`) run molecules in child processes.
- No work stealing: each share stays on the machine it was given to. If a machine is slower than its weight
  says, the others finish first and wait; measured throughput (`--weights auto`) keeps that small.
- The archive of a large run folder (cubes, trajectories) is a large download; prefer the synced folder.
