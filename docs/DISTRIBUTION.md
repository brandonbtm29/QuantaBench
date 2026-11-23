# Distribution: installers, updates and launchers (0.5 design, DIST-1)

Status 2026-11-23: a design for Brandon's OK. **PR 1** (DIST-1, built) gives today's git checkouts an Update button
and terminal-free launchers on all three OSes. **PR 2**, the installers built in CI on tags, waits for the OK on the
recommendation below. ROADMAP.md, "0.5: Distribution", lists the goals.

## What has to be installed

| Layer | What | Where it comes from today |
|---|---|---|
| QuantaBench | pure Python (`src/quantabench`), the app icon (`src/quantabench/assets`) | git checkout, `pip install -e` |
| Core packages | PySide6, VTK/PyVista, RDKit, NumPy/SciPy, ASE, psutil, platformdirs | PyPI wheels (all three OSes) |
| GPU layer | PyTorch: CPU, CUDA, ROCm or Apple's MPS build | Windows: PyPI's CPU build, AMD's ROCm 7.2.1 wheels (Python 3.12 only, `.venv-rocm`), or NVIDIA's index. Linux: download.pytorch.org (CPU, CUDA, ROCm). macOS: PyPI (CPU + MPS) |
| ML and xTB in Python | aimnet, the dxtb fork (a git tag), tad-libcint, tblite (PyPI on Linux and macOS; the fork's wheels from a release page on Windows) | `scripts/fork-packages.json`, read by the installers and by Update |
| External programs | xtb, CREST, MOPAC, Psi4 (conda-forge), g-xTB (pinned by URL and SHA-256), LAMMPS (the Windows installer's `lmp.exe`, conda-forge on the Mac) | `envs/*.yml` with Miniforge (`--with-engines`), CREST through WSL on Windows |

Three things make QuantaBench harder to package than a plain Qt app:

1. **PyTorch comes in builds**, and the right one depends on the GPU (ROCm on the workstation's Radeon, MPS on the
   Mac, CUDA on an NVIDIA machine, CPU elsewhere). AMD's Windows ROCm wheels exist only on AMD's own index, for
   Python 3.12. A ROCm or CUDA build is gigabytes.
2. **QuantaBench starts Python processes of its own.** Lanes run molecules in worker processes (`dist/lanes.py`),
   the CPCM worker runs `sys.executable -m quantabench.chem.cpcm_worker`, and Update relaunches the app. They expect
   a real interpreter with QuantaBench importable.
3. **The engines are not pip packages**, Psi4 lives in a conda environment of its own with its own pins
   (`envs/psi4.yml`), and CREST has no native Windows build yet (FORK-5).

## Today (PR 1): git checkouts

- `scripts/install_{windows.ps1,macos.sh,linux.sh}` clone-and-install: `.venv` (and `.venv-rocm` on Windows with
  `-Gpu rocm`), PyTorch first, `pip install -e ".[all,dev]"` plus the fork packages, optional engines, launchers.
- **Help > Check for updates...** and `quantabench-cli update` follow **main** (why: below). They fast-forward only,
  never force, and refuse with local changes, on another branch or with commits of the checkout's own. They show the
  new CHANGELOG.md sections, reinstall changed dependencies the installers' way (installed extras, fork packages,
  PyTorch pinned) into every environment of the checkout, go back to the old commit if pip fails, wait while this
  window or this computer's worker has work, and offer to restart the app and the worker. Details: INSTALL.md,
  "Updating".
- **Launchers** (`quantabench-cli shortcuts`, Help > Create app shortcuts..., and the installers): Start menu
  shortcuts to `pythonw.exe -m quantabench` with the icon and the app's taskbar ID on Windows,
  `~/Applications/QuantaBench.app` on macOS, `quantabench.desktop` on Linux. One per environment
  (`QuantaBench`, `QuantaBench (AMD GPU)`).

**Why main and not releases.** Every pull request is merged into main after CI passes, several a day; releases are
tags on main a few weeks apart (0.3.0 on 2026-11-18, main is 100+ commits past it a week later). Brandon's machines
run fixes the day they land, and machines that share a batch should run the same commit (`grid build` warns when
they differ). Brandon's other apps (Capillary-Flow-Porometry-Analyzer, Kalends and the lab apps) update to the
newest release tag instead; that is the right model for the **installed** builds below, which get no unreleased
work. The dialog shows how far main is past the newest release.

## The options for 0.5

### A. conda `constructor` installers built in CI

One installer per OS (`.exe`, `.pkg` or `.sh`; constructor builds only for the OS it runs on, so three CI runners)
holding conda packages: Python, Qt, VTK, RDKit and the engines from conda-forge, shortcuts through menuinst.
napari ships this way and dropped its Briefcase installers for it: conda bundles the system libraries, so the
installer "makes very few assumptions about the underlying system installation" ([napari packaging][napari]).
Constructor signs Windows installers with SignTool or AzureSignTool and macOS PKGs with an installer identity plus
an application identity for notarization ([constructor how-to][constructor]).

- **For:** offline install with the engines in one file; a proven path for a scientific Qt app; signing built in.
- **Against:**
  - QuantaBench, the dxtb fork, the tblite Windows build and aimnet would all need conda packages of our own
    (a channel to build and keep up).
  - AMD's ROCm PyTorch for Windows is pip-only, so the ROCm variant needs a pip step after the install anyway.
  - The installer is gigabytes with PyTorch: one per GPU variant, or one without PyTorch plus a pip step.
  - Updates mean `conda update` in the installed environment or a new full installer. No small "pull" update.
  - Psi4 still needs a second environment.

### B. A frozen app: PyInstaller or Briefcase

- **PyInstaller** freezes the interpreter and packages into one folder (`.exe` / `.app`).
  - QuantaBench imports torch, aimnet, dxtb and ovito lazily on purpose. Every lazy import becomes a hidden-import
    rule to maintain.
  - A frozen `sys.executable` is the app, not Python, so lanes, the CPCM worker and `-m` subprocesses need
    special entry points.
  - A frozen app cannot `pip install`: no new fork build, no optional extra and no PyTorch variant without a new
    bundle. Each GPU variant is a separate multi-gigabyte download, and the engines stay outside.
  - Updates replace the whole bundle.
- **Briefcase** bundles a real standalone Python plus pip wheels, and makes MSI, `.app`/`.dmg` (with signing and
  notarization) and AppImage/Flatpak.
  - Its real interpreter keeps `-m` subprocesses working, and its signing is built in.
  - But it takes wheels only (no engines), still needs a build per PyTorch variant, and has no updater.
  - napari moved off it.

### C. A small bootstrap installer that builds the environment (recommended)

A light installer per OS, built in CI on each `v*` tag, does what the install scripts already do, without git:

1. Put a private Python 3.12 in the install folder. `uv` can download a standalone CPython in seconds; the
   installers' current "find Python 3.12" step is the fallback.
2. Create the environment and install **the release's wheel** (built in the same CI run and attached to the GitHub
   release), the fork packages from `scripts/fork-packages.json`, and the PyTorch build the GPU wizard chose.
3. Make the launchers with the same code as PR 1 (`quantabench.launchers`): Start menu shortcut, `.app`,
   `.desktop`.
4. Optional components, as check boxes: the engines (micromamba, a single binary, creates `qb-tools` and
   `qb-psi4` from `envs/*.yml` in the install folder) and a ROCm or CUDA environment.

Per OS:

- **Windows:** an Inno Setup `.exe` (a few MB; per-user install in `%LOCALAPPDATA%\Programs\QuantaBench`, no admin
  rights). It carries `uv.exe`, `micromamba.exe` and the bootstrap script.
- **macOS:** a signed and notarized `QuantaBench.app` in a `.dmg`. On first start the app builds its environment in
  `~/Library/Application Support/QuantaBench`. Before signing exists, the same `.app` with instructions.
- **Linux:** an install script (`curl ... | sh`) or an AppImage of the bootstrap. A conda-forge recipe later, if
  anyone asks.

Why C:

- **It reuses what works.** The install scripts, `fork-packages.json`, the update logic (extras, PyTorch pin, fork
  reinstall) and the launchers are already tested on the workstation, the MacBook and in CI.
- **Real Python and pip.** Lanes, the CPCM worker, Update and optional extras keep working unchanged.
- **PyTorch stays a choice made on the machine** (the wizard), not a matrix of installers.
- **Small download, small updates:** an update is one wheel plus changed packages, not gigabytes.
- **The engines are optional and stay conda's** (they are today).

The cost: the first install needs the internet (PyPI, the PyTorch index, GitHub), and two installs a month apart
can resolve different package versions. Two remedies: pin a lock file per release (`uv pip compile` in CI, CPU
variant), and test the installer in CI on every tag. If an offline or IT-managed install is ever needed (a lab PC
without internet), option A remains: the wheel built for C is what a conda recipe would package.

## Updates of installed builds

- **Channel: releases.** Help > Check for updates asks GitHub for the newest `vX.Y.Z` release (the API, with
  `git ls-remote --tags` as fallback, as Brandon's lab apps do). It shows the release notes, or the CHANGELOG.md
  sections for that version (Porometry's `notes_since`).
- **Update:** download the wheel and run pip in the app's environment with the same rules as today: installed
  extras, fork packages, PyTorch pinned with a constraints file. If it fails, the environment goes back
  (`pip install` of the previous wheel), and the app restarts.
- **Windows locks the files of a running program.** The upgrade runs in a helper process after the window closes
  (Kalends' `update_helper.py` pattern: a copy of a small stdlib script stops the app, installs, starts the new
  version and rolls back if it does not come up). This also closes UPD-1's open item for in-app upgrades on
  Windows.
- **A checkout keeps following main.** The "release" channel applies to installs without `.git`; `update.check`
  already tells them apart ("not-a-checkout"). A setting can switch a tester's install to "main" (git) later.
- **Workers:** the same restart as PR 1 (LaunchAgent, scheduled task). `grid build` compares versions instead of
  commits between installed machines.

## The first-run GPU wizard

When the app starts in an environment without PyTorch, or from Settings > Compute > "GPU setup...":

1. **Detect.** `resources.detect_isolated` (in a child process, as the hardware report does already) reads the
   GPU: vendor, model, architecture (gfx1200 for the RX 9060 XT), VRAM and the driver.
2. **Recommend:**
   - NVIDIA: CUDA (pytorch.org's index).
   - AMD on Windows: ROCm, for the GPUs AMD's wheels support (RDNA3/RDNA4), as `.venv-rocm` does today.
   - AMD on Linux: ROCm (pytorch.org).
   - Apple Silicon: PyPI's build (MPS).
   - Otherwise: CPU.

   It shows the download size and says what each choice enables (AIMNet2 and dxtb on the GPU), with a link to
   docs/PLATFORMS.md.
3. **Install** into a second environment, `env-rocm` or `env-cuda`, with progress, so the CPU environment stays as
   a fallback. Then a second launcher (`QuantaBench (AMD GPU)`), as PR 1 makes for `.venv-rocm`.
4. **Check** that a tensor runs on the device and an AIMNet2 single point works (what `quantabench-cli doctor`
   reports), then save the choice in settings (per machine).

Skip and later are always possible. CPU-only works for everything but GPU refinement.

## Where installs and settings live

| | Dev checkout (today) | Installed build (0.5) |
|---|---|---|
| Code | the git checkout (main) | Windows `%LOCALAPPDATA%\Programs\QuantaBench`, macOS `~/Library/Application Support/QuantaBench/app`, Linux `~/.local/share/quantabench` |
| Python environments | `.venv`, `.venv-rocm` in the checkout | `env`, `env-rocm` / `env-cuda` in the install folder |
| Settings | `settings_location.json` in the checkout points at `~/SyncThing/QuantaBench/Settings` | the same pointer in the platform's config folder (`settings.set_settings_dir` already falls back there without a checkout) |
| Worker state | `platformdirs` state folder (`worker/`) | the same |
| Runs | the output folder in settings | the same |
| Engines | conda environments (`~/miniforge3/envs/qb-tools`, `qb-psi4`), CREST in WSL | micromamba environments in the install folder, or the existing ones (engines are found by path, as today) |
| Launchers | `quantabench-cli shortcuts` | made by the installer, same code |

Uninstall removes the install folder and the launchers. It never removes settings, runs or worker state.

## How the dev checkouts carry on

- The workstation, the MacBook and the lab PC keep their git checkouts, following main. Help > Check for updates
  works there now (PR 1). Clones kept in step from another computer (commits pushed over SSH) follow the same rules
  as the app: never force, skip a clone with local changes, report dependency changes.
- `quantabench-cli shortcuts` (or Help > Create app shortcuts...) gives the existing checkouts the icon launchers
  once. Nothing else changes for them. The installers will not touch a folder that has `.git`.
- If an installed build ever runs on the same machine as a checkout, it needs its own name and IDs
  ("QuantaBench (release)") so that their Start menu entries and taskbar groups stay apart.

## In-app updates and the cross-machine update script

Brandon keeps a script on his hub computer, `Update-OtherComputers.ps1`, outside this repository. It updates his
apps "across the board" from one list he can extend; QuantaBench is one entry. The script and the in-app update
are two tools for one job, and they follow the same rules.

| | In the app / `quantabench-cli update` | The cross-machine script |
|---|---|---|
| Runs | on the computer it updates; it fetches from GitHub itself | on the hub, after a push; it pushes main into each other computer's clone over SSH (their git cannot log in inside SSH) |
| Moves the checkout | `git pull --ff-only` | fast-forward only |
| Never | forces, stashes or resets local work (`git reset --keep` only takes back its own pull when pip fails) | forces, stashes or resets |
| Local changes, another branch, commits not on GitHub | refuses and says why | skips that clone and lists it |
| A computer that is off or asleep | (not running) | skipped and listed |
| Dependency changes (`pyproject.toml`, `scripts/fork-packages.json`) | installs them into every environment of the checkout | reports them |
| Running programs | waits for the window's jobs and this computer's worker; then offers to restart the app and the worker | starts nothing over SSH; running programs keep the old code until they restart |

**Where they meet.** A clone that the script fast-forwarded has the new code, but its environments still have the
old dependencies. The app's next check finds nothing to pull, so it never sees that change: it only compares what
its own pull brings. The repository's own way to finish that computer is one command, run there with any of its
environments:

```
quantabench-cli update --reinstall --restart-worker
```

- `--reinstall` runs the installers' pip line (installed extras, fork packages, PyTorch kept) in every environment
  of the checkout, with nothing to pull.
- `--restart-worker` then restarts that computer's worker, if it runs this checkout, is idle and is kept by the
  LaunchAgent or the scheduled task.
- The command refuses (exit 1, with the reason) while that worker runs or is about to run a job from the
  checkout, so the script can list the computer as skipped. It never starts the GUI.
- Rerunning the install script for the OS does the same reinstall, more slowly.

So QuantaBench's entry in the script's list needs:

- the dependency files: `pyproject.toml` and `scripts/fork-packages.json`;
- the install command: `<.venv>/python -m quantabench update --reinstall` (add `--restart-worker` to restart an
  idle worker);
- the rule that GUIs are never started over SSH (the same rule as for every app on the list).

Each tool covers a different case:

- **In-app update:** one computer, by hand, when Brandon sits at it. It installs and restarts there.
- **The script:** every computer at once, after a push from the hub.
- **Installed builds (0.5)** are not git clones. The script does not find them (it finds clones by their origin
  URL); they follow releases through the app.

## Code signing and notarization (later)

- **Windows.** An unsigned installer shows SmartScreen's "Windows protected your PC". Where Smart App Control is on,
  unsigned binaries are blocked outright (INSTALL.md, "Smart App Control").
  - Microsoft's Trusted Signing (now Artifact Signing) is open to individual developers, after a photo-ID and
    selfie check, "starting at $9.99/month" ([announcement][trusted]). Some users report it limited to the US and
    Canada.
  - constructor and Inno Setup can both call its signing tool.
  - Sign the installer and the launcher stub. The fork's tblite DLLs (our build) should be signed too, or Smart App
    Control may block them.
- **macOS.**
  - A downloaded app that is not signed with a Developer ID and notarized is quarantined. Since macOS Sequoia,
    Control-click no longer overrides Gatekeeper: the user has to allow the app in System Settings > Privacy &
    Security ([Apple][apple]).
  - Signing needs the Apple Developer Program ($99 a year), then `codesign` with the hardened runtime,
    `notarytool` and `stapler`, all scriptable in CI with the certificate kept in a GitHub secret.
  - The launcher that PR 1 makes on the Mac itself is not quarantined (it was not downloaded), which is why it
    needs no signing.
- **Linux.** No signing needed. Publish SHA-256 sums with each release.

## PR 2: the installers, built in CI on tags (plan)

1. `.github/workflows/release.yml`, on `push: tags: ["v*"]` and by hand (`workflow_dispatch`, a dry run without
   publishing). Matrix: windows-latest, macos-15 (Apple Silicon), ubuntu-latest.
2. Build the wheel and sdist (hatch) once. Build the bootstrap per OS: Inno Setup on Windows, a `.app` in a
   `.dmg` on macOS, the script or AppImage on Linux.
3. Smoke-install each in CI: CPU PyTorch, no engines, then `quantabench-cli doctor` and an offscreen GUI start.
4. A draft GitHub release with the installers, the wheel, `SHA256SUMS` and the CHANGELOG section. Brandon
   publishes it.
5. Updater for installed builds (the release channel above), the GPU wizard and the helper-process upgrade on
   Windows.
6. Signing and notarization once the accounts exist (the steps are no-ops without the secrets).

### Open questions for Brandon

1. Is C right: a bootstrap installer with a wheel and a private Python, not constructor or a frozen app?
2. Per-user installs (no admin rights, `%LOCALAPPDATA%\Programs`, `~/Library/Application Support`): OK?
3. GPU wizard on Windows with a supported Radeon: install ROCm by default, or ask every time? CPU always stays.
4. Engines in the installer: opt-in check boxes (the conda environments are large, Psi4's especially), or a
   separate "Install engines..." in the app?
5. Signing: when to pay for Trusted Signing ($9.99/month) and the Apple Developer Program ($99/year)? Until then
   the installers are unsigned and the docs explain SmartScreen and Gatekeeper.
6. Release cadence: installed builds see only releases. Tag more often (every one or two weeks)?
7. Should an installed build be able to switch to "main" (testers), or stay release-only?

[napari]: https://melissawm.github.io/napari.github.io/0.4.17/developers/packaging.html
[constructor]: https://conda.github.io/constructor/howto
[trusted]: https://techcommunity.microsoft.com/blog/microsoftsecurityandcompliance/-/4273554
[apple]: https://developer.apple.com/news/?id=saqachfa
