#!/usr/bin/env bash
# Install or update QuantaBench on macOS (Apple Silicon). Safe to rerun: every step that is already done is
# skipped. Written for the bash 3.2 that ships with macOS.
#
#   bash scripts/install_macos.sh [options]
#
# Options:
#   --with-engines       also set up the external programs: Miniforge (only if no conda, mamba or
#                        micromamba is found), then the conda environments envs/tools.yml (xtb, CREST,
#                        MOPAC) and envs/psi4.yml (Psi4), created or updated
#   --settings-dir DIR   keep settings in DIR, e.g. ~/SyncThing/QuantaBench/Settings
#   --gpu auto|mps|cpu   accepted for symmetry with the other installers: PyPI's macOS PyTorch is one build
#                        that runs on the CPU and on the Apple GPU (mps)
#   --torch-index URL    install PyTorch from this package index instead (e.g. a nightly)
#   --replace-torch      reinstall PyTorch from --torch-index even though one is installed
#   --python PATH        the Python (3.12 recommended) to create .venv with
#   --conda PATH         the conda, mamba or micromamba to use for --with-engines
#   --no-app             don't create ~/Applications/QuantaBench.app
#   --reinstall          run `pip install -e` even when pyproject.toml has not changed
#   --dry-run            print every action without doing it
#   -h, --help           show this help
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
VENV="$REPO/.venv"
VPY="$VENV/bin/python"
EXTRAS="all,dev"
# tblite's newest wheels need macOS 15; never build it from source (an older wheel is used on macOS 14)
PIP_FLAGS="--only-binary tblite"
MINIFORGE_DIR="$HOME/miniforge3"
MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
APP="$HOME/Applications/QuantaBench.app"

DRY_RUN=0 WITH_ENGINES=0 SETTINGS_DIR="" GPU="auto" TORCH_INDEX="" REPLACE_TORCH=0 PYTHON="" CONDA=""
MAKE_APP=1 REINSTALL=0

# ------------------------------------------------------------------------------------------ helpers
usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; }
step() { printf '\n==> %s\n' "$*"; }
info() { printf '    %s\n' "$*"; }
ok() { printf '    [ok] %s\n' "$*"; }
warn() { printf '    [!] %s\n' "$*" >&2; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }
quote() {                                 # one argument, quoted for copy-pasting into a shell
  case "$1" in
    "" | *[!A-Za-z0-9_./=:,+@%-]*) printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")" ;;
    *) printf '%s' "$1" ;;
  esac
}
show() { printf '    $'; for a in "$@"; do printf ' %s' "$(quote "$a")"; done; }
run() {                                   # print a command (copy-pastable), then run it unless --dry-run
  show "$@"
  printf '\n'
  if [ "$DRY_RUN" = 1 ]; then return 0; fi
  "$@"
}
run_quiet() {                             # like run, without the command's standard output
  show "$@"
  printf ' >/dev/null\n'
  if [ "$DRY_RUN" = 1 ]; then return 0; fi
  "$@" >/dev/null
}
write_file() {                            # write_file PATH CONTENT: only when the content differs
  local path=$1 content=$2
  if [ -f "$path" ] && [ "$(cat "$path")" = "$content" ]; then
    ok "$path is up to date"
    return 0
  fi
  if [ "$DRY_RUN" = 1 ]; then
    info "would write $path:"
    printf '%s\n' "$content" | sed 's/^/        | /'
    return 0
  fi
  mkdir -p "$(dirname "$path")"
  printf '%s\n' "$content" > "$path"
  info "wrote $path"
}
make_executable() { if [ -x "$1" ]; then return 0; fi; run chmod +x "$1"; }
py_version() { "$1" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null; }
sha256() { "$ANYPY" -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$1"; }
need_value() { [ $# -ge 2 ] && [ -n "$2" ] || die "$1 needs a value (see --help)"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --with-engines) WITH_ENGINES=1 ;;
    --settings-dir) need_value "$@"; SETTINGS_DIR=$2; shift ;;
    --settings-dir=*) SETTINGS_DIR=${1#*=} ;;
    --gpu) need_value "$@"; GPU=$2; shift ;;
    --gpu=*) GPU=${1#*=} ;;
    --torch-index) need_value "$@"; TORCH_INDEX=$2; shift ;;
    --torch-index=*) TORCH_INDEX=${1#*=} ;;
    --replace-torch) REPLACE_TORCH=1 ;;
    --python) need_value "$@"; PYTHON=$2; shift ;;
    --python=*) PYTHON=${1#*=} ;;
    --conda) need_value "$@"; CONDA=$2; shift ;;
    --conda=*) CONDA=${1#*=} ;;
    --no-app) MAKE_APP=0 ;;
    --reinstall) REINSTALL=1 ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1 (see --help)" ;;
  esac
  shift
done

case "$GPU" in
  auto|mps|cpu) ;;
  cuda|rocm) die "--gpu $GPU is not available on macOS: PyTorch uses the Apple GPU (mps) here" ;;
  *) die "--gpu must be auto, mps or cpu on macOS (got '$GPU')" ;;
esac
[ "$(uname -s)" = Darwin ] || die "this is the macOS installer; on Linux or WSL use scripts/install_linux.sh"
unset PYTHONPATH                          # the checks below must see the venv, nothing else
cd "$REPO"

printf 'QuantaBench installer (macOS)%s\n' "$([ "$DRY_RUN" = 1 ] && echo ': dry run, nothing is changed')"
info "repository: $REPO"
MACOS=$(sw_vers -productVersion 2>/dev/null || echo 0)
MACOS_MAJOR=${MACOS%%.*}
info "macOS $MACOS on $(uname -m)"
if [ "$(uname -m)" != arm64 ]; then
  warn "PyTorch has no Intel-Mac builds any more, so AIMNet2 and dxtb cannot be installed here; this installer targets Apple Silicon"
fi
if [ "$MACOS_MAJOR" -lt 14 ] 2>/dev/null; then
  warn "PyTorch's macOS wheels need macOS 14 or newer"
fi

# ------------------------------------------------------------------------------------- 1. Python
step "Python"
PY=""
ANYPY=""
if [ -x "$VPY" ]; then
  v=$(py_version "$VPY") || die "$VENV is broken (its Python no longer runs, e.g. after \`brew upgrade\`).
Remove it with:  rm -rf \"$VENV\"   and run this script again."
  ok ".venv exists (Python $v); no system Python needed"
  ANYPY=$VPY
else
  if [ -n "$PYTHON" ]; then
    v=$(py_version "$PYTHON") || die "--python $PYTHON does not run"
    case "$v" in 3.1[1-9]) PY=$(command -v "$PYTHON") ;; *) die "--python $PYTHON is Python $v; QuantaBench needs 3.11 or newer (3.12 recommended)" ;; esac
  else
    for c in python3.12 /opt/homebrew/bin/python3.12 /usr/local/bin/python3.12 \
             /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 python3; do
      if command -v "$c" >/dev/null 2>&1 && [ "$(py_version "$c")" = 3.12 ]; then
        PY=$(command -v "$c")
        break
      fi
    done
  fi
  if [ -z "$PY" ]; then
    msg="Python 3.12 was not found (the python3 that comes with macOS is older). Install it, then run this script again:
    Homebrew:   brew install python@3.12
    python.org: the macOS installer for the newest 3.12.x from https://www.python.org/downloads/macos/
  Or pass another Python 3.11+ with --python PATH."
    [ "$DRY_RUN" = 1 ] || die "$msg"
    warn "$msg"
    PY=python3.12
  else
    ok "found $PY (Python $(py_version "$PY"))"
  fi
  ANYPY=$PY
fi
command -v "$ANYPY" >/dev/null 2>&1 || ANYPY=python3

# ------------------------------------------------------------------------- 2. virtual environment
step "Virtual environment (.venv)"
if [ ! -x "$VPY" ] && [ -e "$VENV/pyvenv.cfg" ]; then
  die "$VENV is a virtual environment for another system (no bin/python). Leave it alone and install from a separate clone."
fi
if [ -x "$VPY" ]; then
  ok "$VENV"
else
  run "$PY" -m venv "$VENV"
  run "$VPY" -m pip install --quiet --upgrade pip
fi

# ------------------------------------------------------------------------------------- 3. PyTorch
step "PyTorch (one macOS build: CPU and Apple GPU)"
torch_version() { [ -x "$VPY" ] && "$VPY" -c 'import importlib.metadata as m; print(m.version("torch"))' 2>/dev/null || true; }
install_torch() {
  if [ -n "$TORCH_INDEX" ]; then run "$VPY" -m pip install torch --index-url "$TORCH_INDEX"
  else run "$VPY" -m pip install torch
  fi || die "PyTorch did not install (output above)"
}
TV=$(torch_version)
if [ -z "$TV" ]; then
  install_torch
elif [ -n "$TORCH_INDEX" ] && [ "$REPLACE_TORCH" = 1 ]; then
  info "replacing torch $TV with the one from $TORCH_INDEX (--replace-torch)"
  run "$VPY" -m pip uninstall -y torch
  install_torch
else
  ok "torch $TV"
  [ -z "$TORCH_INDEX" ] || info "--torch-index is ignored while a torch is installed; add --replace-torch to reinstall"
fi

# ---------------------------------------------------------------------------------- 4. QuantaBench
step "QuantaBench (pip install -e \".[$EXTRAS]\")"
STAMP="$VENV/.quantabench-install"
want=""
if [ -x "$VPY" ]; then
  want="$(sha256 "$REPO/pyproject.toml") extras=$EXTRAS $PIP_FLAGS"
fi
installed_here() { "$VPY" -c 'import os, sys, quantabench; sys.exit(0 if os.path.realpath(quantabench.__file__).startswith(os.path.realpath(sys.argv[1]) + os.sep) else 1)' "$REPO/src" 2>/dev/null; }
if [ "$REINSTALL" = 0 ] && [ -n "$want" ] && [ -f "$STAMP" ] && [ "$(cat "$STAMP")" = "$want" ] && installed_here; then
  ok "installed from this checkout; pyproject.toml unchanged since the last install"
else
  CONSTRAINTS="$VENV/.torch-constraint.txt"
  TV=$(torch_version)
  # shellcheck disable=SC2086  # PIP_FLAGS is a list of words
  if [ -n "$TV" ]; then
    # keep the installed PyTorch: an extra that wants another torch fails loudly instead of swapping it
    write_file "$CONSTRAINTS" "torch==$TV"
    run "$VPY" -m pip install -e ".[$EXTRAS]" $PIP_FLAGS --constraint "$CONSTRAINTS" ||
      die "pip install failed (output above). If it wants a different torch, see docs/INSTALL.md."
  else
    run "$VPY" -m pip install -e ".[$EXTRAS]" $PIP_FLAGS || die "pip install failed (output above)"
  fi
  if [ "$DRY_RUN" = 0 ]; then
    printf '%s\n' "$(sha256 "$REPO/pyproject.toml") extras=$EXTRAS $PIP_FLAGS" > "$STAMP"
  fi
fi

# ------------------------------------------------------------------------------- 5. settings folder
if [ -n "$SETTINGS_DIR" ]; then
  step "Settings folder"
  current=""
  if [ "$DRY_RUN" = 0 ] || installed_here; then
    current=$("$VPY" -c 'import os, sys
from quantabench import settings
want = os.path.realpath(os.path.expanduser(sys.argv[1]))
have = os.path.realpath(str(settings.settings_dir()))
print("same" if have == want else have)' "$SETTINGS_DIR") || current=""
  fi
  if [ "$current" = same ]; then
    ok "settings already in $SETTINGS_DIR"
  else
    [ -n "$current" ] && info "currently: $current"
    [ -d "$SETTINGS_DIR" ] || info "$SETTINGS_DIR does not exist yet: it is created (SyncThing can sync into it later)"
    [ -n "${QUANTABENCH_SETTINGS_DIR:-}" ] && warn "QUANTABENCH_SETTINGS_DIR is set and overrides this folder"
    run_quiet "$VENV/bin/quantabench-cli" settings --set-dir "$SETTINGS_DIR"
    [ "$DRY_RUN" = 1 ] || info "settings are now read from $SETTINGS_DIR"
  fi
fi

# ---------------------------------------------------------------------- 6. external programs (conda)
conda_kind() { case "$(basename "$1")" in micromamba*) echo micromamba ;; *) echo conda ;; esac; }
find_conda() {
  if [ -n "$CONDA" ]; then
    CONDA_TOOL=$(command -v "$CONDA" 2>/dev/null || true)
    [ -n "$CONDA_TOOL" ] || die "--conda $CONDA not found"
    return 0
  fi
  # conda first (it has `env update`); Miniforge ships conda and mamba
  for c in conda mamba micromamba; do
    if command -v "$c" >/dev/null 2>&1; then CONDA_TOOL=$(command -v "$c"); return 0; fi
  done
  for c in "$MINIFORGE_DIR/bin/conda" "$HOME/mambaforge/bin/conda" "$HOME/miniconda3/bin/conda" \
           "$HOME/anaconda3/bin/conda" "/opt/homebrew/Caskroom/miniforge/base/bin/conda" \
           "/opt/homebrew/bin/micromamba" "$HOME/.local/bin/micromamba"; do
    if [ -x "$c" ]; then CONDA_TOOL=$c; return 0; fi
  done
  return 1
}
env_prefix() {                            # env_prefix NAME: the environment's folder, or nothing
  "$CONDA_TOOL" env list --json 2>/dev/null | "$ANYPY" -c '
import json, os, sys
try:
    envs = json.load(sys.stdin).get("envs", [])
except ValueError:
    envs = []
for p in envs:
    if os.path.basename(p.rstrip("/")) == sys.argv[1]:
        print(p)
        break' "$1" || true
}
setup_env() {                             # setup_env FILE: create or update one environment
  local file=$1 name prefix stamp hash kind
  name=$(awk '/^name:/ { print $2; exit }' "$file")
  [ -n "$name" ] || die "$file has no name: line"
  kind=$(conda_kind "$CONDA_TOOL")
  prefix=$(env_prefix "$name")
  hash=$(sha256 "$file")
  if [ -n "$prefix" ]; then
    stamp="$prefix/.quantabench-env.sha256"
    if [ -f "$stamp" ] && [ "$(cat "$stamp")" = "$hash" ]; then
      ok "$name is up to date with $(basename "$file") ($prefix)"
      return 0
    fi
    info "$name exists ($prefix): updating it from $(basename "$file")"
    if [ "$kind" = micromamba ]; then run "$CONDA_TOOL" install -y -n "$name" -f "$file"
    else run "$CONDA_TOOL" env update -n "$name" -f "$file"
    fi
  else
    if [ "$kind" = micromamba ]; then run "$CONDA_TOOL" create -y -f "$file"
    else run "$CONDA_TOOL" env create -f "$file"
    fi
    prefix=$(env_prefix "$name")
  fi
  if [ "$DRY_RUN" = 0 ] && [ -n "$prefix" ]; then printf '%s\n' "$hash" > "$prefix/.quantabench-env.sha256"; fi
}
if [ "$WITH_ENGINES" = 1 ]; then
  step "External programs: xtb, CREST, MOPAC, Psi4 (conda environments)"
  CONDA_TOOL=""
  if find_conda; then
    ok "using $CONDA_TOOL"
  elif [ -e "$MINIFORGE_DIR" ]; then
    die "$MINIFORGE_DIR exists but has no bin/conda; fix or remove it, or pass --conda PATH"
  else
    info "no conda, mamba or micromamba found: installing Miniforge into $MINIFORGE_DIR (no shell changes)"
    installer="${TMPDIR:-/tmp}/Miniforge3-$$.sh"
    run curl -fsSL --retry 3 -o "$installer" "$MINIFORGE_URL"
    run bash "$installer" -b -p "$MINIFORGE_DIR"
    run rm -f "$installer"
    CONDA_TOOL="$MINIFORGE_DIR/bin/conda"
    info "to use conda in a terminal later: $MINIFORGE_DIR/bin/conda init zsh"
  fi
  setup_env "$REPO/envs/tools.yml"
  setup_env "$REPO/envs/psi4.yml"
  info "QuantaBench finds these environments by itself; \`quantabench-cli doctor\` below lists the programs"
fi

# ------------------------------------------------------------------------------------ 7. launchers
step "Launchers (no Terminal window)"
make_executable "$REPO/QuantaBench.command"
info "QuantaBench.command in the repository: double-click it in Finder"
if [ "$MAKE_APP" = 1 ]; then
  write_file "$APP/Contents/Info.plist" '<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>QuantaBench</string>
  <key>CFBundleDisplayName</key><string>QuantaBench</string>
  <key>CFBundleIdentifier</key><string>io.github.brandonbtm29.quantabench</string>
  <key>CFBundleExecutable</key><string>QuantaBench</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>LSMinimumSystemVersion</key><string>14.0</string>
  <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>'
  write_file "$APP/Contents/MacOS/QuantaBench" "#!/bin/bash
# Made by scripts/install_macos.sh: starts QuantaBench from the checkout's .venv without a Terminal window.
# Rerun the installer if the checkout moves.
REPO=$(quote "$REPO")
export PATH=\"/opt/homebrew/bin:/usr/local/bin:\$PATH\"   # git for Help > Check for updates...
if [ ! -x \"\$REPO/.venv/bin/quantabench\" ]; then
  /usr/bin/osascript -e 'display alert \"QuantaBench is not installed\" message \"Run scripts/install_macos.sh in the QuantaBench folder.\"'
  exit 1
fi
cd \"\$REPO\" || exit 1
exec \"\$REPO/.venv/bin/quantabench\""
  make_executable "$APP/Contents/MacOS/QuantaBench"
  info "$APP: open it from Launchpad, Spotlight or ~/Applications (drag it to the Dock to keep it there)"
fi

# --------------------------------------------------------------------------------------- 8. doctor
step "quantabench-cli doctor"
run "$VENV/bin/quantabench-cli" doctor || warn "doctor reported problems (above); docs/INSTALL.md has fixes"

step "Done"
if [ "$MAKE_APP" = 1 ]; then info "Start QuantaBench: ~/Applications/QuantaBench.app, or double-click QuantaBench.command"
else info "Start QuantaBench: double-click QuantaBench.command, or run \"$VENV/bin/quantabench\""
fi
info "Command line: \"$VENV/bin/quantabench-cli\" (or: source .venv/bin/activate)"
info "Update later: Help > Check for updates... in the app, quantabench-cli update, or run this script again."
