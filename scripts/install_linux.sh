#!/usr/bin/env bash
# Install or update QuantaBench on Linux or WSL2. Safe to rerun: every step that is already done is skipped.
#
#   bash scripts/install_linux.sh [options]
#
# Options:
#   --with-engines       also set up the external programs: Miniforge (only if no conda, mamba or
#                        micromamba is found), then the conda environments envs/tools.yml (xtb, CREST,
#                        MOPAC) and envs/psi4.yml (Psi4), created or updated
#   --settings-dir DIR   keep settings in DIR, e.g. ~/SyncThing/QuantaBench/Settings
#   --gpu cpu|rocm|cuda|auto
#                        PyTorch build (default cpu). cpu: download.pytorch.org's CPU wheels (small);
#                        cuda: PyPI's build (NVIDIA); rocm: download.pytorch.org's ROCm wheels (AMD);
#                        auto: cuda if nvidia-smi works, rocm if /dev/kfd exists, else cpu
#   --torch-index URL    install PyTorch from this package index instead (e.g. AMD's ROCm wheels)
#   --replace-torch      replace an installed PyTorch of a different build (default: keep it, or ask)
#   --python PATH        the Python (3.12 recommended) to create .venv with
#   --conda PATH         the conda, mamba or micromamba to use for --with-engines
#   --no-launcher        don't write the application-menu entry (quantabench.desktop)
#   --reinstall          run `pip install -e` even when pyproject.toml has not changed
#   --dry-run            print every action without doing it
#   -h, --help           show this help
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
VENV="$REPO/.venv"
VPY="$VENV/bin/python"
EXTRAS="all,dev"
# Added to `pip install -e` from scripts/fork-packages.json ("linux"), which `quantabench-cli update` reads too:
# tad-libcint<0.4 for GFN2-xTB in PyPI's dxtb (docs/INSTALL.md). Read in step 4.
FORK_PKGS=()
PIP_FLAGS=""
TORCH_CPU_INDEX="https://download.pytorch.org/whl/cpu"
TORCH_ROCM_INDEX="https://download.pytorch.org/whl/rocm10.0"   # pytorch.org's ROCm build, Nov 2026
MINIFORGE_DIR="$HOME/miniforge3"
MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
APPS_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"

DRY_RUN=0 WITH_ENGINES=0 SETTINGS_DIR="" GPU="cpu" TORCH_INDEX="" REPLACE_TORCH=0 PYTHON="" CONDA=""
LAUNCHER=1 REINSTALL=0

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
py_version() { "$1" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null; }
sha256() { "$ANYPY" -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$1"; }
fork_values() {                           # fork_values PLATFORM KEY: scripts/fork-packages.json values, one per line
  "$ANYPY" -c 'import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))[sys.argv[2]].get(sys.argv[3]) or []
print("\n".join(value.values() if isinstance(value, dict) else value))' "$REPO/scripts/fork-packages.json" "$1" "$2"
}
load_forks() {                            # load_forks PLATFORM: sets FORK_PKGS (packages) and PIP_FLAGS (pip options)
  local key flag values line
  FORK_PKGS=()
  PIP_FLAGS=""
  for key in packages only_binary find_links; do
    values=$(fork_values "$1" "$key") || die "scripts/fork-packages.json is missing or not valid (error above)"
    case "$key" in only_binary) flag="--only-binary" ;; find_links) flag="--find-links" ;; *) flag="" ;; esac
    while IFS= read -r line; do
      [ -n "$line" ] || continue
      if [ -z "$flag" ]; then FORK_PKGS+=("$line"); else PIP_FLAGS="${PIP_FLAGS:+$PIP_FLAGS }$flag $line"; fi
    done <<< "$values"
  done
}
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
    --no-launcher) LAUNCHER=0 ;;
    --reinstall) REINSTALL=1 ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1 (see --help)" ;;
  esac
  shift
done

case "$GPU" in cpu|rocm|cuda|auto) ;; *) die "--gpu must be cpu, rocm, cuda or auto (got '$GPU')" ;; esac
[ "$(uname -s)" = Linux ] || die "this is the Linux/WSL installer; on macOS use scripts/install_macos.sh"
unset PYTHONPATH                          # the checks below must see the venv, nothing else
cd "$REPO"

printf 'QuantaBench installer (Linux/WSL)%s\n' "$([ "$DRY_RUN" = 1 ] && echo ': dry run, nothing is changed')"
info "repository: $REPO"
IS_WSL=0
if grep -qi microsoft /proc/version 2>/dev/null; then
  IS_WSL=1
  info "running in WSL2: the GUI uses WSLg; GPU work is more reliable on native Windows (docs/INSTALL.md)"
  case "$REPO" in
    /mnt/[a-z]/*) warn "this checkout is on a Windows drive: a Linux venv there is slow. A clone in your WSL home is better." ;;
  esac
fi

# ------------------------------------------------------------------------------------- 1. Python
step "Python"
PY=""
ANYPY=""
if [ -x "$VPY" ]; then
  v=$(py_version "$VPY") || die "$VENV is broken (its Python no longer runs, e.g. after a system upgrade).
Remove it with:  rm -rf \"$VENV\"   and run this script again."
  ok ".venv exists (Python $v); no system Python needed"
  ANYPY=$VPY
else
  if [ -n "$PYTHON" ]; then
    v=$(py_version "$PYTHON") || die "--python $PYTHON does not run"
    case "$v" in 3.1[1-9]) PY=$(command -v "$PYTHON") ;; *) die "--python $PYTHON is Python $v; QuantaBench needs 3.11 or newer (3.12 recommended)" ;; esac
  else
    for c in python3.12 /usr/bin/python3.12 /usr/local/bin/python3.12 "$HOME/.local/bin/python3.12" python3; do
      if command -v "$c" >/dev/null 2>&1 && [ "$(py_version "$c")" = 3.12 ]; then
        PY=$(command -v "$c")
        break
      fi
    done
  fi
  if [ -z "$PY" ]; then
    msg="Python 3.12 was not found. Install it, then run this script again:
    Ubuntu 24.04 / Debian 13:  sudo apt install python3.12 python3.12-venv
    Ubuntu 22.04:              sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt install python3.12 python3.12-venv
    Fedora:                    sudo dnf install python3.12
    any distribution:          uv python install 3.12   (https://docs.astral.sh/uv/), then --python \"\$(uv python find 3.12)\"
  Or pass another Python 3.11+ with --python PATH."
    [ "$DRY_RUN" = 1 ] || die "$msg"
    warn "$msg"
    PY=python3.12
  else
    ok "found $PY (Python $(py_version "$PY"))"
    if ! "$PY" -c 'import ensurepip, venv' 2>/dev/null; then
      [ "$DRY_RUN" = 1 ] || die "$PY cannot create virtual environments (no ensurepip). On Debian/Ubuntu: sudo apt install python3.12-venv"
      warn "$PY has no ensurepip: sudo apt install python3.12-venv"
    fi
  fi
  ANYPY=$PY
fi
command -v "$ANYPY" >/dev/null 2>&1 || ANYPY=python3

# ------------------------------------------------------------------------- 2. virtual environment
step "Virtual environment (.venv)"
if [ ! -x "$VPY" ] && [ -e "$VENV/pyvenv.cfg" ]; then
  # e.g. the Windows .venv of a checkout on /mnt/c: creating a Linux venv on top would break it
  die "$VENV is a virtual environment for another system (no bin/python). Leave it alone and install from
a separate clone (in WSL: a folder in your Linux home, e.g. ~/GitHub\\ Projects/QuantaBench)."
fi
if [ -x "$VPY" ]; then
  ok "$VENV"
else
  run "$PY" -m venv "$VENV"
  run "$VPY" -m pip install --quiet --upgrade pip
fi

# ------------------------------------------------------------------------------------- 3. PyTorch
step "PyTorch"
if [ "$GPU" = auto ]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then GPU=cuda
  elif [ -e /dev/kfd ]; then GPU=rocm
  else GPU=cpu
  fi
  info "--gpu auto: using the $GPU build"
fi
torch_version() { [ -x "$VPY" ] && "$VPY" -c 'import importlib.metadata as m; print(m.version("torch"))' 2>/dev/null || true; }
torch_flavor() {                          # build of an installed torch version: cpu, rocm or cuda
  case "$1" in
    *+cpu*) echo cpu ;;
    *+rocm*) echo rocm ;;
    *+cu*) echo cuda ;;
    *) echo cuda ;;                       # no local tag: PyPI's Linux build, which bundles CUDA
  esac
}
install_torch() {
  if [ -n "$TORCH_INDEX" ]; then run "$VPY" -m pip install torch --index-url "$TORCH_INDEX"
  elif [ "$GPU" = cpu ]; then run "$VPY" -m pip install torch --index-url "$TORCH_CPU_INDEX"
  elif [ "$GPU" = rocm ]; then run "$VPY" -m pip install torch --index-url "$TORCH_ROCM_INDEX"
  else run "$VPY" -m pip install torch
  fi || die "PyTorch did not install. If download.pytorch.org is unreachable, rerun with
  --torch-index https://pypi.org/simple   (PyPI's build: it runs on the CPU too, but includes ~3 GB of CUDA libraries)"
}
TV=$(torch_version)
if [ -z "$TV" ]; then
  install_torch
else
  have=$(torch_flavor "$TV")
  if [ "$have" = "$GPU" ] && [ -z "$TORCH_INDEX" ]; then
    ok "torch $TV ($have build)"
  elif [ "$REPLACE_TORCH" = 1 ]; then
    info "replacing torch $TV ($have build) with the ${TORCH_INDEX:-$GPU} build (--replace-torch)"
    run "$VPY" -m pip uninstall -y torch
    install_torch
  else
    answer=n
    if [ -t 0 ] && [ "$DRY_RUN" = 0 ]; then
      read -r -p "    torch $TV ($have build) is installed; you asked for ${TORCH_INDEX:-$GPU}. Replace it? [y/N] " answer || answer=n
    fi
    case "$answer" in
      y|Y|yes|YES)
        run "$VPY" -m pip uninstall -y torch
        install_torch ;;
      *) ok "keeping torch $TV ($have build); rerun with --replace-torch to switch to ${TORCH_INDEX:-$GPU}" ;;
    esac
  fi
fi

# ---------------------------------------------------------------------------------- 4. QuantaBench
step "QuantaBench (pip install -e \".[$EXTRAS]\")"
STAMP="$VENV/.quantabench-install"
load_forks linux
want=""
if [ -x "$VPY" ]; then
  want="$(sha256 "$REPO/pyproject.toml") extras=$EXTRAS ${FORK_PKGS[*]:-}${PIP_FLAGS:+ $PIP_FLAGS}"
fi
installed_here() { "$VPY" -c 'import os, sys, quantabench; sys.exit(0 if os.path.realpath(quantabench.__file__).startswith(os.path.realpath(sys.argv[1]) + os.sep) else 1)' "$REPO/src" 2>/dev/null; }
if [ "$REINSTALL" = 0 ] && [ -n "$want" ] && [ -f "$STAMP" ] && [ "$(cat "$STAMP")" = "$want" ] && installed_here; then
  ok "installed from this checkout; pyproject.toml unchanged since the last install"
else
  CONSTRAINTS="$VENV/.torch-constraint.txt"
  TV=$(torch_version)
  # shellcheck disable=SC2086  # PIP_FLAGS is a list of words
  if [ -n "$TV" ]; then
    # keep the PyTorch build chosen above: an extra that wants another torch fails loudly instead of swapping it
    write_file "$CONSTRAINTS" "torch==$TV"
    run "$VPY" -m pip install -e ".[$EXTRAS]" ${FORK_PKGS[@]+"${FORK_PKGS[@]}"} $PIP_FLAGS --constraint "$CONSTRAINTS" ||
      die "pip install failed (output above). If it wants a different torch, see docs/INSTALL.md."
  else
    run "$VPY" -m pip install -e ".[$EXTRAS]" ${FORK_PKGS[@]+"${FORK_PKGS[@]}"} $PIP_FLAGS || die "pip install failed (output above)"
  fi
  if [ "$DRY_RUN" = 0 ]; then
    printf '%s\n' "$(sha256 "$REPO/pyproject.toml") extras=$EXTRAS ${FORK_PKGS[*]:-}${PIP_FLAGS:+ $PIP_FLAGS}" > "$STAMP"
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
           "$HOME/anaconda3/bin/conda" "$HOME/.local/bin/micromamba" "$HOME/micromamba/bin/micromamba"; do
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
  elif [ -x "$MINIFORGE_DIR/bin/conda" ]; then
    CONDA_TOOL="$MINIFORGE_DIR/bin/conda"
  elif [ -e "$MINIFORGE_DIR" ]; then
    die "$MINIFORGE_DIR exists but has no bin/conda; fix or remove it, or pass --conda PATH"
  else
    info "no conda, mamba or micromamba found: installing Miniforge into $MINIFORGE_DIR (no shell changes)"
    installer="${TMPDIR:-/tmp}/Miniforge3-$$.sh"
    if command -v curl >/dev/null 2>&1; then run curl -fsSL --retry 3 -o "$installer" "$MINIFORGE_URL"
    else run wget -q -O "$installer" "$MINIFORGE_URL"
    fi
    run bash "$installer" -b -p "$MINIFORGE_DIR"
    run rm -f "$installer"
    CONDA_TOOL="$MINIFORGE_DIR/bin/conda"
    info "to use conda in a terminal later: $MINIFORGE_DIR/bin/conda init"
  fi
  setup_env "$REPO/envs/tools.yml"
  setup_env "$REPO/envs/psi4.yml"
  info "QuantaBench finds these environments by itself; \`quantabench-cli doctor\` below lists the programs"
fi

# ------------------------------------------------------------------------------------- 7. launcher
if [ "$LAUNCHER" = 1 ]; then
  step "Launcher (application menu, no terminal window)"
  case "$VENV" in                         # Exec= is "quoted"; these would need escaping
    *[\"\`\$\\%]*) die "the path $VENV has a character a .desktop file cannot hold (\" \` \$ \\ %); use --no-launcher" ;;
  esac
  write_file "$APPS_DIR/quantabench.desktop" "[Desktop Entry]
Type=Application
Name=QuantaBench
GenericName=Molecular modeling workbench
Comment=Build conformers, refine them with ML potentials or xTB, and look at the results
Exec=\"$VENV/bin/quantabench\"
Icon=applications-science
Terminal=false
Categories=Education;Science;Chemistry;
StartupNotify=true"
  if [ "$DRY_RUN" = 0 ] && command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database "$APPS_DIR" >/dev/null 2>&1 || true
  fi
fi

# ------------------------------------------------------------------------------- 8. system checks
step "System libraries"
xcb_cursor=""
for lib in /usr/lib/*/libxcb-cursor.so.0 /usr/lib64/libxcb-cursor.so.0 /usr/lib/libxcb-cursor.so.0; do
  if [ -e "$lib" ]; then xcb_cursor=$lib; break; fi
done
if [ -z "$xcb_cursor" ] && { ldconfig -p 2>/dev/null || /sbin/ldconfig -p 2>/dev/null; } | grep -q 'libxcb-cursor\.so\.0'; then
  xcb_cursor=ldconfig
fi
if [ -n "$xcb_cursor" ]; then
  ok "libxcb-cursor0 (needed by the Qt GUI)"
else
  warn "libxcb-cursor0 is missing; the GUI will not start without it. Debian/Ubuntu: sudo apt install libxcb-cursor0"
fi

# --------------------------------------------------------------------------------------- 9. doctor
step "quantabench-cli doctor"
run "$VENV/bin/quantabench-cli" doctor || warn "doctor reported problems (above); docs/INSTALL.md has fixes"

step "Done"
[ "$LAUNCHER" = 1 ] && info "Start QuantaBench from the application menu, or run: \"$VENV/bin/quantabench\""
[ "$IS_WSL" = 1 ] && info "In WSL, WSLg also lists it in the Windows Start menu."
info "Command line: \"$VENV/bin/quantabench-cli\" (or: source .venv/bin/activate)"
info "Update later: Help > Check for updates... in the app, quantabench-cli update, or run this script again."
