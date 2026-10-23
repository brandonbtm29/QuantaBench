#!/usr/bin/env bash
# Set up a headless Linux (cloud) development environment for QuantaBench.
# Used by the nightly agent runs; also fine on any Ubuntu/Debian box. Idempotent.
#
#   bash scripts/cloud_setup.sh            # creates ../.venv-qb next to the repo
#   source ../.venv-qb/bin/activate
#
# Notes for restricted networks: download.pytorch.org and storage.googleapis.com (AIMNet2 weights)
# may be blocked. PyPI's Linux torch wheel works on CPU; AIMNet2 tests use stand-ins.
set -euo pipefail
cd "$(dirname "$0")/.."
VENV="${QB_VENV:-../.venv-qb}"

if command -v apt-get >/dev/null 2>&1; then
  # Qt xcb plugin, OpenGL for VTK, a virtual display, and MPI for the lammps wheel
  (apt-get install -y -qq libxcb-cursor0 libgl1 libegl1 xvfb libmpich12 >/dev/null 2>&1 ||
   (apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq libxcb-cursor0 libgl1 libegl1 xvfb libmpich12 >/dev/null 2>&1)) ||
   echo "warning: could not apt-get install system libraries (not root?)"
fi

command -v uv >/dev/null 2>&1 || pip install -q uv
[ -x "$VENV/bin/python" ] || uv venv -q -p 3.12 "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
uv pip install -q "torch>=2.8"
uv pip install -q -e ".[all,dev]" "tad-libcint<0.4"
# dxtb pins tad-mctc; tad-libcint may upgrade it, so put the pin back
uv pip install -q "tad-mctc==0.7.0" 2>/dev/null || true
python -c "import quantabench, torch, rdkit; print('QuantaBench', quantabench.__version__, '| torch', torch.__version__)"
echo "Run tests with: xvfb-run -a python -m pytest -q"
