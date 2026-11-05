#!/bin/bash
# Double-click in Finder to start QuantaBench (macOS). The app starts on its own, not as a child of this
# Terminal window, and the window closes again. For no Terminal window at all, use
# ~/Applications/QuantaBench.app, which scripts/install_macos.sh creates.
cd "$(dirname "$0")" || exit 1
if [ ! -x .venv/bin/python ] || [ ! -x .venv/bin/quantabench ]; then
  echo "QuantaBench is not installed in this folder yet. Run:"
  echo "    bash scripts/install_macos.sh"
  read -r -p "Press Return to close this window. " _
  exit 1
fi
# Both processes run in a new session without this window's terminal, so closing the window neither stops
# QuantaBench nor asks "terminate running processes?". The second one closes this window after we exit.
CLOSE='tell application "Terminal" to close (every window whose name contains "QuantaBench.command")'
# shellcheck disable=SC2016  # $0 belongs to the inner /bin/sh
.venv/bin/python -c 'import subprocess, sys
quiet = dict(start_new_session=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
subprocess.Popen([sys.argv[1]], **quiet)
subprocess.Popen(["/bin/sh", "-c", "sleep 1; exec /usr/bin/osascript -e \"$0\"", sys.argv[2]], **quiet)' \
  "$PWD/.venv/bin/quantabench" "$CLOSE" || {
  read -r -p "QuantaBench did not start. Press Return to close this window. " _
  exit 1
}
exit 0
