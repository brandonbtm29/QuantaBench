"""``python -m quantabench`` opens the GUI; ``python -m quantabench <command>`` runs the CLI."""

import sys


def main() -> int:
    if len(sys.argv) > 1:
        from .cli import main as cli_main

        return cli_main()
    from .gui.app import main as gui_main

    return gui_main()


if __name__ == "__main__":
    code = main()
    from .resources import hard_exit_if_needed

    hard_exit_if_needed(code)          # AMD's Windows ROCm runtime can hang at shutdown
    raise SystemExit(code)
