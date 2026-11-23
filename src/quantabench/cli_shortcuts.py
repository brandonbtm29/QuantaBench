"""Command line for the app launchers (``quantabench.launchers``): ``shortcuts``.

    quantabench-cli shortcuts                 Start menu shortcuts / QuantaBench.app / an application-menu entry
    quantabench-cli shortcuts --desktop       ... and on the Desktop
    quantabench-cli shortcuts --remove        take them away again
    quantabench-cli shortcuts --dry-run       say what would be written

One launcher per environment of the checkout (``.venv``, ``.venv-rocm``, ...), each starting the app without a
terminal window, with the QuantaBench icon. The installers run this; run it again when the checkout moves.
"""

from __future__ import annotations

DESCRIPTION = (
    "Make launchers that start QuantaBench without a terminal window, with its icon: Start menu shortcuts on Windows "
    "(they also group the running window with a pinned taskbar icon), ~/Applications/QuantaBench.app on macOS, "
    "an application-menu entry (quantabench.desktop) on Linux. One per Python environment of the checkout: .venv is "
    "\"QuantaBench\", .venv-rocm \"QuantaBench (AMD GPU)\". They point at the checkout, so updates need no new "
    "launcher; run this again when the checkout moves (docs/INSTALL.md).")


def register(sub) -> None:
    p = sub.add_parser("shortcuts", help="make launchers with the app icon that hide the terminal window",
                       description=DESCRIPTION)
    p.add_argument("--desktop", action="store_true", help="also put them on the Desktop")
    p.add_argument("--remove", action="store_true", help="remove the launchers this command makes")
    p.add_argument("--dry-run", action="store_true", help="say what would be written or removed; change nothing")
    p.set_defaults(func=shortcuts_command)


def shortcuts_command(args) -> int:
    from . import launchers
    from .settings import repo_root

    repo = repo_root()
    variants = launchers.find_variants(repo)
    print(f"QuantaBench checkout: {repo or '(not a checkout: the running Python environment)'}")
    for v in variants:
        print(f"  {v.label}: {v.python()}")
    try:
        paths = launchers.create(repo, desktop=args.desktop, dry_run=args.dry_run, remove=args.remove,
                                 variants=variants)
    except (OSError, RuntimeError) as exc:
        print(f"Could not {'remove' if args.remove else 'make'} the launchers: {exc}")
        return 1
    verb = "Removed" if args.remove else "Wrote"
    for line in paths:
        print(f"  {line}" if args.dry_run else f"  {verb} {line}")
    if args.remove and not paths:
        print("  Nothing to remove.")
    if not args.dry_run and not args.remove:
        print("Start QuantaBench from there; the launchers follow updates of this checkout.")
    return 0
