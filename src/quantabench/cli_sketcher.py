"""Command line for the Sketch dock's editor (``sketcher_assets``): ``quantabench-cli sketcher``.

    quantabench-cli sketcher                          # where the editor is, or what is missing
    quantabench-cli sketcher --install                # download Schrödinger's Sketcher now (17.5 MB, sha256-pinned)
    quantabench-cli sketcher --install --from-zip ~/Downloads/sketcher-2026-4-wasm.zip   # offline: a zip got by hand
    quantabench-cli sketcher --use ~/src/sketcher/build/wasm    # this machine: a self-built (newer) version
    quantabench-cli sketcher --use ""                 # back to the pinned release

``register`` adds the command to the parser of ``quantabench.cli``.
"""

from __future__ import annotations

import sys
import time

DESCRIPTION = (
    "The Sketch dock runs Schrödinger's 2D Sketcher (BSD-3-Clause, https://github.com/schrodinger/sketcher), whose "
    "WebAssembly build is downloaded on first use into the programs folder, pinned by sha256. --install fetches it "
    "ahead of time (before going offline, or for every user of a shared machine); --from-zip installs a zip "
    "downloaded by hand. The dock itself also needs Qt WebEngine: pip install \"quantabench[editor]\".")


def register(sub) -> None:
    p = sub.add_parser("sketcher", help="the 2D sketcher's files: status, download, or a folder of your own",
                       description=DESCRIPTION)
    p.add_argument("--install", action="store_true", help="download, check and unpack the pinned release")
    p.add_argument("--from-zip", metavar="ZIP", help="with --install: use this zip instead of downloading")
    p.add_argument("--force", action="store_true", help="with --install: unpack again even when it is there")
    p.add_argument("--use", metavar="DIR", help="this machine: use the editor files in DIR (a folder with "
                                                "wasm_shell.html, qtloader.js, Sketcher.js, Sketcher.wasm); "
                                                "'' = the pinned release again")
    p.add_argument("--machine", help="with --use: set it for this machine name instead of this one")
    p.set_defaults(func=_sketcher)


def _sketcher(args) -> int:
    from . import settings as settings_mod
    from . import sketcher_assets as assets
    from .gui import sketcher_web

    if args.from_zip and not args.install:
        print("--from-zip goes with --install", file=sys.stderr)
        return 2
    if args.use is not None:
        from pathlib import Path

        folder = args.use.strip()
        if folder:
            missing = assets.missing_files(Path(folder).expanduser())
            if missing:
                print(f"{folder} lacks {', '.join(missing)}: not a Sketcher build folder", file=sys.stderr)
                return 1
            folder = settings_mod.portable(folder)
        s = settings_mod.load()
        machine = settings_mod.machine_key(args.machine) if args.machine else settings_mod.this_machine()
        s.set_engine_path_for(assets.ENGINE_KEY, folder, machine)
        settings_mod.save(s)
        print(f"sketcher on {machine}: {folder or f'the pinned release {assets.RELEASE}'}")
    if args.install:
        last = [0.0]

        def progress(done, total):
            now = time.monotonic()
            if now - last[0] >= 1.0 or done >= total:
                last[0] = now
                print(f"  {done / 1e6:5.1f} of {total / 1e6:.1f} MB", flush=True)

        try:
            assets.install(zip_path=args.from_zip, force=args.force, progress=progress, log=print)
        except assets.SketcherAssetError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
    print(assets.status_text())
    if not sketcher_web.webengine_installed():
        print("  Qt WebEngine is not installed, so the Sketch dock cannot show it: " + sketcher_web.INSTALL_HINT)
    loc = assets.locate()
    return 0 if loc.ok or not (args.install or args.use) else 1
