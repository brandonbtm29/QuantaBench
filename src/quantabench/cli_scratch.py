"""Command line for QuantaBench's scratch folder (``chem.external``): ``scratch``.

    quantabench-cli scratch                      what is in it: folders of running jobs, left behind, older ones
    quantabench-cli scratch --clean [--dry-run]  clean what ended processes left (as every run does when it starts)
    quantabench-cli scratch --clean --older-than 0   also remove the kept logs of failed jobs now

On Windows with CREST through WSL, also the job folders inside WSL (``chem.wsl_scratch``).
"""

from __future__ import annotations

DESCRIPTION = (
    "QuantaBench's scratch folder (Settings > Engines > scratch folder; default <temp>/QuantaBench) holds one folder "
    "per external job (xtb, CREST, Psi4). A finished job's folder is removed; a failed one keeps its inputs and logs "
    "for a few days, without its bulky files. A process that is killed (a lane past the time limit, a crash, a "
    "closed window) cannot clean up: every run cleans what ended processes left when it starts, and a lane's when "
    "it is killed. This command shows the folder and does the same cleaning on demand; a folder of a process that "
    "is still running is never touched. On Windows with CREST through WSL, CREST runs in a folder inside WSL "
    "(/dev/shm or WSL's disk; engines.crest_wsl_scratch): those folders are listed and cleaned too.")


def register(sub) -> None:
    p = sub.add_parser("scratch", help="show or clean QuantaBench's scratch folder (left-behind job files)",
                       description=DESCRIPTION)
    p.add_argument("--clean", action="store_true",
                   help="remove what ended processes left: their bulky files now, the folders (logs) once older "
                        "than --older-than")
    p.add_argument("--older-than", type=float, default=None, metavar="DAYS",
                   help="with --clean: remove left-behind folders older than this whole, logs included (default 3; "
                        "0 = all of them)")
    p.add_argument("--dry-run", action="store_true", help="with --clean: say what would be freed, change nothing")
    p.set_defaults(func=scratch_command)


def _gb(n: int) -> str:
    return f"{n / 1e9:.2f} GB"


def scratch_command(args) -> int:
    from .chem import external

    root = external.scratch_root()
    print(f"Scratch folder: {root}")
    rows = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        info = external._owner(d)
        state = ("running" if external._owner_running(info) else "left behind") if info else "no owner record"
        rows.append((state, external._folder_bytes(d), external._prunable_bytes(d)))
    for state in ("running", "left behind", "no owner record"):
        sel = [r for r in rows if r[0] == state]
        if sel:
            print(f"  {state:16s} {len(sel):5d} folder(s)  {_gb(sum(r[1] for r in sel)):>10s}"
                  f"  (bulky files {_gb(sum(r[2] for r in sel))})")
    if not rows:
        print("  empty")
    wsl = _wsl_rows()
    if not args.clean:
        if any(r[0] != "running" for r in rows):
            print("`quantabench-cli scratch --clean` removes what ended processes left (--dry-run to see first).")
        return 0
    keep = external.SCRATCH_KEEP_DAYS if args.older_than is None else max(0.0, float(args.older_than))
    res = external.sweep_scratch(keep_days=keep, dry_run=args.dry_run)
    print(("Would free " if args.dry_run else "Freed ") + f"{_gb(res.freed_bytes)}: {res.folders} left-behind "
          f"folder(s), {res.removed} {'to remove' if args.dry_run else 'removed'} whole, {res.pruned} kept with "
          f"their logs; {res.running} of running jobs left alone"
          + (f"; {res.wsl_removed} job folder(s) inside WSL removed with them" if res.wsl_removed else ""))
    if wsl:
        from .chem import wsl_scratch

        try:
            w = wsl_scratch.sweep(dry_run=args.dry_run)
        except wsl_scratch.WslScratchError as exc:
            print(f"Inside WSL: not cleaned ({exc})")
        else:
            print(f"Inside WSL: {'would free' if args.dry_run else 'freed'} {_gb(w.freed_bytes)} in {w.folders} "
                  f"job folder(s) of ended runs; {w.running} of running jobs left alone")
    return 0


def _wsl_rows() -> bool:
    """Print the job folders inside WSL (CREST through WSL on Windows, ``chem.wsl_scratch``); True if there are any to
    look at."""
    from .chem import wsl_scratch

    try:
        if not wsl_scratch.in_use():
            return False
        pr = wsl_scratch.probe()
    except Exception as exc:  # noqa: BLE001 - WSL not answering: say so, the rest of the command goes on
        print(f"Inside WSL ({wsl_scratch.SHM_ROOT}, WSL's disk): not reachable ({exc})")
        return False
    print(f"Inside WSL: {wsl_scratch.SHM_ROOT} and {pr.disk_root}")
    old = {j.path for j in wsl_scratch.stale(pr.jobs)}
    for state, sel in (("running", [j for j in pr.jobs if j.path not in old]),
                       ("left behind", [j for j in pr.jobs if j.path in old])):
        if sel:
            print(f"  {state:16s} {len(sel):5d} folder(s)  {_gb(sum(j.size for j in sel)):>10s}")
    if not pr.jobs:
        print("  empty")
    return True
