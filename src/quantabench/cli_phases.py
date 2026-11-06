"""Command line for phases (``chem.phases``): ``build --phases gas,water`` and ``pair``.

    quantabench-cli build --file x.sdf --method tblite-gfn2 --no-protonate --phases gas,water --run-dir OUT
    quantabench-cli build --phases gas,water --resume OUT          # continue every phase (and start a new one)
    quantabench-cli pair OUT/gas OUT/water --out paired.csv        # or: pair OUT

``register`` adds both to the parser of ``quantabench.cli``. Each phase is an ordinary ``build`` call into its own
run folder (``OUT/gas``, ``OUT/water``) with ``--solvent`` set to the phase, so every other build option (lanes,
chunks, --retry-failed, the memory guard, the time limit) means the same as for one run.
"""

from __future__ import annotations

import argparse
import copy
import sys
from collections.abc import Callable
from pathlib import Path

PHASES_HELP = ("compute the same input in several phases, e.g. gas,water (any solvent names; 'gas' = no solvent): "
               "one normal run folder per phase, <run dir>/gas and <run dir>/water, each resumable on its own, then "
               "<run dir>/paired.csv with gas_*, water_* and delta_* columns and dG_solv_kcal per molecule (see "
               "`quantabench-cli pair -h`); --resume RUN_DIR continues every phase")
PAIR_DESCRIPTION = (
    "Join the runs of one input in several phases (build --phases, or separate builds with --solvent) into one "
    "table, one row per molecule, matched by id AND input InChIKey (never by row order). Columns: the identity "
    "columns once (id, name, input_inchikey, inchikey, canonical_smiles, charge, ...), pair_status and pair_notes, "
    "dG_solv_kcal = E_water - E_gas of each phase's best conformer at its own geometry (ALPB's total energy already "
    "contains the solvation free energy), dG_solv_G_kcal = G_water - G_gas when both ran frequencies, "
    "dG_solv_vertical_kcal = E_water - E_gas at the water geometry, then gas_<col>, water_<col> and delta_<col> "
    "(water - gas) for every column of summary.csv and descriptors.csv. Energies refer to 1 mol/L gas -> 1 mol/L "
    "solution (add 1.89 kcal/mol at 298 K for 1 atm gas). Unfinished runs are read from their job tables: missing "
    "molecules show in pair_status. See docs/ENGINES.md, 'Phases'.")


def register(sub, build_parser: argparse.ArgumentParser) -> None:
    """Add ``--phases`` to ``build`` (dispatching to ``build_phases`` when given) and the ``pair`` command."""
    build_parser.add_argument("--phases", metavar="LIST", help=PHASES_HELP)
    plain_build = build_parser.get_default("func")

    def build(args) -> int:
        return build_phases(args, plain_build) if getattr(args, "phases", None) else plain_build(args)

    build_parser.set_defaults(func=build)
    p = sub.add_parser("pair", help="one table of the gas and solvent runs of an input: gas_*, water_*, delta_*, "
                                    "dG_solv_kcal", description=PAIR_DESCRIPTION)
    p.add_argument("run_dirs", nargs="+", metavar="RUN_DIR",
                   help="run folders (one or more per phase: chunks are combined), or the folder build --phases "
                        "wrote (its phase folders are used)")
    p.add_argument("--out", help="CSV to write (default: paired.csv next to the run folders)")
    p.add_argument("--reference", metavar="PHASE",
                   help="the phase differences are taken to (default: gas when present, else the first run's)")
    p.set_defaults(func=pair_command)


def _default_out(run_dirs: list[Path]) -> Path:
    from .chem import phases

    parents = {d.resolve().parent for d in run_dirs}
    return (parents.pop() if len(parents) == 1 else Path.cwd()) / phases.PAIRED_NAME


def pair_command(args) -> int:
    from .chem import phases

    dirs = phases.expand_run_dirs(args.run_dirs)
    out = Path(args.out).expanduser() if args.out else _default_out(dirs)
    try:
        s = phases.write_paired(dirs, out, reference=args.reference, log=lambda m: print(m, file=sys.stderr))
    except (ValueError, FileNotFoundError) as exc:
        print(exc, file=sys.stderr)
        return 2
    for label, src in s.sources.items():
        print(f"{label}: {', '.join(src)}")
    print(s.text())
    return 0


def _stored_solvent(folder: Path, solvent: str) -> str:
    """The solvent as the run folder stored it when it names the same phase ('none', 'H2O'): a plain build keeps
    ``--solvent`` as typed, and a resume with another spelling of it would be refused as a settings change."""
    from .chem import jobs, solvents

    try:
        stored = jobs.open_run(folder).config.get("solvent") or ""
        return stored if solvents.canonical(stored) == solvent else solvent
    except (ValueError, OSError):
        return solvent


def _seed_phase(src: Path, dst: Path, solvent: str) -> None:
    """Start the job table of a phase that a resumed ``--phases`` run does not have yet, from a sibling phase:
    its stored settings with this phase's solvent, and its inputs."""
    from dataclasses import replace

    from .chem import jobs

    st = jobs.open_run(src)
    cfg = replace(jobs.stored_config(st.config), solvent=solvent)
    with jobs.JobStore(dst) as store:
        store.init_run(cfg, st.chunk)
        store.add_inputs(jobs.load_inputs(src))


def build_phases(args, build: Callable[[argparse.Namespace], int]) -> int:
    """``build --phases``: one ``build`` per phase into ``<root>/<phase>``, then ``<root>/paired.csv``."""
    from . import settings as settings_mod
    from .chem import jobs, phases
    from .chem.pipeline import new_run_dir

    try:
        solvents = phases.parse_phases(args.phases)
        if len(solvents) < 2:
            raise ValueError("give at least two phases (one phase is a plain build: --solvent)")
    except ValueError as exc:
        print(f"--phases {args.phases}: {exc}", file=sys.stderr)
        return 2
    if getattr(args, "solvent", None) is not None:
        print("--phases sets the solvent of each phase: drop --solvent", file=sys.stderr)
        return 2
    if args.resume and args.run_dir:
        print("give --resume or --run-dir, not both", file=sys.stderr)
        return 2
    if args.resume:
        root = Path(args.resume).expanduser()
        if not root.is_dir():
            print(f"{root}: no such folder", file=sys.stderr)
            return 2
    elif args.run_dir:
        root = Path(args.run_dir).expanduser()
    else:
        base = args.out or settings_mod.load().output_path()
        root = new_run_dir(base, "phases_" + "_".join(phases.phase_label(s) for s in solvents))
    dirs = phases.phase_dirs(root, solvents)
    have_inputs = bool(args.file or args.molecules)
    codes = []
    for k, (solvent, (label, folder)) in enumerate(zip(solvents, dirs.items()), 1):
        print(f"\n=== phase {k}/{len(dirs)}: {label} -> {folder} ===", flush=True)
        a = copy.copy(args)
        a.phases, a.solvent = None, solvent
        a.resume, a.run_dir = None, str(folder)
        if args.resume:
            if (folder / jobs.DB_NAME).is_file():
                a.resume, a.run_dir = str(folder), None
                a.solvent = _stored_solvent(folder, solvent)
            elif not have_inputs:
                src = next((d for d in dirs.values() if (d / jobs.DB_NAME).is_file()), None)
                if src is None:
                    print(f"{root} has no phase run folders: start it with the input (--file ...) and --run-dir",
                          file=sys.stderr)
                    return 2
                _seed_phase(src, folder, solvent)
                print(f"started {folder} with the inputs and settings of {src} ({label} phase)")
                a.resume, a.run_dir = str(folder), None
        codes.append(build(a))
        if codes[-1] == 2:          # a usage error (settings mismatch, run in use): the other phases would repeat it
            return 2
    print(f"\n=== paired table: {root / phases.PAIRED_NAME} ===")
    try:
        s = phases.write_paired(list(dirs.values()), root / phases.PAIRED_NAME,
                                log=lambda m: print(m, file=sys.stderr))
        print(s.text())
    except (ValueError, FileNotFoundError) as exc:
        print(f"paired table not written: {exc}", file=sys.stderr)
        return max([*codes, 1])
    return max(codes)
