"""Command line for comparing conformer sets (``chem.conformer_compare``): ``conformers compare``.

    quantabench-cli conformers compare --a C3A_P1_DFT_Top5.spartan C3A_P1_All_Conformers_Top30.spartan C3A_P1.spartan \
        --include "*_DFT_Top5.spartan" "*_All_Conformers_Top30.spartan" --b runs/val_water --out cmp_water
    quantabench-cli conformers compare --a runs/rdkit_gas --b runs/crest_gas --a-name RDKit --b-name CREST --out cmp
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

DESCRIPTION = (
    "Compare the conformers two programs found for the same molecules: side A (the reference: Spartan documents, a "
    "Spartan import's long table + SDF, or a QuantaBench run) against side B (a QuantaBench run folder, or any of the "
    "same). Molecules are joined by constitution; conformers are matched by a symmetric heavy-atom RMSD that copes "
    "with the equivalent F and O atoms of PFAS (the sibling-group factorization in chem/conformer_compare.py), the "
    "backbone and polar-H torsions and the stereo configuration (mirror images count as the same conformer). "
    "Reports, per molecule and overall: how many of A's conformers within 5/10 kJ/mol of A's minimum B found and the "
    "reverse, whether the two lowest are the same, their RMSD and torsion differences; with --rescore both sets are "
    "re-scored at one common level (GFN2-xTB single points, gas and ALPB, and with --relax a short optimization) to "
    "tell a missed conformer (search) from a different ranking (energies), and the dipoles of each program's lowest "
    "conformer split the dipole gap into method and conformer. Writes molecules.csv, levels.csv, conformers.csv, "
    "summary.csv and report.md into --out (and rescore_cache.jsonl, which a rerun reuses).")


def register(sub) -> None:
    p = sub.add_parser("conformers", help="compare conformer sets (Spartan vs QuantaBench, RDKit vs CREST, ...)")
    csub = p.add_subparsers(dest="conformers_command", required=True)
    c = csub.add_parser("compare", help="match two programs' conformers, coverage, search vs ranking, dipoles",
                        description=DESCRIPTION)
    c.add_argument("--a", nargs="+", required=True, metavar="PATH",
                   help="side A (the reference): .spartan documents or folders of them, a Spartan import's long table "
                        "(.csv) with its .sdf (import --conformers all --table --sdf), or QuantaBench run folders")
    c.add_argument("--b", nargs="+", required=True, metavar="PATH", help="side B (the probe), the same kinds")
    c.add_argument("--out", required=True, help="output folder")
    c.add_argument("--a-name", default="", help="name of side A in the report (default: its program)")
    c.add_argument("--b-name", default="", help="name of side B in the report (default: its program)")
    c.add_argument("--include", nargs="+", default=[], metavar="GLOB",
                   help="Spartan documents whose conformers are used (fnmatch on the file name, e.g. "
                        "'*_DFT_Top5.spartan'); the others only resolve labels (default: all)")
    c.add_argument("--only", help="compare only these molecules: a CSV with an id column, or a comma list of ids")
    c.add_argument("--a-level", default="", help="the level side A ranks by (a substring, e.g. B3LYP; default: its best)")
    c.add_argument("--b-level", default="", help="the level side B ranks by (default: its best)")
    c.add_argument("--rmsd", type=float, default=0.5, help="same conformer: heavy-atom RMSD at most this, Å (0.5)")
    c.add_argument("--torsion", type=float, default=60.0,
                   help="same conformer: backbone and polar-H torsions within this, degrees (60; 0 = not used)")
    c.add_argument("--no-mirror", action="store_true", help="do not count mirror images as the same conformer")
    c.add_argument("--windows", default="5,10", help="energy windows above each side's minimum, kJ/mol (5,10)")
    c.add_argument("--rescore", default="", metavar="PHASES",
                   help="common level: GFN2-xTB (tblite) single points in these phases, e.g. gas,water (ALPB)")
    c.add_argument("--relax", default="", metavar="PHASE",
                   help="also optimize every conformer at GFN2-xTB in this phase (gas or a solvent) for the ranking")
    c.add_argument("--phase", default="", help="the phase the common level decides in (default: side B's run's)")
    c.add_argument("--fmax", type=float, default=0.03, help="optimization: force convergence, eV/Å (0.03)")
    c.add_argument("--steps", type=int, default=300, help="optimization: at most this many steps (300)")
    c.add_argument("--workers", type=int, default=1, help="processes for the common level (1)")
    c.set_defaults(func=compare_command)


def _only(spec: str) -> set[str]:
    if not spec:
        return set()
    p = Path(spec).expanduser()
    if p.is_file():
        import csv

        with p.open(newline="", encoding="utf-8-sig") as fh:
            return {(r.get("id") or r.get("label") or "").strip() for r in csv.DictReader(fh)} - {""}
    return {t.strip() for t in spec.split(",") if t.strip()}


def compare_command(args) -> int:
    from .chem import conformer_compare as cc

    try:
        windows = tuple(float(x) for x in args.windows.split(",") if x.strip())
        a = cc.read_sources(args.a, include=args.include, log=print)
        b = cc.read_sources(args.b, include=args.include, log=print)
    except (OSError, ValueError, RuntimeError) as exc:
        print(exc, file=sys.stderr)
        return 2
    only = _only(args.only or "")
    if only:
        keep = {s.key for s in a + b if only & ({s.label} | set(s.labels))}
        a, b = [s for s in a if s.key in keep], [s for s in b if s.key in keep]
    if not a or not b:
        print("no conformer sets on side " + ("A" if not a else "B"), file=sys.stderr)
        return 2
    a_name = args.a_name or a[0].program
    b_name = args.b_name or b[0].program
    if a_name == b_name:
        a_name, b_name = f"{a_name} (A)", f"{b_name} (B)"
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    keys = {s.key for s in a} & {s.key for s in b}
    phases = [p.strip() for p in args.rescore.split(",") if p.strip()]
    relax = args.relax.strip()
    if relax and relax not in phases:
        phases.append(relax)
    if phases:
        joined = [s for s in a + b if s.key in keys]
        try:
            n = cc.rescore(joined, phases=phases, relax=relax, fmax=args.fmax, steps=args.steps, workers=args.workers,
                           cache=out / "rescore_cache.jsonl", log=print)
        except ImportError as exc:
            print(f"the common level needs tblite ({exc})", file=sys.stderr)
            return 2
        print(f"common level: {n} conformers computed")
    comp = cc.compare(a, b, level_a=args.a_level, level_b=args.b_level, phase=args.phase, rmsd_cut=args.rmsd,
                      torsion_cut=args.torsion, mirror=not args.no_mirror, windows=windows, a_name=a_name,
                      b_name=b_name, log=print)
    inputs = {"A": f"{a_name}: " + ", ".join(map(str, args.a)), "B": f"{b_name}: " + ", ".join(map(str, args.b))}
    if args.include:
        inputs["Spartan documents used"] = ", ".join(args.include)
    if phases:
        inputs["common level"] = (f"GFN2-xTB (tblite) single points in {', '.join(phases)}"
                                  + (f"; optimization in {relax} (fmax {args.fmax} eV/Å, at most {args.steps} steps)"
                                     if relax else ""))
    paths = cc.write_outputs(comp, out, inputs)
    (out / "settings.json").write_text(json.dumps({**comp.settings, "a": args.a, "b": args.b, "include": args.include,
                                                   "rescore": phases, "relax": relax, "fmax": args.fmax,
                                                   "steps": args.steps}, indent=2) + "\n", encoding="utf-8")
    s = comp.summary[0] if comp.summary else {}
    print(f"{len(comp.molecules)} molecules compared ({len(comp.only_a)} only in {a_name}, {len(comp.only_b)} only in "
          f"{b_name}) -> {out}")
    print(f"  same lowest conformer: {s.get('same_best', 0)}/{s.get('n_with_bests', 0)}; {a_name}'s lowest found by "
          f"{b_name}: {s.get('a_best_found', 0)}; search {s.get('search', 0)}, ranking {s.get('ranking', 0)}")
    for p in paths:
        print(f"  {p.name}")
    return 0
