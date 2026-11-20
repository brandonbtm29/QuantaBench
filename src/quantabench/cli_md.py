"""``quantabench-cli md liquid-slab`` and ``md surface-tension`` (MD-2): liquid-vapour slabs of pure liquids in
LAMMPS and their surface tension.

    quantabench-cli md liquid-slab --smiles "FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F" --name c6f14 -o DIR
        [--ff oplsaa|openff|gaff2] [--dispersion ewald|cutoff12|cutoff16|cutoff18] [--size pilot|full]
        [--molecules N] [--sigma-ref 12.33 --density-ref 1.67] [--run]
    quantabench-cli md liquid-slab --list-ff
    quantabench-cli md surface-tension DIR [--skip-ns 2] [--block-ns 1] [--json]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def register(msub) -> None:
    ls = msub.add_parser("liquid-slab", help="LAMMPS liquid-vapour slab of a pure liquid (bulk NPT, then NVT slab) "
                                             "for its surface tension")
    ls.add_argument("--smiles", help="the molecule (neutral; one fragment)")
    ls.add_argument("--name", default="", help="a short name for files and reports")
    ls.add_argument("-o", "--out", help="folder for system.data, ff.params, in.bulk, in.slab, README.txt")
    ls.add_argument("--ff", default="oplsaa", help="force field: oplsaa (default), openff, gaff2 (--list-ff)")
    ls.add_argument("--offxml", default=None, help="with --ff openff: the SMIRNOFF file (default openff-2.3.0.offxml)")
    ls.add_argument("--overrides", help="JSON file of Lennard-Jones overrides by atom type: "
                                        '{"lj": {"f": {"epsilon": .., "sigma": ..}}, "source": "..."}')
    ls.add_argument("--dispersion", default="ewald",
                    help="ewald (lj/long/coul/long + pppm/disp; default), cutoff12, cutoff16, cutoff18 "
                         "(lj/cut/coul/long + pppm)")
    ls.add_argument("--size", choices=["pilot", "full"], default="pilot",
                    help="pilot: 4 x 4 nm lateral, full: 5 x 5 nm; both a 10 nm liquid slab")
    ls.add_argument("--molecules", type=int, help="number of molecules (sets the slab thickness; the lateral size stays)")
    ls.add_argument("--lateral", type=float, help="lateral box edge, Å (overrides --size)")
    ls.add_argument("--thickness", type=float, help="liquid slab thickness at the packing density, Å")
    ls.add_argument("--vacuum", type=float, default=50.0, help="vacuum on each side of the slab, Å (default 50)")
    ls.add_argument("--temperature", type=float, default=298.15, help="K (default 298.15)")
    ls.add_argument("--density", type=float, help="g/cm³ to pack at (default: QuantaBench's estimate)")
    ls.add_argument("--packing", choices=["auto", "packmol", "lattice"], default="auto",
                    help="auto: Packmol if found, else a lattice and an NVT compression")
    ls.add_argument("--bulk-ns", type=float, default=1.0)
    ls.add_argument("--settle-ns", type=float, default=2.0)
    ls.add_argument("--production-ns", type=float, default=10.0)
    ls.add_argument("--seed", type=int, default=20261120)
    ls.add_argument("--sigma-ref", type=float, help="measured σ (mN/m) the report compares against")
    ls.add_argument("--density-ref", type=float, help="measured density (g/cm³) the report compares against")
    ls.add_argument("--ref-note", default="", help="where the reference values come from")
    ls.add_argument("--list-ff", action="store_true", help="list the force fields and whether they are installed")
    ls.add_argument("--run", action="store_true",
                    help="run both stages now with the LAMMPS Python module (blocks; for short checks)")
    ls.add_argument("--run-var", action="append", default=[], metavar="NAME=VALUE",
                    help="with --run: LAMMPS variables, e.g. --run-var bulk_ns=0.02 --run-var production_ns=0.05")
    ls.set_defaults(func=_liquid_slab)

    st = msub.add_parser("surface-tension", help="σ ± error, densities and checks of a liquid-slab folder")
    st.add_argument("dir")
    st.add_argument("--skip-ns", type=float, help="time skipped before averaging (default: the setup's settle time)")
    st.add_argument("--block-ns", type=float, default=1.0, help="block length for the error bar (default 1 ns)")
    st.add_argument("--json", action="store_true", help="print the full JSON report")
    st.set_defaults(func=_surface_tension)


def _liquid_slab(args) -> int:
    from .md import forcefield as F
    from .md import liquid as L

    if args.list_ff:
        F.describe_backends()
        return 0
    if not args.smiles or not args.out:
        print("--smiles and -o/--out are required (or --list-ff)", file=sys.stderr)
        return 2
    try:
        L.parse_dispersion(args.dispersion)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    overrides = None
    if args.overrides:
        overrides = json.loads(Path(args.overrides).read_text(encoding="utf-8"))
    try:
        mol = F.prepare_molecule(args.smiles, seed=args.seed)
        rho, how = F.estimate_density(mol)
        tm = F.type_molecule(args.smiles, args.name, args.ff, mol=mol, offxml=args.offxml or F.DEFAULT_OPENFF,
                             overrides=overrides)
    except F.ForceFieldError as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return 2
    cfg = L.LiquidSlabConfig(temperature=args.temperature, dispersion=args.dispersion, size=args.size,
                             lateral=args.lateral, slab_thickness=args.thickness, n_molecules=args.molecules,
                             vacuum=args.vacuum, density=args.density, packing=args.packing, bulk_ns=args.bulk_ns,
                             settle_ns=args.settle_ns, production_ns=args.production_ns, seed=args.seed,
                             reference_sigma=args.sigma_ref, reference_density=args.density_ref,
                             reference_note=args.ref_note)
    try:
        s = L.write_liquid_slab(tm, args.out, cfg, density_estimate=rho, density_source=how)
    except (ValueError, RuntimeError) as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {', '.join(p.name for p in s.files)} to {s.out_dir}")
    print(f"  {tm.label}")
    print(f"  {s.n_molecules} molecules ({s.n_atoms} atoms), packed by {s.packing} at {s.density_estimate:.3f} g/cm3 "
          f"({s.density_source}); box at that density {' x '.join(f'{v:.1f}' for v in s.target_box)} A")
    print(f"  dispersion {s.config.dispersion}, cutoff {s.config.cutoff:g} A, vacuum {s.config.vacuum:g} A per side, "
          f"bulk {s.config.bulk_ns:g} ns, slab {s.config.settle_ns:g} + {s.config.production_ns:g} ns")
    for n in tm.notes + s.notes:
        print(f"  note: {n}")
    print(f"  Run: cd {s.out_dir} && lmp -in in.bulk && lmp -in in.slab   (OpenMP, MPI and GPU commands: README.txt)")
    print(f"  Then: quantabench-cli md surface-tension {s.out_dir}")
    if args.run:
        from . import resources

        if not resources.has_package("lammps"):
            print("--run needs the LAMMPS Python module: pip install 'quantabench[md]'", file=sys.stderr)
            return 1
        variables = {}
        for item in args.run_var:
            k, sep, v = item.partition("=")
            if not sep:
                print(f"--run-var wants NAME=VALUE (got {item!r})", file=sys.stderr)
                return 2
            variables[k.strip()] = v.strip()
        print("Running LAMMPS (bulk, then slab; this blocks until both are done)...")
        L.run_liquid_slab(s.out_dir, variables=variables)
        return _surface_tension(_Args(dir=str(s.out_dir), skip_ns=None, block_ns=1.0, json=False))
    return 0


class _Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _surface_tension(args) -> int:
    from .md import surface_tension as S

    folder = Path(args.dir).expanduser()
    if not (folder / "slab_box.txt").is_file():
        print(f"{folder}: no slab_box.txt (has in.slab run yet?)", file=sys.stderr)
        return 2
    try:
        r = S.analyze(folder, skip_ns=args.skip_ns, block_ns=args.block_ns)
    except (ValueError, OSError) as exc:
        print(f"{folder}: {exc}", file=sys.stderr)
        return 1
    js, md = S.write_report(folder, r)
    if args.json:
        print(json.dumps(r, indent=1))
    else:
        print(S.format_report(r))
        print(f"Wrote {js.name} and {md.name} to {folder}")
    return 0
