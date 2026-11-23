"""``quantabench-cli md liquid-slab`` and ``md surface-tension`` (MD-2): liquid-vapour slabs of pure liquids in
LAMMPS and their surface tension.

    quantabench-cli md liquid-slab --smiles "FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F" --name c6f14 -o DIR
        [--ff oplsaa|openff|gaff2] [--long-range auto|ewald|cutoff] [--cutoff RC] [--purpose screen|reference]
        [--gpu auto|yes|no] [--coulomb-accuracy A] [--disp-grid spacing|accuracy] [--disp-gewald G]
        [--disp-spacing H] [--disp-accuracy REAL KSPACE] [--dispersion ewald|cutoff12|cutoff16|...]
        [--size pilot|full] [--molecules N] [--sigma-ref 12.33 --density-ref 1.67] [--run]
        (the long-range options and their recommendation: md.long_range, MD-LR)
    quantabench-cli md liquid-slab --list-ff
    quantabench-cli md surface-tension DIR [--skip-ns 2] [--block-ns 1] [--json]
    quantabench-cli md estimate (DIR | --atoms N --ns NS) [--dispersion ewald] [--geometry slab|bulk]
        [--machine NAME] [--all-variants] [--include-unverified] [--json]
                                                 run time from each machine's MD benchmark (MD-BENCH-1/2)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

LR_PURPOSES = ("screen", "reference")


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
    lr = ls.add_argument_group("long-range interactions (md.long_range; without a choice the recommendation for this "
                               "setup is used, and it is printed either way)")
    lr.add_argument("--long-range", default="auto", metavar="METHOD",
                    help="auto (the recommendation; default), ewald (full long-range dispersion: lj/long/coul/long + "
                         "pppm/disp, CPU only) or cutoff (LJ cutoff + analytic tail: lj/cut/coul/long + pppm, GPU or "
                         "CPU)")
    lr.add_argument("--cutoff", type=float, metavar="RC",
                    help="LJ cutoff (screening length), Å, 8-30: the real-space cutoff of the LJ and Coulomb terms "
                         "(default: ewald 12, cutoff 16)")
    lr.add_argument("--purpose", choices=sorted(LR_PURPOSES), default="screen",
                    help="what the recommendation aims at: screen (σ to a few %%, the fastest on the machine; "
                         "default) or reference (numerically converged full long-range dispersion)")
    lr.add_argument("--gpu", default="auto", metavar="auto|yes|no",
                    help="can LAMMPS use a GPU on the machine that runs the folder? auto (default): this computer's "
                         "LAMMPS (lmp -h: the GPU package and a GPU; the MD benchmark's CPU/GPU check)")
    lr.add_argument("--coulomb-accuracy", type=float, metavar="A",
                    help="Coulomb PPPM relative accuracy (default: ewald 1e-6, cutoff 1e-5)")
    lr.add_argument("--disp-grid", choices=["spacing", "accuracy"],
                    help="ewald's dispersion k-space: spacing (G from the cutoff and a mesh from the box; the "
                         "slab-safe default) or accuracy (LAMMPS picks both from --disp-accuracy)")
    lr.add_argument("--disp-gewald", type=float, metavar="G", help="ewald, spacing: dispersion G, 1/Å (default 2.77/rc)")
    lr.add_argument("--disp-spacing", type=float, metavar="H",
                    help="ewald, spacing: largest dispersion-mesh spacing, Å (default 0.52/G)")
    lr.add_argument("--disp-accuracy", type=float, nargs=2, metavar=("REAL", "KSPACE"),
                    help="ewald, accuracy: force/disp/real and force/disp/kspace, kcal/mol/Å (default 1e-5 2e-5; "
                         "implies --disp-grid accuracy)")
    lr.add_argument("--dispersion", default=None,
                    help="the method and cutoff in one word, as before: ewald, cutoff12, cutoff16, cutoff18, ... "
                         "(= --long-range cutoff --cutoff 16)")
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

    es = msub.add_parser("estimate", help="run time of an MD job on each machine, from its MD benchmark "
                                          "(quantabench-cli machines --bench md)")
    es.add_argument("dir", nargs="?", help="a liquid-slab folder (its atoms, bulk + slab ns and dispersion)")
    es.add_argument("--atoms", type=int, help="atoms of the job")
    es.add_argument("--ns", type=float, help="simulated time, ns")
    es.add_argument("--dispersion", help="ewald, cutoff12, cutoff16, ... (default: any measured variant)")
    es.add_argument("--geometry", choices=["slab", "bulk"], default="slab",
                    help="the job's: slab (a liquid-slab run, the default) uses the slab benchmark, bulk the bulk "
                         "box's; a machine without that one falls back to the other, noted")
    es.add_argument("--machine", help="only this machine (default: every machine in the registry)")
    es.add_argument("--all-variants", action="store_true", help="every measured variant, not only the fastest")
    es.add_argument("--include-unverified", action="store_true",
                    help="also variants whose CPU/GPU pressure check failed")
    es.add_argument("--json", action="store_true")
    es.set_defaults(func=_estimate)


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
        explicit = _long_range_choice(args)
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
    cfg = L.LiquidSlabConfig(temperature=args.temperature, size=args.size,
                             lateral=args.lateral, slab_thickness=args.thickness, n_molecules=args.molecules,
                             vacuum=args.vacuum, density=args.density, packing=args.packing, bulk_ns=args.bulk_ns,
                             settle_ns=args.settle_ns, production_ns=args.production_ns, seed=args.seed,
                             reference_sigma=args.sigma_ref, reference_density=args.density_ref,
                             reference_note=args.ref_note)
    try:
        cfg, record = _apply_long_range(cfg, explicit, args, tm, args.density or rho)
    except ValueError as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return 2
    try:
        s = L.write_liquid_slab(tm, args.out, cfg, density_estimate=rho, density_source=how, long_range=record)
    except (ValueError, RuntimeError) as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {', '.join(p.name for p in s.files)} to {s.out_dir}")
    print(f"  {tm.label}")
    print(f"  {s.n_molecules} molecules ({s.n_atoms} atoms), packed by {s.packing} at {s.density_estimate:.3f} g/cm3 "
          f"({s.density_source}); box at that density {' x '.join(f'{v:.1f}' for v in s.target_box)} A")
    print(f"  dispersion {s.config.dispersion}, cutoff {s.config.cutoff:g} A, vacuum {s.config.vacuum:g} A per side, "
          f"bulk {s.config.bulk_ns:g} ns, slab {s.config.settle_ns:g} + {s.config.production_ns:g} ns")
    _print_long_range(s.config, record)
    for n in tm.notes + s.notes:
        print(f"  note: {n}")
    _runtime_estimate(s)
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


def _long_range_choice(args) -> dict:
    """The long-range settings the user chose explicitly (only those): method, cutoff, accuracies."""
    from .md import long_range as LR

    out: dict = {}
    if args.dispersion:
        d = args.dispersion.strip().lower()
        method = LR.method_of(d)
        if method == "cutoff":
            rc = d[len("cutoff"):]
            try:
                out["cutoff"] = LR.check_cutoff(float(rc))
            except ValueError:
                raise ValueError(f"--dispersion {args.dispersion!r}: ewald, or cutoff12 / cutoff16 / cutoff18 "
                                 "(any cutoffN, 8-30 Å)") from None
        elif d != "ewald":
            raise ValueError(f"--dispersion {args.dispersion!r}: ewald, or cutoff12 / cutoff16 / cutoff18")
        out["method"] = method
    if args.long_range and args.long_range.strip().lower() != "auto":
        m = LR.normalize_method(args.long_range)
        if "method" in out and out["method"] != m:
            raise ValueError(f"--dispersion {args.dispersion} and --long-range {args.long_range} disagree")
        out["method"] = m
    if args.cutoff is not None:
        rc = LR.check_cutoff(args.cutoff)
        if "cutoff" in out and abs(out["cutoff"] - rc) > 1e-9:
            raise ValueError(f"--dispersion {args.dispersion} and --cutoff {args.cutoff:g} disagree")
        out["cutoff"] = rc
    if args.coulomb_accuracy is not None:
        if not 0 < args.coulomb_accuracy < 1:
            raise ValueError("--coulomb-accuracy: a relative accuracy between 0 and 1, e.g. 1e-6")
        out["kspace_accuracy"] = args.coulomb_accuracy
    if args.disp_accuracy:
        out["disp_grid"] = "accuracy"
        out["disp_real_accuracy"], out["disp_kspace_accuracy"] = (float(x) for x in args.disp_accuracy)
    if args.disp_grid:
        if args.disp_accuracy and args.disp_grid != "accuracy":
            raise ValueError("--disp-accuracy goes with --disp-grid accuracy")
        out["disp_grid"] = args.disp_grid
    for k in ("disp_gewald", "disp_spacing"):
        v = getattr(args, k)
        if v is not None:
            if v <= 0:
                raise ValueError(f"--{k.replace('_', '-')} must be positive")
            out[k] = v
    if any(k.startswith("disp_") for k in out):
        out.setdefault("method", "ewald")           # the --disp-* options choose full long-range dispersion
    if out.get("disp_grid") == "accuracy" and ("disp_gewald" in out or "disp_spacing" in out):
        raise ValueError("--disp-gewald and --disp-spacing go with --disp-grid spacing")
    if out.get("method") == "cutoff" and any(k.startswith("disp_") for k in out):
        raise ValueError("the --disp-* options are for --long-range ewald (pppm/disp)")
    return out


def _apply_long_range(cfg, explicit: dict, args, tm, density: float):
    """(config with the long-range settings, the record for README/JSON): the recommendation for this setup, and the
    user's explicit choices on top of it."""
    from dataclasses import replace

    from .md import liquid as L
    from .md import long_range as LR

    base = cfg.resolved()
    n, box = L.plan_box(tm, density, base, check=False)
    slab_box = (box[0], box[1], box[2] + 2.0 * float(base.vacuum))
    machine = LR.machine_for(args.gpu)
    rec = LR.recommend_long_range("slab", slab_box, n_molecules=n, n_atoms=n * tm.n_atoms, machine=machine,
                                  purpose=args.purpose)
    method = explicit.get("method", rec.method)
    if "cutoff" in explicit:
        rc = explicit["cutoff"]
    elif method == rec.method:
        rc = rec.cutoff
    else:
        rc = LR.DEFAULT_CUTOFF[method]
    chosen = LR.settings(method, rc, **{k: v for k, v in explicit.items() if k not in ("method", "cutoff")})
    new = replace(cfg, **LR.config_kwargs(chosen))
    record = {"chosen_by": "user" if explicit else "recommendation", "explicit": dict(explicit),
              "machine": machine.to_dict(), "purpose": args.purpose, "recommendation": rec.to_dict(),
              "differences": LR.differences(chosen, rec),
              "warnings": LR.check_settings(chosen, "slab", slab_box, machine, args.purpose),
              "notes": LR.notes_for(chosen, "slab", machine)}
    return new, record


def _print_long_range(cfg, record: dict) -> None:
    from .md import long_range as LR

    rec = record["recommendation"]
    if record["machine"].get("note"):
        print(f"  machine: {record['machine']['note']}")
    print(f"  {rec['summary']}")
    chosen = LR.settings_of(cfg)
    text = LR.describe(chosen, rec["inputs"].get("box"))
    if record["chosen_by"] == "recommendation":
        print(f"  Using it: {text} (choose with --long-range, --cutoff, --coulomb-accuracy, --disp-*)")
    elif record["differences"]:
        print(f"  Your choice: {text}")
        print(f"  It differs from the recommendation in: {'; '.join(record['differences'])}")
    else:
        print(f"  Your choice is the recommendation: {text}")
    for w in record["warnings"]:
        print(f"  warning: {w}")
    for n in record.get("notes", []):
        print(f"  note: {n}")


class _Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _runtime_estimate(s) -> None:
    """Print, and add to README.txt, how long bulk + slab take on each machine with an MD benchmark (MD-BENCH-1)."""
    from .md import bench

    ns = s.config.bulk_ns + s.config.settle_ns + s.config.production_ns
    lx, ly, lz = s.target_box
    try:
        lines = bench.estimate_lines(s.n_atoms, ns, s.config.dispersion, geometry="slab",
                                     box=(lx, ly, lz + 2.0 * float(s.config.vacuum)))
    except Exception as exc:  # noqa: BLE001 - an estimate never stops a setup
        lines = []
        print(f"  Run time: no estimate ({type(exc).__name__}: {exc})")
    if not lines:
        print("  Run time: no machine has an MD benchmark yet (quantabench-cli machines --bench md on each)")
        return
    head = (f"{ns:g} ns of {s.n_atoms:,} atoms ({s.config.dispersion}), from each machine's MD benchmark "
            "(quantabench-cli machines --bench md):")
    print(f"  Run time, {head}")
    for line in lines:
        print(f"    {line}")
    readme = s.out_dir / "README.txt"
    note = ("The hours come from each machine's slab benchmark (a bulk-box benchmark leaves out the slab's vacuum "
            "mesh: a lower bound) and include the bulk stage at the slab's speed. quantabench-cli md estimate <this "
            "folder> repeats this with newer benchmarks.")
    with open(readme, "a", encoding="utf-8") as fh:
        fh.write("\nRUN TIME ESTIMATE\n  " + head + "\n" + "".join(f"    {x}\n" for x in lines) + f"  {note}\n")


def _estimate(args) -> int:
    from .dist import profile as profile_mod
    from .md import bench

    atoms, ns, dispersion = args.atoms, args.ns, args.dispersion
    geometry, box = getattr(args, "geometry", "slab") or "slab", None
    if args.dir:
        try:
            meta = json.loads((Path(args.dir).expanduser() / "liquid_setup.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"{args.dir}: no readable liquid_setup.json ({exc})", file=sys.stderr)
            return 2
        cfg = meta.get("config") or {}
        atoms = atoms or int(meta.get("n_atoms") or 0)
        ns = ns or sum(float(cfg.get(k) or 0.0) for k in ("bulk_ns", "settle_ns", "production_ns"))
        dispersion = dispersion or cfg.get("dispersion")
        target = meta.get("box_target_A") or []
        if geometry == "slab" and len(target) == 3:
            box = (float(target[0]), float(target[1]), float(target[2]) + 2.0 * float(cfg.get("vacuum") or 0.0))
    if not atoms or not ns or atoms <= 0 or ns <= 0:
        print("give a liquid-slab folder, or --atoms and --ns", file=sys.stderr)
        return 2
    if dispersion:
        try:
            dispersion = bench.parse_variant(f"cpu-{dispersion}").dispersion
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 2
    profiles = [p for p in profile_mod.load_profiles() if not args.machine or p.name == args.machine]
    rows = []
    for p in profiles:
        est = bench.estimate_hours(p.md_throughput, atoms, ns, dispersion=dispersion,
                                   include_unverified=args.include_unverified, geometry=geometry, box=box)
        for e in (est if args.all_variants else est[:1]):
            rows.append({"machine": p.name, "variant": e.variant, "geometry": e.geometry, "hours": round(e.hours, 2),
                         "ns_per_day": round(e.ns_per_day, 4), "measured_ns_per_day": e.measured_ns_per_day,
                         "measured_atoms": e.measured_atoms, "threads": e.threads, "check": e.check, "note": e.note})
    if args.json:
        print(json.dumps({"atoms": atoms, "ns": ns, "dispersion": dispersion, "geometry": geometry, "box_A": box,
                          "estimates": rows}, indent=1))
        return 0
    print(f"{ns:g} ns of {atoms:,} atoms ({dispersion or 'any dispersion'}, {geometry}), from each machine's MD "
          "benchmark:")
    if not rows:
        print("  no machine has a matching MD benchmark (quantabench-cli machines --bench md on each)")
        return 0
    for r in rows:
        flag = "" if r["check"] in ("", "ok") else f"  [pressure check: {r['check']}: unverified]"
        print(f"  {r['machine']:24s} {r['variant']:14s} {bench.format_hours(r['hours']):>18s}  "
              f"({bench.fmt_ns(r['ns_per_day'])} ns/day here; measured {bench.fmt_ns(r['measured_ns_per_day'])} on "
              f"{r['measured_atoms']:,} atoms, {r['geometry']} box" + (f", {r['note']}" if r["note"] else "") + ")"
              + flag)
    print("  Hours scale the benchmark's ns/day by its atoms / these atoms (with the job's box, its k-space share by "
          "the box volume: docs/WORKERS.md, \"The MD benchmark\").")
    return 0


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
