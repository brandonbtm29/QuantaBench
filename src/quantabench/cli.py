"""Command line interface.

    quantabench-cli doctor [--bench] [--json]
    quantabench-cli build "OC(=O)C(F)(F)..." PFOA --method aimnet2 --ph 7
    quantabench-cli build --file cohort.csv --smiles-col SMILES --id-col DTXSID
    quantabench-cli slab --nx 16 --ny 9 --layers 3 --molecule "OC(=O)C(F)(F)..." -o pfoa_on_graphite.data
    quantabench-cli settings [--set-dir ~/SyncThing/QuantaBench/Settings]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from . import __version__, resources
from . import settings as settings_mod


def _doctor(args) -> int:
    rep = resources.detect()
    if args.json:
        print(json.dumps(rep.to_dict(), indent=2))
        return 0
    print(f"QuantaBench {__version__}")
    print(f"  OS            {rep.os}{' (WSL2)' if rep.wsl else ''}, Python {rep.python}")
    print(f"  CPU           {rep.cpu_name}, {rep.physical_cores} cores / {rep.logical_cores} threads, "
          f"{rep.ram_gb:.0f} GB RAM")
    print(f"  PyTorch       {rep.torch_version or 'not installed'} (backend: {rep.torch_backend})")
    for g in rep.gpus:
        print(f"  GPU {g.index}         {g.name} [{g.arch}] {g.total_memory_gb:.1f} GB via {g.backend.upper()}")
    print("  Libraries")
    broken = []
    for label, mod, ok, msg in resources.import_checks():
        print(f"    {'OK ' if ok else '-- '} {label:30s} {'' if ok else msg}")
        if not ok:
            broken.append(mod)
            hint = resources.explain_import_error(msg)
            if hint:
                print(f"         {hint}")
    from .chem import engines

    print("  Engines")
    for key, info in engines.ENGINES.items():
        ok, why = engines.available(key)
        print(f"    {'OK ' if ok else '-- '} {info.label:30s} {why}")
    print("  Optional      " + ", ".join(f"{k}={'yes' if v else 'no'}" for k, v in rep.optional_packages.items()))
    print(f"  Settings      {settings_mod.settings_dir() / settings_mod.SETTINGS_FILE}")
    for note in rep.notes:
        print(f"  note: {note}")
    if args.bench:
        if any(m.startswith("rdkit") or m.startswith("ase") for m in broken):
            print("\nBenchmark skipped: fix the library errors above first.")
        else:
            _bench(rep)
    return 1 if broken else 0


def _bench(rep) -> None:
    if not resources.has_package("torch"):
        print("\nBenchmark skipped: PyTorch not installed.")
        return
    print("\nBenchmark")
    devices = ["cpu"] + [f"cuda:{g.index}" for g in rep.gpus if g.backend in ("cuda", "rocm")]
    for dev in devices:
        try:
            print(f"  matmul {dev:8s} {resources.matmul_benchmark(dev):9.1f} GFLOP/s")
        except Exception as exc:
            print(f"  matmul {dev:8s} FAILED: {exc}")
    import time

    from .chem import engines
    from .chem.inputs import from_smiles
    from .chem.pipeline import BuildConfig, run_molecule

    pfoa = from_smiles("OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F", id="PFOA")
    for key in ("gfn1-xtb", "gfn2-xtb", "aimnet2"):
        ok, why = engines.available(key)
        if not ok:
            print(f"  {key:9s} skipped ({why})")
            continue
        for dev in devices:
            cfg = BuildConfig(method=key, max_conformers=4, refine_top=2, write_files=False, xtb_properties=False)
            t = time.perf_counter()
            try:
                res = run_molecule(pfoa, cfg, device=dev, threads=resources.current_threads())
            except Exception as exc:  # e.g. a DLL blocked while importing an engine
                print(f"  {key:9s} {dev:8s} FAILED: {type(exc).__name__}: {exc}")
                continue
            dt = time.perf_counter() - t
            status = res.status if res.device == dev else f"{res.status}, fell back to {res.device}"
            print(f"  {key:9s} {dev:8s} PFOA anion, 2 conformers: {dt:6.1f} s ({status})"
                  + (f" {res.error}" if res.error else ""))


def _build(args) -> int:
    from .chem import inputs as inp_mod
    from .chem.pipeline import BuildConfig, run

    items = []
    for f in args.file or []:
        items += inp_mod.read_file(f, args.smiles_col, args.id_col)
    for text in args.molecules:
        items += inp_mod.parse_text(text)
    if not items:
        print("Nothing to build: give SMILES, names, or --file.", file=sys.stderr)
        return 2
    s = settings_mod.load()
    cfg = BuildConfig.from_settings(s)
    overrides = {k: v for k, v in {
        "method": args.method, "ph": args.ph, "max_conformers": args.max_confs, "refine_top": args.refine_top,
        "energy_window_kcal": args.window, "threads": args.threads, "gpu": args.gpu, "fmax": args.fmax,
        "output_dir": args.out, "protonation_method": args.protonation,
    }.items() if v is not None}
    if args.no_protonate:
        overrides["protonate"] = False
    if args.formats:
        overrides["formats"] = tuple(x.strip() for x in args.formats.split(",") if x.strip())
    cfg = replace(cfg, **overrides)

    def progress(i, n, stage, ident):
        if stage == "start":
            print(f"[{i + 1}/{n}] {ident}")

    results, run_dir = run(items, cfg, progress=progress, log=print)
    ok = sum(r.status == "ok" for r in results)
    print(f"\n{ok}/{len(results)} succeeded." + (f" Output: {run_dir}" if run_dir else ""))
    return 0 if ok == len(results) else 1


def _slab(args) -> int:
    from ase.io import write

    from .build import surfaces
    from .chem.export import mol_to_atoms

    slab = surfaces.graphite_slab(args.nx, args.ny, args.layers, args.vacuum, args.fix_layers)
    system = slab
    if args.molecule:
        from .chem.inputs import parse_text, read_file
        from .chem.pipeline import BuildConfig, run_molecule

        src = Path(args.molecule)
        items = read_file(src) if src.suffix and src.is_file() else parse_text(args.molecule)
        cfg = replace(BuildConfig.from_settings(settings_mod.load()), method=args.method or "auto", write_files=False)
        threads = resources.apply_thread_limits(cfg.threads)
        device = resources.select_device(cfg.gpu, cfg.gpu_index)
        res = run_molecule(items[0], cfg, device, threads, log=print)
        if res.status != "ok":
            print(res.error, file=sys.stderr)
            return 1
        mol_atoms = mol_to_atoms(res.mol, res.conformers[0].conf_id)
        system = surfaces.place_on_surface(slab, mol_atoms, args.height, args.orientation)
        charges = res.conformers[0].charges
        if charges is not None:
            q = [0.0] * len(slab) + list(charges)
            system.set_initial_charges(q)
    clashes = surfaces.min_distance_report(system, 1.0)
    if clashes:
        print(f"warning: {len(clashes)} atom pairs closer than 1.0 Å", file=sys.stderr)
    out = Path(args.output)
    if out.suffix in (".data", ".lmp"):
        write(out, system, format="lammps-data", atom_style="charge", units="real", masses=True)
    else:
        write(out, system)
    print(f"Wrote {len(system)} atoms to {out} (cell {system.cell.lengths().round(2).tolist()} Å)")
    return 0


def _settings(args) -> int:
    if args.set_dir is not None:
        new = settings_mod.set_settings_dir(args.set_dir or None)
        print(f"Settings folder: {new}")
    path = settings_mod.settings_dir() / settings_mod.SETTINGS_FILE
    s = settings_mod.load()
    if args.init and not path.exists():
        settings_mod.save(s)
    print(f"# {path}{'' if path.exists() else ' (not created yet: defaults shown)'}")
    print(json.dumps(s.to_dict(), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="quantabench-cli", description="QuantaBench command line tools")
    p.add_argument("--version", action="version", version=f"QuantaBench {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("doctor", help="report hardware, GPU backend, engines and settings")
    d.add_argument("--bench", action="store_true", help="also time matmul and each engine on CPU and GPU")
    d.add_argument("--json", action="store_true")
    d.set_defaults(func=_doctor)

    b = sub.add_parser("build", help="SMILES/name/file -> conformers -> optimized geometries + descriptors")
    b.add_argument("molecules", nargs="*", help="SMILES (optionally 'SMILES id') or names to look up in PubChem")
    b.add_argument("--file", action="append", help="SDF, MOL, XYZ, PDB, SMI or CSV (repeatable)")
    b.add_argument("--smiles-col"), b.add_argument("--id-col")
    b.add_argument("--method", choices=["auto", "mmff", "gfn1-xtb", "gfn2-xtb", "aimnet2"])
    b.add_argument("--ph", type=float)
    b.add_argument("--no-protonate", action="store_true", help="keep the input charge state")
    b.add_argument("--protonation", choices=["rules", "dimorphite"])
    b.add_argument("--max-confs", type=int)
    b.add_argument("--refine-top", type=int)
    b.add_argument("--window", type=float, help="energy window, kcal/mol")
    b.add_argument("--fmax", type=float)
    b.add_argument("--threads", type=int, help="CPU threads (0 = all but one)")
    b.add_argument("--gpu", choices=["auto", "on", "off"])
    b.add_argument("--out", help="output folder (default from settings)")
    b.add_argument("--formats", help="comma list: xyz,xyz-all,extxyz,sdf,pdb,mol,lammps")
    b.set_defaults(func=_build)

    s = sub.add_parser("slab", help="graphite slab, optionally with a molecule placed above it")
    s.add_argument("--nx", type=int, default=16), s.add_argument("--ny", type=int, default=9)
    s.add_argument("--layers", type=int, default=3)
    s.add_argument("--vacuum", type=float, default=30.0)
    s.add_argument("--fix-layers", type=int, default=1)
    s.add_argument("--molecule", help="SMILES, name, or structure file to place on the surface")
    s.add_argument("--method", choices=["auto", "mmff", "gfn1-xtb", "gfn2-xtb", "aimnet2"])
    s.add_argument("--height", type=float, default=3.5)
    s.add_argument("--orientation", choices=["flat", "upright", "as-is"], default="flat")
    s.add_argument("-o", "--output", default="system.extxyz")
    s.set_defaults(func=_slab)

    st = sub.add_parser("settings", help="show settings, or move them (e.g. into a synced folder)")
    st.add_argument("--set-dir", nargs="?", const="", help="new settings folder ('' = default)")
    st.add_argument("--init", action="store_true", help="write settings.json with defaults if missing")
    st.set_defaults(func=_settings)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
