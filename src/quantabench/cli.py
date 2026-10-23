"""Command line interface.

    quantabench-cli doctor [--bench] [--json]
    quantabench-cli build "OC(=O)C(F)(F)..." PFOA --method aimnet2 --ph 7 [--freq]
    quantabench-cli build --file cohort.xlsx --smiles-pref qsar --chunk 1/3     # resumable batch run
    quantabench-cli build --resume RUN_DIR [--retry-failed]
    quantabench-cli status RUN_DIR | merge OUT_DIR RUN_DIR...
    quantabench-cli freq "OC(=O)C(F)(F)F TFA" --method gfn1-xtb                # frequencies + thermo + IR
    quantabench-cli slab --nx 16 --ny 9 --layers 3 --molecule "OC(=O)C(F)(F)..." -o pfoa_on_graphite.data
                         [--lammps-dir DIR]
    quantabench-cli md smd --system pfoa_graphite.extxyz -o DIR [--gpu] [--run]
    quantabench-cli md parse DIR
    quantabench-cli settings [--set-dir ~/SyncThing/QuantaBench/Settings]

Batch runs: ``build`` uses the resumable job table (``chem.jobs``: ``jobs.sqlite`` in the run folder)
whenever more than one molecule is built or ``--run-dir``/``--resume``/``--chunk`` is given, so an
interrupted batch continues where it stopped. A single molecule without those options runs directly.
"""

from __future__ import annotations

import argparse
import json
import socket
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


METHODS = ["auto", "mmff", "gfn1-xtb", "gfn2-xtb", "aimnet2"]


def _read_inputs(args) -> list:
    from .chem import inputs as inp_mod

    items = []
    for f in args.file or []:
        items += inp_mod.read_file(f, args.smiles_col, args.id_col, sheet=getattr(args, "sheet", None),
                                   smiles_preference=getattr(args, "smiles_pref", None), log=print)
    for text in args.molecules:
        items += inp_mod.parse_text(text)
    return items


def _config(args, base=None, **extra):
    from .chem.pipeline import BuildConfig

    cfg = base or BuildConfig.from_settings(settings_mod.load())
    overrides = {k: v for k, v in {
        "method": args.method, "ph": getattr(args, "ph", None), "max_conformers": getattr(args, "max_confs", None),
        "refine_top": getattr(args, "refine_top", None), "energy_window_kcal": getattr(args, "window", None),
        "threads": getattr(args, "threads", None), "gpu": getattr(args, "gpu", None),
        "fmax": getattr(args, "fmax", None), "output_dir": getattr(args, "out", None),
        "protonation_method": getattr(args, "protonation", None), "batched": getattr(args, "batched", None),
        "freq_hessian": getattr(args, "freq_hessian", None), "symmetry_number": getattr(args, "symmetry_number", None),
        "temperature_k": getattr(args, "temperature", None),
    }.items() if v is not None}
    # --protonate/--no-protonate and --freq/--no-freq: None = keep settings.json (the GUI saves its
    # checkboxes there), so a box ticked in the GUI can always be overridden from the command line
    if getattr(args, "protonate", None) is not None:
        overrides["protonate"] = args.protonate
    if getattr(args, "freq", None) is not None:
        overrides["frequencies"] = args.freq
    if getattr(args, "formats", None):
        overrides["formats"] = tuple(x.strip() for x in args.formats.split(",") if x.strip())
    cfg = replace(cfg, **overrides, **extra)
    if cfg.frequencies and "spectrum" not in cfg.formats:
        cfg = replace(cfg, formats=(*cfg.formats, "spectrum"))
    return cfg


def _progress(i, n, stage, ident):
    if stage == "start":
        print(f"[{i + 1}/{n}] {ident}")


def _build(args) -> int:
    from .chem import jobs
    from .chem.pipeline import BuildConfig, new_run_dir, run

    run_dir = args.resume or args.run_dir
    if args.resume and not (Path(args.resume).expanduser() / jobs.DB_NAME).is_file():
        print(f"{args.resume} is not a QuantaBench run folder (no {jobs.DB_NAME}).", file=sys.stderr)
        return 2
    try:
        items = _read_inputs(args)
        chunk = jobs.parse_chunk(args.chunk) if args.chunk else None
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    if not items and not args.resume:
        print("Nothing to build: give SMILES, names, --file, or --resume RUN_DIR.", file=sys.stderr)
        return 2
    base = None
    if args.resume:  # continue with the science settings the run was started with; options given now still win
        stored = jobs.open_run(args.resume).config
        known = {k: stored[k] for k in jobs.SCIENCE_FIELDS if k in stored}
        base = replace(BuildConfig.from_settings(settings_mod.load()), **known)
        if "formats" in stored:
            base = replace(base, formats=tuple(stored["formats"]))
    cfg = _config(args, base)

    resumable = bool(run_dir or chunk or args.retry_failed or len(items) > 1)
    if not resumable:
        results, out = run(items, cfg, progress=_progress, log=print)
    else:
        if run_dir is None:
            label = items[0].id if len(items) == 1 else f"batch_{len(items)}"
            if chunk:
                label += f"_{jobs.chunk_label(chunk)}_{socket.gethostname()}"
            run_dir = new_run_dir(cfg.output_dir or settings_mod.load().output_path(), label)
        try:
            results, out = jobs.run_resumable(items or None, cfg, run_dir, progress=_progress, log=print,
                                              retry_failed=args.retry_failed, chunk=chunk, force=args.force)
        except (jobs.ConfigMismatchError, jobs.RunInUseError) as exc:
            print(exc, file=sys.stderr)
            return 2
        print(jobs.open_run(out).summary())
    ok = sum(r.status == "ok" for r in results)
    print(f"\n{ok}/{len(results)} succeeded in this call." + (f" Output: {out}" if out else ""))
    return 0 if ok == len(results) else 1


def _status(args) -> int:
    from .chem import jobs

    try:
        st = jobs.open_run(args.run_dir)
    except (ValueError, FileNotFoundError, OSError) as exc:
        print(exc, file=sys.stderr)
        return 2
    print(f"{st.run_dir}\n  {st.summary()}\n  started {st.created} with QuantaBench {st.version}, "
          f"method {st.config.get('method')}")
    for ident, err in st.failures:
        print(f"  failed: {ident}: {err}")
    return 0


def _merge(args) -> int:
    from .chem import jobs

    problems: list[str] = []

    def note(msg: str) -> None:
        problems.append(msg)
        print(msg, file=sys.stderr)

    written = jobs.merge_tables(args.run_dirs, args.out, log=note)
    for p in written:
        print(f"wrote {p}")
    if not written:
        print("Nothing merged.", file=sys.stderr)
    return 0 if written and not problems else 1


def _print_vibrations(name: str, vib: dict) -> None:
    t = vib.get("thermo") or {}
    print(f"\n{name}: {vib['n_modes']} modes ({vib['hessian_method']} Hessian, {vib['method']}), "
          f"{vib['n_imaginary']} imaginary, lowest {vib['lowest_freq_cm1']:.1f} cm-1")
    if t:
        print(f"  T = {t['temperature_k']:.2f} K, symmetry number {t['symmetry_number']}, "
              f"{'qRRHO' if t['qrrho'] else 'RRHO'} entropy")
        for label, key, unit in (("ZPE", "zpe_kcal", "kcal/mol"), ("H - E", "h_corr_kcal", "kcal/mol"),
                                 ("G - E", "g_corr_kcal", "kcal/mol"), ("S", "entropy_cal_molK", "cal/(mol K)"),
                                 ("E", "potential_energy_ev", "eV"), ("H", "enthalpy_kcal", "kcal/mol"),
                                 ("G", "gibbs_kcal", "kcal/mol")):
            print(f"  {label:6s} {t[key]:14.4f} {unit}")
    ints = vib.get("ir_intensities_km_mol")
    if ints:
        top = sorted(range(len(ints)), key=lambda i: -ints[i])[:6]
        print("  strongest IR bands: " + ", ".join(f"{vib['frequencies_cm1'][i]:.0f} cm-1 ({ints[i]:.0f} km/mol)"
                                                   for i in sorted(top, key=lambda i: vib['frequencies_cm1'][i])))
    for w in vib.get("warnings", []):
        print(f"  note: {w}")


def _freq(args) -> int:
    """Frequencies, thermochemistry and IR for each molecule (best conformer), or for a geometry as is."""
    if args.as_is:
        return _freq_as_is(args)
    from .chem.pipeline import run

    items = _read_inputs(args)
    if not items:
        print("Nothing to do: give SMILES, names or --file.", file=sys.stderr)
        return 2
    cfg = _config(args, formats=tuple(f for f in ("xyz", "extxyz", "spectrum")), frequencies=True)
    results, out = run(items, cfg, progress=_progress, log=print if args.verbose else (lambda _m: None))
    ok = 0
    for r in results:
        if r.status != "ok":
            print(f"\n{r.input.id}: FAILED {r.error}")
            continue
        if not r.vibrations:
            print(f"\n{r.input.id}: no frequencies ({'; '.join(r.notes)})")
            continue
        ok += 1
        _print_vibrations(r.input.id, r.vibrations)
    if out:
        print(f"\nIR spectra, mode tables and geometries: {out}")
    return 0 if ok == len(results) else 1


def _freq_as_is(args) -> int:
    from ase.io import read

    from .chem import engines, export, vibrations

    if not args.file or args.molecules:
        print("--as-is takes structure files (--file), used without a conformer search.", file=sys.stderr)
        return 2
    s = settings_mod.load()
    device = resources.select_device(args.gpu or s.resources.gpu, s.resources.gpu_index)
    resources.apply_thread_limits(args.threads if args.threads is not None else s.resources.threads)
    method = args.method or "gfn1-xtb"
    out_dir = Path(args.out or s.output_path()).expanduser()
    status = 0
    if method != "auto":
        ok, why = engines.available(method)
        if not ok and args.method:
            print(f"{engines.ENGINES[method].label} is not available: {why}\n"
                  "Install it (see docs/INSTALL.md) or choose another --method (mmff needs nothing extra).",
                  file=sys.stderr)
            return 2
        if not ok:               # the default engine is missing on a core install: use the best available one
            print(f"{engines.ENGINES[method].label} is not available ({why}); using the best available engine")
            method = "auto"
    for f in args.file:
        stem = Path(f).stem
        try:
            atoms = read(f)
            m = engines.resolve_auto(atoms.numbers) if method == "auto" else method
            if m != method:
                print(f"{stem}: {engines.ENGINES[m].label}")
            vr = vibrations.analyze(atoms, m, args.charge, args.multiplicity, device=device,
                                    temperature=args.temperature or 298.15, symmetry_number=args.symmetry_number or 0,
                                    hessian=args.freq_hessian or "auto",
                                    preoptimize_fmax=None if args.no_opt else (args.fmax or 0.01),
                                    log=print if args.verbose else None)
        except Exception as exc:
            print(f"{stem}: FAILED {type(exc).__name__}: {exc}", file=sys.stderr)
            status = 1
            continue
        _print_vibrations(stem, vr.to_dict())
        out_dir.mkdir(parents=True, exist_ok=True)
        for p in export.write_vibrations(vr.to_dict(), out_dir, stem):
            print(f"  wrote {p}")
    return status


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
    if args.lammps_dir:
        if not args.molecule:
            print("--lammps-dir needs --molecule (the adsorbate to pull)", file=sys.stderr)
            return 2
        from .md.lammps import write_smd_setup

        try:
            setup = write_smd_setup(system, args.lammps_dir, gpu=args.lammps_gpu,
                                    title=f"{items[0].id} on graphite")
        except ValueError as exc:
            print(f"LAMMPS setup refused: {exc}", file=sys.stderr)
            return 1
        print(f"LAMMPS steered-MD setup in {setup.out_dir} (see README.txt)")
    return 0


def _md_smd(args) -> int:
    from ase.io import read

    from .md.lammps import SMDConfig, run_lammps, write_smd_setup

    atoms = read(args.system)
    try:
        setup = write_smd_setup(atoms, args.out, temperature=args.temperature, pull_velocity=args.velocity,
                                spring_k=args.spring, pull_distance=args.distance, gpu=args.gpu,
                                config=SMDConfig(graphite=args.graphite), title=Path(args.system).stem)
    except ValueError as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {', '.join(p.name for p in setup.paths)} to {setup.out_dir}")
    print(f"  {setup.n_surface} surface atoms, {len(setup.molecules)} adsorbate molecule(s), "
          f"net charge {setup.net_charge:+.3f} e")
    for n in setup.notes:
        print(f"  note: {n}")
    print(f"  Run: cd {setup.out_dir} && lmp -in in.smd   (GPU and MPI commands: README.txt)")
    if args.run:
        if not resources.has_package("lammps"):
            print("--run needs the LAMMPS Python module: pip install 'quantabench[md]'", file=sys.stderr)
            return 1
        print("Running LAMMPS (this blocks until the pull is done)...")
        run_lammps(setup.out_dir)
        return _md_summary(setup.out_dir)
    return 0


def _md_summary(folder: Path) -> int:
    from .md.lammps_log import drag_force_from_setup, estimate_detachment, parse_log, read_pull_force

    folder = Path(folder).expanduser()
    if not folder.is_dir():
        print(f"{folder}: no such folder", file=sys.stderr)
        return 2
    status = 0
    log_path = folder / "log.lammps"
    if log_path.is_file():
        log = parse_log(log_path)
        print(f"{log_path}: LAMMPS {log.version or '?'}, {len(log.blocks)} run block(s), "
              f"{len(log.warnings)} warning(s), {len(log.errors)} error(s)")
        for b in log.blocks:
            steps = b["Step"] if "Step" in b.columns else None
            span = f"steps {int(steps[0])}-{int(steps[-1])}" if steps is not None and len(steps) else f"{len(b)} rows"
            last = ", ".join(f"{c}={b.data[-1, i]:.4g}" for i, c in enumerate(b.columns) if c != "Step")
            print(f"  {span}" + (f", {b.loop_time:.1f} s" if b.loop_time else "") + (f": {last}" if len(b) else ""))
        for e in log.errors:
            print(f"  ERROR {e}")
        if log.lost_atoms:
            print("  Lost atoms: check for overlaps or a too-large time step.")
        status = 0 if log.ok else 1
    else:
        print(f"no log.lammps in {folder}")
    pull_path = folder / "pull_force.txt"
    if pull_path.is_file():
        try:
            print(estimate_detachment(read_pull_force(pull_path),
                                      drag_force_kcal_mol_A=drag_force_from_setup(folder)).summary())
        except ValueError as exc:
            print(f"{pull_path.name}: {exc}")
    else:
        print(f"no pull_force.txt in {folder}")
    return status


def _md_parse(args) -> int:
    return _md_summary(Path(args.dir))


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


def export_formats() -> list[str]:
    # kept in sync with chem.export.FORMATS by tests/test_cli.py; not imported here so that `doctor`
    # still starts when RDKit or ASE cannot load
    return ["xyz", "xyz-all", "extxyz", "sdf", "pdb", "mol", "lammps", "spectrum"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="quantabench-cli", description="QuantaBench command line tools")
    p.add_argument("--version", action="version", version=f"QuantaBench {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("doctor", help="report hardware, GPU backend, engines and settings")
    d.add_argument("--bench", action="store_true", help="also time matmul and each engine on CPU and GPU")
    d.add_argument("--json", action="store_true")
    d.set_defaults(func=_doctor)

    def inputs_args(q):
        q.add_argument("molecules", nargs="*", help="SMILES (optionally 'SMILES id') or names to look up in PubChem")
        q.add_argument("--file", action="append",
                       help="SDF, MOL, XYZ, PDB, SMI, CSV or Excel (.xlsx/.xlsm) file (repeatable)")
        q.add_argument("--smiles-col"), q.add_argument("--id-col")
        q.add_argument("--sheet", help="Excel sheet name or 0-based index (default: the first sheet)")
        q.add_argument("--smiles-pref", choices=["smiles", "qsar", "ms"],
                       help="which SMILES column of a CompTox export to prefer (falls back row by row)")

    def engine_args(q):
        q.add_argument("--method", choices=METHODS)
        q.add_argument("--fmax", type=float, help="force convergence, eV/Å")
        q.add_argument("--threads", type=int, help="CPU threads (0 = all but one)")
        q.add_argument("--gpu", choices=["auto", "on", "off"])
        q.add_argument("--out", help="output folder (default from settings)")

    def freq_args(q):
        q.add_argument("--freq-hessian", choices=["auto", "analytic", "finite"],
                       help="auto = analytic (dxtb) when it fits in memory, else finite differences")
        q.add_argument("--symmetry-number", type=int,
                       help="rotational symmetry number σ for every molecule (default 0: detected per molecule "
                            "from its point group)")
        q.add_argument("--temperature", type=float, help="K, for Boltzmann weights and thermochemistry")

    b = sub.add_parser("build", help="SMILES/name/file -> conformers -> optimized geometries + descriptors",
                       description="Batches (more than one molecule, or --run-dir/--resume/--chunk) run through "
                                   "the resumable job table in the run folder.")
    inputs_args(b)
    engine_args(b)
    b.add_argument("--ph", type=float)
    b.add_argument("--protonate", action=argparse.BooleanOptionalAction, default=None,
                   help="assign the charge state at --ph (--no-protonate keeps the input's; default from settings)")
    b.add_argument("--protonation", choices=["rules", "dimorphite"])
    b.add_argument("--max-confs", type=int)
    b.add_argument("--refine-top", type=int)
    b.add_argument("--window", type=float, help="energy window, kcal/mol")
    b.add_argument("--formats", help="comma list: " + ",".join(export_formats()))
    b.add_argument("--batched", choices=["auto", "on", "off"],
                   help="refine all conformers in one torch batch (auto: only on a GPU)")
    b.add_argument("--freq", action=argparse.BooleanOptionalAction, default=None,
                   help="frequencies, IR and thermochemistry of the best conformer (adds the 'spectrum' format); "
                        "default from settings")
    freq_args(b)
    b.add_argument("--run-dir", help="run folder for the job table (created if missing; continues if present)")
    b.add_argument("--resume", metavar="RUN_DIR", help="continue an interrupted run with its stored settings")
    b.add_argument("--chunk", metavar="K/N", help="build only chunk K of N (1-based) of the list, e.g. 2/3")
    b.add_argument("--retry-failed", action="store_true", help="also redo molecules that failed before")
    b.add_argument("--force", action="store_true", help="continue a run folder even if the settings differ")
    b.set_defaults(func=_build)

    st = sub.add_parser("status", help="progress of a (resumable) run folder")
    st.add_argument("run_dir")
    st.set_defaults(func=_status)
    mg = sub.add_parser("merge", help="combine summary/descriptors tables of several run folders (chunks)")
    mg.add_argument("out")
    mg.add_argument("run_dirs", nargs="+")
    mg.set_defaults(func=_merge)

    fq = sub.add_parser("freq", help="frequencies, thermochemistry (ZPE/H/S/G) and IR spectrum",
                        description="Builds each molecule (conformer search + refinement with --method), then "
                                    "analyzes its best conformer. With --as-is, structure files are analyzed "
                                    "as given (after a tight re-optimization unless --no-opt).")
    inputs_args(fq)
    engine_args(fq)
    freq_args(fq)
    fq.add_argument("--max-confs", type=int)
    fq.add_argument("--refine-top", type=int)
    fq.add_argument("--as-is", action="store_true", help="use the geometry in --file directly (no conformer search)")
    fq.add_argument("--no-opt", action="store_true", help="with --as-is: skip the re-optimization")
    fq.add_argument("--charge", type=int, default=0, help="with --as-is: total charge")
    fq.add_argument("--multiplicity", type=int, default=1, help="with --as-is: spin multiplicity")
    fq.add_argument("-v", "--verbose", action="store_true")
    fq.set_defaults(func=_freq)

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
    s.add_argument("--lammps-dir", help="also write a LAMMPS steered-MD setup (in.smd, system.data) here")
    s.add_argument("--lammps-gpu", action="store_true", help="with --lammps-dir: use the LAMMPS GPU package")
    s.set_defaults(func=_slab)

    md = sub.add_parser("md", help="LAMMPS steered MD: write setups, summarize results")
    msub = md.add_subparsers(dest="md_command", required=True)
    sm = msub.add_parser("smd", help="adsorbate pulled off a frozen slab (rigid adsorbate, UFF/graphite LJ)")
    sm.add_argument("--system", required=True, help="extxyz/traj with tags (0 = surface, 1 = adsorbate) and charges")
    sm.add_argument("-o", "--out", required=True, help="folder for system.data, in.smd, README.txt")
    sm.add_argument("--temperature", type=float, help="K (default 300)")
    sm.add_argument("--velocity", type=float, help="pull velocity, Å/fs (default 1e-4)")
    sm.add_argument("--spring", type=float, help="spring constant, kcal/mol/Å² (default 10)")
    sm.add_argument("--distance", type=float, help="pull distance, Å (default 15)")
    sm.add_argument("--graphite", choices=["amber", "steele"], default="amber", help="graphite LJ parameters")
    sm.add_argument("--gpu", action="store_true", help="LAMMPS GPU package (package gpu ... neigh no)")
    sm.add_argument("--run", action="store_true", help="run it now with the LAMMPS Python module (blocks)")
    sm.set_defaults(func=_md_smd)
    mp = msub.add_parser("parse", help="summarize log.lammps and pull_force.txt (detachment work, peak force)")
    mp.add_argument("dir")
    mp.set_defaults(func=_md_parse)

    se = sub.add_parser("settings", help="show settings, or move them (e.g. into a synced folder)")
    se.add_argument("--set-dir", nargs="?", const="", help="new settings folder ('' = default)")
    se.add_argument("--init", action="store_true", help="write settings.json with defaults if missing")
    se.set_defaults(func=_settings)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
