"""Command line interface.

    quantabench-cli doctor [--bench] [--json]
    quantabench-cli build "OC(=O)C(F)(F)..." PFOA --method aimnet2 --ph 7 [--freq]
    quantabench-cli build --file cohort.xlsx --smiles-pref qsar --chunk 1/3     # resumable batch run
    quantabench-cli build --resume RUN_DIR [--retry-failed]
    quantabench-cli status RUN_DIR | merge OUT_DIR RUN_DIR...
    quantabench-cli build "...PFBA" --conformers crest --method xtb-gfn2 --solvent water --qm wb97x-d/def2-svp --cubes
    quantabench-cli freq "OC(=O)C(F)(F)F TFA" --method gfn1-xtb                # frequencies + thermo + IR
    quantabench-cli import psi4_output.dat ORCA.out [--out DIR]                 # other programs' outputs (cclib)
    quantabench-cli import batches/*.spartan --table all.csv [--sdf all.sdf] [--wide qsar.csv]   # Spartan -> one table
    quantabench-cli sample qsar.csv -o val300.csv [-n 300 --seed 20261105]     # stratified validation set
    quantabench-cli validate RUN_GAS RUN_WATER --spartan qsar.csv --out DIR     # QuantaBench vs Spartan
    quantabench-cli conformers compare --a docs/*.spartan --b RUN --rescore gas,water --relax water --out DIR
                                                                                # whose conformers, search vs ranking
    quantabench-cli slab --nx 16 --ny 9 --layers 3 --molecule "OC(=O)C(F)(F)..." -o pfoa_on_graphite.data
                         [--lammps-dir DIR]
    quantabench-cli md smd --system pfoa_graphite.extxyz -o DIR [--gpu] [--run]
    quantabench-cli md parse DIR
    quantabench-cli settings [--set-dir ~/SyncThing/QuantaBench/Settings]
    quantabench-cli update [--check] [--upgrade-packages]                       # git pull + reinstall if needed

Splitting work (``quantabench.dist``):

    quantabench-cli build --file cohort.xlsx --method gfn1-xtb --lanes auto     # parallel processes here
    quantabench-cli lanes auto                                                  # what a lane spec means here
    quantabench-cli machines [--refresh] [--bench mmff,gfn1-xtb]                # machine registry + throughput
    quantabench-cli build --file cohort.xlsx --method mmff --chunk-weights auto # this machine's weighted share;
        # the first machine stores the split (settings/machines/splits), the others reuse it for that list
    quantabench-cli worker                                                      # serve jobs over Tailscale
    quantabench-cli grid build --file cohort.xlsx --method gfn2-xtb --machines btm-nova,local --wait
    quantabench-cli grid status|cancel|merge GRID_ID

Batch runs: ``build`` uses the resumable job table (``chem.jobs``: ``jobs.sqlite`` in the run folder)
whenever more than one molecule is built or ``--run-dir``/``--resume``/``--chunk`` is given, so an
interrupted batch continues where it stopped. A single molecule without those options runs directly.
"""

from __future__ import annotations

import argparse
import contextlib
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
        from .dist import profile

        out = rep.to_dict()
        if args.bench:  # progress goes to stderr, so stdout stays one JSON document
            prof, path = profile.refresh_from_cli(True, out=sys.stderr)
            if prof is None:
                return 130
            out["profile_saved"] = str(path)
        else:
            prof = profile.local_profile()
        out["profile"] = prof.to_dict()
        print(json.dumps(out, indent=2))
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
    for key, info in engines.all_engines().items():
        ok, why = engines.available(key)
        print(f"    {'OK ' if ok else '-- '} {info.label:30s} {why}")
    for name, err in engines.PLUGIN_ERRORS.items():
        print(f"    !!  plugin {name}: {err}")
    gpu_devs = [f"cuda:{g.index}" if g.backend in ("cuda", "rocm") else g.backend for g in rep.gpus]
    if gpu_devs:
        print("  GPU engines")
        for dev in gpu_devs:
            _doctor_gpu_engines(dev)
    from .chem import external

    print("  External programs")
    for name, info, hint in external.report():
        if info is None:
            print(f"    --  {name:30s} not found: {hint}")
        else:
            print(f"    OK  {name + ' ' + info.version:30s} {info.path} ({info.source})")
    py = external.find_python_with("psi4", "psi4_python")
    print(f"    {'OK ' if py else '-- '} {'psi4 Python':30s} {py or 'no interpreter that can import psi4'}")
    print("  Optional      " + ", ".join(f"{k}={'yes' if v else 'no'}" for k, v in rep.optional_packages.items()))
    print(f"  Settings      {settings_mod.settings_dir() / settings_mod.SETTINGS_FILE}")
    from .dist import profile

    print(f"  Machine       {profile.machine_name()} (registry {profile.registry_dir()}: `quantabench-cli machines`)")
    for note in rep.notes:
        print(f"  note: {note}")
    if args.bench:
        if any(m.startswith("rdkit") or m.startswith("ase") for m in broken):
            print("\nBenchmark skipped: fix the library errors above first.")
        else:
            _bench(rep)
    return 1 if broken else 0


def _doctor_gpu_engines(dev: str) -> None:
    """Where AIMNet2 and dxtb run when given ``dev`` (no model weights loaded, no dxtb run)."""
    from .chem import engines, torch_compat

    if engines.available("aimnet2")[0]:
        try:
            from .chem import aimnet_torch

            backend, why = aimnet_torch.choose_backend(dev)
            label = "stock aimnet (NVIDIA Warp)" if backend == "stock" else "Warp-free aimnet_torch"
            print(f"    {dev:8s} AIMNet2: {label}" + (f" ({why})" if why else ""))
        except Exception as exc:
            print(f"    {dev:8s} AIMNet2: cannot check ({type(exc).__name__}: {exc})")
    if engines.available("gfn1-xtb")[0]:
        try:
            run_on, why = torch_compat.dxtb_device(dev)
            print(f"    {dev:8s} dxtb: " + (torch_compat.describe(dev) if run_on == dev
                                             else why.replace("dxtb ran on", "runs on", 1)))
        except Exception as exc:
            print(f"    {dev:8s} dxtb: cannot check ({type(exc).__name__}: {exc})")


def _bench(rep) -> None:
    """matmul GFLOP/s per device, then molecules/hour per engine and device on the PFAS benchmark set
    (``dist.profile.benchmark``), saved to this machine's profile in the machine registry."""
    print("\nBenchmark")
    if resources.has_package("torch"):
        devices = ["cpu"] + [f"cuda:{g.index}" for g in rep.gpus if g.backend in ("cuda", "rocm")]
        for dev in devices:
            try:
                print(f"  matmul {dev:8s} {resources.matmul_benchmark(dev):9.1f} GFLOP/s")
            except Exception as exc:
                print(f"  matmul {dev:8s} FAILED: {exc}")
    else:
        print("  matmul skipped: PyTorch not installed (MMFF is still timed)")
    from .dist import profile

    profile.refresh_from_cli(True)


def method_choices() -> list[str]:
    from .chem import engines

    return engines.method_keys()


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
        "temperature_k": getattr(args, "temperature", None), "solvent": getattr(args, "solvent", None),
        "optimizer": getattr(args, "optimizer", None), "conformer_search": getattr(args, "conformers", None),
        "crest_method": getattr(args, "crest_method", None), "crest_fallback": getattr(args, "crest_fallback", None),
        "qm_method": getattr(args, "qm", None),
        "qm_task": getattr(args, "qm_task", None), "qm_top": getattr(args, "qm_top", None),
        "memory_budget_gb": getattr(args, "memory_budget", None),
        "molecule_timeout_s": getattr(args, "molecule_timeout", None),
    }.items() if v is not None}
    if getattr(args, "crest_full", False):
        overrides["crest_quick"] = False
    if getattr(args, "qm_freq", None) is not None:
        overrides["qm_frequencies"] = args.qm_freq
    if getattr(args, "cubes", None) is not None:
        overrides["qm_cubes"] = args.cubes
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


read_inputs = _read_inputs      # public names for quantabench.dist (grid build)
config_from_args = _config


def _progress(i, n, stage, ident):
    if stage == "start":
        print(f"[{i + 1}/{n}] {ident}")


def _build(args) -> int:
    from .chem import jobs, pipeline
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
    if args.chunk_weights:  # weighted split (dist.plan): chunk becomes (i, n, weights), stored with the run
        from .dist import plan, profile

        try:
            chunk, weights, warnings = plan.resolve_chunk(args.chunk_weights, chunk,
                                                         profile.bench_method(cfg.method, cfg.solvent),
                                                         profile.load_profiles(), ids=[x.id for x in items])
            print(plan.chunk_plan(weights, chunk, [x.id for x in items], warnings) + "\n")
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 2

    resources.set_molecule_limits(cfg)   # memory budget and time limit of the embedding and the Hessian

    resumable = bool(run_dir or chunk or args.retry_failed or len(items) > 1)
    if not resumable:
        results, out = run(items, cfg, progress=_progress, log=print)
    else:
        if run_dir is None:
            label = items[0].id if len(items) == 1 else f"batch_{len(items)}"
            if chunk:
                label += f"_{jobs.chunk_label(chunk)}_{socket.gethostname()}"
            run_dir = new_run_dir(cfg.output_dir or settings_mod.load().output_path(), label)
        from .dist import profile as profile_mod

        try:  # an explicit --lanes that does not fit is an error; this machine's setting falls back to one lane
            lanes, _spec = profile_mod.resolve_lanes(args.lanes, cfg, log=print)
        except ValueError as exc:
            print(f"--lanes {args.lanes}: {exc}", file=sys.stderr)
            return 2
        try:
            results, out = jobs.run_resumable(items or None, cfg, run_dir, progress=_progress, log=print,
                                              retry_failed=args.retry_failed, chunk=chunk, force=args.force,
                                              lanes=lanes, keep_results=False,
                                              speed_aware=args.speed_aware_lanes)
        except (jobs.ConfigMismatchError, jobs.RunInUseError) as exc:
            print(exc, file=sys.stderr)
            return 2
        print(jobs.open_run(out).summary())
    ok = sum(r.status == "ok" for r in results)
    fell_back = pipeline.fallback_count(results)
    print(f"\n{ok}/{len(results)} succeeded in this call"
          + (f" ({fell_back} with RDKit conformers because CREST failed: summary.csv conformer_fallback)"
             if fell_back else "") + "." + (f" Output: {out}" if out else ""))
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
    for ident, why in st.fallbacks:
        print(f"  RDKit conformers (asked for CREST): {ident}: {why}")
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


def _import_spartan(args, files: list[str]) -> int:
    """Spartan documents: one table (one row per molecule and file) and optionally one SDF."""
    from .chem import io_spartan
    from .chem.pipeline import new_run_dir

    ok, why = io_spartan.available()
    if not ok:
        print(f"olefile is not installed ({why})", file=sys.stderr)
        return 2
    if args.table:
        table = Path(args.table).expanduser()
    else:
        out = Path(args.out).expanduser() if args.out else new_run_dir(settings_mod.load().output_path(), "import")
        table = out / "spartan.csv"
    wide = Path(args.wide).expanduser() if args.wide else None
    s = io_spartan.write_table(files, table, sdf=Path(args.sdf).expanduser() if args.sdf else None,
                               conformers=args.conformers, log=print if len(files) <= 20 or args.verbose else None,
                               wide=wide)
    for e in s.errors[:50] if not args.verbose else s.errors:
        print(f"  {e}", file=sys.stderr)
    if len(s.errors) > 50 and not args.verbose:
        print(f"  ... {len(s.errors) - 50} more (-v lists all)", file=sys.stderr)
    print(f"{s.files} Spartan document(s), {s.molecules} molecules, {s.rows} rows -> {table}"
          + (f" (+ {args.sdf})" if args.sdf else ""))
    for reason, n in sorted(s.skipped.items(), key=lambda kv: -kv[1]):
        print(f"  {n} molecule(s) not read: {reason}", file=sys.stderr)
    if s.label_sources:
        print("  labels: " + ", ".join(f"{n} {src}" for src, n in sorted(s.label_sources.items(),
                                                                        key=lambda kv: -kv[1])))
    if wide:
        print(f"  wide table: {s.wide_rows} rows ({s.wide_rows - s.wide_ambiguous} labels, {s.wide_ambiguous} "
              f"ambiguous structures) -> {wide}")
        for p, level in s.prefixes.items():
            print(f"    {p}_*: {level or '(level unknown)'}")
    if s.duplicate_labels:
        print(f"  {s.duplicate_labels} label(s) occur in more than one molecule: all rows kept, duplicate_label = 1")
    if s.files_failed:
        print(f"  {s.files_failed} document(s) could not be read", file=sys.stderr)
    return 1 if s.files_failed or s.errors else 0


def _import(args) -> int:
    """Read other programs' output files with cclib: summary, then the usual exports per file. Spartan
    documents (``.spartan``) go to ``chem.io_spartan`` instead: one table for all of them."""
    from .chem import export, io_cclib, io_spartan
    from .chem.pipeline import new_run_dir

    spartan = [f for f in args.files if io_spartan.is_spartan(f)]
    others = [f for f in args.files if not io_spartan.is_spartan(f)]
    status = _import_spartan(args, spartan) if spartan else 0
    if not others:
        return status
    if args.table or args.sdf or args.wide:
        print("note: --table, --sdf and --wide apply to Spartan documents; the other files get the usual "
              "per-file exports", file=sys.stderr)
    ok, why = io_cclib.available()
    if not ok:
        print(f"cclib is not installed ({why}); pip install quantabench[interop]", file=sys.stderr)
        return 2
    out = Path(args.out).expanduser() if args.out else new_run_dir(settings_mod.load().output_path(), "import")
    formats = tuple(args.formats) if args.formats else (*export.DEFAULT_FORMATS, "spectrum")
    results = []
    for f in others:
        try:
            calc = io_cclib.read_output(f)
            res = io_cclib.to_result(calc, f)
        except Exception as exc:
            print(f"{Path(f).name}: FAILED {exc}", file=sys.stderr)
            status = 1
            continue
        print("\n".join(calc.summary()))
        if res.vibrations:
            _print_vibrations(res.input.id, res.vibrations)
        ids = {r.input.id for r in results}
        base, n = res.input.id, 2
        while res.input.id in ids:             # two files with the same name from different folders
            res.input.id = f"{base}_{n}"
            n += 1
        files = export.write_molecule(res, out, formats)
        res.files = [str(p) for p in files]
        results.append(res)
        print(f"  wrote {len(files)} files to {out} ({', '.join(p.name for p in files)})\n")
    if len(results) > 1:
        export.write_batch_tables(results, out)
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


def _update(args) -> int:
    """``update --check``: fetch and report. ``update``: git pull --ff-only, then pip install -e if the
    dependencies or the fork packages changed (quantabench.update). ``--upgrade-packages`` also upgrades the
    installed packages, with or without new commits. 0 = up to date or updated, 1 = refused or failed."""
    from . import update as update_mod

    upgrade = getattr(args, "upgrade_packages", False)
    status = update_mod.check()
    print(status.summary())
    if status.state in ("error", "not-a-checkout"):
        return 1
    if args.check:
        if status.can_update:
            print("Run `quantabench-cli update` (or Help > Check for updates... in the app) to update; add "
                  "--upgrade-packages to also upgrade the installed packages.")
        return 0
    nothing_to_pull = status.state in ("up-to-date", "ahead")
    if not status.can_update and not (upgrade and nothing_to_pull):
        return 0 if nothing_to_pull else 1
    print()
    extra = {"upgrade_packages": True, "upgrade_strategy": args.upgrade_strategy} if upgrade else {}
    res = update_mod.update(status=status, log=print, **extra)
    print(res.message)
    return 0 if res.ok else 1


def export_formats() -> list[str]:
    # kept in sync with chem.export.FORMATS by tests/test_cli.py; not imported here so that `doctor`
    # still starts when RDKit or ASE cannot load
    return ["xyz", "xyz-all", "extxyz", "sdf", "pdb", "mol", "lammps", "spectrum"]


def add_input_args(q: argparse.ArgumentParser) -> None:
    """Molecule inputs (positional SMILES/names, --file and its column options); read by ``read_inputs``."""
    q.add_argument("molecules", nargs="*", help="SMILES (optionally 'SMILES id') or names to look up in PubChem")
    q.add_argument("--file", action="append",
                   help="SDF, MOL, XYZ, PDB, SMI, CSV or Excel (.xlsx/.xlsm) file (repeatable)")
    q.add_argument("--smiles-col", help="tables: the SMILES column (default: smiles, qsar_ready_smiles, ...)")
    q.add_argument("--id-col", help="tables: the id column; SD files: the SD property to use as id, e.g. CAS_RN "
                                    "(default: DTXSID, ID, then the title line)")
    q.add_argument("--sheet", help="Excel sheet name or 0-based index (default: the first sheet)")
    q.add_argument("--smiles-pref", choices=["smiles", "qsar", "ms"],
                   help="which SMILES column of a CompTox export to prefer (falls back row by row)")


def add_engine_args(q: argparse.ArgumentParser, *, method_help: str | None = None, threads_help: str | None = None,
                    gpu_help: str | None = None, out_help: str | None = None) -> None:
    """Engine, solvent, optimizer and resources; the help texts can be adapted (``grid build``)."""
    q.add_argument("--method", choices=method_choices(), help=method_help)
    q.add_argument("--solvent", help="implicit solvent for engines that have one (water, octanol, ...; "
                                     "'none' = gas phase)")
    q.add_argument("--optimizer", choices=["lbfgs", "fire", "native"],
                   help="native = the engine's own optimizer (xtb, MOPAC) when it has one")
    q.add_argument("--fmax", type=float, help="force convergence, eV/Å")
    q.add_argument("--threads", type=int, help=threads_help or "CPU threads (0 = all but one)")
    q.add_argument("--gpu", choices=["auto", "on", "off"], help=gpu_help)
    q.add_argument("--out", help=out_help or "output folder (default from settings)")


def add_freq_args(q: argparse.ArgumentParser) -> None:
    q.add_argument("--freq-hessian", choices=["auto", "analytic", "finite"],
                   help="auto = analytic (dxtb) when it fits in memory, else finite differences")
    q.add_argument("--symmetry-number", type=int,
                   help="rotational symmetry number σ for every molecule (default 0: detected per molecule "
                        "from its point group)")
    q.add_argument("--temperature", type=float, help="K, for Boltzmann weights and thermochemistry")


def add_build_science_args(q: argparse.ArgumentParser) -> None:
    """The science options of ``build`` beyond the engine ones (``config_from_args`` reads them); shared with
    ``grid build`` so both accept the same settings."""
    q.add_argument("--ph", type=float)
    q.add_argument("--conformers", choices=["rdkit", "crest"],
                   help="conformer search: RDKit ETKDGv3 + MMFF94s (default) or CREST iMTD-GC (external)")
    q.add_argument("--crest-method", choices=["gfn2", "gfnff", "gfn2//gfnff", "gfn1"])
    q.add_argument("--crest-full", action="store_true", help="full CREST run instead of --quick (much slower)")
    q.add_argument("--crest-fallback", choices=["rdkit", "fail"],
                   help="when CREST is missing or fails for a molecule: 'rdkit' (default) builds it with RDKit's "
                        "conformers and flags it (note, summary.csv conformer_fallback, a count at the end); 'fail' "
                        "marks the molecule failed")
    q.add_argument("--qm", metavar="FUNCTIONAL/BASIS",
                   help="QM stage with Psi4 on the lowest conformers, e.g. wb97x-d/def2-svp")
    q.add_argument("--qm-task", choices=["energy", "optimize"], help="QM single point (default) or optimization")
    q.add_argument("--qm-top", type=int, help="number of lowest conformers for the QM stage (default 1)")
    q.add_argument("--qm-freq", action=argparse.BooleanOptionalAction, default=None,
                   help="QM frequencies + IR + thermochemistry of the best QM conformer")
    q.add_argument("--cubes", action=argparse.BooleanOptionalAction, default=None,
                   help="QM density + ESP cubes and molecular-surface (GIPF) descriptors")
    q.add_argument("--protonate", action=argparse.BooleanOptionalAction, default=None,
                   help="assign the charge state at --ph (--no-protonate keeps the input's; default from settings)")
    q.add_argument("--protonation", choices=["rules", "dimorphite"])
    q.add_argument("--max-confs", type=int)
    q.add_argument("--refine-top", type=int)
    q.add_argument("--window", type=float, help="energy window, kcal/mol")
    q.add_argument("--formats", help="comma list: " + ",".join(export_formats()))
    q.add_argument("--batched", choices=["auto", "on", "off"],
                   help="refine all conformers in one torch batch (auto: only on a GPU)")
    q.add_argument("--freq", action=argparse.BooleanOptionalAction, default=None,
                   help="frequencies, IR and thermochemistry of the best conformer (adds the 'spectrum' format); "
                        "default from settings")
    add_freq_args(q)


# ----------------------------------------------------------------------------------------- validation (VAL-1)
def _csv_list(text: str) -> list[str]:
    return [t.strip() for t in (text or "").split(",") if t.strip()]


def _sample(args) -> int:
    from .chem import validate as val

    try:
        table = val.read_spartan(args.spartan, log=print)
        cohorts = val.read_cohorts(args.cohorts) if args.cohorts else None
        bins = [int(x) for x in _csv_list(args.size_bins)]
        elements = None if args.elements.lower() == "any" else _csv_list(args.elements)
        s = val.sample(table, n=args.n, seed=args.seed, require=args.require, elements=elements,
                       charges=_csv_list(args.charges), min_heavy=args.min_heavy,
                       max_heavy=args.max_heavy if args.max_heavy > 0 else None, size_bins=bins,
                       strata=[{"charge": "charge_class"}.get(s, s) for s in _csv_list(args.stratify)],
                       prefer=args.prefer, cohorts=cohorts, allow_charge_mismatch=args.allow_charge_mismatch,
                       allow_fragments=args.allow_fragments, include_failed=args.include_failed)
    except (OSError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 2
    out = Path(args.output).expanduser()
    val.write_csv(out, val.sample_rows(s, table, bins, cohorts))
    design = out.with_suffix(".design.json")
    design.write_text(json.dumps(s.design, indent=2) + "\n", encoding="utf-8")
    d = s.design
    print(f"{d['n_selected']} of {d['eligible']} eligible molecules -> {out} (design: {design.name})")
    print(f"  with PM6 (gas): {d['with_pm6']}, with DFT (water): {d['with_dft_water']}, both: {d['with_both']}")
    for k, q in d["quotas"].items():
        print(f"  {k:<28} {q['quota']:>4} of {q['available']}")
    for why, n in sorted(s.excluded.items(), key=lambda kv: -kv[1]):
        print(f"  excluded: {n} {why}")
    return 0 if s.rows else 1


def _validate(args) -> int:
    from .chem import validate as val

    try:
        table = val.read_spartan(args.spartan, log=print)
        records = val.read_quantabench(args.runs)
    except (OSError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 2
    if not records:
        print("No finished QuantaBench molecules in " + ", ".join(args.runs), file=sys.stderr)
        return 2
    res = val.validate(records, table, args.out, top=args.top, include_failed=args.include_failed,
                       plots=not args.no_plots, log=print)
    joined = sum(1 for m in res.matches if m.rows)
    print(f"{joined} of {len(res.matches)} QuantaBench molecules joined to Spartan rows; {len(res.pairs)} property "
          f"pairs -> {res.out_dir}")
    for s in res.summary:
        if s["group_type"] == "all":
            print(f"  {s['pair']:<60} N={s['n']:<4} MAE={val._g(s['mae'])} r={val._g(s['pearson_r'])} "
                  f"({s['comparison']})")
    for w in sorted(set(res.warnings))[:5]:
        print(f"  warning: {w}", file=sys.stderr)
    return 0 if joined else 1


def register_validation(sub) -> None:
    """``sample`` (a stratified validation set from Spartan's wide table) and ``validate`` (a QuantaBench run
    against it); see chem/validate.py and docs/VALIDATION.md."""
    from .chem import validate as val

    sp = sub.add_parser("sample", help="pick a reproducible, stratified validation set from a Spartan wide table",
                        description="Reads a Spartan wide QSAR table (import --wide) and writes a CSV of SMILES, id "
                                    "and cohort that `build --file` reads: about N molecules stratified by cohort "
                                    "(the document name's prefix, C1A_, C3A_, ...) and charge class, molecules "
                                    "with DFT-water and PM6 values first, spread over size bins, with a seed.")
    sp.add_argument("spartan", help="the Spartan wide table (CSV)")
    sp.add_argument("-o", "--output", required=True, help="the sample CSV to write (+ <name>.design.json)")
    sp.add_argument("-n", type=int, default=300, help="molecules to pick (default 300)")
    sp.add_argument("--seed", type=int, default=val.SEED, help=f"random seed (default {val.SEED})")
    sp.add_argument("--require", choices=val.REQUIRE, default="any",
                    help="Spartan values a molecule must have: any (PM6 or DFT-water, default), pm6, dft, both, none")
    sp.add_argument("--prefer", choices=val.PREFER, default="dft",
                    help="within a stratum, take molecules with DFT-water values first (default), PM6 first, or none")
    sp.add_argument("--elements", default=",".join(val.AIMNET2_ELEMENTS),
                    help="allowed elements, comma-separated, or 'any' (default: AIMNet2's)")
    sp.add_argument("--charges", default=",".join(val.CHARGE_CLASSES), help="charge classes (neutral,anion,cation)")
    sp.add_argument("--min-heavy", type=int, default=1, help="fewest heavy atoms (default 1)")
    sp.add_argument("--max-heavy", type=int, default=val.DEFAULT_MAX_HEAVY,
                    help=f"most heavy atoms, 0 = no limit (default {val.DEFAULT_MAX_HEAVY})")
    sp.add_argument("--size-bins", default=",".join(map(str, val.DEFAULT_SIZE_BINS)),
                    help="heavy-atom bin edges (default 10,20,35: <=10, 11-20, 21-35, >35)")
    sp.add_argument("--stratify", default="cohort,charge",
                    help="strata: any of cohort, charge, size (default cohort,charge)")
    sp.add_argument("--cohorts", help="CSV of label -> cohort overriding the cohort from the document names")
    sp.add_argument("--allow-charge-mismatch", action="store_true",
                    help="keep rows whose Spartan charge differs from the structure Spartan drew")
    sp.add_argument("--allow-fragments", action="store_true", help="keep multi-fragment structures (salts)")
    sp.add_argument("--include-failed", action="store_true", help="count values of failed Spartan jobs")
    sp.set_defaults(func=_sample)

    vp = sub.add_parser("validate", help="compare a QuantaBench run with a Spartan wide table",
                        description="Joins QuantaBench molecules (run folders, merged tables or the paired gas/water "
                                    "table) to the Spartan wide table by InChIKey, canonical SMILES, then the "
                                    "stereo-free InChIKey, cross-checks the CAS id, and compares dipoles, HOMO/LUMO/"
                                    "gap, polarizabilities, areas/volumes and charges like with like (gas vs PM6, "
                                    "water vs DFT in water). Writes validation.csv, join.csv, summary.csv, report.md "
                                    "and parity plots (matplotlib) into --out.")
    vp.add_argument("runs", nargs="+", metavar="RUN", help="run folder(s) or table(s) (gas and water runs may be "
                                                           "given together)")
    vp.add_argument("--spartan", required=True, help="the Spartan wide table (CSV)")
    vp.add_argument("--out", required=True, help="output folder")
    vp.add_argument("--top", type=int, default=10, help="outliers listed per pair (default 10)")
    vp.add_argument("--include-failed", action="store_true", help="use values of failed Spartan jobs")
    vp.add_argument("--no-plots", action="store_true", help="skip the parity plots")
    vp.set_defaults(func=_validate)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="quantabench-cli", description="QuantaBench command line tools")
    p.add_argument("--version", action="version", version=f"QuantaBench {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("doctor", help="report hardware, GPU backend, engines and settings")
    d.add_argument("--bench", action="store_true", help="also time matmul and each engine on CPU and GPU")
    d.add_argument("--json", action="store_true")
    d.set_defaults(func=_doctor)

    b = sub.add_parser("build", help="SMILES/name/file -> conformers -> optimized geometries + descriptors",
                       description="Batches (more than one molecule, or --run-dir/--resume/--chunk) run through "
                                   "the resumable job table in the run folder.")
    add_input_args(b)
    add_engine_args(b)
    add_build_science_args(b)
    b.add_argument("--lanes", help="parallel worker processes on this machine: 1 (default), auto, N, or a list "
                                   "like gpu:1x4,cpu:5x2 (lanes x threads each); see docs/PLATFORMS.md")
    b.add_argument("--speed-aware-lanes", action=argparse.BooleanOptionalAction, default=None,
                   help="near the end of a batch, slower lanes leave the last molecules to clearly faster ones "
                        "(a GPU lane) when those finish them first; default from settings (on)")
    b.add_argument("--chunk-weights", metavar="SPEC",
                   help="weighted split: 3,1 (with --chunk K/N), name=weight pairs in any order (chunks are "
                        "the names sorted; this machine's is found by name), or auto (the machine registry's "
                        "throughput, stored for the list on first use so every machine reuses the same split); "
                        "see `quantabench-cli machines`")
    b.add_argument("--run-dir", help="run folder for the job table (created if missing; continues if present)")
    b.add_argument("--resume", metavar="RUN_DIR", help="continue an interrupted run with its stored settings")
    b.add_argument("--chunk", metavar="K/N", help="build only chunk K of N (1-based) of the list, e.g. 2/3")
    b.add_argument("--retry-failed", action="store_true", help="also redo molecules that failed before")
    b.add_argument("--force", action="store_true", help="continue a run folder even if the settings differ")
    b.add_argument("--memory-budget", type=float, metavar="GB",
                   help="memory per process for the dxtb steps: larger estimates take a cheaper path or skip the "
                        "step with a note (0 = available memory / lanes, at least 1.5 GB; < 0 = no guard; "
                        "default from settings); see docs/ENGINES.md")
    b.add_argument("--molecule-timeout", type=float, metavar="S",
                   help="wall-clock limit per molecule, s (0 = none; default from settings, 1800): a lane past it is "
                        "killed and restarted, one process stops at the next checkpoint; the molecule is marked "
                        "failed 'time limit (S s)' and --retry-failed redoes it")
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
    add_input_args(fq)
    add_engine_args(fq)
    add_freq_args(fq)
    fq.add_argument("--max-confs", type=int)
    fq.add_argument("--refine-top", type=int)
    fq.add_argument("--as-is", action="store_true", help="use the geometry in --file directly (no conformer search)")
    fq.add_argument("--no-opt", action="store_true", help="with --as-is: skip the re-optimization")
    fq.add_argument("--charge", type=int, default=0, help="with --as-is: total charge")
    fq.add_argument("--multiplicity", type=int, default=1, help="with --as-is: spin multiplicity")
    fq.add_argument("-v", "--verbose", action="store_true")
    fq.set_defaults(func=_freq)

    im = sub.add_parser("import", help="read Gaussian/ORCA/Psi4/xtb/MOPAC/... output files (cclib) and Spartan "
                                       "documents, and export them",
                        description="Parses each output file with cclib (last geometry, energies, charges, "
                                    "frequencies, IR intensities, normal modes), prints a summary and writes "
                                    "the usual files (XYZ, SDF, extXYZ, IR spectrum and mode table) per file. "
                                    "Spartan documents (.spartan) are read into ONE table instead: one row per "
                                    "molecule and document, keyed by the label (--table, --sdf).")
    im.add_argument("files", nargs="+", metavar="FILE", help="program output files and/or .spartan documents")
    im.add_argument("--out", help="output folder (default: a new run folder in the settings' output folder)")
    im.add_argument("--formats", nargs="+", choices=export_formats(),
                    help="export formats (default: xyz sdf extxyz spectrum)")
    im.add_argument("--table", help="Spartan documents: the CSV to write (default: OUT/spartan.csv)")
    im.add_argument("--sdf", help="Spartan documents: also write all geometries to this SDF")
    im.add_argument("--wide", help="Spartan documents: also write a QSAR table with one row per resolved label "
                                   "and each level of theory's best result as prefixed columns (pm6_homo_ev, "
                                   "dft_homo_ev, conf_n_conformers, ...)")
    im.add_argument("--conformers", choices=["best", "all"], default="best",
                    help="Spartan conformer libraries: the lowest-energy conformer per molecule (default) or one "
                         "row per saved conformer")
    im.add_argument("-v", "--verbose", action="store_true", help="one line per Spartan document")
    im.set_defaults(func=_import)

    register_validation(sub)        # sample, validate (chem.validate)

    s = sub.add_parser("slab", help="graphite slab, optionally with a molecule placed above it")
    s.add_argument("--nx", type=int, default=16), s.add_argument("--ny", type=int, default=9)
    s.add_argument("--layers", type=int, default=3)
    s.add_argument("--vacuum", type=float, default=30.0)
    s.add_argument("--fix-layers", type=int, default=1)
    s.add_argument("--molecule", help="SMILES, name, or structure file to place on the surface")
    s.add_argument("--method", choices=method_choices())
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

    up = sub.add_parser("update", help="update this QuantaBench checkout (git pull --ff-only, reinstall if "
                                       "dependencies changed; --upgrade-packages also upgrades the packages)")
    up.add_argument("--check", action="store_true", help="only fetch and report how far behind origin/main it is")
    up.add_argument("--upgrade-packages", action="store_true",
                    help="also upgrade the installed packages with pip (also when there is nothing to pull). "
                         "PyTorch stays the installed build, and the fork packages of scripts/fork-packages.json "
                         "are kept and reinstalled from their pinned source")
    up.add_argument("--upgrade-strategy", choices=["eager", "only-if-needed"], default="eager",
                    help="with --upgrade-packages: eager (default) upgrades every dependency to the newest version "
                         "pyproject.toml allows; only-if-needed upgrades QuantaBench and the fork packages, and "
                         "dependencies only where they must move")
    up.set_defaults(func=_update)

    from .dist import cli as dist_cli

    dist_cli.register(sub)      # machines, worker, grid (quantabench.dist)
    from .cli_phases import register as register_phases

    register_phases(sub, b)     # build --phases, pair (quantabench.cli_phases)
    from .cli_pubchem import register as register_pubchem

    register_pubchem(sub)       # pubchem enrich, pubchem lookup (quantabench.cli_pubchem)
    from .cli_conformers import register as register_conformers

    register_conformers(sub)    # conformers compare (quantabench.chem.conformer_compare)

    se = sub.add_parser("settings", help="show settings, or move them (e.g. into a synced folder)")
    se.add_argument("--set-dir", nargs="?", const="", help="new settings folder ('' = default)")
    se.add_argument("--init", action="store_true", help="write settings.json with defaults if missing")
    se.set_defaults(func=_settings)
    return p


def _apply_gpu_policies() -> None:
    """Settings > Compute's dxtb linear-algebra and GPU modes for commands that do not build a BuildConfig
    (freq, doctor, ...); ``build`` passes them through BuildConfig as well. Bad settings never stop a command."""
    from . import settings as settings_mod
    from .chem import torch_compat

    try:
        torch_compat.apply_settings(settings_mod.load(), strict=False)
    except Exception:  # noqa: BLE001, S110 - unreadable or invalid settings: keep the defaults
        pass


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _apply_gpu_policies()
    return args.func(args)


def safe_console_streams(streams=None) -> None:
    """Never fail on a character the console cannot show. On Windows a redirected stdout uses the ANSI code
    page (cp1252), so ``build --help`` (σ, Å) or a molecule named β-... raised UnicodeEncodeError; such
    characters now print as ``?``. UTF-8 streams are left alone."""
    for s in streams if streams is not None else (sys.stdout, sys.stderr):
        enc = (getattr(s, "encoding", "") or "").lower().replace("-", "").replace("_", "")
        if s is None or enc in ("utf8", "utf8sig") or not hasattr(s, "reconfigure"):
            continue
        with contextlib.suppress(Exception):
            s.reconfigure(errors="replace")


def entry() -> None:
    """Console-script entry point (``quantabench-cli``): ``main`` plus a hard exit where the GPU runtime
    would hang at shutdown (``resources.hard_exit_if_needed``). Tests call ``main`` directly."""
    safe_console_streams()
    code = main()
    resources.hard_exit_if_needed(code)
    raise SystemExit(code)


if __name__ == "__main__":
    entry()
