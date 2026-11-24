"""MD benchmark of a machine (MD-BENCH-1, MD-BENCH-2): ns/day of a LAMMPS liquid on each accelerator, and run times.

The molecule benchmark (``dist.profile.benchmark``) gives molecules/hour per engine; this one gives what an MD job
needs to know before it starts: how many ns/day this machine's LAMMPS runs a molecular liquid, on the CPU (OpenMP)
and on the GPU package, with the long-range options of MD-LR (``md.long_range``). It is opt-in, because it takes a
few minutes::

    quantabench-cli machines --bench md [--md-size slab|small|pilot-slab|pilot] [--md-variants cpu-ewald,...]
                             [--md-threads N] [--md-lammps PATH] [--max-seconds 60]
    quantabench-cli machines --bench default,md        (the molecule engines as well)
    quantabench-cli worker --bench-methods default,md  (the worker's start-up benchmark)
    quantabench-cli md estimate --atoms 9780 --ns 13 [--dispersion ewald] [--geometry slab|bulk]

The systems
-----------
Liquid C6F14 (perfluorohexane), OPLS-AA with the Watkins-Jorgensen perfluoroalkane types (``md.forcefield``), at
QuantaBench's density estimate (``SYSTEMS``):

``slab``        (the default, MD-BENCH-2) 190 molecules (3,800 atoms) in a 40 x 40 Å slab about 39 Å thick, with half
                its thickness of vacuum on each side (box about 40 x 40 x 78 Å). It times what a surface-tension run
                does: the slab's styles (no tail correction for a cutoff), pppm/disp's dispersion mesh of the slab box
                with its vacuum, the pressure tensor every 10 steps and the density profile. The lateral edge is the
                pilot's (so the x and y meshes are too), and box / liquid = 2, as in every ``md liquid-slab`` size
                (40 or 50 Å lateral, 100 Å of liquid, 2 x 50 Å of vacuum), so its cost per atom is theirs.
``pilot-slab``  the pilot slab itself: 489 molecules (9,780 atoms), 40 x 40 x 100 Å of liquid and 50 Å of vacuum on
                each side. For checking the atom scaling below on a machine once (about twice as long).
``small``       a bulk cube: 150 molecules (3,000 atoms, 37 Å), the bulk NPT stage's styles (tail correction).
``pilot``       the pilot's 489 molecules as a bulk box.

Each is packed by ``md.liquid`` (Packmol, else a lattice), minimized, compressed to the estimated density in 1 ps of
NVT and minimized again; a slab is then opened along z as ``in.slab`` does it (``md.liquid.open_box_lines``) and
relaxed for 1 ps of NVT. Once per QuantaBench version and size, in the per-machine cache
(``<cache>/md_bench/<version>-b<MD_BENCH_VERSION>/<size>``). If that fails, its inputs and logs stay in
``<cache>/md_bench/failed-<size>-<time>`` and the reason names how LAMMPS ended (``exit_reason``).

The run
-------
Each variant is its own LAMMPS process (no ``clear``: ``in.run``, bulk, ``clear`` and slab in one process, hung on
the GPU package): read the cached box, Maxwell velocities at 298.15 K from a fixed seed, ``run 0`` (energies and
pressure of the start, the consistency check below), 100 steps of warm-up, then NVT (Nosé-Hoover, 1 fs) for at
most ``MAX_STEPS`` steps or ``max_seconds`` of wall time (``timer timeout``), whichever comes first. ns/day is
LAMMPS' own ``Performance:`` line of the timed run. Variants are ``<accelerator>-<dispersion>``:

``cpu``     ``-sf omp -pk omp N`` (OPENMP package; plain styles without it), N threads (default: the thread budget
            of a one-lane run, at most the physical cores: LAMMPS' OpenMP styles gain little from SMT). A LAMMPS whose
            OPENMP package was built without OpenMP (``lmp -h``: "OPENMP package API: Serial", the macOS PyPI wheel)
            runs on one thread, and the record says so
``gpu``     ``liquid.gpu_switches``: ``-sf hybrid gpu omp -pk gpu 1 [device_type generic] pair/only on -pk omp N`` (GPU
            package; where ``lmp -h`` lists it), cutoff variants only: LAMMPS has no GPU version of
            ``lj/long/coul/long`` or ``pppm/disp``
``kokkos``  ``-k on g 1 -sf kk`` (KOKKOS with a GPU backend) or ``-k on t N -sf kk`` (OpenMP backend); only
            when asked for

The dispersion variants are MD-LR's options with their defaults (``md.long_range.settings``): ``ewald``
(``lj/long/coul/long`` 12 Å + ``pppm/disp``: Coulomb 1e-6, G = 2.77 / rc and a dispersion mesh no coarser than
0.52 / G, from the box it runs in) and ``cutoff12``/``cutoff16``/``cutoff18`` (``lj/cut/coul/long`` + ``pppm``
1e-6, 1e-5 before OPENMM-1; the bulk with LAMMPS' tail correction, the slab without). Each record names its settings (``long_range``,
``long_range_settings``) and the k-space grids LAMMPS used (``kspace``); a record without ``long_range`` timed the
coarser settings of before MD-LR.

The CPU/GPU check
-----------------
On the workstation the GPU package's NPT expanded a C6F14 box to half its density while the CPU held it, so a fast
GPU number alone is not to be trusted. Every non-CPU variant is compared with the CPU (OpenMP) run of the same
dispersion at ``run 0``, on the same configuration and velocities: the pressure must agree within
``PRESSURE_TOL_ATM`` and the potential energy within ``ENERGY_TOL_ABS`` + ``ENERGY_TOL_REL`` x the size of its
terms (mixed precision explains far less than that). ``check`` is "ok", "mismatch" (the variant is *unverified*:
estimates leave it out unless asked) or "skipped" (no CPU reference). The run-0 numbers are stored, so a mismatch
shows which term differs.

The registry
------------
``MachineProfile.md_throughput`` (``dist.profile``)::

    {"bench_version": 2, "date": "...", "quantabench_version": "...", "lammps": "22 Jul 2025 - Update 4",
     "lammps_source": "PATH", "packages": ["GPU", "KSPACE", "MOLECULE", "OPENMP", ...],
     "accelerators": {"GPU": {"api": "OpenCL", "precision": "mixed"}, ...}, "gpu_present": true,
     "system": {"size": "slab", "geometry": "slab", "atoms": 3800, "box_A": [40, 40, 78.2], ...},
     "variants": {"cpu-ewald@slab": {"ns_per_day": 17.1, "atoms": 3800, "threads": 8, "geometry": "slab",
                                     "box_A": [...], "lammps_build": "...", "long_range": "...",
                                     "kspace": {...}, "breakdown": {"Pair": 61.2, "Kspace": 25.0, ...},
                                     "run0": {"pe": ..., "press": ...}},
                  "gpu-cutoff16@slab": {"ns_per_day": ..., "check": "ok|mismatch|skipped", ...},
                  "cpu-ewald": {...a bulk box (small, pilot) or a record from before MD-BENCH-2...}},
     "notes": [...], "skipped": "" | why nothing was measured}

Keys are ``<variant>`` for the bulk boxes and ``<variant>@slab`` for the slab boxes (``record_key``), so a machine
keeps both. A profile without ``md_throughput`` (an older QuantaBench) loads with ``{}``. Variants measured earlier
and not measured again are kept (``dist.profile.local_profile``); each carries its own atoms, threads, LAMMPS and
date.

Scaling to another system (``scaled_rate``)
-------------------------------------------
ns/day is scaled by the atom ratio: rate(N) = rate_measured x N_measured / N (the cost of PPPM MD grows about
linearly with the atoms at a fixed density). When the job's box is known and the record has its box and LAMMPS'
timing breakdown, the k-space share f (``breakdown["Kspace"]``, %) scales with the box volume instead (its meshes
follow the box): time per step ∝ (1 - f) x N / N_measured + f x V / V_measured. For the ``md liquid-slab`` sizes
(box / liquid = 2, like the ``slab`` benchmark) both rules agree; a thinner slab in the same 50 Å of vacuum has more
mesh per atom, and the second rule sees that. ``estimate_hours`` prefers records of the job's geometry (``slab`` for
``md liquid-slab``) and falls back to the other, with a note: a bulk record leaves out the slab's vacuum mesh.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
import shutil
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from . import long_range as LR

MD_BENCH_VERSION = 2       # 2: slab boxes (MD-BENCH-2); records name their geometry, k-space grids and timings
SMILES = "FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"     # perfluorohexane, C6F14
MOLECULE = "C6F14"
FORCEFIELD = "oplsaa"
SEED = 20261120
#: ``--md-size``: the benchmark boxes. ``config``: ``LiquidSlabConfig`` settings of the packed liquid; ``vacuum``: Å of
#: vacuum on each side of a slab, or "half" (half the liquid's thickness: box / liquid = 2, as in every md liquid-slab
#: size); without it the box is bulk
SYSTEMS: dict[str, dict] = {
    "slab": {"config": {"lateral": 40.0, "n_molecules": 190}, "vacuum": "half"},   # 3,800 atoms, ~40 x 40 x 78 Å
    "small": {"config": {"lateral": 37.0, "n_molecules": 150}},                    # 3,000 atoms, bulk cube
    "pilot-slab": {"config": {"size": "pilot"}, "vacuum": 50.0},                   # MD-2's pilot slab, 9,780 atoms
    "pilot": {"config": {"size": "pilot"}},                                        # its atoms as a bulk box
}
DEFAULT_SIZE = "slab"
GEOMETRIES = ("slab", "bulk")
SLAB_RELAX_STEPS = 1000     # 1 ps of NVT after the slab's box is opened (the cut surfaces relax)
TEMPERATURE = 298.15        # K
TIMESTEP = 1.0              # fs
TDAMP = 100.0               # fs
ACCELERATORS = ("cpu", "gpu", "kokkos")
#: OPENMM-1: the same box timed by OpenMM (``md.openmm_engine.bench``) on its GPU platform or its CPU one
OPENMM_ACCELERATORS = ("openmm-gpu", "openmm-cpu")
ALL_ACCELERATORS = ACCELERATORS + OPENMM_ACCELERATORS
DEFAULT_VARIANTS = ("cpu-ewald", "cpu-cutoff12", "cpu-cutoff16", "gpu-cutoff12", "gpu-cutoff16")
#: ``--md-variants openmm`` (OPENMM-1): the OpenMM variant worth timing where OpenMM has a GPU; never a default
OPENMM_VARIANTS = ("openmm-gpu-cutoff16",)
MAX_STEPS = 10_000          # the timed run stops here or at max_seconds, whichever comes first
WARMUP_STEPS = 100
DEFAULT_MAX_SECONDS = 60.0
THERMO_EVERY = 1000
#: the CPU/GPU check at run 0 (same configuration and velocities)
PRESSURE_TOL_ATM = 20.0
ENERGY_TOL_ABS = 0.5        # kcal/mol
ENERGY_TOL_REL = 1.0e-4     # of |evdwl| + |ecoul| + |elong| + |bonded|
RUN0_KEYS = ("pe", "press", "pxx", "pyy", "pzz", "evdwl", "ecoul", "elong", "ebond", "eangle", "edihed", "ke")
#: the version in ``lmp -h`` (the same regex as ``chem.external.PROGRAMS["lammps"]``)
VERSION_RE = (r"Large-scale Atomic/Molecular Massively Parallel Simulator - (\d{1,2} \w{3} \d{4}"
              r"(?: - Update \d+)?)")
REQUIRED_PACKAGES = ("KSPACE", "MOLECULE")
GPU_KOKKOS_BACKENDS = ("CUDA", "HIP", "SYCL", "OPENMPTARGET")


def _noop(_msg: str) -> None:
    return None


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# ------------------------------------------------------------------------------------------- variants
@dataclass(frozen=True)
class Variant:
    accelerator: str        # "cpu" | "gpu" | "kokkos"
    dispersion: str         # "ewald" | "cutoff12" | ...

    @property
    def name(self) -> str:
        return f"{self.accelerator}-{self.dispersion}"


def dispersion_cutoff(dispersion: str) -> tuple[str, float]:
    """("ewald", 12.0) | ("cutoff", 16.0); the same rules as ``md.liquid.parse_dispersion`` (which needs numpy and
    RDKit to import), checked against it in the tests."""
    s = dispersion.strip().lower()
    if s == "ewald":
        return "ewald", 12.0
    m = re.fullmatch(r"cutoff(\d+(?:\.\d+)?)", s)
    if m and 8.0 <= float(m.group(1)) <= 30.0:
        return "cutoff", float(m.group(1))
    raise ValueError(f"dispersion {dispersion!r}: ewald, or cutoff12 / cutoff16 / cutoff18 (8-30 Å)")


def parse_variant(name: str) -> Variant:
    s = name.strip().lower()
    if s.startswith("openmm-"):                       # openmm-gpu-cutoff16, openmm-cpu-ewald; openmm-gpu = cutoff16
        kind_, _, disp = s[len("openmm-"):].partition("-")
        accel = f"openmm-{kind_}"
        if accel not in OPENMM_ACCELERATORS:
            raise ValueError(f"MD variant {name!r}: openmm-gpu-<dispersion> or openmm-cpu-<dispersion>")
        kind, rc = dispersion_cutoff(disp or "cutoff16")
        return Variant(accel, "ewald" if kind == "ewald" else f"cutoff{rc:g}")
    accel, sep, disp = s.partition("-")
    if not sep or accel not in ACCELERATORS:
        raise ValueError(f"MD variant {name!r}: <{'|'.join(ALL_ACCELERATORS)}>-<ewald|cutoff12|cutoff16|...>")
    kind, rc = dispersion_cutoff(disp)
    return Variant(accel, "ewald" if kind == "ewald" else f"cutoff{rc:g}")


def parse_variants(text: str | Sequence[str] | None) -> list[str] | None:
    """``--md-variants``: a comma list (or a sequence) of variant names, "default" for ``DEFAULT_VARIANTS``, "openmm"
    for ``OPENMM_VARIANTS``;
    None or empty = the default (only the variants this LAMMPS can run). Raises ValueError on a bad name."""
    if text is None:
        return None
    items = [x.strip() for x in (text.split(",") if isinstance(text, str) else text) if x and x.strip()]
    if not items:
        return None
    out: list[str] = []
    for item in items:
        group = {"default": DEFAULT_VARIANTS, "openmm": OPENMM_VARIANTS}.get(item.lower(), (item,))
        for name in group:
            v = parse_variant(name).name
            if v not in out:
                out.append(v)
    return out


def geometry_of(size: str) -> str:
    """"slab" for a benchmark box with vacuum (``SYSTEMS[size]["vacuum"]``), else "bulk"."""
    if size not in SYSTEMS:
        raise ValueError(f"MD benchmark size {size!r}: choose {', '.join(SYSTEMS)}")
    return "slab" if SYSTEMS[size].get("vacuum") else "bulk"


def record_key(variant: str, geometry: str = "bulk") -> str:
    """The registry key of a variant measured in a box of ``geometry``: "cpu-ewald" (bulk) or "cpu-ewald@slab"."""
    return variant if geometry == "bulk" else f"{variant}@{geometry}"


def split_key(key: str) -> tuple[str, str]:
    """("cpu-ewald", "slab") from "cpu-ewald@slab"; ("cpu-ewald", "bulk") from "cpu-ewald" (and every record from
    before MD-BENCH-2). The variant part is not validated here (``parse_variant``)."""
    name, sep, geo = str(key).partition("@")
    return (name, geo) if sep and geo in GEOMETRIES else (str(key), "bulk")


def _order(name: str) -> tuple:
    variant, geo = split_key(name)
    try:
        v = parse_variant(variant)
    except ValueError:
        return (9, 0.0, 9, name)
    kind, rc = dispersion_cutoff(v.dispersion)
    return (ALL_ACCELERATORS.index(v.accelerator), 0.0 if kind == "ewald" else rc, GEOMETRIES.index(geo), name)


# ------------------------------------------------------------------------------------------- LAMMPS
@dataclass
class LammpsInfo:
    path: str
    source: str = ""                     # "PATH", "settings", "conda env ...", ...
    version: str = ""                    # "22 Jul 2025 - Update 4"
    packages: list[str] = field(default_factory=list)
    accelerators: dict[str, dict] = field(default_factory=dict)   # {"GPU": {"api": "OpenCL", "precision": "mixed"}}
    gpu_present: bool | None = None      # "Compatible GPU present: yes|no" (GPU package builds); None = not said

    def has(self, package: str) -> bool:
        return package.upper() in self.packages

    @property
    def kokkos_on_gpu(self) -> bool:
        api = str(self.accelerators.get("KOKKOS", {}).get("api", "")).upper()
        return any(b in api.split() for b in GPU_KOKKOS_BACKENDS)

    @property
    def omp_serial(self) -> bool:
        """The OPENMP package is there but was built without OpenMP ("OPENMP package API: Serial", e.g. the macOS
        PyPI wheel): its /omp styles run on one thread whatever ``-pk omp N`` says."""
        return self.has("OPENMP") and str(self.accelerators.get("OPENMP", {}).get("api", "")).strip().lower() == "serial"

    def build_text(self) -> str:
        """'22 Jul 2025 - Update 4; OPENMP: OpenMP, double; GPU: OpenCL, mixed; from conda env qb-lammps'."""
        parts = [self.version or "version unknown"]
        for name in sorted(self.accelerators):
            a = self.accelerators[name]
            parts.append(f"{name}: " + ", ".join(str(a[k]) for k in ("api", "precision") if a.get(k)))
        if self.source:
            parts.append(f"from {self.source}")
        return "; ".join(parts)


def parse_lmp_help(text: str) -> dict:
    """Version, installed packages, accelerator APIs and whether a GPU was found, from ``lmp -h``."""
    m = re.search(VERSION_RE, text)
    lines = text.splitlines()
    packages: list[str] = []
    for i, line in enumerate(lines):
        if line.strip() != "Installed packages:":
            continue
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        while j < len(lines) and lines[j].strip():
            packages += [t for t in lines[j].split() if re.fullmatch(r"[A-Z0-9][A-Z0-9_-]*", t)]
            j += 1
        break
    accel: dict[str, dict] = {}
    for a in re.finditer(r"^\s*([A-Za-z0-9_-]+) package (API|precision): (.+?)\s*$", text, re.M):
        accel.setdefault(a.group(1).upper(), {})[a.group(2).lower()] = a.group(3).strip()
    g = re.search(r"Compatible GPU present:\s*(yes|no)", text, re.I)
    return {"version": m.group(1) if m else "", "packages": sorted(set(packages)), "accelerators": accel,
            "gpu_present": None if g is None else g.group(1).lower() == "yes"}


def find_lammps(refresh: bool = False, path: str | None = None) -> tuple[LammpsInfo | None, str]:
    """(LAMMPS, "") or (None, why not), described by ``lmp -h``: ``path`` when given (``--md-lammps``), else the
    executable ``chem.external`` finds (this machine's ``lammps`` path in the settings, ``QUANTABENCH_LAMMPS``, PATH,
    this environment, conda environments).

    An explicit choice that points at nothing (this machine's settings path or ``QUANTABENCH_LAMMPS``,
    ``external.configured_problems``) is refused, not passed over: ``find_program`` would go on to another LAMMPS
    (on the MacBook, 2026-11-21, a variable mangled on its way through PowerShell and SSH sent the benchmark to the
    venv's PyPI wheel), and a benchmark of the wrong program is worse than none."""
    from ..chem import external

    if path:
        p = Path(os.path.expanduser(str(path).strip()))
        if not p.is_file():
            return None, f"--md-lammps {path}: no such file"
        info = external.ProgramInfo("lammps", str(p), "given (--md-lammps)")
    else:
        try:
            info = external.find_program("lammps", version=False, refresh=refresh)
            problems = external.configured_problems("lammps")
        except Exception as exc:  # noqa: BLE001 - a broken settings path must not break a profile
            return None, f"could not look for LAMMPS ({type(exc).__name__}: {exc})"
        if problems and not (info and info.source.startswith(("settings (this machine)", "environment variable"))):
            other = f"; the next one found is {info.path} ({info.source.split(' (')[0]})" if info else ""
            return None, (f"{'; '.join(problems)}{other}. Not timing a LAMMPS you did not choose: fix or unset it, or "
                          "give the program with --md-lammps PATH")
        if info is None:
            return None, "LAMMPS not found (" + external.PROGRAMS["lammps"].install_hint + ")"
    if info.via_wsl:
        return None, f"LAMMPS {info.path} runs through WSL; the MD benchmark times native programs only"
    from .. import proctree

    probe = proctree.run_probe([info.path, "-h"], timeout=120)       # bounded whatever lmp's children do (ENG-3)
    if probe.error:
        return None, f"{info.path} -h failed ({probe.error})"
    h = parse_lmp_help(probe.text)
    if not h["version"] and not h["packages"]:
        why = "no answer within 120 s" if probe.timed_out else f"exit {probe.returncode}"
        return None, f"{info.path} -h did not print a LAMMPS version ({why})"
    return LammpsInfo(info.path, info.source, h["version"], h["packages"], h["accelerators"], h["gpu_present"]), ""


def openmm_platforms(refresh: bool = False) -> dict:
    """``md.openmm_engine.probe_platforms`` of this machine (probed once per process)."""
    from . import openmm_engine as OE

    return OE.probe_platforms(refresh=refresh)


def openmm_can_run(v: Variant) -> tuple[bool, str]:
    desc = openmm_platforms()
    if desc.get("error"):
        return False, desc["error"]
    if v.accelerator == "openmm-gpu" and not any(p["name"] in ("CUDA", "HIP", "OpenCL") and p.get("precisions")
                                                 for p in desc.get("platforms", [])):
        return False, f"OpenMM {desc.get('version')} sees no GPU platform here"
    return True, ""


def can_run(info: LammpsInfo, v: Variant) -> tuple[bool, str]:
    missing = [p for p in REQUIRED_PACKAGES if info.packages and not info.has(p)]
    if v.accelerator.startswith("openmm"):          # the box is LAMMPS-prepared; OpenMM times it
        return (False, f"this LAMMPS lacks {' and '.join(missing)} (it prepares the box)") if missing else \
            openmm_can_run(v)
    if missing:
        return False, f"this LAMMPS lacks the {' and '.join(missing)} package{'s' if len(missing) > 1 else ''}"
    if v.accelerator != "cpu" and v.dispersion == "ewald":
        return False, (f"{v.name}: LAMMPS has no {v.accelerator.upper()} version of lj/long/coul/long and "
                       "pppm/disp (ewald runs on the CPU)")
    if v.accelerator == "gpu":
        if not info.has("GPU"):
            hint = " (the Windows installer's lmp.exe has one)" if sys.platform == "win32" else ""
            return False, f"this LAMMPS has no GPU package{hint}"
        if info.gpu_present is False:
            return False, "LAMMPS' GPU package found no compatible GPU (lmp -h)"
    if v.accelerator == "kokkos" and not info.has("KOKKOS"):
        return False, "this LAMMPS has no KOKKOS package"
    return True, ""


def accelerator_args(info: LammpsInfo, accelerator: str, threads: int) -> tuple[list[str], int, str]:
    """(command-line switches, CPU threads, short label) for a variant's accelerator."""
    if accelerator == "cpu":
        if info.omp_serial:              # /omp styles, one thread: what this build can do
            return ["-sf", "omp", "-pk", "omp", "1"], 1, "omp-serial"
        if info.has("OPENMP") or not info.packages:
            return ["-sf", "omp", "-pk", "omp", str(threads)], threads, "omp"
        return [], 1, "none"
    if accelerator == "gpu":
        from .liquid import gpu_switches
        opencl = str(info.accelerators.get("GPU", {}).get("api", "")).lower() == "opencl"
        omp = (1 if info.omp_serial else threads) if info.has("OPENMP") else None
        return gpu_switches(opencl, omp), (omp or 1), "gpu"
    if info.kokkos_on_gpu:
        return ["-k", "on", "g", "1", "-sf", "kk"], 1, "kokkos-gpu"
    return ["-k", "on", "t", str(threads), "-sf", "kk"], threads, "kokkos-omp"


def default_threads() -> int:
    """The thread budget of a one-lane run on this machine (its threads minus the free ones, ``priority``), at most
    its physical cores."""
    from .. import priority, resources
    from .. import settings as settings_mod

    try:
        s = settings_mod.load()
        level, free = s.priority_for(), s.free_threads_for()
    except Exception:  # noqa: BLE001 - defaults without settings
        level, free = priority.DEFAULT_PRIORITY, priority.DEFAULT_FREE_THREADS
    return max(1, min(priority.thread_budget(0, free, level), resources.physical_cores()))


# ------------------------------------------------------------------------------------------- parsing output
_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|[-+]?inf|nan"
PERF_RE = re.compile(rf"^Performance:\s*({_NUM}) ns/day,\s*({_NUM}) hours/ns,\s*({_NUM}) timesteps/s"
                     rf"(?:,\s*({_NUM}) ([kMG]?)atom-step/s)?", re.M | re.I)
LOOP_RE = re.compile(rf"^Loop time of ({_NUM}) on (\d+) procs for (\d+) steps with (\d+) atoms", re.M)
CPU_RE = re.compile(rf"^\s*({_NUM})% CPU use with (\d+) MPI tasks x (\d+) OpenMP threads", re.M)
DEVICE_RE = re.compile(r"^\s*Device (\d+): (.+?)\s*$", re.M)


def parse_performance(text: str) -> dict | None:
    """The last ``Performance:`` line of a LAMMPS log (with the ``Loop time`` and ``% CPU use`` lines after
    it), or None. Keys: ns_per_day, hours_per_ns, timesteps_per_s, katom_steps_per_s, loop_s, procs, steps,
    atoms, mpi, omp_threads, cpu_percent (the last ones when present)."""
    perf = list(PERF_RE.finditer(text))
    if not perf:
        return None
    p = perf[-1]
    out: dict = {"ns_per_day": float(p.group(1)), "hours_per_ns": float(p.group(2)),
                 "timesteps_per_s": float(p.group(3))}
    if p.group(4):
        out["katom_steps_per_s"] = float(p.group(4)) * {"": 1e-3, "k": 1.0, "M": 1e3, "G": 1e6}[p.group(5)]
    loops = [m for m in LOOP_RE.finditer(text) if m.start() < p.start()]   # LAMMPS prints Loop time first
    if loops:
        m = loops[-1]
        out.update(loop_s=float(m.group(1)), procs=int(m.group(2)), steps=int(m.group(3)), atoms=int(m.group(4)))
    cpus = [m for m in CPU_RE.finditer(text) if m.start() > p.start()]
    if cpus:
        m = cpus[0]
        out.update(cpu_percent=float(m.group(1)), mpi=int(m.group(2)), omp_threads=int(m.group(3)))
    return out


def parse_tagged(text: str, tag: str) -> dict[str, float] | None:
    """``QB_RUN0 pe -1840.7 press 1249.5 ...`` (a ``print`` of the benchmark input) -> {"pe": -1840.7, ...}; the
    last such line, None if there is none. The echoed ``print "QB_RUN0 ..."`` command never matches."""
    found = None
    for line in text.splitlines():
        if not line.startswith(tag + " "):
            continue
        toks = line.split()[1:]
        vals: dict[str, float] = {}
        for k, v in zip(toks[::2], toks[1::2]):
            with contextlib.suppress(ValueError):
                vals[k] = float(v)
        found = vals
    return found


def lammps_errors(text: str) -> list[str]:
    return [ln.strip() for ln in text.splitlines() if ln.startswith("ERROR")]


_KSPACE_G_RE = re.compile(rf"^\s*(Coulomb |Dispersion )?G vector \(1/distance\)\s*=\s*({_NUM})", re.M)
_KSPACE_GRID_RE = re.compile(r"^\s*(Coulomb |Dispersion )?grid\s*=\s*(\d+)\s+(\d+)\s+(\d+)", re.M)


def parse_kspace(text: str) -> dict:
    """The k-space settings LAMMPS reports at its last set-up: {"coulomb_g": 0.25, "coulomb_grid": [20, 20, 45],
    "dispersion_g": 0.2308, "dispersion_grid": [18, 18, 36]} (the dispersion keys with pppm/disp only); {} when the
    log has none."""
    out: dict = {}
    for m in _KSPACE_G_RE.finditer(text):
        out[f"{(m.group(1) or 'Coulomb ').strip().lower()}_g"] = float(m.group(2))
    for m in _KSPACE_GRID_RE.finditer(text):
        out[f"{(m.group(1) or 'Coulomb ').strip().lower()}_grid"] = [int(m.group(k)) for k in (2, 3, 4)]
    return out


_BREAKDOWN_ROW_RE = re.compile(rf"^(\w+)\s*\|.*\|\s*({_NUM})\s*$")


def parse_breakdown(text: str) -> dict[str, float]:
    """LAMMPS' last "MPI task timing breakdown" as {section: % of the loop time}: {"Pair": 61.2, "Kspace": 25.0,
    "Neigh": 6.1, ...}; {} when there is none. The k-space share feeds ``scaled_rate``."""
    at = text.rfind("MPI task timing breakdown:")
    if at < 0:
        return {}
    out: dict[str, float] = {}
    for line in text[at:].splitlines()[3:]:
        m = _BREAKDOWN_ROW_RE.match(line.strip())
        if not m:
            break
        out[m.group(1)] = float(m.group(2))
    return out


#: the signal numbers every POSIX system shares (for ``exit_reason`` where ``signal`` lacks them, on Windows)
POSIX_SIGNALS = {1: "SIGHUP", 2: "SIGINT", 3: "SIGQUIT", 4: "SIGILL", 6: "SIGABRT", 8: "SIGFPE", 9: "SIGKILL",
                 11: "SIGSEGV", 13: "SIGPIPE", 14: "SIGALRM", 15: "SIGTERM"}


def exit_reason(code: int | None) -> str:
    """What an exit code says about how a program ended: "" for an ordinary exit status, else e.g. "killed by
    SIGTERM (signal 15) from outside QuantaBench ...". A signal shows as -N (Python), 128 + N (a shell between, the way
    the 2026-11-21 MacBook run reported 143) or 256 - N (PyPI's ``lmp`` wrapper, ``sys.exit(-N)``)."""
    import signal

    if code is None:
        return ""
    n, how = 0, ""
    if code < 0:
        n = -code
    elif 128 < code <= 128 + 64:
        n, how = code - 128, f" (exit code {code} = 128 + {code - 128}, as a shell reports a signal)"
    elif 256 - 64 <= code < 256:
        n, how = 256 - code, f" (exit code {code} = 256 - {256 - code}: PyPI's lmp wrapper passing on a signal)"
    if not n:
        return ""
    try:
        name = signal.Signals(n).name
    except ValueError:                   # Windows has no SIGKILL, SIGHUP, ...: the codes come from POSIX machines
        name = POSIX_SIGNALS.get(n, "")
    head = f"killed by {name} (signal {n}){how}" if name else f"killed by signal {n}{how}"
    if name in ("SIGTERM", "SIGINT", "SIGHUP"):
        return head + (": stopped from outside QuantaBench (QuantaBench itself stops LAMMPS only at its time limit "
                       "or on a cancel, and says so)")
    if name == "SIGKILL":
        return head + ": often the system running out of memory, or a forced stop"
    if name in ("SIGSEGV", "SIGBUS", "SIGABRT", "SIGILL", "SIGFPE"):
        return head + ": LAMMPS crashed"
    return head


# ------------------------------------------------------------------------------------------- the check
def check_consistency(ref: Mapping[str, float] | None, got: Mapping[str, float] | None, *,
                      pressure_tol: float = PRESSURE_TOL_ATM, energy_tol_abs: float = ENERGY_TOL_ABS,
                      energy_tol_rel: float = ENERGY_TOL_REL) -> tuple[str, str]:
    """("ok" | "mismatch" | "skipped", detail): a variant's run-0 pressure and potential energy against the CPU's on
    the same configuration and velocities. The energy tolerance grows with the size of its terms (mixed
    precision on a GPU is relative); a mismatch names the terms that differ."""
    if not ref or "press" not in ref or "pe" not in ref:
        return "skipped", "no CPU reference at run 0"
    if not got or "press" not in got or "pe" not in got:
        return "skipped", "no run-0 values from this variant"
    terms = ("evdwl", "ecoul", "elong")
    nonbonded = sum(ref.get(k, 0.0) for k in terms)
    scale = sum(abs(ref.get(k, 0.0)) for k in terms) + abs(ref["pe"] - nonbonded)
    e_tol = energy_tol_abs + energy_tol_rel * scale
    dp, de = got["press"] - ref["press"], got["pe"] - ref["pe"]
    detail = (f"pressure {got['press']:,.1f} vs CPU {ref['press']:,.1f} atm (diff {dp:+,.1f}, tolerance "
              f"{pressure_tol:g}); pe {got['pe']:,.3f} vs {ref['pe']:,.3f} kcal/mol (diff {de:+,.3f}, tolerance "
              f"{e_tol:.2g})")
    finite = all(math.isfinite(x) for x in (dp, de))
    if finite and abs(dp) <= pressure_tol and abs(de) <= e_tol:
        return "ok", detail
    off = [k for k in (*terms, "ebond", "eangle", "edihed", "pxx", "pyy", "pzz")
           if k in ref and k in got and abs(got[k] - ref[k]) > (pressure_tol if k.startswith("p") else e_tol)]
    return "mismatch", detail + (f"; differing: {', '.join(off)}" if off else "")


def check_energy(ref: Mapping[str, float] | None, got: Mapping[str, float] | None) -> tuple[str, str]:
    """An OpenMM variant's run-0 potential energy against LAMMPS' CPU run on the same configuration (OpenMM gives no
    pressure tensor on a single-precision GPU worth checking; σ is computed afterwards in double precision)."""
    if not ref or "pe" not in ref:
        return "skipped", "no CPU reference at run 0"
    if not got or "pe" not in got:
        return "skipped", "no run-0 energy from this variant"
    terms = ("evdwl", "ecoul", "elong")
    nonbonded = sum(ref.get(k, 0.0) for k in terms)
    scale = sum(abs(ref.get(k, 0.0)) for k in terms) + abs(ref["pe"] - nonbonded)
    tol = ENERGY_TOL_ABS + ENERGY_TOL_REL * scale
    de = got["pe"] - ref["pe"]
    detail = f"pe {got['pe']:,.3f} vs LAMMPS CPU {ref['pe']:,.3f} kcal/mol (diff {de:+,.3f}, tolerance {tol:.2g})"
    return ("ok" if math.isfinite(de) and abs(de) <= tol else "mismatch"), detail


def run_openmm_variant(work: Path, v: Variant, threads: int, *, steps: int = MAX_STEPS,
                       max_seconds: float = DEFAULT_MAX_SECONDS, geometry: str = "bulk",
                       box: Sequence[float] | None = None, should_stop: Callable[[], bool] | None = None) -> dict:
    """Time ``v`` (an ``openmm-gpu`` or ``openmm-cpu`` variant) on the benchmark box in ``work`` with OpenMM, in the
    OpenMM environment (``md.openmm_engine.bench``): the same model, Langevin NVT at 298.15 K, 1 fs."""
    from ..chem import external
    from . import openmm_engine as OE

    kind, rc = dispersion_cutoff(v.dispersion)
    rec: dict = {"accelerator": "openmm", "engine": "openmm", "dispersion": v.dispersion, "cutoff_A": rc,
                 "timestep_fs": TIMESTEP, "date": _now(), "bench_version": MD_BENCH_VERSION, "geometry": geometry,
                 "long_range": LR.describe(LR.settings(kind, rc), box if kind == "ewald" else None), "error": ""}
    if box:
        rec["box_A"] = [round(float(x), 4) for x in box]
    py, why = OE.find_openmm_python()
    if py is None:
        rec["error"] = why
        return rec
    n = 1 if v.accelerator == "openmm-gpu" else max(1, int(threads))
    cmd = [py, "-m", "quantabench.md.openmm_engine", "bench", "bench_start.data", "ff.params", "--molecule",
           "molecule.json", "--dispersion", v.dispersion, "--geometry", geometry,
           "--platform", "gpu" if v.accelerator == "openmm-gpu" else "CPU", "--threads", str(n),
           "--steps", str(int(steps)), "--max-seconds", f"{max_seconds:g}"]
    env = external.program_env(n, {k: val for k, val in OE.openmm_env(py, n).items()
                                   if k in ("PYTHONPATH", "OPENMM_CPU_THREADS")})
    try:
        run = external.run_program(cmd, work, threads=n, env=env, log_name=f"out.{v.name}", should_stop=should_stop,
                                   check=False, timeout=2.0 * max_seconds + 300.0)
    except external.ExternalProgramError as exc:
        rec["error"] = str(exc).splitlines()[0]
        return rec
    lines = [x for x in run.text().splitlines() if x.startswith("{")]
    if run.returncode != 0 or not lines:
        rec["error"] = f"OpenMM exited with code {run.returncode}: {run.tail(3)}"
        return rec
    rec.update(json.loads(lines[-1]))
    rec["threads"] = n
    return rec


def verified(rec: Mapping) -> bool:
    """A variant whose numbers can be used: a CPU one (the reference), or one whose run-0 check passed."""
    return str(rec.get("check", "") or "") in ("", "ok")


# ------------------------------------------------------------------------------------------- the system
def cache_root() -> Path:
    """Per-machine cache (not the synced settings folder); ``QUANTABENCH_CACHE_DIR`` overrides it."""
    from platformdirs import user_cache_dir

    root = os.environ.get("QUANTABENCH_CACHE_DIR", "").strip() or user_cache_dir("QuantaBench", appauthor=False)
    return Path(root)


def system_dir(size: str) -> Path:
    from .. import __version__

    if size not in SYSTEMS:
        raise ValueError(f"MD benchmark size {size!r}: choose {', '.join(SYSTEMS)}")
    return cache_root() / "md_bench" / f"{__version__}-b{MD_BENCH_VERSION}" / size


def _read_meta(d: Path) -> dict | None:
    try:
        meta = json.loads((d / "bench_system.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    ok = isinstance(meta, dict) and all((d / f).is_file() for f in ("bench_start.data", "ff.params", "molecule.json"))
    return meta if ok else None


def _vacuum_A(size: str, liquid_lz: float) -> float:
    """Å of vacuum on each side of ``size``'s slab (0 for a bulk box)."""
    v = SYSTEMS[size].get("vacuum")
    if not v:
        return 0.0
    return 0.5 * float(liquid_lz) if v == "half" else float(v)


def _prep_input(tm, target_box: Sequence[float], vacuum: float = 0.0) -> str:
    from . import liquid as L

    cfg = L.LiquidSlabConfig(dispersion="cutoff12").resolved()
    lx, ly, lz = (float(x) for x in target_box)
    lines = ["# QuantaBench MD benchmark (MD-BENCH-1/2): the start configuration, made once per version and size",
             "# minimize the packing, NVT-compress to the estimated density in 1 ps, relax 0.5 ps, minimize again"
             + (f"; then open a slab with {vacuum:.4g} A of vacuum on each side (as in.slab) and relax it 1 ps"
                if vacuum else ""),
             "", f"variable rc index {cfg.cutoff:g}", "variable kacc index 1e-4", ""]
    lines += L._styles(tm, cfg, "bulk")
    lines += ["read_data       system.data", "include         ff.params"]
    lines += L._kspace(tm, cfg)
    lines += ["neighbor        2.0 bin", "neigh_modify    delay 0 every 1 check yes",
              "thermo_style    custom step temp press pe density", "thermo          500",
              "min_style       cg", "minimize        1.0e-4 1.0e-6 1000 10000", "reset_timestep  0",
              f"velocity        all create {TEMPERATURE:g} {SEED} dist gaussian mom yes rot yes",
              "timestep        0.5",
              f"fix             nvt all nvt temp {TEMPERATURE:g} {TEMPERATURE:g} 50.0",
              f"fix             squeeze all deform 1 x final 0.0 {lx:.6f} y final 0.0 {ly:.6f} z final 0.0 {lz:.6f} "
              "remap x",
              "run             2000", "unfix           squeeze", "run             1000", "unfix           nvt",
              "minimize        1.0e-4 1.0e-6 1000 10000"]
    if vacuum:
        lines += ["", f"variable        vacuum index {vacuum:.6f}", *L.open_box_lines(),
                  "reset_timestep  0", "timestep        1.0",
                  f"velocity        all create {TEMPERATURE:g} {SEED + 1} dist gaussian mom yes rot yes",
                  f"fix             nvt all nvt temp {TEMPERATURE:g} {TEMPERATURE:g} {TDAMP:g}",
                  "fix             nodrift all momentum 100 linear 1 1 1",
                  "thermo_style    custom step temp press pe density",
                  f"run             {SLAB_RELAX_STEPS}", "unfix           nodrift", "unfix           nvt"]
    else:
        lines += ["variable        lz0 equal $(lz)"]
    lines += ["write_data      bench_start.data nocoeff",
              'print           "QB_PREP density $(density:%.6f) lx $(lx:%.6f) ly $(ly:%.6f) lz $(lz:%.6f) '
              'liquid_lz $(v_lz0:%.6f) pe $(pe:%.6f)"', ""]
    return "\n".join(lines)


def _keep_failed(tmp: Path, size: str) -> Path | None:
    """Move a failed set-up folder aside with its inputs and logs (the bulky files pruned); its new path, or None."""
    from ..chem import external

    if not tmp.is_dir():
        return None
    dest = tmp.parent / f"failed-{size}-{datetime.now():%Y%m%d-%H%M%S}"
    try:
        external.prune_scratch(tmp)
        tmp.rename(dest)
    except OSError:
        return None
    return dest


def prepare_system(size: str, lammps: LammpsInfo, threads: int, *, log: Callable[[str], None] = _noop,
                   should_stop: Callable[[], bool] | None = None) -> dict:
    """The cached benchmark box of ``size`` (``SYSTEMS``): its metadata, made with ``lammps`` on the CPU when it is
    not cached yet (seconds to a few minutes). The folder appears whole or not at all (built next to it, then
    renamed), so a benchmark that is stopped, or two that run at once, never leave half a system. A set-up that
    fails keeps its inputs and logs in ``failed-<size>-<time>`` next to it, and the error says where and how LAMMPS
    ended (``exit_reason``)."""
    from .. import __version__
    from ..chem import external
    from ..chem.optimize import Cancelled
    from . import forcefield as F
    from . import liquid as L

    final = system_dir(size)
    meta = _read_meta(final)
    if meta is not None:
        return meta
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.parent / f".{size}.{os.getpid()}.{time.time_ns()}"
    geometry = geometry_of(size)
    log(f"  md: building the {size} benchmark box ({geometry}; once per QuantaBench version)...")
    t0 = time.perf_counter()
    try:
        mol = F.prepare_molecule(SMILES, seed=SEED)
        rho, how = F.estimate_density(mol)
        tm = F.type_molecule(SMILES, MOLECULE, FORCEFIELD, mol=mol)
        cfg = L.LiquidSlabConfig(dispersion="cutoff12", packing="auto", seed=SEED, **SYSTEMS[size]["config"])
        setup = L.write_liquid_slab(tm, tmp, cfg, density_estimate=rho, density_source=how)
        for name in ("in.bulk", "in.slab", "in.run", "README.txt"):
            (tmp / name).unlink(missing_ok=True)
        shutil.rmtree(tmp / "packing", ignore_errors=True)
        vacuum = _vacuum_A(size, setup.target_box[2])
        (tmp / "molecule.json").write_text(json.dumps(tm.to_dict()) + "\n", encoding="utf-8")
        (tmp / "in.prep").write_text(_prep_input(tm, setup.target_box, vacuum), encoding="utf-8")
        args, n, _label = accelerator_args(lammps, "cpu", threads)
        run = external.run_program([lammps.path, "-nocite", "-log", "log.prep", *args, "-in", "in.prep"], tmp,
                                   threads=n, log_name="out.prep", should_stop=should_stop, timeout=3600.0,
                                   check=False)
        text = ""
        with contextlib.suppress(OSError):
            text = (tmp / "log.prep").read_text(encoding="utf-8", errors="replace")
        done = parse_tagged(text, "QB_PREP")
        if run.returncode != 0 or done is None or not (tmp / "bench_start.data").is_file():
            errs = lammps_errors(text) or lammps_errors(run.text())
            if errs:
                why = errs[-1]
            elif run.returncode != 0:
                why = (f"LAMMPS ({lammps.path}) exited with code {run.returncode} after {run.seconds:.0f} s"
                       + (f", {exit_reason(run.returncode)}" if exit_reason(run.returncode) else ""))
            else:
                why = "the preparation run wrote no bench_start.data"
            raise RuntimeError(why)
        lx, ly, lz = (done.get(k, 0.0) for k in ("lx", "ly", "lz"))
        liquid_lz = done.get("liquid_lz") or lz
        mass = tm.mass * setup.n_molecules / F.AVOGADRO                     # g
        meta = {"bench_version": MD_BENCH_VERSION, "quantabench_version": __version__, "size": size,
                "geometry": geometry, "molecule": MOLECULE, "smiles": SMILES, "forcefield": tm.label,
                "molecules": setup.n_molecules, "atoms": setup.n_atoms,
                "density_g_cm3": round(mass / (lx * ly * liquid_lz * 1e-24), 4) if lx * ly * liquid_lz else 0.0,
                "box_A": [round(x, 4) for x in (lx, ly, lz)], "liquid_lz_A": round(liquid_lz, 4),
                "vacuum_A": round(vacuum, 4), "packing": setup.packing,
                "density_source": how, "temperature_K": TEMPERATURE, "timestep_fs": TIMESTEP,
                "prep_lammps": lammps.version, "prep_threads": n,
                "prep_seconds": round(time.perf_counter() - t0, 1), "created": _now()}
        (tmp / "bench_system.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
        try:
            tmp.rename(final)
        except OSError:
            if _read_meta(final) is None:      # not another benchmark that finished first
                raise
            shutil.rmtree(tmp, ignore_errors=True)
            return _read_meta(final) or meta
    except Cancelled:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    except Exception as exc:
        kept = _keep_failed(tmp, size)
        shutil.rmtree(tmp, ignore_errors=True)
        if kept is not None:
            raise RuntimeError(f"{exc}; its inputs and logs are in {kept}") from exc
        raise
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    shape = (f"a {meta['box_A'][0]:.0f} x {meta['box_A'][1]:.0f} Å slab, {meta['liquid_lz_A']:.0f} Å of liquid and "
             f"{meta['vacuum_A']:.0f} Å of vacuum on each side" if geometry == "slab" else "bulk")
    log(f"  md: {meta['atoms']:,} atoms ({meta['molecules']} x {MOLECULE}, {meta['packing']} packing, {shape}) at "
        f"{meta['density_g_cm3']:.3f} g/cm3, ready in {meta['prep_seconds']:.0f} s")
    return meta


# ------------------------------------------------------------------------------------------- inputs and runs
def _hms(seconds: float) -> str:
    s = max(1, int(round(seconds)))
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def bench_input(tm, dispersion: str, *, steps: int, warmup: int = WARMUP_STEPS,
                max_seconds: float = DEFAULT_MAX_SECONDS, label: str = "", geometry: str = "bulk") -> str:
    """The input of one variant: ``md.liquid``'s styles for ``dispersion`` (the bulk stage's, or with ``geometry``
    "slab" the slab stage's and its surface-tension outputs: the pressure tensor every ``pressure_every`` steps, the
    density profile and ``fix momentum``), run 0 (printed as ``QB_RUN0 ...``), and with ``steps`` > 0 a warm-up and
    the timed NVT run (after ``QB_TIMED``)."""
    from .. import __version__
    from . import liquid as L

    if geometry not in GEOMETRIES:
        raise ValueError(f"geometry {geometry!r}: {' or '.join(GEOMETRIES)}")
    cfg = L.LiquidSlabConfig(dispersion=dispersion).resolved()
    slab = geometry == "slab"
    lines = [f"# QuantaBench {__version__}: MD benchmark (MD-BENCH-{'2' if slab else '1'}), {geometry} box"
             + (f", {label}" if label else ""),
             f"# {tm.name}, {tm.label}; NVT at {TEMPERATURE:g} K, {TIMESTEP:g} fs, from bench_start.data",
             f"# long range: {L._ascii(LR.describe(LR.settings_of(cfg)))}",
             "# run 0: energies and pressure of the start (the CPU/GPU check); then a warm-up and the timed run", "",
             f"variable rc             index {cfg.cutoff:g}", f"variable kacc           index {cfg.kspace_accuracy:g}",
             ""]
    lines += L._styles(tm, cfg, "slab" if slab else "bulk")
    lines += ["read_data       bench_start.data", "include         ff.params"]
    lines += L._kspace(tm, cfg)              # pppm/disp's mesh from this box (the slab's, vacuum and all)
    lines += ["neighbor        2.0 bin", "neigh_modify    delay 0 every 1 check yes",
              f"velocity        all create {TEMPERATURE:g} {SEED} dist gaussian mom yes rot yes",
              f"timestep        {TIMESTEP:g}",
              f"fix             nvt all nvt temp {TEMPERATURE:g} {TEMPERATURE:g} {TDAMP:g}"]
    if slab:
        lines += ["# what a surface-tension run computes as it goes (md.liquid.slab_input)",
                  f"fix             nodrift all momentum {cfg.momentum_every} linear 1 1 1",
                  f"fix             ptens all ave/time {cfg.pressure_every} 1 {cfg.pressure_every} c_thermo_press[1] "
                  'c_thermo_press[2] c_thermo_press[3] file bench_pressure.txt format " %.8g"',
                  f"compute         zbins all chunk/atom bin/1d z lower {cfg.profile_bin:g} units box",
                  f"fix             dprof all ave/chunk {cfg.profile_every} {cfg.profile_repeat} {cfg.profile_freq} "
                  "zbins density/mass file bench_profile.txt"]
    lines += ["thermo_style    custom step temp press pe ke etotal evdwl ecoul elong",
              "thermo_modify   flush yes", f"thermo          {THERMO_EVERY}",
              "run             0 post no",
              'print           "QB_RUN0 ' + " ".join(f"{k} $({k}:%.10g)" for k in RUN0_KEYS) + '"']
    if steps > 0:
        lines += [f"run             {int(warmup)}", 'print           "QB_TIMED"',
                  f"timer           timeout {_hms(max_seconds)} every 10",
                  f"run             {int(steps)}",
                  'print           "QB_END temp $(temp:%.6g) press $(press:%.6g)"']
    return "\n".join(lines) + "\n"


def run_variant(lammps: LammpsInfo, tm, work: Path, v: Variant, threads: int, *, steps: int = MAX_STEPS,
                max_seconds: float = DEFAULT_MAX_SECONDS, should_stop: Callable[[], bool] | None = None,
                tag: str = "", geometry: str = "bulk", box: Sequence[float] | None = None) -> dict:
    """Run one variant in ``work`` (which holds bench_start.data and ff.params) as its own LAMMPS process; the
    record of ``md_throughput["variants"]`` (``error`` set when it failed). ``steps`` 0: run 0 only (a CPU
    reference for the check). ``box``: the benchmark box (Å), for the record. ``Cancelled`` from ``should_stop``
    propagates."""
    from ..chem import external

    name = tag or v.name
    args, n, label = accelerator_args(lammps, v.accelerator, threads)
    kind, rc = dispersion_cutoff(v.dispersion)
    lr = LR.settings(kind, rc)
    rec: dict = {"accelerator": label, "dispersion": v.dispersion, "cutoff_A": rc, "threads": n,
                 "args": " ".join(args), "timestep_fs": TIMESTEP, "lammps": lammps.version,
                 "lammps_build": lammps.build_text(), "date": _now(), "bench_version": MD_BENCH_VERSION,
                 "geometry": geometry, "long_range": LR.describe(lr, box if kind == "ewald" else None),
                 "long_range_settings": lr, "error": ""}
    if box:
        rec["box_A"] = [round(float(x), 4) for x in box]
    if label == "omp-serial":
        rec["note"] = "this LAMMPS's OPENMP package has no OpenMP threading (lmp -h: API Serial): one thread"
    if v.accelerator == "gpu":
        gpu = lammps.accelerators.get("GPU", {})
        rec["gpu_api"], rec["gpu_precision"] = gpu.get("api", ""), gpu.get("precision", "")
    (work / f"in.{name}").write_text(bench_input(tm, v.dispersion, steps=steps, max_seconds=max_seconds,
                                                 label=name, geometry=geometry), encoding="utf-8")
    cmd = [lammps.path, "-nocite", "-log", f"log.{name}", *args, "-in", f"in.{name}"]
    try:
        run = external.run_program(cmd, work, threads=n, log_name=f"out.{name}", should_stop=should_stop,
                                   check=False, timeout=2.0 * max_seconds + 300.0)
        returncode = run.returncode
    except external.ExternalProgramError as exc:      # timed out, or could not start
        rec["error"] = str(exc).splitlines()[0]
        return rec
    text = ""
    with contextlib.suppress(OSError):
        text = (work / f"log.{name}").read_text(encoding="utf-8", errors="replace")
    screen = run.text()
    errors = lammps_errors(text) or lammps_errors(screen)
    run0 = parse_tagged(text, "QB_RUN0") or parse_tagged(screen, "QB_RUN0")
    if run0:
        rec["run0"] = run0
    kspace = parse_kspace(text)
    if kspace:
        rec["kspace"] = kspace
    dev = DEVICE_RE.search(screen) or DEVICE_RE.search(text)
    if dev and v.accelerator != "cpu":
        rec["device"] = dev.group(2)[:120]
    if errors or returncode != 0:
        why = exit_reason(returncode)
        rec["error"] = errors[-1] if errors else (f"LAMMPS exited with code {returncode}" + (f" ({why})" if why else "")
                                                   + f": {run.tail(3)}")
        return rec
    if steps <= 0:
        return rec
    timed = text.split("\nQB_TIMED", 1)[1] if "\nQB_TIMED" in text else ""
    perf = parse_performance(timed)
    if perf is None or not perf.get("ns_per_day"):
        rec["error"] = "no Performance line after the timed run"
        return rec
    rec["ns_per_day"] = round(perf["ns_per_day"], 4)
    rec.update({k: perf[k] for k in ("hours_per_ns", "timesteps_per_s", "katom_steps_per_s", "steps", "atoms",
                                     "procs", "omp_threads") if k in perf})
    rec["seconds"] = round(perf.get("loop_s", 0.0), 2)
    breakdown = parse_breakdown(timed)
    if breakdown:
        rec["breakdown"] = breakdown
    end = parse_tagged(text, "QB_END") or {}
    if "temp" in end:
        rec["final_temp_K"] = end["temp"]
        if not math.isfinite(end["temp"]) or end["temp"] > 2.0 * TEMPERATURE:
            rec["note"] = f"unstable run: T = {end['temp']:g} K at the end"
    return rec


def plan_variants(lammps: LammpsInfo, requested: Sequence[str] | None
                  ) -> tuple[list[Variant], dict[str, str], list[str]]:
    """(variants to run, CPU first; {asked-for variant: why it cannot run}; notes on default variants left out)."""
    explicit = requested is not None
    names = list(requested) if explicit else list(DEFAULT_VARIANTS)
    run: list[Variant] = []
    errors: dict[str, str] = {}
    notes: list[str] = []
    for name in names:
        v = parse_variant(name)
        ok, why = can_run(lammps, v)
        if ok:
            if v not in run:
                run.append(v)
        elif explicit:
            errors[v.name] = why
        elif why not in notes:
            notes.append(why)
    run.sort(key=lambda v: _order(v.name))
    return run, errors, notes


def run_md_benchmark(variants: Sequence[str] | None = None, *, size: str = DEFAULT_SIZE, threads: int = 0,
                     max_seconds: float = DEFAULT_MAX_SECONDS, steps: int = MAX_STEPS,
                     should_stop: Callable[[], bool] | None = None, log: Callable[[str], None] | None = None,
                     lammps: LammpsInfo | None = None, lammps_path: str | None = None) -> dict:
    """Benchmark ``variants`` (default: those of ``DEFAULT_VARIANTS`` this LAMMPS can run) on the ``size`` box and
    return the ``md_throughput`` record (module docstring); each variant's key is ``record_key(name, geometry)``.
    ``lammps_path``: this executable instead of the one QuantaBench finds (``--md-lammps``). Never raises for a
    missing or failing LAMMPS: the record says why (``skipped``, a variant's ``error``). A stop (``should_stop``)
    sets ``cancelled``."""
    from .. import __version__
    from ..chem import external
    from ..chem.optimize import Cancelled
    from . import forcefield as F

    log = log or _noop
    stop = should_stop or (lambda: False)
    t_all = time.perf_counter()
    out: dict = {"bench_version": MD_BENCH_VERSION, "date": _now(), "quantabench_version": __version__,
                 "variants": {}, "notes": [], "skipped": ""}
    geometry = geometry_of(size)                     # ValueError for an unknown size
    if lammps is None:
        lammps, why = find_lammps(path=lammps_path)
        if lammps is None:
            log(f"  md: skipped, {why}")
            out["skipped"] = why
            return out
    out.update(lammps=lammps.version, lammps_source=lammps.source, packages=list(lammps.packages),
               accelerators=dict(lammps.accelerators), gpu_present=lammps.gpu_present)
    todo, refused, notes = plan_variants(lammps, variants)
    if lammps.omp_serial:
        notes.append("this LAMMPS's OPENMP package was built without OpenMP (lmp -h: OPENMP package API: Serial), so "
                     "its CPU variants run on one thread; a conda-forge or self-built LAMMPS has threads")
    out["notes"] = notes
    for name, why in refused.items():
        out["variants"][record_key(name, geometry)] = {"error": why, "date": _now(), "lammps": lammps.version,
                                                      "geometry": geometry}
        log(f"  md {name:16s} not run: {why}")
    for note in notes:
        log(f"  md: {note}")
    if not todo:
        out["skipped"] = "no MD variant can run with this LAMMPS"
        return out
    n_threads = int(threads) if threads and threads > 0 else default_threads()
    log(f"  md: LAMMPS {lammps.version or '?'} ({lammps.path}; {lammps.source or 'given'}); {size} box ({geometry}); "
        f"variants {', '.join(v.name for v in todo)}")
    try:
        system = prepare_system(size, lammps, n_threads, log=log, should_stop=stop)
    except Cancelled:
        out["cancelled"] = True
        return out
    except Exception as exc:  # noqa: BLE001 - reported in the record, never fatal for the profile
        head = "" if isinstance(exc, RuntimeError) else f"{type(exc).__name__}: "
        out["skipped"] = f"the {size} benchmark box could not be built: {head}{str(exc).splitlines()[0]}"
        log(f"  md: skipped, {out['skipped']}")
        return out
    out["system"] = {k: system.get(k) for k in ("size", "geometry", "atoms", "molecules", "molecule", "forcefield",
                                                "density_g_cm3", "box_A", "liquid_lz_A", "vacuum_A", "packing",
                                                "temperature_K", "timestep_fs")}
    out["system"].update(geometry=geometry, ensemble="NVT")
    box = system.get("box_A") or None
    src = system_dir(size)
    tm = F.TypedMolecule.from_dict(json.loads((src / "molecule.json").read_text(encoding="utf-8")))
    work = external.make_scratch_dir("md_bench")
    for f in ("bench_start.data", "ff.params", "molecule.json"):
        shutil.copy2(src / f, work / f)
    refs: dict[str, dict | None] = {}
    failed = False
    try:
        for v in todo:
            if stop():
                out["cancelled"] = True
                break
            if v.accelerator != "cpu" and v.dispersion not in refs:     # the CPU reference at run 0 only
                ref = run_variant(lammps, tm, work, Variant("cpu", v.dispersion), n_threads, steps=0,
                                  max_seconds=max_seconds, should_stop=stop, tag=f"ref-cpu-{v.dispersion}",
                                  geometry=geometry, box=box)
                refs[v.dispersion] = ref.get("run0")
            if v.accelerator.startswith("openmm"):
                rec = run_openmm_variant(work, v, n_threads, steps=steps, max_seconds=max_seconds, geometry=geometry,
                                         box=box, should_stop=stop)
            else:
                rec = run_variant(lammps, tm, work, v, n_threads, steps=steps, max_seconds=max_seconds,
                                  should_stop=stop, geometry=geometry, box=box)
            rec["size"] = size
            rec.setdefault("atoms", system.get("atoms"))
            if v.accelerator == "cpu":
                refs.setdefault(v.dispersion, rec.get("run0"))
            elif v.accelerator.startswith("openmm"):
                rec["check"], rec["check_detail"] = check_energy(refs.get(v.dispersion), rec.get("run0"))
            else:
                rec["check"], rec["check_detail"] = check_consistency(refs.get(v.dispersion), rec.get("run0"))
                if refs.get(v.dispersion):
                    rec["reference_run0"] = refs[v.dispersion]
            key = record_key(v.name, geometry)
            out["variants"][key] = rec
            failed = failed or bool(rec.get("error"))
            log("  " + _log_line(key, rec))
    except Cancelled:
        out["cancelled"] = True
    finally:
        keep = failed and not out.get("cancelled")
        if keep:
            external.prune_scratch(work)
            log(f"  md: LAMMPS inputs and logs of the failed variant(s) kept in {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)
    out["seconds"] = round(time.perf_counter() - t_all, 1)
    log(f"  md benchmark took {out['seconds']:.0f} s")
    return out


def _log_line(name: str, rec: Mapping) -> str:
    if rec.get("error"):
        return f"md {name:18s} FAILED: {rec['error']}"
    txt = (f"md {name:18s} {fmt_ns(rec['ns_per_day']):>6s} ns/day ({int(rec.get('atoms') or 0):,} atoms, "
           f"{_context(rec)}, {int(rec.get('steps') or 0):,} steps in {float(rec.get('seconds') or 0):.1f} s"
           + (f", on {rec['device']}" if rec.get("device") else "") + ")")
    if rec.get("check"):
        txt += f"; check {rec['check']}: {rec.get('check_detail', '')}"
    if rec.get("note"):
        txt += f"; {rec['note']}"
    return txt


# ------------------------------------------------------------------------------------------- the registry
def clean_record(value) -> dict:
    """An ``md_throughput`` value read from JSON: a dict whose ``variants`` is a dict of dicts with a finite,
    non-negative ``ns_per_day`` where there is one (anything else is dropped); other keys as they are."""
    if not isinstance(value, dict):
        return {}
    out = {str(k): v for k, v in value.items()}
    variants = out.get("variants")
    if "variants" in out:
        clean: dict[str, dict] = {}
        for k, rec in (variants.items() if isinstance(variants, dict) else ()):
            if not isinstance(rec, dict):
                continue
            rec = dict(rec)
            x = rec.get("ns_per_day")
            if x is not None and not (isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
                                      and x >= 0):
                rec.pop("ns_per_day")
            clean[str(k)] = rec
        out["variants"] = clean
    return out


def merge(previous: Mapping | None, new: Mapping | None) -> dict:
    """``new`` with the variants of ``previous`` it did not measure again (each carries its own atoms, threads,
    LAMMPS and date); a re-measured variant replaces the old one, a failure included. A new record that measured
    nothing (LAMMPS missing, a stop) keeps the previous one, with its reason noted."""
    prev = clean_record(dict(previous or {}))
    cur = clean_record(dict(new or {}))
    if not cur:
        return prev
    if not cur.get("variants") and prev.get("variants"):
        why = cur.get("skipped") or ("cancelled" if cur.get("cancelled") else "")
        if why:
            prev["last_attempt"] = {"date": cur.get("date", ""), "skipped": why}
        return prev
    variants = {**(prev.get("variants") or {}), **(cur.get("variants") or {})}
    return {**cur, "variants": variants}


def rated(md: Mapping | None) -> dict[str, dict]:
    """{variant: record} of the variants with a measured ns/day."""
    variants = (md or {}).get("variants") or {}
    return {k: v for k, v in variants.items() if isinstance(v, dict) and (v.get("ns_per_day") or 0) > 0}


# ------------------------------------------------------------------------------------------- display
def fmt_ns(x: float) -> str:
    x = float(x)
    return f"{x:,.0f}" if x >= 100 else f"{x:.3g}"


def _context(rec: Mapping) -> str:
    label = str(rec.get("accelerator", ""))
    if label == "openmm":
        return f"OpenMM {rec.get('platform', '')} {rec.get('precision', '')}".strip()
    if label in ("gpu", "kokkos-gpu"):
        return str(rec.get("gpu_api") or "GPU") if label == "gpu" else "KOKKOS GPU"
    return f"{int(rec.get('threads') or 1)} thr"


def describe_lines(md: Mapping | None) -> list[str]:
    """Lines for ``quantabench-cli machines``: 'md: cpu-ewald 6.5 ns/day (9,780 atoms, 6 thr), gpu-cutoff16 10.6
    ns/day (9,780 atoms, OpenCL) [pressure check: mismatch] (LAMMPS 4 Jul 2026)', then failures."""
    md = md or {}
    if not md:
        return []
    lines = []
    good = rated(md)
    if good:
        parts, prev = [], None
        for name in sorted(good, key=_order):
            rec = good[name]
            txt = f"{name} {fmt_ns(rec['ns_per_day'])} ns/day"
            ctx = f"{int(rec.get('atoms') or 0):,} atoms, {_context(rec)}"
            if ctx != prev:
                txt += f" ({ctx})"
                prev = ctx
            if rec.get("check"):
                txt += f" [pressure check: {rec['check']}]"
            parts.append(txt)
        lines.append("  md:         " + ", ".join(parts) + (f" (LAMMPS {md['lammps']})" if md.get("lammps") else ""))
    elif md.get("skipped"):
        lines.append(f"  md:         not measured ({md['skipped']})")
    for name, rec in sorted(((md.get("variants") or {}).items()), key=lambda kv: _order(kv[0])):
        if isinstance(rec, dict) and rec.get("error") and name not in good:
            lines.append(f"  md benchmark {name} failed: {rec['error']}")
    return lines


def short_text(md: Mapping | None) -> str:
    """'cpu-ewald 6.5, gpu-cutoff16 10.6 (unverified)' for the Machines dock ("" when not measured)."""
    good = rated(md)
    parts = [f"{k} {fmt_ns(v['ns_per_day'])}" + ("" if verified(v) else " (unverified)")
             for k, v in sorted(good.items(), key=lambda kv: _order(kv[0]))]
    return ", ".join(parts)


# ------------------------------------------------------------------------------------------- estimates
@dataclass
class Estimate:
    variant: str                   # "cpu-ewald"
    hours: float
    ns_per_day: float              # for the job's atoms (and box)
    measured_ns_per_day: float
    measured_atoms: int
    threads: int
    check: str = ""
    note: str = ""
    geometry: str = "bulk"         # of the benchmark box it comes from
    key: str = ""                  # its registry key ("cpu-ewald@slab")

    @property
    def verified(self) -> bool:
        return self.check in ("", "ok")


def _volume(box: Sequence[float] | None) -> float:
    try:
        v = math.prod(float(x) for x in box) if box and len(box) == 3 else 0.0
    except (TypeError, ValueError):
        return 0.0
    return v if math.isfinite(v) and v > 0 else 0.0


def scaled_rate(rec: Mapping, atoms: int, box: Sequence[float] | None = None) -> tuple[float, str]:
    """(ns/day of ``atoms`` atoms, how it was scaled) from a benchmark record (module docstring, "Scaling"): the
    measured ns/day x measured atoms / ``atoms``; with ``box`` (the job's, Å) and the record's box and k-space share
    (``breakdown["Kspace"]``), that share scales with the box volume instead. ``how`` is "" for the atom rule."""
    measured = float(rec["ns_per_day"])
    m_atoms = int(rec.get("atoms") or 0) or int(atoms)
    n_ratio = float(atoms) / m_atoms
    f = (rec.get("breakdown") or {}).get("Kspace") if isinstance(rec.get("breakdown"), Mapping) else None
    v_job, v_rec = _volume(box), _volume(rec.get("box_A"))
    if v_job and v_rec and isinstance(f, (int, float)) and 0.0 <= f <= 100.0:
        share = float(f) / 100.0
        cost = (1.0 - share) * n_ratio + share * v_job / v_rec
        if cost > 0 and abs(cost - n_ratio) > 0.005 * n_ratio:
            return measured / cost, (f"its k-space share ({share:.0%}) scaled by the box volume "
                                     f"(x{v_job / v_rec:.2f}) and the rest by the atoms (x{n_ratio:.2f})")
    return measured / n_ratio, ""


def estimate_hours(md: Mapping | None, atoms: int, ns: float, *, dispersion: str | None = None,
                   accelerator: str | None = None, include_unverified: bool = False, geometry: str = "slab",
                   box: Sequence[float] | None = None) -> list[Estimate]:
    """Wall hours for ``ns`` of MD of ``atoms`` atoms from a machine's ``md_throughput``, one per usable variant,
    fastest first, by ``scaled_rate`` (``box``: the job's box, used with records of the same geometry).

    ``geometry``: the job's, "slab" (an ``md liquid-slab`` run) or "bulk". Records measured in that geometry are
    used; an accelerator with none falls back to the other geometry's, noted (a bulk box has no vacuum mesh, so for
    a slab its hours are a lower bound). ``dispersion``: only that variant (a cutoff that was not measured is scaled
    from the nearest measured cutoff on the same accelerator by (rc_measured/rc)³, the pair cost; noted). Unverified
    variants (a failed CPU/GPU check) only with ``include_unverified``."""
    if atoms <= 0 or ns <= 0:
        raise ValueError("atoms and ns must be positive")
    if geometry not in GEOMETRIES:
        raise ValueError(f"geometry {geometry!r}: {' or '.join(GEOMETRIES)}")
    want = parse_variant(f"cpu-{dispersion}").dispersion if dispersion else None
    out: list[Estimate] = []
    by_accel: dict[str, list[tuple[str, str, dict, str]]] = {}
    for key, rec in rated(md).items():
        name, geo = split_key(key)
        try:
            v = parse_variant(name)
        except ValueError:
            continue
        if accelerator and v.accelerator != accelerator:
            continue
        by_accel.setdefault(v.accelerator, []).append((v.dispersion, geo, rec, key))
    for accel, every in by_accel.items():
        mine = [x for x in every if x[1] == geometry]
        fallback = ""
        if not mine:
            mine = every
            fallback = ("from the bulk benchmark: the slab's vacuum mesh was not timed, so a lower bound"
                        if geometry == "slab" else "from the slab benchmark")
        items = [x for x in mine if include_unverified or verified(x[2])]
        picks: list[tuple[str, tuple, float, str]] = []
        if want is None:
            picks = [(x[0], x, 1.0, "") for x in items]
        elif any(x[0] == want for x in mine):          # measured: that number or none (never scaled around a
            picks = [(want, x, 1.0, "") for x in items if x[0] == want]    # variant that failed its check)
        elif want != "ewald":
            _, rc = dispersion_cutoff(want)
            cuts = [(abs(dispersion_cutoff(x[0])[1] - rc), x) for x in items if x[0] != "ewald"]
            if cuts:
                _, x = min(cuts, key=lambda t: t[0])
                rm = dispersion_cutoff(x[0])[1]
                picks = [(want, x, (rm / rc) ** 3, f"scaled from {accel}-{x[0]} by ({rm:g}/{rc:g})^3")]
        for disp, (_d, geo, rec, key), factor, note in picks:
            rate, how = scaled_rate(rec, atoms, box if geo == geometry else None)
            rate *= factor
            old = ("timed the coarser pppm/disp settings of before MD-LR (faster): measure again"
                   if disp == "ewald" and not rec.get("long_range") else "")
            notes = [n for n in (note, how, fallback, old) if n]
            out.append(Estimate(f"{accel}-{disp}", ns / rate * 24.0, rate, float(rec["ns_per_day"]),
                                int(rec.get("atoms") or 0) or atoms, int(rec.get("threads") or 1),
                                str(rec.get("check", "") or ""), "; ".join(notes), geo, key))
    return sorted(out, key=lambda e: e.hours)


def best_estimate(md: Mapping | None, atoms: int, ns: float, **kw) -> Estimate | None:
    est = estimate_hours(md, atoms, ns, **kw)
    return est[0] if est else None


def format_hours(hours: float) -> str:
    if hours < 1.0:
        return f"{hours * 60:.0f} min"
    if hours < 48.0:
        return f"{hours:.1f} h"
    return f"{hours / 24:.1f} days ({hours:,.0f} h)"


def estimate_lines(atoms: int, ns: float, dispersion: str | None = None, profiles: Iterable | None = None,
                   this: str | None = None, *, geometry: str = "slab",
                   box: Sequence[float] | None = None) -> list[str]:
    """One line per machine in the registry with an MD benchmark: the fastest verified variant for ``ns`` of
    ``atoms`` atoms (and a faster unverified one, flagged). [] when no machine has one."""
    from ..dist import profile as profile_mod

    if profiles is None:
        try:
            profiles = profile_mod.load_profiles()
            this = this or profile_mod.machine_name()
        except Exception:  # noqa: BLE001 - no registry: no estimate
            return []
    lines = []
    kw = {"dispersion": dispersion, "geometry": geometry, "box": box}
    for p in profiles:
        md = getattr(p, "md_throughput", None) or {}
        if not rated(md):
            continue
        best = best_estimate(md, atoms, ns, **kw)
        anyv = best_estimate(md, atoms, ns, include_unverified=True, **kw)
        who = p.name + (" (this machine)" if this and p.name == this else "")
        if best is None and anyv is None:
            lines.append(f"{who}: no benchmark of {dispersion or 'any'} variant")
            continue
        txt = f"{who}: no verified variant" if best is None else (
            f"{who}: {format_hours(best.hours)} with {best.variant} ({fmt_ns(best.ns_per_day)} ns/day at "
            f"{atoms:,} atoms; measured {fmt_ns(best.measured_ns_per_day)} on {best.measured_atoms:,} "
            f"({best.geometry} box)" + (f", {best.note}" if best.note else "") + ")")
        if anyv is not None and (best is None or anyv.hours < best.hours) and not anyv.verified:
            txt += (f"; {anyv.variant} would take {format_hours(anyv.hours)} but is unverified (pressure check: "
                    f"{anyv.check or 'none'})")
        lines.append(txt)
    return lines
