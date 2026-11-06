"""The xtb program (Grimme group) as engines: GFN2-xTB, GFN1-xTB and GFN-FF with ALPB/GBSA solvation.

Engines registered here (``engines.PLUGINS`` imports this module lazily; importing it runs nothing):

=========  =========================  =============  =========  ======================================
key        label                      family         prefix     notes
=========  =========================  =============  =========  ======================================
xtb-gfn2   GFN2-xTB (xtb program)     tight binding  gfn2       dipole, gap, static polarizability
xtb-gfn1   GFN1-xTB (xtb program)     tight binding  gfn1       dipole, gap
gfnff      GFN-FF (xtb program)       force field    gfnff      EEQ charges; dipole from the charges
=========  =========================  =============  =========  ======================================

xtb >= 6.5 is required (built-in parameter files, ``<name>.engrad`` output); 6.7.1 is recommended and is
what the parsers below were checked against. Everything runs through ``external.run_program``, so it
works natively (Linux, macOS, the Windows zip) and as ``wsl:xtb`` inside WSL.

Which xtb output is read (xtb 6.7.1, xyz input ``calc.xyz``, ``--grad --json``):

- energy and gradient: ``calc.engrad`` (ORCA format, Eh and Eh/bohr, 12 decimals). It is rewritten on
  every run. The Turbomole-style ``gradient`` and ``energy`` files are only a fallback: older versions
  append a new cycle to them when they already exist, which is a trap in a reused folder.
- atomic charges: ``charges`` (GFN1: Mulliken, GFN2: Mulliken from the GFN2 density) or
  ``gfnff_charges`` (GFN-FF EEQ charges at the current geometry); ``xtbout.json`` as a fallback.
- dipole and HOMO-LUMO gap: ``xtbout.json`` (only GFN1/GFN2 write it; the log prints the dipole with
  3 decimals only). For GFN-FF the dipole is Σ qᵢ rᵢ of its charges (what xtb prints as "q only").
- static polarizability (GFN2 only) and the solvation term ``-> Gsolv``: the log's property printout.
- **Ions:** xtb reports the dipole about the coordinate origin, so for a charged molecule it depends
  on where the molecule sits (a carboxylate shifted by 10 Å gains 48 D). Everything here is shifted to
  the centre of mass (μ - Q·R_com), the convention ``descriptors.dipole_debye`` uses.

The ASE calculator keeps one scratch folder per instance (under ``external.scratch_root()``, removed on
``close()`` or when the calculator is garbage-collected, unless ``settings.engines.keep_scratch``): xtb
reads its ``xtbrestart`` there, so the SCF of the next geometry starts from the last charges (water:
9 -> 3 SCF iterations), and GFN-FF reuses its ``gfnff_topo`` topology. Nothing is written to the
user's folders.

Cancellation: the calculator polls an optional ``should_stop`` (factory option); a single point takes
milliseconds to seconds, so ASE-driven optimizations stop between steps anyway. ``native_optimize`` and
``properties`` pass ``should_stop`` to ``run_program``, which kills xtb mid-run.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import weakref
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from . import engines, external
from .engines import BOHR_ANG, HARTREE_EV

MIN_VERSION = (6, 5)
RECOMMENDED_VERSION = "6.7.1"

EV_PER_EH_BOHR = HARTREE_EV / BOHR_ANG     # 51.42208619 eV/Å per Eh/bohr
AU_DEBYE = 2.541746473                     # 1 e·bohr in debye
AU_POLARIZABILITY_A3 = 0.1481847           # 1 bohr³ in Å³
KCAL_PER_HARTREE = 627.509474

#: (key, label, family, prefix, xtb method flags)
METHODS: dict[str, tuple[str, str, str, tuple[str, ...]]] = {
    "xtb-gfn2": ("GFN2-xTB (xtb program)", "tight binding", "gfn2", ("--gfn", "2")),
    "xtb-gfn1": ("GFN1-xTB (xtb program)", "tight binding", "gfn1", ("--gfn", "1")),
    "gfnff": ("GFN-FF (xtb program)", "force field", "gfnff", ("--gfnff",)),
}

#: xtb's optimization levels as xtb 6.7.1 prints them: (name, energy change Eh, gradient norm Eh/bohr).
OPT_LEVELS: tuple[tuple[str, float, float], ...] = (
    ("crude", 5e-4, 1e-2), ("sloppy", 1e-4, 6e-3), ("loose", 5e-5, 4e-3), ("lax", 2e-5, 2.5e-3),
    ("normal", 5e-6, 1e-3), ("tight", 1e-6, 8e-4), ("vtight", 1e-7, 2e-4), ("extreme", 5e-8, 5e-5),
)

INPUT = "calc"   # base name of the geometry file xtb reads; xtb names calc.engrad after it

#: files a run writes that must not survive into the next run (xtbrestart and gfnff_topo are kept)
_OUTPUTS = ("gradient", "energy", "charges", "gfnff_charges", "wbo", "xtbout.json", "xtbtopo.mol",
            ".sccnotconverged", "xtbopt.xyz", "xtbopt.log", ".xtboptok", "xtblast.xyz", "NOT_CONVERGED",
            "xtb.log")
_MOLECULE_STATE = ("xtbrestart", "gfnff_topo", "gfnff_adjacency", "gfnff_lists.json")


# ----------------------------------------------------------------------------- availability
def available() -> tuple[bool, str]:
    info = external.find_program("xtb")
    if info is None:
        return False, f"xtb not found ({external.PROGRAMS['xtb'].install_hint})"
    if info.version and external.version_tuple(info.version) < MIN_VERSION:
        return False, (f"xtb {info.version} at {info.path} is too old: QuantaBench needs xtb >= 6.5 "
                       f"({RECOMMENDED_VERSION} recommended; {external.PROGRAMS['xtb'].install_hint})")
    return True, ""


def opt_level(fmax: float) -> str:
    """The loosest xtb ``--opt`` level that guarantees max |F| <= ``fmax`` (eV/Å).

    xtb converges on the gradient *norm* (all atoms together, Eh/bohr) and the energy change; every
    atomic force is at most the norm, so ``gnorm_threshold * 51.42 <= fmax`` is sufficient:

    crude 0.514, sloppy 0.309, loose 0.206, lax 0.129, normal 0.051, tight 0.041, vtight 0.0103,
    extreme 0.0026 eV/Å. The default fmax 0.03 maps to vtight, 0.05 to tight, 0.1 to normal;
    anything below 0.0026 uses extreme (and ``converged`` then checks the final forces).
    """
    for name, _de, gnorm in OPT_LEVELS:
        if gnorm * EV_PER_EH_BOHR <= fmax:
            return name
    return OPT_LEVELS[-1][0]


# ----------------------------------------------------------------------------- input / output
def _symbols(numbers: Sequence[int]) -> list[str]:
    from ase.data import chemical_symbols

    return [chemical_symbols[int(z)] for z in numbers]


def write_xyz(path: Path, numbers: Sequence[int], positions_A, comment: str = "") -> None:
    pos = np.asarray(positions_A, dtype=float).reshape(-1, 3)
    lines = [str(len(pos)), comment]
    lines += [f"{s:<3s}{x:22.12f}{y:22.12f}{z:22.12f}" for s, (x, y, z) in zip(_symbols(numbers), pos)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_xyz_frames(text: str) -> list[tuple[list[str], np.ndarray, str]]:
    """Frames of a (multi-frame) XYZ file: (symbols, positions Å, comment line)."""
    lines = text.splitlines()
    frames, i = [], 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        n = int(lines[i].split()[0])
        comment = lines[i + 1] if i + 1 < len(lines) else ""
        rows = [ln.split() for ln in lines[i + 2:i + 2 + n]]
        if len(rows) != n:
            raise ValueError(f"truncated XYZ frame ({len(rows)} of {n} atoms)")
        symbols = [r[0].capitalize() for r in rows]
        pos = np.array([[float(v) for v in r[1:4]] for r in rows])
        frames.append((symbols, pos, comment))
        i += 2 + n
    return frames


def parse_engrad(text: str) -> tuple[float, np.ndarray, np.ndarray | None]:
    """(energy Eh, gradient (N, 3) Eh/bohr, coordinates (N, 3) bohr or None) from an ORCA-style .engrad."""
    values = [ln.split() for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    n = int(values[0][0])
    energy = float(values[1][0])
    grad = np.array([float(v[0]) for v in values[2:2 + 3 * n]]).reshape(n, 3)
    coords = None
    rows = values[2 + 3 * n:2 + 4 * n]
    if len(rows) == n and all(len(r) >= 4 for r in rows):
        coords = np.array([[float(x) for x in r[1:4]] for r in rows])
    return energy, grad, coords


def parse_turbomole_gradient(text: str, n_atoms: int) -> tuple[float, np.ndarray]:
    """Last cycle of a Turbomole ``gradient`` file: (energy Eh, gradient (N, 3) Eh/bohr)."""
    cycles = list(re.finditer(r"cycle\s*=\s*\d+\s+SCF energy\s*=\s*(\S+)", text))
    if not cycles:
        raise ValueError("no cycle in the gradient file")
    last = cycles[-1]
    rows = text[last.end():].splitlines()[1:]
    grad_rows = [r.replace("D", "E").split() for r in rows[n_atoms:2 * n_atoms]]
    return float(last.group(1)), np.array([[float(x) for x in r[:3]] for r in grad_rows])


def _float(pattern: str, text: str) -> float | None:
    hits = re.findall(pattern, text)
    return float(hits[-1]) if hits else None


def parse_log(text: str) -> dict[str, float | bool | int | None]:
    """Numbers from xtb's printout (the last occurrence wins, i.e. the final single point of an --opt)."""
    out: dict[str, float | bool | int | None] = {
        "energy_eh": _float(r"::\s*total energy\s+(-?\d+\.\d+)\s+Eh", text),
        "gnorm": _float(r"::\s*gradient norm\s+(\d+\.\d+)\s+Eh", text),
        "gap_ev": _float(r"::\s*HOMO-LUMO gap\s+(-?\d+\.\d+)\s+eV", text),
        "gsolv_eh": _float(r"::\s*->\s*Gsolv\s+(-?\d+\.\d+)\s+Eh", text),
        # "Mol. α(0) /au : 9.428750" (GFN2 only); \S+ instead of α so the log's encoding does not matter
        "alpha_au": _float(r"Mol\.\s+\S+\(0\)\s*/au\s*:\s*(-?\d+\.\d+)", text),
        "scf_failed": bool(re.search(r"did not converge|SCF not converged|sccnotconverged", text, re.I)),
    }
    m = re.search(r"GEOMETRY OPTIMIZATION CONVERGED AFTER\s+(\d+)\s+ITERATIONS", text)
    f = re.search(r"FAILED TO CONVERGE GEOMETRY OPTIMIZATION IN\s+(\d+)\s+ITERATIONS", text)
    out["opt_converged"] = bool(m)
    out["opt_steps"] = int(m.group(1)) if m else int(f.group(1)) if f else len(re.findall(r"\.\s+CYCLE\s+\d+\s+\.", text))
    return out


@dataclass
class XtbResult:
    """One xtb run, in xtb's units (Eh, bohr, e·bohr); dipole about the centre of mass."""

    energy_eh: float
    gradient: np.ndarray | None        # (N, 3) Eh/bohr, None when not requested
    charges: np.ndarray | None
    dipole_au: np.ndarray | None       # about the centre of mass
    gap_ev: float | None
    alpha_au: float | None
    gsolv_eh: float | None
    seconds: float
    log_text: str


def _masses(numbers) -> np.ndarray:
    from ase.data import atomic_masses

    return np.asarray([atomic_masses[int(z)] for z in numbers], dtype=float)


_BAD_ESCAPE = re.compile(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})')   # a backslash that starts no JSON escape


def read_xtbout(path: Path) -> dict:
    """``xtbout.json``, or {} when missing or unreadable. xtb writes its "program call" line verbatim, so on
    Windows the path's backslashes make the file invalid JSON (``Invalid \\escape``); those are escaped and the
    file read again, rather than losing the dipole, gap and charges to the charge-only fallback."""
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    for attempt in (text, _BAD_ESCAPE.sub(r"\\\\", text)):
        try:
            data = json.loads(attempt)
        except ValueError:
            continue
        return data if isinstance(data, dict) else {}
    return {}


def dipole_about_com(numbers, positions_A, charge: int, dipole_origin_au=None, charges=None) -> np.ndarray | None:
    """Dipole (e·bohr) about the centre of mass, from xtb's origin-based dipole or from charges."""
    pos = np.asarray(positions_A, dtype=float).reshape(-1, 3) / BOHR_ANG
    m = _masses(numbers)
    com = (pos * m[:, None]).sum(0) / m.sum()
    if dipole_origin_au is not None:
        return np.asarray(dipole_origin_au, dtype=float).reshape(3) - float(charge) * com
    if charges is not None:
        return ((pos - com) * np.asarray(charges, dtype=float).reshape(-1, 1)).sum(0)
    return None


def read_results(folder: Path, numbers, positions_A, charge: int, method: str, log_text: str,
                 seconds: float = 0.0, grad: bool = True, stem: str = INPUT) -> XtbResult:
    """Collect one run's results from ``folder`` (see the module docstring for the sources)."""
    n = len(numbers)
    info = parse_log(log_text)
    energy, gradient = info["energy_eh"], None
    engrad = folder / f"{stem}.engrad"
    if engrad.is_file():
        energy, gradient, _coords = parse_engrad(engrad.read_text(encoding="utf-8", errors="replace"))
    elif grad and (folder / "gradient").is_file():
        energy, gradient = parse_turbomole_gradient((folder / "gradient").read_text(errors="replace"), n)
    if energy is None:
        raise external.ExternalProgramError("xtb wrote no energy\n" + "\n".join(log_text.splitlines()[-30:]))
    if grad and (gradient is None or gradient.shape != (n, 3)):
        raise external.ExternalProgramError("xtb wrote no gradient\n" + "\n".join(log_text.splitlines()[-30:]))
    data = read_xtbout(folder / "xtbout.json")
    charges = None
    for name in ("gfnff_charges", "charges") if method == "gfnff" else ("charges",):
        if (folder / name).is_file():
            vals = (folder / name).read_text().split()
            if len(vals) == n:
                charges = np.array([float(v) for v in vals])
                break
    if charges is None and len(data.get("partial charges", [])) == n:
        charges = np.asarray(data["partial charges"], dtype=float)
    dip0 = data.get("dipole / a.u.")
    dipole = dipole_about_com(numbers, positions_A, charge, dip0, None if dip0 is not None else charges)
    gap = data.get("HOMO-LUMO gap / eV", info["gap_ev"])
    return XtbResult(float(energy), gradient, charges, dipole, None if method == "gfnff" else gap,
                     info["alpha_au"], info["gsolv_eh"], seconds, log_text)


# ----------------------------------------------------------------------------- running
def _timeout() -> float | None:
    t = float(getattr(external._engine_settings(), "timeout_s", 0.0) or 0.0)
    return t if t > 0 else None


def solvent_args(solvent: str, model: str = "alpb") -> list[str]:
    """``["--alpb", "water"]`` for a canonical solvent name, ``[]`` for the gas phase."""
    if not solvent:
        return []
    from . import solvents

    kw = solvents.for_engine(solvent, model)
    if kw is None:
        raise ValueError(f"xtb has no {model.upper()} parameters for {solvent}")
    return [f"--{model.lower()}", str(kw)]


def command(method: str, *, charge: int = 0, multiplicity: int = 1, solvent: str = "", model: str = "alpb",
            grad: bool = True, opt_level_name: str = "", cycles: int = 0, program=None,
            input_name: str = INPUT + ".xyz", accuracy: float = 1.0) -> list[str]:
    """The xtb command line (file names relative to the run folder, so it also works through WSL)."""
    program = program or external.require_program("xtb")
    cmd = [*program.command(), input_name, *METHODS[method][3]]
    if opt_level_name:
        cmd += ["--opt", opt_level_name, "--cycles", str(max(1, int(cycles)))]
    if grad:
        cmd.append("--grad")
    cmd += ["--chrg", str(int(charge)), "--uhf", str(max(0, int(multiplicity) - 1))]
    if accuracy and method != "gfnff" and abs(float(accuracy) - 1.0) > 1e-12:
        cmd += ["--acc", f"{float(accuracy):g}"]
    cmd += solvent_args(solvent, model)
    if method != "gfnff":
        cmd.append("--json")
    return cmd


def _clean(folder: Path, stem: str = INPUT, molecule: bool = False) -> None:
    for name in (*_OUTPUTS, f"{stem}.engrad", *(_MOLECULE_STATE if molecule else ())):
        try:
            (folder / name).unlink()
        except FileNotFoundError:
            pass


def _fail(what: str, run: external.ProgramRun, folder: Path) -> external.ExternalProgramError:
    text = run.text()
    scf = (folder / ".sccnotconverged").exists() or parse_log(text)["scf_failed"]
    why = "the SCF did not converge" if scf else f"exit code {run.returncode}"
    return external.ExternalProgramError(f"xtb {what} failed ({why})\n{run.tail(30)}")


def run_xtb(folder: Path, method: str, numbers, positions_A, *, charge: int = 0, multiplicity: int = 1,
            solvent: str = "", model: str = "alpb", threads: int = 1, grad: bool = True,
            should_stop: Callable[[], bool] | None = None, log: Callable[[str], None] | None = None,
            program=None, accuracy: float = 1.0) -> XtbResult:
    """One single point in ``folder`` (restart files there are reused). Raises ExternalProgramError
    with the log tail when xtb fails (e.g. SCF not converged)."""
    program = program or external.require_program("xtb")
    folder.mkdir(parents=True, exist_ok=True)
    _clean(folder)
    write_xyz(folder / f"{INPUT}.xyz", numbers, positions_A, "QuantaBench")
    cmd = command(method, charge=charge, multiplicity=multiplicity, solvent=solvent, model=model, grad=grad,
                  program=program, accuracy=accuracy)
    env = external.program_env(threads, via_wsl=program.via_wsl)
    run = external.run_program(cmd, folder, threads=threads, env=env, timeout=_timeout(), log_name="xtb.log",
                               should_stop=should_stop, log=log, check=False)
    if run.returncode != 0:
        raise _fail(f"{METHODS[method][0]} single point", run, folder)
    return read_results(folder, numbers, positions_A, charge, method, run.text(), run.seconds, grad=grad)


# ----------------------------------------------------------------------------- ASE calculator
def _remove(path: str) -> None:
    shutil.rmtree(path, ignore_errors=True)


class XtbCalculator(Calculator):
    """ASE calculator that runs the xtb program for every geometry (energy, forces, dipole, charges).

    Units as in ASE: eV, eV/Å, e·Å (dipole about the centre of mass), e. One scratch folder per
    instance; call ``close()`` (or use it as a context manager) to remove it early.

    ``accuracy`` is xtb's ``--acc`` (GFN1/GFN2). The default 0.1 is 10x tighter than xtb's: with SCF
    restarts the forces at one geometry otherwise scatter by ~5e-5 eV/Å (6e-6 with 0.1; ethanol, same
    run time), which matters for finite-difference Hessians.
    """

    implemented_properties: ClassVar[list[str]] = ["energy", "free_energy", "forces", "dipole", "charges"]

    def __init__(self, numbers, method: str = "xtb-gfn2", charge: int = 0, multiplicity: int = 1,
                 solvent: str = "", threads: int = 1, solvation_model: str = "alpb",
                 should_stop: Callable[[], bool] | None = None, accuracy: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        if method not in METHODS:
            raise ValueError(f"unknown xtb method {method!r}")
        self.method = method
        self.numbers = [int(z) for z in numbers]
        self.charge = int(charge)
        self.multiplicity = int(multiplicity)
        self.solvent = solvent or ""
        self.solvation_model = solvation_model
        solvent_args(self.solvent, solvation_model)          # unknown solvent: fail now, not mid-run
        self.threads = max(1, int(threads or 1))
        self.should_stop = should_stop
        self.accuracy = float(accuracy)
        self.program = external.require_program("xtb")
        self.n_runs = 0
        self.last: XtbResult | None = None
        self._dir: Path | None = None
        self._finalizer = None

    # -- scratch folder
    @property
    def directory_path(self) -> Path:
        """This calculator's scratch folder (created on first use)."""
        if self._dir is None:
            d = Path(tempfile.mkdtemp(prefix=f"xtb_{self.method}_", dir=external.scratch_root()))
            self._dir = d
            if not bool(getattr(external._engine_settings(), "keep_scratch", False)):
                self._finalizer = weakref.finalize(self, _remove, str(d))
        return self._dir

    def close(self) -> None:
        if self._finalizer is not None:
            self._finalizer()
        self._finalizer = None
        self._dir = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):  # weakref.finalize also covers interpreter exit; this just makes it prompt
        try:
            self.close()
        except Exception:  # noqa: S110 - never raise from a finalizer
            pass

    def same_setup(self, method: str, charge: int, multiplicity: int, solvent: str) -> bool:
        return (self.method, self.charge, self.multiplicity, self.solvent) == (
            method, int(charge), int(multiplicity), solvent or "")

    # -- results
    def store(self, atoms, res: XtbResult) -> None:
        """Put an xtb result (computed at ``atoms``' geometry) into ``self.results`` in ASE units."""
        e = res.energy_eh * HARTREE_EV
        self.results = {"energy": e, "free_energy": e}
        if res.gradient is not None:
            self.results["forces"] = -res.gradient * EV_PER_EH_BOHR
        if res.charges is not None:
            self.results["charges"] = np.asarray(res.charges, dtype=float)
        if res.dipole_au is not None:
            self.results["dipole"] = np.asarray(res.dipole_au, dtype=float) * BOHR_ANG
        self.atoms = atoms.copy()
        self.last = res

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        numbers = [int(z) for z in self.atoms.numbers]
        folder = self.directory_path
        if numbers != self.numbers:          # another molecule: its restart/topology files do not apply
            _clean(folder, molecule=True)
            self.numbers = numbers
        if self.should_stop is not None and self.should_stop():
            from .optimize import Cancelled

            raise Cancelled()
        res = run_xtb(folder, self.method, numbers, self.atoms.positions, charge=self.charge,
                      multiplicity=self.multiplicity, solvent=self.solvent, model=self.solvation_model,
                      threads=self.threads, grad=True, should_stop=self.should_stop, program=self.program,
                      accuracy=self.accuracy)
        self.n_runs += 1
        self.store(self.atoms, res)


def _factory(method: str):
    def factory(numbers, charge: int = 0, multiplicity: int = 1, device: str = "cpu", **options) -> XtbCalculator:
        return XtbCalculator(numbers, method, charge, multiplicity, solvent=options.get("solvent", "") or "",
                             threads=options.get("threads", 1) or 1,
                             solvation_model=options.get("solvation_model", "alpb") or "alpb",
                             should_stop=options.get("should_stop"), accuracy=options.get("accuracy", 0.1))

    return factory


# ----------------------------------------------------------------------------- native optimizer
def native_optimize(atoms, fmax: float = 0.03, max_steps: int = 500, should_stop=None, log=None, *,
                    method: str = "xtb-gfn2", charge: int = 0, multiplicity: int = 1, solvent: str = "",
                    threads: int = 1, solvation_model: str = "alpb", **_options):
    """Optimize ``atoms`` in place with xtb's own ANCopt optimizer (``xtb --opt <level> --cycles N``).

    The level comes from ``opt_level(fmax)``; ``max_steps`` becomes ``--cycles``. The final forces are
    read from the gradient xtb writes for the optimized geometry (``--grad``), so the returned fmax is
    exact. ``converged`` needs both xtb's "GEOMETRY OPTIMIZATION CONVERGED" and fmax <= the target
    (xtb 6.7.1 writes ``.xtboptok`` and exits 0 even when it runs out of cycles).
    When ``atoms.calc`` is an ``XtbCalculator`` with the same setup, the run uses its folder (restart
    files) and its ``results`` are set to the optimized geometry (energy, forces, charges, dipole).
    """
    from .optimize import Cancelled, OptResult

    if should_stop is not None and should_stop():
        raise Cancelled()
    numbers = [int(z) for z in atoms.numbers]
    calc = atoms.calc if isinstance(atoms.calc, XtbCalculator) else None
    if calc is not None and not calc.same_setup(method, charge, multiplicity, solvent):
        calc = None
    if len(numbers) == 1:            # nothing to optimize (xtb's ANCopt needs at least two atoms)
        work = calc or XtbCalculator(numbers, method, charge, multiplicity, solvent, threads, solvation_model)
        try:
            e = work.get_potential_energy(atoms)
        finally:
            if work is not calc:
                work.close()
        return OptResult(float(e), True, 0, 0.0)
    program = calc.program if calc is not None else external.require_program("xtb")
    level = opt_level(fmax)
    cmd = command(method, charge=charge, multiplicity=multiplicity, solvent=solvent, model=solvation_model,
                  grad=True, opt_level_name=level, cycles=max_steps, program=program)
    env = external.program_env(threads, via_wsl=program.via_wsl)

    def run_in(folder: Path):
        _clean(folder)
        write_xyz(folder / f"{INPUT}.xyz", numbers, atoms.positions, "QuantaBench")
        run = external.run_program(cmd, folder, threads=threads, env=env, timeout=_timeout(), log_name="xtb.log",
                                   should_stop=should_stop, check=False)
        text = run.text()
        if run.returncode != 0:
            raise _fail(f"{METHODS[method][0]} optimization", run, folder)
        opt_file = folder / "xtbopt.xyz"
        if not opt_file.is_file():
            raise external.ExternalProgramError(f"xtb wrote no xtbopt.xyz\n{run.tail(30)}")
        symbols, pos, comment = read_xyz_frames(opt_file.read_text(encoding="utf-8", errors="replace"))[-1]
        if symbols != _symbols(numbers):
            raise external.ExternalProgramError("xtbopt.xyz has a different atom order than the input")
        res = read_results(folder, numbers, pos, charge, method, text, run.seconds, grad=True)
        engrad = folder / f"{INPUT}.engrad"
        if engrad.is_file():                  # make sure the gradient belongs to the final geometry
            _e, _g, coords = parse_engrad(engrad.read_text(encoding="utf-8", errors="replace"))
            if coords is not None and np.abs(coords * BOHR_ANG - pos).max() > 1e-4:
                res.gradient = None
        if res.gradient is None:
            m = re.search(r"gnorm:\s*(\S+)", comment)
            gnorm = float(m.group(1)) if m else float("nan")
            fmax_final = gnorm * EV_PER_EH_BOHR        # an upper bound
        else:
            fmax_final = float(np.sqrt((res.gradient ** 2).sum(1)).max() * EV_PER_EH_BOHR)
        return pos, res, parse_log(text), fmax_final

    if calc is not None:
        pos, res, info, fmax_final = run_in(calc.directory_path)
    else:
        with external.work_dir(f"xtb_opt_{method}") as d:
            pos, res, info, fmax_final = run_in(d)
    atoms.set_positions(pos)
    if calc is not None:
        calc.n_runs += 1
        # After --opt, xtbout.json holds the dipole of the *first* SCF (the start geometry), while the
        # energy, charges and engrad belong to the final one (CF3COOH: 3.11 D stored vs 3.52 D at the
        # optimum; for ions the error was larger still). One cheap single point at the final geometry
        # makes every result, the dipole included, belong to the geometry the atoms now have.
        calc.reset()
        calc.calculate(atoms)
    converged = bool(info["opt_converged"]) and fmax_final <= fmax
    if log is not None and info["opt_converged"] and not converged:
        log(f"  xtb --opt {level} converged, but max force {fmax_final:.4f} eV/Å > fmax {fmax}")
    return OptResult(res.energy_eh * HARTREE_EV, converged, int(info["opt_steps"] or 0), fmax_final)


def _native(method: str):
    def run(atoms, fmax, max_steps, should_stop=None, log=None, **options):
        return native_optimize(atoms, fmax, max_steps, should_stop, log, method=method, **options)

    return run


# ----------------------------------------------------------------------------- properties
def properties(numbers, positions_A, charge: int = 0, multiplicity: int = 1, *, method: str = "xtb-gfn2",
               solvent: str = "", threads: int = 1, solvation_model: str = "alpb",
               should_stop: Callable[[], bool] | None = None, **_options) -> dict:
    """One xtb single point at this geometry (same level and solvent as the refinement).

    ``<prefix>_energy_ev``, ``_dipole_debye`` (about the centre of mass), ``_gap_ev`` (HOMO-LUMO; not
    for GFN-FF), ``_d4_polarizability_A3`` (GFN2: the D4 dispersion model's molecular α(0), xtb's "Mol. α(0) /au" x 0.1481847; a charge-scaled sum of atomic reference polarizabilities, not the GFN2 electronic response), and with a
    solvent ``_gsolv_kcal`` = E(solution) - E(gas) at this geometry (a second, gas-phase single point;
    this includes the cost of polarizing the solute, like CENSO's xtb Gsolv) and ``_gsolv_term_kcal``
    (the "-> Gsolv" term xtb prints, evaluated with the solution-phase density). Plus "charges".
    """
    label, _family, prefix, _flags = METHODS[method]
    program = external.require_program("xtb")
    kw = {"charge": charge, "multiplicity": multiplicity, "model": solvation_model, "threads": threads,
          "grad": False, "should_stop": should_stop, "program": program}
    notes: list[str] = []
    with external.work_dir(f"xtb_props_{method}") as d:
        res = run_xtb(d, method, numbers, positions_A, solvent=solvent, **kw)
        gas = None
        if solvent:
            try:
                gas = run_xtb(d / "gas", method, numbers, positions_A, solvent="", **kw)
            except external.ExternalProgramError as exc:
                notes.append(f"{label}: gas-phase reference failed ({str(exc).splitlines()[0]}); "
                             f"{prefix}_gsolv_kcal not computed")
    out: dict = {f"{prefix}_energy_ev": res.energy_eh * HARTREE_EV}
    if res.dipole_au is not None:
        out[f"{prefix}_dipole_debye"] = float(np.linalg.norm(res.dipole_au) * AU_DEBYE)
    if res.gap_ev is not None:
        out[f"{prefix}_gap_ev"] = float(res.gap_ev)
    if method == "xtb-gfn2" and res.alpha_au is not None:
        out[f"{prefix}_d4_polarizability_A3"] = float(res.alpha_au * AU_POLARIZABILITY_A3)
    if solvent:
        if gas is not None:
            out[f"{prefix}_gsolv_kcal"] = (res.energy_eh - gas.energy_eh) * KCAL_PER_HARTREE
        if res.gsolv_eh is not None:
            out[f"{prefix}_gsolv_term_kcal"] = res.gsolv_eh * KCAL_PER_HARTREE
    if res.charges is not None:
        out["charges"] = [float(q) for q in res.charges]
    if method == "gfnff":
        notes.append(f"{label}: dipole from the GFN-FF EEQ charges (no electron density)")
    out["_notes"] = notes
    return out


def _props(method: str):
    def run(numbers, positions_A, charge=0, multiplicity=1, **options):
        return properties(numbers, positions_A, charge, multiplicity, method=method, **options)

    return run


# ----------------------------------------------------------------------------- registration
_DESCRIPTIONS = {
    "xtb-gfn2": "GFN2-xTB run by the xtb program (>= 6.5, 6.7.1 recommended): ALPB/GBSA solvents, its own "
                "fast optimizer, dipole, gap and polarizability. CPU.",
    "xtb-gfn1": "GFN1-xTB run by the xtb program: ALPB/GBSA solvents, its own optimizer. CPU.",
    "gfnff": "GFN-FF force field run by the xtb program: generic, fast, for large or unusual molecules; "
             "ALPB/GBSA solvents. CPU.",
}

for _key, (_label, _family, _prefix, _flags) in METHODS.items():
    engines.register(
        engines.EngineInfo(_key, _label, _DESCRIPTIONS[_key], False, frozenset(range(1, 87)), family=_family,
                           solvation=("alpb", "gbsa"), external="xtb", prop_prefix=_prefix),
        available, _factory(_key), _props(_key), _native(_key),
        dipole_origin="com")   # calc.results["dipole"] is about the centre of mass (safe for ions)
