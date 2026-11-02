"""MOPAC engines: the PM7 and PM6-D3H4X semi-empirical (NDDO) methods, run as the external ``mopac`` program.

Each energy/gradient is one MOPAC job (``job.mop`` -> ``job.out`` + ``job.aux``) in a scratch folder from
``external.work_dir``, run with ``external.run_program`` (cancellable, thread caps, works through WSL).
MOPAC 22 and later are open source (no license key, no interactive prompt); older MOPAC2016 builds are
reported as unavailable.

Energy. The ASE energy is the **heat of formation** ΔHf (kcal/mol converted to eV), because MOPAC's
gradients are derivatives of ΔHf (checked by finite differences: water and methanol agree to
0.002 kcal/mol/Å with ``PRECISE``). ΔHf differs from a total energy by per-element reference terms, so
conformer energy differences are the same either way; but do not compare it with other engines' energies.
``PRECISE`` (100x tighter SCF) is used by default: without it the gradients carry ~0.02 kcal/mol/Å of noise,
which matters for finite-difference Hessians.

Open shells use UHF with the spin-state keyword (DOUBLET, TRIPLET, ...): analytic UHF gradients, unlike
MOPAC's default half-electron RHF for odd electron counts. Implicit solvent: COSMO from the dielectric
constant (``EPS=``).

``native_optimize`` runs MOPAC's own optimizer (EF) in one job instead of ASE driving MOPAC step by step.
ASE's ``fmax`` (largest atomic force) is translated to MOPAC's ``GNORM`` (norm of the whole gradient
vector) as ``GNORM = fmax`` in kcal/mol/Å: the norm bounds every atomic force, so a converged MOPAC run
also meets ``fmax`` (it is a little stricter than ASE for larger molecules).

Properties: ΔHf, dipole, HOMO/LUMO/gap, Koopmans ionization energy, atomic charges, the COSMO solvent-
accessible area and volume (MOPAC computes this surface in the gas phase too; good QSAR size descriptors)
and, for closed shells, the static polarizability from MOPAC's TDHF ``POLAR`` (gas phase, like the
tblite polarizability; MOPAC's POLAR does not work with UHF).

Elements: H-La and Lu-Bi (the PM6/PM7 main-group and d-block parameters). MOPAC also accepts Ce-Yb, but
only as Sparkle +3 cores, valid only for Ln(III) complexes, so they are left out.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from . import engines, external

KCAL_PER_EV = 23.060548                 # same constant as conformers.KCAL_PER_EV
DEBYE_E_ANG = 0.20819434                # 1 Debye in e·Å (ase.units.Debye)
MIN_VERSION = (22,)
ELEMENTS = frozenset(range(1, 58)) | frozenset(range(71, 84))
SPIN_KEYWORDS = {1: "SINGLET", 2: "DOUBLET", 3: "TRIPLET", 4: "QUARTET", 5: "QUINTET", 6: "SEXTET",
                 7: "SEPTET", 8: "OCTET", 9: "NONET"}
JOB = "job"


def available() -> tuple[bool, str]:
    info = external.find_program("mopac")
    if info is None:
        return False, f"MOPAC not found: {external.PROGRAMS['mopac'].install_hint}"
    if not info.version:
        return False, (f"could not read the version of {info.path} (MOPAC >= 22 prints it with --version); "
                       "MOPAC2016 and older are not supported: install MOPAC 22+ (open source)")
    if external.version_tuple(info.version) < MIN_VERSION:
        return False, f"MOPAC {info.version} is too old: need MOPAC >= 22 (open source, no license key)"
    return True, ""


# ---------------------------------------------------------------------------- input
def keywords(method: str, charge: int = 0, multiplicity: int = 1, *, solvent: str = "", threads: int = 1,
             task: str = "gradient", gnorm: float | None = None, cycles: int | None = None, polar: bool = False,
             precise: bool = True) -> str:
    """MOPAC keyword line. ``task``: "gradient" (1SCF GRADIENTS), "opt" (GNORM/CYCLES) or "sp" (1SCF)."""
    kw = [method]
    if task == "gradient":
        kw += ["1SCF", "GRADIENTS"]
    elif task == "sp":
        kw.append("1SCF")
    elif task == "opt":
        if gnorm is not None:
            kw.append(f"GNORM={gnorm:.4f}")
            if gnorm < 0.01:
                kw.append("LET")          # MOPAC resets GNORM < 0.01 to 0.01 unless LET is given
        if cycles is not None:
            kw.append(f"CYCLES={max(1, int(cycles))}")
    else:
        raise ValueError(f"unknown MOPAC task {task!r}")
    if polar:   # static alpha only: no 0.25/0.5 eV frequencies, no hyperpolarizabilities (10x faster for PFOA)
        kw.append("POLAR(E=(0.0),BETA=0,GAMMA=0)")
    if precise:
        kw.append("PRECISE")
    kw.append(f"CHARGE={int(charge)}")
    if multiplicity > 1:
        if multiplicity not in SPIN_KEYWORDS:
            raise ValueError(f"MOPAC supports multiplicities up to 9, not {multiplicity}")
        kw += ["UHF", SPIN_KEYWORDS[multiplicity]]
    if solvent:
        from . import solvents

        kw.append(f"EPS={float(solvents.for_engine(solvent, 'cosmo')):.2f}")
    kw += ["AUX(PRECISION=9)", f"THREADS={max(1, int(threads))}"]
    return " ".join(kw)


def write_input(path: Path, numbers, positions_ang, keyword_line: str, title: str = "QuantaBench") -> None:
    """``job.mop``: keywords, title, comment, then Cartesian coordinates with every coordinate flagged (1)
    so that gradients are printed (and optimized) for all of them."""
    from ase.data import chemical_symbols

    lines = [keyword_line, re.sub(r"\s+", " ", title).strip()[:70] or "QuantaBench", "generated by QuantaBench"]
    for z, (x, y, w) in zip(numbers, np.asarray(positions_ang, float).reshape(-1, 3)):
        lines.append(f"{chemical_symbols[int(z)]:<2s} {x:16.10f} 1 {y:16.10f} 1 {w:16.10f} 1")
    Path(path).write_text("\n".join(lines) + "\n", encoding="ascii")


def check_electrons(numbers, charge: int, multiplicity: int) -> None:
    n = int(np.sum(np.asarray(numbers, int))) - int(charge)
    if n < 0 or (n + multiplicity - 1) % 2:
        raise ValueError(f"{n} electrons cannot have multiplicity {multiplicity}")


# ---------------------------------------------------------------------------- output
#: ``KEY=...`` or ``KEY[n]=...``; names may hold units and spaces (``AREA:SQUARE ANGSTROMS``). Data lines
#: of arrays never contain "=", so every match starts a new entry.
_KEY = re.compile(r"^\s*([A-Z][A-Z0-9_.:/()*+\- ]*?)(?:\[(\d+)\])?=(.*)$")


def _value(token: str):
    t = token.strip().strip('"')
    try:
        return float(t.replace("D", "E").replace("d", "e"))
    except ValueError:
        return t


def _scalar(rest: str):
    rest = rest.strip()
    if rest.startswith('"') or len(rest.split()) != 1:
        return rest.strip('"').strip()
    return _value(rest)


def parse_aux(text: str) -> dict[str, list]:
    """Every ``KEY=value`` and ``KEY[n]=`` entry of a MOPAC .aux file, as key -> list of values (one per
    occurrence: optimizations repeat the ``*_UPDATED`` entries every cycle). Keys keep their unit suffix
    (``HEAT_OF_FORMATION:KCAL/MOL``) and drop the array size; arrays become float arrays (or lists of str)."""
    out: dict[str, list] = {}
    key, size, buf = None, 0, []

    def flush():
        nonlocal key, buf
        if key is not None:
            vals = [_value(t) for t in buf]
            out.setdefault(key, []).append(np.array(vals, float) if all(isinstance(v, float) for v in vals)
                                           else vals)
        key, buf = None, []

    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):        # banners and "# Lower half triangle only"
            continue
        m = _KEY.match(line)
        if m:
            flush()
            name, n, rest = m.group(1).strip(), m.group(2), m.group(3)
            if n is None:
                out.setdefault(name, []).append(_scalar(rest))
                continue
            key, size, buf = name, int(n), rest.split()
        elif key is not None:
            buf.extend(s.split())
        if key is not None and len(buf) >= size:
            flush()
    flush()
    return out


def last(aux: dict[str, list], key: str, default=None):
    vals = aux.get(key)
    return vals[-1] if vals else default


def mopac_messages(out_text: str) -> list[str]:
    """Lines of MOPAC's "Error and normal termination messages" box (empty for a clean run)."""
    m = re.search(r"Error and normal termination messages reported in this calculation(.*?)(?:\*{20,}|\Z)",
                  out_text, re.S)
    if not m:
        return []
    lines = [ln.strip().strip("*").strip() for ln in m.group(1).splitlines()]
    return [ln for ln in lines if ln and ln != "JOB ENDED NORMALLY"]


def frontier_orbitals(aux: dict[str, list]) -> tuple[float | None, float | None]:
    """(HOMO, LUMO) in eV from the eigenvalues and occupancies (RHF, or both UHF spin channels)."""
    channels = []
    for prefix in ("", "ALPHA_", "BETA_"):
        eps = last(aux, f"{prefix}EIGENVALUES")
        occ = last(aux, f"{prefix}MOLECULAR_ORBITAL_OCCUPANCIES")
        if isinstance(eps, np.ndarray) and isinstance(occ, np.ndarray) and len(eps) == len(occ):
            channels.append((eps, occ))
    if not channels:
        return None, None
    occupied = [e[o > 0.5] for e, o in channels]
    virtual = [e[o <= 0.5] for e, o in channels]
    homo = max((float(x.max()) for x in occupied if x.size), default=None)
    lumo = min((float(x.min()) for x in virtual if x.size), default=None)
    return homo, lumo


_POLAR = re.compile(r"ISOTROPIC AVERAGE ALPHA\s*=\s*(\S+)\s*A\.U\.\s*=\s*(\S+)\s*ANG\.\*\*3")


def polarizability_from_out(out_text: str) -> float | None:
    """Static isotropic polarizability (Å³) from a POLAR job: the first "ISOTROPIC AVERAGE ALPHA" of the
    "CALCULATION OF STATIC FIELD QUANTITIES" section (MOPAC prints it in a.u. and Å³)."""
    i = out_text.find("STATIC FIELD QUANTITIES")
    m = _POLAR.search(out_text, max(i, 0))
    return float(m.group(2)) if m else None


@dataclass
class MopacJob:
    aux: dict[str, list]
    out: str
    seconds: float

    def array(self, key: str, shape=(-1, 3)) -> np.ndarray:
        v = last(self.aux, key)
        if not isinstance(v, np.ndarray):
            raise external.ExternalProgramError(f"MOPAC output has no {key}")
        return v.reshape(shape)

    @property
    def finished(self) -> bool:
        """False when MOPAC stopped before its final SCF (e.g. an optimization hit CYCLES)."""
        return last(self.aux, "HEAT_OF_FORMATION:KCAL/MOL") is not None

    @property
    def heat_of_formation_kcal(self) -> float:
        return float(last(self.aux, "HEAT_OF_FORMATION:KCAL/MOL"))

    def messages(self) -> list[str]:
        """MOPAC's error/termination messages (``ERROR_MESSAGE`` in the .aux, then the .out summary box)."""
        msgs = [str(m) for m in self.aux.get("ERROR_MESSAGE", [])] + mopac_messages(self.out)
        return list(dict.fromkeys(m.strip() for m in msgs if str(m).strip()))


def run_job(numbers, positions_ang, keyword_line: str, *, threads: int = 1, title: str = "QuantaBench",
            prefix: str = "mopac", should_stop: Callable[[], bool] | None = None,
            log: Callable[[str], None] | None = None, timeout: float | None = None,
            partial_ok: bool = False) -> MopacJob:
    """Write ``job.mop``, run MOPAC on it in a scratch folder and parse ``job.aux``/``job.out``.
    MOPAC exits with code 0 even when it rejects a job, so success means a heat of formation in the .aux;
    otherwise ``ExternalProgramError`` carries MOPAC's own messages. ``partial_ok``: also accept an
    optimization that stopped early but left per-cycle ``*_UPDATED`` results (``MopacJob.finished`` is False)."""
    info = external.require_program("mopac")
    with external.work_dir(prefix) as d:
        write_input(d / f"{JOB}.mop", numbers, positions_ang, keyword_line, title)
        run = external.run_program([*info.command(), f"{JOB}.mop"], d, threads=threads, log_name="mopac.log",
                                   should_stop=should_stop, log=log, timeout=timeout)
        out_file, aux_file = d / f"{JOB}.out", d / f"{JOB}.aux"
        out = out_file.read_text(encoding="utf-8", errors="replace") if out_file.is_file() else ""
        aux = parse_aux(aux_file.read_text(encoding="utf-8", errors="replace")) if aux_file.is_file() else {}
        job = MopacJob(aux, out, run.seconds)
        if not job.finished and not (partial_ok and last(aux, "HEAT_OF_FORM_UPDATED:KCAL/MOL") is not None):
            msgs = job.messages() or [ln for ln in (out or run.text()).splitlines() if ln.strip()][-5:]
            raise external.ExternalProgramError("MOPAC failed: " + "; ".join(msgs))
        return job


def _results(job: MopacJob, n_atoms: int) -> dict:
    """ASE results (eV, eV/Å, e·Å, e) from a MOPAC job."""
    e = job.heat_of_formation_kcal / KCAL_PER_EV
    res = {"energy": e, "free_energy": e}
    grad = last(job.aux, "GRADIENTS:KCAL/MOL/ANGSTROM")
    if isinstance(grad, np.ndarray) and grad.size == 3 * n_atoms:
        res["forces"] = -grad.reshape(-1, 3) / KCAL_PER_EV
    dip = last(job.aux, "DIP_VEC:DEBYE")
    if isinstance(dip, np.ndarray) and dip.size == 3:
        res["dipole"] = dip * DEBYE_E_ANG
    q = last(job.aux, "ATOM_CHARGES")
    if isinstance(q, np.ndarray) and q.size == n_atoms:
        res["charges"] = q.copy()
    return res


# ---------------------------------------------------------------------------- ASE calculator
class MopacCalculator(Calculator):
    """ASE calculator running ``mopac`` once per geometry (1SCF GRADIENTS). Energy = ΔHf in eV."""

    implemented_properties: ClassVar[list[str]] = ["energy", "free_energy", "forces", "dipole", "charges"]

    def __init__(self, method: str = "PM7", charge: int = 0, multiplicity: int = 1, solvent: str = "",
                 threads: int = 1, precise: bool = True, should_stop: Callable[[], bool] | None = None,
                 timeout: float | None = None, **kwargs):
        super().__init__(**kwargs)
        self.method = method
        self.charge = int(charge)
        self.multiplicity = int(multiplicity)
        self.solvent = solvent or ""
        self.threads = max(1, int(threads))
        self.precise = precise
        self.should_stop = should_stop
        self.timeout = timeout
        self.last_job: MopacJob | None = None
        self.n_jobs = 0

    def keyword_line(self, task: str = "gradient", **kw) -> str:
        return keywords(self.method, self.charge, self.multiplicity, solvent=self.solvent, threads=self.threads,
                        task=task, precise=self.precise, **kw)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        check_electrons(self.atoms.numbers, self.charge, self.multiplicity)
        job = run_job(self.atoms.numbers, self.atoms.positions, self.keyword_line(), threads=self.threads,
                      prefix=f"mopac_{self.method}", should_stop=self.should_stop, timeout=self.timeout)
        res = _results(job, len(self.atoms))
        if "forces" not in res:
            raise external.ExternalProgramError("MOPAC printed no gradients")
        self.results.update(res)
        self.last_job = job
        self.n_jobs += 1


# ---------------------------------------------------------------------------- engine hooks
def _factory(method: str):
    def factory(numbers, charge: int = 0, multiplicity: int = 1, device: str = "cpu", **options) -> MopacCalculator:
        return MopacCalculator(method, charge, multiplicity, solvent=options.get("solvent") or "",
                               threads=int(options.get("threads") or 1))
    factory.__name__ = f"mopac_{method.split('-')[0].lower()}_factory"
    return factory


def _native_optimize(method: str):
    def native_optimize(atoms, fmax: float, max_steps: int, should_stop=None, log=None, *, charge: int = 0,
                        multiplicity: int = 1, solvent: str = "", threads: int = 1, **options):
        """MOPAC's own optimizer in one job; updates ``atoms`` in place and returns ``optimize.OptResult``."""
        from .optimize import OptResult

        check_electrons(atoms.numbers, charge, multiplicity)
        gnorm = float(fmax) * KCAL_PER_EV
        kw = keywords(method, charge, multiplicity, solvent=solvent or "", threads=int(threads or 1), task="opt",
                      gnorm=gnorm, cycles=max_steps)
        job = run_job(atoms.numbers, atoms.positions, kw, threads=int(threads or 1), prefix=f"mopac_{method}_opt",
                      should_stop=should_stop, partial_ok=True)
        if job.finished:
            pos = job.array("ATOM_X_OPT:ANGSTROMS")
            res = _results(job, len(atoms))
        else:   # stopped early (CYCLES reached): MOPAC skips its final SCF, so take the last cycle's point
            pos = job.array("ATOM_X_UPDATED:ANGSTROMS")
            e = float(last(job.aux, "HEAT_OF_FORM_UPDATED:KCAL/MOL")) / KCAL_PER_EV
            grad = job.array("GRADIENTS_UPDATED:KCAL/MOL/ANGSTROM")
            res = {"energy": e, "free_energy": e, "forces": -grad / KCAL_PER_EV} if grad.shape == pos.shape else {}
        if pos.shape != atoms.positions.shape:
            raise external.ExternalProgramError("MOPAC returned a geometry with a different number of atoms")
        if "forces" not in res:
            raise external.ExternalProgramError("MOPAC printed no final gradients")
        atoms.positions = pos
        calc = atoms.calc
        if isinstance(calc, MopacCalculator):   # cache the final point: no extra MOPAC run for energy/charges
            calc.atoms = atoms.copy()
            calc.results = dict(res)
            calc.last_job = job
        f = res["forces"]
        fmax_final = float(np.sqrt((f**2).sum(axis=1)).max()) if len(f) else 0.0
        steps = len(job.aux.get("HEAT_OF_FORM_UPDATED:KCAL/MOL", []))
        converged = fmax_final <= fmax * (1 + 1e-6)
        if not converged and log is not None:
            why = "; ".join(job.messages()) or "no message"
            log(f"    MOPAC {method} optimizer stopped at fmax {fmax_final:.3f} eV/Å after {steps} cycles ({why})")
        return OptResult(float(res["energy"]), converged, steps, fmax_final)
    native_optimize.__name__ = f"mopac_{method.split('-')[0].lower()}_optimize"
    return native_optimize


def _properties(method: str, prefix: str, label: str):
    def properties(numbers, positions, charge: int = 0, multiplicity: int = 1, *, solvent: str = "",
                   threads: int = 1, polarizability: bool = True, **options) -> dict:
        check_electrons(numbers, charge, multiplicity)
        solvent = solvent or ""
        threads = int(threads or 1)
        notes: list[str] = []
        polar_here = polarizability and multiplicity == 1 and not solvent
        job = run_job(numbers, positions, keywords(method, charge, multiplicity, solvent=solvent, threads=threads,
                                                   task="sp", polar=polar_here),
                      threads=threads, prefix=f"mopac_{method}_props")
        aux = job.aux
        out: dict = {f"{prefix}_heat_of_formation_kcal": job.heat_of_formation_kcal}
        for key, name in (("DIPOLE:DEBYE", "dipole_debye"), ("IONIZATION_POTENTIAL:EV", "ionization_ev"),
                          ("AREA:SQUARE ANGSTROMS", "cosmo_area_A2"), ("VOLUME:CUBIC ANGSTROMS", "cosmo_volume_A3")):
            v = last(aux, key)
            if isinstance(v, float):
                out[f"{prefix}_{name}"] = v
        homo, lumo = frontier_orbitals(aux)
        if homo is not None:
            out[f"{prefix}_homo_ev"] = homo
        if lumo is not None:
            out[f"{prefix}_lumo_ev"] = lumo
        if homo is not None and lumo is not None:
            out[f"{prefix}_gap_ev"] = lumo - homo
        if polarizability:
            alpha = None
            if multiplicity > 1:
                notes.append(f"{label} polarizability skipped: MOPAC POLAR does not work with UHF (open shell)")
            elif polar_here:
                alpha = polarizability_from_out(job.out)
            else:     # in a solvent: the polarizability is a gas-phase property (as for tblite)
                gas = run_job(numbers, positions, keywords(method, charge, 1, threads=threads, task="sp", polar=True),
                              threads=threads, prefix=f"mopac_{method}_polar")
                alpha = polarizability_from_out(gas.out)
                notes.append(f"{label} polarizability: gas phase (other {prefix} properties: COSMO {solvent})")
            if alpha is not None:
                out[f"{prefix}_polarizability_iso_A3"] = alpha
            elif multiplicity == 1:
                notes.append(f"{label} polarizability: MOPAC POLAR printed no static polarizability")
        q = last(aux, "ATOM_CHARGES")
        if isinstance(q, np.ndarray) and q.size == len(numbers):
            out["charges"] = [float(x) for x in q]
        if notes:
            out["_notes"] = notes
        return out
    properties.__name__ = f"{prefix}_properties"
    return properties


for _key, _label, _method, _prefix, _desc in (
    ("mopac-pm7", "PM7 (MOPAC)", "PM7", "pm7",
     "PM7 semi-empirical (MOPAC program): heat of formation, COSMO solvation, polarizability. CPU."),
    ("mopac-pm6", "PM6-D3H4X (MOPAC)", "PM6-D3H4X", "pm6",
     "PM6 with D3 dispersion and H4X hydrogen/halogen-bond corrections (MOPAC program). CPU."),
):
    engines.register(
        engines.EngineInfo(_key, _label, _desc, False, ELEMENTS, family="semi-empirical", solvation=("cosmo",),
                           external="mopac", prop_prefix=_prefix),
        available, _factory(_method), _properties(_method, _prefix, _label), _native_optimize(_method))
