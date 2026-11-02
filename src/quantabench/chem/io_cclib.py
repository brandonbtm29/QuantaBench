"""Import output files of other programs through cclib (Gaussian, ORCA, Psi4, Q-Chem, GAMESS, NWChem,
xtb, MOPAC, Turbomole, ...).

CONTRACT (implemented by the spectra/interop builder):

    available() -> tuple[bool, str]
    read_output(path) -> ImportedCalculation
        program, version, method/basis when known, charge, multiplicity, numbers, positions_A (last geometry),
        trajectory_A (all geometries, for optimizations), energies_ev (SCF/total per step), final energy_ev,
        charges {scheme: list}, dipole_debye, frequencies_cm1, ir_intensities_km_mol, displacements (n,N,3),
        homo_ev/lumo_ev, warnings
    to_atoms(calc) -> ase.Atoms; to_mol(calc) -> RDKit Mol (bonds perceived with rdDetermineBonds, charge set)
    to_vibrations_dict(calc) -> dict shaped like VibrationalResult.to_dict(include_modes=True) (so the spectra
        panel and export.write_vibrations accept it)
    to_result(calc, path) -> a built MoleculeResult (one conformer) for MainWindow.add_result and the CLI

Units: cclib 1.8 reports energies (scfenergies, mpenergies, ccenergies, moenergies) in eV, coordinates in Å,
dipoles in Debye, frequencies in cm⁻¹ and IR intensities in km/mol. cclib converts Hartree to eV with an older
factor (27.21138505) than QuantaBench (CODATA 2018, ``engines.HARTREE_EV``); energies are rescaled to
QuantaBench's factor so that an imported Psi4 energy equals the one QuantaBench's own QM stage would report
(the difference is 3e-6 Eh for water).

Compatibility shims for cclib 1.8.1 (each one only touches the program it is for):

- Psi4 1.9 conda builds print ``Git: Rev {} zzzzzzz``, which cclib turns into the invalid version
  ``1!1.9.1.dev+-zzzzzzz`` and aborts; the empty ``{}`` is dropped before parsing.
- Psi4 >= 1.7 prints the "Thermochemistry Energy Analysis" block in a layout cclib 1.8.1 asserts against
  (``Vibrational ZPVE``); that header is renamed so cclib skips the block, and the ZPVE is read here.
- MOPAC 22+ prints ``MOPAC v23.x``, which cclib's file-type detection (``MOPAC20..``) does not recognize; the
  MOPAC parser is then used directly. cclib reads frequencies but no IR intensities or normal modes from MOPAC.
- xtb does not print normal modes in its standard output; it writes them to ``g98.out`` (Gaussian format) in
  the same folder, which is read for the displacements when its frequencies match.
- Programs that list the six (five) zero translation/rotation modes (xtb, ORCA) have them removed.
"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .engines import HARTREE_EV

#: Atomic-charge schemes in the order one is picked for the 3D view's charge coloring.
CHARGE_PREFERENCE = ("mbis", "cm5", "hirshfeld", "chelpg", "merz-kollman", "resp", "natural", "lowdin",
                     "mulliken")

_PSI4_BANNER = "An Open-Source Ab Initio Electronic Structure Package"
_PSI4_ZPVE = re.compile(r"^\s*Vibrational ZPVE\s+\S+\s+\[kcal/mol\]\s+\S+\s+\[kJ/mol\]\s+(\S+)\s+\[Eh\]", re.M)
#: Functional names some programs print in upper case, keyed by the upper-case form.
_FUNCTIONAL_CASE = {"WB97X-D": "wB97X-D", "WB97X-D3": "wB97X-D3", "WB97X-D3BJ": "wB97X-D3BJ", "WB97X-V": "wB97X-V",
                    "WB97M-V": "wB97M-V", "WB97M-D3BJ": "wB97M-D3BJ", "WB97X": "wB97X", "WB97": "wB97",
                    "R2SCAN": "r2SCAN", "R2SCAN-3C": "r2SCAN-3c", "TPSSH": "TPSSh", "B97-3C": "B97-3c",
                    "PBEH-3C": "PBEh-3c", "WPBE": "wPBE", "CAM-B3LYP": "CAM-B3LYP"}


def _cclib_hartree_ev() -> float:
    from cclib.parser.utils import convertor

    return float(convertor(1.0, "hartree", "eV"))


def _qb_ev(values, factor: float) -> np.ndarray:
    """cclib eV -> QuantaBench eV (same Hartree value, QuantaBench's Hartree-to-eV factor)."""
    return np.asarray(values, dtype=float) / factor * HARTREE_EV


def available() -> tuple[bool, str]:
    from ..resources import has_package

    return (True, "") if has_package("cclib") else (False, "pip install cclib")


@dataclass
class ImportedCalculation:
    """What QuantaBench takes from another program's output file (energies in eV, lengths in Å)."""

    path: str
    program: str                                   # cclib's package name: "Psi4", "xTB", "MOPAC", "ORCA", ...
    version: str = ""
    method: str = ""                               # "wB97X-D", "HF", "GFN2-xTB", "PM7" ("" when unknown)
    basis: str = ""                                # "def2-SVP" ("" for semi-empirical methods or unknown)
    charge: int = 0
    multiplicity: int = 1
    numbers: list[int] = field(default_factory=list)
    positions_A: np.ndarray | None = None          # (N, 3), last geometry in the file
    trajectory_A: np.ndarray | None = None         # (n_geometries, N, 3), every geometry printed
    energies_ev: list[float] = field(default_factory=list)     # SCF (or total) energy per step
    energy_ev: float | None = None                 # final energy (highest correlated level found)
    energy_kind: str = "total"                     # "total" | "heat of formation" (MOPAC)
    charges: dict[str, list[float]] = field(default_factory=dict)
    dipole_debye: float | None = None
    dipole_vector_debye: list[float] | None = None
    frequencies_cm1: list[float] | None = None     # vibrational modes only, imaginary as negative
    ir_intensities_km_mol: list[float] | None = None
    displacements: np.ndarray | None = None        # (n_modes, N, 3) as printed (not renormalized)
    reduced_masses_amu: list[float] | None = None
    force_constants_mdyn_A: list[float] | None = None
    zpve_ev: float | None = None                   # the program's own zero-point energy, when printed
    homo_ev: float | None = None
    lumo_ev: float | None = None
    optimized: bool | None = None                  # optimization converged (None: not an optimization)
    success: bool | None = None                    # the program reported a normal termination
    warnings: list[str] = field(default_factory=list)
    compat: list[str] = field(default_factory=list)   # shims applied to read the file (see module docs)

    @property
    def n_atoms(self) -> int:
        return len(self.numbers)

    @property
    def symbols(self) -> list[str]:
        from ase.data import chemical_symbols

        return [chemical_symbols[z] for z in self.numbers]

    @property
    def level(self) -> str:
        """``"wB97X-D/def2-SVP"``, ``"GFN2-xTB"``, ``"PM7"`` or ``""``."""
        return f"{self.method}/{self.basis}" if self.method and self.basis else self.method

    @property
    def label(self) -> str:
        """Method label for results: ``"psi4: wB97X-D/def2-SVP"``."""
        prog = self.program.lower() or "imported"
        return f"{prog}: {self.level}" if self.level else prog

    def summary(self) -> list[str]:
        """Human-readable lines for the CLI."""
        lines = [f"{Path(self.path).name}: {self.program} {self.version}".rstrip()
                 + (f", {self.level}" if self.level else ""),
                 f"  {self.n_atoms} atoms, charge {self.charge:+d}, multiplicity {self.multiplicity}, "
                 f"{0 if self.trajectory_A is None else len(self.trajectory_A)} geometries"
                 + ("" if self.optimized is None else f", optimization {'converged' if self.optimized else 'NOT converged'}")]
        if self.energy_ev is not None and self.energy_kind == "total":
            lines.append(f"  final energy: {self.energy_ev:.6f} eV ({self.energy_ev / HARTREE_EV:.8f} Eh)")
        elif self.energy_ev is not None:
            from .vibrations import KCAL_PER_EV

            lines.append(f"  final {self.energy_kind}: {self.energy_ev * KCAL_PER_EV:.4f} kcal/mol")
        if self.homo_ev is not None:
            lines.append(f"  HOMO {self.homo_ev:.3f} eV" + (f", LUMO {self.lumo_ev:.3f} eV, gap "
                                                           f"{self.lumo_ev - self.homo_ev:.3f} eV"
                                                           if self.lumo_ev is not None else ""))
        if self.dipole_debye is not None:
            lines.append(f"  dipole {self.dipole_debye:.3f} D")
        if self.charges:
            lines.append(f"  charges: {', '.join(sorted(self.charges))}")
        if self.frequencies_cm1:
            f = self.frequencies_cm1
            n_imag = sum(1 for x in f if x < 0)
            lines.append(f"  {len(f)} vibrational modes, {n_imag} imaginary, {f[0]:.1f} to {f[-1]:.1f} cm-1"
                         + ("" if self.ir_intensities_km_mol else ", no IR intensities")
                         + ("" if self.displacements is not None else ", no normal modes"))
        lines += [f"  note: {w}" for w in self.warnings]
        return lines


# ----------------------------------------------------------------------------------------- reading
def _head(path: Path, size: int = 65536) -> str:
    with path.open("rb") as fh:
        return fh.read(size).decode("utf-8", errors="replace")


def _psi4_compat(text: str, compat: list[str]) -> str:
    fixed = re.sub(r"(Git: Rev )\{\}\s*", r"\1", text)
    if fixed != text:
        compat.append("Psi4 version banner 'Git: Rev {}' normalized")
    if "Vibrational ZPVE" in fixed and "==> Thermochemistry Energy Analysis <==" in fixed:
        fixed = fixed.replace("==> Thermochemistry Energy Analysis <==",
                              "==> Thermochemistry Energy Analysis (ZPVE layout) <==")
        compat.append("Psi4 >= 1.7 thermochemistry block skipped by cclib (ZPVE read by QuantaBench)")
    return fixed


def _parse(path: Path, compat: list[str]):
    """cclib data for ``path`` with the shims of the module docstring; raises ValueError when unreadable."""
    import cclib

    log = io.StringIO()
    kw = {"loglevel": logging.ERROR, "logstream": log}
    head = _head(path)
    try:
        if _PSI4_BANNER in head and "Psi4" in head:
            text = _psi4_compat(path.read_text(encoding="utf-8", errors="replace"), compat)
            data = cclib.parser.Psi4(io.StringIO(text), **kw).parse()
            return data, text
        quiet = logging.getLogger("cclib")
        level = quiet.level
        quiet.setLevel(logging.ERROR)           # "Unable to determine the type of logfile" is handled below
        try:
            parser = cclib.io.ccopen(str(path), **kw)
        finally:
            quiet.setLevel(level)
        if parser is None and re.search(r"\bMOPAC v\d", head):
            compat.append("MOPAC 22+ banner: cclib's MOPAC parser used directly")
            parser = cclib.parser.MOPAC(str(path), **kw)
        if parser is None:
            raise ValueError(f"{path.name}: cclib does not recognize this file (not an output of a supported "
                             "program, or a version cclib does not know)")
        return parser.parse(), None
    except ValueError:
        raise
    except Exception as exc:
        detail = " ".join(line.strip() for line in log.getvalue().splitlines() if line.strip())[-300:]
        raise ValueError(f"{path.name}: cclib could not parse it ({type(exc).__name__}: {exc})"
                         + (f"; {detail}" if detail else "")) from exc


def _float_list(values) -> list[float]:
    return [float(x) for x in np.asarray(values, dtype=float).ravel()]


def _expected_modes(numbers, positions) -> tuple[int, str]:
    from .vibrations import detect_geometry, isotope_masses

    n = len(numbers)
    geometry = detect_geometry(positions, isotope_masses(numbers))
    if geometry == "monatomic":
        return 0, geometry
    return 3 * n - (5 if geometry == "linear" else 6), geometry


def _vibrations(data, calc: ImportedCalculation, path: Path) -> None:
    freqs = np.asarray(getattr(data, "vibfreqs", []), dtype=float)
    if not len(freqs):
        return
    n_atoms = calc.n_atoms
    keep = np.arange(len(freqs))
    n_vib, _geometry = _expected_modes(calc.numbers, calc.positions_A)
    if len(freqs) > n_vib:
        # xtb and ORCA list the translations and rotations as (near) zero modes: drop the smallest |ν|
        drop = np.argsort(np.abs(freqs), kind="stable")[: len(freqs) - n_vib]
        keep = np.array(sorted(set(range(len(freqs))) - set(drop.tolist())), int)
        calc.compat.append(f"{len(drop)} translation/rotation modes removed")
    calc.frequencies_cm1 = _float_list(freqs[keep])

    def pick(attr: str):
        v = getattr(data, attr, None)
        if v is None:
            return None
        v = np.asarray(v, dtype=float)
        return v[keep] if len(v) == len(freqs) else None

    irs = pick("vibirs")
    calc.ir_intensities_km_mol = None if irs is None else _float_list(irs)
    rm = pick("vibrmasses")
    calc.reduced_masses_amu = None if rm is None else _float_list(rm)
    fc = pick("vibfconsts")
    calc.force_constants_mdyn_A = None if fc is None else _float_list(fc)
    disps = getattr(data, "vibdisps", None)
    if disps is not None:
        d = np.asarray(disps, dtype=float)
        if d.shape == (len(freqs), n_atoms, 3):
            calc.displacements = d[keep]
        else:
            calc.warnings.append(f"normal modes have shape {d.shape}, expected ({len(freqs)}, {n_atoms}, 3): ignored")
    if calc.displacements is None and calc.program.lower() == "xtb":
        _xtb_g98_modes(calc, path)


def _xtb_g98_modes(calc: ImportedCalculation, path: Path) -> None:
    """Normal modes from the ``g98.out`` that xtb --hess/--ohess writes next to its output."""
    g98 = path.parent / "g98.out"
    if not g98.is_file():
        calc.warnings.append("xtb prints no normal modes; g98.out (written by xtb --hess) not found next to the "
                             "output, so modes cannot be animated")
        return
    import cclib

    try:
        g = cclib.parser.Gaussian(str(g98), loglevel=logging.CRITICAL, logstream=io.StringIO()).parse()
        f = np.asarray(g.vibfreqs, dtype=float)
        d = np.asarray(g.vibdisps, dtype=float)
    except Exception as exc:
        calc.warnings.append(f"g98.out next to the xtb output could not be read ({type(exc).__name__})")
        return
    ours = np.asarray(calc.frequencies_cm1, dtype=float)
    if f.shape != ours.shape or d.shape != (len(ours), calc.n_atoms, 3) or np.abs(f - ours).max() > 1.0:
        calc.warnings.append("g98.out next to the xtb output belongs to another calculation (frequencies differ): "
                             "not used")
        return
    calc.displacements = d
    calc.compat.append("normal modes read from xtb's g98.out")
    if calc.reduced_masses_amu is None and getattr(g, "vibrmasses", None) is not None:
        calc.reduced_masses_amu = _float_list(g.vibrmasses)


def _homo_lumo(data, factor: float) -> tuple[float | None, float | None]:
    mo = getattr(data, "moenergies", None)
    homos = getattr(data, "homos", None)
    if mo is None or homos is None or not len(mo):
        return None, None
    homos = np.asarray(homos, int).ravel()[-len(mo):]       # xtb's parser appends one index per SCF cycle
    occ, virt = [], []
    for spin, h in zip(mo, homos):
        e = _qb_ev(spin, factor)
        if 0 <= h < len(e):
            occ.append(e[h])
        if 0 <= h + 1 < len(e):
            virt.append(e[h + 1])
    return (float(max(occ)) if occ else None), (float(min(virt)) if virt else None)


def _method_basis(data, head: str) -> tuple[str, str]:
    md = data.metadata
    methods = [m for m in md.get("methods") or [] if m]
    functional = md.get("functional") or ""
    method = functional if functional and (not methods or methods[-1] in ("DFT", "TD-DFT")) else \
        (methods[-1] if methods else functional)
    if not method and md.get("package") == "MOPAC":
        m = re.search(r"^\s*(\S+) CALCULATION RESULTS", head, re.M)
        method = m.group(1) if m else ""
    method = _FUNCTIONAL_CASE.get(str(method).upper(), str(method))
    basis = str(md.get("basis_set") or "")
    if basis.lower().startswith("def2-"):                   # DEF2-SVP -> def2-SVP
        basis = "def2-" + basis[5:].upper()
    m = re.fullmatch(r"(?i)((?:aug|jun|may|apr|heavy-aug)-)?cc-pv(\w)z(.*)", basis)
    if m:                                                   # CC-PVDZ -> cc-pVDZ, AUG-CC-PVTZ -> aug-cc-pVTZ
        basis = f"{(m.group(1) or '').lower()}cc-pV{m.group(2).upper()}Z{m.group(3)}"
    return method, basis


def _zpve_ev(raw: float | None, freqs, factor: float) -> float | None:
    """cclib's ``zpve`` in eV. cclib documents Hartree, but its xtb parser stores eV; the unit that agrees with
    ½Σν of the frequencies is taken (Hartree when there are no frequencies to compare with)."""
    if raw is None:
        return None
    as_hartree = float(raw) * HARTREE_EV
    if freqs is None or not len(freqs):
        return as_hartree
    from ase import units

    as_ev = float(_qb_ev(float(raw), factor))
    f = np.asarray(freqs, dtype=float)
    harmonic = 0.5 * float(f[f > 0].sum()) * units.invcm
    return min((as_hartree, as_ev), key=lambda v: abs(v - harmonic))


def read_output(path: str | Path) -> ImportedCalculation:
    """Parse an output file with cclib. Raises ValueError when cclib cannot read it or it has no geometry."""
    ok, why = available()
    if not ok:
        raise RuntimeError(f"cclib is not installed ({why})")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    compat: list[str] = []
    data, text = _parse(path, compat)
    if data is None or not hasattr(data, "atomcoords") or not hasattr(data, "atomnos"):
        raise ValueError(f"{path.name}: no geometry found (cclib read {type(data).__name__})")
    md = data.metadata
    program = str(md.get("package") or type(data).__name__)
    version = str(md.get("legacy_package_version") or md.get("package_version") or "")
    head = text[:65536] if text is not None else _head(path)
    method, basis = _method_basis(data, head)
    coords = np.asarray(data.atomcoords, dtype=float).reshape(-1, len(data.atomnos), 3)
    calc = ImportedCalculation(path=str(path), program=program, version=version, method=method, basis=basis,
                               numbers=[int(z) for z in data.atomnos], positions_A=coords[-1].copy(),
                               trajectory_A=coords, compat=compat)
    warnings = calc.warnings

    if hasattr(data, "charge"):
        calc.charge = int(data.charge)
    else:
        warnings.append("total charge not found in the file: 0 assumed")
    if hasattr(data, "mult"):
        calc.multiplicity = int(data.mult)
    else:
        n_el = sum(calc.numbers) - calc.charge
        calc.multiplicity = 1 if n_el % 2 == 0 else 2
        warnings.append(f"multiplicity not found in the file: {calc.multiplicity} assumed from the electron count")

    factor = _cclib_hartree_ev()
    if hasattr(data, "scfenergies"):
        calc.energies_ev = _float_list(_qb_ev(data.scfenergies, factor))
        calc.energy_ev = calc.energies_ev[-1] if calc.energies_ev else None
    for attr in ("mpenergies", "ccenergies"):         # a correlated final energy wins over the SCF one
        v = getattr(data, attr, None)
        if v is not None and np.size(v):
            calc.energy_ev = float(_qb_ev(np.asarray(v, dtype=float).ravel()[-1], factor))
    if program == "MOPAC":
        calc.energy_kind = "heat of formation"
        warnings.append("MOPAC energies are heats of formation, not total energies")

    calc.charges = {str(k): _float_list(v) for k, v in (getattr(data, "atomcharges", None) or {}).items()
                    if np.size(v) == calc.n_atoms}
    moments = getattr(data, "moments", None)
    if moments is not None and len(moments) > 1 and np.size(moments[1]) == 3:
        vec = np.asarray(moments[1], dtype=float)
        calc.dipole_vector_debye = _float_list(vec)
        calc.dipole_debye = float(np.linalg.norm(vec))
    calc.homo_ev, calc.lumo_ev = _homo_lumo(data, factor)
    if hasattr(data, "optdone"):
        done = data.optdone
        calc.optimized = bool(done) if isinstance(done, bool) else bool(len(done))
        if not calc.optimized:
            warnings.append("geometry optimization did not converge: the last geometry is not a stationary point")
    calc.success = md.get("success")
    _vibrations(data, calc, path)
    if getattr(data, "zpve", None) is not None:
        calc.zpve_ev = _zpve_ev(data.zpve, calc.frequencies_cm1, factor)
    elif text is not None:
        m = _PSI4_ZPVE.search(text)
        if m:
            calc.zpve_ev = float(m.group(1)) * HARTREE_EV
    if calc.frequencies_cm1 is not None and calc.optimized is False:
        warnings.append("frequencies of an unconverged geometry are approximate")
    return calc


# ----------------------------------------------------------------------------------------- conversions
def to_atoms(calc: ImportedCalculation):
    from ase import Atoms

    atoms = Atoms(numbers=calc.numbers, positions=np.asarray(calc.positions_A, dtype=float))
    q = preferred_charges(calc)
    if q is not None:
        atoms.set_initial_charges(q)
    atoms.info.update(energy_eV=calc.energy_ev, method=calc.label, total_charge=calc.charge)
    return atoms


def preferred_charges(calc: ImportedCalculation) -> list[float] | None:
    for scheme in CHARGE_PREFERENCE:
        if scheme in calc.charges:
            return list(calc.charges[scheme])
    return next(iter(calc.charges.values()), None) if calc.charges else None


def to_mol(calc: ImportedCalculation):
    """RDKit molecule (atom order kept, one conformer) with bonds and formal charges from rdDetermineBonds;
    when bond orders cannot be assigned for the total charge, connectivity only (single bonds)."""
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds

    xyz = f"{calc.n_atoms}\n{Path(calc.path).stem}\n" + "\n".join(
        f"{s} {x:.8f} {y:.8f} {z:.8f}" for s, (x, y, z) in zip(calc.symbols, calc.positions_A))
    raw = Chem.MolFromXYZBlock(xyz)
    if raw is None:
        raise ValueError("RDKit could not read the geometry")
    mol = Chem.Mol(raw)
    try:
        rdDetermineBonds.DetermineBonds(mol, charge=calc.charge)
        mol.SetProp("qb_bonds", "rdDetermineBonds")
    except Exception as exc:
        mol = Chem.Mol(raw)
        rdDetermineBonds.DetermineConnectivity(mol)
        mol.UpdatePropertyCache(strict=False)
        mol.SetProp("qb_bonds", "connectivity only")
        calc.warnings.append(f"bond orders could not be assigned for charge {calc.charge:+d} "
                             f"({type(exc).__name__}): connectivity only")
    return mol


def _unit_modes(d: np.ndarray) -> np.ndarray:
    norm = np.sqrt((d.reshape(len(d), -1) ** 2).sum(1))
    norm[norm == 0] = 1.0
    return d / norm[:, None, None]


def to_vibrations_dict(calc: ImportedCalculation, temperature: float = 298.15, pressure: float = 101325.0,
                       symmetry_number: int = 0, qrrho: bool = True, imag_tol_cm1: float = 20.0) -> dict | None:
    """``VibrationalResult.to_dict(include_modes=True)``-shaped dict (None without frequencies).

    Displacements are renormalized to unit Cartesian norm per mode (QuantaBench's convention). Missing reduced
    masses and force constants are computed from the modes (μ = Σ mᵢ|dᵢ|², k = μω²), NaN without modes.
    Thermochemistry (QuantaBench's ideal-gas RRHO/qRRHO, ``vibrations.thermochemistry``) is computed from the
    imported frequencies when the file has the full set of 3N-6 (3N-5) modes; otherwise ``thermo`` is None.
    """
    from . import vibrations as vib

    if not calc.frequencies_cm1:
        return None
    freqs = np.asarray(calc.frequencies_cm1, dtype=float)
    pos = np.asarray(calc.positions_A, dtype=float)
    masses = vib.isotope_masses(calc.numbers)
    n_expected, geometry = _expected_modes(calc.numbers, pos)
    warnings = [w for w in calc.warnings if "frequenc" in w or "mode" in w]
    disp = None if calc.displacements is None else _unit_modes(np.asarray(calc.displacements, dtype=float))
    nan = [float("nan")] * len(freqs)
    if calc.reduced_masses_amu is not None:
        reduced = list(calc.reduced_masses_amu)
    elif disp is not None:
        reduced = _float_list((masses[None, :, None] * disp**2).sum((1, 2)))
    else:
        reduced = nan
    if calc.force_constants_mdyn_A is not None and any(abs(k) > 0 for k in calc.force_constants_mdyn_A):
        fconst = list(calc.force_constants_mdyn_A)
    else:   # k = μ ω² (xtb's g98.out prints zeros)
        lam = np.sign(freqs) * (freqs / vib._EIG_TO_CM1) ** 2
        fconst = _float_list(lam * np.asarray(reduced, dtype=float) * vib._EV_A2_TO_MDYN_A)
    n_imag = int((freqs < -imag_tol_cm1).sum())
    if n_imag:
        warnings.append(f"{n_imag} imaginary frequenc{'y' if n_imag == 1 else 'ies'} (lowest {freqs.min():.1f} cm-1): "
                        "not a minimum; imaginary modes are excluded from the thermochemistry")
    thermo = None
    if len(freqs) == n_expected and calc.energy_ev is not None:
        from ase import Atoms

        atoms = Atoms(numbers=calc.numbers, positions=pos)
        sigma = symmetry_number or vib.rotational_symmetry_number(calc.numbers, pos, masses)
        use = np.abs(freqs[freqs >= -imag_tol_cm1])
        use = use[use > 1e-3]
        t = vib.thermochemistry(atoms, use, calc.energy_ev, temperature, pressure, sigma, calc.multiplicity,
                                geometry, masses, qrrho)
        thermo = t.to_dict()
        if calc.energy_kind != "total":
            warnings.append(f"thermochemistry built on the {calc.energy_kind}: only the corrections "
                            "(H - E, G - E, ZPE, S) are meaningful")
    elif len(freqs) != n_expected:
        warnings.append(f"{len(freqs)} modes in the file, {n_expected} expected: no thermochemistry")
    ints = calc.ir_intensities_km_mol
    sf = vib.SCALE_FACTORS.get(calc.level.lower())
    d = {
        "method": calc.label, "hessian_method": "imported", "charge": calc.charge,
        "multiplicity": calc.multiplicity, "device": "", "n_atoms": calc.n_atoms, "geometry": geometry,
        "frequencies_cm1": _float_list(freqs), "reduced_masses_amu": [float(x) for x in reduced],
        "force_constants_mdyn_A": [float(x) for x in fconst],
        "ir_intensities_km_mol": None if ints is None else [float(x) for x in ints],
        "ir_source": f"{calc.program} output" if ints is not None else "", "n_imaginary": n_imag,
        "thermo": thermo, "fd_step_A": None, "max_force_ev_A": 0.0,
        "scale_factor": sf.value if sf else 1.0, "scale_factor_source": sf.source if sf else "",
        "imag_tol_cm1": imag_tol_cm1, "warnings": warnings, "timings": {},
        "n_modes": len(freqs), "lowest_freq_cm1": float(freqs[0]) if len(freqs) else None,
        "symbols": calc.symbols, "positions_A": np.round(pos, 6).tolist(),
        "source_file": str(calc.path), "program": calc.program,
    }
    if calc.zpve_ev is not None:
        d["program_zpve_kcal"] = calc.zpve_ev * vib.KCAL_PER_EV
    if disp is not None:
        d["displacements"] = np.round(disp, 5).tolist()
    return d


def to_result(calc: ImportedCalculation, path: str | Path | None = None):
    """A built ``MoleculeResult``: status "ok", one conformer (the last geometry), the file's method as
    ``result.method`` ("psi4: wB97X-D/def2-SVP"), charges, dipole and the vibrations when present."""
    from rdkit import Chem

    from . import descriptors as desc
    from .inputs import MoleculeInput, safe_id
    from .pipeline import ConformerResult, MoleculeResult

    path = Path(path or calc.path)
    mol = to_mol(calc)
    try:
        smiles = Chem.MolToSmiles(Chem.RemoveHs(mol, sanitize=False))
    except Exception:
        smiles = ""
    inp = MoleculeInput(id=safe_id(path.stem, "imported"), smiles=smiles, name=path.name,
                        source=f"imported ({calc.program}): {path}", mol=Chem.Mol(mol), charge=calc.charge,
                        multiplicity=calc.multiplicity, meta={"program": calc.program, "version": calc.version})
    charges = preferred_charges(calc)
    conf = ConformerResult(rank=0, conf_id=mol.GetConformer().GetId(), energy_ev=float(calc.energy_ev or 0.0),
                           weight=1.0, converged=calc.optimized is not False,
                           steps=0 if calc.trajectory_A is None else len(calc.trajectory_A),
                           charges=charges, dipole_debye=calc.dipole_debye)
    d: dict = {}
    try:
        mol_sanitized = Chem.Mol(mol)
        Chem.SanitizeMol(mol_sanitized)
        d.update(desc.graph_descriptors(mol_sanitized))
    except Exception:
        from rdkit.Chem import rdMolDescriptors

        d["formula"] = rdMolDescriptors.CalcMolFormula(mol)
    if calc.energy_ev is not None:
        d["energy_eV"] = calc.energy_ev
    if calc.homo_ev is not None:
        d["homo_ev"] = calc.homo_ev
    if calc.lumo_ev is not None:
        d["lumo_ev"] = calc.lumo_ev
        if calc.homo_ev is not None:
            d["gap_ev"] = calc.lumo_ev - calc.homo_ev
    if calc.dipole_debye is not None:
        d["dipole_debye"] = calc.dipole_debye
    vibs = to_vibrations_dict(calc)
    if vibs and vibs.get("thermo"):
        t = vibs["thermo"]
        d.update(vib_n_imaginary=vibs["n_imaginary"], vib_lowest_freq_cm1=vibs["lowest_freq_cm1"],
                 vib_zpe_kcal=t["zpe_kcal"], vib_h_corr_kcal=t["h_corr_kcal"], vib_g_corr_kcal=t["g_corr_kcal"],
                 vib_entropy_cal_molK=t["entropy_cal_molK"])
    notes = [f"imported from {path.name}" + (f" ({calc.program} {calc.version})" if calc.version else "")]
    notes += calc.warnings
    return MoleculeResult(input=inp, status="ok", smiles=smiles, charge=calc.charge, multiplicity=calc.multiplicity,
                          method=calc.label, device="", forcefield="", n_embedded=0, conformers=[conf], mol=mol,
                          notes=notes, descriptors=d, vibrations=vibs, conformer_search="imported")
