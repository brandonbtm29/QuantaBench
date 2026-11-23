"""OpenMM as a second MD engine (OPENMM-1): the LAMMPS model QuantaBench writes, as an OpenMM ``System``.

``md.liquid`` writes a liquid-slab folder for LAMMPS (``system.data``, ``ff.params``, ``liquid_setup.json``). This
module reads the same files and builds the same model in OpenMM, so one folder runs on either engine and the two
can be compared term by term (``tests/test_md_openmm.py``: LAMMPS ``run 0`` against OpenMM, per style)::

    model = load_model("system.data", "ff.params")          # LAMMPS data + coefficients, numpy only
    phys = physics_from_setup(read_setup(folder), "slab")   # styles, mixing, 1-4 scaling, long range
    system = build_system(model, phys)                      # needs openmm

What is translated (LAMMPS ``real`` units -> OpenMM's nm, kJ/mol, ps):

* ``bond_style harmonic`` E = K (r - r0)² and ``angle_style harmonic`` E = K (θ - θ0)²: OpenMM's harmonic forces
  have a ½, so k = 2K.
* ``dihedral_style opls`` E = ½K1(1 + cos φ) + ½K2(1 - cos 2φ) + ½K3(1 + cos 3φ) + ½K4(1 - cos 4φ): periodic
  torsions n = 1..4 with k = K_n / 2 and phase 0 (odd n) or π (even n). ``fourier`` E = Σ K_i [1 + cos(n_i φ - d_i)]
  and ``harmonic`` E = K [1 + d cos(nφ)] are periodic torsions as they stand. ``improper_style cvff``
  E = K [1 + d cos(nφ)] (d = ±1) is a periodic torsion over the same four atoms (LAMMPS measures φ as for a
  dihedral, i-j-k-l), phase 0 for d = 1 and π for d = -1. Anything else is refused by name.
* ``special_bonds lj 0 0 f coul 0 0 g``: 1-2 and 1-3 pairs excluded, 1-4 pairs scaled; both engines find the 1-4
  pairs from the bonds (a pair that is also 1-2 or 1-3 in a ring is excluded, not scaled).
* Lennard-Jones mixing: Lorentz-Berthelot (``arithmetic``) is OpenMM's ``NonbondedForce`` itself. Geometric
  (OPLS-AA), and any explicit ``pair_coeff i j`` off-diagonal entry, go through a ``CustomNonbondedForce`` with a
  table of A = 4εσ¹² and B = 4εσ⁶ per type pair; the ``NonbondedForce`` then carries the charges and the 1-4
  exceptions (with the mixed 1-4 σ and ε) only.
* Long range: Coulomb is PME (LAMMPS' PPPM; OpenMM's tolerance takes ``kspace_accuracy``). The LJ is either cut
  at rc (``lj="cutoff"``: LAMMPS ``lj/cut/coul/long``; ``tail`` adds the homogeneous tail correction, as the bulk
  NPT's ``pair_modify tail yes``), or LJ-PME (``lj="pme"``: LAMMPS ``lj/long/coul/long`` + ``pppm/disp``, the
  "ewald" variant). OpenMM's LJ-PME uses Lorentz-Berthelot within rc and geometric C6 in reciprocal space, which is
  LAMMPS' ``mix/disp geom`` for the arithmetic force fields; for geometric ones a short-range table adds the
  geometric-minus-LB difference within rc, so both parts are geometric, as in LAMMPS. The dispersion splitting and
  mesh follow ``md.long_range`` (G = 2.77 / rc, mesh ceil(L / h)) unless told otherwise.

Force groups (``GROUPS``) let a caller evaluate parts of the energy: bonds 0, angles 1, dihedrals 2, impropers 3,
the ``NonbondedForce`` 4 (Coulomb, 1-4 pairs, and LB or LJ-PME Lennard-Jones), the table LJ 5; ``NONBONDED`` =
{4, 5}. Pressure and stress are never taken from a single-precision GPU here (``md.openmm_run`` and
``md.openmm_sigma`` say how σ is computed).

The readers are numpy only; ``build_system`` and the platform helpers import ``openmm`` when called, so the rest
of QuantaBench imports this module without OpenMM installed. ``python -m quantabench.md.openmm_engine energies
DATA PARAMS --setup liquid_setup.json --stage bulk|slab`` prints the energy terms as JSON (the parity tests use it
through the OpenMM environment's Python).
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

KCAL = 4.184                        # kJ per kcal
ATM_TO_BAR = 1.01325
BAR_TO_ATM = 1.0 / ATM_TO_BAR
#: g/mol·(Å/fs)² in kcal/mol: 1e-3 kg/mol · 1e10 m²/s² / 4184 J/kcal
KE_UNIT = 1e7 / 4184.0
BOLTZ_KCAL = 0.0019872043           # kcal/mol/K
GROUPS = {"bond": 0, "angle": 1, "dihedral": 2, "improper": 3, "nonbonded": 4, "lj": 5}
NONBONDED = {4, 5}
#: ``add_cutoff_shift``'s group: never integrated, only added to the strained energies of σ samples
SHIFT_GROUP = 6
SUPPORTED_STYLES = {"bond": ("harmonic",), "angle": ("harmonic",),
                    "dihedral": ("opls", "fourier", "harmonic"), "improper": ("cvff",)}


class OpenMMModelError(ValueError):
    """A LAMMPS model this converter cannot translate (a style, a special_bonds setting, a missing coefficient)."""


# ------------------------------------------------------------------------------------------- the LAMMPS files
@dataclass
class LammpsModel:
    """A LAMMPS ``atom_style full`` data file and its coefficients, atoms in id order (0-based indices below)."""

    n_atoms: int
    ids: np.ndarray                       # LAMMPS atom ids in file order of this model (sorted)
    mol: np.ndarray                       # molecule id per atom
    types: np.ndarray                     # atom type per atom
    charges: np.ndarray                   # e
    masses: np.ndarray                    # g/mol per atom
    positions: np.ndarray                 # Å, unwrapped (x + image · L), as in the file
    velocities: np.ndarray | None         # Å/fs, or None
    box_lo: np.ndarray                    # Å
    box_len: np.ndarray                   # Å (orthogonal boxes only)
    bonds: list[tuple[int, int, int]] = field(default_factory=list)                 # (i, j, type)
    angles: list[tuple[int, int, int, int]] = field(default_factory=list)
    dihedrals: list[tuple[int, int, int, int, int]] = field(default_factory=list)
    impropers: list[tuple[int, int, int, int, int]] = field(default_factory=list)
    type_masses: dict[int, float] = field(default_factory=dict)
    pair: dict[int, tuple[float, float]] = field(default_factory=dict)              # type -> (ε kcal/mol, σ Å)
    pair_ij: dict[tuple[int, int], tuple[float, float]] = field(default_factory=dict)
    coeffs: dict[str, dict[int, list[float]]] = field(default_factory=dict)       # bond/angle/dihedral/improper
    step: int | None = None               # "timestep = N" of a write_data file
    title: str = ""

    def molecules(self) -> list[np.ndarray]:
        """Atom indices per molecule, in molecule-id order."""
        order = np.argsort(self.mol, kind="stable")
        ids, starts = np.unique(self.mol[order], return_index=True)
        return [order[s:e] for s, e in zip(starts, [*starts[1:], len(order)])]

    @property
    def total_mass(self) -> float:
        return float(self.masses.sum())


_HEADER_COUNTS = ("atoms", "bonds", "angles", "dihedrals", "impropers")
_SECTIONS = ("Masses", "Atoms", "Velocities", "Bonds", "Angles", "Dihedrals", "Impropers", "Pair Coeffs",
             "PairIJ Coeffs", "Bond Coeffs", "Angle Coeffs", "Dihedral Coeffs", "Improper Coeffs")


def read_data_sections(path: str | Path) -> tuple[str, dict[str, tuple[float, float]], dict[str, list[list[str]]]]:
    """(title line, box {"x": (lo, hi), ...}, sections {name: rows of tokens}) of a LAMMPS data file. Comments
    are dropped; a triclinic tilt line is refused."""
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    title = lines[0] if lines else ""
    box: dict[str, tuple[float, float]] = {}
    out: dict[str, list[list[str]]] = {}
    sec = None
    for raw in lines[1:]:
        s = raw.split("#", 1)[0].strip()
        if not s:
            continue
        if s in _SECTIONS:
            sec = s
            out.setdefault(sec, [])
            continue
        w = s.split()
        if len(w) == 4 and w[2] in ("xlo", "ylo", "zlo"):
            box[w[2][0]] = (float(w[0]), float(w[1]))
            continue
        if len(w) >= 6 and w[3:6] == ["xy", "xz", "yz"]:
            if any(abs(float(t)) > 1e-12 for t in w[:3]):
                raise OpenMMModelError(f"{path}: a triclinic box is not supported")
            continue
        if sec is None:
            continue                       # the header counts
        if w[0][0].isalpha():
            sec = None
            continue
        out[sec].append(w)
    return title, box, out


def read_params(path: str | Path) -> dict:
    """``ff.params`` (``pair_coeff``, ``bond_coeff``, ...): {"pair": {t: (ε, σ)}, "pair_ij": {(i, j): (ε, σ)},
    "bond": {t: [coeffs]}, "angle", "dihedral", "improper"}. Wildcards are refused."""
    p: dict = {"pair": {}, "pair_ij": {}, "bond": {}, "angle": {}, "dihedral": {}, "improper": {}}
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        w = line.split("#", 1)[0].split()
        if not w or not w[0].endswith("_coeff"):
            continue
        kind = w[0][:-len("_coeff")]
        if kind == "pair":
            try:
                i, j, eps, sig = int(w[1]), int(w[2]), float(w[3]), float(w[4])
            except (IndexError, ValueError):
                raise OpenMMModelError(f"{path}: cannot read {line.strip()!r} (wildcards are not supported)") from None
            if i == j:
                p["pair"][i] = (eps, sig)
            else:
                p["pair_ij"][(min(i, j), max(i, j))] = (eps, sig)
        elif kind in p:
            try:
                p[kind][int(w[1])] = [float(v) for v in w[2:]]
            except ValueError:
                raise OpenMMModelError(f"{path}: cannot read {line.strip()!r}") from None
    return p


def load_model(data: str | Path, params: str | Path | None = None) -> LammpsModel:
    """The model of a LAMMPS data file (``write_data`` files with velocities too) and its ``ff.params``
    (coefficients in the data file itself are read as well; ``params`` wins)."""
    title, box, sec = read_data_sections(data)
    if not all(k in box for k in "xyz"):
        raise OpenMMModelError(f"{data}: no box")
    lo = np.array([box[k][0] for k in "xyz"], dtype=float)
    length = np.array([box[k][1] - box[k][0] for k in "xyz"], dtype=float)
    rows = sorted(sec.get("Atoms", []), key=lambda w: int(w[0]))
    if not rows:
        raise OpenMMModelError(f"{data}: no Atoms section")
    if len(rows[0]) < 7:
        raise OpenMMModelError(f"{data}: Atoms must be atom_style full (id mol type q x y z)")
    ids = np.array([int(w[0]) for w in rows])
    index = {int(a): i for i, a in enumerate(ids)}
    mol = np.array([int(w[1]) for w in rows])
    types = np.array([int(w[2]) for w in rows])
    q = np.array([float(w[3]) for w in rows])
    x = np.array([[float(v) for v in w[4:7]] for w in rows])
    img = np.array([[int(v) for v in w[7:10]] if len(w) >= 10 else [0, 0, 0] for w in rows])
    tmass = {int(w[0]): float(w[1]) for w in sec.get("Masses", [])}
    missing = sorted(set(types.tolist()) - set(tmass))
    if missing:
        raise OpenMMModelError(f"{data}: no mass for atom types {missing}")
    v = None
    if sec.get("Velocities"):
        v = np.zeros((len(ids), 3))
        for w in sec["Velocities"]:
            v[index[int(w[0])]] = [float(t) for t in w[1:4]]

    def terms(name: str, k: int) -> list[tuple[int, ...]]:
        return [(*(index[int(a)] for a in w[2:2 + k]), int(w[1])) for w in sec.get(name, [])]

    coeffs: dict = {"bond": {}, "angle": {}, "dihedral": {}, "improper": {}}
    pair: dict[int, tuple[float, float]] = {}
    pair_ij: dict[tuple[int, int], tuple[float, float]] = {}
    for kind, name in (("bond", "Bond Coeffs"), ("angle", "Angle Coeffs"), ("dihedral", "Dihedral Coeffs"),
                       ("improper", "Improper Coeffs")):
        for w in sec.get(name, []):
            coeffs[kind][int(w[0])] = [float(t) for t in w[1:]]
    for w in sec.get("Pair Coeffs", []):
        pair[int(w[0])] = (float(w[1]), float(w[2]))
    if params is not None:
        p = read_params(params)
        pair.update(p["pair"])
        pair_ij.update(p["pair_ij"])
        for kind in coeffs:
            coeffs[kind].update(p[kind])
    m = re.search(r"timestep\s*=\s*(\d+)", title)
    return LammpsModel(n_atoms=len(ids), ids=ids, mol=mol, types=types, charges=q,
                       masses=np.array([tmass[t] for t in types]), positions=x + img * length, velocities=v,
                       box_lo=lo, box_len=length, bonds=terms("Bonds", 2), angles=terms("Angles", 3),
                       dihedrals=terms("Dihedrals", 4), impropers=terms("Impropers", 4), type_masses=tmass,
                       pair=pair, pair_ij=pair_ij, coeffs=coeffs, step=int(m.group(1)) if m else None, title=title)


# ------------------------------------------------------------------------------------------- the physics
@dataclass
class Physics:
    """How the model's interactions are evaluated (one stage of a run: the bulk NPT or the slab)."""

    styles: dict[str, str] = field(default_factory=lambda: {"bond": "harmonic", "angle": "harmonic",
                                                            "dihedral": "opls", "improper": "cvff"})
    mixing: str = "geometric"            # "geometric" | "arithmetic"
    special_lj: tuple[float, float, float] = (0.0, 0.0, 0.5)
    special_coul: tuple[float, float, float] = (0.0, 0.0, 0.5)
    cutoff_A: float = 12.0
    lj: str = "cutoff"                   # "cutoff" (lj/cut/coul/long) | "pme" (LJ-PME: LAMMPS' "ewald" variant)
    tail: bool = False                   # homogeneous LJ tail correction (cutoff only; the bulk NPT)
    ewald_tolerance: float = 1e-5        # Coulomb PME (OpenMM's ewaldErrorTolerance)
    disp_gewald: float | None = None     # LJ-PME splitting G, 1/Å (None: 2.77 / rc)
    disp_spacing: float | None = None    # LJ-PME mesh spacing, Å (None: OpenMM's choice from the tolerance)

    def to_dict(self) -> dict:
        return asdict(self)


def physics_from_setup(meta: Mapping, stage: str) -> Physics:
    """The physics of ``stage`` ("bulk" or "slab") of a liquid-slab folder, from its ``liquid_setup.json``: the
    molecule's styles, mixing and 1-4 scaling, and the config's long-range settings (``md.long_range``)."""
    mol = meta.get("molecule") or {}
    cfg = meta.get("config") or {}
    disp = str(cfg.get("dispersion") or "ewald").lower()
    ewald = disp == "ewald"
    rc = float(cfg.get("cutoff") or (12.0 if ewald else float(disp[6:] or 12.0)))
    acc = cfg.get("kspace_accuracy")
    styles = dict(mol.get("styles") or {})
    return Physics(styles=styles, mixing=str(mol.get("mixing") or "geometric"),
                   special_lj=tuple(mol.get("special_lj") or (0.0, 0.0, 0.5)),
                   special_coul=tuple(mol.get("special_coul") or (0.0, 0.0, 0.5)),
                   cutoff_A=rc, lj="pme" if ewald else "cutoff", tail=(not ewald and stage == "bulk"),
                   ewald_tolerance=float(acc) if acc else (1e-6 if ewald else 1e-5),
                   disp_gewald=(float(cfg["disp_gewald"]) if ewald and cfg.get("disp_gewald") else None),
                   disp_spacing=(float(cfg["disp_spacing"]) if ewald and cfg.get("disp_spacing") else None))


def read_setup(folder: str | Path) -> dict:
    return json.loads((Path(folder) / "liquid_setup.json").read_text(encoding="utf-8"))


def _check(model: LammpsModel, phys: Physics) -> None:
    for kind, terms in (("bond", model.bonds), ("angle", model.angles), ("dihedral", model.dihedrals),
                        ("improper", model.impropers)):
        if not terms:
            continue
        style = phys.styles.get(kind, "")
        if style not in SUPPORTED_STYLES[kind]:
            raise OpenMMModelError(f"{kind}_style {style or '(none given)'}: the OpenMM engine translates "
                                   f"{', '.join(SUPPORTED_STYLES[kind])}")
        missing = sorted({t[-1] for t in terms} - set(model.coeffs.get(kind, {})))
        if missing:
            raise OpenMMModelError(f"no {kind}_coeff for types {missing}")
    if any(phys.special_lj[:2]) or any(phys.special_coul[:2]):
        raise OpenMMModelError("special_bonds with non-zero 1-2 or 1-3 factors is not translated")
    if phys.mixing not in ("geometric", "arithmetic"):
        raise OpenMMModelError(f"pair_modify mix {phys.mixing}: geometric or arithmetic only")
    if phys.lj not in ("cutoff", "pme"):
        raise OpenMMModelError(f"lj {phys.lj!r}: cutoff or pme")
    missing = sorted(set(model.types.tolist()) - set(model.pair))
    if missing:
        raise OpenMMModelError(f"no pair_coeff for atom types {missing}")
    if np.any(model.box_len <= 2.0 * phys.cutoff_A):
        raise OpenMMModelError(f"box {np.round(model.box_len, 2).tolist()} Å is not more than twice the cutoff "
                               f"{phys.cutoff_A:g} Å")


def _mixed(model: LammpsModel, a: int, b: int, mixing: str) -> tuple[float, float]:
    """(ε, σ) of the type pair (a, b): an explicit pair_coeff a b, else the mixing rule."""
    key = (min(a, b), max(a, b))
    if key in model.pair_ij:
        return model.pair_ij[key]
    (ea, sa), (eb, sb) = model.pair[a], model.pair[b]
    if mixing == "arithmetic":
        return math.sqrt(ea * eb), 0.5 * (sa + sb)
    return math.sqrt(ea * eb), math.sqrt(sa * sb)


def _torsions(style: str, coeffs: Sequence[float]) -> list[tuple[int, float, float]]:
    """[(periodicity, phase rad, k kJ/mol)] of one dihedral or improper type."""
    c = list(coeffs)
    out: list[tuple[int, float, float]] = []
    if style == "opls":                             # K1..K4
        for n, k in enumerate(c[:4], start=1):
            if k != 0.0:
                out.append((n, 0.0 if n % 2 else math.pi, 0.5 * k * KCAL))
    elif style == "fourier":                        # m K1 n1 d1 ... (or, ff.params of md.liquid: "1 K n d")
        m = int(round(c[0]))
        for i in range(m):
            k, n, d = c[1 + 3 * i:4 + 3 * i]
            out.append((int(round(n)), math.radians(d), k * KCAL))
    elif style in ("harmonic", "cvff"):             # K d n: E = K [1 + d cos(nφ)], d = ±1
        k, d, n = c[:3]
        if abs(abs(d) - 1.0) > 1e-9:
            raise OpenMMModelError(f"{style} with d = {d:g}: only d = +1 or -1")
        out.append((int(round(n)), 0.0 if d > 0 else math.pi, k * KCAL))
    else:
        raise OpenMMModelError(f"style {style!r}")
    return out


def build_system(model: LammpsModel, phys: Physics, *, box_A: Sequence[float] | None = None):
    """The OpenMM ``System`` of ``model`` under ``phys`` (see the module docstring). ``box_A`` overrides the data
    file's box (lengths, Å)."""
    import openmm as mm

    _check(model, phys)
    box = np.asarray(box_A if box_A is not None else model.box_len, dtype=float)
    s = mm.System()
    for m in model.masses:
        s.addParticle(float(m))
    s.setDefaultPeriodicBoxVectors(mm.Vec3(box[0] / 10, 0, 0), mm.Vec3(0, box[1] / 10, 0), mm.Vec3(0, 0, box[2] / 10))
    if model.bonds:
        f = mm.HarmonicBondForce()
        for i, j, t in model.bonds:
            k, r0 = model.coeffs["bond"][t][:2]
            f.addBond(int(i), int(j), r0 / 10.0, 2.0 * k * KCAL * 100.0)
        f.setForceGroup(GROUPS["bond"])
        f.setUsesPeriodicBoundaryConditions(True)
        s.addForce(f)
    if model.angles:
        f = mm.HarmonicAngleForce()
        for i, j, k_, t in model.angles:
            k, th0 = model.coeffs["angle"][t][:2]
            f.addAngle(int(i), int(j), int(k_), math.radians(th0), 2.0 * k * KCAL)
        f.setForceGroup(GROUPS["angle"])
        f.setUsesPeriodicBoundaryConditions(True)
        s.addForce(f)
    for kind, terms in (("dihedral", model.dihedrals), ("improper", model.impropers)):
        if not terms:
            continue
        f = mm.PeriodicTorsionForce()
        cache: dict[int, list] = {}
        for i, j, k_, l_, t in terms:
            if t not in cache:
                cache[t] = _torsions(phys.styles[kind], model.coeffs[kind][t])
            for n, phase, k in cache[t]:
                f.addTorsion(int(i), int(j), int(k_), int(l_), n, phase, k)
        f.setForceGroup(GROUPS[kind])
        f.setUsesPeriodicBoundaryConditions(True)
        s.addForce(f)
    _nonbonded(s, model, phys, box)
    return s


def _nonbonded(s, model: LammpsModel, phys: Physics, box: np.ndarray) -> None:
    import openmm as mm

    rc = phys.cutoff_A / 10.0
    nb = mm.NonbondedForce()
    nb.setNonbondedMethod(mm.NonbondedForce.LJPME if phys.lj == "pme" else mm.NonbondedForce.PME)
    nb.setCutoffDistance(rc)
    nb.setEwaldErrorTolerance(float(phys.ewald_tolerance))
    nb.setUseSwitchingFunction(False)
    nb.setUseDispersionCorrection(bool(phys.tail) and phys.lj == "cutoff")
    if phys.lj == "pme":
        g = (phys.disp_gewald or 2.77 / phys.cutoff_A) * 10.0          # 1/Å -> 1/nm
        if phys.disp_spacing:
            n = [max(1, math.ceil(float(x) / float(phys.disp_spacing) - 1e-9)) for x in box]
            nb.setLJPMEParameters(g, *n)
        else:
            nb.setLJPMEParameters(g, 0, 0, 0)                          # the mesh from the tolerance
    types = model.types
    eps = np.array([model.pair[t][0] for t in types]) * KCAL
    sig = np.array([model.pair[t][1] for t in types]) / 10.0
    for i in range(model.n_atoms):
        nb.addParticle(float(model.charges[i]), float(sig[i]), float(eps[i]))
    bonds = [(int(i), int(j)) for i, j, _t in model.bonds]
    nb.createExceptionsFromBonds(bonds, float(phys.special_coul[2]), float(phys.special_lj[2]))
    table = phys.mixing == "geometric" or bool(model.pair_ij)
    # 1-4 pairs: σ and ε by the force field's own rule (createExceptionsFromBonds mixed them by LB)
    for k in range(nb.getNumExceptions()):
        i, j, qq, _s14, e14 = nb.getExceptionParameters(k)
        if e14.value_in_unit(mm.unit.kilojoule_per_mole) == 0.0 or not table:
            continue
        e_ij, s_ij = _mixed(model, int(types[i]), int(types[j]), phys.mixing)
        nb.setExceptionParameters(k, i, j, qq, s_ij / 10.0, float(phys.special_lj[2]) * e_ij * KCAL)
    nb.setForceGroup(GROUPS["nonbonded"])
    s.addForce(nb)
    if not table:
        return
    utypes = sorted(set(types.tolist()))
    tindex = {t: n for n, t in enumerate(utypes)}
    nt = len(utypes)
    a_tab = np.zeros((nt, nt))
    b_tab = np.zeros((nt, nt))
    for p, ta in enumerate(utypes):
        for q, tb in enumerate(utypes):
            e_ij, s_ij = _mixed(model, ta, tb, phys.mixing)
            e_ij *= KCAL
            s_ij /= 10.0
            a, b = 4.0 * e_ij * s_ij**12, 4.0 * e_ij * s_ij**6
            if phys.lj == "pme":            # LJ-PME: the NonbondedForce already has LB within rc; add the difference
                (ea, sa), (eb, sb) = model.pair[ta], model.pair[tb]
                e_lb, s_lb = math.sqrt(ea * eb) * KCAL, 0.5 * (sa + sb) / 10.0
                a -= 4.0 * e_lb * s_lb**12
                b -= 4.0 * e_lb * s_lb**6
            a_tab[p, q], b_tab[p, q] = a, b
    cn = mm.CustomNonbondedForce("(acoef(type1, type2)/r6 - bcoef(type1, type2))/r6; r6 = r^6")
    cn.addTabulatedFunction("acoef", mm.Discrete2DFunction(nt, nt, a_tab.ravel(order="F").tolist()))
    cn.addTabulatedFunction("bcoef", mm.Discrete2DFunction(nt, nt, b_tab.ravel(order="F").tolist()))
    cn.addPerParticleParameter("type")
    for t in types:
        cn.addParticle([float(tindex[int(t)])])
    cn.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
    cn.setCutoffDistance(rc)
    cn.setUseSwitchingFunction(False)
    cn.setUseLongRangeCorrection(bool(phys.tail) and phys.lj == "cutoff")
    for k in range(nb.getNumExceptions()):
        i, j, *_ = nb.getExceptionParameters(k)
        cn.addExclusion(i, j)
    if phys.lj == "cutoff":                  # the table is the whole LJ: the NonbondedForce keeps the charges only
        for i in range(model.n_atoms):
            q, _s, _e = nb.getParticleParameters(i)
            nb.setParticleParameters(i, q, 0.1, 0.0)
        nb.setUseDispersionCorrection(False)
    cn.setForceGroup(GROUPS["lj"])
    s.addForce(cn)


def add_cutoff_shift(system, model: LammpsModel, phys: Physics) -> int | None:
    """For a plain LJ cutoff: a force group (``SHIFT_GROUP``) whose energy is −U_ij(rc) for every pair within rc and
    whose force is zero. Added to the energies of a strained box, it cancels the jump of the truncated LJ when a pair
    crosses rc, so a finite difference of the energy gives the mechanical virial, as LAMMPS' pressure does (its
    forces have no impulsive part at rc); without it, σ from the energy picks up that impulsive term (about +1 mN/m
    for hexane at rc = 8 Å, less at larger rc). The integrator must leave the group out
    (``setIntegrationForceGroups``). None for LJ-PME (its real-space jump at rc is under 2 % of the plain one's)."""
    import openmm as mm

    if phys.lj != "cutoff":
        return None
    rc = phys.cutoff_A
    utypes = sorted(set(model.types.tolist()))
    tindex = {t: n for n, t in enumerate(utypes)}
    nt = len(utypes)
    tab = np.zeros((nt, nt))
    for p, ta in enumerate(utypes):
        for q, tb in enumerate(utypes):
            e_ij, s_ij = _mixed(model, ta, tb, phys.mixing)
            sr6 = (s_ij / rc) ** 6
            tab[p, q] = 4.0 * e_ij * (sr6 * sr6 - sr6) * KCAL
    f = mm.CustomNonbondedForce("-ucut(type1, type2)")
    f.addTabulatedFunction("ucut", mm.Discrete2DFunction(nt, nt, tab.ravel(order="F").tolist()))
    f.addPerParticleParameter("type")
    for t in model.types:
        f.addParticle([float(tindex[int(t)])])
    f.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
    f.setCutoffDistance(rc / 10.0)
    f.setUseLongRangeCorrection(False)
    nb = next(x for x in (system.getForce(i) for i in range(system.getNumForces()))
              if isinstance(x, mm.NonbondedForce))
    for k in range(nb.getNumExceptions()):
        i, j, *_ = nb.getExceptionParameters(k)
        f.addExclusion(i, j)
    f.setForceGroup(SHIFT_GROUP)
    system.addForce(f)
    return SHIFT_GROUP


# ------------------------------------------------------------------------------------------- contexts and energies
def platform_names() -> list[str]:
    import openmm as mm

    return [mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())]


def platform_properties(name: str, precision: str | None = None, threads: int | None = None,
                        device: str | None = None) -> dict[str, str]:
    props: dict[str, str] = {}
    if name in ("CUDA", "OpenCL", "HIP") and precision:
        props["Precision"] = precision
    if name in ("CUDA", "OpenCL", "HIP") and device not in (None, ""):
        props["DeviceIndex"] = str(device)
    if name == "CPU" and threads:
        props["Threads"] = str(int(threads))
    return props


def make_context(system, integrator, platform: str = "CPU", *, precision: str | None = None,
                 threads: int | None = None, device: str | None = None):
    import openmm as mm

    plat = mm.Platform.getPlatformByName(platform)
    return mm.Context(system, integrator, plat, platform_properties(platform, precision, threads, device))


def set_state(context, model: LammpsModel, *, velocities: bool = True) -> None:
    """Positions (box origin moved to 0), the box and, when the file has them, the velocities of ``model``."""
    import openmm as mm

    b = model.box_len / 10.0
    context.setPeriodicBoxVectors(mm.Vec3(b[0], 0, 0), mm.Vec3(0, b[1], 0), mm.Vec3(0, 0, b[2]))
    context.setPositions((model.positions - model.box_lo) / 10.0)
    if velocities and model.velocities is not None:
        context.setVelocities(model.velocities * 100.0)          # Å/fs -> nm/ps


def group_energies(context, groups: Mapping[str, int] | None = None) -> dict[str, float]:
    """Potential energy per force group, kcal/mol (only the groups the system has)."""
    groups = dict(GROUPS if groups is None else groups)
    present = {f.getForceGroup() for f in context.getSystem().getForces()}
    out = {}
    for name, g in groups.items():
        if g in present:
            e = context.getState(getEnergy=True, groups={g}).getPotentialEnergy()
            out[name] = e.value_in_unit(e.unit) / KCAL
    return out


def energies(model: LammpsModel, phys: Physics, platform: str = "CPU", *, precision: str | None = None,
             threads: int | None = 2) -> dict[str, float]:
    """kcal/mol: bond, angle, dihedral, improper, nonbonded (Coulomb + LJ + 1-4 pairs + tail), pe, and ke from
    the file's velocities (if any): the terms LAMMPS' ``run 0`` prints, grouped the same way."""
    import openmm as mm

    system = build_system(model, phys)
    ctx = make_context(system, mm.VerletIntegrator(0.001), platform, precision=precision, threads=threads)
    set_state(ctx, model)
    g = group_energies(ctx)
    out = {k: g.get(k, 0.0) for k in ("bond", "angle", "dihedral", "improper")}
    out["nonbonded"] = g.get("nonbonded", 0.0) + g.get("lj", 0.0)
    out["pe"] = sum(out.values())
    if model.velocities is not None:
        out["ke"] = float(0.5 * (model.masses[:, None] * model.velocities**2).sum() * KE_UNIT)
    return out


# ------------------------------------------------------------------------------------------- the platforms here
def describe_platforms(timeout_note: bool = False) -> dict:
    """What this OpenMM sees (run in the OpenMM environment): {"version", "platforms": [{"name", "speed",
    "precisions": [...], "devices": [...], "error"}]}. A GPU platform's precisions are tried by creating a tiny
    context with each ("mixed" and "double" fail on Apple GPUs, which have no FP64)."""
    import openmm as mm

    out: dict = {"version": mm.__version__, "platforms": []}
    s = mm.System()
    s.addParticle(1.0)
    s.addParticle(1.0)
    f = mm.HarmonicBondForce()
    f.addBond(0, 1, 0.1, 1000.0)
    s.addForce(f)
    for i in range(mm.Platform.getNumPlatforms()):
        p = mm.Platform.getPlatform(i)
        rec: dict = {"name": p.getName(), "speed": p.getSpeed(), "precisions": [], "devices": []}
        cands = ("single", "mixed", "double") if p.getName() in ("CUDA", "OpenCL", "HIP") else (None,)
        for prec in cands:
            try:
                ctx = mm.Context(s, mm.VerletIntegrator(0.001), p, platform_properties(p.getName(), prec))
                ctx.setPositions([mm.Vec3(0, 0, 0), mm.Vec3(0.1, 0, 0)])
                ctx.getState(getEnergy=True)
                if prec:
                    rec["precisions"].append(prec)
                else:
                    rec["precisions"] = ["double" if p.getName() == "Reference" else "mixed"]
                for key in ("DeviceName", "OpenCLPlatformName", "CudaDeviceIndex", "OpenCLPlatformIndex"):
                    try:
                        val = p.getPropertyValue(ctx, key)
                    except Exception:  # noqa: BLE001, S112 - not a property of this platform
                        continue
                    if key == "DeviceName" and val and val not in rec["devices"]:
                        rec["devices"].append(val)
                    elif key == "OpenCLPlatformName":
                        rec["opencl_platform"] = val
                del ctx
            except Exception as exc:  # noqa: BLE001 - a platform or precision this machine lacks
                if prec in (None, "single"):
                    rec["error"] = str(exc).splitlines()[0][:200]
        out["platforms"].append(rec)
    return out


def best_platform(desc: Mapping) -> tuple[str, str]:
    """(platform, precision) a run uses by default: a GPU platform that works (CUDA, HIP, then OpenCL) with mixed
    precision where it has it, else single; else the CPU."""
    plats = {p["name"]: p for p in desc.get("platforms", []) if p.get("precisions")}
    for name in ("CUDA", "HIP", "OpenCL"):
        if name in plats:
            precs = plats[name]["precisions"]
            return name, "mixed" if "mixed" in precs else precs[0]
    return ("CPU", "mixed") if "CPU" in plats else ("Reference", "double")


# ------------------------------------------------------------------------------------------- the OpenMM environment
ENV_NAME = "qb-openmm"
SETTING_KEY = "openmm_python"


def _env_python(prefix: Path) -> Path | None:
    for c in (prefix / "python.exe", prefix / "bin" / "python3", prefix / "bin" / "python"):
        if c.is_file():
            return c
    return None


def find_openmm_python() -> tuple[str | None, str]:
    """(a Python that can import openmm, where it was found), without importing openmm here. Order: this machine's
    ``openmm_python`` in the settings (Settings > Engines), ``QUANTABENCH_OPENMM_PYTHON``, this interpreter, then the
    conda/mamba environments: ``qb-openmm`` (``envs/openmm.yml``) first, then any other with an ``openmm`` package."""
    import importlib.util

    from ..chem import external

    with_cfg = ""
    try:
        with_cfg = external._configured_paths().get(SETTING_KEY, ("", ""))[0]
    except Exception:  # noqa: BLE001 - no settings: the other places
        with_cfg = ""
    for path, how in ((with_cfg, "settings"), (os.environ.get("QUANTABENCH_OPENMM_PYTHON", "").strip(),
                                                "QUANTABENCH_OPENMM_PYTHON")):
        if path and Path(os.path.expanduser(path)).is_file():
            return os.path.expanduser(path), how
    if importlib.util.find_spec("openmm") is not None:
        return sys.executable, "this environment"
    prefixes = external.conda_env_prefixes()
    prefixes.sort(key=lambda p: p.name != ENV_NAME)
    for prefix in prefixes:
        if prefix.name == ENV_NAME or any((prefix / "conda-meta").glob("openmm-[0-9]*.json")):
            py = _env_python(prefix)
            if py is not None:
                return str(py), f"conda env {prefix.name}"
    return None, (f"no Python with OpenMM: conda env create -f envs/openmm.yml (environment {ENV_NAME}), or set "
                  "Settings > Engines > OpenMM Python")


def openmm_env(python: str, threads: int | None = None) -> dict[str, str]:
    """The environment of a process running QuantaBench's OpenMM code in ``python``: QuantaBench's source first on
    PYTHONPATH (the OpenMM environment has only openmm and numpy), and the CPU platform's thread count."""
    src = str(Path(__file__).resolve().parents[2])
    env = dict(os.environ)
    if Path(python).resolve() != Path(sys.executable).resolve() or "PYTHONPATH" in env:
        env["PYTHONPATH"] = os.pathsep.join([src, *[p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]])
    if threads:
        env["OPENMM_CPU_THREADS"] = str(int(threads))
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


_PROBES: dict[str, dict] = {}


def probe_platforms(python: str | None = None, timeout: float = 120.0, refresh: bool = False) -> dict:
    """``describe_platforms`` as run by the OpenMM environment's Python (``doctor``, ``machines``): {"python",
    "source", "version", "platforms"} or {"error"}. Probed once per process and Python (``refresh`` again)."""
    how = ""
    if python is None:
        python, how = find_openmm_python()
    if python is None:
        return {"error": how}
    if not refresh and python in _PROBES:
        return dict(_PROBES[python])
    out = _probe(python, how, timeout)
    _PROBES[python] = out                     # a failure too: a broken environment is not probed again and again
    return dict(out)


def _probe(python: str, how: str, timeout: float) -> dict:
    from .. import proctree

    probe = proctree.run_probe([python, "-m", "quantabench.md.openmm_engine", "platforms"], timeout=timeout,
                               env=openmm_env(python))
    lines = [x for x in (probe.stdout or "").splitlines() if x.strip().startswith("{")]
    if probe.returncode != 0 or not lines:
        tail = (probe.text or "").strip().splitlines()[-1:] or ["no output"]
        return {"python": python, "source": how, "error": f"OpenMM probe failed: {tail[0][:200]}"}
    out = json.loads(lines[-1])
    out.update(python=python, source=how)
    return out


def platform_lines(desc: Mapping) -> list[str]:
    """Human lines for ``doctor``: one per platform, with the precisions it allows."""
    if desc.get("error"):
        return [desc["error"]]
    L = [f"OpenMM {desc.get('version')} ({desc.get('python')}, {desc.get('source') or 'found'})"]
    for p in desc.get("platforms", []):
        if not p.get("precisions"):
            L.append(f"  {p['name']}: not usable ({p.get('error', 'no precision worked')})")
            continue
        dev = f" on {', '.join(p['devices'])}" if p.get("devices") else ""
        note = ""
        if p["name"] in ("OpenCL", "CUDA", "HIP") and "mixed" not in p["precisions"]:
            note = " (no mixed or double: this GPU has no FP64, as Apple GPUs)"
        L.append(f"  {p['name']}{dev}: precision {', '.join(p['precisions'])}{note}")
    name, prec = best_platform(desc)
    L.append(f"  MD runs use {name} ({prec}) unless told otherwise"
             + ("; without a GPU platform OpenMM is much slower than LAMMPS here (geometric mixing runs as a custom "
                "force on the CPU): use LAMMPS" if name in ("CPU", "Reference") else ""))
    return L


# ------------------------------------------------------------------------------------------- the MD benchmark
def bench(data: str | Path, params: str | Path, molecule: Mapping, dispersion: str, *, geometry: str = "slab",
          platform: str = "gpu", threads: int = 1, steps: int = 10_000, max_seconds: float = 60.0,
          temperature: float = 298.15, seed: int = 20261120) -> dict:
    """``md.bench``'s OpenMM variant: the benchmark box under ``dispersion`` (``md.long_range``'s default settings,
    the slab's physics for a slab box, the bulk's with its tail for a bulk one), Langevin NVT at 1 fs; run 0's
    potential energy, then up to ``steps`` steps or ``max_seconds`` after 100 steps of warm-up. ``platform`` "gpu": the
    fastest GPU platform at its best precision (``best_platform``)."""
    import time

    import openmm as mm

    from . import long_range as LR

    kind = "ewald" if dispersion == "ewald" else "cutoff"
    rc = 12.0 if kind == "ewald" else float(dispersion[len("cutoff"):])
    s = LR.settings(kind, rc)
    meta = {"molecule": dict(molecule), "config": {"dispersion": dispersion, "cutoff": rc,
                                                   "kspace_accuracy": s["kspace_accuracy"],
                                                   "disp_gewald": s.get("disp_gewald"),
                                                   "disp_spacing": s.get("disp_spacing")}}
    phys = physics_from_setup(meta, "slab" if geometry == "slab" else "bulk")
    model = load_model(data, params)
    if platform.lower() == "gpu":
        name, prec = best_platform(describe_platforms())
        if name in ("CPU", "Reference"):
            raise RuntimeError("OpenMM sees no GPU platform here")
    else:
        name, prec = platform, "mixed"
    system = build_system(model, phys)
    system.addForce(mm.CMMotionRemover(1000))
    integ = mm.LangevinMiddleIntegrator(temperature, 1.0, 0.001)
    integ.setRandomNumberSeed(seed % 2**31)
    ctx = make_context(system, integ, name, precision=prec, threads=threads)
    set_state(ctx, model, velocities=False)
    pe0 = ctx.getState(getEnergy=True).getPotentialEnergy()._value / KCAL
    ctx.setVelocitiesToTemperature(temperature, seed)
    integ.step(100)
    ctx.getState(getEnergy=True)
    t0 = time.perf_counter()
    done = 0
    while done < steps and time.perf_counter() - t0 < max_seconds:
        n = min(500, steps - done)
        integ.step(n)
        ctx.getState(getEnergy=True)
        done += n
    wall = time.perf_counter() - t0
    st = ctx.getState(getEnergy=True)
    temp = 2 * st.getKineticEnergy()._value / ((3 * model.n_atoms - 3) * 0.00831446261815324)
    dev = ""
    try:
        dev = ctx.getPlatform().getPropertyValue(ctx, "DeviceName")
    except Exception:  # noqa: BLE001 - the CPU platform has no device name
        dev = ""
    nsday = done * 1e-6 / (wall / 86400.0) if wall > 0 else 0.0
    out = {"ns_per_day": round(nsday, 4), "steps": done, "seconds": round(wall, 2), "atoms": model.n_atoms,
           "timesteps_per_s": round(done / wall, 3) if wall > 0 else None, "platform": name, "precision": prec,
           "openmm": mm.__version__, "run0": {"pe": pe0}, "final_temp_K": round(temp, 2)}
    if dev:
        out["device"] = str(dev)[:120]
    if not math.isfinite(temp) or temp > 2.0 * temperature:
        out["note"] = f"unstable run: T = {temp:g} K at the end"
    return out


# ------------------------------------------------------------------------------------------- command line
def _main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m quantabench.md.openmm_engine")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("energies", help="energy terms (kcal/mol) of a LAMMPS model in OpenMM, as JSON")
    e.add_argument("data")
    e.add_argument("params")
    e.add_argument("--setup", help="liquid_setup.json (styles, mixing, long range)")
    e.add_argument("--stage", default="slab", choices=["bulk", "slab"])
    e.add_argument("--physics", help="a Physics as JSON (instead of --setup)")
    e.add_argument("--platform", default="CPU")
    e.add_argument("--precision")
    sub.add_parser("platforms", help="the platforms and precisions this OpenMM can use, as JSON")
    b = sub.add_parser("bench", help="md.bench's OpenMM variant on a benchmark box, as JSON")
    b.add_argument("data")
    b.add_argument("params")
    b.add_argument("--molecule", required=True, help="molecule.json (a TypedMolecule)")
    b.add_argument("--dispersion", default="cutoff16")
    b.add_argument("--geometry", default="slab", choices=["slab", "bulk"])
    b.add_argument("--platform", default="gpu")
    b.add_argument("--threads", type=int, default=1)
    b.add_argument("--steps", type=int, default=10_000)
    b.add_argument("--max-seconds", type=float, default=60.0)
    args = ap.parse_args(argv)
    if args.cmd == "platforms":
        print(json.dumps(describe_platforms()))
        return 0
    if args.cmd == "bench":
        mol = json.loads(Path(args.molecule).read_text(encoding="utf-8"))
        print(json.dumps(bench(args.data, args.params, mol, args.dispersion, geometry=args.geometry,
                               platform=args.platform, threads=args.threads, steps=args.steps,
                               max_seconds=args.max_seconds)))
        return 0
    if args.physics:
        phys = Physics(**json.loads(args.physics))
    elif args.setup:
        phys = physics_from_setup(json.loads(Path(args.setup).read_text(encoding="utf-8")), args.stage)
    else:
        phys = Physics()
    model = load_model(args.data, args.params)
    print(json.dumps(energies(model, phys, args.platform, precision=args.precision)))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("OPENMM_CPU_THREADS", "2")
    sys.exit(_main())
