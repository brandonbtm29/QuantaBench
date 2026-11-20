"""Force fields for liquid MD (MD-2): a SMILES becomes a typed molecule with charges, Lennard-Jones types and
bonded terms in LAMMPS ``real`` units, with the source of every parameter.

Backends (``BACKENDS``, chosen with ``type_molecule(..., ff=)``):

``oplsaa``
    OPLS-AA (Jorgensen et al. 1996) with Watkins & Jorgensen's 2001 perfluoroalkane types, typed in process by
    ``md.oplsaa`` (alkanes, perfluoroalkanes, (fluoro)benzenes; everything else is refused). Geometric mixing,
    1-4 scaling 0.5/0.5. Needs nothing beyond the core install.
``openff``
    OpenFF Sage (``openff-2.3.0.offxml`` by default) with AM1-BCC charges from AmberTools' sqm, exported by
    OpenFF Interchange's LAMMPS writer. Lorentz-Berthelot mixing, 1-4 scaling 0.5 (LJ) / 0.8333 (Coulomb).
``gaff2``
    GAFF2 with AM1-BCC charges: antechamber, parmchk2 and tleap from AmberTools, read back with ParmEd.
    Lorentz-Berthelot mixing, 1-4 scaling 0.5 (LJ, scnb 2.0) / 0.8333 (Coulomb, scee 1.2).

The last two are the cross-check route and cover any organic molecule their force field covers. OpenFF and
AmberTools are conda-forge packages (no PyPI wheels), so they run in their own environment through
``md/ff_driver.py``, the way Psi4 does (``chem.qm``): QuantaBench finds that environment's Python next to its
``antechamber`` (``chem.external``: settings key ``antechamber``, ``QUANTABENCH_ANTECHAMBER``, PATH, conda envs) and
hands the driver the molecule as a MOL block, so the atom order is QuantaBench's. Install::

    conda create -n qbmd -c conda-forge python=3.12 ambertools packmol openff-toolkit-base \\
        openff-interchange-base openff-forcefields

Hook for refits: ``TypedMolecule.apply_overrides`` replaces Lennard-Jones parameters by atom type (for example
Träg & Zahn's GAFF2 fluorocarbon refit, J. Mol. Model. 2019, 25, 39, once its numbers are sourced) and records the
source; the setup JSON then lists the overridden types.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from rdkit import Chem

from . import oplsaa as O

AVOGADRO = 6.02214076e23
DRIVER = Path(__file__).with_name("ff_driver.py")
DEFAULT_OPENFF = "openff-2.3.0.offxml"


class ForceFieldError(ValueError):
    """A force field cannot type this molecule, or its backend is not installed."""


@dataclass
class LJType:
    name: str
    element: str
    mass: float          # g/mol
    epsilon: float       # kcal/mol
    sigma: float         # Å
    source: str


@dataclass
class TermType:
    """One bonded type: LAMMPS style and its coefficients in ``real`` units, as ``*_coeff`` takes them."""

    style: str
    coeffs: list[float]
    source: str


@dataclass
class TypedMolecule:
    name: str
    smiles: str
    forcefield: str                       # backend key: "oplsaa" | "openff" | "gaff2"
    label: str                            # what the force field is, in words
    elements: list[str]
    positions: np.ndarray                 # (n, 3) Å, one conformer
    atom_types: list[str]                 # per atom: key of lj_types
    charges: np.ndarray                   # e
    lj_types: dict[str, LJType]
    bonds: list[tuple[int, int, str]] = field(default_factory=list)
    angles: list[tuple[int, int, int, str]] = field(default_factory=list)
    dihedrals: list[tuple[int, int, int, int, str]] = field(default_factory=list)
    impropers: list[tuple[int, int, int, int, str]] = field(default_factory=list)
    bond_types: dict[str, TermType] = field(default_factory=dict)
    angle_types: dict[str, TermType] = field(default_factory=dict)
    dihedral_types: dict[str, TermType] = field(default_factory=dict)
    improper_types: dict[str, TermType] = field(default_factory=dict)
    styles: dict[str, str] = field(default_factory=dict)   # bond / angle / dihedral / improper
    mixing: str = "geometric"
    special_lj: tuple[float, float, float] = (0.0, 0.0, 0.5)
    special_coul: tuple[float, float, float] = (0.0, 0.0, 0.5)
    formal_charge: int = 0
    notes: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    versions: dict[str, str] = field(default_factory=dict)
    overrides: dict[str, dict] = field(default_factory=dict)

    @property
    def n_atoms(self) -> int:
        return len(self.elements)

    @property
    def mass(self) -> float:
        return float(sum(self.lj_types[t].mass for t in self.atom_types))

    @property
    def net_charge(self) -> float:
        return float(np.sum(self.charges))

    def check(self) -> None:
        """Consistency checks a writer relies on; raises ``ForceFieldError``."""
        n = self.n_atoms
        if self.positions.shape != (n, 3) or len(self.atom_types) != n or len(self.charges) != n:
            raise ForceFieldError("atoms, positions, types and charges disagree in length")
        if abs(self.net_charge - self.formal_charge) > 1e-3:
            raise ForceFieldError(f"charges sum to {self.net_charge:+.4f} e, the formal charge is {self.formal_charge:+d}")
        for terms, types, k in ((self.bonds, self.bond_types, 2), (self.angles, self.angle_types, 3),
                                (self.dihedrals, self.dihedral_types, 4), (self.impropers, self.improper_types, 4)):
            for t in terms:
                if t[k] not in types or any(not 0 <= i < n for i in t[:k]):
                    raise ForceFieldError(f"bad term {t}")
        for t in self.atom_types:
            if t not in self.lj_types:
                raise ForceFieldError(f"atom type {t} has no Lennard-Jones parameters")

    def apply_overrides(self, overrides: dict) -> None:
        """Replace Lennard-Jones parameters by type name: ``{"lj": {"f": {"epsilon": .., "sigma": ..}},
        "source": "who and where"}`` (kcal/mol, Å). An unknown type name is an error, not ignored."""
        src = str(overrides.get("source") or "user override")
        for name, p in (overrides.get("lj") or {}).items():
            if name not in self.lj_types:
                raise ForceFieldError(f"override for unknown atom type {name!r} (types: {', '.join(self.lj_types)})")
            old = self.lj_types[name]
            self.lj_types[name] = LJType(name, old.element, old.mass, float(p.get("epsilon", old.epsilon)),
                                         float(p.get("sigma", old.sigma)), f"{src} (was ε {old.epsilon:.5g}, "
                                         f"σ {old.sigma:.5g}: {old.source})")
            self.overrides[name] = {"epsilon": self.lj_types[name].epsilon, "sigma": self.lj_types[name].sigma,
                                    "source": src}

    def to_dict(self) -> dict:
        d = asdict(self)
        d["positions"] = np.asarray(self.positions).round(6).tolist()
        d["charges"] = [float(x) for x in self.charges]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> TypedMolecule:
        d = dict(d)
        d["positions"] = np.asarray(d["positions"], dtype=float)
        d["charges"] = np.asarray(d["charges"], dtype=float)
        d["lj_types"] = {k: LJType(**v) for k, v in d["lj_types"].items()}
        for key in ("bond_types", "angle_types", "dihedral_types", "improper_types"):
            d[key] = {k: TermType(**v) for k, v in d.get(key, {}).items()}
        for key, k in (("bonds", 2), ("angles", 3), ("dihedrals", 4), ("impropers", 4)):
            d[key] = [tuple([*map(int, t[:k]), str(t[k])]) for t in d.get(key, [])]
        d["special_lj"] = tuple(d.get("special_lj", (0, 0, 0.5)))
        d["special_coul"] = tuple(d.get("special_coul", (0, 0, 0.5)))
        return cls(**d)


# ------------------------------------------------------------------------------------------- the molecule
def prepare_molecule(smiles: str, seed: int = 42) -> Chem.Mol:
    """The molecule of ``smiles`` with explicit hydrogens and its lowest MMFF94s conformer (QuantaBench's own
    conformer search, ``chem.conformers``: ETKDGv3 + MMFF94s, up to 30 conformers). No protonation step: a
    pure liquid is the neutral molecule as written."""
    from ..chem import conformers

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ForceFieldError(f"cannot parse SMILES {smiles!r}")
    if len(Chem.GetMolFrags(mol)) != 1:
        raise ForceFieldError("one molecule per liquid: the SMILES has several fragments")
    mol = Chem.AddHs(mol)
    cs = conformers.generate(mol, max_conformers=30, seed=seed, timeout_s=0)
    best = cs.mol.GetConformer(cs.conf_ids[0])
    out = Chem.Mol(cs.mol)
    out.RemoveAllConformers()
    out.AddConformer(Chem.Conformer(best), assignId=True)
    return out


def _topology(mol: Chem.Mol) -> tuple[list[tuple[int, int]], list[tuple[int, int, int]], list[tuple[int, int, int, int]]]:
    bonds = sorted((min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()), max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
                   for b in mol.GetBonds())
    nb = {a.GetIdx(): sorted(n.GetIdx() for n in a.GetNeighbors()) for a in mol.GetAtoms()}
    angles = [(i, j, k) for j in nb for ii, i in enumerate(nb[j]) for k in nb[j][ii + 1:]]
    dihedrals = []
    for j, k in bonds:
        for i in nb[j]:
            if i == k:
                continue
            for m in nb[k]:
                if m in (j, i):
                    continue
                dihedrals.append((i, j, k, m))
    return bonds, angles, dihedrals


# ----------------------------------------------------------------------------------------------- OPLS-AA
def _type_oplsaa(mol: Chem.Mol, name: str, smiles: str) -> TypedMolecule:
    try:
        types = O.assign_types(mol)
    except O.OplsTypingError as exc:
        raise ForceFieldError(str(exc)) from exc
    bonds, angles, dihedrals = _topology(mol)
    lj = {t: LJType(t, O.ATOM_TYPES[t].element, O.ATOM_TYPES[t].mass, O.ATOM_TYPES[t].epsilon_kcal,
                    O.ATOM_TYPES[t].sigma_A, O.ATOM_TYPES[t].source) for t in dict.fromkeys(types)}
    tm = TypedMolecule(
        name=name, smiles=smiles, forcefield="oplsaa",
        label="OPLS-AA (Jorgensen et al. 1996) with the Watkins-Jorgensen 2001 perfluoroalkane types",
        elements=[a.GetSymbol() for a in mol.GetAtoms()], positions=mol.GetConformer().GetPositions(),
        atom_types=types, charges=np.array([O.ATOM_TYPES[t].charge for t in types]), lj_types=lj,
        styles={"bond": "harmonic", "angle": "harmonic", "dihedral": "opls", "improper": "cvff"},
        mixing=O.MIXING, special_lj=O.SPECIAL_LJ, special_coul=O.SPECIAL_COUL,
        formal_charge=Chem.GetFormalCharge(mol),
        references=[O.REFERENCES[k] for k in ("jorgensen1996", "watkins2001", "jorgensen2004", "gromacs", "foyer")],
        versions={"oplsaa_table": f"GROMACS {O.GROMACS_COMMIT[:10]}, foyer {O.FOYER_COMMIT[:10]}"},
    )
    for i, j in bonds:
        key = O.bond_key(types[i], types[j])
        label = "-".join(key)
        if label not in tm.bond_types:
            b0, kb, line, comment = O.BONDS[key]
            tm.bond_types[label] = TermType("harmonic", list(O.lammps_bond(key)),
                                            f"GROMACS oplsaa.ff/ffbonded.itp line {line} ({comment}) = foyer oplsaa.xml")
        tm.bonds.append((i, j, label))
    for i, j, k in angles:
        key = O.angle_key(types[i], types[j], types[k])
        label = "-".join(key)
        if label not in tm.angle_types:
            line, comment = O.ANGLES[key][2:]
            tm.angle_types[label] = TermType("harmonic", list(O.lammps_angle(key)),
                                             f"GROMACS oplsaa.ff/ffbonded.itp line {line} ({comment}) = foyer oplsaa.xml")
        tm.angles.append((i, j, k, label))
    for i, j, k, m in dihedrals:
        label, v, source = O.dihedral_parameters(types[i], types[j], types[k], types[m])
        if label not in tm.dihedral_types:
            tm.dihedral_types[label] = TermType("opls", list(v), source)
        tm.dihedrals.append((i, j, k, m, label))
    for a in mol.GetAtoms():
        if types[a.GetIdx()] in ("opls_145", "opls_718"):
            n1, n2, n3 = sorted(n.GetIdx() for n in a.GetNeighbors())
            if "Z-CA-X-Y" not in tm.improper_types:
                tm.improper_types["Z-CA-X-Y"] = TermType(
                    "cvff", list(O.lammps_improper_ca()),
                    f"GROMACS oplsaa.ff/ffbonded.itp line {O.IMPROPER_CA_LINE} (improper_Z_CA_X_Y); not in foyer's XML")
            tm.impropers.append((n1, n2, a.GetIdx(), n3, "Z-CA-X-Y"))
    if any(t in O.PERFLUORO_CARBONS for t in types):
        tm.notes.append("perfluoroalkyl C-C-C-C torsions use Watkins & Jorgensen's own torsion (GROMACS "
                        "dih_perfluoroalkane_CT_CT_CT_CT), not the hydrocarbon one a bond-type lookup would give")
    if mol.GetRingInfo().NumRings() and any(t in O.PERFLUORO_CARBONS for t in types):
        tm.notes.append("a ring of perfluoro carbons: Watkins & Jorgensen fitted linear chains")
    return tm


# --------------------------------------------------------------------------------------- external backends
def driver_python() -> str | None:
    """The Python of the environment with AmberTools (and OpenFF), found next to its antechamber."""
    from ..chem import external

    return external.find_python_with("antechamber")


def _run_driver(task: dict, timeout: float = 1800.0) -> dict:
    py = driver_python()
    if py is None:
        raise ForceFieldError("AmberTools not found: the openff and gaff2 force fields run in a conda environment "
                              "with ambertools (and openff-toolkit-base, openff-interchange-base, openff-forcefields): "
                              "see quantabench.md.forcefield")
    env = dict(os.environ)
    bindir = str(Path(py).parent)
    env["PATH"] = bindir + os.pathsep + env.get("PATH", "")
    prefix = Path(py).parent.parent if os.name != "nt" else Path(py).parent
    if not env.get("AMBERHOME") and (prefix / "dat" / "leap").is_dir():
        env["AMBERHOME"] = str(prefix)
    with tempfile.TemporaryDirectory(prefix="qb_ff_") as tmp:
        tpath, rpath = Path(tmp) / "task.json", Path(tmp) / "result.json"
        tpath.write_text(json.dumps(task), encoding="utf-8")
        cp = subprocess.run([py, "-I", str(DRIVER), str(tpath), str(rpath)], cwd=tmp, env=env,  # noqa: S603
                            capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL, errors="replace")
        if not rpath.is_file():
            tail = "\n".join((cp.stderr or cp.stdout or "").strip().splitlines()[-15:])
            raise ForceFieldError(f"force-field driver failed (exit {cp.returncode}):\n{tail}")
        result = json.loads(rpath.read_text(encoding="utf-8"))
    if not result.get("ok"):
        raise ForceFieldError(f"{task['backend']}: {result.get('error', 'failed')}")
    return result


def _type_external(mol: Chem.Mol, name: str, smiles: str, backend: str, offxml: str) -> TypedMolecule:
    task = {"backend": backend, "molblock": Chem.MolToMolBlock(mol), "net_charge": Chem.GetFormalCharge(mol),
            "offxml": offxml, "name": name}
    r = _run_driver(task)
    elements = [a.GetSymbol() for a in mol.GetAtoms()]
    if r["elements"] != elements:
        raise ForceFieldError(f"{backend}: the driver changed the atom order")
    d = {k: r[k] for k in ("label", "atom_types", "lj_types", "bonds", "angles", "dihedrals", "impropers",
                           "bond_types", "angle_types", "dihedral_types", "improper_types", "styles", "mixing",
                           "special_lj", "special_coul", "notes", "references", "versions")}
    d.update(name=name, smiles=smiles, forcefield=backend, elements=elements, charges=r["charges"],
             positions=mol.GetConformer().GetPositions(), formal_charge=Chem.GetFormalCharge(mol))
    tm = TypedMolecule.from_dict(d)
    # AM1-BCC charges come rounded from sqm/antechamber (GAFF2's sum to ±0.002 e for C6F14 and hexane): spread a
    # residue below 0.01 e evenly so the molecule is exactly neutral, and say so
    resid = tm.formal_charge - tm.net_charge
    if abs(resid) > 0.01:
        raise ForceFieldError(f"{backend}: charges sum to {tm.net_charge:+.4f}, not {tm.formal_charge:+d}")
    if abs(resid) > 1e-9:
        tm.charges = tm.charges + resid / tm.n_atoms
        tm.notes.append(f"the charges summed to {tm.formal_charge - resid:+.5f} e (rounding); {resid / tm.n_atoms:+.2e} e "
                        "added to every atom")
    return tm


@dataclass(frozen=True)
class Backend:
    key: str
    description: str
    external: bool


BACKENDS: dict[str, Backend] = {
    "oplsaa": Backend("oplsaa", "OPLS-AA + Watkins-Jorgensen perfluoroalkanes (alkanes, perfluoroalkanes, "
                                "(fluoro)benzenes), in process", False),
    "openff": Backend("openff", f"OpenFF Sage ({DEFAULT_OPENFF}) + AM1-BCC, through OpenFF Interchange (conda env)", True),
    "gaff2": Backend("gaff2", "GAFF2 + AM1-BCC through AmberTools antechamber/parmchk2/tleap and ParmEd (conda env)", True),
}


def backend_available(key: str) -> tuple[bool, str]:
    if key not in BACKENDS:
        return False, f"unknown force field {key!r} (choose from {', '.join(BACKENDS)})"
    if not BACKENDS[key].external:
        return True, ""
    py = driver_python()
    return (py is not None), (py or "AmberTools not found (see quantabench.md.forcefield)")


def type_molecule(smiles: str, name: str = "", ff: str = "oplsaa", *, mol: Chem.Mol | None = None, seed: int = 42,
                  offxml: str = DEFAULT_OPENFF, overrides: dict | None = None) -> TypedMolecule:
    """Type ``smiles`` (or ``mol``: explicit H and one conformer, atom order kept) with force field ``ff``."""
    if ff not in BACKENDS:
        raise ForceFieldError(f"unknown force field {ff!r} (choose from {', '.join(BACKENDS)})")
    if mol is None:
        mol = prepare_molecule(smiles, seed=seed)
    elif mol.GetNumConformers() == 0:
        raise ForceFieldError("the molecule needs a 3D conformer")
    if Chem.GetFormalCharge(mol) != 0:
        raise ForceFieldError("a pure liquid of ions is not set up here (net charge must be 0)")
    smiles = smiles or Chem.MolToSmiles(Chem.RemoveHs(mol))
    name = name or smiles
    tm = _type_oplsaa(mol, name, smiles) if ff == "oplsaa" else _type_external(mol, name, smiles, ff, offxml)
    if overrides:
        tm.apply_overrides(overrides)
    tm.check()
    return tm


def molar_volume_A3(tm: TypedMolecule, density_g_cm3: float) -> float:
    """Volume per molecule (Å³) at a mass density."""
    return tm.mass / density_g_cm3 / AVOGADRO * 1e24


def estimate_density(mol: Chem.Mol) -> tuple[float, str]:
    """Liquid density guess for packing the box (g/cm³, how): QuantaBench's packing-coefficient estimate
    (``chem.estimates``, about 3 % on organic liquids), else Girolami's rule, else 1.0."""
    from ..chem import estimates

    d, _notes = estimates.estimate(mol, d4=False)
    for key, how in (("density_packing_g_cm3", "packing-coefficient estimate (chem.estimates)"),
                     ("density_girolami_g_cm3", "Girolami estimate (chem.estimates)")):
        v = d.get(key)
        if v and math.isfinite(v) and 0.3 < v < 4.0:
            return float(v), how
    return 1.0, "no estimate: 1.0 g/cm³ assumed"


def describe_backends(file=sys.stdout) -> None:
    for key, b in BACKENDS.items():
        ok, why = backend_available(key)
        print(f"{key:7s} {'yes' if ok else 'no ':3s} {b.description}" + ("" if ok else f" [{why}]"), file=file)
