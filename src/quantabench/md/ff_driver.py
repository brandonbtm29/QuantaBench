"""Force-field driver for md.forcefield's ``openff`` and ``gaff2`` backends (MD-2).

Runs inside the conda environment that has AmberTools (and OpenFF), never in QuantaBench's own: it imports only
the standard library, ``openff.*`` and ``parmed``. Usage (md.forcefield does this)::

    python -I ff_driver.py task.json result.json

task: ``{"backend": "openff" | "gaff2", "molblock": MOL V2000 with explicit H and 3D coordinates,
"net_charge": 0, "offxml": "openff-2.3.0.offxml", "name": ...}``. The atom order of the MOL block is kept.

result: per-atom elements, type names and charges; Lennard-Jones types (ε kcal/mol, σ Å); bonds, angles, dihedrals
and impropers with LAMMPS ``real``-unit coefficients (bond/angle harmonic with K as LAMMPS takes it, dihedral
fourier, improper cvff), mixing rule, 1-4 scaling, versions, and a source string for every type. On failure
``{"ok": false, "error": ...}``.

Unit conventions: OpenFF Interchange's LAMMPS writer already converts (its bond and angle K are LAMMPS', i.e. half
of the SMIRNOFF k). Amber (ParmEd) bonds and angles use E = k (x - x0)², the same convention as LAMMPS harmonic;
Amber torsions E = Σ k [1 + cos(nφ - γ)] are LAMMPS ``dihedral_style fourier`` terms; impropers with γ = 0 or 180°
are ``improper_style cvff`` (K, d = cos γ, n). Lennard-Jones: σ = 2 R_min/2 / 2^(1/6).
"""

from __future__ import annotations

import glob
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
import warnings
from pathlib import Path

TWO_SIXTH = 2.0 ** (1.0 / 6.0)


def _conda_version(name: str) -> str:
    prefix = Path(sys.executable).resolve().parent.parent
    for p in sorted(glob.glob(str(prefix / "conda-meta" / f"{name}-[0-9]*.json"))):
        try:
            return json.loads(Path(p).read_text(encoding="utf-8"))["version"]
        except (OSError, ValueError, KeyError):
            pass
    return ""


def _sections(text: str) -> dict[str, list[list[str]]]:
    """A LAMMPS data file's sections: name -> rows of tokens (comments dropped)."""
    out: dict[str, list[list[str]]] = {}
    names = ("Masses", "Pair Coeffs", "Bond Coeffs", "Angle Coeffs", "Dihedral Coeffs", "Improper Coeffs",
             "Atoms", "Velocities", "Bonds", "Angles", "Dihedrals", "Impropers")
    cur = None
    for line in text.splitlines():
        s = line.split("#")[0].strip()
        head = next((n for n in names if s == n or s.startswith(n + " ")), None)
        if head:
            cur = head
            out[cur] = []
            continue
        if not s:
            continue
        if cur and s.split()[0].lstrip("-").replace(".", "", 1).isdigit():
            out[cur].append(s.split())
        elif cur:
            cur = None
    return out


ELEMENT_BY_MASS = {1: "H", 12: "C", 14: "N", 16: "O", 19: "F", 31: "P", 32: "S", 35: "Cl", 80: "Br", 127: "I",
                   28: "Si", 11: "B"}


def openff(task: dict, work: Path) -> dict:
    warnings.filterwarnings("ignore")
    import numpy as np
    import openff.interchange
    import openff.toolkit
    from openff.interchange import Interchange
    from openff.toolkit import ForceField, Molecule
    from openff.units import unit

    (work / "in.mol").write_text(task["molblock"], encoding="utf-8")
    mol = Molecule.from_file(str(work / "in.mol"), file_format="MOL", allow_undefined_stereo=True)
    mol.assign_partial_charges("am1bcc", use_conformers=[mol.conformers[0]])
    ff = ForceField(task.get("offxml") or "openff-2.3.0.offxml")
    labels = ff.label_molecules(mol.to_topology())[0]
    ic = Interchange.from_smirnoff(ff, [mol], charge_from_molecules=[mol])
    ic.box = np.eye(3) * 5.0 * unit.nanometer
    ic.to_lammps(str(work / "off"))
    data = (work / "off.lmp").read_text(encoding="utf-8")
    inp = next(work.glob("off*.in")).read_text(encoding="utf-8")
    sec = _sections(data)
    ffname = task.get("offxml") or "openff-2.3.0.offxml"
    n = mol.n_atoms
    elements = [a.symbol for a in mol.atoms]
    atoms = sorted(sec["Atoms"], key=lambda r: int(r[0]))
    if len(atoms) != n:
        raise RuntimeError("Interchange wrote a different number of atoms")
    vdw = {k[0]: p for k, p in labels["vdW"].items()}
    mass = {int(r[0]): float(r[1]) for r in sec["Masses"]}
    pair = {int(r[0]): (float(r[1]), float(r[2])) for r in sec["Pair Coeffs"]}
    num_name: dict[int, str] = {}       # Interchange's numeric type -> element_vdWid
    num_param: dict[int, object] = {}
    atom_types = []
    for i, r in enumerate(atoms):
        t = int(r[2])
        num_name.setdefault(t, f"{elements[i]}_{vdw[i].id}")
        num_param.setdefault(t, vdw[i])
        atom_types.append(num_name[t])
    lj = {nm: {"name": nm, "element": nm.split("_")[0], "mass": mass[t], "epsilon": pair[t][0], "sigma": pair[t][1],
               "source": f"OpenFF {ffname} vdW {num_param[t].id} ({num_param[t].smirks})"}
          for t, nm in num_name.items()}
    charges = [float(r[3]) for r in atoms]

    def lookup(handler: str, idx: tuple[int, ...]):
        lab = labels[handler]
        for k in (idx, idx[::-1]):
            if k in lab:
                return lab[k]
        for k, p in lab.items():         # impropers: Interchange may order the atoms differently
            if set(k) == set(idx):
                return p
        return None

    def terms(sec_name: str, coeff_name: str, k: int, handler: str, style: str):
        coeffs = {int(r[0]): [float(x) for x in r[1:]] for r in sec.get(coeff_name, [])}
        rows, types = [], {}
        for r in sec.get(sec_name, []):
            t = int(r[1])
            idx = tuple(int(x) - 1 for x in r[2:2 + k])
            p = lookup(handler, idx)
            pid = p.id if p is not None else "?"
            label = f"{pid}:{t}"
            if label not in types:
                c = coeffs[t]
                if style == "fourier":
                    c = c[1:]        # Interchange writes "m K n d" with m = 1 term per type
                types[label] = {"style": style, "coeffs": c,
                                "source": f"OpenFF {ffname} {pid}" + (f" ({p.smirks})" if p is not None else "")}
            rows.append([*idx, label])
        return rows, types

    bonds, btypes = terms("Bonds", "Bond Coeffs", 2, "Bonds", "harmonic")
    angles, atypes = terms("Angles", "Angle Coeffs", 3, "Angles", "harmonic")
    dihedrals, dtypes = terms("Dihedrals", "Dihedral Coeffs", 4, "ProperTorsions", "fourier")
    impropers, itypes = terms("Impropers", "Improper Coeffs", 4, "ImproperTorsions", "cvff")
    special_lj, special_coul = [0.0, 0.0, 0.5], [0.0, 0.0, 0.8333333333]
    styles = {"bond": "harmonic", "angle": "harmonic", "dihedral": "fourier", "improper": "cvff"}
    mixing = "arithmetic"
    for line in inp.splitlines():
        tok = line.split()
        if not tok:
            continue
        if tok[0] == "special_bonds" and "lj" in tok and "coul" in tok:
            i, j = tok.index("lj"), tok.index("coul")
            special_lj = [float(x) for x in tok[i + 1:i + 4]]
            special_coul = [float(x) for x in tok[j + 1:j + 4]]
        elif tok[0] in ("bond_style", "angle_style", "dihedral_style", "improper_style"):
            styles[tok[0].split("_")[0]] = tok[1]
        elif tok[0] == "pair_modify" and "mix" in tok:
            mixing = tok[tok.index("mix") + 1]
    return {"label": f"OpenFF {ffname.removesuffix('.offxml')} with AM1-BCC charges (AmberTools sqm), "
                     "exported by OpenFF Interchange",
            "elements": elements, "atom_types": atom_types, "charges": charges, "lj_types": lj,
            "bonds": bonds, "angles": angles, "dihedrals": dihedrals, "impropers": impropers,
            "bond_types": btypes, "angle_types": atypes, "dihedral_types": dtypes, "improper_types": itypes,
            "styles": styles, "mixing": mixing, "special_lj": special_lj, "special_coul": special_coul,
            "notes": [f"AM1-BCC charges from one conformer (QuantaBench's), sqm via the OpenFF toolkit {openff.toolkit.__version__}"],
            "references": [
                "Boothroyd et al. Development and benchmarking of Open Force Field 2.0.0: the Sage small molecule "
                "force field. J. Chem. Theory Comput. 2023, 19, 3251-3275. doi:10.1021/acs.jctc.3c00039 "
                f"(the parameters used: {ffname})",
                "Jakalian, Jack, Bayly. Fast, efficient generation of high-quality atomic charges. AM1-BCC model: "
                "II. J. Comput. Chem. 2002, 23, 1623-1641. doi:10.1002/jcc.10128"],
            "versions": {"openff-toolkit": openff.toolkit.__version__, "openff-interchange": openff.interchange.__version__,
                         "openff-forcefields": _conda_version("openff-forcefields"), "ambertools": _conda_version("ambertools"),
                         "offxml": ffname}}


def _run(cmd: list[str], work: Path, log: str) -> None:
    exe = shutil.which(cmd[0])
    if exe is None:
        raise RuntimeError(f"{cmd[0]} not found on PATH")
    with open(work / log, "w") as fh:
        cp = subprocess.run([exe, *cmd[1:]], cwd=work, stdout=fh, stderr=subprocess.STDOUT,  # noqa: S603
                            stdin=subprocess.DEVNULL, timeout=3600)
    if cp.returncode != 0:
        tail = "\n".join((work / log).read_text(errors="replace").splitlines()[-12:])
        raise RuntimeError(f"{' '.join(cmd[:3])} failed (exit {cp.returncode}):\n{tail}")


def _canon(*types: str) -> str:
    """The label of a bonded type read either way round (f-c3 and c3-f are one type)."""
    return "-".join(min(tuple(types), tuple(types)[::-1]))


def gaff2(task: dict, work: Path) -> dict:
    import parmed

    (work / "in.mol").write_text(task["molblock"], encoding="utf-8")
    q = int(task.get("net_charge", 0))
    _run(["antechamber", "-i", "in.mol", "-fi", "mdl", "-o", "mol.mol2", "-fo", "mol2", "-c", "bcc", "-at", "gaff2",
          "-nc", str(q), "-rn", "MOL", "-pf", "y"], work, "antechamber.log")
    _run(["parmchk2", "-i", "mol.mol2", "-f", "mol2", "-o", "mol.frcmod", "-s", "gaff2"], work, "parmchk2.log")
    (work / "leap.in").write_text("source leaprc.gaff2\nloadamberparams mol.frcmod\nm = loadmol2 mol.mol2\n"
                                  "saveamberparm m mol.prmtop mol.inpcrd\nquit\n")
    _run(["tleap", "-f", "leap.in"], work, "tleap.log")
    frcmod = (work / "mol.frcmod").read_text(encoding="utf-8")
    missing = [ln for ln in frcmod.splitlines() if "ATTN" in ln]
    s = parmed.load_file(str(work / "mol.prmtop"), str(work / "mol.inpcrd"))
    ref = [ln.split() for ln in task["molblock"].splitlines()[4:4 + len(s.atoms)]]
    elements = []
    for a, r in zip(s.atoms, ref):
        el = r[3]
        if abs(a.xx - float(r[0])) > 1e-2 or abs(a.xy - float(r[1])) > 1e-2 or abs(a.xz - float(r[2])) > 1e-2:
            raise RuntimeError("antechamber/tleap changed the atom order")
        elements.append(el)
    lj, atom_types = {}, []
    for a in s.atoms:
        atom_types.append(a.type)
        if a.type not in lj:
            # prmtop stores single-precision-like values (0.0832 reads back as 0.08319999988): round to gaff2.dat's digits
            eps, rmin = round(float(a.epsilon), 6), round(float(a.rmin), 6)
            lj[a.type] = {"name": a.type, "element": elements[a.idx], "mass": float(a.mass), "epsilon": eps,
                          "sigma": 2.0 * rmin / TWO_SIXTH,
                          "source": f"GAFF2 type {a.type} (gaff2.dat, AmberTools {_conda_version('ambertools')}; "
                                    "R_min/2 and epsilon)"}
    btypes, bonds = {}, []
    for b in s.bonds:
        lab = _canon(b.atom1.type, b.atom2.type)
        c = [round(float(b.type.k), 6), round(float(b.type.req), 6)]
        key = f"{lab}:{c[0]}:{c[1]}"
        btypes.setdefault(key, {"style": "harmonic", "coeffs": c, "source": f"GAFF2 bond {lab} (gaff2.dat or parmchk2)"})
        bonds.append([b.atom1.idx, b.atom2.idx, key])
    atypes, angles = {}, []
    for t in s.angles:
        lab = _canon(t.atom1.type, t.atom2.type, t.atom3.type)
        c = [round(float(t.type.k), 6), round(float(t.type.theteq), 6)]
        key = f"{lab}:{c[0]}:{c[1]}"
        atypes.setdefault(key, {"style": "harmonic", "coeffs": c, "source": f"GAFF2 angle {lab} (gaff2.dat or parmchk2)"})
        angles.append([t.atom1.idx, t.atom2.idx, t.atom3.idx, key])
    dtypes, dihedrals, itypes, impropers = {}, [], {}, []
    scee, scnb = set(), set()
    for d in s.dihedrals:
        types4 = (d.atom1.type, d.atom2.type, d.atom3.type, d.atom4.type)
        lab = "-".join(types4) if d.improper else _canon(*types4)     # an improper's order is its definition
        idx = [d.atom1.idx, d.atom2.idx, d.atom3.idx, d.atom4.idx]
        terms = d.type if isinstance(d.type, parmed.DihedralTypeList) else [d.type]
        for t in terms:
            if d.improper:
                cosg = math.cos(math.radians(t.phase))
                if abs(abs(cosg) - 1) > 1e-6:
                    raise RuntimeError(f"improper {lab} has phase {t.phase}: not representable with cvff")
                c = [round(float(t.phi_k), 6), int(round(cosg)), int(t.per)]
                key = f"{lab}:{c[0]}:{c[1]}:{c[2]}"
                itypes.setdefault(key, {"style": "cvff", "coeffs": c, "source": f"GAFF2 improper {lab} (gaff2.dat or parmchk2)"})
                impropers.append([*idx, key])
            else:
                c = [round(float(t.phi_k), 6), int(t.per), round(float(t.phase), 6)]
                key = f"{lab}:{c[0]}:{c[1]}:{c[2]}"
                dtypes.setdefault(key, {"style": "fourier", "coeffs": c,
                                        "source": f"GAFF2 torsion {lab} (gaff2.dat or parmchk2)"})
                dihedrals.append([*idx, key])
                scee.add(round(float(t.scee), 4))
                scnb.add(round(float(t.scnb), 4))
    if len(scee) > 1 or len(scnb) > 1:
        raise RuntimeError(f"mixed 1-4 scaling factors {scee} {scnb}: not representable with one special_bonds")
    sc = scee.pop() if scee else 1.2
    sn = scnb.pop() if scnb else 2.0
    notes = ["AM1-BCC charges from antechamber -c bcc on one conformer (QuantaBench's)"]
    if missing:
        notes.append(f"parmchk2 marked {len(missing)} parameter(s) as guessed (ATTN): " + "; ".join(m.strip() for m in missing))
    return {"label": "GAFF2 with AM1-BCC charges (AmberTools antechamber, parmchk2, tleap; read with ParmEd)",
            "elements": elements, "atom_types": atom_types, "charges": [float(a.charge) for a in s.atoms],
            "lj_types": lj, "bonds": bonds, "angles": angles, "dihedrals": dihedrals, "impropers": impropers,
            "bond_types": btypes, "angle_types": atypes, "dihedral_types": dtypes, "improper_types": itypes,
            "styles": {"bond": "harmonic", "angle": "harmonic", "dihedral": "fourier", "improper": "cvff"},
            "mixing": "arithmetic", "special_lj": [0.0, 0.0, 1.0 / sn], "special_coul": [0.0, 0.0, 1.0 / sc],
            "notes": notes,
            "references": [
                "Wang, Wolf, Caldwell, Kollman, Case. Development and testing of a general amber force field. "
                "J. Comput. Chem. 2004, 25, 1157-1174. doi:10.1002/jcc.20035 (GAFF; GAFF2 is its AmberTools update)",
                "Jakalian, Jack, Bayly. Fast, efficient generation of high-quality atomic charges. AM1-BCC model: "
                "II. J. Comput. Chem. 2002, 23, 1623-1641. doi:10.1002/jcc.10128"],
            "versions": {"ambertools": _conda_version("ambertools"), "parmed": parmed.__version__}}


def main(argv: list[str]) -> int:
    task_path, result_path = Path(argv[1]), Path(argv[2])
    task = json.loads(task_path.read_text(encoding="utf-8"))
    try:
        with tempfile.TemporaryDirectory(prefix="qbff_") as tmp:
            work = Path(tmp)
            fn = {"openff": openff, "gaff2": gaff2}[task["backend"]]
            out = fn(task, work)
        out["ok"] = True
    except Exception as exc:  # noqa: BLE001 - reported back to QuantaBench as data
        out = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
    result_path.write_text(json.dumps(out))
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    raise SystemExit(main(sys.argv))
