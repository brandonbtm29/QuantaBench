"""Writers for single molecules, conformer ensembles and batch summaries."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from ase import Atoms
from ase.io import write as ase_write
from rdkit import Chem

if TYPE_CHECKING:
    from .pipeline import MoleculeResult

FORMATS = {
    "xyz": "Best conformer, XYZ",
    "xyz-all": "All conformers, multi-frame XYZ",
    "extxyz": "Best conformer, extended XYZ with charges",
    "sdf": "All conformers, SDF with energies and weights",
    "pdb": "Best conformer, PDB",
    "mol": "Best conformer, MDL Molfile",
    "lammps": "Best conformer, LAMMPS data file (atom_style charge, real units, padded box)",
}
DEFAULT_FORMATS = ("xyz", "sdf", "extxyz")


def mol_to_atoms(mol: Chem.Mol, conf_id: int = -1) -> Atoms:
    conf = mol.GetConformer(conf_id)
    return Atoms([a.GetSymbol() for a in mol.GetAtoms()], positions=np.array(conf.GetPositions()))


def write_lammps_data(atoms: Atoms, path: Path, charges=None, padding: float = 10.0) -> None:
    a = atoms.copy()
    a.center(vacuum=padding)
    a.pbc = True
    if charges is not None:
        a.set_initial_charges(np.asarray(charges, dtype=float))
    ase_write(path, a, format="lammps-data", atom_style="charge", units="real", masses=True, specorder=None)


def write_molecule(result: MoleculeResult, out_dir: Path, formats=DEFAULT_FORMATS) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = result.input.id
    written: list[Path] = []
    mol = result.mol
    if mol is None or not result.conformers:
        return written
    best = result.conformers[0]
    best_atoms = mol_to_atoms(mol, best.conf_id)
    comment = (f"{result.input.name or stem} | {result.method} | E={best.energy_ev:.6f} eV | "
               f"charge={result.charge} | smiles={result.smiles}")
    for fmt in formats:
        if fmt == "xyz":
            p = out_dir / f"{stem}.xyz"
            ase_write(p, best_atoms, format="xyz", comment=comment)
        elif fmt == "xyz-all":
            p = out_dir / f"{stem}_conformers.xyz"
            frames = []
            for c in result.conformers:
                at = mol_to_atoms(mol, c.conf_id)
                at.info.update(rank=c.rank, energy_eV=c.energy_ev, rel_kcal=c.rel_kcal, weight=c.weight)
                frames.append(at)
            ase_write(p, frames, format="extxyz")
        elif fmt == "extxyz":
            p = out_dir / f"{stem}.extxyz"
            at = best_atoms.copy()
            if best.charges is not None:
                at.set_initial_charges(np.asarray(best.charges))
            at.info.update(energy_eV=best.energy_ev, method=result.method, total_charge=result.charge,
                           smiles=result.smiles)
            ase_write(p, at, format="extxyz")
        elif fmt == "sdf":
            p = out_dir / f"{stem}_conformers.sdf"
            w = Chem.SDWriter(str(p))
            for c in result.conformers:
                mol.SetProp("_Name", f"{stem} conf {c.rank}")
                for key, val in (("QB_Rank", c.rank), ("QB_Method", result.method), ("QB_Energy_eV", c.energy_ev),
                                 ("QB_RelEnergy_kcal", c.rel_kcal), ("QB_BoltzmannWeight", c.weight),
                                 ("QB_Converged", int(c.converged))):
                    mol.SetProp(key, f"{val:.6f}" if isinstance(val, float) else str(val))
                w.write(mol, confId=c.conf_id)
            w.close()
        elif fmt == "pdb":
            p = out_dir / f"{stem}.pdb"
            Chem.MolToPDBFile(mol, str(p), confId=best.conf_id)
        elif fmt == "mol":
            p = out_dir / f"{stem}.mol"
            Chem.MolToMolFile(mol, str(p), confId=best.conf_id)
        elif fmt == "lammps":
            p = out_dir / f"{stem}.lammps.data"
            write_lammps_data(best_atoms, p, best.charges)
        else:
            raise ValueError(f"Unknown export format {fmt!r} (known: {', '.join(FORMATS)})")
        written.append(p)
    summary = out_dir / f"{stem}_summary.json"
    summary.write_text(json.dumps(result.to_dict(), indent=2, default=str) + "\n", encoding="utf-8")
    written.append(summary)
    return written


def write_batch_tables(results: list[MoleculeResult], run_dir: Path) -> list[Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = [r.summary_row() for r in results]
    desc_rows = [{"id": r.input.id, **r.descriptors} for r in results if r.descriptors]
    out = []
    for name, data in (("summary.csv", rows), ("descriptors.csv", desc_rows)):
        if not data:
            continue
        keys: list[str] = []
        for row in data:
            keys += [k for k in row if k not in keys]
        path = run_dir / name
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(data)
        out.append(path)
    return out
