"""Descriptors for QSAR/QSPR work, from the 2D graph and the 3D conformer ensemble.

3D values are reported for the lowest-energy conformer and as Boltzmann-weighted averages,
because flexible PFAS chains change volume and surface area noticeably between conformers.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, rdFreeSASA, rdMolDescriptors

DEBYE_PER_E_ANG = 4.80320471


def graph_descriptors(mol: Chem.Mol) -> dict:
    heavy = Chem.RemoveHs(mol)
    n_c = sum(1 for a in heavy.GetAtoms() if a.GetSymbol() == "C")
    n_f = sum(1 for a in heavy.GetAtoms() if a.GetSymbol() == "F")
    f_count = [sum(1 for nb in a.GetNeighbors() if nb.GetSymbol() == "F")
               for a in heavy.GetAtoms() if a.GetSymbol() == "C"]
    return {
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "mol_weight": Descriptors.MolWt(mol),
        "formal_charge": sum(a.GetFormalCharge() for a in mol.GetAtoms()),
        "heavy_atoms": heavy.GetNumAtoms(),
        "n_carbon": n_c,
        "n_fluorine": n_f,
        "f_to_c_ratio": n_f / n_c if n_c else 0.0,
        "n_cf2": sum(1 for n in f_count if n == 2),
        "n_cf3": sum(1 for n in f_count if n == 3),
        "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "fraction_csp3": rdMolDescriptors.CalcFractionCSP3(mol),
        "tpsa": rdMolDescriptors.CalcTPSA(mol),
        "logp_crippen": Crippen.MolLogP(mol),
        "hbd": rdMolDescriptors.CalcNumHBD(mol),
        "hba": rdMolDescriptors.CalcNumHBA(mol),
    }


def _sasa(mol: Chem.Mol, conf_id: int) -> float:
    radii = rdFreeSASA.classifyAtoms(mol)
    return float(rdFreeSASA.CalcSASA(mol, radii, confIdx=conf_id))


def dipole_debye(positions: np.ndarray, charges: np.ndarray, masses: np.ndarray) -> float:
    """Dipole from point charges about the center of mass (the convention for ions)."""
    com = (positions * masses[:, None]).sum(0) / masses.sum()
    mu = ((positions - com) * charges[:, None]).sum(0)
    return float(np.linalg.norm(mu) * DEBYE_PER_E_ANG)


def conformer_descriptors(mol: Chem.Mol, conf_id: int) -> dict:
    return {
        "volume_A3": float(AllChem.ComputeMolVolume(mol, confId=conf_id, gridSpacing=0.2)),
        "sasa_A2": _sasa(mol, conf_id),
        "radius_of_gyration_A": rdMolDescriptors.CalcRadiusOfGyration(mol, confId=conf_id),
        "asphericity": rdMolDescriptors.CalcAsphericity(mol, confId=conf_id),
        "npr1": rdMolDescriptors.CalcNPR1(mol, confId=conf_id),
        "npr2": rdMolDescriptors.CalcNPR2(mol, confId=conf_id),
    }


def ensemble_descriptors(mol: Chem.Mol, conf_ids: list[int], weights: np.ndarray) -> dict:
    if not conf_ids:
        return {}
    per = [conformer_descriptors(mol, cid) for cid in conf_ids]
    out = {f"{k}_best": v for k, v in per[0].items()}
    for k in per[0]:
        out[f"{k}_boltz"] = float(sum(w * p[k] for w, p in zip(weights, per)))
    return out
