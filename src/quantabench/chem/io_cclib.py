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
"""

from __future__ import annotations


def available() -> tuple[bool, str]:
    from ..resources import has_package

    return (True, "") if has_package("cclib") else (False, "pip install cclib")
