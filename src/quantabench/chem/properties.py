"""Electronic properties from GFN1-xTB (dxtb) at a fixed geometry: atomic charges, dipole moment
and static polarizability.

Charges and the charge-based dipole need only dxtb's default PyTorch integrals. The polarizability
needs an electric-field interaction, i.e. dipole integrals from the libcint backend; without it
(PyPI's dxtb on Windows or macOS; the dxtb fork fixes that, FORK-1) the polarizability is skipped with a note.

In an implicit solvent (``tblite_xtb_properties``) the same GFN1-xTB quantities come from tblite with the
run's solvation model (ALPB by default): dxtb has no implicit solvent. The two implementations agree in
the gas phase (Mulliken charges within 5e-5 e, the charge dipole within 1e-4 D, the polarizability within
5e-4 Å^3 for acetate, PFPrA and methylammonium), so a gas/water difference of these columns is the solvent's
effect, not the program's. The solvated polarizability is the finite-field response with the field and the
continuum in one SCF (central differences of the dipole; it matches -d2E/dF2 of the same calculations to
1e-4 Å^3, ``tests/test_properties_solvent.py``).

These are the Spartan-style inputs to the "quantum parachor" idea (polarizability, dipole, volume).
GFN1-xTB uses a minimal basis, so absolute polarizabilities are low (roughly half of DFT values);
use them for trends and as ML features, or recalibrate against a DFT subset.
"""

from __future__ import annotations

import numpy as np

from ..resources import has_package
from .descriptors import dipole_debye
from .engines import BOHR_ANG, has_libcint, libcint_status

AU_POLARIZABILITY_A3 = 0.1481847
#: Above this static polarizability per heavy atom the xTB response is not physical (iodine is ~5 A^3).
MAX_ALPHA_PER_HEAVY_ATOM_A3 = 10.0


def available() -> bool:
    return has_package("dxtb") and has_package("torch")


def tblite_available() -> bool:
    """tblite can compute the xTB properties (in any solvent it has parameters for)."""
    return has_package("tblite")


def unphysical_polarizability(eigenvalues_A3, numbers, charge: int = 0) -> str:
    """Why a polarizability tensor (eigenvalues, Å^3) cannot be physical, "" if it can. A gas-phase
    multiply charged anion (a phosphate or a disulfonate at pH 7) can have an unbound HOMO in GFN1-xTB:
    the field response then diverges (4e5 Å^3 for a C10 perfluoroalkyl phosphate dianion)."""
    eig = np.asarray(eigenvalues_A3, dtype=float)
    heavy = max(1, int(np.count_nonzero(np.asarray(numbers) > 1)))
    if np.all(np.isfinite(eig)) and eig.min() >= 0 and eig.mean() <= MAX_ALPHA_PER_HEAVY_ATOM_A3 * heavy:
        return ""
    return (f"{eig.mean():.4g} Å^3 (eigenvalues {eig.min():.3g} to {eig.max():.3g}) is unphysical for {heavy} heavy "
            f"atoms at charge {charge:+d}" + (" (gas-phase anion: unbound electrons)" if charge < 0 else ""))


def tblite_polarizability_A3(numbers, positions_ang, charge: int = 0, multiplicity: int = 1) -> np.ndarray:
    """The GFN1-xTB static polarizability tensor (Å^3) from tblite: central finite differences of the dipole in
    ±field (``eng_tblite.TBLiteCalculator.polarizability_au``), gas phase. The same quantity as dxtb's autograd
    polarizability (39.1012 vs 39.1009 Å^3 for deca-alanine) without its autograd graph (``chem.memory_guard``)."""
    from .eng_tblite import AU_POLARIZABILITY_A3 as BOHR3_A3
    from .eng_tblite import TBLiteCalculator

    calc = TBLiteCalculator(numbers, "GFN1-xTB", charge, multiplicity)
    calc.singlepoint(np.asarray(positions_ang, float))
    return calc.polarizability_au(np.asarray(positions_ang, float)) * BOHR3_A3


def tblite_xtb_properties(numbers, positions_ang, masses, charge: int = 0, multiplicity: int = 1, *,
                          solvent: str = "", solvation_model: str = "alpb", polarizability: bool = True) -> dict:
    """``xtb_properties`` from tblite's GFN1-xTB in ``solvent`` (canonical name, "" = gas phase): Mulliken charges,
    the charge dipole about the center of mass and the finite-field polarizability, all with the same solvation.
    ``_solvation`` holds the model used ("ALPB(water)")."""
    from .eng_tblite import TBLiteCalculator, polarizability_summary

    pos = np.asarray(positions_ang, float)
    calc = TBLiteCalculator(numbers, "GFN1-xTB", charge, multiplicity, solvent=solvent or "",
                            solvation_model=solvation_model or "alpb")
    res = calc.singlepoint(pos)
    q = np.asarray(res.get("charges"), float)
    out: dict = {"xtb_charges": [float(x) for x in q],
                 "xtb_dipole_debye": dipole_debye(pos, q, np.asarray(masses, float)),
                 "_solvation": calc.solvation_label}
    notes = list(calc.notes)
    if polarizability:
        alpha_au = calc.polarizability_au(pos)
        eig = np.linalg.eigvalsh(0.5 * (alpha_au + alpha_au.T)) * AU_POLARIZABILITY_A3
        why = unphysical_polarizability(eig, numbers, charge)
        if why:
            notes.append(f"xTB polarizability discarded: {why}")
        else:
            out["xtb_polarizability_iso_A3"], out["xtb_polarizability_aniso_A3"] = polarizability_summary(alpha_au)
    if notes:
        out["_notes"] = notes
    return out


def xtb_properties(numbers, positions_ang, masses, charge: int = 0, multiplicity: int = 1,
                   device: str = "cpu", polarizability: bool = True, polarizability_engine: str = "dxtb") -> dict:
    """GFN1-xTB charges, charge dipole and (``polarizability``) static polarizability. ``polarizability_engine``
    "tblite" takes the polarizability from tblite's finite-field GFN1-xTB instead of dxtb's autograd (the memory
    guard's cheaper path; the charges and dipole stay dxtb's)."""
    import dxtb
    import torch
    from dxtb.components.field import new_efield

    from . import torch_compat

    dev = torch.device(torch_compat.dxtb_device(device)[0])     # CPU where dxtb is slower on the GPU
    dtype = torch_compat.compute_dtype(dev)
    z = torch.as_tensor(np.asarray(numbers), device=dev)
    pos = torch.tensor(np.asarray(positions_ang) / BOHR_ANG, device=dev, dtype=dtype)
    kwargs = {"chrg": charge}
    if multiplicity > 1:
        kwargs["spin"] = multiplicity - 1
    opts = {"verbosity": 0}

    calc = dxtb.Calculator(z, dxtb.GFN1_XTB, opts=opts, device=dev, dtype=dtype)
    with torch_compat.linalg_routing(dev):
        q_orb = calc.get_mulliken_charges(pos, **kwargs)
    q = calc.ihelp.reduce_orbital_to_atom(q_orb).detach().cpu().double().numpy()
    out = {
        "xtb_charges": [float(x) for x in q],
        "xtb_dipole_debye": dipole_debye(np.asarray(positions_ang), q, np.asarray(masses)),
    }
    if polarizability and polarizability_engine == "dxtb" and not has_libcint():
        out["_notes"] = [f"xTB polarizability skipped: needs dxtb's libcint backend ({libcint_status()[1]})"]
    elif polarizability:
        if polarizability_engine == "tblite":
            alpha = tblite_polarizability_A3(numbers, positions_ang, charge, multiplicity)
        elif polarizability_engine == "dxtb":
            field = torch.zeros(3, device=dev, dtype=dtype, requires_grad=True)
            fcalc = dxtb.Calculator(z, dxtb.GFN1_XTB, interaction=new_efield(field), opts=opts, device=dev,
                                    dtype=dtype)
            with torch_compat.linalg_routing(dev):
                alpha = fcalc.get_polarizability(pos, **kwargs).detach().cpu().double().numpy() * AU_POLARIZABILITY_A3
        else:
            raise ValueError(f"polarizability_engine must be 'dxtb' or 'tblite', not {polarizability_engine!r}")
        eig = np.linalg.eigvalsh(0.5 * (alpha + alpha.T))
        why = unphysical_polarizability(eig, numbers, charge)
        if why:
            out["_notes"] = [f"xTB polarizability discarded: {why}"]
        else:
            out["xtb_polarizability_iso_A3"] = float(eig.mean())
            out["xtb_polarizability_aniso_A3"] = float(eig.max() - eig.min())
    return out
