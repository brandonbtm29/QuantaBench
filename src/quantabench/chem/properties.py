"""Electronic properties from GFN1-xTB (dxtb) at a fixed geometry: atomic charges, dipole moment
and static polarizability.

Charges and the charge-based dipole need only dxtb's default PyTorch integrals. The polarizability
needs an electric-field interaction, i.e. dipole integrals from the libcint backend; without it
(PyPI's dxtb on Windows or macOS; the dxtb fork fixes that, FORK-1) the polarizability is skipped with a note.

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


def xtb_properties(numbers, positions_ang, masses, charge: int = 0, multiplicity: int = 1,
                   device: str = "cpu", polarizability: bool = True) -> dict:
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
    if polarizability and not has_libcint():
        out["_notes"] = [f"xTB polarizability skipped: needs dxtb's libcint backend ({libcint_status()[1]})"]
    elif polarizability:
        field = torch.zeros(3, device=dev, dtype=dtype, requires_grad=True)
        fcalc = dxtb.Calculator(z, dxtb.GFN1_XTB, interaction=new_efield(field), opts=opts, device=dev, dtype=dtype)
        with torch_compat.linalg_routing(dev):
            alpha = fcalc.get_polarizability(pos, **kwargs).detach().cpu().double().numpy() * AU_POLARIZABILITY_A3
        eig = np.linalg.eigvalsh(0.5 * (alpha + alpha.T))
        why = unphysical_polarizability(eig, numbers, charge)
        if why:
            out["_notes"] = [f"xTB polarizability discarded: {why}"]
        else:
            out["xtb_polarizability_iso_A3"] = float(eig.mean())
            out["xtb_polarizability_aniso_A3"] = float(eig.max() - eig.min())
    return out
