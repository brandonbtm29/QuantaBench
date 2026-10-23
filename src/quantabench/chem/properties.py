"""Electronic properties from GFN1-xTB (dxtb) at a fixed geometry: atomic charges, dipole moment
and static polarizability.

These are the Spartan-style inputs to the "quantum parachor" idea (polarizability, dipole, volume).
GFN1-xTB uses a minimal basis, so absolute polarizabilities are low (roughly half of DFT values);
use them for trends and as ML features, or recalibrate against a DFT subset.
"""

from __future__ import annotations

import numpy as np

from ..resources import has_package
from .descriptors import dipole_debye
from .engines import BOHR_ANG

AU_POLARIZABILITY_A3 = 0.1481847


def available() -> bool:
    return has_package("dxtb") and has_package("torch")


def xtb_properties(numbers, positions_ang, masses, charge: int = 0, multiplicity: int = 1,
                   device: str = "cpu", polarizability: bool = True) -> dict:
    import dxtb
    import torch
    from dxtb.components.field import new_efield

    dev = torch.device(device)
    dtype = torch.float64 if dev.type == "cpu" else torch.float32
    z = torch.as_tensor(np.asarray(numbers), device=dev)
    pos = torch.tensor(np.asarray(positions_ang) / BOHR_ANG, device=dev, dtype=dtype)
    kwargs = {"chrg": charge}
    if multiplicity > 1:
        kwargs["spin"] = multiplicity - 1
    opts = {"verbosity": 0}

    calc = dxtb.Calculator(z, dxtb.GFN1_XTB, opts=opts, device=dev, dtype=dtype)
    q_orb = calc.get_mulliken_charges(pos, **kwargs)
    q = calc.ihelp.reduce_orbital_to_atom(q_orb).detach().cpu().double().numpy()
    out = {
        "xtb_charges": [float(x) for x in q],
        "xtb_dipole_debye": dipole_debye(np.asarray(positions_ang), q, np.asarray(masses)),
    }
    if polarizability:
        field = torch.zeros(3, device=dev, dtype=dtype, requires_grad=True)
        fcalc = dxtb.Calculator(z, dxtb.GFN1_XTB, interaction=new_efield(field), opts=opts, device=dev, dtype=dtype)
        alpha = fcalc.get_polarizability(pos, **kwargs).detach().cpu().double().numpy() * AU_POLARIZABILITY_A3
        eig = np.linalg.eigvalsh(0.5 * (alpha + alpha.T))
        out["xtb_polarizability_iso_A3"] = float(eig.mean())
        out["xtb_polarizability_aniso_A3"] = float(eig.max() - eig.min())
    return out
