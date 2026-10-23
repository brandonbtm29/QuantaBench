"""Nonbonded (Lennard-Jones 12-6) parameters for the rigid-adsorbate SMD bridge.

Why these parameters
--------------------
The SMD bridge treats the adsorbate as a rigid body and the slab as frozen, so only the
*intermolecular* adsorbate-surface (and adsorbate-adsorbate) nonbonded terms ever produce motion.
That needs one Lennard-Jones pair per element, not a bonded force field, which is exactly what
UFF provides for the whole periodic table.

* **Adsorbate atoms: UFF** (Rappé, Casewit, Colwell, Goddard and Skiff, *J. Am. Chem. Soc.* 1992,
  114, 10024-10035, Table 1). UFF lists the van der Waals bond distance ``x_i`` (the position of
  the LJ minimum, Å) and well depth ``D_i`` (kcal/mol). For LAMMPS' ``lj/cut`` form
  ``E = 4 ε [(σ/r)^12 - (σ/r)^6]`` the minimum is at ``r = 2^(1/6) σ``, so
  ``ε = D_i`` and ``σ = x_i / 2^(1/6)``.
* **Graphite carbon: AMBER aromatic carbon** (Cornell et al., *J. Am. Chem. Soc.* 1995, 117,
  5179-5197; parm94 atom type CA: R*/2 = 1.9080 Å, ε = 0.0860 kcal/mol, i.e. σ = 3.400 Å).
  It is a common choice for graphene/graphite sheets in biomolecular-style MD. The classic
  Steele graphite potential (W. A. Steele, *Surf. Sci.* 1973, 36, 317-352) uses the same
  σ = 3.40 Å with a shallower ε/k_B = 28.0 K = 0.0556 kcal/mol; select it with
  ``graphite="steele"``. The choice shifts adsorption energies by roughly the ratio
  sqrt(0.086/0.0556) ≈ 1.24 for the carbon-adsorbate term, so report which one you used.
* **Mixing: geometric** for both ε and σ (``pair_modify mix geometric``), which is UFF's own
  combination rule. Note that AMBER itself uses Lorentz-Berthelot (arithmetic σ); with
  σ_graphite = 3.40 Å and UFF σ between 2.6 and 4.0 Å the two rules differ by < 1.5 % in σ_ij.

All parameters can be overridden per type label (``"C_surf"``, ``"F"``, ...) with a dict of
``(epsilon_kcal_mol, sigma_angstrom)`` tuples.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

TWO_POW_SIXTH = 2.0 ** (1.0 / 6.0)

# Rappé et al. 1992, Table 1: element -> (x_i in Å, D_i in kcal/mol).
# Values are those of the generic (first-listed) UFF type for each element; UFF uses one
# nonbonded pair per element regardless of hybridization, so the choice of type does not matter.
UFF_LJ: dict[str, tuple[float, float]] = {
    "H": (2.886, 0.044),
    "He": (2.362, 0.056),
    "Li": (2.451, 0.025),
    "Be": (2.745, 0.085),
    "B": (4.083, 0.180),
    "C": (3.851, 0.105),
    "N": (3.660, 0.069),
    "O": (3.500, 0.060),
    "F": (3.364, 0.050),
    "Ne": (3.243, 0.042),
    "Na": (2.983, 0.030),
    "Mg": (3.021, 0.111),
    "Al": (4.499, 0.505),
    "Si": (4.295, 0.402),
    "P": (4.147, 0.305),
    "S": (4.035, 0.274),
    "Cl": (3.947, 0.227),
    "Ar": (3.868, 0.185),
    "K": (3.812, 0.035),
    "Ca": (3.399, 0.238),
    "Zn": (2.763, 0.124),
    "Se": (4.205, 0.291),
    "Br": (4.189, 0.251),
    "I": (4.500, 0.339),
}

# Graphite / graphene carbon presets: name -> (epsilon kcal/mol, sigma Å, citation)
GRAPHITE_LJ: dict[str, tuple[float, float, str]] = {
    "amber": (0.0860, 3.400, "Cornell et al., J. Am. Chem. Soc. 1995, 117, 5179 (AMBER parm94 CA)"),
    "steele": (0.0556, 3.400, "Steele, Surf. Sci. 1973, 36, 317 (eps/kB = 28.0 K)"),
}
DEFAULT_GRAPHITE = "amber"

UFF_CITATION = "Rappé et al., J. Am. Chem. Soc. 1992, 114, 10024 (UFF, Table 1)"


@dataclass(frozen=True)
class LJ:
    """One LAMMPS ``lj/cut`` pair: ε in kcal/mol, σ in Å (LAMMPS ``units real``)."""

    epsilon: float
    sigma: float
    source: str = ""

    @property
    def r_min(self) -> float:
        return TWO_POW_SIXTH * self.sigma


def uff_lj(element: str) -> LJ:
    """UFF Lennard-Jones parameters of ``element`` converted to (ε, σ)."""
    try:
        x, d = UFF_LJ[element]
    except KeyError:
        raise KeyError(f"No UFF Lennard-Jones parameters tabulated for {element!r}; pass them in "
                       f"params={{'{element}': (epsilon_kcal_mol, sigma_A)}}") from None
    return LJ(epsilon=d, sigma=x / TWO_POW_SIXTH, source=UFF_CITATION)


def graphite_lj(preset: str = DEFAULT_GRAPHITE) -> LJ:
    try:
        eps, sig, cite = GRAPHITE_LJ[preset]
    except KeyError:
        raise ValueError(f"unknown graphite preset {preset!r} (known: {', '.join(GRAPHITE_LJ)})") from None
    return LJ(epsilon=eps, sigma=sig, source=cite)


def mix_geometric(a: LJ, b: LJ) -> LJ:
    """ε_ij = sqrt(ε_i ε_j), σ_ij = sqrt(σ_i σ_j): what ``pair_modify mix geometric`` does."""
    return LJ(math.sqrt(a.epsilon * b.epsilon), math.sqrt(a.sigma * b.sigma), "geometric mixing")


def resolve_type_params(type_labels: Mapping[str, str], surface_labels: set[str], graphite: str = DEFAULT_GRAPHITE,
                        overrides: Mapping[str, tuple[float, float]] | None = None) -> dict[str, LJ]:
    """LJ parameters for every type label.

    ``type_labels`` maps a label (``"C_surf"``, ``"F"``) to its element. Surface carbon gets the
    graphite preset, other surface elements and all adsorbate elements get UFF. ``overrides``
    (label or element -> (ε, σ)) wins over both; a label key beats an element key.
    """
    overrides = dict(overrides or {})
    out: dict[str, LJ] = {}
    for label, element in type_labels.items():
        if label in overrides or element in overrides and label not in surface_labels:
            eps, sig = overrides.get(label, overrides.get(element))
            out[label] = LJ(float(eps), float(sig), "user override")
        elif label in surface_labels and element == "C":
            out[label] = graphite_lj(graphite)
        else:
            out[label] = uff_lj(element)
    return out
