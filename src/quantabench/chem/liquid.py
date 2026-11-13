"""Dimensionless liquid-state groups and a Lifshitz estimate of the surface tension (DESC-3, the feature builder).

Every molecule's descriptors already hold a polarizability α, a dipole μ, a van der Waals volume V and area A, and a
liquid molar volume V_m (``chem.estimates``). These turn them into pure numbers, the forms in which the physics of a
liquid's cohesion is written. With v_m = V_m / N_A the volume one molecule takes in the liquid:

    lorentz_lorenz_bosque        LL = (4π/3) α / v_m = (n² − 1) / (n² + 2), Clausius-Mossotti / Lorentz-Lorenz with
                                 Bosque's additive α (an optical-like polarizability): dispersion cohesion
    refractive_index_bosque      the n that LL implies (LL capped at 0.95)
    dipole_orientation_ratio     ORI = (4π/3) μ² / (3 · 4πε₀ · kT) / v_m: the Debye orientation term, dipole-dipole
                                 cohesion against thermal energy (Boltzmann-averaged dipole of the phase)
    packing_fraction_vdw         η = V_vdW / v_m: how much of the liquid the molecules' van der Waals volume fills
    sphericity_vdw               ψ = π^⅓ (6 V)^⅔ / A: 1 for a sphere, lower for chains and flat molecules
    hamaker_lifshitz_zJ          the Hamaker constant of the liquid across vacuum from Lifshitz theory in
                                 Israelachvili's approximation (Intermolecular and Surface Forces, ch. 13):
                                 A_H = (3/4) kT ((ε − 1)/(ε + 1))² + (3 h ν_e / 16√2) (n² − 1)² / (n² + 1)^(3/2),
                                 ν_e = 3.0e15 s⁻¹, ε from the Debye equation (LL + ORI, capped at 0.97), in zJ
    sigma_lifshitz_mN_m          σ ≈ A_H / (24 π D₀²), D₀ = 0.165 nm: a no-fit estimate of the surface tension from
                                 dispersion (and a bounded polar part). It misses H-bonding and is crude (several
                                 mN/m); it is a feature, not a prediction to report
    sigma_scale_kT_mN_m          kT / v_m^(2/3): the unit of the reduced surface tension σ* = σ / this, the surface
                                 energy of one molecular cross-section in units of kT (the form of Eötvös' rule,
                                 σ V_m^(2/3) ∝ T_c − T). A model of σ* instead of σ does not have to learn the trivial
                                 growth of σ·area with size

Why: a σ model on descriptors interpolates but extrapolates badly to fluorinated liquids, which are less
polarizable per volume and less spherical than most training liquids; the dimensionless groups and the reduced
target describe exactly that (docs/SURFACE_TENSION.md, "Dimensionless groups"). The Analyze table's feature
presets (``analysis.features``) are the same formulas as table formula columns, for run folders made before these
columns existed; ``tests/test_liquid.py`` checks that the two agree.

A missing input gives no value for the groups that need it (never an error). The groups are for neutral molecules
in a liquid; an ion gets them too (from the same inputs), but a liquid of an ion is not defined: read them with care.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

AVOGADRO = 6.02214076e23
BOLTZMANN = 1.380649e-23            # J/K
PLANCK = 6.62607015e-34             # J s
DEBYE_CM = 3.33564e-30              # C m
FOUR_PI_EPS0 = 1.11265005545e-10    # C² / (J m)
A3_PER_CM3_MOL = 1e24 / AVOGADRO    # Å³ per (cm³/mol): one molecule's share of a molar volume
NU_E = 3.0e15                       # s⁻¹: the main UV absorption frequency of Israelachvili's approximation
D0_M = 0.165e-9                     # m: the "cut-off" distance of σ = A_H / (24 π D0²)
LL_CAP = 0.95                       # LL near 1 means a refractive index diverging: cap it
STATIC_CAP = 0.97                   # the Debye function of the static ε, capped (polar liquids need Onsager)

#: the descriptor names this module writes, in order
COLUMNS = ("lorentz_lorenz_bosque", "refractive_index_bosque", "dipole_orientation_ratio", "packing_fraction_vdw",
           "sphericity_vdw", "hamaker_lifshitz_zJ", "sigma_lifshitz_mN_m", "sigma_scale_kT_mN_m")


def orientation_A3_per_D2(temperature_k: float = 298.15) -> float:
    """μ² / (3 · 4πε₀ · kT) in Å³ per D²: the orientation polarizability of a 1 D dipole (8.10 Å³ at 25 °C)."""
    return DEBYE_CM ** 2 / (FOUR_PI_EPS0 * 3 * BOLTZMANN * temperature_k) * 1e30


def _num(v) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def groups(alpha_A3, mu_debye, vdw_volume_A3, vdw_area_A2, molar_volume_cm3,
           temperature_k: float = 298.15) -> dict[str, float]:
    """The groups of the module docstring for one molecule; a group whose inputs are missing (None, NaN) or not
    positive is left out."""
    a, mu, v, area, vm = map(_num, (alpha_A3, mu_debye, vdw_volume_A3, vdw_area_A2, molar_volume_cm3))
    t = float(temperature_k)
    kt = BOLTZMANN * t
    out: dict[str, float] = {}
    vmol = vm * A3_PER_CM3_MOL if vm and vm > 0 else None          # Å³ per molecule in the liquid
    if vmol:
        out["sigma_scale_kT_mN_m"] = kt / (vmol * 1e-30) ** (2 / 3) * 1e3
        if v and v > 0:
            out["packing_fraction_vdw"] = v / vmol
    if v and v > 0 and area and area > 0:
        out["sphericity_vdw"] = math.pi ** (1 / 3) * (6 * v) ** (2 / 3) / area
    ll = (4 * math.pi / 3) * a / vmol if vmol and a is not None and a > 0 else None
    ori = (4 * math.pi / 3) * orientation_A3_per_D2(t) * mu ** 2 / vmol if vmol and mu is not None else None
    if ll is not None:
        out["lorentz_lorenz_bosque"] = ll
        llc = min(ll, LL_CAP)
        n2 = (1 + 2 * llc) / (1 - llc)
        out["refractive_index_bosque"] = math.sqrt(n2)
    if ori is not None:
        out["dipole_orientation_ratio"] = ori
    if ll is not None and ori is not None:
        st = min(ll + ori, STATIC_CAP)
        eps = (1 + 2 * st) / (1 - st)
        a_h = (0.75 * kt * ((eps - 1) / (eps + 1)) ** 2
               + 3 * PLANCK * NU_E / (16 * math.sqrt(2)) * (n2 - 1) ** 2 / (n2 + 1) ** 1.5)
        out["hamaker_lifshitz_zJ"] = a_h * 1e21
        out["sigma_lifshitz_mN_m"] = a_h / (24 * math.pi * D0_M ** 2) * 1e3
    return {k: out[k] for k in COLUMNS if k in out}


def from_descriptors(d: Mapping, temperature_k: float = 298.15) -> dict[str, float]:
    """``groups`` from a molecule's descriptors: Bosque's α, the Boltzmann-averaged dipole (else the best
    conformer's), the best conformer's van der Waals volume and area, the packing molar volume."""
    mu = d.get("dipole_debye_boltz")
    if _num(mu) is None:
        mu = d.get("dipole_debye_best")
    return groups(d.get("polarizability_bosque_A3"), mu, d.get("vdw_volume_A3_best"), d.get("vdw_area_A2_best"),
                  d.get("molar_volume_packing_cm3"), temperature_k)
