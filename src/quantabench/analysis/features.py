"""Feature presets for the Analyze table (DESC-3, the feature builder): ready-made formula columns.

The dimensionless liquid-state groups of ``chem.liquid`` (Lorentz-Lorenz α/v_m and the refractive index it implies,
the dipole orientation ratio, the packing fraction, the sphericity, the Lifshitz Hamaker constant and surface tension,
the kT/v_m^⅔ scale), a few plain ratios of α, μ, V and A, and the reduced target σ* = σ / (kT/v_m^⅔) of any column.

A preset is a formula with placeholders, resolved against the columns the table has:

    {alpha} {mu} {V} {A} {Vm}   inputs: the first of ``INPUTS``' candidates that is a column (a paired run's
                                ``gas_...`` name when the plain one is not there)
    {LL_bosque} ...             another preset: its descriptor column when the run has it (runs since DESC-3
                                write the groups), else that preset's own formula column, added first

so a run folder made before the groups were descriptors gets them as formula columns, and a new run reuses its
descriptors. ``plan`` returns the formula columns to add, dependencies first; ``reduced_target`` the column σ* of a
target. Everything is a formula column: the user sees and can edit the expression (the constants are those of
``chem.liquid`` at 298.15 K). ``tests/test_liquid.py`` checks the formulas against ``chem.liquid.groups``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Collection, Iterable
from dataclasses import dataclass

from ..chem import liquid as lq

#: placeholder -> candidate column names, the first present wins
INPUTS: dict[str, tuple[str, ...]] = {
    "alpha": ("polarizability_bosque_A3",),
    "mu": ("dipole_debye_boltz", "dipole_debye_best", "dipole_debye"),
    "V": ("vdw_volume_A3_best",),
    "A": ("vdw_area_A2_best",),
    "Vm": ("molar_volume_packing_cm3",),
}
INPUT_TEXT = {"alpha": "Bosque polarizability (polarizability_bosque_A3)",
              "mu": "a dipole (dipole_debye_boltz, dipole_debye_best or dipole_debye)",
              "V": "the van der Waals volume (vdw_volume_A3_best)", "A": "the van der Waals area (vdw_area_A2_best)",
              "Vm": "the packing molar volume (molar_volume_packing_cm3)"}

GROUPS = "Dimensionless groups"
RATIOS = "Ratios of α, μ, V and A"


def _c(x: float) -> str:
    return f"{x:.7g}"


_T = 298.15
_KT_ZJ = lq.BOLTZMANN * _T * 1e21
_LL = (4 * math.pi / 3) / lq.A3_PER_CM3_MOL                                   # LL = this · α / V_m
_ORI = (4 * math.pi / 3) * lq.orientation_A3_per_D2(_T) / lq.A3_PER_CM3_MOL   # ORI = this · μ² / V_m
_PSI = math.pi ** (1 / 3) * 6 ** (2 / 3)
_DISP_ZJ = 3 * lq.PLANCK * lq.NU_E / (16 * math.sqrt(2)) * 1e21
_SIG_PER_ZJ = 1e-21 / (24 * math.pi * lq.D0_M ** 2) * 1e3                       # mN/m per zJ
_SCALE = lq.BOLTZMANN * _T / (lq.A3_PER_CM3_MOL * 1e-30) ** (2 / 3) * 1e3        # kT/v_m^⅔ = this / V_m^⅔


@dataclass(frozen=True)
class Preset:
    name: str                  # the formula column it makes
    label: str                 # menu text
    group: str
    template: str
    note: str
    descriptor: str = ""       # the run-table column with the same value (runs since DESC-3)


PRESETS: tuple[Preset, ...] = (
    Preset("LL_bosque", "Lorentz-Lorenz (n²−1)/(n²+2) = (4π/3) α / v_m", GROUPS,
           f"{_c(_LL)}*{{alpha}}/{{Vm}}",
           "Clausius-Mossotti / Lorentz-Lorenz from Bosque's α and the packing molar volume: dispersion cohesion "
           "(chem.liquid)", "lorentz_lorenz_bosque"),
    Preset("n_bosque", "Refractive index n from Lorentz-Lorenz", GROUPS,
           f"sqrt((1 + 2*min({{LL_bosque}}, {lq.LL_CAP}))/(1 - min({{LL_bosque}}, {lq.LL_CAP})))",
           "n from LL (LL capped at 0.95)", "refractive_index_bosque"),
    Preset("ORI", "Dipole orientation ratio μ²/(3·4πε₀kT) per v_m", GROUPS,
           f"{_c(_ORI)}*{{mu}}**2/{{Vm}}",
           "the Debye orientation term at 298.15 K: dipole-dipole cohesion against kT", "dipole_orientation_ratio"),
    Preset("eta_vdw", "Packing fraction η = V_vdW / v_m", GROUPS,
           f"{{V}}/({_c(lq.A3_PER_CM3_MOL)}*{{Vm}})", "how much of the liquid the van der Waals volume fills",
           "packing_fraction_vdw"),
    Preset("psi_vdw", "Sphericity ψ = π^⅓ (6V)^⅔ / A", GROUPS,
           f"{_c(_PSI)}*{{V}}**(2/3)/{{A}}", "1 for a sphere, lower for chains and flat molecules", "sphericity_vdw"),
    Preset("A_H_zJ", "Hamaker constant, Lifshitz (zJ)", GROUPS,
           f"{_c(0.75 * _KT_ZJ)}*((min({{LL_bosque}} + {{ORI}}, {lq.STATIC_CAP})*3/(2 + min({{LL_bosque}} + {{ORI}}, "
           f"{lq.STATIC_CAP})))**2) + {_c(_DISP_ZJ)}*({{n_bosque}}**2 - 1)**2/({{n_bosque}}**2 + 1)**1.5",
           "Israelachvili's approximation of Lifshitz theory across vacuum: a zero-frequency (Debye ε) and a "
           "dispersion term (ν_e = 3e15 s⁻¹)", "hamaker_lifshitz_zJ"),
    Preset("sigma_H", "Surface tension, Lifshitz σ = A_H/(24π D₀²) (mN/m)", GROUPS,
           f"{_c(_SIG_PER_ZJ)}*{{A_H_zJ}}", "a no-fit estimate (D₀ = 0.165 nm): dispersion only, misses H-bonding",
           "sigma_lifshitz_mN_m"),
    Preset("sigma_scale", "Reduced-σ unit kT / v_m^⅔ (mN/m)", GROUPS,
           f"{_c(_SCALE)}/{{Vm}}**(2/3)", "σ* = σ / this: the surface energy of one molecular cross-section in kT",
           "sigma_scale_kT_mN_m"),
    Preset("alpha_per_V", "α / V_vdW (polarizability per volume, dimensionless)", RATIOS, "{alpha}/{V}",
           "Bosque's α over the van der Waals volume (both Å³)"),
    Preset("mu2_per_V", "μ² / V_vdW (D²/Å³)", RATIOS, "{mu}**2/{V}", "dipole strength per volume"),
    Preset("mu2_per_alpha", "μ² / α (D²/Å³): orientation vs induction", RATIOS, "{mu}**2/{alpha}",
           "how much of the cohesion is dipolar rather than dispersive"),
    Preset("A_per_V23", "A / V^⅔ (shape, dimensionless)", RATIOS, "{A}/{V}**(2/3)",
           "surface per volume^⅔: 4.84 for a sphere, larger for chains"),
)
BY_NAME = {p.name: p for p in PRESETS}
_PLACE = re.compile(r"\{(\w+)\}")


class FeatureError(ValueError):
    """A preset cannot be made from the table's columns (the message names what is missing)."""


def _input(key: str, aliases: Collection[str]) -> str | None:
    for cand in INPUTS[key]:
        if cand in aliases:
            return cand
    for cand in INPUTS[key]:                  # a paired run: gas_..., else the first phase that has it
        prefixed = sorted(a for a in aliases if a.endswith("_" + cand))
        if prefixed:
            return next((a for a in prefixed if a.startswith("gas_")), prefixed[0])
    return None


def plan(names: Iterable[str], aliases: Collection[str], formulas: Collection[str] = ()) -> list[tuple[str, str, str]]:
    """The formula columns to add for the presets ``names``: [(name, expression, note)], dependencies first. A
    preset whose descriptor column is in ``aliases`` adds nothing (its users read the descriptor); one whose name
    is already a formula column (``formulas``) is reused as it is. Raises ``FeatureError`` naming missing inputs."""
    aliases = set(aliases)
    formulas = set(formulas)
    out: list[tuple[str, str, str]] = []
    made: dict[str, str] = {}                 # preset -> the column its users read

    def need(name: str, stack: tuple[str, ...] = ()) -> str:
        if name in made:
            return made[name]
        p = BY_NAME[name]
        if p.descriptor and p.descriptor in aliases:
            made[name] = p.descriptor
            return p.descriptor
        if name in formulas:
            made[name] = name
            return name
        if name in stack:
            raise FeatureError(f"circular preset {' -> '.join(stack + (name,))}")
        values: dict[str, str] = {}
        missing: list[str] = []
        for key in dict.fromkeys(_PLACE.findall(p.template)):
            if key in INPUTS:
                got = _input(key, aliases)
                if got is None:
                    missing.append(INPUT_TEXT[key])
                else:
                    values[key] = got
            else:
                values[key] = need(key, stack + (name,))
        if missing:
            raise FeatureError(f"{p.label} needs {', '.join(missing)}: no such column in the table")
        if name in aliases:
            raise FeatureError(f"{name} is already the name of a column that is not this preset")
        expr = _PLACE.sub(lambda m: values[m.group(1)], p.template)
        out.append((name, expr, p.note))
        made[name] = name
        return name

    for n in names:
        if n not in BY_NAME:
            raise FeatureError(f"no preset {n!r}")
        need(n)
    return out


def status(name: str, aliases: Collection[str], formulas: Collection[str] = ()) -> str:
    """"" when ``plan([name])`` would add a column; else why not (already there, or what is missing)."""
    p = BY_NAME[name]
    if p.descriptor and p.descriptor in aliases:
        return f"already a column: {p.descriptor}"
    if name in formulas:
        return f"already a formula column: {name}"
    try:
        plan([name], aliases, formulas)
    except FeatureError as exc:
        return str(exc)
    return ""


def reduced_target(target: str, aliases: Collection[str], formulas: Collection[str] = (),
                   name: str | None = None) -> list[tuple[str, str, str]]:
    """The formula columns for σ* = ``target`` / (kT/v_m^⅔): the scale first when the table needs it, then
    ``<target>_star`` (or ``name``)."""
    if target not in aliases:
        raise FeatureError(f"no column {target}")
    steps = plan(["sigma_scale"], aliases, formulas)
    scale = steps[-1][0] if steps else ("sigma_scale_kT_mN_m" if "sigma_scale_kT_mN_m" in aliases else "sigma_scale")
    col = name or f"{target}_star"
    if col in aliases or col in formulas:
        raise FeatureError(f"{col} is already the name of a column")
    return steps + [(col, f"{target}/{scale}",
                     f"reduced surface tension σ* = {target} · v_m^⅔ / kT (dimensionless; 298.15 K)")]
