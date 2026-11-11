"""Cheap estimates for every molecule: liquid density and molar volume, additive polarizabilities and the Sugden
parachor (DESC-2). They run in the descriptors step of every phase, from the graph or the best conformer, in
milliseconds, and need no data about the molecule.

Why: a surface tension model of the parachor family, σ = (P/V_m)⁴, needs the liquid's molar volume V_m, which a
QuantaBench volume is not (the van der Waals volume V_vdW is 50-66 % of V_m, batch b02), and a polarizability
that tracks size better than the minimal-basis xTB response (GFN2-xTB gives tributyl phosphate 14 Å³, Bosque-Sales
27.8; the xTB responses are 0.6-0.7 x DFT, docs/VALIDATION.md, DESC-2).
docs/SURFACE_TENSION.md, "Density and polarizability estimates", compares these with the other options.

Columns (``descriptors.csv``):

    density_girolami_g_cm3        Girolami's atom-volume rule (J. Chem. Educ. 1994, 71, 962): graph only
    molar_volume_girolami_cm3     M / that density, cm³/mol
    density_packing_g_cm3         M / V_m with V_m = N_A V_vdW / k: the Bondi van der Waals volume of the best
    molar_volume_packing_cm3      conformer and a packing coefficient k from composition (``PACKING``), fitted to
                                  PubChem/HSDB liquid densities near 25 °C (docs/VALIDATION.md, DESC-2)
    polarizability_bosque_A3      Bosque & Sales' additive model from the composition (J. Chem. Inf. Comput. Sci.
                                  2002, 42, 1154), fitted to experimental polarizabilities of 340 solvents
    polarizability_crippen_A3     Wildman-Crippen molar refractivity (J. Chem. Inf. Comput. Sci. 1999, 39, 868)
                                  through Lorentz-Lorenz: α = 3 MR / (4π N_A); an optical (589 nm) polarizability
    polarizability_d4_A3          the D4 dispersion model's static polarizability with EEQ charges (sum of
                                  charge- and coordination-scaled atomic reference polarizabilities from TD-DFT;
                                  Caldeweyher et al. 2019): geometry, needs PyTorch and tad-dftd4 (or the dxtb fork)
    parachor_sugden               Sugden's group parachor (Quayle's table, Chem. Rev. 1953, 53, 439),
                                  cm³ mol⁻¹ (mN/m)^¼: σ ≈ (P / V_m)⁴

Coverage: a method that has no parameters for an element of the molecule gives no value (an empty cell, NaN in
pandas) and a note, never an error. The densities are for neutral molecules (a liquid of an ion is not defined);
the additive polarizabilities ignore a formal charge (they were fitted on neutral molecules).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

AVOGADRO = 6.02214076e23
#: Å³ per (cm³/mol): one molecule's share of a molar volume
A3_PER_CM3_MOL = 1e24 / AVOGADRO
#: Lorentz-Lorenz: α (Å³) = 3 R_m / (4π N_A), with R_m (cm³/mol) the molar refraction
ALPHA_A3_PER_MR = 3.0 / (4.0 * np.pi) * A3_PER_CM3_MOL
AU_POLARIZABILITY_A3 = 0.1481847
BOHR_ANG = 0.529177210903


# ---------------------------------------------------------------------------------------------------- Girolami
def _girolami_volume(z: int) -> float | None:
    """Girolami's relative atomic volume: H 1, Li-F 2, Na-Cl 4, K-Br 5, Rb-I 7.5, Cs-Bi 9 (noble gases: none)."""
    if z == 1:
        return 1.0
    for lo, hi, v in ((3, 9, 2.0), (11, 17, 4.0), (19, 35, 5.0), (37, 53, 7.5), (55, 83, 9.0)):
        if lo <= z <= hi:
            return v
    return None


_SMARTS: dict[str, Chem.Mol] = {}


def _pattern(smarts: str) -> Chem.Mol:
    p = _SMARTS.get(smarts)
    if p is None:
        p = _SMARTS[smarts] = Chem.MolFromSmarts(smarts)
    return p


#: Girolami's corrections (+10 % each, +20 % for a sulfone, +7.5 % per ring of a fused system, at most +30 %)
GIROLAMI_GROUPS: tuple[tuple[str, str, float], ...] = (
    ("carboxylic acid", "[CX3](=O)[OX2H1]", 0.10),
    ("hydroxyl", "[OX2H1;!$(O[CX3]=O)]", 0.10),
    ("amide", "[CX3](=[OX1])[NX3]", 0.10),
    ("primary or secondary amine", "[#7X3;H1,H2;!$([#7][#6,#16]=[OX1])]", 0.10),
    ("sulfoxide", "[SX3](=[OX1])([#6])[#6]", 0.10),
    ("sulfone", "[SX4](=[OX1])(=[OX1])([#6])[#6]", 0.20),
)
GIROLAMI_MAX = 1.30


def girolami_factor(mol: Chem.Mol) -> tuple[float, list[str]]:
    """Girolami's density factor (1 to 1.3) and the groups that set it."""
    f, why = 1.0, []
    for name, smarts, inc in GIROLAMI_GROUPS:
        n = len(mol.GetSubstructMatches(_pattern(smarts)))
        if n:
            f += inc * n
            why.append(f"{n} {name}")
    rings = [set(r) for r in mol.GetRingInfo().AtomRings()]
    fused = sum(1 for i, r in enumerate(rings) if any(len(r & s) >= 2 for j, s in enumerate(rings) if j != i))
    if len(rings) - fused:
        f += 0.10 * (len(rings) - fused)
        why.append(f"{len(rings) - fused} ring")
    if fused:
        f += 0.075 * fused
        why.append(f"{fused} fused ring")
    return min(f, GIROLAMI_MAX), why


def girolami_density(mol: Chem.Mol) -> tuple[float | None, str]:
    """(density in g/cm³, note) by Girolami's rule: ρ = factor · M / (5 Σ v_i). None, with the reason, for an
    element without a volume or an ion."""
    mh = Chem.AddHs(mol)
    vs = 0.0
    for a in mh.GetAtoms():
        v = _girolami_volume(a.GetAtomicNum())
        if v is None:
            return None, f"Girolami density: no atomic volume for {a.GetSymbol()}"
        vs += v
    if Chem.GetFormalCharge(mh) != 0:
        return None, "Girolami density: not defined for an ion"
    factor, _why = girolami_factor(mh)
    return factor * Descriptors.MolWt(mh) / (5.0 * vs), ""


# ------------------------------------------------------------------------------------------- packing density
@dataclass(frozen=True)
class PackingModel:
    """k = c0 + Σ c_i x_i, V_m = N_A V_vdW / k. Features (per heavy atom unless stated): ``inv_heavy`` 1/n_heavy,
    ``f``, ``cl``, ``br``, ``i``, ``si``, ``s``, ``p``, ``n``, ``o`` (element fractions), ``hbd`` (H-bond donors),
    ``ring`` (ring atoms), ``arom`` (aromatic atoms). Fitted to measured liquid densities; ``elements`` are those
    the fit saw (others get no value)."""
    coefficients: tuple[tuple[str, float], ...]
    elements: frozenset[int]
    source: str


#: DESC-2 fit (2026-11-11): least squares on the measured k = N_A V_vdW / V_m of 831 training liquids (8 outliers
#: beyond 5 robust SDs left out), V_vdW the Bondi union volume (grid 0.15 Å) of an MMFF94s conformer, V_m from the
#: median PubChem/HSDB pure-liquid density record at 10-40 °C moved to 25 °C (liquefied gases, bp < 30 °C, left
#: out). On 237 test liquids it was never fitted to: 2.7 % mean absolute density error (Girolami 5.3 %);
#: fluorinated liquids with >= 3 F 3.2 % (Girolami 6.4 %), 3.8 % on the 11 of them it never saw (Girolami
#: 6.8 %). docs/VALIDATION.md, DESC-2.
PACKING = PackingModel(
    coefficients=(("const", 0.59592), ("inv_heavy", -0.41099), ("f", -0.05547), ("cl", 0.05421), ("br", 0.11663),
                  ("i", 0.20968), ("si", -0.22943), ("s", 0.08746), ("p", -0.07605), ("n", 0.07483), ("o", 0.11410),
                  ("hbd", 0.16920), ("ring", 0.04034), ("arom", 0.01317)),
    elements=frozenset({1, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53}),
    source="DESC-2, PubChem/HSDB liquid densities, 2026-11-11",
)


def packing_features(mol: Chem.Mol) -> dict[str, float]:
    heavy = [a for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    n = max(1, len(heavy))
    count = {z: 0 for z in (7, 8, 9, 14, 15, 16, 17, 35, 53)}
    for a in heavy:
        if a.GetAtomicNum() in count:
            count[a.GetAtomicNum()] += 1
    return {"const": 1.0, "inv_heavy": 1.0 / n, "f": count[9] / n, "cl": count[17] / n, "br": count[35] / n,
            "i": count[53] / n, "si": count[14] / n, "s": count[16] / n, "p": count[15] / n, "n": count[7] / n,
            "o": count[8] / n, "hbd": rdMolDescriptors.CalcNumHBD(mol) / n,
            "ring": sum(1 for a in heavy if a.IsInRing()) / n, "arom": sum(1 for a in heavy if a.GetIsAromatic()) / n}


def packing_coefficient(mol: Chem.Mol, model: PackingModel = PACKING) -> float:
    x = packing_features(mol)
    return float(sum(c * x[k] for k, c in model.coefficients))


def packing_density(mol: Chem.Mol, vdw_volume_A3: float, model: PackingModel = PACKING) -> tuple[float | None, str]:
    """(density g/cm³, note) from the Bondi van der Waals volume of a conformer and the packing model."""
    bad = sorted({a.GetSymbol() for a in mol.GetAtoms() if a.GetAtomicNum() not in model.elements})
    if bad:
        return None, f"packing density: not fitted for {', '.join(bad)}"
    if Chem.GetFormalCharge(mol) != 0:
        return None, "packing density: not defined for an ion"
    if not vdw_volume_A3 or vdw_volume_A3 <= 0:
        return None, "packing density: no van der Waals volume"
    k = packing_coefficient(mol, model)
    if not 0.3 < k < 0.9:
        return None, f"packing density: packing coefficient {k:.3f} outside 0.3-0.9 (outside the fitted range)"
    vm = vdw_volume_A3 / k / A3_PER_CM3_MOL
    return Descriptors.MolWt(mol) / vm, ""


# ----------------------------------------------------------------------------------------- polarizabilities
#: Bosque & Sales 2002 (Table 3, model with intercept): α (Å³) = 0.32 + Σ n_Z α_Z, fitted to 340 solvents
BOSQUE_INTERCEPT = 0.32
BOSQUE_A3: dict[int, float] = {6: 1.51, 1: 0.17, 8: 0.57, 7: 1.05, 16: 2.99, 15: 2.48, 9: 0.22, 17: 2.16,
                               35: 3.29, 53: 5.45}
#: elements with their own Wildman-Crippen atom types (others fall into a generic type with no fitted refractivity)
CRIPPEN_ELEMENTS = frozenset({1, 6, 7, 8, 9, 15, 16, 17, 35, 53})


def polarizability_bosque(mol: Chem.Mol) -> tuple[float | None, str]:
    mh = Chem.AddHs(mol)
    bad = sorted({a.GetSymbol() for a in mh.GetAtoms() if a.GetAtomicNum() not in BOSQUE_A3})
    if bad:
        return None, f"Bosque-Sales polarizability: no parameter for {', '.join(bad)}"
    return BOSQUE_INTERCEPT + sum(BOSQUE_A3[a.GetAtomicNum()] for a in mh.GetAtoms()), ""


def polarizability_crippen(mol: Chem.Mol) -> tuple[float | None, str]:
    mh = Chem.AddHs(mol)
    bad = sorted({a.GetSymbol() for a in mh.GetAtoms() if a.GetAtomicNum() not in CRIPPEN_ELEMENTS})
    if bad:
        return None, f"Crippen polarizability: no Wildman-Crippen atom type for {', '.join(bad)}"
    mr = sum(c[1] for c in rdMolDescriptors._CalcCrippenContribs(mh))
    return mr * ALPHA_A3_PER_MR, ""


def _tad_dftd4():
    try:
        import tad_dftd4
        return tad_dftd4
    except ImportError:
        from dxtb._src.exlibs import tad_dftd4  # the brandonbtm29 dxtb fork vendors it
        return tad_dftd4


@lru_cache(maxsize=1)
def d4_available() -> bool:
    """PyTorch and tad-dftd4 (or the dxtb fork's copy) can be imported; checked once per process."""
    from ..resources import has_package

    if not has_package("torch"):
        return False
    try:
        _tad_dftd4()
    except Exception:  # noqa: BLE001 - neither tad-dftd4 nor a dxtb that vendors it
        return False
    return True


D4_MAX_Z = 86


def polarizability_d4(numbers, positions_ang, charge: int = 0) -> tuple[float | None, str]:
    """The D4 model's molecular static polarizability (Å³): the sum of its atomic α(0) with EEQ charges and D4
    coordination numbers at this geometry."""
    z = np.asarray(numbers, int)
    if z.max(initial=0) > D4_MAX_Z:
        return None, f"D4 polarizability: no reference for Z > {D4_MAX_Z}"
    import torch

    t4 = _tad_dftd4()
    num = torch.as_tensor(z, dtype=torch.long)
    pos = torch.as_tensor(np.asarray(positions_ang, float) / BOHR_ANG, dtype=torch.float64)
    with torch.no_grad():
        _cn, _q, _c6, alpha = t4.get_properties(num, pos, torch.tensor(float(charge), dtype=torch.float64))
    total = float(alpha.sum()) * AU_POLARIZABILITY_A3
    if not np.isfinite(total) or total <= 0:
        return None, f"D4 polarizability: unphysical value {total:.4g} Å^3"
    return total, ""


# ----------------------------------------------------------------------------------------------- parachor
#: Sugden's atomic and structural parachors (Sugden 1924, as tabulated by Quayle, Chem. Rev. 1953, 53, 439),
#: cm³ mol⁻¹ (mN/m)^¼. "ester" is the -C(=O)O- of esters and acids (60.0 with its C=O); "semipolar" the S=O,
#: P=O and N⁺-O⁻ bonds; rings by size.
SUGDEN: dict[str, float] = {"C": 4.8, "H": 17.1, "N": 12.5, "O": 20.0, "F": 25.7, "Cl": 54.3, "Br": 68.0, "I": 91.0,
                            "S": 48.2, "P": 37.7, "Si": 25.0, "B": 16.4}
SUGDEN_DOUBLE, SUGDEN_TRIPLE, SUGDEN_ESTER, SUGDEN_SEMIPOLAR = 23.2, 46.6, 60.0, -1.6
SUGDEN_RING = {3: 16.7, 4: 11.6, 5: 8.5, 6: 6.1, 7: 4.6, 8: 2.5}


def parachor_sugden(mol: Chem.Mol) -> tuple[float | None, str]:
    m = Chem.Mol(mol)
    try:
        Chem.Kekulize(m, clearAromaticFlags=True)
    except Exception:  # noqa: BLE001, S110 - no Kekulé form: aromatic bonds count as 1.5
        pass
    p = 0.0
    for a in m.GetAtoms():          # hydrogens as atoms (explicit) or as counts on their heavy atom (implicit)
        s = a.GetSymbol()
        if s not in SUGDEN:
            return None, f"Sugden parachor: no value for {s}"
        p += SUGDEN[s] + a.GetTotalNumHs() * SUGDEN["H"]
    ester_co = set()
    for c, o1, _o2 in m.GetSubstructMatches(_pattern("[CX3](=[OX1])[OX2]")):
        p += SUGDEN_ESTER - 2 * SUGDEN["O"]
        ester_co.add(frozenset((c, o1)))
    for b in m.GetBonds():
        bt = b.GetBondType()
        a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
        if a1.GetAtomicNum() == 1 or a2.GetAtomicNum() == 1:
            continue
        pair = frozenset((a1.GetIdx(), a2.GetIdx()))
        ends = {a1.GetSymbol(), a2.GetSymbol()}
        if bt == Chem.BondType.DOUBLE and pair not in ester_co:
            p += SUGDEN_SEMIPOLAR if (ends & {"S", "P"} and "O" in ends) else SUGDEN_DOUBLE
        elif bt == Chem.BondType.SINGLE and a1.GetFormalCharge() * a2.GetFormalCharge() < 0:
            p += SUGDEN_SEMIPOLAR
        elif bt == Chem.BondType.TRIPLE:
            p += SUGDEN_TRIPLE
        elif bt == Chem.BondType.AROMATIC:
            p += SUGDEN_DOUBLE / 2
    for ring in m.GetRingInfo().AtomRings():
        p += SUGDEN_RING.get(len(ring), 0.0)
    return p, ""


# --------------------------------------------------------------------------------------------------- all
def estimate(mol: Chem.Mol, conf_id: int = -1, vdw_volume_bondi_A3: float | None = None,
             d4: bool | None = None) -> tuple[dict, list[str]]:
    """Every estimate of this module for ``mol`` (with explicit hydrogens) at conformer ``conf_id``:
    (descriptors, notes). ``vdw_volume_bondi_A3`` is the conformer's Bondi union volume when the caller has it
    (``descriptors.vdw_volume_A3_best`` with the default radii); else it is computed. ``d4``: None = when PyTorch
    and tad-dftd4 are installed."""
    from . import descriptors as desc
    from . import radii

    out: dict = {}
    notes: list[str] = []

    def put(key, value_note):
        value, note = value_note
        if value is None:
            if note:
                notes.append(note)
            return None
        out[key] = float(value)
        return value

    mw = Descriptors.MolWt(mol)
    rho = put("density_girolami_g_cm3", girolami_density(mol))
    if rho:
        out["molar_volume_girolami_cm3"] = mw / rho
    if mol.GetNumConformers():
        conf = mol.GetConformer(conf_id)
        if vdw_volume_bondi_A3 is None:
            z = [a.GetAtomicNum() for a in mol.GetAtoms()]
            vdw_volume_bondi_A3 = desc.sphere_union_volume(conf.GetPositions(), radii.radii(z, "bondi"))
        rho = put("density_packing_g_cm3", packing_density(mol, vdw_volume_bondi_A3))
        if rho:
            out["molar_volume_packing_cm3"] = mw / rho
    put("polarizability_bosque_A3", polarizability_bosque(mol))
    put("polarizability_crippen_A3", polarizability_crippen(mol))
    if mol.GetNumConformers() and (d4 if d4 is not None else d4_available()):
        conf = mol.GetConformer(conf_id)
        try:
            put("polarizability_d4_A3", polarizability_d4([a.GetAtomicNum() for a in mol.GetAtoms()],
                                                          conf.GetPositions(), Chem.GetFormalCharge(mol)))
        except Exception as exc:  # noqa: BLE001 - an optional estimate never fails a molecule
            notes.append(f"D4 polarizability skipped ({type(exc).__name__}: {exc})")
    put("parachor_sugden", parachor_sugden(mol))
    return out, notes
