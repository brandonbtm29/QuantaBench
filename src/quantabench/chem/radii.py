"""Van der Waals radii for molecular areas and volumes.

Two sets, by atomic number (Å):

- ``bondi`` (the default): Bondi's radii (*J. Phys. Chem.* 1964, 68, 441) completed by Mantina et al.
  (*J. Phys. Chem. A* 2009, 113, 5806), as ``ase.data.vdw_radii`` tabulates them, 2.0 Å where neither has a
  value. The 3D viewer's space-filling style and the charge-ESP surfaces (``chem.surfaces``) use them too.
- ``spartan``: the table Spartan '24/'26 shows under Preferences > VDW Radii (its defaults, read off the
  dialog, 2026-11-05). Spartan's CPK area and volume (``CPKAREA``, ``CPKVOLUME``) are the area and volume of
  the union of atom spheres with these radii: on the C3A B3LYP/C-PCM documents the grid volume of that union
  reproduces ``CPKVOLUME`` to 0.1 % and the area ``CPKAREA`` to 0.5 % (``cpk_volume_A3``/``cpk_area_A2`` in
  ``chem.descriptors``). They differ most from Bondi for carbon (1.92 vs 1.70 Å), boron (2.00 vs 1.92) and the
  heavier halogens; hydrogen, nitrogen and oxygen are Bondi's.

``radii(numbers, name, overrides)`` gives per-atom radii; ``overrides`` (element symbol or atomic number ->
Å) replaces single elements, as Spartan's table can be edited (``settings.SurfaceSettings.radii_overrides``).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np

#: Radius for an element a set has no value for (Å).
FALLBACK_A = 2.0

#: Spartan '24/'26 Preferences > VDW Radii, defaults (Å), by atomic number (1-111).
SPARTAN: dict[int, float] = {
    1: 1.2000, 2: 1.1700, 3: 1.1025, 4: 1.2348, 5: 2.0000, 6: 1.9200, 7: 1.5500, 8: 1.5200, 9: 1.5138,
    10: 1.4598, 11: 1.3428, 12: 1.3590, 13: 2.0250, 14: 2.1000, 15: 1.8657, 16: 1.8153, 17: 1.7820, 18: 1.7406,
    19: 1.7154, 20: 1.5291, 21: 1.4832, 22: 1.4283, 23: 1.4148, 24: 1.3608, 25: 1.3320, 26: 1.3104, 27: 1.2924,
    28: 1.2753, 29: 1.5723, 30: 1.2438, 31: 1.9719, 32: 1.9260, 33: 1.9035, 34: 1.8918, 35: 1.8855, 36: 1.8630,
    37: 1.8513, 38: 1.5291, 39: 1.6389, 40: 1.5057, 41: 1.4058, 42: 1.4238, 43: 1.3734, 44: 1.3491, 45: 1.3338,
    46: 1.3041, 47: 1.4166, 48: 1.2816, 49: 2.0079, 50: 1.9764, 51: 1.9890, 52: 2.0115, 53: 2.0250, 54: 1.9818,
    55: 2.0331, 56: 1.6668, 57: 1.5849, 58: 1.6002, 59: 1.6227, 60: 1.6092, 61: 1.5957, 62: 1.5840, 63: 1.5723,
    64: 1.5156, 65: 1.5525, 66: 1.5426, 67: 1.5336, 68: 1.5264, 69: 1.5183, 70: 1.5102, 71: 1.6380, 72: 1.4139,
    73: 1.4265, 74: 1.3815, 75: 1.3293, 76: 1.4040, 77: 1.2780, 78: 1.2393, 79: 1.4814, 80: 1.2168, 81: 1.9566,
    82: 1.9332, 83: 1.9665, 84: 2.1195, 85: 2.1375, 86: 2.1438, 87: 2.2050, 88: 1.6551, 89: 1.5651, 90: 1.5282,
    91: 1.5408, 92: 1.5273, 93: 1.5408, 94: 1.5408, 95: 1.5219, 96: 1.4967, 97: 1.5021, 98: 1.4904, 99: 1.4850,
    100: 1.4787, 101: 1.4733, 102: 1.4616, 103: 1.4562, 104: 1.4000, 105: 1.4000, 106: 1.4000, 107: 1.3000,
    108: 1.4000, 109: 1.3000, 110: 1.2000, 111: 1.5000,
}

SETS = ("bondi", "spartan")
LABELS = {"bondi": "Bondi (Mantina where Bondi has none)", "spartan": "Spartan '24/'26 defaults"}


def _bondi(z: int) -> float:
    from ase.data import vdw_radii

    if 0 < z < len(vdw_radii):
        r = float(vdw_radii[z])
        if np.isfinite(r) and r > 0:
            return r
    return FALLBACK_A


def default_radius(z: int, name: str = "bondi") -> float:
    """The set's radius for atomic number ``z`` (Å), :data:`FALLBACK_A` where it has none."""
    name = (name or "bondi").lower()
    if name == "spartan":
        return SPARTAN.get(int(z), FALLBACK_A)
    if name == "bondi":
        return _bondi(int(z))
    raise ValueError(f"unknown radii set {name!r} (known: {', '.join(SETS)})")


def _override_key(key) -> int | None:
    from ase.data import atomic_numbers

    if isinstance(key, int):
        return key
    s = str(key).strip()
    if s.isdigit():
        return int(s)
    return atomic_numbers.get(s[:1].upper() + s[1:].lower())


def normalize_overrides(overrides: Mapping | None) -> dict[int, float]:
    """``{atomic number: Å}`` from a mapping keyed by symbol ("C", "cl") or number; non-positive or unreadable
    entries are dropped."""
    out: dict[int, float] = {}
    for k, v in (overrides or {}).items():
        z = _override_key(k)
        try:
            r = float(v)
        except (TypeError, ValueError):
            continue
        if z and r > 0:
            out[z] = r
    return out


def radii(numbers: Iterable[int], name: str = "bondi", overrides: Mapping | None = None) -> np.ndarray:
    """Per-atom radii (Å) of ``numbers`` from set ``name``, with per-element ``overrides``."""
    over = normalize_overrides(overrides)
    return np.array([over.get(int(z), default_radius(int(z), name)) for z in numbers], dtype=float)


def table(name: str = "bondi", max_z: int = 111) -> dict[int, float]:
    """``{atomic number: radius}`` of a set for the settings dialog."""
    return {z: default_radius(z, name) for z in range(1, max_z + 1)}
