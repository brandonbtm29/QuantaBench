"""Shared atom selection, so the 2D and 3D views stay in sync.

Both viewers hold a reference to one ``SelectionModel``. A click in either view calls
``select``/``toggle``. Both views listen to ``changed`` and redraw their highlights.
Indices are always **full-molecule atom indices** (explicit hydrogens included), the same indices
used by RDKit ``mol.GetAtoms()`` and ASE ``Atoms``.

The order of selection is kept, because measurements depend on it:
2 atoms = distance, 3 = angle (middle atom is the vertex), 4 = dihedral.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QObject, Signal

#: Selection accent shared by the 2D and 3D views, so a picked atom looks the same in both (amber).
ACCENT_RGB = (1.0, 0.69, 0.0)
ACCENT_HEX = "#ffaf00"   # what RDKit writes for ACCENT_RGB


class SelectionModel(QObject):
    changed = Signal(list)          # ordered list of selected atom indices

    def __init__(self, parent=None):
        super().__init__(parent)
        self._atoms: list[int] = []

    # ------------------------------------------------------------------ state
    @property
    def atoms(self) -> list[int]:
        return list(self._atoms)

    def select(self, indices, additive: bool = False) -> None:
        new = list(self._atoms) if additive else []
        for i in indices:
            i = int(i)
            if i not in new:
                new.append(i)
        self._set(new)

    def toggle(self, index: int) -> None:
        index = int(index)
        new = list(self._atoms)
        if index in new:
            new.remove(index)
        else:
            new.append(index)
        self._set(new)

    def clear(self) -> None:
        self._set([])

    def _set(self, new: list[int]) -> None:
        if new != self._atoms:
            self._atoms = new
            self.changed.emit(list(new))


def measure(positions: np.ndarray, atoms: list[int]) -> tuple[str, float] | None:
    """Measurement for the last 2-4 selected atoms: ('distance', Å) | ('angle', deg) | ('dihedral', deg)."""
    if len(atoms) < 2:
        return None
    sel = atoms[-4:] if len(atoms) >= 4 else atoms
    p = np.asarray(positions, dtype=float)[sel]
    if len(sel) == 2:
        return "distance", float(np.linalg.norm(p[1] - p[0]))
    if len(sel) == 3:
        a, b = p[0] - p[1], p[2] - p[1]
        cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        return "angle", float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
    b0, b1, b2 = p[0] - p[1], p[2] - p[1], p[3] - p[2]
    b1n = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    x = np.dot(v, w)
    y = np.dot(np.cross(b1n, v), w)
    return "dihedral", float(np.degrees(np.arctan2(y, x)))


def format_measurement(m: tuple[str, float] | None) -> str:
    if m is None:
        return ""
    kind, value = m
    return f"{kind}: {value:.3f} Å" if kind == "distance" else f"{kind}: {value:.2f}°"
