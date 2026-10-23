"""Compatibility shim: the 3D viewer now lives in :mod:`quantabench.gui.viewer3d`.

``from quantabench.gui.viewer import MoleculeViewer`` keeps working (``MoleculeViewer`` is
``MoleculeViewer3D``), as do the bond and element helpers.
"""

from __future__ import annotations

from .viewer3d import (
    COLOR_MODES,
    LABEL_MODES,
    MODES,
    STYLES,
    MoleculeViewer3D,
    atom_radii,
    bonds_from_distances,
    bonds_from_rdkit,
    element_counts,
)

MoleculeViewer = MoleculeViewer3D

__all__ = ["COLOR_MODES", "LABEL_MODES", "MODES", "STYLES", "MoleculeViewer", "MoleculeViewer3D", "atom_radii",
           "bonds_from_distances", "bonds_from_rdkit", "element_counts"]
