"""Molecular surface panel: density isosurface colored by ESP, from QM cubes or point charges.

CONTRACT (implemented by the surfaces builder):

    install(win: MainWindow) -> None
        Adds a "Surfaces" dock (right area, tabified with Pipeline) and Analyze/View menu actions. For the
        selected molecule (``win.current_result()`` / ``win.resultChanged``) offers: surface kind (density
        0.002 au from ``res.qm["files"]["density"]``, or vdW/SAS from charges), color by (ESP, none), opacity,
        color range (symmetric auto / fixed kcal/mol), colormap (bwr, coolwarm), and shows the GIPF
        descriptors. Rendering goes into ``win.viewer.plotter`` as an extra actor named "qb_surface" that is
        removed when the molecule changes; it must not break picking (set ``pickable=False``).
"""

from __future__ import annotations


def install(win) -> None:
    """No-op until implemented."""
