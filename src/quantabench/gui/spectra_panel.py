"""IR spectrum panel (pyqtgraph) with experimental overlay and normal-mode animation.

CONTRACT (implemented by the spectra/interop builder):

    install(win: MainWindow) -> None
        Adds a "Spectra" dock (bottom area, tabified with Results) and a View menu toggle. Listens to the
        selected molecule: ``win.current_result()`` returns the MoleculeResult or None; ``win.resultChanged``
        (Signal(object)) fires when the selection or its results change. Plots ``res.vibrations`` (and
        ``res.qm["vibrations"]`` when present, as a second trace) as sticks + Lorentzian broadening (FWHM spin
        box, scale factor), wavenumber axis reversed, clicking near a peak selects that mode and animates it in
        ``win.viewer`` (MoleculeViewer3D) (``set_positions(..., emit=False)`` with a QTimer, amplitude slider, stop restores
        the geometry). "Load experimental..." overlays a CSV (x, y) or JCAMP-DX file, normalized.
"""

from __future__ import annotations


def install(win) -> None:
    """No-op until implemented."""
