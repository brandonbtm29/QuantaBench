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

Details
-------
- Broadening: each stick at ``scale·ν̃`` with intensity I becomes ``I γ² / ((x - scale·ν̃)² + γ²)``, γ = FWHM/2, so
  an isolated band peaks at its stick height (km/mol). Normalized, the curve equals ``vibrations.ir_spectrum``.
  Imaginary modes are listed in the table (as ``123.4i``) but not plotted.
- Mode sets: "computed" (``res.vibrations``) and "qm" (``res.qm["vibrations"]``); each keeps its own scale factor
  (default ``vib["scale_factor"]``). The table, the click selection and the animation follow the set chosen in
  the "Modes" box.
- Animation: ``base + A sin(2πt/T) d`` with d scaled so that its largest atomic displacement is 1, i.e. the
  amplitude slider sets the largest atomic excursion (Å); T = 1 s on screen whatever the frequency. The base is
  the frequency geometry ``vib["positions_A"]`` when present, else the shown structure (atom counts must match).
  Stop restores the coordinates shown before Play. The animation stops when the molecule (or the conformer in
  the 3D view) changes, when the panel is hidden or closed, and when the window closes. Each frame is one
  ``set_positions``, whose cost is the synchronous render: with the viewer's SSAA that is ~170 ms for 25 atoms on
  software OpenGL (llvmpipe), so the timer interval stretches to 1.5x the frame time to keep the UI responsive,
  and ``viewer.set_interactive(True/False)`` (cheaper anti-aliasing while animating) is called when the viewer
  has it.
- Experimental overlay (``chem.spectra_io``): normalized to the maximum of the first computed trace between
  400 and 4000 cm⁻¹ (to 1 when there is none); kept when another molecule is selected, until "Clear".
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PySide6.QtCore import QElapsedTimer, QEvent, QObject, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

VIEW_RANGE = (400.0, 4000.0)       # default x range (shown 4000 -> 400)
GRID_STEP = 1.0                    # cm⁻¹
GRID_MIN = 50.0
FPS = 25
PERIOD_S = 1.0                     # one vibration per second on screen
DEFAULT_AMPLITUDE = 0.25           # Å, largest atomic excursion

# Reference categorical palette (light, dark): computed, QM; experimental uses the secondary text ink.
_LIGHT = {"bg": "#fcfcfb", "fg": "#0b0b0b", "muted": "#52514e", "s1": "#2a78d6", "s2": "#eb6834"}
_DARK = {"bg": "#1a1a19", "fg": "#f0efec", "muted": "#c3c2b7", "s1": "#3987e5", "s2": "#d95926"}


# ======================================================================================= data
@dataclass(eq=False)
class ModeSet:
    """One set of normal modes (a ``VibrationalResult.to_dict(include_modes=True)``-shaped dict)."""

    key: str                       # "computed" | "qm"
    label: str                     # legend text
    vib: dict
    scale: float = 1.0
    freqs: np.ndarray = field(init=False)
    ints: np.ndarray | None = field(init=False)
    disp: np.ndarray | None = field(init=False)
    base: np.ndarray | None = field(init=False)

    def __post_init__(self) -> None:
        v = self.vib
        self.freqs = np.asarray(v.get("frequencies_cm1") or [], dtype=float)
        ints = v.get("ir_intensities_km_mol")
        self.ints = None if ints is None or len(ints) != len(self.freqs) else np.asarray(ints, dtype=float)
        d = v.get("displacements")
        disp = None if d is None else np.asarray(d, dtype=float)
        self.disp = disp if disp is not None and disp.ndim == 3 and disp.shape[0] == len(self.freqs) \
            and disp.shape[2] == 3 else None
        pos = v.get("positions_A")
        base = None if pos is None else np.asarray(pos, dtype=float).reshape(-1, 3)
        self.base = base if base is not None and self.disp is not None and len(base) == self.disp.shape[1] else None
        self.scale = float(v.get("scale_factor") or 1.0)

    @property
    def n_modes(self) -> int:
        return len(self.freqs)

    @property
    def uniform(self) -> bool:
        """No IR intensities: every mode gets the same weight."""
        return self.ints is None

    def weights(self) -> np.ndarray:
        return np.ones_like(self.freqs) if self.ints is None else self.ints

    def column(self, name: str, i: int) -> float | None:
        vals = self.vib.get(name)
        if vals is None or i >= len(vals) or vals[i] is None:
            return None
        v = float(vals[i])
        return None if math.isnan(v) else v


def _method_label(method: str) -> str:
    from ..chem import engines

    info = engines.all_engines().get(method)
    return info.label if info is not None else method


def mode_sets(res) -> list[ModeSet]:
    """The plottable mode sets of a MoleculeResult (computed first, then QM)."""
    out: list[ModeSet] = []
    if res is None:
        return out
    vib = getattr(res, "vibrations", None)
    if vib and vib.get("frequencies_cm1"):
        out.append(ModeSet("computed", _method_label(vib.get("method") or res.method), vib))
    qm = getattr(res, "qm", None) or {}
    qv = qm.get("vibrations")
    if qv and qv.get("frequencies_cm1"):
        out.append(ModeSet("qm", "QM " + str(qm.get("level") or qv.get("method") or ""), qv))
    return out


def broaden(freqs, weights, fwhm: float, scale: float, x: np.ndarray) -> np.ndarray:
    """Σ Iₖ γ² / ((x - scale·ν̃ₖ)² + γ²) over the real modes (peak height Iₖ for an isolated band)."""
    nu = np.asarray(freqs, dtype=float)
    w = np.asarray(weights, dtype=float)
    keep = nu > 0
    c, w = nu[keep] * scale, w[keep]
    g2 = (0.5 * fwhm) ** 2
    y = np.zeros_like(x, dtype=float)
    for i in range(0, len(c), 256):                       # bounded memory for large molecules
        y += (w[None, i:i + 256] * g2 / ((x[:, None] - c[None, i:i + 256]) ** 2 + g2)).sum(1)
    return y


def sticks_xy(freqs, weights, scale: float, norm: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Vertex pairs (x, 0)-(x, I) for ``connect='pairs'``."""
    nu = np.asarray(freqs, dtype=float)
    keep = nu > 0
    x = np.repeat(nu[keep] * scale, 2)
    y = np.zeros_like(x)
    y[1::2] = np.asarray(weights, dtype=float)[keep] / norm
    return x, y


def _fmt_freq(f: float) -> str:
    return f"{abs(f):.1f}i" if f < 0 else f"{f:.1f}"


# ======================================================================================= widget
class SpectraPanel(QWidget):
    """IR spectra of the selected molecule, the mode table and the mode animation."""

    modeSelected = Signal(int)

    def __init__(self, win, parent=None):
        super().__init__(parent)
        import pyqtgraph as pg

        self._pg = pg
        self.win = win
        self.result = None
        self.sets: list[ModeSet] = []
        self.active = 0                       # index into self.sets for table / click / animation
        self.selected: int | None = None      # mode index in the active set
        self.experimental = None              # spectra_io.Spectrum
        self.frame_times_ms: list[float] = []
        self._items: list = []
        self._anim_atoms = None
        self._anim_base = None
        self._anim_dir = None
        self._restore = None
        self._timer = QTimer(self)
        self._timer.setInterval(int(1000 / FPS))
        self._timer.timeout.connect(self._tick)
        self._clock = QElapsedTimer()
        self._syncing = False

        # ---- controls
        self.set_box = QComboBox()
        self.set_box.setToolTip("Which normal modes the table, the click selection and the animation use")
        self.set_box.currentIndexChanged.connect(self._on_set_changed)
        self.fwhm = QDoubleSpinBox()
        self.fwhm.setRange(1.0, 300.0)
        self.fwhm.setValue(20.0)
        self.fwhm.setSingleStep(2.0)
        self.fwhm.setSuffix(" cm⁻¹")
        self.fwhm.setToolTip("Full width at half maximum of the Lorentzian bands")
        self.fwhm.valueChanged.connect(self._replot)
        self.scale = QDoubleSpinBox()
        self.scale.setDecimals(4)
        self.scale.setRange(0.5, 1.5)
        self.scale.setSingleStep(0.005)
        self.scale.setValue(1.0)
        self.scale.setToolTip("Frequency scale factor of the selected mode set (default: the method's)")
        self.scale.valueChanged.connect(self._on_scale)
        self.normalize = QCheckBox("Normalize")
        self.normalize.setToolTip("Scale every trace to a maximum of 1 (off: IR intensity in km/mol)")
        self.normalize.toggled.connect(self._replot)
        self.load_btn = QPushButton("Load experimental...")
        self.load_btn.setToolTip("Overlay a measured spectrum: CSV/TSV (two columns) or JCAMP-DX (.jdx, .dx)")
        self.load_btn.clicked.connect(lambda: self.load_experimental())
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setToolTip("Remove the experimental overlay")
        self.clear_btn.clicked.connect(self.clear_experimental)
        self.clear_btn.setEnabled(False)
        self.export_btn = QPushButton("Export...")
        self.export_btn.setToolTip("Save the broadened spectrum as shown (CSV)")
        self.export_btn.clicked.connect(lambda: self.export_csv())

        top = QHBoxLayout()
        top.setContentsMargins(4, 2, 4, 2)
        top.addWidget(QLabel("Modes"))
        top.addWidget(self.set_box, 1)
        top.addWidget(QLabel("FWHM"))
        top.addWidget(self.fwhm)
        top.addWidget(QLabel("Scale"))
        top.addWidget(self.scale)
        top.addWidget(self.normalize)
        top.addStretch(1)
        for b in (self.load_btn, self.clear_btn, self.export_btn):
            top.addWidget(b)

        # ---- plot
        self.plot = pg.PlotWidget()
        self.plot.setMinimumHeight(120)
        pi = self.plot.getPlotItem()
        pi.invertX(True)
        pi.setLabel("bottom", "Wavenumber (cm⁻¹)")
        pi.showGrid(x=True, y=True, alpha=0.15)
        pi.setMenuEnabled(False)
        pi.getViewBox().setAutoVisible(y=True)           # y range from the bands in view, not a 290 cm⁻¹ torsion
        self.legend = pi.addLegend(offset=(0, 8))
        # top center: 2500-2000 cm⁻¹ is the emptiest part of most IR spectra (on the reversed axis)
        self.legend.anchor(itemPos=(0.5, 0.0), parentPos=(0.5, 0.0), offset=(0, 8))
        self.plot.setXRange(*VIEW_RANGE, padding=0)
        self.marker = pg.InfiniteLine(angle=90, movable=False)
        self.marker.setVisible(False)
        self.marker.setZValue(10)
        pi.addItem(self.marker, ignoreBounds=True)
        self.plot.scene().sigMouseClicked.connect(self._on_plot_click)

        # ---- mode table + animation
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["#", "ν (cm⁻¹)", "Scaled", "IR (km/mol)", "μ (amu)"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.currentCellChanged.connect(self._on_table_row)
        self.info = QLabel("")
        self.info.setWordWrap(True)
        self.info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.play_btn = QPushButton("Play")
        self.play_btn.setToolTip("Animate the selected mode in the 3D view")
        self.play_btn.clicked.connect(self.play)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setToolTip("Stop and restore the geometry shown before")
        self.stop_btn.clicked.connect(lambda: self.stop())
        self.stop_btn.setEnabled(False)
        self.amp = QSlider(Qt.Orientation.Horizontal)
        self.amp.setRange(5, 100)                        # 0.05 - 1.00 Å
        self.amp.setValue(int(round(DEFAULT_AMPLITUDE * 100)))
        self.amp.setToolTip("Largest atomic displacement of the animation")
        self.amp_label = QLabel()
        self.amp.valueChanged.connect(self._update_amp_label)
        self._update_amp_label()
        anim = QHBoxLayout()
        anim.addWidget(self.play_btn)
        anim.addWidget(self.stop_btn)
        anim.addWidget(QLabel("Amplitude"))
        anim.addWidget(self.amp, 1)
        anim.addWidget(self.amp_label)
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(self.table, 1)
        rl.addWidget(self.info)
        rl.addLayout(anim)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(self.status)
        ll.addWidget(self.plot, 1)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(left)
        self.splitter.addWidget(right)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 2)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(top)
        lay.addWidget(self.splitter, 1)
        self._apply_theme()
        self._update_enabled()

    # ================================================================== public API
    @property
    def playing(self) -> bool:
        return self._timer.isActive()

    @property
    def active_set(self) -> ModeSet | None:
        return self.sets[self.active] if 0 <= self.active < len(self.sets) else None

    def set_result(self, res) -> None:
        """Show the vibrations of ``res`` (a MoleculeResult or None). Stops a running animation."""
        self.stop(restore=True)
        self.result = res
        self.sets = mode_sets(res)
        self.active = 0
        self.selected = None
        self.set_box.blockSignals(True)
        self.set_box.clear()
        for ms in self.sets:
            self.set_box.addItem(ms.label, ms.key)
        self.set_box.blockSignals(False)
        if res is None:
            self.status.setText("Select a built molecule to see its IR spectrum.")
        elif not self.sets:
            self.status.setText("No frequencies for this molecule: tick \"Frequencies + thermochemistry\" in the "
                                "Properties stage and run it, or import an output file with frequencies.")
        else:
            self.status.setText("")
        self.status.setVisible(bool(self.status.text()))
        self._load_set()
        if self.experimental is not None or self.sets:
            self.plot.setXRange(*VIEW_RANGE, padding=0)

    def select_set(self, index: int) -> None:
        if 0 <= index < len(self.sets):
            self.set_box.setCurrentIndex(index)

    def select_mode(self, index: int | None) -> None:
        """Select mode ``index`` of the active set (table row, plot marker, info line)."""
        ms = self.active_set
        if ms is None or index is None or not 0 <= index < ms.n_modes:
            self.selected = None
            self.marker.setVisible(False)
            self.info.setText("")
            self._update_enabled()
            return
        restart = self.playing
        if restart:
            self.stop(restore=True)
        self.selected = int(index)
        if not self._syncing:
            self._syncing = True
            try:
                self.table.setCurrentCell(self.selected, 0)
                self.table.scrollToItem(self.table.item(self.selected, 0))
            finally:
                self._syncing = False
        f = float(ms.freqs[self.selected])   # with a NumPy scalar, f > 0 is a numpy.bool: Qt refuses it under NumPy 2
        self.marker.setValue(abs(f) * ms.scale)
        self.marker.setVisible(f > 0)
        bits = [f"Mode {self.selected + 1}: {_fmt_freq(f)} cm⁻¹"]
        if ms.scale != 1.0:
            bits[0] += f" (×{ms.scale:.4g} → {_fmt_freq(f * ms.scale)})"
        if ms.ints is not None:
            bits.append(f"{ms.ints[self.selected]:.1f} km/mol")
        mu, k = ms.column("reduced_masses_amu", self.selected), ms.column("force_constants_mdyn_A", self.selected)
        if mu is not None:
            bits.append(f"μ {mu:.3f} amu")
        if k is not None:
            bits.append(f"k {k:.3f} mdyn/Å")
        if f < 0:
            bits.append("imaginary")
        self.info.setText(", ".join(bits))
        self._update_enabled()
        self.modeSelected.emit(self.selected)
        if restart:
            self.play()

    def select_near(self, x: float) -> int | None:
        """Select the real mode of the active set whose (scaled) frequency is closest to ``x``, when it lies
        within max(3 FWHM, 5 % of the visible range); returns the mode index or None."""
        ms = self.active_set
        if ms is None or not ms.n_modes:
            return None
        pos = ms.freqs * ms.scale
        real = np.where(ms.freqs > 0)[0]
        if not len(real):
            return None
        i = int(real[np.argmin(np.abs(pos[real] - x))])
        (x0, x1), _ = self.plot.getPlotItem().vb.viewRange()
        tol = max(3.0 * self.fwhm.value(), 0.05 * abs(x1 - x0))
        if abs(pos[i] - x) > tol:
            return None
        self.select_mode(i)
        return i

    def curve(self, key: str = "computed") -> tuple[np.ndarray, np.ndarray] | None:
        """(x, y) of a computed trace as displayed (normalized or km/mol), or None."""
        ms = next((m for m in self.sets if m.key == key), None)
        if ms is None:
            return None
        x, y, _norm = self._trace(ms)
        return x, y

    def _trace(self, ms: ModeSet) -> tuple[np.ndarray, np.ndarray, float]:
        """Grid, displayed curve and the divisor applied to it (1 in km/mol mode)."""
        x = self._grid()
        y = broaden(ms.freqs, ms.weights(), self.fwhm.value(), ms.scale, x)
        norm = float(y.max()) if self._normalized() and y.size and y.max() > 0 else 1.0
        return x, y / norm, norm

    def experimental_curve(self) -> tuple[np.ndarray, np.ndarray] | None:
        """(x, y) of the experimental overlay as displayed (normalized to the first computed trace)."""
        spec = self.experimental
        if spec is None:
            return None
        x, y = np.asarray(spec.x, float), np.asarray(spec.y, float)
        win = (x >= VIEW_RANGE[0]) & (x <= VIEW_RANGE[1])
        ref = np.abs(y[win]).max() if win.any() else (np.abs(y).max() if y.size else 0.0)
        target = 1.0
        if self.sets:
            cx, cy = self.curve(self.sets[0].key)
            inside = (cx >= VIEW_RANGE[0]) & (cx <= VIEW_RANGE[1])
            target = float(cy[inside].max()) if inside.any() and cy[inside].max() > 0 else 1.0
        return x, (y / ref * target if ref > 0 else y)

    def load_experimental(self, path: str | Path | None = None) -> bool:
        from ..chem import spectra_io

        if path is None:
            path, _ = QFileDialog.getOpenFileName(
                self, "Load experimental spectrum", "",
                "Spectra (*.csv *.tsv *.txt *.dat *.jdx *.dx *.jcamp);;JCAMP-DX (*.jdx *.dx *.jcamp);;"
                "Tables (*.csv *.tsv *.txt *.dat);;All files (*)")
            if not path:
                return False
        try:
            spec = spectra_io.read_spectrum(path)
        except Exception as exc:
            QMessageBox.warning(self, "Load experimental spectrum", f"{Path(path).name}: {exc}")
            return False
        self.experimental = spec
        for w in spec.warnings:
            self._log(f"{Path(path).name}: {w}")
        self.clear_btn.setEnabled(True)
        self._replot()
        return True

    def clear_experimental(self) -> None:
        self.experimental = None
        self.clear_btn.setEnabled(False)
        self._replot()

    def export_csv(self, path: str | Path | None = None) -> Path | None:
        """Write the broadened spectrum as shown: ``wavenumber_cm-1,intensity_<normalized|km_mol>`` (a
        ``qm_intensity_...`` column too when a QM trace is shown)."""
        from ..chem import spectra_io

        if not self.sets:
            QMessageBox.information(self, "Export spectrum", "No computed spectrum to export.")
            return None
        if path is None:
            stem = self.result.input.id if self.result is not None else "spectrum"
            path, _ = QFileDialog.getSaveFileName(self, "Export spectrum", f"{stem}_ir.csv", "CSV (*.csv)")
            if not path:
                return None
        unit = "normalized" if self._normalized() else "km_mol"
        cols: dict[str, np.ndarray] = {}
        for ms in self.sets:
            x, y = self.curve(ms.key)
            cols.setdefault("wavenumber_cm-1", x)
            cols[("" if ms.key == "computed" else f"{ms.key}_") + f"intensity_{unit}"] = y
        out = spectra_io.write_table(path, cols)
        self._log(f"IR spectrum written to {out}")
        return out

    # ------------------------------------------------------------------ animation
    def play(self) -> bool:
        """Animate the selected mode of the active set in ``win.viewer``; False (with a message) if it cannot."""
        ms = self.active_set
        viewer = getattr(self.win, "viewer", None)
        if ms is None or self.selected is None or viewer is None:
            return False
        if ms.disp is None:
            self._say("This result has no normal-mode displacements to animate.")
            return False
        atoms = viewer.atoms
        if atoms is None or getattr(viewer, "plotter", None) is None:
            self._say("The 3D view is not available.")
            return False
        d = ms.disp[self.selected]
        if len(atoms) != len(d):
            self._say(f"The 3D view shows {len(atoms)} atoms but the modes have {len(d)}: cannot animate.")
            return False
        self.stop(restore=True)
        base = ms.base if ms.base is not None else atoms.positions.copy()
        dmax = float(np.sqrt((d**2).sum(1)).max())
        if dmax <= 0:
            return False
        self._restore = atoms.positions.copy()
        self._anim_atoms = atoms
        self._anim_base = np.asarray(base, float).copy()
        self._anim_dir = d / dmax
        self.frame_times_ms = []
        self._interactive(viewer, True)
        self._clock.start()
        self._timer.setInterval(int(1000 / FPS))
        self._timer.start()
        self._update_enabled()
        self._tick()
        return True

    @staticmethod
    def _interactive(viewer, on: bool) -> None:
        """Cheaper anti-aliasing while animating, when the viewer offers it (``set_interactive``)."""
        fn = getattr(viewer, "set_interactive", None)
        if callable(fn):
            fn(on)

    def stop(self, restore: bool = True) -> None:
        """Stop the animation; restore the coordinates shown before Play when the 3D view still shows them."""
        was = self._timer.isActive()
        self._timer.stop()
        viewer = getattr(self.win, "viewer", None)
        closing = getattr(self.win, "_closing", False)
        if was and viewer is not None and not closing:
            self._interactive(viewer, False)
        if was and restore and viewer is not None and self._anim_atoms is not None \
                and viewer.atoms is self._anim_atoms and self._restore is not None and not closing:
            viewer.set_positions(self._restore, emit=False)
        self._anim_atoms = None
        self._restore = None
        self._update_enabled()

    def _tick(self) -> None:
        viewer = getattr(self.win, "viewer", None)
        if viewer is None or getattr(self.win, "_closing", False) or viewer.atoms is not self._anim_atoms:
            self.stop(restore=False)       # another structure is shown now (or the window closes)
            return
        t = self._clock.elapsed() / 1000.0
        a = self.amplitude * math.sin(2.0 * math.pi * t / PERIOD_S)
        start = time.perf_counter()
        viewer.set_positions(self._anim_base + a * self._anim_dir, emit=False)
        self.frame_times_ms.append(1000.0 * (time.perf_counter() - start))
        if len(self.frame_times_ms) > 500:
            del self.frame_times_ms[:250]
        # A slow renderer (software OpenGL) gets a longer interval, so input events still get through;
        # the phase follows the clock, so the period stays 1 s and only the frame rate drops.
        recent = sorted(self.frame_times_ms[-5:])[len(self.frame_times_ms[-5:]) // 2]
        self._timer.setInterval(max(int(1000 / FPS), int(1.5 * recent)))

    @property
    def amplitude(self) -> float:
        return self.amp.value() / 100.0

    # ================================================================== internals
    def _normalized(self) -> bool:
        return self.normalize.isChecked() or any(ms.uniform for ms in self.sets)

    def _grid(self) -> np.ndarray:
        top = VIEW_RANGE[1]
        for ms in self.sets:
            if ms.n_modes and ms.freqs.max() > 0:
                top = max(top, float(ms.freqs.max() * ms.scale) + 5 * self.fwhm.value())
        return np.arange(GRID_MIN, top + GRID_STEP, GRID_STEP)

    def _load_set(self) -> None:
        ms = self.active_set
        self.scale.blockSignals(True)
        self.scale.setValue(ms.scale if ms else 1.0)
        self.scale.blockSignals(False)
        uniform = any(m.uniform for m in self.sets)
        self.normalize.setEnabled(not uniform)
        if uniform:
            self.normalize.setToolTip("No IR intensities in this result: all modes have the same weight")
        self._fill_table()
        self._replot()
        if ms is not None and ms.n_modes:
            # the strongest band in the default view (else the strongest real mode, else the first mode)
            pos = ms.freqs * ms.scale
            real = np.where(ms.freqs > 0)[0]
            shown = real[(pos[real] >= VIEW_RANGE[0]) & (pos[real] <= VIEW_RANGE[1])]
            pool = shown if len(shown) else real
            w = ms.weights()
            self.select_mode(int(pool[np.argmax(w[pool])]) if len(pool) else 0)
        else:
            self.select_mode(None)

    def _fill_table(self) -> None:
        ms = self.active_set
        self._syncing = True
        try:
            self.table.setRowCount(0)
            if ms is None:
                return
            self.table.setRowCount(ms.n_modes)
            for i, f in enumerate(ms.freqs):
                mu = ms.column("reduced_masses_amu", i)
                vals = [str(i + 1), _fmt_freq(f), _fmt_freq(f * ms.scale),
                        "" if ms.ints is None else f"{ms.ints[i]:.1f}", "" if mu is None else f"{mu:.3f}"]
                for col, v in enumerate(vals):
                    item = QTableWidgetItem(v)
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                    self.table.setItem(i, col, item)
        finally:
            self._syncing = False

    def _replot(self, *_args) -> None:
        pg = self._pg
        pi = self.plot.getPlotItem()
        for it in self._items:
            pi.removeItem(it)
        self._items = []
        self.legend.clear()
        c = self._colors()
        norm = self._normalized()
        for ms, color in zip(self.sets, (c["s1"], c["s2"])):
            x, y, divisor = self._trace(ms)
            label = ms.label + (f" (×{ms.scale:.4g})" if ms.scale != 1.0 else "")
            sx, sy = sticks_xy(ms.freqs, ms.weights(), ms.scale, divisor)
            sticks = pg.PlotDataItem(sx, sy, connect="pairs", pen=pg.mkPen(color, width=1.5))
            line = pg.PlotDataItem(x, y, pen=pg.mkPen(color, width=2), name=label)
            for it in (sticks, line):
                pi.addItem(it)
                self._items.append(it)
        exp = self.experimental_curve()
        if exp is not None:
            ex, ey = exp
            pen = pg.mkPen(c["muted"], width=1.5)
            name = f"Experimental: {self.experimental.title}"
            if self.experimental.peaks:
                sx = np.repeat(ex, 2)
                sy = np.zeros_like(sx)
                sy[1::2] = ey
                item = pg.PlotDataItem(sx, sy, connect="pairs", pen=pen, name=name)
            else:
                item = pg.PlotDataItem(ex, ey, pen=pen, name=name)
            item.setZValue(-1)
            pi.addItem(item)
            self._items.append(item)
        uniform = any(ms.uniform for ms in self.sets)
        pi.setLabel("left", "Relative intensity (no IR intensities)" if uniform else
                    "Normalized intensity" if norm else "IR intensity (km/mol)")
        self.legend.setVisible(bool(self._items))      # it is the only place that names the method
        pi.enableAutoRange(axis="y")
        if self.selected is not None and self.active_set is not None:
            self.marker.setValue(abs(self.active_set.freqs[self.selected]) * self.active_set.scale)

    def _colors(self) -> dict:
        dark = self.palette().color(QPalette.ColorRole.Window).lightness() < 128
        return _DARK if dark else _LIGHT

    def _apply_theme(self) -> None:
        pg = self._pg
        c = self._colors()
        self.plot.setBackground(c["bg"])
        pi = self.plot.getPlotItem()
        for name in ("bottom", "left"):
            ax = pi.getAxis(name)
            ax.setPen(pg.mkPen(c["muted"]))
            ax.setTextPen(pg.mkPen(c["fg"]))
        self.legend.setLabelTextColor(c["fg"])
        self.legend.setBrush(pg.mkBrush(c["bg"] + "d9"))      # ~85 % opaque surface: bands stay faintly visible
        self.marker.setPen(pg.mkPen(c["fg"], width=1, style=Qt.PenStyle.DashLine))
        self._replot()

    def changeEvent(self, event) -> None:
        if event.type() in (QEvent.Type.PaletteChange, QEvent.Type.ApplicationPaletteChange) \
                and hasattr(self, "legend"):
            self._apply_theme()
        super().changeEvent(event)

    def _update_enabled(self) -> None:
        ms = self.active_set
        can = ms is not None and ms.disp is not None and self.selected is not None
        self.play_btn.setEnabled(can and not self.playing)
        self.play_btn.setToolTip("Animate the selected mode in the 3D view" if ms is None or ms.disp is not None
                                 else "No normal-mode displacements in this result")
        self.stop_btn.setEnabled(self.playing)
        self.export_btn.setEnabled(bool(self.sets))
        for w in (self.fwhm, self.scale, self.set_box):
            w.setEnabled(bool(self.sets))

    @Slot(bool)
    def on_dock_visibility(self, visible: bool) -> None:
        """The dock was closed or its tab left: stop (and restore) the animation."""
        if not visible:
            self.stop(restore=True)

    def _update_amp_label(self, *_args) -> None:
        self.amp_label.setText(f"{self.amplitude:.2f} Å")

    def _on_set_changed(self, index: int) -> None:
        if 0 <= index < len(self.sets) and index != self.active:
            self.stop(restore=True)
            self.active = index
            self.selected = None
            self._load_set()

    def _on_scale(self, value: float) -> None:
        ms = self.active_set
        if ms is None:
            return
        ms.scale = float(value)
        self._fill_table()
        self._replot()
        if self.selected is not None:
            self.select_mode(self.selected)

    def _on_table_row(self, row, _col, _prow, _pcol) -> None:
        if self._syncing or row < 0:
            return
        self._syncing = True
        try:
            self.select_mode(row)
        finally:
            self._syncing = False

    def _on_plot_click(self, ev) -> None:
        if ev.button() != Qt.MouseButton.LeftButton:
            return
        vb = self.plot.getPlotItem().vb
        if not vb.sceneBoundingRect().contains(ev.scenePos()):
            return
        self.select_near(vb.mapSceneToView(ev.scenePos()).x())

    def _log(self, msg: str) -> None:
        log = getattr(self.win, "log", None)
        if callable(log):
            log(msg)

    def _say(self, msg: str) -> None:
        self.info.setText(msg)
        self._log(msg)


class _CloseWatcher(QObject):
    """Stops the animation when the main window starts closing (before the 3D view is torn down)."""

    def __init__(self, panel: SpectraPanel, win):
        super().__init__(panel)
        self.panel = panel
        win.installEventFilter(self)

    def eventFilter(self, obj, event) -> bool:
        if event.type() == QEvent.Type.Close:
            self.panel.stop(restore=True)
        return False


def _menu(win, title: str):
    for action in win.menuBar().actions():
        if action.menu() is not None and action.text().replace("&", "") == title:
            return action.menu()
    return None


def install(win) -> None:
    panel = SpectraPanel(win)
    dock = QDockWidget("Spectra", win)
    dock.setObjectName("spectra")
    dock.setWidget(panel)
    win.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)
    results = getattr(win, "results_dock", None)
    if results is not None:
        win.tabifyDockWidget(results, dock)
        results.raise_()
    toggle = dock.toggleViewAction()
    toggle.setText("Spectra")
    toggle.setToolTip("IR spectrum, normal modes and their animation")
    view = _menu(win, "View")
    if view is not None:
        actions = view.actions()
        after = results.toggleViewAction() if results is not None else None
        idx = actions.index(after) if after in actions else -1
        if 0 <= idx < len(actions) - 1:
            view.insertAction(actions[idx + 1], toggle)
        else:
            view.addAction(toggle)
    dock.visibilityChanged.connect(panel.on_dock_visibility)     # a slot: disconnected when the panel goes
    panel._close_watcher = _CloseWatcher(panel, win)
    win.spectra_dock = dock
    win.spectra_panel = panel
    win.resultChanged.connect(panel.set_result)
    panel.set_result(win.current_result())
