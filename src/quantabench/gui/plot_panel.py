"""The Plot dock (UI-3): a scatter plot of two numeric columns of the Table dock (pyqtgraph).

- **X and Y** are any numeric columns of the table (shown or not: formula columns, descriptors, the best energy in the
  display unit); **Colour** by a third (viridis, missing values grey). The points are the rows the table shows, so
  the table's filter filters the plot too; rows without both values are left out.
- **Fit line:** the least-squares line with its slope, intercept, R² and n (``formula.linear_fit``).
- **Click a point** to select that molecule (in the Molecules dock and the table); the selected molecule is ringed.
  Hovering a point shows its molecule and values.
- The choices are part of the table's state (``TablePanel.plot_settings``), kept with it.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QCheckBox, QComboBox, QDockWidget, QGridLayout, QLabel, QVBoxLayout, QWidget

from ..analysis import formula as fm
from .selection import ACCENT_HEX as ACCENT  # the selection's amber, as in the 2D and 3D views

_LIGHT = {"bg": "#fcfcfb", "fg": "#0b0b0b", "muted": "#52514e", "point": "#2a78d6", "line": "#eb6834",
          "missing": "#a8a7a2"}
_DARK = {"bg": "#1a1a19", "fg": "#f0efec", "muted": "#c3c2b7", "point": "#3987e5", "line": "#d95926",
         "missing": "#6b6a66"}
NONE = ""


class PlotPanel(QWidget):
    def __init__(self, win, table, parent=None):
        super().__init__(parent)
        import pyqtgraph as pg

        self._pg = pg
        self.win = win
        self.table = table
        self.model = table.model
        self._ids = np.empty(0, dtype=object)
        self._xy = np.empty((0, 2))
        self.fit_info: tuple[float, float, float, int] | None = None
        self._filling = False

        self.x_box, self.y_box, self.c_box = QComboBox(), QComboBox(), QComboBox()
        for box, name in ((self.x_box, "plot_x"), (self.y_box, "plot_y"), (self.c_box, "plot_color")):
            box.setObjectName(name)
            box.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            box.setMinimumContentsLength(8)
            box.currentIndexChanged.connect(self._on_choice)
        self.fit_box = QCheckBox("Fit line")
        self.fit_box.setToolTip("Least-squares line y = a·x + b, with R² and n")
        self.fit_box.setChecked(True)
        self.fit_box.toggled.connect(self._on_choice)
        self.stats = QLabel()
        self.stats.setObjectName("plot_stats")
        self.stats.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.stats.setWordWrap(True)
        self.legend = QLabel()
        self.legend.setStyleSheet("color: gray;")
        self.legend.setWordWrap(True)

        self.plot = pg.PlotWidget()
        self.plot.setMinimumSize(200, 160)
        pi = self.plot.getPlotItem()
        pi.showGrid(x=True, y=True, alpha=0.15)
        pi.setMenuEnabled(False)
        self.scatter = pg.ScatterPlotItem(size=7, pxMode=True, hoverable=True, tip=self._tip,
                                          hoverSize=10, hoverPen=pg.mkPen(ACCENT, width=1.5))
        self.scatter.sigClicked.connect(self._on_clicked)
        pi.addItem(self.scatter)
        self.line = pg.PlotDataItem()
        pi.addItem(self.line)
        self.marker = pg.ScatterPlotItem(size=16, pxMode=True, brush=None, pen=pg.mkPen(ACCENT, width=2.5))
        self.marker.setZValue(10)
        pi.addItem(self.marker)

        grid = QGridLayout()             # two rows: the dock stays narrow enough for the 3D view
        grid.setContentsMargins(4, 2, 4, 2)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(2)
        for r, c, label, box in ((0, 0, "X", self.x_box), (0, 2, "Y", self.y_box), (1, 0, "Colour", self.c_box)):
            grid.addWidget(QLabel(label), r, c)
            grid.addWidget(box, r, c + 1)
        grid.addWidget(self.fit_box, 1, 2, 1, 2)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        lay.addLayout(grid)
        lay.addWidget(self.plot, 1)
        lay.addWidget(self.stats)
        lay.addWidget(self.legend)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self.redraw)
        for sig in (self.model.refreshed, self.model.layoutChanged, self.model.modelReset):
            sig.connect(self.schedule)
        win.mol_list.currentIdChanged.connect(self._mark)
        win.displayChanged.connect(self.schedule)
        self._apply_theme()
        self.fill_boxes()
        self.redraw()

    # ------------------------------------------------------------------ choices
    def _settings(self) -> dict:
        return self.table.plot_settings

    def fill_boxes(self) -> None:
        """The numeric columns in the X, Y and Colour boxes, keeping the choices (by column key)."""
        cols = self.model.numeric_columns()
        st = self._settings()
        keys = [c.key for c in cols]
        want_x = st.get("x") if st.get("x") in keys else next(
            (k for k in ("mol_weight", "molar_volume_packing_cm3") if k in keys),
            next((k for k in keys if k not in ("seq",)), ""))
        want_y = st.get("y") if st.get("y") in keys else next(
            (k for k in ("parachor_sugden", "polarizability_bosque_A3", "best_energy_eV") if k in keys and k != want_x),
            next((k for k in keys if k not in ("seq", want_x)), ""))
        want_c = st.get("color") if st.get("color") in keys else NONE
        self._filling = True
        try:
            for box, want, none in ((self.x_box, want_x, False), (self.y_box, want_y, False),
                                    (self.c_box, want_c, True)):
                box.clear()
                if none:
                    box.addItem("None", NONE)
                for c in cols:
                    title = self.model.title(c.key)
                    box.addItem(title if title == c.alias else f"{title} ({c.alias})", c.key)
                    box.setItemData(box.count() - 1, f"In formulas: {c.alias}", Qt.ItemDataRole.ToolTipRole)
                i = box.findData(want)
                box.setCurrentIndex(i if i >= 0 else 0)
            self.fit_box.setChecked(bool(st.get("fit", True)))
        finally:
            self._filling = False

    def _on_choice(self, *_args) -> None:
        if self._filling:
            return
        st = self._settings()
        st.update(x=self.x_box.currentData() or "", y=self.y_box.currentData() or "",
                  color=self.c_box.currentData() or NONE, fit=self.fit_box.isChecked())
        self.table.save_state()
        self.redraw()

    def set_axes(self, x: str, y: str, color: str = NONE, fit: bool | None = None) -> None:
        """Choose the columns by key (tests and scripts); a column the table does not have yet is kept as the choice
        and used once it has it."""
        st = self._settings()
        st.update(x=x, y=y, color=color or NONE)
        if fit is not None:
            st["fit"] = bool(fit)
        self.table.save_state()
        self.fill_boxes()
        self.redraw()

    # ------------------------------------------------------------------ drawing
    def schedule(self, *_args) -> None:
        if not self._timer.isActive():
            self._timer.start()

    def redraw(self) -> None:
        self._timer.stop()
        pg = self._pg
        keys = [c.key for c in self.model.numeric_columns()]
        if [self.x_box.itemData(i) for i in range(self.x_box.count())] != keys:
            self.fill_boxes()
        c = self._colors()
        xk, yk, ck = self.x_box.currentData(), self.y_box.currentData(), self.c_box.currentData()
        pi = self.plot.getPlotItem()
        self.fit_info = None
        if not xk or not yk or self.model.column(xk) is None or self.model.column(yk) is None:
            self.scatter.setData([])
            self.line.setData([], [])
            self.marker.setData([])
            self._ids = np.empty(0, dtype=object)
            self.stats.setText("No numeric columns yet: build molecules or open a run folder.")
            self.legend.setText("")
            return
        view = self.model.view_positions()
        x = np.asarray(self.model.display_values(xk), dtype=float)[view]
        y = np.asarray(self.model.display_values(yk), dtype=float)[view]
        ids = np.array(self.model.view_ids(), dtype=object)
        ok = np.isfinite(x) & np.isfinite(y)
        x, y, ids = x[ok], y[ok], ids[ok]
        self._ids = ids
        self._xy = np.column_stack([x, y]) if len(x) else np.empty((0, 2))
        brush = pg.mkBrush(c["point"])
        self.legend.setText("")
        brushes = None
        if ck and self.model.column(ck) is not None:
            v = np.asarray(self.model.display_values(ck), dtype=float)[view][ok]
            fin = np.isfinite(v)
            if fin.any():
                lo, hi = float(np.min(v[fin])), float(np.max(v[fin]))
                norm = np.zeros_like(v)
                if hi > lo:
                    norm[fin] = (v[fin] - lo) / (hi - lo)
                cmap = pg.colormap.get("viridis")
                colors = cmap.mapToQColor(np.clip(norm, 0, 1))
                missing = QColor(c["missing"])
                brushes = [pg.mkBrush(col if f else missing) for col, f in zip(colors, fin)]
                self.legend.setText(f"Colour: {self.model.title(ck)} from {lo:.4g} (dark violet) to {hi:.4g} "
                                    f"(yellow){'; grey: no value' if not fin.all() else ''}")
        self.scatter.setData(x=x, y=y, data=ids, brush=brushes if brushes is not None else brush,
                             pen=pg.mkPen(c["bg"], width=0.5))
        pi.setLabel("bottom", self.model.title(xk))
        pi.setLabel("left", self.model.title(yk))
        n = len(x)
        text = f"{n:,} point{'s' if n != 1 else ''}"
        hidden = int((~ok).sum())
        if hidden:
            text += f" ({hidden:,} rows without both values)"
        if self.fit_box.isChecked():
            fit = fm.linear_fit(x, y)
            self.fit_info = fit
            if fit is not None:
                slope, icpt, r2, nfit = fit
                xs = np.array([x.min(), x.max()])
                self.line.setData(xs, slope * xs + icpt, pen=pg.mkPen(c["line"], width=2))
                text = (f"y = {slope:.5g}·x {'+' if icpt >= 0 else '-'} {abs(icpt):.5g}    R² = {r2:.4f}    "
                        f"n = {nfit:,}") + ("" if not hidden else f"    ({hidden:,} rows without both values)")
            else:
                self.line.setData([], [])
        else:
            self.line.setData([], [])
        self.stats.setText(text)
        self._mark(self.win.mol_list.current_id())

    def _mark(self, ident: str = "") -> None:
        hit = np.flatnonzero(self._ids == ident) if ident and len(self._ids) else []
        if len(hit):
            self.marker.setData(pos=self._xy[hit[:1]])
        else:
            self.marker.setData([])

    def _tip(self, x, y, data) -> str:
        xk, yk = self.x_box.currentData(), self.y_box.currentData()
        return f"{data}\n{self.model.title(xk)}: {x:.5g}\n{self.model.title(yk)}: {y:.5g}"

    def _on_clicked(self, _item, points, _ev=None) -> None:
        if points is None or not len(points):
            return
        ident = points[0].data()
        if ident:
            self.table.select_molecule(str(ident))

    def point_pos(self, ident: str):
        """Data coordinates of ``ident``'s point (None when it is not plotted)."""
        hit = np.flatnonzero(self._ids == ident) if len(self._ids) else []
        return tuple(self._xy[hit[0]]) if len(hit) else None

    # ------------------------------------------------------------------ theme
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

    def changeEvent(self, event) -> None:  # noqa: N802 - Qt's name
        if event.type() in (QEvent.Type.PaletteChange, QEvent.Type.ApplicationPaletteChange) and hasattr(self, "plot"):
            self._apply_theme()
            self.schedule()
        super().changeEvent(event)


def install(win) -> None:
    """The "plot" dock; needs the Table dock (``table_panel.install`` first) and pyqtgraph."""
    from .table_panel import add_view_toggle

    table = getattr(win, "table_panel", None)
    if table is None:
        raise RuntimeError("the Table dock is not installed")
    panel = PlotPanel(win, table)
    dock = QDockWidget("Plot", win)
    dock.setObjectName("plot")
    dock.setWidget(panel)
    win.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
    dock.hide()                                   # the workspaces show it (Analyze)
    dock.visibilityChanged.connect(lambda on: table.model.set_active("plot", on))
    add_view_toggle(win, dock, "Plot", "Scatter plot of two table columns, colour by a third, fit line")
    win.plot_dock = dock
    win.plot_panel = panel
