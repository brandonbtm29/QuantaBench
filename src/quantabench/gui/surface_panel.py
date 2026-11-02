"""Molecular surface panel: density isosurface colored by ESP, from QM cubes or point charges.

CONTRACT (implemented by the surfaces builder):

    install(win: MainWindow) -> None
        Adds a "Surfaces" dock (right area, tabified with Pipeline) and Analyze/View menu actions. For the
        selected molecule (``win.current_result()`` / ``win.resultChanged``) offers: surface kind (density
        0.002 au from ``res.qm["files"]["density"]``, or vdW/SAS from charges), color by (ESP, none), opacity,
        color range (symmetric auto / fixed kcal/mol), colormap (bwr, coolwarm), and shows the GIPF
        descriptors. Rendering goes into ``win.viewer.plotter`` as an extra actor named "qb_surface" that is
        removed when the molecule changes; it must not break picking (set ``pickable=False``).

How it fits in: the dock's toggle is View > Surfaces panel (found by the menu's title). The panel owns one
actor, :data:`ACTOR`, in the 3D view's renderer, and exposes itself as ``win.surface_panel``. The viewer's own scene
rebuilds (style, colors, another conformer) leave foreign actors alone and emit ``sceneRebuilt``; the panel
then checks whether the surface still matches the displayed atoms and recomputes or removes it. The viewer
picks atoms by exact ray-sphere intersection in NumPy, not through VTK's pickers, so a surface in front of
the atoms never takes a click (the actor is still added with ``pickable=False`` for VTK pickers).

Sources: the QM density isosurface (``res.qm["files"]["density"]``, ESP from ``["esp"]``) when the QM stage
wrote cubes; otherwise a vdW or solvent-accessible surface with the ESP of the atomic point charges (the
displayed conformer's charges, else the best conformer's, else the QM stage's MBIS/Lowdin/Mulliken charges,
else RDKit Gasteiger charges). Meshes are computed on a worker thread (``gui.workers.FunctionWorker``) unless the
molecule is small enough to take a few tens of milliseconds, and cached per molecule, geometry and settings.

Colormaps follow matplotlib's direction (the first color is the most negative ESP): ``bwr`` paints negative
regions blue and positive ones red; "Reverse" (on by default) gives the Spartan/GaussView convention (red = negative) that
the viewer's charge coloring uses.
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

ACTOR = "qb_surface"
SOURCES = (("qm", "QM density isosurface"), ("vdw", "van der Waals (point charges)"),
           ("sas", "Solvent-accessible, 1.4 Å probe (point charges)"))
# control colors (0-255), evenly spaced from the most negative to the most positive value (matplotlib order)
COLORMAPS: dict[str, np.ndarray] = {
    "bwr": np.array([(0, 0, 255), (255, 255, 255), (255, 0, 0)], float),
    "coolwarm": np.array([(59, 76, 192), (98, 130, 234), (141, 176, 254), (185, 208, 249), (221, 220, 220),
                          (245, 196, 172), (244, 152, 122), (221, 95, 75), (180, 4, 38)], float),
    "RdBu_r": np.array([(5, 48, 97), (32, 101, 171), (67, 147, 195), (144, 196, 221), (209, 229, 240),
                        (247, 246, 246), (253, 219, 199), (243, 164, 129), (214, 96, 77), (177, 24, 43),
                        (103, 0, 31)], float),
}
PLAIN_RGB = (0.78, 0.80, 0.86)          # surface color when not colored by ESP
SYNC_ATOM_LIMIT = 40                    # charge surfaces of molecules this small take < 50 ms: no worker
GEOMETRY_TOL_A = 1e-4                   # displayed atoms moved more than this: the surface is stale
QM_RMSD_WARN, QM_RMSD_MAX = 0.05, 0.5   # Å between the cube's atoms and the displayed ones
CACHE_SIZE = 8


def colormap_rgb(name: str, n: int = 256, reverse: bool = False) -> np.ndarray:
    """(n, 3) floats in [0, 1], negative to positive."""
    stops = COLORMAPS[name][::-1] if reverse else COLORMAPS[name]
    x = np.linspace(0.0, 1.0, len(stops))
    t = np.linspace(0.0, 1.0, n)
    return np.column_stack([np.interp(t, x, stops[:, k]) for k in range(3)]) / 255.0


def _color_word(rgb) -> str:
    r, g, b = (float(c) for c in rgb)
    if b > r + 0.15:
        return "blue"
    if r > b + 0.15:
        return "red"
    return "white"


@dataclass
class SurfaceData:
    mesh: object                       # pyvista.PolyData with point data "esp_kcal" when there is an ESP
    esp: np.ndarray | None
    descriptors: dict
    atom_positions: np.ndarray         # geometry (Å) the surface belongs to
    seconds: float
    label: str
    notes: list[str] = field(default_factory=list)


# ====================================================================== computations (any thread)
def compute_qm(density_path: str, esp_path: str | None, iso: float) -> SurfaceData:
    from ..chem import cube as cube_mod
    from ..chem import surfaces as sf

    t = time.perf_counter()
    dens = cube_mod.read_cube(density_path)
    esp = cube_mod.read_cube(esp_path) if esp_path else None
    mesh, values = sf.qm_esp_surface(dens, esp, iso)
    desc = sf.gipf_descriptors(mesh, values)
    return SurfaceData(mesh, values, desc, dens.positions_A, time.perf_counter() - t,
                       f"QM density {iso:g} au" + (", QM ESP" if values is not None else ""))


def compute_charges(numbers, positions, charges, kind: str, label: str) -> SurfaceData:
    from ..chem import surfaces as sf

    t = time.perf_counter()
    mesh, values = sf.charge_esp_surface(numbers, positions, charges, kind=kind)
    desc = sf.gipf_descriptors(mesh, values)
    return SurfaceData(mesh, values, desc, np.asarray(positions, float).copy(), time.perf_counter() - t, label)


def gasteiger_charges(mol) -> np.ndarray:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    m = Chem.Mol(mol)
    AllChem.ComputeGasteigerCharges(m)
    q = np.array([a.GetDoubleProp("_GasteigerCharge") if a.HasProp("_GasteigerCharge") else 0.0
                  for a in m.GetAtoms()])
    return np.nan_to_num(q)


def pick_charges(res, row: int) -> tuple[np.ndarray, str]:
    """Charges (e) for the point-charge ESP of conformer ``row`` and where they come from."""
    n = res.mol.GetNumAtoms()
    origin = next((note.split(":", 1)[1].strip() for note in res.notes if note.startswith("atomic charges:")), "")
    confs = res.conformers
    if 0 <= row < len(confs) and confs[row].charges is not None and len(confs[row].charges) == n:
        return np.asarray(confs[row].charges, float), f"conformer {row + 1} charges" + (f" ({origin})" if origin else "")
    if confs and confs[0].charges is not None and len(confs[0].charges) == n:
        return np.asarray(confs[0].charges, float), "best conformer's charges" + (f" ({origin})" if origin else "")
    qm_q = ((res.qm or {}).get("properties") or {}).get("charges") or {}
    for key in ("mbis", "lowdin", "mulliken"):
        q = qm_q.get(key)
        if q is not None and len(q) == n:
            return np.asarray(q, float), f"QM {key.upper() if key == 'mbis' else key.capitalize()} charges"
    return gasteiger_charges(res.mol), "Gasteiger charges (RDKit)"


def qm_cube_files(res) -> tuple[str | None, str | None]:
    files = (getattr(res, "qm", None) or {}).get("files") or {}
    dens, esp = files.get("density"), files.get("esp")
    dens = str(dens) if dens and Path(dens).is_file() else None
    esp = str(esp) if esp and Path(esp).is_file() else None
    return dens, esp


def rigid_fit(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Rotation ``R`` and shift ``t`` minimizing |source @ R.T + t - target| (Kabsch), and the RMSD left."""
    a, b = np.asarray(source, float), np.asarray(target, float)
    ca, cb = a.mean(axis=0), b.mean(axis=0)
    u, _s, vt = np.linalg.svd((a - ca).T @ (b - cb))
    d = np.sign(np.linalg.det(vt.T @ u.T)) or 1.0
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    shift = cb - ca @ rot.T
    rmsd = float(np.sqrt(np.mean(np.sum((a @ rot.T + shift - b) ** 2, axis=1))))
    return rot, shift, rmsd


def transformed(data: SurfaceData, rot: np.ndarray, shift: np.ndarray, positions: np.ndarray) -> SurfaceData:
    """``data`` moved rigidly (mesh points and normals); it then belongs to ``positions``."""
    if np.allclose(rot, np.eye(3), atol=1e-9) and np.allclose(shift, 0.0, atol=1e-9):
        mesh = data.mesh
    else:
        mesh = data.mesh.copy()
        mesh.points = np.asarray(data.mesh.points) @ rot.T + shift
        normals = data.mesh.point_data.get("Normals")
        if normals is not None:
            mesh.point_data["Normals"] = np.asarray(normals) @ rot.T
            mesh.point_data.active_normals_name = "Normals"
    return SurfaceData(mesh, data.esp, data.descriptors, np.asarray(positions, float).copy(), data.seconds,
                       data.label, list(data.notes))


def _digest(*arrays) -> str:
    h = hashlib.blake2b(digest_size=16)
    for a in arrays:
        h.update(np.ascontiguousarray(np.round(np.asarray(a, float), 5)).tobytes())
    return h.hexdigest()


# ====================================================================== panel
class SurfacePanel(QWidget):
    def __init__(self, win, parent=None):
        super().__init__(parent)
        self.win = win
        self.viewer = win.viewer
        self.sync_atom_limit = SYNC_ATOM_LIMIT
        self.current: SurfaceData | None = None
        self._actor = None
        self._key = None                 # key of ``current``
        self._wanted = None              # key of the latest request (a late worker result for another is cached only)
        self._running: set = set()
        self._cache: OrderedDict = OrderedDict()
        self._preferred = "qm"
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(0)
        self._timer.timeout.connect(self.refresh)
        self._build()
        if self.viewer.plotter is None:
            self.message.setText("Surfaces need the 3D view, which is not available here"
                                 + (f" ({self.viewer.error})" if self.viewer.error else "") + ".")
            for w in self._controls:
                w.setEnabled(False)
            return
        win.resultChanged.connect(self._on_result_changed)
        self.viewer.sceneRebuilt.connect(self._on_scene_changed)
        self.viewer.coordinatesChanged.connect(self._on_scene_changed)
        self._update_sources(win.current_result())

    # ------------------------------------------------------------------ widgets
    def _build(self) -> None:
        lay = QVBoxLayout(self)
        self.message = QLabel("Select a built molecule and tick Show.")
        self.message.setWordWrap(True)
        self.message.setStyleSheet("color: gray;")
        lay.addWidget(self.message)
        form = QFormLayout()
        self.show_box = QCheckBox("Show surface")
        self.show_box.setToolTip("Draw the molecular surface in the 3D view (atoms can still be clicked through it)")
        form.addRow(self.show_box)
        self.source_box = QComboBox()
        for key, label in SOURCES:
            self.source_box.addItem(label, key)
        self.source_box.setToolTip("QM: electron-density isosurface from the QM stage's cube files.\n"
                                   "vdW / SAS: union of Bondi spheres (SAS: + 1.4 Å), ESP from atomic point charges.")
        form.addRow("Surface", self.source_box)
        self.iso_box = QDoubleSpinBox()
        self.iso_box.setDecimals(4)
        self.iso_box.setRange(0.0001, 0.1)
        self.iso_box.setSingleStep(0.0005)
        self.iso_box.setValue(0.002)
        self.iso_box.setSuffix(" au")
        self.iso_box.setToolTip("Electron density isovalue (e/bohr³): 0.001 au is the GIPF/Bader envelope, 0.002 au "
                                "is closer to crystal-packing volumes")
        form.addRow("Isovalue", self.iso_box)
        self.color_box = QComboBox()
        self.color_box.addItem("Electrostatic potential", "esp")
        self.color_box.addItem("None", "none")
        form.addRow("Color", self.color_box)
        row = QHBoxLayout()
        self.cmap_box = QComboBox()
        for name in COLORMAPS:
            self.cmap_box.addItem(name, name)
        self.reverse_box = QCheckBox("Reverse")
        self.reverse_box.setToolTip("Checked (default): red = negative, blue = positive, as in Spartan, GaussView "
                                    "and QuantaBench's charge coloring. Unchecked: matplotlib's direction.")
        self.reverse_box.setChecked(True)
        row.addWidget(self.cmap_box, 1)
        row.addWidget(self.reverse_box)
        form.addRow("Colormap", row)
        row = QHBoxLayout()
        self.range_box = QComboBox()
        self.range_box.addItem("Auto (symmetric)", "auto")
        self.range_box.addItem("Fixed ±", "fixed")
        self.range_value = QDoubleSpinBox()
        self.range_value.setRange(0.5, 500.0)
        self.range_value.setValue(30.0)
        self.range_value.setSuffix(" kcal/mol")
        row.addWidget(self.range_box, 1)
        row.addWidget(self.range_value, 1)
        form.addRow("Range", row)
        self.opacity_box = QDoubleSpinBox()
        self.opacity_box.setRange(0.05, 1.0)
        self.opacity_box.setSingleStep(0.05)
        self.opacity_box.setValue(0.85)
        form.addRow("Opacity", self.opacity_box)
        lay.addLayout(form)
        self.legend_bar = QFrame()
        self.legend_bar.setFixedHeight(14)
        self.legend_bar.setFrameShape(QFrame.Shape.StyledPanel)
        lay.addWidget(self.legend_bar)
        ticks = QHBoxLayout()
        self.legend_lo, self.legend_mid, self.legend_hi = QLabel(), QLabel("0"), QLabel()
        self.legend_mid.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.legend_hi.setAlignment(Qt.AlignmentFlag.AlignRight)
        for w in (self.legend_lo, self.legend_mid, self.legend_hi):
            ticks.addWidget(w, 1)
        lay.addLayout(ticks)
        self.legend_text = QLabel()
        self.legend_text.setWordWrap(True)
        lay.addWidget(self.legend_text)
        self.readout = QLabel()
        self.readout.setWordWrap(True)
        self.readout.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self.readout)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: gray;")
        lay.addWidget(self.status)
        lay.addStretch(1)
        self._controls = [self.show_box, self.source_box, self.iso_box, self.color_box, self.cmap_box,
                          self.reverse_box, self.range_box, self.range_value, self.opacity_box]
        self.show_box.toggled.connect(self._schedule)
        self.source_box.activated.connect(self._on_source_picked)
        self.source_box.currentIndexChanged.connect(self._on_source_changed)
        self.iso_box.valueChanged.connect(self._schedule)
        for w in (self.color_box, self.cmap_box, self.range_box):
            w.currentIndexChanged.connect(self._restyle)
        self.reverse_box.toggled.connect(self._restyle)
        self.range_value.valueChanged.connect(self._restyle)
        self.opacity_box.valueChanged.connect(self._restyle)
        self.range_value.setEnabled(False)
        self._on_source_changed()
        self._update_legend()

    # ------------------------------------------------------------------ state
    @property
    def plotter(self):
        return self.viewer.plotter

    def source(self) -> str:
        return self.source_box.currentData()

    def actor_shown(self) -> bool:
        return self.plotter is not None and ACTOR in self.plotter.renderer.actors

    @property
    def busy(self) -> bool:
        return bool(self._running) or self._timer.isActive()

    def _schedule(self, *_args) -> None:
        if self.plotter is not None:
            self._timer.start()

    def _on_source_picked(self, _index: int) -> None:
        self._preferred = self.source()

    def _on_source_changed(self, *_args) -> None:
        self.iso_box.setEnabled(self.source() == "qm")
        self._schedule()

    def _update_sources(self, res) -> None:
        """Enable the QM entry only when the molecule has a density cube; keep the user's choice if possible."""
        dens, _ = qm_cube_files(res) if res is not None else (None, None)
        item = self.source_box.model().item(0)
        item.setEnabled(dens is not None)
        item.setToolTip("" if dens else "No density cube: run the QM stage with cubes (CLI: build --qm ... --cubes)")
        want = self._preferred if (self._preferred != "qm" or dens) else "vdw"
        self.source_box.blockSignals(True)
        self.source_box.setCurrentIndex(self.source_box.findData(want))
        self.source_box.blockSignals(False)
        self.iso_box.setEnabled(want == "qm")

    def _on_result_changed(self, res) -> None:
        self.remove_surface()             # whatever is shown belongs to the previous molecule
        self._update_sources(res)
        self._schedule()

    def _on_scene_changed(self, *_args) -> None:
        if self.current is not None and not self._fits(self.current):
            self.remove_surface()
        self._schedule()

    def _fits(self, data: SurfaceData) -> bool:
        """The surface was made for exactly the displayed atoms (else it is removed and recomputed/realigned)."""
        atoms = self.viewer.atoms
        if atoms is None or len(atoms) != len(data.atom_positions):
            return False
        return float(np.max(np.abs(atoms.positions - data.atom_positions), initial=0.0)) <= GEOMETRY_TOL_A

    # ------------------------------------------------------------------ requests
    def _request(self):
        """``((key, callable, args), "")`` for what should be shown now, or ``(None, why not)``."""
        res = self.win.current_result()
        atoms = self.viewer.atoms
        if res is None or res.mol is None:
            return None, "Select a built molecule (surfaces need a finished build)."
        if atoms is None or len(atoms) != res.mol.GetNumAtoms():
            return None, "The 3D view shows another structure (e.g. the Surface Builder's slab)."
        src = self.source()
        if src == "qm":
            dens, esp = qm_cube_files(res)
            if dens is None:
                return None, "This molecule has no QM density cube; choose a vdW or SAS surface."
            iso = round(float(self.iso_box.value()), 6)
            stamp = tuple(Path(p).stat().st_mtime_ns for p in (dens, esp) if p)
            key = ("qm", res.input.id, dens, esp, iso, stamp)
            return (key, compute_qm, (dens, esp, iso)), ""
        row = max(0, self.win.conf_table.currentRow()) if hasattr(self.win, "conf_table") else 0
        charges, label = pick_charges(res, row)
        pos = atoms.positions.copy()
        key = (src, res.input.id, _digest(pos, charges, atoms.numbers))
        label = f"{'SAS' if src == 'sas' else 'vdW'} surface, point-charge ESP from {label}"
        return (key, compute_charges, (atoms.numbers.copy(), pos, charges, src, label)), ""

    def refresh(self) -> None:
        """Bring the 3D view in line with the controls: compute (or reuse) the surface, or remove it."""
        if self.plotter is None:
            return
        if not self.show_box.isChecked():
            self._wanted = None
            self.remove_surface()
            self.message.setText("Tick Show to draw the surface.")
            return
        req, why = self._request()
        if req is None:
            self._wanted = None
            self.remove_surface()
            self.message.setText(why)
            return
        key, fn, args = req
        self._wanted = key
        if key == self._key and self.current is not None and self.actor_shown():
            self._restyle()
            return
        if key in self._cache:
            self._cache.move_to_end(key)
            self._display(key, self._cache[key])
            return
        self.remove_surface()
        if self._running:                # one job at a time; _finished() comes back here for the newest request
            self.message.setText("Computing the surface...")
            return
        if fn is compute_charges and len(args[0]) <= self.sync_atom_limit:
            try:
                data = fn(*args)
            except Exception as exc:
                self._failed(key, f"{type(exc).__name__}: {exc}")
                return
            self._finished(key, data)
            return
        from . import workers

        self._running.add(key)
        self.message.setText("Computing the surface...")
        w = workers.FunctionWorker(fn, *args)
        workers.start(w, self.win, on_done=lambda data, k=key: self._finished(k, data),
                      on_failed=lambda msg, k=key: self._failed(k, msg))

    def _finished(self, key, data: SurfaceData) -> None:
        self._running.discard(key)
        self._cache[key] = data
        while len(self._cache) > CACHE_SIZE:
            self._cache.popitem(last=False)
        if key == self._wanted and self.show_box.isChecked():
            self._display(key, data)
        elif self._wanted is not None:
            self._schedule()             # the request changed meanwhile: serve the newest one

    def _failed(self, key, msg: str) -> None:
        self._running.discard(key)
        if key == self._wanted:
            self.remove_surface()
            self.message.setText(f"Surface failed: {msg}")
        elif self._wanted is not None:
            self._schedule()

    # ------------------------------------------------------------------ drawing
    def _display(self, key, data: SurfaceData) -> None:
        notes = list(data.notes)
        if key[0] == "qm":
            atoms = self.viewer.atoms
            if atoms is None or len(atoms) != len(data.atom_positions):
                self.remove_surface()
                self.message.setText("The QM cubes belong to another structure (the atoms differ).")
                return
            # the cubes may be in the QM program's frame (Psi4 recenters and reorients unless told not to), and
            # Move mode turns the molecule rigidly: superimpose the cube's atoms on the displayed ones
            rot, shift, rmsd = rigid_fit(data.atom_positions, atoms.positions)
            if rmsd > QM_RMSD_MAX:
                self.remove_surface()
                self.message.setText(f"The QM cubes were computed for another geometry (RMSD {rmsd:.2f} Å after "
                                     "superposition). Select the QM conformer, or use a vdW/SAS surface.")
                return
            if rmsd > QM_RMSD_WARN:
                notes.append(f"density from the QM geometry (RMSD {rmsd:.2f} Å from the displayed atoms)")
            data = transformed(data, rot, shift, atoms.positions)
        self.current, self._key = data, key
        self._add_actor()
        self.message.setText(data.label)
        self.status.setText(f"{data.mesh.n_points} vertices, {data.seconds * 1000:.0f} ms"
                            + "".join(f"; {n}" for n in notes))
        self._show_descriptors(data)

    def _lut(self, vmax: float):
        import vtk

        lut = vtk.vtkLookupTable()
        rgb = colormap_rgb(self.cmap_box.currentData(), 256, self.reverse_box.isChecked())
        lut.SetNumberOfTableValues(len(rgb))
        for i, (r, g, b) in enumerate(rgb):
            lut.SetTableValue(i, r, g, b, 1.0)
        lut.SetRange(-vmax, vmax)
        lut.SetNanColor(0.5, 0.5, 0.5, 1.0)
        lut.Build()
        return lut

    def color_range(self) -> float:
        if self.range_box.currentData() == "fixed" or self.current is None or self.current.esp is None:
            return float(self.range_value.value())
        return float(max(np.max(np.abs(self.current.esp)), 1e-3))

    def _add_actor(self) -> None:
        import vtk

        data = self.current
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(data.mesh)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        prop = actor.GetProperty()
        prop.SetInterpolationToPhong()
        prop.SetSpecular(0.25)
        prop.SetSpecularPower(30.0)
        prop.SetDiffuse(0.85)
        prop.SetAmbient(0.15)
        self.plotter.add_actor(actor, name=ACTOR, reset_camera=False, pickable=False, render=False)
        self._actor = actor
        self._restyle()

    def _restyle(self, *_args) -> None:
        """Colors, range and opacity changes: only the actor's mapper and property change, no recompute."""
        self._update_legend()
        if self.plotter is not None:
            self.range_value.setEnabled(self.range_box.currentData() == "fixed")
        if not self.actor_shown() or self.current is None or self._actor is None:
            return
        actor = self._actor
        mapper = actor.GetMapper()
        prop = actor.GetProperty()
        by_esp = self.color_box.currentData() == "esp" and self.current.esp is not None
        if by_esp:
            mapper.SetScalarModeToUsePointFieldData()
            mapper.SelectColorArray("esp_kcal")
            mapper.SetLookupTable(self._lut(self.color_range()))
            mapper.SetUseLookupTableScalarRange(True)
            mapper.SetColorModeToMapScalars()
            mapper.ScalarVisibilityOn()
        else:
            mapper.ScalarVisibilityOff()
            prop.SetColor(*PLAIN_RGB)
        opacity = float(self.opacity_box.value())
        prop.SetOpacity(opacity)
        # a closed surface with outward normals: culling its back faces keeps a translucent shell from
        # blending its far side over the near side in the wrong order (no depth peeling needed)
        prop.SetBackfaceCulling(opacity < 0.999)
        self.plotter.render()

    def remove_surface(self) -> None:
        if self.plotter is not None and ACTOR in self.plotter.renderer.actors:
            self.plotter.remove_actor(ACTOR, render=False)
            self.plotter.render()
        self.current, self._key = None, None
        self.readout.setText("")
        self.status.setText("")

    def _update_legend(self) -> None:
        rgb = colormap_rgb(self.cmap_box.currentData(), 9, self.reverse_box.isChecked())
        stops = ", ".join(f"stop:{i / (len(rgb) - 1):.3f} rgb({int(r * 255)}, {int(g * 255)}, {int(b * 255)})"
                          for i, (r, g, b) in enumerate(rgb))
        self.legend_bar.setStyleSheet(f"background: qlineargradient(x1:0, y1:0, x2:1, y2:0, {stops});")
        esp = self.color_box.currentData() == "esp"
        self.legend_bar.setVisible(esp)
        for w in (self.legend_lo, self.legend_mid, self.legend_hi):
            w.setVisible(esp)
        if not esp:
            self.legend_text.setText("Surface colored uniformly.")
            return
        vmax = self.color_range()
        self.legend_lo.setText(f"−{vmax:.1f}")
        self.legend_hi.setText(f"+{vmax:.1f}")
        self.legend_text.setText(f"ESP, kcal/mol: {_color_word(rgb[0])} = negative (electron-rich), "
                                 f"{_color_word(rgb[-1])} = positive (electron-poor); values beyond ±{vmax:.1f} "
                                 "are saturated.")

    def _show_descriptors(self, data: SurfaceData) -> None:
        d = data.descriptors
        rows = [("Area", "surf_area_A2", "Å²", 2), ("Volume", "surf_volume_A3", "Å³", 2)]
        if data.esp is not None:
            rows += [("V<sub>S,min</sub>", "esp_min_kcal", "kcal/mol", 2),
                     ("V<sub>S,max</sub>", "esp_max_kcal", "kcal/mol", 2),
                     ("V̄<sub>S</sub>", "esp_mean_kcal", "kcal/mol", 2),
                     ("V̄<sub>S</sub><sup>+</sup>", "esp_mean_pos_kcal", "kcal/mol", 2),
                     ("V̄<sub>S</sub><sup>−</sup>", "esp_mean_neg_kcal", "kcal/mol", 2),
                     ("σ²<sub>+</sub>", "esp_sigma2_pos", "(kcal/mol)²", 1),
                     ("σ²<sub>−</sub>", "esp_sigma2_neg", "(kcal/mol)²", 1),
                     ("σ²<sub>tot</sub>", "esp_sigma2_tot", "(kcal/mol)²", 1),
                     ("ν (balance)", "esp_nu", "", 3),
                     ("Π (local polarity)", "esp_pi", "kcal/mol", 2),
                     ("Area V > 0", "esp_area_pos_A2", "Å²", 1),
                     ("Area V < 0", "esp_area_neg_A2", "Å²", 1)]
        body = "".join(f"<tr><td>{label}</td><td align='right'>{d[key]:.{nd}f}</td><td>{unit}</td></tr>"
                       for label, key, unit, nd in rows if key in d)
        self.readout.setText("<b>Surface descriptors</b> (Murray-Politzer, area-weighted)"
                             f"<table cellspacing='4'>{body}</table>")


# ====================================================================== install
def _find_menu(win, title: str):
    bar = win.menuBar()
    for act in bar.actions():
        menu = act.menu()
        if menu is not None and act.text().replace("&", "").strip().lower() == title.lower():
            return menu
    return None


def install(win) -> None:
    """Add the Surfaces dock (tabified with Pipeline) and its toggle in the View menu."""
    panel = SurfacePanel(win)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.setWidget(panel)
    dock = QDockWidget("Surfaces", win)
    dock.setObjectName("surfaces")
    dock.setWidget(scroll)
    win.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
    pipeline = getattr(win, "pipeline_dock", None)
    if isinstance(pipeline, QDockWidget):
        win.tabifyDockWidget(pipeline, dock)
        pipeline.raise_()
    toggle = dock.toggleViewAction()
    toggle.setText("Surfaces panel")
    view = _find_menu(win, "View")
    if view is not None:
        before = getattr(win, "a_ovito", None)
        if isinstance(before, QObject) and before in view.actions():
            view.insertAction(before, toggle)
        else:
            view.addAction(toggle)
    win.surface_panel = panel
    win.surface_dock = dock
