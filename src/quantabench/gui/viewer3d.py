"""Interactive 3D molecular viewer (PyVista/VTK inside Qt).

Interaction follows the conventions shared by Avogadro, PyMOL and VMD, so it feels familiar:

==========================  ===================================================================
Mouse / key                 Action
==========================  ===================================================================
left-drag                   rotate the view (trackball camera)
middle-drag, Shift+left     pan
wheel, right-drag           zoom
click an atom               select it (replaces the selection); click on empty space clears
Ctrl+click                  add/remove an atom from the selection
double-click an atom        rotate about that atom from now on (moves the focal point)
R                           reset the view (the main window's View > Reset view)
X / Y / Z                   look down the x / y / z axis (Shift: from the negative side)
P                           toggle perspective / orthographic projection
+ / -                       zoom in / out
Ctrl+arrows                 turn the view 15 degrees
Esc                         clear the selection (the main window's Selection keys)
==========================  ===================================================================

These are the default keys: they are named actions of ``gui.keymap`` (``key_actions``), which Settings > Controls
rebinds; VTK's and pyvista's own single-letter keys are switched off.

``set_mode('select')`` makes a plain left-drag draw a selection box instead of rotating.
``set_mode('move')`` grabs atoms instead of the camera, like Avogadro's manipulate tool: left-drag
rotates the grabbed atoms rigidly about their centroid (a virtual trackball in screen space) and
Shift+left-drag translates them in the view plane. Pressing on an atom grabs its whole molecule
(connected fragment; on a surface system, the adsorbate, ``tag == 1``, see
``build.surfaces.TAG_ADSORBATE``), or the selection when that atom is selected. Pressing on empty
space grabs the selection, else the adsorbate, else everything, so a PFAS molecule can be oriented
above a graphite slab without touching the slab. A click in move mode never changes the selection.

``set_mode('build')`` is the Model Kit's mode (UI-7, ``gui.model_kit``): a drag still rotates the camera, but a
click builds instead of selecting (Ctrl+click still selects). The view stays a view: the kit gives it
the free valences to draw (``set_stubs``) and gets ``buildClicked(kind, index, point, shift)`` back, where ``kind``
is ``"stub"`` (``index`` is the row of the stubs given), ``"atom"`` or ``"space"`` (``point`` is the clicked point on
the focal plane, where a first atom goes). In build mode hydrogens are drawn as stubs, not atoms. A stub and an atom
under the cursor: the one nearer to the viewer wins.

Rendering: atoms are one instanced glyph actor (``vtkGlyph3DMapper``: one sphere mesh, drawn once per
atom on the GPU) and bonds are one instanced cylinder actor (two half-cylinders per bond, each in its
atom's color). Moving atoms therefore only rewrites a few NumPy arrays, never rebuilds meshes, and a
20,000-atom slab stays interactive. Picking is an exact ray–sphere intersection in NumPy against the
radii that are actually drawn, so it does not depend on the GPU's selection buffer and is unit-testable.

Radii: ball-and-stick uses 0.25 Å + 0.2 x covalent radius (Cordero et al., *Dalton Trans.* 2008,
2832); space filling uses the van der Waals radii in ``ase.data.vdw_radii`` (Bondi, *J. Phys. Chem.*
1964, 68, 441, completed with Mantina et al., *J. Phys. Chem. A* 2009, 113, 5806). Element colors are
the Jmol scheme. Charge coloring follows the electrostatic-potential convention used by Spartan and
Avogadro: red = negative, white = 0, blue = positive, on a scale symmetric about zero.

If OpenGL/VTK is unavailable (remote desktop, headless CI) the widget shows a message instead of
crashing the app, and ``.error`` says why.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, chemical_symbols, covalent_radii, vdw_radii
from ase.data.colors import jmol_colors
from PySide6.QtCore import QEvent, Qt, QTimer, Signal
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from . import keymap
from .selection import ACCENT_RGB, SelectionModel, format_measurement, measure

STYLES = ("Ball and stick", "Licorice", "Space filling", "Wireframe")
MODES = ("rotate", "select", "move", "build")
LABEL_MODES = ("none", "element", "index", "element+index", "charge")
COLOR_MODES = ("element", "charge", "tag")

CLICK_TOLERANCE_PX = 4.0          # press/release closer than this is a click, not a drag
WIREFRAME_PICK_RADIUS = 0.35      # Å; wireframe draws no spheres, so pick within this radius
LICORICE_RADIUS = 0.18            # Å
BOND_RADIUS = {"Ball and stick": 0.11, "Licorice": LICORICE_RADIUS}
KEY_ZOOM = 1.25                   # one zoom key press
KEY_TURN_DEG = 15.0               # one turn or roll key press
HALO_PAD = 0.28                   # Å added to the drawn radius for the selection halo
HALO_COLOR = ACCENT_RGB              # same amber as the 2D view's highlights
MEASURE_COLOR = (1.0, 0.80, 0.20)
DASH, GAP = 0.16, 0.10            # Å, dashed measurement lines
MAX_LABELS = 5000                 # beyond this, labels are unreadable and slow; they are skipped
STUB_LENGTH = 0.6                 # Å, a free valence drawn in build mode, from the atom's surface
STUB_RADIUS = 0.08                # Å
STUB_PICK_RADIUS = 0.3            # Å around a stub's tip
#: stub colours by the bond order a free valence takes: teal single, magenta double, orange triple
STUB_RGB = {1: (0, 168, 150), 2: (205, 80, 200), 3: (235, 120, 30)}

# ColorBrewer RdBu stops (Brewer 2003), reversed so negative charge is red as in ESP maps
_DIVERGING_STOPS = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
_DIVERGING_RGB = np.array([[178, 24, 43], [239, 138, 98], [247, 247, 247], [103, 169, 207], [33, 102, 172]], float)
# categorical colors for tags: 0 = surface (neutral gray), 1 = adsorbate (orange), then others
_TAG_RGB = np.array([[150, 156, 166], [255, 140, 40], [60, 180, 170], [170, 110, 220], [230, 80, 120],
                     [120, 190, 60]], np.uint8)


# ====================================================================== pure helpers
def bonds_from_rdkit(mol) -> np.ndarray:
    return np.array([[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()], dtype=int).reshape(-1, 2)


def bonds_from_distances(atoms: Atoms, scale: float = 1.15) -> np.ndarray:
    from ase.neighborlist import neighbor_list

    if len(atoms) < 2:
        return np.zeros((0, 2), int)
    cut = [covalent_radii[z] * scale for z in atoms.numbers]
    i, j = neighbor_list("ij", atoms, cut, self_interaction=False)
    pairs = np.column_stack([i, j])
    pairs = pairs[pairs[:, 0] < pairs[:, 1]]
    # drop bonds that wrap across periodic boundaries (drawn as long sticks otherwise)
    if len(pairs):
        d = np.linalg.norm(atoms.positions[pairs[:, 0]] - atoms.positions[pairs[:, 1]], axis=1)
        pairs = pairs[d < 3.0]
    return pairs.reshape(-1, 2).astype(int)


def element_counts(atoms: Atoms) -> str:
    from collections import Counter

    sym = {v: k for k, v in atomic_numbers.items()}
    c = Counter(atoms.numbers)
    return " ".join(f"{sym[z]}{n}" for z, n in sorted(c.items(), key=lambda t: (t[0] != 6, t[0] != 1, t[0])))


def atom_radii(numbers: np.ndarray, style: str) -> np.ndarray:
    """Radii (Å) that ``style`` draws; 0 for wireframe, which draws no spheres."""
    z = np.asarray(numbers, dtype=int)
    if style == "Space filling":
        return np.nan_to_num(vdw_radii[z], nan=1.8)
    if style == "Ball and stick":
        return 0.25 + 0.2 * covalent_radii[z]
    if style == "Licorice":
        return np.full(len(z), LICORICE_RADIUS)
    return np.zeros(len(z))


def sphere_resolution(n_atoms: int) -> int:
    """Facets per sphere: smooth for molecules, coarse for slabs (the GPU draws n x facets)."""
    if n_atoms < 500:
        return 24
    if n_atoms < 2000:
        return 16
    if n_atoms < 8000:
        return 10
    return 8


def diverging_rgb(values: np.ndarray, vmax: float) -> np.ndarray:
    """Red (negative) - white (0) - blue (positive), symmetric: ``vmax`` maps to full saturation."""
    t = np.clip(np.asarray(values, float) / max(float(vmax), 1e-12), -1.0, 1.0)
    rgb = np.column_stack([np.interp(t, _DIVERGING_STOPS, _DIVERGING_RGB[:, k]) for k in range(3)])
    return np.rint(rgb).astype(np.uint8)


def tag_rgb(tags: np.ndarray) -> np.ndarray:
    tags = np.asarray(tags, int)
    return _TAG_RGB[np.mod(tags, len(_TAG_RGB))]


def ray_sphere_pick(origin: np.ndarray, direction: np.ndarray, centers: np.ndarray,
                    radii: np.ndarray) -> tuple[int, float] | None:
    """Nearest sphere hit by the ray ``origin + t * direction`` (t >= 0).

    Returns ``(row, t)`` for the row of ``centers`` hit first, or None. Exact geometry: the ray hits
    a sphere when its closest approach to the center is within the radius; the entry point is at
    ``t = tc - sqrt(r² - d²)``. If the ray starts inside a sphere, its exit point is used.
    """
    if len(centers) == 0:
        return None
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    oc = np.asarray(centers, float) - np.asarray(origin, float)
    tc = oc @ d
    d2 = np.einsum("ij,ij->i", oc, oc) - tc * tc
    r2 = np.asarray(radii, float) ** 2
    hit = np.where(d2 <= r2)[0]
    if len(hit) == 0:
        return None
    th = np.sqrt(np.maximum(r2[hit] - d2[hit], 0.0))
    t0, t1 = tc[hit] - th, tc[hit] + th
    t = np.where(t0 >= 0, t0, t1)
    ok = t >= 0
    if not ok.any():
        return None
    k = np.argmin(np.where(ok, t, np.inf))
    return int(hit[k]), float(t[k])


def fragment_of(n_atoms: int, bonds: np.ndarray, atom: int) -> np.ndarray:
    """Indices of the connected fragment (through ``bonds``, (M, 2) pairs) that contains ``atom``."""
    adj: list[list[int]] = [[] for _ in range(n_atoms)]
    for a, b in np.asarray(bonds, int).reshape(-1, 2):
        adj[a].append(b)
        adj[b].append(a)
    seen = {int(atom)}
    stack = [int(atom)]
    while stack:
        for nb in adj[stack.pop()]:
            if nb not in seen:
                seen.add(int(nb))
                stack.append(int(nb))
    return np.array(sorted(seen), int)


def rotation_matrix(axis: Sequence[float], angle: float) -> np.ndarray:
    """Rodrigues rotation matrix for ``angle`` radians about ``axis`` (right-hand rule)."""
    a = np.asarray(axis, float)
    n = np.linalg.norm(a)
    if n < 1e-12 or angle == 0.0:
        return np.eye(3)
    x, y, z = a / n
    c, s = np.cos(angle), np.sin(angle)
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + s * k + (1 - c) * (k @ k)


def dashed_segments(points: np.ndarray, pairs: Sequence[tuple[int, int]], dash: float = DASH,
                    gap: float = GAP) -> tuple[np.ndarray, np.ndarray]:
    """Points and VTK line connectivity for dashed lines between ``points[i]`` and ``points[j]``.

    OpenGL core profiles have no line stipple, so dashes are real geometry.
    """
    pts, lines = [], []
    for i, j in pairs:
        a, b = np.asarray(points[i], float), np.asarray(points[j], float)
        length = float(np.linalg.norm(b - a))
        if length < 1e-9:
            continue
        u = (b - a) / length
        # whole number of dash+gap periods, stretched so the line starts and ends on a dash
        n = max(1, round((length + gap) / (dash + gap)))
        period = (length + gap) / n
        on = max(period - gap, 0.5 * period)
        for k in range(n):
            s = k * period
            lines += [2, len(pts), len(pts) + 1]
            pts += [a + s * u, a + min(s + on, length) * u]
    return np.array(pts, float).reshape(-1, 3), np.array(lines, dtype=np.int64)


def charges_from_rdkit(mol) -> np.ndarray | None:
    """Per-atom partial charges stored on an RDKit molecule (xTB/Gasteiger props), if any."""
    for prop in ("xtb_charge", "PartialCharge", "_GasteigerCharge"):
        if all(a.HasProp(prop) for a in mol.GetAtoms()) and mol.GetNumAtoms():
            try:
                q = np.array([float(a.GetProp(prop)) for a in mol.GetAtoms()])
            except ValueError:
                continue
            if np.all(np.isfinite(q)):
                return q
    return None


def charges_from_atoms(atoms: Atoms) -> np.ndarray | None:
    """Charges from a calculator result or ``initial_charges`` (extxyz writes them there)."""
    calc = getattr(atoms, "calc", None)
    if calc is not None:
        q = getattr(calc, "results", {}).get("charges")
        if q is not None and len(q) == len(atoms):
            return np.asarray(q, float)
    if "initial_charges" in atoms.arrays:
        q = atoms.get_initial_charges()
        if np.any(q != 0):
            return np.asarray(q, float)
    return None


@dataclass
class _Gesture:
    """State of one left-button press until its release."""
    x: float
    y: float
    shift: bool = False
    ctrl: bool = False
    kind: str = "click"           # click | camera | move | translate | box | double
    dragged: bool = False
    last: tuple[float, float] = (0.0, 0.0)
    indices: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    start: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    pivot: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rot: np.ndarray = field(default_factory=lambda: np.eye(3))
    shift_vec: np.ndarray = field(default_factory=lambda: np.zeros(3))


# ====================================================================== widget
class MoleculeViewer3D(QWidget):
    """3D view of an ASE ``Atoms`` or RDKit molecule with Avogadro-style mouse interaction."""

    measurementChanged = Signal(str)      # "distance: 1.234 Å", "angle: ...", "" when none
    coordinatesChanged = Signal(object)   # np.ndarray (N, 3), full new positions after a move
    #: The scene was rebuilt (new structure, style, colors, hydrogens) or cleared. Actors added by others
    #: (e.g. the Surfaces panel's "qb_surface") are kept; listeners check whether they still fit ``atoms``.
    sceneRebuilt = Signal()
    #: A click in build mode: (kind "stub" | "atom" | "space", stub row or atom index (-1 for space), world point
    #: (3,) of the stub tip, atom centre or the focal plane, Shift held).
    buildClicked = Signal(str, int, object, bool)

    def __init__(self, parent=None, selection: SelectionModel | None = None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.plotter = None
        self.error = ""
        self.style = STYLES[0]
        self.show_hydrogens = True
        self.show_cell = True
        self.mode = "rotate"
        self.label_mode = "none"
        self.color_mode = "element"
        self.selection = selection if selection is not None else SelectionModel(self)
        self.selection.changed.connect(self._on_selection_changed)
        self._atoms: Atoms | None = None
        self._bonds = np.zeros((0, 2), int)
        self._charges: np.ndarray | None = None
        self._radii = np.zeros(0)          # drawn radius per atom (full indexing)
        self._vis = np.zeros(0, int)       # full indices of displayed atoms
        self._bond_vis = np.zeros((0, 2), int)
        self._colors = np.zeros((0, 3), np.uint8)
        self._names: set[str] = set()      # actors owned by the scene (removed on rebuild)
        self._atom_pd = self._bond_pd = self._wire_pd = self._label_pd = None
        self._measure_text = ""
        self._gesture: _Gesture | None = None
        self._camera_drag = False
        self._style_obj = None
        self._band_actor = None
        self._stubs: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None   # build mode: atom, direction, order
        self._stub_tips = np.zeros((0, 3))
        # One render per event-loop pass: a new molecule rebuilds the scene, clears the selection, and panels
        # add their actors, each asking for a render; under software OpenGL each costs ~65 ms (SCALE-1).
        self.renders = 0                   # renders done (tests count them)
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(0)
        self._render_timer.timeout.connect(self.render_now)
        try:
            import pyvista as pv
            from pyvistaqt import QtInteractor

            pv.global_theme.smooth_shading = True
            # auto_update=False: pyvistaqt's default re-renders 5 times a second forever, idle or not
            self.plotter = QtInteractor(self, auto_update=False)
            lay.addWidget(self.plotter.interactor)
            self._aa = ""
            self._set_anti_aliasing(0)
            self._set_background()
            self._install_interaction()
            self.plotter.add_axes(interactive=False, line_width=2, labels_off=False)
        except Exception as exc:  # no OpenGL, missing VTK, headless
            self.plotter = None
            self.error = f"3D view unavailable: {type(exc).__name__}: {exc}"
            msg = QLabel(self.error)
            msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
            msg.setWordWrap(True)
            lay.addWidget(msg)
        self.key_actions = self._make_key_actions()
        if self.plotter is not None:      # the VTK widget takes every key itself: the actions see them first
            self._key_dispatcher = keymap.KeyDispatcher(self.plotter.interactor)

    def _make_key_actions(self) -> dict:
        """The 3D view's own keys as named actions (``gui.keymap``, scope "view3d"), with their default keys; the main
        window's keymap applies the user's."""
        cmds = {
            "view3d.along_x": lambda: self.view_along("x"), "view3d.along_y": lambda: self.view_along("y"),
            "view3d.along_z": lambda: self.view_along("z"),
            "view3d.along_minus_x": lambda: self.view_along("x", negative=True),
            "view3d.along_minus_y": lambda: self.view_along("y", negative=True),
            "view3d.along_minus_z": lambda: self.view_along("z", negative=True),
            "view3d.projection": self.toggle_projection,
            "view3d.zoom_in": lambda: self.zoom(KEY_ZOOM), "view3d.zoom_out": lambda: self.zoom(1.0 / KEY_ZOOM),
            "view3d.rotate_left": lambda: self.turn(-KEY_TURN_DEG, 0.0),
            "view3d.rotate_right": lambda: self.turn(KEY_TURN_DEG, 0.0),
            "view3d.rotate_up": lambda: self.turn(0.0, -KEY_TURN_DEG),
            "view3d.rotate_down": lambda: self.turn(0.0, KEY_TURN_DEG),
            "view3d.roll_ccw": lambda: self.roll(KEY_TURN_DEG), "view3d.roll_cw": lambda: self.roll(-KEY_TURN_DEG),
        }
        return {name: keymap.local_action(self, name, fn) for name, fn in cmds.items()}

    def run_command(self, name: str) -> bool:
        """Trigger the 3D view's action ``name`` ("view3d.along_x", ...); False when there is no such action."""
        a = self.key_actions.get(name)
        if a is None:
            return False
        a.trigger()
        return True

    # ================================================================== public API
    @property
    def atoms(self) -> Atoms | None:
        """The displayed structure (a copy owned by the viewer; moves update it)."""
        return self._atoms

    def show_atoms(self, atoms: Atoms, bonds: np.ndarray | None = None, reset_camera: bool = True,
                   charges: Sequence[float] | None = None) -> None:
        """Show ``atoms``. Bonds default to covalent-radius distance criteria; ``charges`` (e) default
        to the calculator's or the extxyz ``initial_charges`` if present."""
        if charges is None:
            charges = charges_from_atoms(atoms)
        self._atoms = atoms.copy()
        self._bonds = bonds_from_distances(self._atoms) if bonds is None else np.asarray(bonds, int).reshape(-1, 2)
        self._charges = None if charges is None else np.asarray(charges, float).reshape(-1)
        if self._charges is not None and len(self._charges) != len(atoms):
            self._charges = None
        sel = self.selection.atoms
        if sel and max(sel) >= len(atoms):
            self.selection.clear()           # indices refer to another molecule
        self._rebuild(reset_camera)

    def show_rdkit(self, mol, conf_id: int = -1, reset_camera: bool = True,
                   charges: Sequence[float] | None = None) -> None:
        conf = mol.GetConformer(conf_id)
        atoms = Atoms([a.GetSymbol() for a in mol.GetAtoms()], positions=np.array(conf.GetPositions()))
        if charges is None:
            charges = charges_from_rdkit(mol)
        self.show_atoms(atoms, bonds_from_rdkit(mol), reset_camera, charges=charges)

    def set_style(self, style: str) -> None:
        self.style = style
        self._rebuild(False)

    def set_show_hydrogens(self, on: bool) -> None:
        self.show_hydrogens = bool(on)
        self._rebuild(False)

    def set_label_mode(self, mode: str) -> None:
        if mode not in LABEL_MODES:
            raise ValueError(f"label mode must be one of {LABEL_MODES}")
        self.label_mode = mode
        self._update_labels()
        self._render()

    def set_color_mode(self, mode: str) -> None:
        if mode not in COLOR_MODES:
            raise ValueError(f"color mode must be one of {COLOR_MODES}")
        self.color_mode = mode
        self._rebuild(False)

    def set_mode(self, mode: str) -> None:
        """'rotate' (drag moves the camera), 'select' (drag draws a box), 'move' (drag moves atoms), 'build' (a
        click builds, see the module docstring)."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        was = self._build_view()
        self.mode = mode
        if self._build_view() != was:
            self._rebuild(False)             # hydrogens become stubs, or atoms again

    def set_stubs(self, atoms=None, directions=None, orders=None, redraw: bool = True) -> None:
        """Free valences to draw as stubs in build mode: the atom each one starts from, its unit direction and the
        bond order it takes (1, 2 or 3; colour). None removes them. ``redraw=False`` when the structure is shown
        next anyway (``show_rdkit``), so the scene is rebuilt once."""
        if atoms is None:
            self._stubs = None
        else:
            a = np.asarray(atoms, int).reshape(-1)
            d = np.asarray(directions, float).reshape(-1, 3)
            o = np.ones(len(a), int) if orders is None else np.asarray(orders, int).reshape(-1)
            if not len(a) == len(d) == len(o):
                raise ValueError("one direction and one order per stub")
            self._stubs = (a, d, o)
        if redraw:
            self._rebuild(False)

    @property
    def stub_tips(self) -> np.ndarray:
        """(S, 3) tips of the stubs drawn now (the points a click picks), in the order given to ``set_stubs``."""
        return self._stub_tips

    def clear(self) -> None:
        self._atoms = None
        self._charges = None
        self._bonds = np.zeros((0, 2), int)
        self._remove_scene()
        self.selection.clear()
        self._render()
        self.sceneRebuilt.emit()

    def reset_camera(self, min_extent: float = 0.0) -> None:
        """Default (isometric) orientation, zoomed to fit, rotating about the center of the structure. The view
        spans at least ``min_extent`` Å (the Model Kit's first atom is not blown up to fill the view)."""
        if self.plotter is not None:
            self._reset_view(min_extent)
            self._render()

    def _reset_view(self, min_extent: float = 0.0) -> None:
        p = self.plotter
        p.camera_position = p.get_default_cam_pos()
        if self._atoms is not None and len(self._atoms):
            # fit the atoms, not the cell: a slab's 30 Å of vacuum would otherwise shrink the structure
            pos = self._atoms.positions
            pad = (float(self._radii.max()) if len(self._radii) else 1.0) + 0.5
            lo, hi = pos.min(axis=0) - pad, pos.max(axis=0) + pad
            grow = np.maximum(0.0, float(min_extent) - (hi - lo)) / 2
            lo, hi = lo - grow, hi + grow
            # the renderer's, not the plotter's: Plotter.reset_camera renders whatever ``render`` says
            p.renderer.reset_camera(render=False, bounds=(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]))
        else:
            p.renderer.reset_camera(render=False)
        # pyvista resets the camera on the first real render unless told the camera is set,
        # which would silently undo double-click recentering and axis views
        p.camera_set = True

    def view_along(self, axis: str, negative: bool = False) -> None:
        """Look down the x, y or z axis (camera on the + side unless ``negative``)."""
        if self.plotter is None:
            return
        view = {"x": self.plotter.view_yz, "y": self.plotter.view_xz, "z": self.plotter.view_xy}[axis.lower()]
        view(negative=negative, render=False)
        self.plotter.renderer.reset_camera(render=False)
        self.plotter.camera_set = True
        self._render()

    def toggle_projection(self) -> bool:
        """Switch perspective/orthographic; returns True when orthographic is now on."""
        if self.plotter is None:
            return False
        cam = self.plotter.renderer.GetActiveCamera()
        if cam.GetParallelProjection():
            self.plotter.disable_parallel_projection()
        else:
            self.plotter.enable_parallel_projection()
        self._render()
        return bool(cam.GetParallelProjection())

    def zoom(self, factor: float) -> None:
        """Zoom the camera by ``factor`` (> 1 closer)."""
        if self.plotter is None:
            return
        cam = self.plotter.renderer.GetActiveCamera()
        if cam.GetParallelProjection():
            cam.SetParallelScale(cam.GetParallelScale() / factor)
        else:
            cam.Dolly(factor)
        self.plotter.renderer.ResetCameraClippingRange()
        self.plotter.camera_set = True
        self._render()

    def turn(self, right_deg: float, down_deg: float) -> None:
        """Turn the structure ``right_deg`` to the right (about the screen's vertical axis) and ``down_deg`` down
        (its top toward the viewer), by moving the camera around the focal point."""
        if self.plotter is None:
            return
        cam = self.plotter.renderer.GetActiveCamera()
        if right_deg:
            cam.Azimuth(-right_deg)
        if down_deg:
            cam.Elevation(down_deg)
            cam.OrthogonalizeViewUp()
        self.plotter.renderer.ResetCameraClippingRange()
        self.plotter.camera_set = True
        self._render()

    def roll(self, ccw_deg: float) -> None:
        """Roll the structure ``ccw_deg`` counter-clockwise in the plane of the screen."""
        if self.plotter is None:
            return
        self.plotter.renderer.GetActiveCamera().Roll(-ccw_deg)
        self.plotter.camera_set = True
        self._render()

    def atom_count(self) -> int:
        """Atoms in the structure shown (0 when nothing is)."""
        return 0 if self._atoms is None else len(self._atoms)

    def center_on_atom(self, index: int) -> None:
        """Make atom ``index`` the center of rotation without turning the view (camera translates)."""
        if self.plotter is None or self._atoms is None:
            return
        cam = self.plotter.renderer.GetActiveCamera()
        target = self._atoms.positions[int(index)]
        delta = target - np.array(cam.GetFocalPoint())
        cam.SetFocalPoint(*target)
        cam.SetPosition(*(np.array(cam.GetPosition()) + delta))
        self.plotter.renderer.ResetCameraClippingRange()
        self._render()

    def screenshot(self, path, scale: int = 2, transparent: bool = False) -> None:
        """Save a PNG at ``scale`` x the window resolution (for papers and slides).

        VTK renders the enlarged image in tiles; 3D geometry scales by itself, but text sizes and
        line widths are in pixels, so they are enlarged for the capture and restored afterwards."""
        if self.plotter is None:
            return
        scale = max(1, int(scale))
        restore = self._scale_overlays(scale) if scale > 1 else []
        try:
            self.render_now()
            self.plotter.screenshot(str(path), transparent_background=transparent, scale=scale)
        finally:
            for fn in restore:
                fn()
            self._render()

    def _scale_overlays(self, factor: int) -> list:
        """Multiply pixel-sized text and line widths of the scene's actors; returns undo callbacks."""
        undo = []
        actors = self.plotter.renderer.actors
        for name in self._names:
            actor = actors.get(name)
            mapper = actor.GetMapper() if actor is not None and hasattr(actor, "GetMapper") else None
            tp = mapper.GetLabelTextProperty() if hasattr(mapper, "GetLabelTextProperty") else None
            if tp is not None:
                size = tp.GetFontSize()
                tp.SetFontSize(size * factor)
                undo.append(lambda tp=tp, size=size: tp.SetFontSize(size))
            elif actor is not None and hasattr(actor, "GetProperty") and hasattr(actor.GetProperty(), "GetLineWidth"):
                prop = actor.GetProperty()
                width = prop.GetLineWidth()
                prop.SetLineWidth(width * factor)
                undo.append(lambda prop=prop, width=width: prop.SetLineWidth(width))
        return undo

    def close_viewer(self) -> None:
        self._render_timer.stop()
        if self.plotter is not None:
            self.plotter.close()
            self.plotter = None

    # ------------------------------------------------------------------ moving atoms
    def move_group(self) -> np.ndarray:
        """Atoms that a move-mode drag acts on: the selection, else the adsorbate, else everything."""
        if self._atoms is None:
            return np.zeros(0, int)
        sel = self.selection.atoms
        if sel:
            return np.array(sel, int)
        tags = self._atoms.get_tags()
        if len(tags) and np.any(tags == 1):
            return np.where(tags == 1)[0]
        return np.arange(len(self._atoms))

    def grab_group(self, atom: int | None) -> np.ndarray:
        """Atoms a move-mode press on ``atom`` grabs (Avogadro's manipulate tool): the selection when ``atom``
        is selected, else ``atom``'s connected fragment (in a tagged surface system, the adsorbate when
        ``atom`` is a surface atom, its tag group otherwise). ``None`` (empty space) -> :meth:`move_group`."""
        if self._atoms is None:
            return np.zeros(0, int)
        if atom is None:
            return self.move_group()
        sel = self.selection.atoms
        if sel and atom in sel:
            return np.array(sel, int)
        tags = self._atoms.get_tags()
        if len(tags) and np.any(tags == 1):
            if tags[atom] == 0:
                return np.where(tags == 1)[0]
            return np.where(tags == tags[atom])[0]
        return fragment_of(len(self._atoms), self._bonds, atom)

    def rotate_atoms(self, indices, axis, angle_deg: float, center=None) -> None:
        """Rigidly rotate ``indices`` by ``angle_deg`` about ``axis`` through ``center`` (default: centroid)."""
        idx = np.asarray(indices, int)
        pos = self._atoms.positions.copy()
        c = pos[idx].mean(axis=0) if center is None else np.asarray(center, float)
        pos[idx] = (pos[idx] - c) @ rotation_matrix(axis, np.radians(angle_deg)).T + c
        self.set_positions(pos)

    def translate_atoms(self, indices, vector) -> None:
        pos = self._atoms.positions.copy()
        pos[np.asarray(indices, int)] += np.asarray(vector, float)
        self.set_positions(pos)

    def set_positions(self, positions: np.ndarray, emit: bool = True) -> None:
        """Replace all coordinates (Å) without rebuilding the scene; bonds keep their topology."""
        if self._atoms is None:
            return
        self._atoms.positions = np.asarray(positions, float).reshape(-1, 3)
        self._refresh_geometry()
        if emit:
            self.coordinatesChanged.emit(self._atoms.positions.copy())

    # ------------------------------------------------------------------ coordinates & picking
    def _projection(self) -> tuple[np.ndarray, tuple[float, float, float, float]]:
        """World -> NDC matrix and the renderer viewport (x0, y0, w, h) in display pixels."""
        ren = self.plotter.renderer
        cam = ren.GetActiveCamera()
        m = cam.GetCompositeProjectionTransformMatrix(ren.GetTiledAspectRatio(), -1, 1)
        mat = np.array([[m.GetElement(i, j) for j in range(4)] for i in range(4)])
        x0, y0 = ren.GetOrigin()
        w, h = ren.GetSize()
        return mat, (float(x0), float(y0), float(max(w, 1)), float(max(h, 1)))

    def world_to_display(self, points: np.ndarray) -> np.ndarray:
        """(N, 3) world points -> (N, 3) display (x px, y px from the bottom, NDC depth)."""
        mat, (x0, y0, w, h) = self._projection()
        p = np.atleast_2d(np.asarray(points, float))
        hom = np.column_stack([p, np.ones(len(p))]) @ mat.T
        ndc = hom[:, :3] / hom[:, 3:4]
        return np.column_stack([x0 + (ndc[:, 0] + 1) * 0.5 * w, y0 + (ndc[:, 1] + 1) * 0.5 * h, ndc[:, 2]])

    def display_to_world(self, x: float, y: float, depth: float) -> np.ndarray:
        """Inverse of ``world_to_display`` for one point at NDC ``depth`` (-1 near, 1 far)."""
        mat, (x0, y0, w, h) = self._projection()
        ndc = np.array([2 * (x - x0) / w - 1, 2 * (y - y0) / h - 1, depth, 1.0])
        p = np.linalg.solve(mat, ndc)
        return p[:3] / p[3]

    def display_ray(self, x: float, y: float) -> tuple[np.ndarray, np.ndarray]:
        """Ray (origin on the near plane, unit direction) through display pixel (x, y).
        Works for perspective and orthographic cameras alike."""
        near = self.display_to_world(x, y, -1.0)
        far = self.display_to_world(x, y, 1.0)
        d = far - near
        return near, d / np.linalg.norm(d)

    def pick_atom(self, display_x: float, display_y: float) -> int | None:
        """Full-molecule index of the displayed atom under display pixel (x, y from the bottom), or None.

        Hidden hydrogens are not pickable, but the returned index always counts them."""
        hit = self._atom_hit(display_x, display_y)
        return None if hit is None else hit[0]

    def _atom_hit(self, display_x: float, display_y: float) -> tuple[int, float] | None:
        """(full index, distance along the ray) of the displayed atom under the pixel, or None."""
        if self.plotter is None or self._atoms is None or len(self._vis) == 0:
            return None
        origin, direction = self.display_ray(display_x, display_y)
        radii = self._radii[self._vis]
        if self.style == "Wireframe":
            radii = np.full(len(self._vis), WIREFRAME_PICK_RADIUS)
        hit = ray_sphere_pick(origin, direction, self._atoms.positions[self._vis], radii)
        return None if hit is None else (int(self._vis[hit[0]]), hit[1])

    def pick_stub(self, display_x: float, display_y: float) -> tuple[int, float] | None:
        """(row, distance along the ray) of the build-mode stub whose tip is under the pixel, or None."""
        if self.plotter is None or not len(self._stub_tips):
            return None
        origin, direction = self.display_ray(display_x, display_y)
        return ray_sphere_pick(origin, direction, self._stub_tips, np.full(len(self._stub_tips), STUB_PICK_RADIUS))

    def focal_plane_point(self, display_x: float, display_y: float) -> np.ndarray:
        """World point under the pixel at the depth of the camera's focal point (where a click in empty space
        puts a new atom)."""
        cam = self.plotter.renderer.GetActiveCamera()
        depth = self.world_to_display(np.array(cam.GetFocalPoint()))[0, 2]
        return self.display_to_world(display_x, display_y, depth)

    # ================================================================== mouse logic (VTK-free)
    def _mouse_press(self, x: float, y: float, shift: bool = False, ctrl: bool = False,
                     double: bool = False) -> bool:
        """Left press at display (x, y). Returns True when the viewer consumes it (camera must not move)."""
        g = _Gesture(x=x, y=y, shift=shift, ctrl=ctrl, last=(x, y))
        self._gesture = g
        if double:
            g.kind = "double"
            idx = self.pick_atom(x, y)
            if idx is not None:
                self.center_on_atom(idx)
            return True
        if self.mode == "move" and self._atoms is not None:
            g.kind = "translate" if shift else "move"
            g.indices = self.grab_group(self.pick_atom(x, y))
            g.start = self._atoms.positions[g.indices].copy()
            g.pivot = g.start.mean(axis=0) if len(g.start) else np.zeros(3)
            return True
        if self.mode == "select" and not shift:
            g.kind = "box"
            return True
        g.kind = "camera"
        return False

    def _mouse_move(self, x: float, y: float) -> bool:
        g = self._gesture
        if g is None or g.kind in ("camera", "double"):
            return False
        if not g.dragged and np.hypot(x - g.x, y - g.y) < CLICK_TOLERANCE_PX:
            return True
        g.dragged = True
        if g.kind == "box":
            self._draw_band(g.x, g.y, x, y)
        elif g.kind in ("move", "translate") and len(g.indices):
            dx, dy = x - g.last[0], y - g.last[1]
            if g.kind == "move":
                right, up, _ = self._camera_frame()
                w, h = self._projection()[1][2:]
                angle = np.pi * np.hypot(dx, dy) / min(w, h)   # a drag across the view = 180°
                g.rot = rotation_matrix(-dy * right + dx * up, angle) @ g.rot
            else:
                center = g.pivot + g.shift_vec
                px, py, pz = self.world_to_display(center)[0]
                g.shift_vec = g.shift_vec + self.display_to_world(px + dx, py + dy, pz) - \
                    self.display_to_world(px, py, pz)
            pos = self._atoms.positions.copy()
            pos[g.indices] = (g.start - g.pivot) @ g.rot.T + g.pivot + g.shift_vec
            self.set_positions(pos, emit=False)
        g.last = (x, y)
        return True

    def _mouse_release(self, x: float, y: float) -> bool:
        g, self._gesture = self._gesture, None
        if g is None or g.kind == "double":
            return g is not None
        moved = g.dragged or np.hypot(x - g.x, y - g.y) >= CLICK_TOLERANCE_PX
        if g.kind == "box":
            self._clear_band()
            if moved:
                self._box_select(g.x, g.y, x, y, additive=g.ctrl)
                return True
        if g.kind in ("move", "translate"):
            if g.dragged:
                self.coordinatesChanged.emit(self._atoms.positions.copy())
            return True                  # a click in move mode does not change what is selected
        if not moved:
            self._click(x, y, g.ctrl, g.shift)
        return g.kind != "camera"

    def _click(self, x: float, y: float, ctrl: bool, shift: bool = False) -> None:
        if self.mode == "build" and not ctrl:
            self._build_click(x, y, shift)
            return
        idx = self.pick_atom(x, y)
        if idx is None:
            if not ctrl:
                self.selection.clear()
        elif ctrl:
            self.selection.toggle(idx)
        else:
            self.selection.select([idx])

    def _build_click(self, x: float, y: float, shift: bool) -> None:
        """Build mode: say what was clicked (``buildClicked``); the selection does not change."""
        if self.plotter is None:
            return
        stub = self.pick_stub(x, y)
        atom = self._atom_hit(x, y)
        if stub is not None and (atom is None or stub[1] <= atom[1]):
            self.buildClicked.emit("stub", stub[0], self._stub_tips[stub[0]].copy(), shift)
        elif atom is not None:
            self.buildClicked.emit("atom", atom[0], self._atoms.positions[atom[0]].copy(), shift)
        else:
            self.buildClicked.emit("space", -1, self.focal_plane_point(x, y), shift)

    def _box_select(self, x0: float, y0: float, x1: float, y1: float, additive: bool) -> None:
        if self._atoms is None or len(self._vis) == 0:
            return
        d = self.world_to_display(self._atoms.positions[self._vis])
        inside = ((d[:, 0] >= min(x0, x1)) & (d[:, 0] <= max(x0, x1)) & (d[:, 1] >= min(y0, y1))
                  & (d[:, 1] <= max(y0, y1)) & (np.abs(d[:, 2]) <= 1))
        self.selection.select(self._vis[inside], additive=additive)

    def _camera_frame(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Unit vectors (right, up, toward viewer) of the screen, in world coordinates."""
        cam = self.plotter.renderer.GetActiveCamera()
        dop = np.array(cam.GetDirectionOfProjection())
        right = np.cross(dop, np.array(cam.GetViewUp()))
        right /= np.linalg.norm(right)
        up = np.cross(right, dop)
        return right, up / np.linalg.norm(up), -dop

    # ================================================================== VTK wiring
    def _install_interaction(self) -> None:
        import vtk

        p = self.plotter
        style = vtk.vtkInteractorStyleTrackballCamera()
        p.iren.style = style
        self._style_obj = style
        iren = p.iren.interactor

        def info():
            x, y = iren.GetEventPosition()
            return float(x), float(y)

        def on_press(_obj, _ev):
            x, y = info()
            consumed = self._mouse_press(x, y, shift=bool(iren.GetShiftKey()), ctrl=bool(iren.GetControlKey()),
                                         double=iren.GetRepeatCount() > 0)
            self._camera_drag = not consumed
            if self._camera_drag:
                style.OnLeftButtonDown()

        def on_move(_obj, _ev):
            if not self._mouse_move(*info()):
                style.OnMouseMove()

        def on_release(_obj, _ev):
            self._mouse_release(*info())
            if self._camera_drag:
                style.OnLeftButtonUp()
                self._camera_drag = False

        style.AddObserver("LeftButtonPressEvent", on_press)
        style.AddObserver("MouseMoveEvent", on_move)
        style.AddObserver("LeftButtonReleaseEvent", on_release)
        # VTK's built-in single-letter keys (w/s wireframe, e/q exit, 3 stereo...) would fight ours, which are
        # actions (``key_actions``, gui.keymap)
        style.AddObserver("CharEvent", lambda *_: None)
        p.iren.clear_key_event_callbacks()   # pyvista binds q = close window, b/v/C pickers, ...

    def _draw_band(self, x0: float, y0: float, x1: float, y1: float) -> None:
        import pyvista as pv
        import vtk

        pts = np.array([[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0]], float)
        poly = pv.PolyData(pts, lines=np.array([5, 0, 1, 2, 3, 0]))
        if self._band_actor is None:
            mapper = vtk.vtkPolyDataMapper2D()
            self._band_actor = vtk.vtkActor2D()
            self._band_actor.SetMapper(mapper)
            self._band_actor.GetProperty().SetColor(*HALO_COLOR)
            self._band_actor.GetProperty().SetLineWidth(1.5)
            self.plotter.renderer.AddViewProp(self._band_actor)
        self._band_actor.GetMapper().SetInputData(poly)
        self._band_actor.SetVisibility(True)
        self._render()

    def _clear_band(self) -> None:
        if self._band_actor is not None:
            self._band_actor.SetVisibility(False)
            self._render()

    # ================================================================== drawing
    def _render(self) -> None:
        """Ask for a render at the next pass of the event loop; requests until then share it."""
        if self.plotter is not None:
            self._render_timer.start()

    request_render = _render

    def render_now(self) -> None:
        """Render at once (a pending request is then done)."""
        self._render_timer.stop()
        if self.plotter is not None:
            self.renders += 1
            self.plotter.render()

    def changeEvent(self, event) -> None:
        """Follow a light/dark theme switch at runtime (the 2D view does the same)."""
        super().changeEvent(event)
        if event.type() in (QEvent.Type.PaletteChange, QEvent.Type.ApplicationPaletteChange) \
                and self.plotter is not None:
            dark = self.palette().window().color().lightness() < 128
            if dark != getattr(self, "_dark", None):
                self._set_background()
                if self._atoms is not None:
                    self._rebuild(False)  # label and charge-bar text colors depend on the theme
                self._render()

    def _set_background(self) -> None:
        if self.plotter is None:
            return
        self._dark = self.palette().window().color().lightness() < 128
        self.plotter.set_background("#1e1f24" if self._dark else "#f4f5f7", top="#2b2d35" if self._dark else "#ffffff")

    def _set_anti_aliasing(self, n_atoms: int) -> None:
        """SSAA looks best but renders the scene several times; FXAA keeps big slabs fluid."""
        want = "ssaa" if n_atoms <= 2000 else "fxaa"
        if want != self._aa:
            if self._aa:
                self.plotter.disable_anti_aliasing()
            self.plotter.enable_anti_aliasing(want)
            self._aa = want

    def set_interactive(self, on: bool) -> None:
        """Cheaper anti-aliasing while coordinates animate: FXAA instead of SSAA (which renders the scene
        several times per frame). False restores the usual choice for the atom count."""
        if self.plotter is None:
            return
        if on and self._aa == "ssaa":
            self.plotter.disable_anti_aliasing()
            self.plotter.enable_anti_aliasing("fxaa")
            self._aa = "fxaa"
        elif not on:
            self._set_anti_aliasing(len(self._atoms) if self._atoms is not None else 0)

    def _add(self, name: str, actor) -> None:
        self.plotter.add_actor(actor, name=name, reset_camera=False, pickable=False, render=False)
        self._names.add(name)

    def _remove(self, name: str) -> None:
        if name in self._names:
            self.plotter.remove_actor(name, render=False)
            self._names.discard(name)

    def _remove_scene(self) -> None:
        if self.plotter is None:
            return
        for name in list(self._names):
            self._remove(name)
        self._atom_pd = self._bond_pd = self._wire_pd = self._label_pd = None
        self._stub_tips = np.zeros((0, 3))
        self._set_measure_text("")

    def _atom_colors(self) -> np.ndarray:
        atoms = self._atoms
        if self.color_mode == "charge" and self._charges is not None:
            return diverging_rgb(self._charges, self._charge_vmax())
        if self.color_mode == "tag":
            return tag_rgb(atoms.get_tags())
        return (jmol_colors[atoms.numbers] * 255).round().astype(np.uint8)

    def _charge_vmax(self) -> float:
        return float(max(np.max(np.abs(self._charges)), 1e-3)) if self._charges is not None else 1.0

    def _rebuild(self, reset_camera: bool) -> None:
        """Full scene rebuild: new molecule, style, color mode or hydrogen visibility."""
        if self.plotter is None:
            return
        self._remove_scene()
        if self._atoms is None:
            self._render()
            self.sceneRebuilt.emit()
            return
        import pyvista as pv
        import vtk

        atoms = self._atoms
        n = len(atoms)
        self._set_anti_aliasing(n)
        keep = np.ones(n, bool) if self.show_hydrogens and not self._build_view() else atoms.numbers != 1
        self._vis = np.where(keep)[0]
        self._radii = atom_radii(atoms.numbers, self.style)
        self._colors = self._atom_colors()
        b = self._bonds
        self._bond_vis = b[keep[b[:, 0]] & keep[b[:, 1]]] if len(b) else np.zeros((0, 2), int)
        res = sphere_resolution(len(self._vis))
        material = {"specular": 0.35, "specular_power": 25.0, "diffuse": 0.85, "ambient": 0.12}

        if self.style != "Wireframe" and len(self._vis):
            pd = pv.PolyData(atoms.positions[self._vis])
            pd.point_data["radius"] = self._radii[self._vis]
            pd.point_data["rgb"] = self._colors[self._vis]
            sphere = vtk.vtkSphereSource()
            sphere.SetRadius(1.0)
            sphere.SetThetaResolution(res)
            sphere.SetPhiResolution(res)
            self._atom_pd = pd
            self._add("atoms", self._glyph_actor(pd, sphere.GetOutputPort(), "radius", material))

        if len(self._bond_vis) and self.style in BOND_RADIUS:
            cyl = vtk.vtkCylinderSource()        # along +y, centered; turn it into x in [0, 1]
            cyl.SetRadius(BOND_RADIUS[self.style])
            cyl.SetHeight(1.0)
            cyl.SetResolution(max(8, res // 2 + 4))
            cyl.CappingOff()
            tf = vtk.vtkTransform()
            tf.Translate(0.5, 0, 0)
            tf.RotateZ(-90)
            tfp = vtk.vtkTransformPolyDataFilter()
            tfp.SetTransform(tf)
            tfp.SetInputConnection(cyl.GetOutputPort())
            starts, dirs, scale = self._bond_arrays()
            pd = pv.PolyData(starts)
            pd.point_data["dir"] = dirs
            pd.point_data["scale"] = scale
            pd.point_data["rgb"] = np.vstack([self._colors[self._bond_vis[:, 0]], self._colors[self._bond_vis[:, 1]]])
            self._bond_pd = pd
            self._add("bonds", self._glyph_actor(pd, tfp.GetOutputPort(), "scale", material, orient="dir"))
        elif len(self._bond_vis) and self.style == "Wireframe":
            self._wire_pd = self._wire_mesh()
            actor = self._mesh_actor(self._wire_pd, line_width=3.0, rgb="rgb")
            self._add("bonds", actor)

        if self.show_cell and atoms.cell.rank == 3 and any(atoms.pbc):
            corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], float) @ atoms.cell.array
            edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
            lines = np.hstack([[2, i, j] for i, j in edges])
            self._add("cell", self._mesh_actor(pv.PolyData(corners, lines=lines), color=(0.54, 0.56, 0.6),
                                               line_width=1.5))

        if self.color_mode == "charge" and self._charges is not None:
            self._add_charge_bar()
        self._draw_stubs()
        self._update_labels()
        self._update_highlights()
        if reset_camera:
            self._reset_view()
        self.plotter.renderer.ResetCameraClippingRange()
        self._render()
        self.sceneRebuilt.emit()

    def _glyph_actor(self, pd, source_port, scale_array: str, material: dict, orient: str | None = None):
        import vtk

        m = vtk.vtkGlyph3DMapper()
        m.SetInputData(pd)
        m.SetSourceConnection(source_port)
        m.SetScaleArray(scale_array)
        m.ScalingOn()
        if orient:
            m.SetScaleModeToScaleByVectorComponents()
            m.SetOrientationArray(orient)
            m.SetOrientationModeToDirection()
            m.OrientOn()
        else:
            m.SetScaleModeToScaleByMagnitude()
            m.OrientOff()
        m.SetScalarModeToUsePointFieldData()
        m.SelectColorArray("rgb")
        m.SetColorModeToDirectScalars()
        m.ScalarVisibilityOn()
        actor = vtk.vtkActor()
        actor.SetMapper(m)
        prop = actor.GetProperty()
        prop.SetInterpolationToPhong()
        prop.SetSpecular(material["specular"])
        prop.SetSpecularPower(material["specular_power"])
        prop.SetDiffuse(material["diffuse"])
        prop.SetAmbient(material["ambient"])
        return actor

    def _mesh_actor(self, pd, color=None, line_width: float = 1.0, rgb: str | None = None, opacity: float = 1.0):
        import vtk

        m = vtk.vtkPolyDataMapper()
        m.SetInputData(pd)
        if rgb:
            m.SetScalarModeToUseCellFieldData()
            m.SelectColorArray(rgb)
            m.SetColorModeToDirectScalars()
            m.ScalarVisibilityOn()
        else:
            m.ScalarVisibilityOff()
        actor = vtk.vtkActor()
        actor.SetMapper(m)
        prop = actor.GetProperty()
        if color is not None:
            prop.SetColor(*color)
        prop.SetLineWidth(line_width)
        prop.SetOpacity(opacity)
        prop.LightingOff()
        return actor

    def _build_view(self) -> bool:
        """Build mode with stubs to draw: hydrogens are shown as stubs, not atoms."""
        return self.mode == "build" and self._stubs is not None

    def _stub_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Start points (on the atom's drawn surface), directions and orders of the stubs on atoms that exist."""
        a, d, o = self._stubs
        ok = (a >= 0) & (a < len(self._atoms))
        a, d, o = a[ok], d[ok], o[ok]
        r = self._radii[a] if len(a) else np.zeros(0)
        starts = self._atoms.positions[a] + d * r[:, None]
        return starts, d, o

    def _draw_stubs(self) -> None:
        """The free valences as short cylinders (build mode); their tips are what ``pick_stub`` hits."""
        self._remove("build_stubs")
        self._stub_tips = np.zeros((0, 3))
        if not self._build_view() or self._atoms is None:
            return
        import pyvista as pv
        import vtk

        starts, dirs, orders = self._stub_arrays()
        if len(self._stubs[0]) != len(starts):
            return                           # stubs of another structure: drawn after the next show_rdkit
        self._stub_tips = starts + dirs * STUB_LENGTH
        if not len(starts):
            return
        cyl = vtk.vtkCylinderSource()            # along +y, centered; turn it into x in [0, 1] as for bonds
        cyl.SetRadius(STUB_RADIUS)
        cyl.SetHeight(1.0)
        cyl.SetResolution(12)
        tf = vtk.vtkTransform()
        tf.Translate(0.5, 0, 0)
        tf.RotateZ(-90)
        tfp = vtk.vtkTransformPolyDataFilter()
        tfp.SetTransform(tf)
        tfp.SetInputConnection(cyl.GetOutputPort())
        pd = pv.PolyData(starts)
        pd.point_data["dir"] = dirs
        pd.point_data["scale"] = np.column_stack([np.full(len(starts), STUB_LENGTH), np.ones(len(starts)),
                                                  np.ones(len(starts))])
        pd.point_data["rgb"] = np.array([STUB_RGB.get(int(k), STUB_RGB[1]) for k in orders], np.uint8)
        material = {"specular": 0.2, "specular_power": 10.0, "diffuse": 0.8, "ambient": 0.25}
        self._add("build_stubs", self._glyph_actor(pd, tfp.GetOutputPort(), "scale", material, orient="dir"))

    def _bond_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Two half-cylinders per bond: start points, directions and (length, 1, 1) scales."""
        pos = self._atoms.positions
        a, c = pos[self._bond_vis[:, 0]], pos[self._bond_vis[:, 1]]
        mid = 0.5 * (a + c)
        starts = np.vstack([a, c])
        dirs = np.vstack([mid - a, mid - c])
        length = np.linalg.norm(dirs, axis=1)
        scale = np.column_stack([length, np.ones_like(length), np.ones_like(length)])
        return starts, dirs, scale

    def _wire_mesh(self):
        import pyvista as pv

        pos = self._atoms.positions
        b = self._bond_vis
        a, c = pos[b[:, 0]], pos[b[:, 1]]
        mid = 0.5 * (a + c)
        nb = len(b)
        pts = np.vstack([a, mid, c])
        lines = np.vstack([np.column_stack([np.full(nb, 2), np.arange(nb), np.arange(nb, 2 * nb)]),
                           np.column_stack([np.full(nb, 2), np.arange(2 * nb, 3 * nb), np.arange(nb, 2 * nb)])])
        mesh = pv.PolyData(pts, lines=lines.ravel())
        mesh.cell_data["rgb"] = np.vstack([self._colors[b[:, 0]], self._colors[b[:, 1]]])
        return mesh

    def _add_charge_bar(self) -> None:
        import vtk

        vmax = self._charge_vmax()
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(256)
        lut.SetRange(-vmax, vmax)
        rgb = diverging_rgb(np.linspace(-vmax, vmax, 256), vmax) / 255.0
        for i, (r, g, b) in enumerate(rgb):
            lut.SetTableValue(i, r, g, b, 1.0)
        lut.Build()
        bar = vtk.vtkScalarBarActor()
        bar.SetLookupTable(lut)
        bar.SetTitle("q (e)")
        bar.SetNumberOfLabels(5)
        bar.SetLabelFormat("%+.2f")
        bar.SetOrientationToVertical()
        bar.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        bar.GetPositionCoordinate().SetValue(0.90, 0.08)
        bar.SetWidth(0.07)
        bar.SetHeight(0.45)
        fg = (0.92, 0.92, 0.92) if getattr(self, "_dark", False) else (0.1, 0.1, 0.12)
        for tp in (bar.GetTitleTextProperty(), bar.GetLabelTextProperty()):
            tp.SetColor(*fg)
            tp.ShadowOff()
            tp.ItalicOff()
            tp.BoldOff()
        self._add("charge_bar", bar)

    # ------------------------------------------------------------------ overlays
    def _text_actor(self, points: np.ndarray, texts: list[str], size: int, bold: bool = False,
                    color=None, background: bool = False):
        import pyvista as pv
        import vtk

        pd = pv.PolyData(np.asarray(points, float).reshape(-1, 3))
        arr = vtk.vtkStringArray()
        arr.SetName("label")
        for t in texts:
            arr.InsertNextValue(t)
        pd.GetPointData().AddArray(arr)
        m = vtk.vtkLabeledDataMapper()
        m.SetInputData(pd)
        m.SetLabelModeToLabelFieldData()
        m.SetFieldDataName("label")
        tp = m.GetLabelTextProperty()
        # No devicePixelRatio factor: pyvistaqt sets the render window DPI to 72 x DPR, and VTK already
        # scales point sizes by it (multiplying again drew labels 1.5x/2x too large at 150%/200%).
        tp.SetFontSize(int(size))
        tp.SetBold(bold)
        tp.ItalicOff()
        tp.SetJustificationToCentered()
        tp.SetVerticalJustificationToCentered()
        fg = color if color is not None else ((0.95, 0.95, 0.95) if getattr(self, "_dark", False) else (0.08, 0.08, 0.1))
        tp.SetColor(*fg)
        if background:                   # measurement: sit just above the anchor, on a dark plate
            tp.SetVerticalJustificationToBottom()
            tp.SetBackgroundColor(0.08, 0.08, 0.1)
            tp.SetBackgroundOpacity(0.7)
            tp.ShadowOff()
        else:
            tp.ShadowOn()
        actor = vtk.vtkActor2D()
        actor.SetMapper(m)
        return actor, pd

    def _label_texts(self, idx: np.ndarray) -> list[str]:
        mode = self.label_mode
        if mode == "charge":
            if self._charges is None:
                return []
            return [f"{q:+.2f}" for q in self._charges[idx]]
        sym = [chemical_symbols[z] for z in self._atoms.numbers[idx]]
        if mode == "element":
            return sym
        if mode == "index":
            return [str(i) for i in idx]
        return [f"{s}{i}" for s, i in zip(sym, idx)]

    def _update_labels(self) -> None:
        if self.plotter is None:
            return
        self._remove("labels")
        self._label_pd = None
        if self._atoms is None or self.label_mode == "none" or len(self._vis) == 0 or len(self._vis) > MAX_LABELS:
            return
        texts = self._label_texts(self._vis)
        if not texts:
            return
        actor, self._label_pd = self._text_actor(self._atoms.positions[self._vis], texts, 12, bold=True)
        self._add("labels", actor)

    def _update_highlights(self) -> None:
        """Halos and measurement only; atoms and bonds are untouched (cheap on every selection change)."""
        if self.plotter is None:
            return
        import pyvista as pv
        import vtk

        self._remove("halo")
        self._remove("measure_lines")
        self._remove("measure_label")
        sel = [i for i in self.selection.atoms if self._atoms is not None and i < len(self._atoms)]
        if self._atoms is None or not sel:
            self._set_measure_text("")
            return
        pos = self._atoms.positions
        visible = set(self._vis.tolist())
        shown = np.array([i for i in sel if i in visible], int)
        if len(shown):
            pd = pv.PolyData(pos[shown])
            pd.point_data["radius"] = np.maximum(self._radii[shown] + HALO_PAD, 0.45)
            pd.point_data["rgb"] = np.tile((np.array(HALO_COLOR) * 255).astype(np.uint8), (len(shown), 1))
            sphere = vtk.vtkSphereSource()
            sphere.SetRadius(1.0)
            sphere.SetThetaResolution(24)
            sphere.SetPhiResolution(24)
            actor = self._glyph_actor(pd, sphere.GetOutputPort(), "radius",
                                      {"specular": 0.0, "specular_power": 1.0, "diffuse": 0.6, "ambient": 0.5})
            actor.GetProperty().SetOpacity(0.38)
            self._add("halo", actor)

        # 2-4 atoms only: with a larger (box) selection a "last four atoms" dihedral is noise
        m = measure(pos, sel) if 2 <= len(sel) <= 4 else None
        text = format_measurement(m)
        if m is not None:
            used = sel[-4:]
            pairs = list(zip(range(len(used) - 1), range(1, len(used))))
            p = pos[used]
            pts, lines = dashed_segments(p, pairs)
            if len(pts):
                self._add("measure_lines", self._mesh_actor(pv.PolyData(pts, lines=lines), color=MEASURE_COLOR,
                                                            line_width=3.0))
            # distance: bond midpoint; angle: the vertex atom; dihedral: middle of the central bond
            anchor = p[1] if len(used) == 3 else 0.5 * (p[len(used) // 2 - 1] + p[len(used) // 2])
            actor, _ = self._text_actor(anchor[None, :], [text], 15, bold=True, color=(1.0, 0.9, 0.55),
                                        background=True)
            self._add("measure_label", actor)
        self._set_measure_text(text)

    def _set_measure_text(self, text: str) -> None:
        if text != self._measure_text:
            self._measure_text = text
            self.measurementChanged.emit(text)

    @property
    def measurement_text(self) -> str:
        return self._measure_text

    def _on_selection_changed(self, _atoms: list) -> None:
        self._update_highlights()
        self._render()

    def _refresh_geometry(self) -> None:
        """After coordinates change: update instanced arrays in place, no mesh rebuild."""
        if self.plotter is None or self._atoms is None:
            return
        pos = self._atoms.positions
        if self._atom_pd is not None:
            self._atom_pd.points = pos[self._vis].copy()
        if self._bond_pd is not None:
            starts, dirs, scale = self._bond_arrays()
            self._bond_pd.points = starts
            self._bond_pd.point_data["dir"] = dirs
            self._bond_pd.point_data["scale"] = scale
        if self._wire_pd is not None:
            nb = len(self._bond_vis)
            a, c = pos[self._bond_vis[:, 0]], pos[self._bond_vis[:, 1]]
            self._wire_pd.points = np.vstack([a, 0.5 * (a + c), c]) if nb else self._wire_pd.points
        if self._label_pd is not None:
            self._label_pd.points = pos[self._vis].copy()
        if "build_stubs" in self._names:
            self._draw_stubs()
        self._update_highlights()
        self.plotter.renderer.ResetCameraClippingRange()
        self._render()


# Backwards-compatible name used by main_window and older code
MoleculeViewer = MoleculeViewer3D
