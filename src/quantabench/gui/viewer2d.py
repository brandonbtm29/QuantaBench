"""2D structure viewer: a crisp RDKit depiction you can pick, pan, zoom and rotate.

Why it is built this way:

* **Vector all the way.** RDKit's ``MolDraw2DSVG`` writes the drawing and ``QGraphicsSvgItem`` paints
  it as vectors at whatever zoom the view has, so labels and bonds stay sharp at any magnification
  (no bitmap is ever scaled).
* **Same atom indices as the 3D view.** The molecule arrives with explicit hydrogens (the pipeline's
  representation). By default the depiction hides them (implicit H, as chemists draw structures), so
  depicted atom ``i`` is not full atom ``i``. Every depicted atom carries its full index, and picking,
  highlighting, index labels and charges all go through that map, so a click here selects the same
  atom in the 3D viewer through the shared :class:`~quantabench.gui.selection.SelectionModel`.
* **Rotation rotates the coordinates, not the picture.** "Grab and rotate" turns the 2D coordinates
  about their centroid and re-renders, so atom labels stay upright and wedges are recomputed from the
  new geometry (also after a mirror flip, which would otherwise invert the drawn stereochemistry).
* **Clean zig-zags for long chains.** CoordGen gives the best general layouts (rings, macrocycles), but
  it draws long perfluoroalkyl chains as crankshafts or curls, and RDKit's legacy depictor draws CF2
  chains as straight lines. When the longest acyclic heavy-atom path is long enough, it is pinned to
  an ideal 120° zig-zag and RDKit's depictor places everything else around it (the two F of each CF2
  then splay outward, the way PFAS are drawn in the literature).
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
from PySide6.QtCore import QByteArray, QEvent, QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QKeyEvent, QMouseEvent, QPainter, QWheelEvent
from PySide6.QtWidgets import QApplication, QGraphicsScene, QGraphicsView, QLabel, QVBoxLayout, QWidget
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Geometry import Point2D, Point3D

from quantabench.gui.selection import ACCENT_HEX, ACCENT_RGB, SelectionModel

__all__ = ["ACCENT_HEX", "ACCENT_RGB", "MoleculeViewer2D", "build_depiction", "compute_2d_coords"]
#: Opacity of the highlight discs and bond bands drawn under the selected atoms.
ACCENT_ALPHA = 0.6

#: Bond length on screen at zoom 1 (px). Fixed so a rotation never rescales the drawing.
BOND_PX = 40.0
#: Pick radius as a fraction of the bond length: generous enough for a label, smaller than half a bond.
PICK_RADIUS = 0.42
#: Longest acyclic heavy-atom path (atoms) from which the chain is laid out as an explicit zig-zag.
ZIGZAG_MIN_ATOMS = 6
#: RDKit's standard 2D bond length (Å-like depiction units).
DEPICTION_BOND = 1.5
FULL_INDEX_PROP = "_qb_full_index"

ZOOM_MIN, ZOOM_MAX = 0.05, 60.0
FIT_MAX_SCALE = 2.5          # don't blow methane up to fill a 4K screen
KEY_ROTATE_DEG = 15.0
KEY_ZOOM = 1.25


# ---------------------------------------------------------------------------------------- depiction
def _longest_acyclic_path(mol: Chem.Mol) -> list[int]:
    """Longest simple path through acyclic heavy atoms (each acyclic component is a tree, so two BFS
    sweeps find its diameter exactly)."""
    ok = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1 and not a.IsInRing()}
    adj = {i: [n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() in ok] for i in ok}

    def farthest(start: int) -> tuple[int, dict[int, int | None]]:
        prev: dict[int, int | None] = {start: None}
        queue = [start]
        for u in queue:
            for v in adj[u]:
                if v not in prev:
                    prev[v] = u
                    queue.append(v)
        return queue[-1], prev

    best: list[int] = []
    seen: set[int] = set()
    for s in sorted(ok):
        if s in seen:
            continue
        a, prev = farthest(s)
        seen |= set(prev)
        b, prev = farthest(a)
        path = [b]
        while prev[path[-1]] is not None:
            path.append(prev[path[-1]])
        if len(path) > len(best):
            best = path
    return best


def compute_2d_coords(mol: Chem.Mol) -> None:
    """Give ``mol`` a single 2D conformer: CoordGen, or a pinned zig-zag for long acyclic chains.

    Bond lengths are normalized to RDKit's 1.5 units and the result is straightened (most bonds at
    30°/90° to the x axis) with minimal rotation, so long molecules come out horizontal.
    """
    mol.RemoveAllConformers()
    path = _longest_acyclic_path(mol)
    prefer = rdDepictor.GetPreferCoordGen()
    try:
        if len(path) >= ZIGZAG_MIN_ATOMS:
            rdDepictor.SetPreferCoordGen(False)  # CoordGen does not honour a coordinate map here
            dx = DEPICTION_BOND * math.cos(math.radians(30.0))
            dy = DEPICTION_BOND * 0.5
            cmap = {atom: Point2D(k * dx, (k % 2) * dy) for k, atom in enumerate(path)}
            rdDepictor.Compute2DCoords(mol, coordMap=cmap)
        else:
            rdDepictor.SetPreferCoordGen(True)
            rdDepictor.Compute2DCoords(mol)
            if mol.GetNumBonds():
                rdDepictor.NormalizeDepiction(mol, canonicalize=1)
    finally:
        rdDepictor.SetPreferCoordGen(prefer)
    if mol.GetNumBonds():
        rdDepictor.StraightenDepiction(mol, minimizeRotation=True)


def build_depiction(mol: Chem.Mol, show_hydrogens: bool = False) -> tuple[Chem.Mol, list[int]]:
    """Depicted molecule (2D coordinates) and the map ``depicted index -> full index``.

    With hydrogens hidden, hydrogens on tetrahedral stereocentres are kept explicit (so a wedge can be
    drawn to them, as RDKit does by default), and every other H becomes implicit on its heavy atom.
    """
    m = Chem.Mol(mol)
    m.RemoveAllConformers()
    for a in m.GetAtoms():
        a.SetIntProp(FULL_INDEX_PROP, a.GetIdx())
        a.SetAtomMapNum(0)
    if not show_hydrogens:
        stereo = (Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
        keep = set()
        for a in m.GetAtoms():
            if a.GetChiralTag() in stereo:
                hs = [n.GetIdx() for n in a.GetNeighbors() if n.GetAtomicNum() == 1]
                keep.update(hs[:1])
        for i in keep:  # mapped atoms survive RemoveHs; the map number is only a temporary marker
            m.GetAtomWithIdx(i).SetAtomMapNum(1)
        params = Chem.RemoveHsParameters()
        params.removeMapped = False
        m = Chem.RemoveHs(m, params, sanitize=False)
        for a in m.GetAtoms():
            a.SetAtomMapNum(0)
        try:
            Chem.SanitizeMol(m)
        except Exception:  # odd valences from a file: still draw it, ring perception is all we need
            m.UpdatePropertyCache(strict=False)
            Chem.FastFindRings(m)
    else:
        try:
            m.GetRingInfo().NumRings()
        except RuntimeError:  # ring info not initialized on an unsanitized input
            Chem.FastFindRings(m)
    for b in m.GetBonds():
        b.SetBondDir(Chem.BondDir.NONE)
    compute_2d_coords(m)
    return m, [a.GetIntProp(FULL_INDEX_PROP) for a in m.GetAtoms()]


def _molecule_key(mol: Chem.Mol) -> tuple:
    """Identity of a molecular graph in a given atom order (conformers excluded)."""
    atoms = tuple((a.GetAtomicNum(), a.GetFormalCharge(), a.GetIsotope()) for a in mol.GetAtoms())
    bonds = tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx(), str(b.GetBondType())) for b in mol.GetBonds()))
    return atoms, bonds


def _cip_labels(mol: Chem.Mol) -> tuple[dict[int, str], dict[int, str]]:
    """CIP descriptors with the new (rule-complete) labeler; atom and bond labels keyed by index."""
    m = Chem.Mol(mol)
    try:
        from rdkit.Chem import rdCIPLabeler

        rdCIPLabeler.AssignCIPLabels(m)
    except Exception:  # pathological graphs can exceed the labeler's limits: label nothing
        return {}, {}
    atoms = {a.GetIdx(): a.GetProp("_CIPCode") for a in m.GetAtoms() if a.HasProp("_CIPCode")}
    bonds = {b.GetIdx(): b.GetProp("_CIPCode") for b in m.GetBonds() if b.HasProp("_CIPCode")}
    return atoms, bonds


def prepare_for_drawing(depict: Chem.Mol) -> Chem.Mol:
    """Kekulized copy with wedges recomputed from the *current* 2D coordinates (no atoms added)."""
    m = Chem.Mol(depict)
    for b in m.GetBonds():
        b.SetBondDir(Chem.BondDir.NONE)
    try:
        return rdMolDraw2D.PrepareMolForDrawing(m, kekulize=True, addChiralHs=False, wedgeBonds=True)
    except Exception:  # unkekulizable aromatic input: draw it aromatic rather than not at all
        return rdMolDraw2D.PrepareMolForDrawing(m, kekulize=False, addChiralHs=False, wedgeBonds=True)


# ---------------------------------------------------------------------------------------- widget
class MoleculeViewer2D(QWidget):
    """RDKit 2D depiction in a ``QGraphicsView``, sharing atom selection with the 3D viewer.

    Mouse: click an atom to select it, Ctrl+click to toggle, click empty space to clear; left-drag on
    empty space pans; Shift+left-drag or right-drag rotates in-plane; the wheel zooms at the cursor;
    double-click fits. Keys: ``+``/``-`` zoom, ``[``/``]`` rotate by 15°, ``F`` fits.
    """

    def __init__(self, parent=None, selection: SelectionModel | None = None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.error = ""
        self.show_hydrogens = False
        self.atom_indices = False
        self.charges: list[float] | None = None
        self.cip_labels = False
        self.fold_hydrogen_charges = True
        self.view: _DepictionView | None = None
        self.scene: QGraphicsScene | None = None
        self._item = None
        self._renderer = None
        self._message = None
        self._full: Chem.Mol | None = None
        self._key: tuple | None = None
        self._depict: Chem.Mol | None = None
        self._to_full: list[int] = []
        self._xform = np.eye(2)                  # accumulated user rotation/flips, re-applied on H toggle
        self._scene_xy = np.zeros((0, 2))         # depicted atom centres in scene coordinates
        self._highlighted: list[int] = []
        self._svg = ""
        self._user_moved = False
        self._selection: SelectionModel | None = None
        try:
            from PySide6.QtSvg import QSvgRenderer  # noqa: F401
            from PySide6.QtSvgWidgets import QGraphicsSvgItem  # noqa: F401

            self.scene = QGraphicsScene(self)
            self.view = _DepictionView(self)
            self.view.setScene(self.scene)
            lay.addWidget(self.view)
        except Exception as exc:  # QtSvg missing from a trimmed PySide6 build
            self.error = f"2D view unavailable: {type(exc).__name__}: {exc}"
            msg = QLabel(self.error)
            msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
            msg.setWordWrap(True)
            lay.addWidget(msg)
        self.set_selection_model(selection or SelectionModel(self))

    # ------------------------------------------------------------------ selection
    @property
    def selection(self) -> SelectionModel:
        return self._selection

    def set_selection_model(self, selection: SelectionModel) -> None:
        if self._selection is not None:
            try:
                self._selection.changed.disconnect(self._on_selection_changed)
            except (RuntimeError, TypeError):
                pass
        self._selection = selection
        selection.changed.connect(self._on_selection_changed)
        self._render()

    def _on_selection_changed(self, _atoms: list) -> None:
        self._render()

    # ------------------------------------------------------------------ public
    def show_rdkit(self, mol: Chem.Mol, conf_id: int = -1, reset_camera: bool = True) -> None:
        """Depict ``mol`` (explicit H and 3D conformers expected; ``conf_id`` is accepted to mirror the
        3D viewer, but a 2D depiction does not depend on the conformer). Showing another conformer of
        the same molecule keeps the user's rotation and zoom."""
        key = _molecule_key(mol)
        same = key == self._key and self._depict is not None
        self._full = mol
        self._key = key
        if not same:
            self._xform = np.eye(2)
            self._rebuild()
        else:
            self._render()
        if reset_camera or not same:
            self.fit()

    def set_show_hydrogens(self, on: bool) -> None:
        self.show_hydrogens = bool(on)
        if self._full is not None:
            self._rebuild()
            self.fit()

    def set_options(self, atom_indices: bool = False, charges=None, cip_labels: bool = False,
                    fold_hydrogen_charges: bool = True) -> None:
        """Atom notes. ``charges`` is one value per *full* atom (e); with hydrogens hidden and
        ``fold_hydrogen_charges`` on, a heavy atom shows its group charge (itself plus its hidden H),
        so the drawn numbers still add up to the net charge. Index labels are full-molecule indices."""
        if charges is not None:
            charges = [float(q) for q in charges]
            if self._full is not None and len(charges) != self._full.GetNumAtoms():
                raise ValueError(f"{len(charges)} charges for {self._full.GetNumAtoms()} atoms")
        self.atom_indices = bool(atom_indices)
        self.charges = charges
        self.cip_labels = bool(cip_labels)
        self.fold_hydrogen_charges = bool(fold_hydrogen_charges)
        self._render()

    def clear(self) -> None:
        self._full = self._depict = self._key = None
        self._to_full = []
        self._scene_xy = np.zeros((0, 2))
        self._highlighted = []
        self._svg = ""
        self._xform = np.eye(2)
        if self.scene is not None:
            self.scene.clear()
        self._item = self._renderer = self._message = None

    def rotate(self, degrees: float) -> None:
        """Rotate the drawing in-plane about its centroid; positive is counter-clockwise on screen."""
        t = math.radians(degrees)
        self._transform(np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]]))

    def flip_horizontal(self) -> None:
        """Mirror left-right. Wedges are recomputed, so the drawn stereochemistry stays correct."""
        self._transform(np.diag([-1.0, 1.0]))

    def flip_vertical(self) -> None:
        self._transform(np.diag([1.0, -1.0]))

    def fit(self) -> None:
        if self.view is None or self._item is None:
            return
        rect = self._item.sceneBoundingRect()
        m = 0.06 * max(rect.width(), rect.height())
        self.view.fitInView(rect.adjusted(-m, -m, m, m), Qt.AspectRatioMode.KeepAspectRatio)
        s = self.view.transform().m11()
        if s > FIT_MAX_SCALE:
            self.view.scale(FIT_MAX_SCALE / s, FIT_MAX_SCALE / s)
        self.view.centerOn(rect.center())
        self._user_moved = False

    def zoom(self, factor: float) -> None:
        if self.view is None:
            return
        s = self.view.transform().m11()
        factor = min(max(factor, ZOOM_MIN / s), ZOOM_MAX / s)
        self.view.scale(factor, factor)
        self._user_moved = True

    # mirror MoleculeViewer so the main window can drive both views the same way
    def reset_camera(self) -> None:
        self.fit()

    def screenshot(self, path: str) -> None:
        self.export(path)

    def close_viewer(self) -> None:
        pass

    def export(self, path, scale: float = 3.0, background: str | None = None, dark: bool = False) -> Path:
        """Write the current depiction (rotation, highlights, notes) as ``.svg`` or ``.png``.

        Exports use the light palette by default (figures go into papers and slides) and a transparent
        background unless ``background`` is a colour such as ``"white"``. PNG is rendered from the same
        SVG at ``scale`` times the on-screen size.
        """
        path = Path(path)
        if self._depict is None:
            raise RuntimeError("nothing to export: no molecule shown")
        svg, _ = self._make_svg(dark=dark, background=background)
        suffix = path.suffix.lower()
        if suffix == ".svg":
            path.write_text(svg, encoding="latin-1", errors="xmlcharrefreplace")
        elif suffix == ".png":
            from PySide6.QtSvg import QSvgRenderer

            renderer = QSvgRenderer(QByteArray(_svg_bytes(svg)))
            size = renderer.defaultSize() * float(scale)
            img = QImage(size, QImage.Format.Format_ARGB32_Premultiplied)
            img.fill(QColor(background) if background else Qt.GlobalColor.transparent)
            painter = QPainter(img)
            painter.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing)
            renderer.render(painter)
            painter.end()
            if not img.save(str(path)):
                raise OSError(f"could not write {path}")
        else:
            raise ValueError(f"unsupported export format {path.suffix!r} (use .svg or .png)")
        return path

    # ------------------------------------------------------------------ introspection (tests, tooltips)
    @property
    def depicted_to_full(self) -> list[int]:
        return list(self._to_full)

    @property
    def coords_2d(self) -> np.ndarray:
        """Depicted atom 2D coordinates (depiction units, y up), shape (n, 2)."""
        if self._depict is None:
            return np.zeros((0, 2))
        return self._depict.GetConformer().GetPositions()[:, :2].copy()

    @property
    def highlighted_atoms(self) -> list[int]:
        """Full indices highlighted in the last render (hidden hydrogens cannot be highlighted)."""
        return list(self._highlighted)

    @property
    def svg(self) -> str:
        return self._svg

    def atom_scene_pos(self, full_index: int) -> QPointF | None:
        try:
            k = self._to_full.index(int(full_index))
        except ValueError:
            return None
        x, y = self._scene_xy[k]
        return QPointF(float(x), float(y))

    def atom_at(self, scene_point) -> int | None:
        """Full index of the atom under ``scene_point`` (a QPointF or (x, y)), or None."""
        if not len(self._scene_xy):
            return None
        p = np.array([scene_point.x(), scene_point.y()] if isinstance(scene_point, QPointF) else scene_point,
                     dtype=float)
        d = np.linalg.norm(self._scene_xy - p, axis=1)
        k = int(np.argmin(d))
        return self._to_full[k] if d[k] <= PICK_RADIUS * BOND_PX else None

    def describe_atom(self, full_index: int) -> str:
        if self._full is None:
            return ""
        a = self._full.GetAtomWithIdx(full_index)
        text = f"{a.GetSymbol()} {full_index}"
        if a.GetFormalCharge():
            text += f" (formal charge {a.GetFormalCharge():+d})"
        if self.charges is not None:
            q = self.charges[full_index]
            text += f"\npartial charge {q:+.3f} e"
            group = self._group_charge(full_index)
            if not self.show_hydrogens and abs(group - q) > 1e-12:
                text += f"\nwith H: {group:+.3f} e"
        return text

    # ------------------------------------------------------------------ interaction (called by the view)
    def _click(self, scene_point: QPointF, toggle: bool) -> None:
        atom = self.atom_at(scene_point)
        if atom is None:
            if not toggle:
                self._selection.clear()
        elif toggle:
            self._selection.toggle(atom)
        else:
            self._selection.select([atom])

    # ------------------------------------------------------------------ internals
    def _is_dark(self) -> bool:
        return self.palette().window().color().lightness() < 128

    def _rebuild(self) -> None:
        if self._full is None:
            return
        try:
            self._depict, self._to_full = build_depiction(self._full, self.show_hydrogens)
            if not np.allclose(self._xform, np.eye(2)):
                self._apply_xform(self._xform)
            self.error = ""
        except Exception as exc:
            self._depict, self._to_full = None, []
            self._fail(f"2D depiction failed: {type(exc).__name__}: {exc}")
            return
        self._render()

    def _transform(self, mat: np.ndarray) -> None:
        if self._depict is None:
            return
        self._xform = mat @ self._xform
        self._apply_xform(mat)
        self._render()

    def _apply_xform(self, mat: np.ndarray) -> None:
        conf = self._depict.GetConformer()
        xy = conf.GetPositions()[:, :2]
        c = xy.mean(axis=0)
        new = (xy - c) @ mat.T + c
        for i, (x, y) in enumerate(new):
            conf.SetAtomPosition(i, Point3D(float(x), float(y), 0.0))

    def _group_charge(self, full_index: int) -> float:
        q = self.charges[full_index]
        if self.show_hydrogens or not self.fold_hydrogen_charges:
            return q
        shown = set(self._to_full)
        atom = self._full.GetAtomWithIdx(full_index)
        return q + sum(self.charges[n.GetIdx()] for n in atom.GetNeighbors()
                       if n.GetAtomicNum() == 1 and n.GetIdx() not in shown)

    def _notes(self, m: Chem.Mol) -> None:
        cip_atoms, cip_bonds = _cip_labels(m) if self.cip_labels else ({}, {})
        for k, a in enumerate(m.GetAtoms()):
            full = self._to_full[k]
            values = []                       # "12: +0.21" reads as "atom 12 has charge +0.21"
            if self.atom_indices:
                values.append(str(full))
            if self.charges is not None:
                values.append(f"{self._group_charge(full):+.2f}")
            parts = [f"({cip_atoms[k]})"] if k in cip_atoms else []
            if values:
                parts.append(": ".join(values))
            if parts:
                a.SetProp("atomNote", " ".join(parts))
        for b in m.GetBonds():
            if b.GetIdx() in cip_bonds:
                b.SetProp("bondNote", f"({cip_bonds[b.GetIdx()]})")

    def _make_svg(self, dark: bool, background: str | None = None) -> tuple[str, np.ndarray]:
        """SVG text and the drawn atom centres (SVG px) for the current depiction and selection."""
        m = prepare_for_drawing(self._depict)
        self._notes(m)
        full_to_dep = {f: k for k, f in enumerate(self._to_full)}
        sel = [full_to_dep[i] for i in (self._selection.atoms if self._selection else []) if i in full_to_dep]
        sel_set = set(sel)
        bonds = [b.GetIdx() for b in m.GetBonds()
                 if b.GetBeginAtomIdx() in sel_set and b.GetEndAtomIdx() in sel_set]
        colour = (*ACCENT_RGB, ACCENT_ALPHA)

        drawer = rdMolDraw2D.MolDraw2DSVG(-1, -1)  # flexicanvas: the canvas grows to fit the molecule
        opts = drawer.drawOptions()
        if dark:
            rdMolDraw2D.SetDarkMode(opts)
        opts.fixedBondLength = BOND_PX
        opts.prepareMolsBeforeDrawing = False      # done above, with wedges from the current coordinates
        opts.annotationFontScale = 0.6
        opts.fillHighlights = True
        opts.continuousHighlight = True
        opts.highlightRadius = 0.35
        opts.padding = 0.08
        if background:
            c = QColor(background)
            opts.setBackgroundColour((c.redF(), c.greenF(), c.blueF(), 1.0))
            opts.clearBackground = True
        else:
            opts.clearBackground = False           # transparent: the view's palette shows through
        drawer.DrawMolecule(m, highlightAtoms=sel, highlightAtomColors={i: colour for i in sel},
                            highlightBonds=bonds, highlightBondColors={i: colour for i in bonds})
        drawer.FinishDrawing()
        coords = np.array([[p.x, p.y] for p in (drawer.GetDrawCoords(i) for i in range(m.GetNumAtoms()))])
        return _clean_svg(drawer.GetDrawingText()), coords.reshape(-1, 2)

    def _render(self) -> None:
        if self.scene is None or self._depict is None:
            return
        from PySide6.QtSvg import QSvgRenderer
        from PySide6.QtSvgWidgets import QGraphicsSvgItem

        try:
            svg, coords = self._make_svg(self._is_dark())
        except Exception as exc:
            self._fail(f"2D drawing failed: {type(exc).__name__}: {exc}")
            return
        renderer = QSvgRenderer(QByteArray(_svg_bytes(svg)))
        if self._message is not None:
            self.scene.removeItem(self._message)
            self._message = None
        if self._item is None:
            self._item = QGraphicsSvgItem()
            # DeviceCoordinateCache re-rasterizes whenever the view transform changes: crisp and cheap
            self._item.setCacheMode(QGraphicsSvgItem.CacheMode.DeviceCoordinateCache)
            self.scene.addItem(self._item)
        self._item.setSharedRenderer(renderer)
        self._renderer = renderer                  # the item does not own a shared renderer
        # RDKit's flexicanvas rounds the canvas to whole pixels, so its scale drifts by ~1% as the
        # drawing turns; undo that on the item so a rotation never makes the molecule "breathe".
        k = 1.0
        xy = self.coords_2d
        if len(xy) > 1:
            drawn = np.sqrt(((coords - coords.mean(axis=0)) ** 2).sum(axis=1).mean())
            model = np.sqrt(((xy - xy.mean(axis=0)) ** 2).sum(axis=1).mean())
            if drawn > 0 and model > 0:
                k = (BOND_PX / DEPICTION_BOND) / (drawn / model)
        # put the depiction's centroid at the scene origin, so rotations pivot in place on screen
        centre = coords.mean(axis=0) if len(coords) else np.zeros(2)
        self._item.setScale(k)
        self._item.setPos(QPointF(-centre[0] * k, -centre[1] * k))
        self._scene_xy = (coords - centre) * k
        self._svg = svg
        full_to_dep = set(self._to_full)
        self._highlighted = [i for i in self._selection.atoms if i in full_to_dep] if self._selection else []

    def _fail(self, message: str) -> None:
        self.error = message
        if self.scene is None:
            return
        self.scene.clear()
        self._item = self._renderer = None
        self._scene_xy = np.zeros((0, 2))
        self._message = self.scene.addText(message)
        self._message.setDefaultTextColor(self.palette().text().color())
        self._message.setTextWidth(400)
        if self.view is not None:
            self.view.resetTransform()
            self.view.centerOn(self._message)

    def changeEvent(self, event: QEvent) -> None:
        if event.type() in (QEvent.Type.PaletteChange, QEvent.Type.ApplicationPaletteChange):
            self._render()
        super().changeEvent(event)


_STYLE = re.compile(r"style='([^']*)'")
_HEX8 = re.compile(r"^#([0-9A-Fa-f]{6})([0-9A-Fa-f]{2})$")


_EMPTY_PATH = re.compile(r"<path\b[^>]*\sd=''[^>]*/>\n?")


def _clean_svg(svg: str) -> str:
    """Make RDKit's SVG strict SVG 1.1, which QtSvg (and Inkscape/Illustrator imports) expect.

    * Translucent highlights come out as 8-digit ``#RRGGBBAA`` hex (CSS4, not SVG 1.1), which QtSvg
      paints black: rewrite them as ``#RRGGBB`` plus ``fill-/stroke-opacity``. Later declarations win
      inside a style attribute, so the opacity goes into the declaration map rather than being appended.
    * A space inside an atom note becomes a path with empty ``d=''``; QtSvg warns on each one.
    """
    svg = _EMPTY_PATH.sub("", svg)

    def fix(match: re.Match) -> str:
        decls: dict[str, str] = {}
        for part in match.group(1).split(";"):
            if ":" in part:
                k, v = part.split(":", 1)
                decls[k.strip()] = v.strip()
        changed = False
        for prop in ("fill", "stroke"):
            m = _HEX8.match(decls.get(prop, ""))
            if m:
                decls[prop] = f"#{m.group(1)}"
                decls[f"{prop}-opacity"] = f"{int(m.group(2), 16) / 255:.3f}"
                changed = True
        if not changed:
            return match.group(0)
        if decls.get("fill") == decls.get("stroke") and "fill-opacity" in decls and "stroke-opacity" in decls:
            # a translucent stroke over the same translucent fill draws a darker ring around each disc
            decls["stroke"] = "none"
            decls.pop("stroke-opacity")
        return "style='" + ";".join(f"{k}:{v}" for k, v in decls.items()) + "'"

    return _STYLE.sub(fix, svg)


def _svg_bytes(svg: str) -> bytes:
    # RDKit declares iso-8859-1 in the XML header; encode to match (notes are ASCII anyway)
    return svg.encode("latin-1", errors="xmlcharrefreplace")


class _DepictionView(QGraphicsView):
    """Mouse and keyboard handling. Kept separate so the widget's public API stays small."""

    def __init__(self, owner: MoleculeViewer2D):
        super().__init__(owner)
        self._owner = owner
        self._mode: str | None = None     # None | "click" | "pan" | "rotate"
        self._press = QPointF()
        self._last = QPointF()
        self._last_angle = 0.0
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.TextAntialiasing
                            | QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.PreventContextMenu)  # right-drag rotates
        self.setSceneRect(QRectF(-1e5, -1e5, 2e5, 2e5))  # room to pan anywhere
        self.setMouseTracking(True)

    def _screen_angle(self, pos: QPointF) -> float:
        c = self.mapFromScene(QPointF(0.0, 0.0))   # the depiction's centroid
        return math.degrees(math.atan2(pos.y() - c.y(), pos.x() - c.x()))

    def mousePressEvent(self, event: QMouseEvent) -> None:
        pos = event.position()
        self._press = self._last = pos
        shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        if event.button() == Qt.MouseButton.RightButton or (event.button() == Qt.MouseButton.LeftButton and shift):
            self._mode = "rotate"
            self._last_angle = self._screen_angle(pos)
            self.viewport().setCursor(Qt.CursorShape.SizeAllCursor)
        elif event.button() == Qt.MouseButton.LeftButton:
            self._mode = "click"
        else:
            self._mode = None
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pos = event.position()
        if self._mode == "rotate":
            angle = self._screen_angle(pos)
            delta = (angle - self._last_angle + 180.0) % 360.0 - 180.0
            self._last_angle = angle
            if delta:
                # screen y points down, so a clockwise drag is a positive screen angle: negate it
                self._owner.rotate(-delta)
        elif self._mode in ("click", "pan"):
            if self._mode == "click" and (pos - self._press).manhattanLength() >= QApplication.startDragDistance():
                self._mode = "pan"
                self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            if self._mode == "pan":
                d = pos - self._last
                self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - round(d.x()))
                self.verticalScrollBar().setValue(self.verticalScrollBar().value() - round(d.y()))
                self._owner._user_moved = True
        else:
            atom = self._owner.atom_at(self.mapToScene(pos.toPoint()))
            self.viewport().setCursor(Qt.CursorShape.PointingHandCursor if atom is not None
                                      else Qt.CursorShape.ArrowCursor)
            self.viewport().setToolTip(self._owner.describe_atom(atom) if atom is not None else "")
        self._last = pos
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._mode == "click" and event.button() == Qt.MouseButton.LeftButton:
            toggle = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
            self._owner._click(self.mapToScene(event.position().toPoint()), toggle)
        self._mode = None
        self.viewport().setCursor(Qt.CursorShape.ArrowCursor)
        event.accept()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._owner.fit()
        event.accept()

    def wheelEvent(self, event: QWheelEvent) -> None:
        steps = event.angleDelta().y() / 120.0
        if steps:
            self._owner.zoom(1.2 ** steps)
        event.accept()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        if key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self._owner.zoom(KEY_ZOOM)
        elif key in (Qt.Key.Key_Minus, Qt.Key.Key_Underscore):
            self._owner.zoom(1.0 / KEY_ZOOM)
        elif key == Qt.Key.Key_BracketLeft:
            self._owner.rotate(KEY_ROTATE_DEG)
        elif key == Qt.Key.Key_BracketRight:
            self._owner.rotate(-KEY_ROTATE_DEG)
        elif key == Qt.Key.Key_F:
            self._owner.fit()
        else:
            super().keyPressEvent(event)
            return
        event.accept()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if not self._owner._user_moved:
            self._owner.fit()
