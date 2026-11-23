"""The Model Kit dock (UI-7): build and edit a molecule in the 3D view.

    install(win)    adds the Model Kit dock (Build workspace), Edit > Undo / Redo and the controller of the 3D view's
                    Build mode (View toolbar) to a MainWindow

**Choosing what to build** goes element first, then geometry:

1. *Element*: a row of tiles for the common elements (``edit.COMMON_ELEMENTS``) and, in the section **Inorganic and
   other elements**, a compact periodic table (periods 1-6). Metals are coordination centres; the other elements
   follow VSEPR.
2. *Geometry*: the shapes that element can take at the chosen charge (``edit.geometry_options``), each drawn as a
   small schematic (bond sites as dots, lone pairs as rings, the polyhedron's edges faint) and named by its shape:
   tetrahedral, pyramidal, bent, trigonal planar, linear, terminal for a main-group atom (with its hybridization in the
   tooltip; a higher valence of S, Se, P, As is a geometry of its own); terminal, linear, trigonal, tetrahedral, square
   planar, trigonal bipyramidal and octahedral for a metal.
3. *Charge* and *Bond to it* (Auto, Single, Double, Triple): a carbonyl oxygen is O, terminal, Double; a nitrile N is
   N, terminal, Triple.

Or a *fragment*: Functional groups, Ring systems, Small substituents, PFAS pieces (``chem.fragments``; the (CF2)n and
CnF2n+1 lengths come from the n box), or any SMILES or InChI typed in the box.

**An edit** is a session on one ``chem.edit.EditableMolecule``. Entering Build mode starts one: on the molecule
selected in the list (its shown conformer when it is built), or on an empty model when nothing is selected (**New**
starts an empty one, **Edit selected** loads the selected molecule). While an edit is open the kit owns the 3D and 2D
views, as the Surface Builder does: selecting another molecule in the list does not change them, and the other mouse
modes (Rotate to select atoms for the edit tools, Move to place a second piece) act on the model. **Add to list**
hands the structure over as a new molecule (``model.to_input``: the 3D coordinates go with it, source "model kit")
and **Replace <id>** gives the molecule the edit started from the new structure; both end the edit. **Close** drops
it.

In Build mode the view draws every free valence as a stub (``MoleculeViewer3D.set_stubs``; hydrogens are stubs there)
and reports clicks (``buildClicked``), which this module turns into edits (MODEL_KIT_GUI.md): a stub attaches the
current choice there, an atom is replaced by the current atom type, a click in empty space places the first atom or
fragment of an empty model (Shift+click: a separate piece), and Ctrl+click selects.

The edit tools act on the selected atoms: Delete (atoms, or the whole piece), bond order (1 -> 2 -> 3), charge ±,
make and break a bond, the measured distance / angle / dihedral (2, 3 or 4 atoms) set to a typed value, Add H and
Minimize (MMFF94; UFF when MMFF94 has no parameters, which the dock says, as for metals; the selection frozen when
asked). A quick local relax runs inside each edit (``model.auto_relax``), so Undo takes it back too. A failed edit
(``EditError``) is reported on the message line and changes nothing.

Every edit is a ``chem.edit.Command``; ``UndoAdaptor`` is the ``model.history`` that puts them on a ``QUndoStack``
(Edit > Undo / Redo, named actions ``edit.undo`` / ``edit.redo``, default keys in ``gui.action_ids``). The handlers
(``click_stub``, ``click_atom``, ``click_space``, ``choose``, ``choose_element``, ``choose_geometry``, the tools) are
plain methods, which the tests call directly.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
from PySide6.QtCore import QPointF, QSize, Qt
from PySide6.QtGui import QAction, QColor, QIcon, QPainter, QPen, QPixmap, QUndoCommand, QUndoStack
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..chem import edit as kit
from ..chem import fragments as frag_lib
from ..chem.edit import AtomType, EditableMolecule, EditError
from ..chem.fragments import Fragment

TITLE = "Model Kit"
SOURCE = "model kit"
FIRST_VIEW_EXTENT = 10.0        # Å the view spans around the first atom of a new model
COMMON_COLUMNS = 6              # common-element tiles per row
GEOMETRY_COLUMNS = 4            # geometry buttons per row
BONDS = (("Auto", None), ("Single", 1), ("Double", 2), ("Triple", 3))
CHARGES = (-3, -2, -1, 0, 1, 2, 3)
#: Fragment menus: (title, fragments). The library's categories, under QuantaBench's own names.
FRAGMENT_MENUS = (("Functional groups", "group"), ("Ring systems", "ring"), ("Small substituents", "substituent"),
                  ("PFAS pieces", "pfas"))

#: The compact periodic table: one string per period, a symbol per column (18 columns, "." = empty). Lanthanides
#: and actinides after La are left out.
PERIODIC_ROWS = (
    "H . . . . . . . . . . . . . . . . He",
    "Li Be . . . . . . . . . . B C N O F Ne",
    "Na Mg . . . . . . . . . . Al Si P S Cl Ar",
    "K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr",
    "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe",
    "Cs Ba La Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn",
)
NOBLE_GASES = frozenset("He Ne Ar Kr Xe Rn".split())
#: Tile colours by kind of element (QuantaBench's own scheme)
_KIND_RGB = {"nonmetal": (214, 226, 240), "halogen": (210, 236, 214), "metalloid": (244, 230, 196),
             "s-metal": (246, 214, 210), "d-metal": (226, 216, 242), "p-metal": (208, 232, 232),
             "noble": (225, 225, 225)}
_HALOGENS = frozenset("F Cl Br I At".split())
_METALLOIDS = frozenset("B Si Ge As Sb Te Po".split())
_S_METALS = frozenset("Li Na K Rb Cs Be Mg Ca Sr Ba".split())
_P_METALS = frozenset("Al Ga In Tl Sn Pb Bi".split())


def element_kind(symbol: str) -> str:
    if symbol in NOBLE_GASES:
        return "noble"
    if symbol in _HALOGENS:
        return "halogen"
    if symbol in _METALLOIDS:
        return "metalloid"
    if symbol in _S_METALS:
        return "s-metal"
    if symbol in _P_METALS:
        return "p-metal"
    if symbol in kit.METALS:
        return "d-metal"
    return "nonmetal"


def element_name(symbol: str) -> str:
    try:
        from ase.data import atomic_names, atomic_numbers

        return atomic_names[atomic_numbers[symbol]].lower()
    except (ImportError, KeyError, IndexError):
        return symbol


# ====================================================================== undo
class _KitCommand(QUndoCommand):
    """One ``chem.edit.Command`` on the QUndoStack. It has already run when pushed (``UndoAdaptor.push``), so the
    first ``redo()``, which ``QUndoStack.push`` calls at once, does nothing."""

    def __init__(self, model: EditableMolecule, command: kit.Command):
        super().__init__(command.text)
        self.model = model
        self.command = command
        self._pushed = False

    def redo(self) -> None:
        if not self._pushed:
            self._pushed = True
            return
        self.command.apply(self.model)

    def undo(self) -> None:
        self.command.undo(self.model)


class UndoAdaptor:
    """``model.history`` for the GUI: runs each command, then puts it on ``stack``. A command that fails raises
    ``EditError`` before anything is pushed, so the stack never holds a failed edit."""

    def __init__(self, model: EditableMolecule, stack: QUndoStack):
        self.model = model
        self.stack = stack

    def push(self, command: kit.Command):
        result = command.apply(self.model)
        self.stack.push(_KitCommand(self.model, command))
        return result


# ====================================================================== geometry icons
#: a fixed oblique view for the schematics: turned 25° about x, then -35° about y
_VIEW = (np.array([[math.cos(-0.61), 0, math.sin(-0.61)], [0, 1, 0], [-math.sin(-0.61), 0, math.cos(-0.61)]])
         @ np.array([[1, 0, 0], [0, math.cos(0.44), -math.sin(0.44)], [0, math.sin(0.44), math.cos(0.44)]]))


def site_directions(t: AtomType) -> tuple[np.ndarray, int]:
    """(unit vectors of all of ``t``'s domains, how many of them are bonds); the rest are lone pairs."""
    if t.hyb in kit.COORDINATION:
        dirs = kit.IDEAL_SITES[t.hyb]
        return dirs, len(dirs)
    n = kit.DOMAINS.get(t.hyb, 4)
    dirs = np.array(kit.domain_directions([], n)) if n > 1 else np.array([[1.0, 0, 0]])
    bonds = max(1, min(n, t.sigma))
    if n == 4:          # upright: a lone pair on top when there is one (pyramidal, bent), else a bond
        up = dirs[-1] if bonds < n else dirs[0]
        dirs = dirs @ kit._align(up, np.array([0.0, 1.0, 0.0])).T
    return dirs, bonds


def geometry_icon(t: AtomType, color: QColor, size: int = 34) -> QIcon:
    """A schematic of ``t``'s geometry: the centre, a dot per bond site, a ring per lone pair, faint polyhedron
    edges."""
    dirs, n_bonds = site_directions(t)
    out = QIcon()
    for scale in (1, 2):
        pm = QPixmap(size * scale, size * scale)
        pm.fill(Qt.GlobalColor.transparent)
        pm.setDevicePixelRatio(scale)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        c = QPointF(size / 2, size / 2)
        r = size * 0.38
        pts = [QPointF(c.x() + r * v[0], c.y() - r * v[1]) for v in (dirs @ _VIEW.T)]
        if len(dirs) >= 3:      # the polyhedron's edges: pairs at the smallest angle between sites
            cos = dirs @ dirs.T
            smallest = min(cos[i, j] for i in range(len(dirs)) for j in range(i))
            faint = QColor(color)
            faint.setAlpha(70)
            p.setPen(QPen(faint, 1.0))
            for i in range(len(dirs)):
                for j in range(i):
                    if cos[i, j] >= smallest - 0.05 or len(dirs) == 3:
                        p.drawLine(pts[i], pts[j])
        pen = QPen(color, max(1.3, size / 24))
        p.setPen(pen)
        for k, q in enumerate(pts):
            if k < n_bonds:
                p.drawLine(c, q)
        p.setBrush(color)
        for k, q in enumerate(pts):
            if k < n_bonds:
                p.drawEllipse(q, size * 0.07, size * 0.07)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(color, 1.0, Qt.PenStyle.DashLine))
        for k, q in enumerate(pts):
            if k >= n_bonds:
                p.drawEllipse(QPointF((c.x() + q.x()) / 2 + (q.x() - c.x()) * 0.25,
                                      (c.y() + q.y()) / 2 + (q.y() - c.y()) * 0.25), size * 0.08, size * 0.08)
        p.setPen(QPen(color, 1.0))
        p.setBrush(QColor(color.red(), color.green(), color.blue(), 160))
        p.drawEllipse(c, size * 0.11, size * 0.11)
        p.end()
        out.addPixmap(pm)
    return out


def _geometry_label(t: AtomType) -> str:
    name = kit.shape_name(t)
    if t.valence is not None and t.hyb not in kit.COORDINATION:
        name += f" ({t.valence})"
    return name.replace("trigonal bipyramidal", "trig. bipyramid").replace("trigonal planar", "trig. planar")


def _geometry_tip(t: AtomType) -> str:
    q = kit.charge_text(t.charge)
    if t.hyb in kit.COORDINATION:
        return (f"{t.symbol}{q}, {t.hyb}: {t.sigma} coordination site{'s' if t.sigma != 1 else ''} (single bonds; "
                "open sites stay empty)")
    lp = kit.lone_pairs(t)
    bits = [f"{t.symbol}{q} {t.hyb}, valence {t.total_valence}", f"{t.sigma} σ bond{'s' if t.sigma != 1 else ''}"]
    if t.pi:
        bits.append(f"{t.pi} π")
    if lp:
        bits.append(f"{lp} lone pair{'s' if lp != 1 else ''}")
    return ": ".join([bits[0], ", ".join(bits[1:])])


# ====================================================================== the dock's widget and controller
class ModelKitPanel(QWidget):
    """The element and geometry pickers, the fragments, the edit tools and the controller of the 3D view's Build mode
    (module docstring)."""

    def __init__(self, win, parent=None):
        super().__init__(parent)
        self.win = win
        self.model: EditableMolecule | None = None
        self.active = False                 # an edit is open: the kit owns the 3D and 2D views
        self.source_id: str | None = None   # the listed molecule this edit started from
        self.source_name = ""
        self.element = "C"
        self.options: list[AtomType] = []   # the element's geometries at the chosen charge
        self.geometry: AtomType | None = None
        self.choice: AtomType | Fragment = kit.atom_type("C sp3")
        self.valences: list[kit.FreeValence] = []   # the stubs, in the order the view draws them
        self.message = ""
        self._shown_n = -1
        self._syncing = False
        self.stack = QUndoStack(self)
        self.stack.setUndoLimit(500)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(4)
        scroll.setWidget(body)
        outer.addWidget(scroll)

        # ---- the edit
        self.editing_label = QLabel()
        self.editing_label.setWordWrap(True)
        lay.addWidget(self.editing_label)
        row = QHBoxLayout()
        self.b_new = QPushButton("New")
        self.b_new.setToolTip("Start an empty model, in Build mode: click in the 3D view to place the first atom")
        self.b_new.clicked.connect(lambda _c=False: self.new() and self._to_build_mode())
        self.b_edit = QPushButton("Edit selected")
        self.b_edit.setToolTip("Edit the molecule selected in the list (its shown conformer when it is built)")
        self.b_edit.clicked.connect(lambda _c=False: self.edit_selected() and self._to_build_mode())
        row.addWidget(self.b_new)
        row.addWidget(self.b_edit)
        lay.addLayout(row)

        # ---- element
        lay.addWidget(_heading("1. Element"))
        grid = QGridLayout()
        grid.setSpacing(2)
        self.element_buttons: dict[str, QToolButton] = {}
        for k, sym in enumerate(kit.COMMON_ELEMENTS):
            b = self._element_tile(sym, 30)
            grid.addWidget(b, k // COMMON_COLUMNS, k % COMMON_COLUMNS)
        lay.addLayout(grid)
        from .collapsible import Section

        table = QWidget()
        tl = QGridLayout(table)
        tl.setContentsMargins(0, 2, 0, 2)
        tl.setHorizontalSpacing(1)
        tl.setVerticalSpacing(1)
        self.table_buttons: dict[str, QToolButton] = {}
        for r, period in enumerate(PERIODIC_ROWS):
            for col, sym in enumerate(period.split()):
                if sym == ".":
                    continue
                b = self._element_tile(sym, 18, in_table=True)
                tl.addWidget(b, r, col)
        legend = QLabel("Tiles: nonmetal, <span style='background:#d2ecd6'>&nbsp;halogen&nbsp;</span> "
                        "<span style='background:#f4e6c4'>&nbsp;metalloid&nbsp;</span> "
                        "<span style='background:#f6d6d2'>&nbsp;s-block&nbsp;</span> "
                        "<span style='background:#e2d8f2'>&nbsp;d-block&nbsp;</span> "
                        "<span style='background:#d0e8e8'>&nbsp;p-block metal&nbsp;</span>. "
                        "Metals are built as coordination centres (UFF for Minimize).")
        legend.setWordWrap(True)
        legend.setStyleSheet("color: gray;")
        tl.addWidget(legend, len(PERIODIC_ROWS), 0, 1, 18)
        self.table_section = Section("Inorganic and other elements", table, key="kit_elements", open_=False)
        lay.addWidget(self.table_section)
        self.element_label = QLabel()
        self.element_label.setWordWrap(True)
        lay.addWidget(self.element_label)

        # ---- geometry, charge, bond
        lay.addWidget(_heading("2. Geometry"))
        self.geometry_grid = QGridLayout()
        self.geometry_grid.setSpacing(2)
        lay.addLayout(self.geometry_grid)
        self.geometry_buttons: list[QToolButton] = []
        row = QHBoxLayout()
        row.addWidget(QLabel("Charge"))
        self.charge_box = QComboBox()
        for q in CHARGES:
            self.charge_box.addItem(f"{q:+d}" if q else "0", q)
        self.charge_box.setCurrentIndex(CHARGES.index(0))
        self.charge_box.setToolTip("Formal charge of the new atom (the geometries follow it: N+ is tetrahedral)")
        self.charge_box.currentIndexChanged.connect(lambda _i: self._on_charge())
        row.addWidget(self.charge_box)
        row.addWidget(QLabel("Bond to it"))
        self.bond_box = QComboBox()
        for text, order in BONDS:
            self.bond_box.addItem(text, order)
        self.bond_box.setToolTip("Order of the bond from the atom it attaches to. Auto: double between two atoms that "
                                 "each still have a π bond to make (C sp2 to C sp2), else single")
        self.bond_box.currentIndexChanged.connect(lambda _i: self._update_choice())
        row.addWidget(self.bond_box)
        row.addStretch(1)
        lay.addLayout(row)

        # ---- fragments
        lay.addWidget(_heading("Or a fragment"))
        grid = QGridLayout()
        grid.setSpacing(2)
        self.menus: dict[str, QMenu] = {}
        for k, (title, category) in enumerate(FRAGMENT_MENUS):
            b = self._menu_button(title, frag_lib.by_category(category))
            b.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            grid.addWidget(b, k // 2, k % 2)
        menu = self.menus["PFAS pieces"]
        menu.addSeparator()
        menu.addAction("(CF2)n chain, n from the box", lambda: self.choose(frag_lib.cf2_chain(self.n_box.value())))
        menu.addAction("CnF2n+1, n from the box", lambda: self.choose(frag_lib.perfluoroalkyl(self.n_box.value())))
        lay.addLayout(grid)
        row = QHBoxLayout()
        self.n_box = QSpinBox()
        self.n_box.setRange(1, 20)
        self.n_box.setValue(6)
        self.n_box.setPrefix("n = ")
        self.n_box.setToolTip("Length of the PFAS pieces' (CF2)n chain and CnF2n+1 group")
        row.addWidget(self.n_box)
        self.text_box = QLineEdit()
        self.text_box.setPlaceholderText("SMILES or InChI")
        self.text_box.setToolTip("A fragment from text: mark where it bonds with [*:1] (else its first atom bonds), "
                                 "or an InChI (its first atom bonds)")
        self.text_box.returnPressed.connect(lambda: self.use_text())
        self.b_text = QPushButton("Use")
        self.b_text.clicked.connect(lambda _c=False: self.use_text())
        row.addWidget(self.text_box, 1)
        row.addWidget(self.b_text)
        lay.addLayout(row)
        self.choice_label = QLabel()
        self.choice_label.setWordWrap(True)
        lay.addWidget(self.choice_label)

        # ---- edit tools
        lay.addWidget(_heading("Edit the selected atoms (Ctrl+click)"))
        grid = QGridLayout()
        grid.setSpacing(2)
        self.b_delete = QToolButton()
        self.b_delete.setText("Delete")
        self.b_delete.setToolTip("Delete the selected atoms with their hydrogens (the arrow: the whole piece)")
        self.b_delete.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self.b_delete.clicked.connect(lambda _c=False: self.delete_selected())
        dm = QMenu(self.b_delete)
        dm.addAction("Delete the selected atoms", lambda: self.delete_selected())
        dm.addAction("Delete the whole piece", lambda: self.delete_selected(whole=True))
        self.b_delete.setMenu(dm)
        tools = [self.b_delete]
        for text, tip, fn in (
                ("Bond order", "Two bonded atoms: single -> double -> triple -> single", self.cycle_bond_order),
                ("Charge +", "One atom: formal charge + 1 (an amine N becomes ammonium)", lambda: self.change_charge(1)),
                ("Charge −", "One atom: formal charge − 1 (a hydroxyl O becomes an oxide)",
                 lambda: self.change_charge(-1)),
                ("Make bond", "Two atoms: bond them (a hydrogen of each makes room)", self.make_bond),
                ("Break bond", "Two bonded atoms: break the bond", self.break_bond),
                ("Add H", "Hydrogens on every open valence (not on metals)", self.add_hydrogens)):
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.clicked.connect(lambda _c=False, fn=fn: fn())
            tools.append(b)
        for k, b in enumerate(tools):
            b.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            grid.addWidget(b, k // 3, k % 3)
        self.tool_buttons = tools
        lay.addLayout(grid)
        row = QHBoxLayout()
        self.value_label = QLabel("Value")
        self.value_box = QDoubleSpinBox()
        self.value_box.setDecimals(3)
        self.value_box.setRange(-180.0, 180.0)
        self.value_box.setKeyboardTracking(False)
        self.value_box.setToolTip("The selected distance (2 atoms), angle (3) or dihedral (4): type a value and Set")
        self.b_set = QPushButton("Set")
        self.b_set.setToolTip("Move the smaller side so the measurement takes this value")
        self.b_set.clicked.connect(lambda _c=False: self.set_value(self.value_box.value()))
        row.addWidget(self.value_label)
        row.addWidget(self.value_box, 1)
        row.addWidget(self.b_set)
        lay.addLayout(row)
        row = QHBoxLayout()
        self.b_minimize = QPushButton("Minimize")
        self.b_minimize.setToolTip("MMFF94; UFF when MMFF94 has no parameters (metals and other elements it lacks); "
                                   "open valences count as hydrogens")
        self.b_minimize.clicked.connect(lambda _c=False: self.minimize())
        self.freeze_box = QCheckBox("Freeze selected")
        self.freeze_box.setToolTip("Keep the selected atoms where they are while minimizing")
        row.addWidget(self.b_minimize)
        row.addWidget(self.freeze_box)
        lay.addLayout(row)
        self.ff_label = QLabel()
        self.ff_label.setWordWrap(True)
        self.ff_label.setStyleSheet("color: gray;")
        lay.addWidget(self.ff_label)
        self.relax_box = QCheckBox("Relax after each edit")
        self.relax_box.setChecked(True)
        self.relax_box.setToolTip("A quick relax of the atoms near each edit with the same force field (part of the "
                                  "edit: Undo takes it back too)")
        self.relax_box.toggled.connect(self._on_relax_toggled)
        lay.addWidget(self.relax_box)
        lay.addStretch(1)

        # ---- footer, always in view: problems, the last message, the hand-off
        foot = QVBoxLayout()
        foot.setContentsMargins(6, 2, 6, 4)
        foot.setSpacing(2)
        self.problems_label = QLabel()
        self.problems_label.setWordWrap(True)
        self.problems_label.setStyleSheet("color: #b06000;")
        foot.addWidget(self.problems_label)
        self.message_label = QLabel()
        self.message_label.setWordWrap(True)
        self.message_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        foot.addWidget(self.message_label)
        row = QHBoxLayout()
        self.b_add = QPushButton("Add to list")
        self.b_add.setToolTip("Add the structure to the molecule list as a new molecule (with its 3D coordinates) "
                              "and end the edit")
        self.b_add.clicked.connect(lambda _c=False: self.add_to_list())
        self.b_replace = QPushButton("Replace")
        self.b_replace.setToolTip("Give the molecule this edit started from the new structure (same ID and name) "
                                  "and end the edit")
        self.b_replace.clicked.connect(lambda _c=False: self.replace_original())
        self.b_close = QPushButton("Close")
        self.b_close.setToolTip("End the edit without adding it (asks when there are changes)")
        self.b_close.clicked.connect(lambda _c=False: self.close_edit())
        for b in (self.b_add, self.b_replace, self.b_close):
            row.addWidget(b)
        foot.addLayout(row)
        outer.addLayout(foot)

        win.viewer.buildClicked.connect(self._on_build_clicked)
        win.viewer.coordinatesChanged.connect(self._on_coordinates_changed)
        win.selection.changed.connect(self._on_selection_changed)
        self.stack.cleanChanged.connect(self._on_clean_changed)
        self.choose("C sp3")
        self._update_state()

    # ================================================================== element, geometry, fragment
    def _element_tile(self, sym: str, size: int, in_table: bool = False) -> QToolButton:
        b = QToolButton()
        b.setText(sym)
        b.setObjectName(f"element_{sym}" + ("_table" if in_table else ""))
        b.setCheckable(True)
        b.setFixedSize(QSize(size + (4 if not in_table else 0), size))
        r, g, bl = _KIND_RGB[element_kind(sym)]
        font = "8pt" if in_table else "10pt"
        b.setStyleSheet(f"QToolButton {{ background: rgb({r},{g},{bl}); color: #1b1d22; border: 1px solid "
                        f"rgb({r - 30},{g - 30},{bl - 30}); border-radius: 3px; font-size: {font}; font-weight: 600; "
                        "padding: 0px; } QToolButton:checked { border: 2px solid #c47a00; } "
                        "QToolButton:disabled { color: #9a9a9a; }")
        if sym in NOBLE_GASES:
            b.setEnabled(False)
            b.setToolTip(f"{sym} ({element_name(sym)}): not built by the kit")
        else:
            b.setToolTip(f"{sym} ({element_name(sym)}), {element_kind(sym).replace('-', ' ')}")
            b.clicked.connect(lambda _c=False, s=sym: self.choose_element(s))
        (self.table_buttons if in_table else self.element_buttons)[sym] = b
        return b

    def choose_element(self, symbol: str, prefer: AtomType | None = None) -> AtomType | None:
        """Make ``symbol`` the element to build with; its geometry buttons follow (keeping the previous shape when
        it has one, ``prefer`` when given). Returns the atom type chosen, None when the element has no geometry at
        the current charge (said on the message line)."""
        self.element = symbol
        charge = self.charge_box.currentData()
        self.options = kit.geometry_options(symbol, charge)
        self._fill_geometries()
        for d in (self.element_buttons, self.table_buttons):
            for sym, b in d.items():
                b.setChecked(sym == symbol)
        what = "coordination centre: choose its geometry" if symbol in kit.METALS else "choose its geometry"
        self.element_label.setText(f"<b>{_html(symbol)}</b> ({_html(element_name(symbol))}), {what}"
                                   + (" (UFF in Minimize)" if symbol not in kit.MMFF_ELEMENTS else ""))
        if not self.options:
            self.geometry = None
            self.say(f"{symbol}{kit.charge_text(charge)}: no geometry at this charge", error=True)
            self._update_choice()
            return None
        keep = prefer or self.geometry
        pick = 0
        if keep is not None:
            same = [k for k, t in enumerate(self.options)
                    if t.hyb == keep.hyb and t.total_valence == keep.total_valence]
            same_hyb = [k for k, t in enumerate(self.options) if t.hyb == keep.hyb]
            similar = [k for k, t in enumerate(self.options) if kit.shape_name(t) == kit.shape_name(keep)]
            pick = (same or same_hyb or similar or [0])[0]
        return self.choose_geometry(pick)

    def _fill_geometries(self) -> None:
        for b in self.geometry_buttons:
            self.geometry_grid.removeWidget(b)
            b.deleteLater()
        self.geometry_buttons = []
        color = self.palette().windowText().color()
        for k, t in enumerate(self.options):
            b = QToolButton()
            b.setText(_geometry_label(t))
            b.setObjectName(f"geometry_{k}")
            b.setIcon(geometry_icon(t, color))
            b.setIconSize(QSize(30, 30))
            b.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
            b.setCheckable(True)
            b.setAutoRaise(True)
            b.setMinimumWidth(78)
            b.setToolTip(_geometry_tip(t))
            b.clicked.connect(lambda _c=False, k=k: self.choose_geometry(k))
            self.geometry_grid.addWidget(b, k // GEOMETRY_COLUMNS, k % GEOMETRY_COLUMNS)
            self.geometry_buttons.append(b)

    def choose_geometry(self, which: int | str) -> AtomType | None:
        """Pick one of the element's geometries: its index, or a hybridization / geometry name (``"sp2"``,
        ``"octahedral"``)."""
        if isinstance(which, str):
            idx = next((k for k, t in enumerate(self.options) if t.hyb == which or kit.shape_name(t) == which), None)
        else:
            idx = which if 0 <= which < len(self.options) else None
        if idx is None:
            self.say(f"{self.element} has no geometry {which!r} here", error=True)
            return None
        self.geometry = self.options[idx]
        for k, b in enumerate(self.geometry_buttons):
            b.setChecked(k == idx)
        return self._update_choice()

    def _on_charge(self) -> None:
        if not self._syncing:
            self.choose_element(self.element)

    def current_atom_type(self) -> AtomType | None:
        """The element and geometry chosen, with the bond order of *Bond to it*."""
        if self.geometry is None:
            return None
        order = self.bond_box.currentData()
        return dataclasses.replace(self.geometry, bond_order=order) if order else self.geometry

    def _update_choice(self) -> AtomType | None:
        if self._syncing:
            return None
        t = self.current_atom_type()
        if t is None:
            return None
        self.choice = t
        self._show_choice()
        return t

    def _show_choice(self) -> None:
        c = self.choice
        if isinstance(c, AtomType):
            bond = {2: ", attached by a double bond", 3: ", attached by a triple bond"}.get(c.bond_order or 0, "")
            self.choice_label.setText(f"Next: <b>{_html(c.symbol + kit.charge_text(c.charge))}</b>, "
                                      f"{_html(_geometry_label(c))}{bond}. A stub attaches it, an atom becomes it.")
        else:
            self.choice_label.setText(f"Next: <b>{_html(c.caption)}</b> ({_html(c.name)}). A stub attaches it.")

    def _menu_button(self, name: str, entries: list[Fragment]) -> QToolButton:
        b = QToolButton()
        b.setText(name)
        b.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(b)
        for f in entries:
            a = menu.addAction(f"{f.caption}  ({f.name})" if f.label else f.name, lambda f=f: self.choose(f))
            if f.note:
                a.setToolTip(f.note)
        menu.setToolTipsVisible(True)
        b.setMenu(menu)
        self.menus[name] = menu
        return b

    def choose(self, choice: str | AtomType | Fragment) -> AtomType | Fragment | None:
        """Make ``choice`` what a click builds: an atom type (``AtomType``, or a name ``edit.atom_type`` reads:
        ``"C sp2"``, ``"=O"``, ``"Fe octahedral"``), a ``Fragment``, or a fragment name (``"COOH"``, ``"(CF2)6"``).
        An atom type sets the element, charge, geometry and bond pickers. Returns the choice, None when unknown."""
        try:
            if isinstance(choice, (AtomType, Fragment)):
                c = choice
            else:
                try:
                    c = kit.atom_type(choice)
                except EditError:
                    c = frag_lib.get(choice)
        except KeyError:
            self.say(f"No atom type or fragment named {choice!r}", error=True)
            return None
        if isinstance(c, Fragment):
            self.choice = c
            for b in self.geometry_buttons:
                b.setChecked(False)
            self._show_choice()
            return c
        self._syncing = True
        try:
            self.charge_box.setCurrentIndex(CHARGES.index(max(min(c.charge, 3), -3)))
            self.bond_box.setCurrentIndex(next(k for k, (_t, o) in enumerate(BONDS) if o == c.bond_order))
        finally:
            self._syncing = False
        self.choose_element(c.symbol, prefer=c)
        if self.geometry is None or (self.geometry.hyb, self.geometry.total_valence) != (c.hyb, c.total_valence):
            self.choice = c                  # a type the pickers do not offer: build it as given
            self._show_choice()
        return self.choice

    def use_text(self, text: str | None = None) -> Fragment | None:
        """The text box: a SMILES (``[*:1]`` marks the attachment; else its first atom bonds) or an InChI."""
        text = (self.text_box.text() if text is None else text).strip()
        if not text:
            return None
        try:
            smiles = text
            if text.startswith("InChI="):
                from rdkit import Chem, rdBase

                with rdBase.BlockLogs():
                    mol = Chem.MolFromInchi(text)
                if mol is None:
                    raise ValueError(f"Not a valid InChI: {text!r}")
                smiles = Chem.MolToSmiles(mol)
            frag = frag_lib.from_smiles(smiles, name=text if smiles != text else "")
        except ValueError as exc:
            self.say(str(exc), error=True)
            return None
        return self.choose(frag)

    # ================================================================== the edit (session)
    def start(self, model: EditableMolecule, source_id: str | None = None, name: str = "") -> None:
        """Open an edit of ``model`` (the views show it from now on)."""
        self.stack.clear()
        if self.model is not None and self._on_model_changed in self.model.listeners:
            self.model.listeners.remove(self._on_model_changed)
        model.auto_relax = self.relax_box.isChecked()
        model.history = UndoAdaptor(model, self.stack)
        model.listeners.append(self._on_model_changed)
        self.model = model
        self.source_id = source_id
        self.source_name = name
        self.active = True
        self.win.selection.clear()
        self._shown_n = -1
        self._refresh(reset_camera=True)
        self.stack.setClean()
        what = f"{source_id}" if source_id else "a new molecule"
        self.say(f"Editing {what}. In Build mode a stub attaches, an atom is replaced"
                 + (", a click in empty space places the first atom." if model.n_atoms == 0 else "."))
        self._update_state()

    def new(self) -> bool:
        if not self._may_discard("New"):
            return False
        self.start(EditableMolecule())
        return True

    def edit_selected(self) -> bool:
        """Edit the molecule selected in the list: its shown conformer when it is built, else its input."""
        row = self.win._current_row()
        if row is None:
            self.say("Select a molecule in the list first.", error=True)
            return False
        if not self._may_discard("Edit selected"):
            return False
        try:
            model = self._model_of(row)
        except (EditError, ValueError) as exc:
            QMessageBox.warning(self, "Edit selected", f"{row.id}: {exc}")
            return False
        self.start(model, row.id, row.name)
        return True

    def _model_of(self, row) -> EditableMolecule:
        from rdkit import Chem

        res = self.win.current_result()
        if res is not None and res.input.id == row.id and res.conformers:
            k = min(max(0, self.win.conf_table.currentRow()), len(res.conformers) - 1)
            mol = Chem.Mol(res.mol)
            conf = Chem.Conformer(mol.GetConformer(res.conformers[k].conf_id))
            mol.RemoveAllConformers()
            mol.AddConformer(conf, assignId=True)
            return EditableMolecule.from_mol(mol)
        return EditableMolecule.from_input(self.win._input_of(row))

    def _may_discard(self, title: str) -> bool:
        if not self.active or self.stack.isClean():
            return True
        return QMessageBox.question(self, title, "Discard the changes of the open edit?") \
            == QMessageBox.StandardButton.Yes

    def close_edit(self) -> bool:
        """End the edit without handing it over."""
        if not self.active:
            return True
        if not self._may_discard("Close"):
            return False
        self.end()
        return True

    def end(self, select: str | None = None) -> None:
        """Give the views back to the molecule list (showing ``select`` when given, else the selected molecule)."""
        if not self.active:
            return
        self.active = False
        if self.model is not None and self._on_model_changed in self.model.listeners:
            self.model.listeners.remove(self._on_model_changed)
        self.stack.clear()
        self.valences = []
        win = self.win
        win.viewer.set_stubs(None, redraw=False)
        win.viewer.clear()
        win.viewer2d.clear()
        win._shown_key = None
        if win.viewer.mode == "build":
            win.set_view_mode("rotate")
        if select is not None and win.mol_list.select_id(select):
            win.mol_list.currentIdChanged.emit(select)      # the views and panels show it (also when it was current)
        else:
            win._on_molecule_selected()
        self._update_state()

    def _to_build_mode(self) -> None:
        if self.win.viewer.mode != "build":
            self.win.set_view_mode("build")

    def enter_build(self) -> None:
        """Build mode was chosen: show the dock and open an edit when none is open."""
        dock = getattr(self.win, "model_kit_dock", None)
        if dock is not None:
            dock.show()
            dock.raise_()
        if not self.active:
            if self.win._current_row() is None or not self.edit_selected():
                self.start(EditableMolecule())

    # ================================================================== hand-off
    def _hand_off_input(self, title: str, ident: str | None, name: str):
        from rdkit.Chem import rdMolDescriptors

        m = self.model
        if m is None or m.n_atoms == 0:
            QMessageBox.information(self, title, "Nothing built yet: click in the 3D view in Build mode.")
            return None
        unpaired = [m.label(i) for i in range(m.n_atoms) if m.symbol(i) != "H"
                    and m.atom_state(i).open > 0 and m.atom_state(i).sigma_free <= 0]
        if unpaired and QMessageBox.question(
                self, title, f"{', '.join(unpaired)} still {'has' if len(unpaired) == 1 else 'have'} a double-bond "
                "valence to fill: as it is, each keeps an unpaired electron (a radical). Use it anyway?") \
                != QMessageBox.StandardButton.Yes:
            return None
        try:
            mol = m.to_mol()
            formula = rdMolDescriptors.CalcMolFormula(mol)
            item = m.to_input(id=ident or f"kit_{formula}", name=name)
        except EditError as exc:
            QMessageBox.warning(self, title, str(exc))
            return None
        item.name = name
        item.source = SOURCE
        return item

    def add_to_list(self) -> bool:
        """Add the structure to the molecule list as a new molecule and end the edit."""
        item = self._hand_off_input("Add to list", None, self.source_name if self.source_id else "")
        if item is None:
            return False
        self.win.add_inputs([item])               # may rename the id to keep ids unique
        self.win.log(f"Model Kit: added {item.id} ({item.smiles})")
        self.end(select=item.id)
        return True

    def replace_original(self) -> bool:
        """Give the molecule the edit started from the new structure (same id, name and place in the list). Its
        results leave the list (the files stay on disk); it is built again at the next Run."""
        win = self.win
        row = win.model.row(self.source_id) if self.source_id else None
        if row is None:
            QMessageBox.information(self, "Replace", "This edit did not start from a molecule in the list (or it "
                                    "has been removed): use Add to list.")
            return False
        if row.id in win.running:
            QMessageBox.information(self, "Replace", f"{row.id} is being built; replace it afterwards.")
            return False
        item = self._hand_off_input("Replace", row.id, row.name)
        if item is None:
            return False
        if (row.status != "pending" or row.folder) and QMessageBox.question(
                self, "Replace", f"Replace {row.id}'s structure with the edit? Its results are removed from the list "
                "(the files on disk stay) and it is built again at the next Run.") != QMessageBox.StandardButton.Yes:
            return False
        from .molecule_list import MoleculeRow

        fresh = MoleculeRow(id=row.id, name=row.name, smiles=item.smiles, source=item.source, input=item)
        win.model.update(row.id, **{f.name: getattr(fresh, f.name) for f in dataclasses.fields(MoleculeRow)
                                    if f.name not in ("id", "seq")})
        win.results.discard(row.id)
        win.log(f"Model Kit: replaced {row.id}'s structure with {item.smiles}")
        self.end(select=row.id)
        return True

    # ================================================================== build-mode clicks
    def _on_build_clicked(self, kind: str, index: int, point, shift: bool) -> None:
        if not self.active:
            return
        if kind == "stub":
            self.click_stub(index)
        elif kind == "atom":
            self.click_atom(index)
        else:
            self.click_space(point, shift)

    def click_stub(self, row: int):
        """A click on stub ``row`` (of ``valences``): attach the current choice there."""
        if not 0 <= row < len(self.valences):
            return None
        v = self.valences[row]
        if isinstance(self.choice, AtomType):
            return self._do(self.model.attach_atom, v, self.choice)
        return self._do(self.model.attach_fragment, v, self.choice)

    def click_atom(self, atom: int):
        """A click on an atom: replace it with the current atom type (a hydrogen takes a fragment too)."""
        m = self.model
        if not 0 <= atom < m.n_atoms:
            return None
        if isinstance(self.choice, AtomType):
            return self._do(m.replace_atom, atom, self.choice)
        if m.symbol(atom) == "H":
            return self._do(m.attach_fragment, atom, self.choice)
        self.say(f"Click a free valence (a stub) to attach {self.choice.caption}; an atom type replaces an atom.",
                 error=True)
        return None

    def click_space(self, point=(0.0, 0.0, 0.0), shift: bool = False):
        """A click in empty space: the first atom or fragment of an empty model, or with Shift a separate piece."""
        m = self.model
        if m.n_atoms and not shift:
            self.say("Click a free valence (a stub) or an atom; Shift+click in empty space adds a separate piece.")
            return None
        p = tuple(float(x) for x in np.asarray(point, float).reshape(3))
        if isinstance(self.choice, AtomType):
            return self._do(m.add_atom, self.choice, p)
        return self._do(m.add_fragment, self.choice, p)

    # ================================================================== edit tools (on the selection)
    def selected(self) -> list[int]:
        """Selected atoms of the model, in selection order (atoms the 2D view adds for open valences left out)."""
        n = self.model.n_atoms if self.model is not None else 0
        return [i for i in self.win.selection.atoms if 0 <= i < n]

    def _need(self, counts: tuple[int, ...] | None, what: str) -> list[int] | None:
        """The selected atoms when there are ``counts`` of them (None: at least one), else None and a message."""
        if not self.active:
            return None
        sel = self.selected()
        if (len(sel) not in counts) if counts else not sel:
            self.say(f"Select {what} first (Ctrl+click in Build mode, or click in Rotate mode).", error=True)
            return None
        return sel

    def delete_selected(self, whole: bool = False):
        """Delete the selected atoms with their hydrogens, or (``whole``) the piece the last selected atom is in."""
        sel = self._need(None, "the atoms to delete" if not whole else "an atom of the piece to delete")
        if sel is None:
            return None
        return self._do(self.model.delete_fragment, sel[-1]) if whole else self._do(self.model.delete_atoms, sel)

    def cycle_bond_order(self):
        sel = self._need((2,), "two bonded atoms")
        if sel is None:
            return None
        i, j = sel
        old = self.model.bond_order(i, j)
        if not old:
            self.say(f"{self.model.label(i)} and {self.model.label(j)} are not bonded: use Make bond.", error=True)
            return None
        error = ""
        for order in (old % 3 + 1, (old + 1) % 3 + 1):
            try:
                return self.model.set_bond_order(i, j, order)
            except EditError as exc:
                error = error or str(exc)
        self.say(error, error=True)
        return None

    def change_charge(self, step: int):
        sel = self._need((1,), "one atom")
        if sel is None:
            return None
        a = sel[0]
        return self._do(self.model.set_charge, a, self.model.atom_state(a).charge + int(step))

    def make_bond(self):
        sel = self._need((2,), "two atoms")
        return None if sel is None else self._do(self.model.make_bond, *sel)

    def break_bond(self):
        sel = self._need((2,), "two bonded atoms")
        return None if sel is None else self._do(self.model.break_bond, *sel)

    def add_hydrogens(self):
        return self._do(self.model.fill_hydrogens) if self.active else None

    def set_value(self, value: float):
        """Give the selected distance (2 atoms), angle (3) or dihedral (4) ``value`` (Å or degrees)."""
        sel = self._need((2, 3, 4), "2, 3 or 4 atoms")
        if sel is None:
            return None
        fn = {2: self.model.set_distance, 3: self.model.set_angle, 4: self.model.set_dihedral}[len(sel)]
        return self._do(fn, *sel, float(value))

    def minimize(self):
        if not self.active:
            return None
        frozen = self.selected() if self.freeze_box.isChecked() else []
        res = self._do(self.model.minimize, frozen)
        if res is not None:
            self.say(f"Minimized with {res.forcefield}: E = {res.energy:.2f} kcal/mol"
                     + ("" if res.converged else " (not converged: Minimize again)")
                     + (f", {len(frozen)} atom(s) frozen" if frozen else "")
                     + (f". {res.note}" if res.note else ""))
        return res

    def _do(self, fn, *args):
        """Run one edit; an ``EditError`` goes to the message line and changes nothing."""
        try:
            result = fn(*args)
        except EditError as exc:
            self.say(str(exc), error=True)
            return None
        self.say("")
        return result

    # ================================================================== views
    def _on_model_changed(self, model: EditableMolecule) -> None:
        if self.active and model is self.model:
            self._refresh()

    def _refresh(self, reset_camera: bool = False) -> None:
        """Show the model in both views, with its free valences as the 3D view's stubs."""
        m, win = self.model, self.win
        self.valences = m.all_free_valences()
        win.viewer.set_stubs([v.atom for v in self.valences], [v.direction for v in self.valences],
                             [v.max_order for v in self.valences], redraw=False)
        first = self._shown_n == 0 and m.n_atoms > 0
        if m.n_atoms != self._shown_n:
            win.selection.clear()                 # indices of another structure
        self._shown_n = m.n_atoms
        if m.n_atoms == 0:
            win.viewer.clear()
            win.viewer2d.clear()
        else:
            win.viewer.show_rdkit(m.mol, reset_camera=reset_camera)
            if first and not reset_camera:        # the first atom of a new model: room to build around it
                win.viewer.reset_camera(min_extent=FIRST_VIEW_EXTENT)
            try:
                win.viewer2d.show_rdkit(m.to_mol(), reset_camera=reset_camera)
            except Exception:  # noqa: BLE001 - a half-built structure RDKit cannot depict: the 3D view still shows it
                win.viewer2d.clear()
        problems = m.problems()
        self.problems_label.setText("" if not problems else "To fix before a calculation: " + "; ".join(problems[:4])
                                    + (f" (and {len(problems) - 4} more)" if len(problems) > 4 else ""))
        ff, why = m.forcefield()
        self.ff_label.setText("" if ff == "none" and m.n_atoms < 2 else
                              f"Force field: {ff}" + (f" ({why})" if why else ""))
        self._update_value()
        self._update_state()

    def _on_coordinates_changed(self, positions) -> None:
        """A Move-mode drag on the model: one undoable step."""
        if not self.active or self.model is None or len(positions) != self.model.n_atoms:
            return
        if not np.allclose(positions, self.model.positions):
            self._do(self.model.set_positions, positions)

    def _on_selection_changed(self, _atoms=None) -> None:
        if self.active:
            self._update_value()

    def _update_value(self) -> None:
        sel = self.selected() if self.active else []
        n = len(sel)
        m = self.model
        self.value_box.blockSignals(True)
        if n == 2:
            self.value_label.setText("Distance")
            self.value_box.setRange(0.1, 100.0)
            self.value_box.setSuffix(" Å")
            self.value_box.setValue(m.distance(*sel))
        elif n == 3:
            self.value_label.setText("Angle")
            self.value_box.setRange(0.1, 179.9)
            self.value_box.setSuffix(" °")
            self.value_box.setValue(m.angle(*sel))
        elif n == 4:
            self.value_label.setText("Dihedral")
            self.value_box.setRange(-180.0, 180.0)
            self.value_box.setSuffix(" °")
            self.value_box.setValue(m.dihedral(*sel))
        else:
            self.value_label.setText("Value")
            self.value_box.setSuffix("")
        self.value_box.blockSignals(False)
        self.value_box.setEnabled(n in (2, 3, 4))
        self.b_set.setEnabled(n in (2, 3, 4))

    def _on_relax_toggled(self, on: bool) -> None:
        if self.model is not None:
            self.model.auto_relax = bool(on)

    def _on_clean_changed(self, _clean: bool) -> None:
        import shiboken6

        if shiboken6.isValid(self.b_add):        # the stack clears itself while the window is being destroyed
            self._update_state()

    def _update_state(self) -> None:
        on = self.active
        for w in (*self.tool_buttons, self.b_minimize, self.b_add, self.b_close):
            w.setEnabled(on)
        self.b_replace.setEnabled(on and self.source_id is not None)
        self.b_replace.setText(f"Replace {self.source_id}" if on and self.source_id else "Replace")
        if not on:
            self.editing_label.setText("No edit open. Choose <b>Build</b> mode or <b>New</b> to start; Build mode "
                                       "edits the selected molecule.")
            self.problems_label.setText("")
            self.ff_label.setText("")
            return
        what = f"<b>{_html(self.source_id)}</b>" if self.source_id else "<b>a new molecule</b>"
        changed = "" if self.stack.isClean() else " (changed)"
        self.editing_label.setText(f"Editing {what}{changed}: {self.model.n_atoms} atoms. "
                                   "Add to list or Replace hands it over.")

    def say(self, text: str, error: bool = False) -> None:
        self.message = text
        self.message_label.setStyleSheet("color: #c0392b;" if error else "color: gray;")
        self.message_label.setText(_html(text))
        if error and text:
            self.win.statusBar().showMessage(text, 8000)


# ====================================================================== helpers
def _heading(text: str) -> QLabel:
    label = QLabel(f"<b>{_html(text)}</b>")
    label.setContentsMargins(0, 4, 0, 0)
    return label


def _html(text: str) -> str:
    import html

    return html.escape(str(text))


def _menu(win, title: str):
    for action in win.menuBar().actions():
        if action.menu() is not None and action.text().replace("&", "") == title:
            return action.menu()
    return None


def install(win) -> None:
    from .action_ids import bind

    panel = ModelKitPanel(win)
    dock = QDockWidget(TITLE, win)
    dock.setObjectName("modelkit")
    dock.setWidget(panel)
    win.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
    dock.hide()
    win.model_kit = panel
    win.model_kit_dock = dock
    win.viewModeChanged.connect(lambda mode: panel.enter_build() if mode == "build" else None)
    toggle = dock.toggleViewAction()
    toggle.setText(TITLE)
    toggle.setToolTip("Build and edit molecules in the 3D view (Build mode)")
    view = _menu(win, "View")
    if view is not None:
        view.addAction(toggle)
    # Edit > Undo / Redo: the model kit's edits (a text field keeps its own undo while it has the focus)
    from .clipboard import _edit_menu

    menu = _edit_menu(win)
    undo = bind(panel.stack.createUndoAction(win, "Undo"), "edit.undo")
    undo.setToolTip("Undo the last Model Kit edit")
    redo = bind(panel.stack.createRedoAction(win, "Redo"), "edit.redo")
    redo.setToolTip("Redo the Model Kit edit")
    first = menu.actions()[0] if menu.actions() else None
    menu.insertAction(first, undo)
    menu.insertAction(first, redo)
    if first is not None:
        menu.insertSeparator(first)
    win.a_undo, win.a_redo = undo, redo
    build = _menu(win, "Build")
    if build is not None:
        a = bind(QAction("Model kit (Build mode)", win), "build.model_kit")
        a.setToolTip("Build or edit in the 3D view with the Model Kit")
        a.triggered.connect(lambda _checked=False: win.set_view_mode("build"))
        build.addAction(a)
        win.a_model_kit = a
