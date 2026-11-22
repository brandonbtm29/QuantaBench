"""The Model Kit dock (UI-7): Spartan-style building in the 3D view.

    install(win)    adds the Model Kit dock (Build workspace), Edit > Undo / Redo (Ctrl+Z, Ctrl+Y) and the controller
                    of the 3D view's Build mode (View toolbar, Ctrl+4) to a MainWindow

An **edit** is a session on one ``chem.edit.EditableMolecule``. Entering Build mode starts one: on the molecule
selected in the list (its shown conformer when it is built, else its input structure), or on an empty model when
nothing is selected (**New** starts an empty one, **Edit selected** loads the selected molecule). While an edit is
open the kit owns the 3D and 2D views, as the Surface Builder does: selecting another molecule in the list does not
change them, and the other mouse modes (Rotate to select atoms for the edit tools, Move to place a second piece) act
on the model. **Add to list** hands the structure over as a new molecule (``model.to_input``: the 3D coordinates go
with it, source "model kit") and **Replace <id>** gives the molecule the edit started from the new structure; both
end the edit, as Spartan's Done does. **Close** drops it.

In Build mode the view draws every free valence as a stub (``MoleculeViewer3D.set_stubs``; hydrogens are stubs
there) and reports clicks (``buildClicked``), which this module turns into edits (MODEL_KIT_GUI.md):

====================  ======================  =====================================================
Click                 Palette choice          Edit
====================  ======================  =====================================================
stub                  atom type               ``attach_atom`` at that free valence
stub                  group / ring / PFAS     ``attach_fragment``
atom                  atom type               ``replace_atom``
empty space           anything                ``add_atom`` / ``add_fragment`` there, when the model is
                                              empty or with Shift (a separate piece)
Ctrl+click            -                       select (for the edit tools), as in Rotate mode
====================  ======================  =====================================================

The edit tools act on the selected atoms: Delete (atoms, or the whole piece), bond order (1 → 2 → 3), charge ±,
make and break a bond, the measured distance / angle / dihedral (2, 3 or 4 atoms) set to a typed value, Add H and
Minimize (MMFF94, UFF when MMFF has no parameters, the selection frozen when asked). A quick local relax runs inside
each edit (``model.auto_relax``), so Undo takes it back too. A failed edit (``EditError``) is reported on the
message line and changes nothing.

Every edit is a ``chem.edit.Command``; ``UndoAdaptor`` is the ``model.history`` that puts them on a ``QUndoStack``
(Edit > Undo / Redo). The handlers (``click_stub``, ``click_atom``, ``click_space``, ``choose``, the tools) are
plain methods, which the tests call directly.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QAction,
    QColor,
    QFont,
    QIcon,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
    QUndoCommand,
    QUndoStack,
)
from PySide6.QtWidgets import (
    QCheckBox,
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
#: the "More" menu: charged and hypervalent atom types after the Organic tab
MORE_TYPES = tuple(k for k in kit.PALETTE if k not in kit.ORGANIC)
COLUMNS = 5                     # atom buttons per row
FIRST_VIEW_EXTENT = 10.0        # Å the view spans around the first atom of a new model


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


# ====================================================================== icons
def _element_rgb(symbol: str) -> QColor:
    from ase.data import atomic_numbers
    from ase.data.colors import jmol_colors

    r, g, b = (jmol_colors[atomic_numbers[symbol]] * 255).round().astype(int)
    return QColor(int(r), int(g), int(b))


def _stub_angles(t: AtomType) -> list[tuple[float, float]]:
    """(angle in degrees, length factor) of the stubs an atom type's icon shows: its sigma bonds in the 2D picture
    of its geometry (a tetrahedron's fourth bond foreshortened)."""
    domains = {4: [(90, 1.0), (205, 1.0), (335, 1.0), (270, 0.55)], 3: [(90, 1.0), (210, 1.0), (330, 1.0)],
               2: [(180, 1.0), (0, 1.0)], 1: [(180, 1.0)]}[kit.DOMAINS.get(t.hyb, 4) if t.symbol != "H" else 1]
    if t.bond_order:                      # =O, =S, ≡N: one multiple bond to the left
        return [(180, 1.0)]
    n = max(1, t.sigma)
    if n == 2 and len(domains) == 4:      # bent (O sp3): the two lower bonds
        return domains[1:3]
    return domains[:n]


def atom_icon(t: AtomType, size: int = 40) -> QIcon:
    """The atom with its free valences in their geometry, as Spartan's Organic tab draws them."""
    out = QIcon()
    for scale in (1, 2):
        pm = QPixmap(size * scale, size * scale)
        pm.fill(Qt.GlobalColor.transparent)
        pm.setDevicePixelRatio(scale)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        c = QPointF(size / 2, size / 2)
        r_atom = size * 0.25
        r_stub = size * 0.48
        order = t.bond_order or 1
        pen = QPen(QColor(110, 115, 125), max(1.4, size / 22))
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        for ang, f in _stub_angles(t):
            u = QPointF(math.cos(math.radians(ang)), -math.sin(math.radians(ang)))
            n = QPointF(-u.y(), u.x())
            for k in range(order):
                off = (k - (order - 1) / 2) * size * 0.09
                p.drawLine(c + n * off, c + u * (r_stub * f) + n * off)
        color = _element_rgb(t.symbol)
        p.setPen(QPen(color.darker(150), 1.0))
        p.setBrush(color)
        p.drawEllipse(c, r_atom, r_atom)
        font = QFont()
        font.setPixelSize(int(size * 0.27))
        font.setBold(True)
        p.setFont(font)
        p.setPen(QColor(20, 20, 20) if color.lightness() > 110 else QColor(245, 245, 245))
        label = t.symbol + ("+" if t.charge > 0 else "-" if t.charge < 0 else "")
        p.drawText(QRectF(c.x() - r_atom * 1.6, c.y() - r_atom, r_atom * 3.2, r_atom * 2),
                   Qt.AlignmentFlag.AlignCenter, label)
        p.end()
        out.addPixmap(pm)
    return out


# ====================================================================== the dock's widget and controller
class ModelKitPanel(QWidget):
    """The palette, the edit tools and the controller of the 3D view's Build mode (module docstring)."""

    def __init__(self, win, parent=None):
        super().__init__(parent)
        self.win = win
        self.model: EditableMolecule | None = None
        self.active = False                 # an edit is open: the kit owns the 3D and 2D views
        self.source_id: str | None = None   # the listed molecule this edit started from
        self.source_name = ""
        self.choice: AtomType | Fragment = kit.atom_type("C sp3")
        self.valences: list[kit.FreeValence] = []   # the stubs, in the order the view draws them
        self.message = ""
        self._shown_n = -1
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
        self.b_new.setToolTip("Start an empty model: click in the 3D view (Build mode) to place the first atom")
        self.b_new.clicked.connect(lambda _c=False: self.new() and self._to_build_mode())
        self.b_edit = QPushButton("Edit selected")
        self.b_edit.setToolTip("Edit the molecule selected in the list (its shown conformer when it is built)")
        self.b_edit.clicked.connect(lambda _c=False: self.edit_selected() and self._to_build_mode())
        row.addWidget(self.b_new)
        row.addWidget(self.b_edit)
        lay.addLayout(row)

        # ---- palette: atoms
        lay.addWidget(_heading("Atoms"))
        grid = QGridLayout()
        grid.setSpacing(2)
        self.atom_buttons: dict[str, QToolButton] = {}   # checked: the current choice (none for a group)
        for k, key in enumerate(kit.ORGANIC):
            t = kit.PALETTE[key]
            b = QToolButton()
            b.setText(key)
            b.setIcon(atom_icon(t))
            b.setIconSize(_size(28))
            b.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
            b.setCheckable(True)
            b.setAutoRaise(True)
            b.setMinimumWidth(54)
            b.setToolTip(_type_tip(t))
            b.clicked.connect(lambda _c=False, key=key: self.choose(key))
            self.atom_buttons[key] = b
            grid.addWidget(b, k // COLUMNS, k % COLUMNS)
        lay.addLayout(grid)

        # ---- palette: groups, rings, PFAS, more, text
        lay.addWidget(_heading("Groups and rings"))
        row = QHBoxLayout()
        self.menus: dict[str, QMenu] = {}
        for name, entries in (("Groups", frag_lib.by_category("group") + [frag_lib.get("CF3")]),
                              ("Rings", frag_lib.by_category("ring")),
                              ("Substituents", frag_lib.by_category("substituent"))):
            row.addWidget(self._menu_button(name, entries))
        lay.addLayout(row)
        row = QHBoxLayout()
        pfas = self._menu_button("PFAS", frag_lib.by_category("pfas"))
        menu = self.menus["PFAS"]
        menu.addSeparator()
        menu.addAction("(CF2)n chain (n from the box)", lambda: self.choose(frag_lib.cf2_chain(self.n_box.value())))
        menu.addAction("CnF2n+1 (n from the box)", lambda: self.choose(frag_lib.perfluoroalkyl(self.n_box.value())))
        row.addWidget(pfas)
        self.n_box = QSpinBox()
        self.n_box.setRange(1, 20)
        self.n_box.setValue(6)
        self.n_box.setPrefix("n = ")
        self.n_box.setToolTip("Length of the PFAS menu's (CF2)n chain and CnF2n+1 group")
        row.addWidget(self.n_box)
        more = QToolButton()
        more.setText("More atoms")
        more.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        more_menu = QMenu(more)
        for key in MORE_TYPES:
            a = more_menu.addAction(atom_icon(kit.PALETTE[key], 24), key, lambda key=key: self.choose(key))
            a.setToolTip(_type_tip(kit.PALETTE[key]))
        more.setMenu(more_menu)
        self.menus["More"] = more_menu
        row.addWidget(more)
        lay.addLayout(row)
        row = QHBoxLayout()
        self.text_box = QLineEdit()
        self.text_box.setPlaceholderText("SMILES or InChI of a group")
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
                ("Bond order", "Two bonded atoms: single → double → triple → single", self.cycle_bond_order),
                ("Charge +", "One atom: formal charge + 1 (an amine N becomes ammonium)", lambda: self.change_charge(1)),
                ("Charge −", "One atom: formal charge − 1 (a hydroxyl O becomes an oxide)",
                 lambda: self.change_charge(-1)),
                ("Make bond", "Two atoms: bond them (a hydrogen of each makes room)", self.make_bond),
                ("Break bond", "Two bonded atoms: break the bond", self.break_bond),
                ("Add H", "Hydrogens on every open valence", self.add_hydrogens)):
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
        self.b_minimize.setToolTip("MMFF94 (UFF when MMFF has no parameters); open valences count as hydrogens")
        self.b_minimize.clicked.connect(lambda _c=False: self.minimize())
        self.freeze_box = QCheckBox("Freeze selected")
        self.freeze_box.setToolTip("Keep the selected atoms where they are while minimizing")
        row.addWidget(self.b_minimize)
        row.addWidget(self.freeze_box)
        lay.addLayout(row)
        self.relax_box = QCheckBox("Relax after each edit")
        self.relax_box.setChecked(True)
        self.relax_box.setToolTip("A quick MMFF relax of the atoms near each edit (part of the edit: Undo takes it "
                                  "back too)")
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
        self.choose(self.choice)
        self._update_state()

    # ================================================================== palette
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
        """Make ``choice`` what a click builds: an atom type (palette key such as ``"C sp2"``, ``"=O"``), a
        ``Fragment``, or a fragment name from ``chem.fragments`` (``"COOH"``, ``"(CF2)6"``). Returns it, None when
        it is unknown (said on the message line)."""
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
        self.choice = c
        key = c.key if isinstance(c, AtomType) else None
        for k, b in self.atom_buttons.items():
            b.setChecked(k == key)
        if isinstance(c, AtomType):
            self.choice_label.setText(f"Building with <b>{c.key}</b>: click a free valence to attach it, an atom "
                                      "to replace it.")
        else:
            self.choice_label.setText(f"Building with <b>{_html(c.caption)}</b> ({_html(c.name)}): click a free "
                                      "valence to attach it.")
        return c

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
        self.say(f"Editing {what}. Build mode (Ctrl+4): click a stub to attach, an atom to replace"
                 + (", in empty space to place the first atom." if model.n_atoms == 0 else "."))
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
            QMessageBox.information(self, title, "Nothing built yet: click in the 3D view in Build mode (Ctrl+4).")
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
        """A click on stub ``row`` (of ``valences``): attach the palette's choice there."""
        if not 0 <= row < len(self.valences):
            return None
        v = self.valences[row]
        if isinstance(self.choice, AtomType):
            return self._do(self.model.attach_atom, v, self.choice)
        return self._do(self.model.attach_fragment, v, self.choice)

    def click_atom(self, atom: int):
        """A click on an atom: replace it with the palette's atom type (a hydrogen takes a group too)."""
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
        """A click in empty space: the first atom or group of an empty model, or with Shift a separate piece."""
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
                     + (f", {len(frozen)} atom(s) frozen" if frozen else ""))
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
            self.editing_label.setText("No edit open. Choose <b>Build</b> mode (Ctrl+4) or <b>New</b> to start; "
                                       "Build mode edits the selected molecule.")
            self.problems_label.setText("")
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


def _size(n: int):
    from PySide6.QtCore import QSize

    return QSize(n, n)


def _html(text: str) -> str:
    import html

    return html.escape(str(text))


def _type_tip(t: AtomType) -> str:
    bits = [f"{t.symbol} {t.hyb}"]
    if t.charge:
        bits.append(f"charge {t.charge:+d}")
    if t.bond_order:
        bits.append({2: "attaches with a double bond", 3: "attaches with a triple bond"}[t.bond_order])
    bits.append(f"{t.sigma} free valence{'s' if t.sigma != 1 else ''}")
    return ", ".join(bits)


def _menu(win, title: str):
    for action in win.menuBar().actions():
        if action.menu() is not None and action.text().replace("&", "") == title:
            return action.menu()
    return None


def install(win) -> None:
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
    toggle.setToolTip("Spartan-style model kit: build in the 3D view (Build mode, Ctrl+4)")
    view = _menu(win, "View")
    if view is not None:
        view.addAction(toggle)
    # Edit > Undo / Redo: the model kit's edits (a text field keeps its own Ctrl+Z while it has the focus)
    from .clipboard import _edit_menu

    menu = _edit_menu(win)
    undo = panel.stack.createUndoAction(win, "Undo")
    undo.setShortcuts([QKeySequence("Ctrl+Z")])
    undo.setToolTip("Undo the last Model Kit edit (Ctrl+Z)")
    redo = panel.stack.createRedoAction(win, "Redo")
    redo.setShortcuts([QKeySequence("Ctrl+Y"), QKeySequence("Ctrl+Shift+Z")])
    redo.setToolTip("Redo the Model Kit edit (Ctrl+Y)")
    first = menu.actions()[0] if menu.actions() else None
    menu.insertAction(first, undo)
    menu.insertAction(first, redo)
    if first is not None:
        menu.insertSeparator(first)
    win.a_undo, win.a_redo = undo, redo
    build = _menu(win, "Build")
    if build is not None:
        a = QAction("Model kit (Build mode)", win)
        a.setToolTip("Build or edit in the 3D view with the Model Kit")
        a.triggered.connect(lambda _checked=False: win.set_view_mode("build"))
        build.addAction(a)
        win.a_model_kit = a
