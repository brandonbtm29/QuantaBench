"""Named actions and their keys (UI-9): the one place the window's keyboard shortcuts are defined.

Every user command with a shortcut is a ``QAction`` whose ``objectName`` is its name in :data:`SPECS` ("file.open",
"view.mode_rotate", "view3d.along_x", ...). A spec gives the action's title, its category on Settings > Controls, its
default keys and its *scope*, where the keys work:

==========  =====================================================================================================
window      anywhere in the main window (``Qt.WindowShortcut``): menus, toolbars, every dock
views       the structure views: the 2D and the 3D view (the "views" splitter), ``Qt.WidgetWithChildrenShortcut``
view3d      the 3D view only
view2d      the 2D view only
molecules   the Molecules dock only
table       Analyze's table only
guide       the Methods guide window (a window of its own)
==========  =====================================================================================================

Two keys conflict when they are the same and their scopes overlap (one scope contains the other): Qt would call such
a key ambiguous and run neither action. Plain keys (letters, arrows, Space, Esc) are only used in the view scopes by
default, so typing in a text field or a list is never taken over; window-wide defaults carry Ctrl (Command on macOS),
Alt or are function keys. docs/CONTROLS.md explains how the defaults were chosen.

Default keys are written as a ``QKeySequence.StandardKey`` where the platform has a standard (Open, Save, Undo,
Paste, Find, Select all...: Command on macOS by itself) or as Qt's portable text ("Ctrl+Shift+O": Ctrl is Command on
macOS). :data:`PRESETS` hold other keymaps as differences from the defaults (QuantaBench's own and ones modelled on
Avogadro 2, VMD and Spartan, from their public documentation). The user's changes are overrides on top of the chosen
preset (:class:`KeymapState`).

The window's :class:`Keymap` binds the actions (``bind``), applies the active keys, keeps the state in the window
settings (per computer, next to the window layout: Windows and macOS have different modifier keys) and exports or
imports it as JSON. Widgets that own view-local actions (the 3D and 2D views, the table) make them with
:func:`local_action` (default keys, no window needed) and call :func:`dispatch_key` from their key handling, so a key
still works when Qt's shortcut map does not see it (an inactive window, a key the VTK widget swallows).
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import shiboken6
from PySide6.QtCore import QEvent, QKeyCombination, QObject, QSettings, Qt, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QWidget

SK = QKeySequence.StandardKey
MAC = sys.platform == "darwin"

#: scope -> the scope that contains it (None: a window of its own)
SCOPES: dict[str, str | None] = {
    "window": None, "views": "window", "view3d": "views", "view2d": "views", "molecules": "window",
    "table": "window", "guide": None,
}
SCOPE_LABELS = {"window": "anywhere in the window", "views": "in the 2D and 3D views", "view3d": "in the 3D view",
                "view2d": "in the 2D view", "molecules": "in the Molecules dock", "table": "in Analyze's table",
                "guide": "in the Methods guide"}


@dataclass(frozen=True)
class ActionSpec:
    name: str
    title: str
    category: str
    default: tuple = ()          # StandardKey or portable strings, in order (the first is shown in menus)
    scope: str = "window"
    note: str = ""               # where it comes from when it is not in this build yet (another PR)


def _ws(n: int) -> str:
    """Workspace keys: Alt+1 ... on Windows and Linux (Ctrl+digits are the mouse modes); Command+Option+1 ... on
    macOS, where Option+digit types characters and Apple's guidelines keep Control for the system."""
    return f"Ctrl+Alt+{n}" if MAC else f"Alt+{n}"


COPY_KINDS = (("smiles", "SMILES"), ("inchi", "InChI"), ("inchikey", "InChIKey"), ("mol", "MOL"), ("xyz", "XYZ"),
              ("cdxml", "CDXML"))

SPECS: tuple[ActionSpec, ...] = (
    # ---------------------------------------------------------------- File
    ActionSpec("file.new_project", "New project", "File", (SK.New,)),
    ActionSpec("file.open", "Open...", "File", (SK.Open,)),
    ActionSpec("file.open_project", "Open project...", "File", ("Ctrl+Shift+O",)),
    ActionSpec("file.open_run_folder", "Open run folder...", "File", ("Ctrl+Shift+R",)),
    ActionSpec("file.import_output", "Import calculation output...", "File"),
    ActionSpec("file.save_project", "Save project", "File", (SK.Save,)),
    ActionSpec("file.save_project_as", "Save project as...", "File", ("Ctrl+Shift+S",)),
    ActionSpec("file.close_project", "Close project", "File", ("Ctrl+W",)),
    ActionSpec("file.export", "Export...", "File", ("Ctrl+E",)),
    ActionSpec("file.open_output_folder", "Open output folder", "File"),
    ActionSpec("file.quit", "Quit", "File", ("Ctrl+Q",)),
    # ---------------------------------------------------------------- Edit
    ActionSpec("edit.undo", "Undo", "Edit", (SK.Undo,), note="Model Kit (UI-7)"),
    ActionSpec("edit.redo", "Redo", "Edit", (SK.Redo, "Ctrl+Shift+Z"), note="Model Kit (UI-7)"),
    ActionSpec("edit.paste", "Paste", "Edit", (SK.Paste,)),
    ActionSpec("edit.copy_molecules", "Copy molecules", "Edit", ("Ctrl+Shift+C",), scope="molecules",
               note="UI-4"),
    *(ActionSpec(f"edit.copy_as.{kind}", f"Copy as {label}", "Edit", note="" if kind in ("smiles", "mol", "cdxml")
                 else "UI-4") for kind, label in COPY_KINDS),
    ActionSpec("edit.find_molecule", "Find a molecule", "Edit", (SK.Find,)),
    # ---------------------------------------------------------------- Selection (the shown molecule's atoms)
    ActionSpec("selection.select_all", "Select all atoms", "Selection", (SK.SelectAll,), scope="views"),
    ActionSpec("selection.clear", "Clear the selection", "Selection", ("Esc", SK.Deselect), scope="views"),
    # ---------------------------------------------------------------- Build
    ActionSpec("build.new_job", "New job...", "Build", ("Ctrl+J",)),
    ActionSpec("build.review", "Review molecules...", "Build", ("F6",)),
    ActionSpec("build.run", "Run", "Build", ("F5",)),
    # Not Esc: the views use Esc to clear the selection. Command+period is macOS's Cancel.
    ActionSpec("build.cancel", "Cancel", "Build", ("Shift+Esc", *(("Ctrl+.",) if MAC else ()))),
    ActionSpec("build.resume", "Resume run folder", "Build"),
    ActionSpec("build.sketch", "Sketch a molecule...", "Build"),
    ActionSpec("build.surface_builder", "Surface builder...", "Build"),
    ActionSpec("build.model_kit", "Model kit (Build mode)", "Build", note="Model Kit (UI-7)"),
    ActionSpec("kit.minimize", "Minimize the Model Kit's structure", "Build", note="Model Kit (UI-7)"),
    # ---------------------------------------------------------------- Mouse modes of the 3D view
    ActionSpec("view.mode_rotate", "Rotate mode", "Mouse mode", ("Ctrl+1",)),
    ActionSpec("view.mode_select", "Select mode", "Mouse mode", ("Ctrl+2",)),
    ActionSpec("view.mode_move", "Move mode", "Mouse mode", ("Ctrl+3",)),
    ActionSpec("view.mode_build", "Build mode (Model Kit)", "Mouse mode", ("Ctrl+4",), note="Model Kit (UI-7)"),
    # ---------------------------------------------------------------- View
    ActionSpec("view.reset", "Reset view", "View", ("R",), scope="views"),
    ActionSpec("view.show_2d", "Show 2D", "View", ("Ctrl+D",)),
    ActionSpec("view.hydrogens_2d", "Hydrogens in 2D", "View"),
    ActionSpec("view.screenshot", "Screenshot...", "View"),
    ActionSpec("view.open_ovito", "Open in OVITO viewport", "View"),
    ActionSpec("view.reset_layout", "Reset layout", "View"),
    ActionSpec("view.reset_all_layouts", "Reset all workspace layouts", "View"),
    ActionSpec("view.dock_all", "Dock all panels", "View", ("Ctrl+Shift+D",)),
    # ---------------------------------------------------------------- Workspaces
    ActionSpec("workspace.build", "Build workspace", "Workspace", (_ws(1),)),
    ActionSpec("workspace.calculate", "Calculate workspace", "Workspace", (_ws(2),)),
    ActionSpec("workspace.simulate", "Simulate workspace", "Workspace", (_ws(3),)),
    ActionSpec("workspace.analyze", "Analyze workspace", "Workspace", (_ws(4),)),
    ActionSpec("workspace.library", "Library workspace", "Workspace", (_ws(5),)),
    # ---------------------------------------------------------------- Playback bar and the molecule list
    ActionSpec("playback.previous", "Previous conformer (or molecule, in the bar's Molecules mode)", "Playback",
               ("Left",), scope="views"),
    ActionSpec("playback.next", "Next conformer (or molecule)", "Playback", ("Right",), scope="views"),
    ActionSpec("playback.first", "First conformer", "Playback", ("Home",), scope="views"),
    ActionSpec("playback.last", "Last conformer", "Playback", ("End",), scope="views"),
    ActionSpec("playback.play", "Play / pause", "Playback", ("Space",), scope="views"),
    ActionSpec("molecules.previous", "Previous molecule in the list", "Playback", ("Up",), scope="views"),
    ActionSpec("molecules.next", "Next molecule in the list", "Playback", ("Down",), scope="views"),
    ActionSpec("molecules.first", "First molecule in the list", "Playback", ("Ctrl+Home",), scope="views"),
    ActionSpec("molecules.last", "Last molecule in the list", "Playback", ("Ctrl+End",), scope="views"),
    # ---------------------------------------------------------------- 3D view
    ActionSpec("view3d.along_x", "Look down the x axis", "3D view", ("X",), scope="view3d"),
    ActionSpec("view3d.along_y", "Look down the y axis", "3D view", ("Y",), scope="view3d"),
    ActionSpec("view3d.along_z", "Look down the z axis", "3D view", ("Z",), scope="view3d"),
    ActionSpec("view3d.along_minus_x", "Look down the x axis from the negative side", "3D view", ("Shift+X",),
               scope="view3d"),
    ActionSpec("view3d.along_minus_y", "Look down the y axis from the negative side", "3D view", ("Shift+Y",),
               scope="view3d"),
    ActionSpec("view3d.along_minus_z", "Look down the z axis from the negative side", "3D view", ("Shift+Z",),
               scope="view3d"),
    ActionSpec("view3d.projection", "Perspective / orthographic projection", "3D view", ("P",), scope="view3d"),
    ActionSpec("view3d.zoom_in", "Zoom in", "3D view", ("+", "="), scope="view3d"),
    ActionSpec("view3d.zoom_out", "Zoom out", "3D view", ("-", "_"), scope="view3d"),
    ActionSpec("view3d.rotate_left", "Turn left", "3D view", ("Ctrl+Left",), scope="view3d"),
    ActionSpec("view3d.rotate_right", "Turn right", "3D view", ("Ctrl+Right",), scope="view3d"),
    ActionSpec("view3d.rotate_up", "Turn up", "3D view", ("Ctrl+Up",), scope="view3d"),
    ActionSpec("view3d.rotate_down", "Turn down", "3D view", ("Ctrl+Down",), scope="view3d"),
    ActionSpec("view3d.roll_ccw", "Roll counter-clockwise", "3D view", scope="view3d"),
    ActionSpec("view3d.roll_cw", "Roll clockwise", "3D view", scope="view3d"),
    # ---------------------------------------------------------------- 2D view
    ActionSpec("view2d.zoom_in", "Zoom in", "2D view", ("+", "="), scope="view2d"),
    ActionSpec("view2d.zoom_out", "Zoom out", "2D view", ("-", "_"), scope="view2d"),
    ActionSpec("view2d.rotate_ccw", "Rotate counter-clockwise", "2D view", ("Ctrl+Left", "["), scope="view2d"),
    ActionSpec("view2d.rotate_cw", "Rotate clockwise", "2D view", ("Ctrl+Right", "]"), scope="view2d"),
    ActionSpec("view2d.fit", "Fit the structure", "2D view", ("F",), scope="view2d"),
    # ---------------------------------------------------------------- Analyze's table
    ActionSpec("table.copy", "Copy cells", "Table", (SK.Copy,), scope="table"),
    ActionSpec("table.copy_with_titles", "Copy cells with column titles", "Table", ("Ctrl+Shift+C",),
               scope="table"),
    ActionSpec("table.select_all", "Select all cells", "Table", (SK.SelectAll,), scope="table"),
    # ---------------------------------------------------------------- Tools and Help
    ActionSpec("tools.settings", "Settings...", "Tools", ("Ctrl+,",)),
    ActionSpec("tools.controls", "Keyboard controls...", "Tools"),
    ActionSpec("tools.hardware_report", "Hardware report", "Tools"),
    ActionSpec("help.methods_guide", "Methods guide...", "Help", (SK.HelpContents,)),
    ActionSpec("help.guide_find", "Find in the Methods guide", "Help", (SK.Find,), scope="guide"),
    ActionSpec("help.check_updates", "Check for updates...", "Help"),
    ActionSpec("help.about", "About QuantaBench", "Help"),
)
BY_NAME: dict[str, ActionSpec] = {s.name: s for s in SPECS}
CATEGORIES = tuple(dict.fromkeys(s.category for s in SPECS))


# ====================================================================== presets


@dataclass(frozen=True)
class Preset:
    key: str
    label: str
    summary: str
    sources: tuple[tuple[str, str], ...] = ()       # (title, URL)
    #: action -> keys (portable text) where this preset differs from QuantaBench's defaults; () = no key
    keys: Mapping[str, tuple[str, ...]] = field(default_factory=dict)


AVOGADRO_URL = "https://avogadro.cc/docs/tools/keyboard-shortcuts.html"
AVOGADRO_FILE_URL = "https://avogadro.cc/docs/menus/file-menu.html"
VMD_URL = "https://tcbg.illinois.edu/Research/vmd/current/ug/node34.html"
SPARTAN_URL = "https://downloads-s3.wavefun.com/support/Spartan26Manual.pdf"

PRESETS: dict[str, Preset] = {
    "quantabench": Preset(
        "quantabench", "QuantaBench (default)",
        "QuantaBench's own keymap: the platform's standard keys (Open, Save, Undo, Paste, Find...), Ctrl+1-4 for "
        "the mouse modes, workspaces on " + ("Command+Option+1-5" if MAC else "Alt+1-5") + ", and plain keys only "
        "in the structure views (arrows step conformers and molecules, X/Y/Z look down an axis)."),
    "avogadro": Preset(
        "avogadro", "Avogadro 2",
        "Avogadro 2's documented keys: tools on Ctrl+1 (Navigate), Ctrl+2 (Draw), Ctrl+4 (Selection) and Ctrl+5 "
        "(Manipulate), select all / none on Ctrl+A / Ctrl+Shift+A, arrows (or WASD, HJKL) turn the 3D view, "
        "Shift+Up/Down zoom, Shift+Left/Right roll. Its arrow keys take the conformer and molecule keys' place.",
        (("Avogadro 2: Keyboard Shortcuts", AVOGADRO_URL), ("Avogadro 2: File menu", AVOGADRO_FILE_URL)),
        {
            "view.mode_rotate": ("Ctrl+1",), "view.mode_build": ("Ctrl+2",), "view.mode_select": ("Ctrl+4",),
            "view.mode_move": ("Ctrl+5",),
            "selection.select_all": ("Ctrl+A",), "selection.clear": ("Ctrl+Shift+A",),
            "kit.minimize": ("Ctrl+Alt+O",),
            "file.new_project": ("Ctrl+N",), "file.open": ("Ctrl+O",), "file.close_project": ("Ctrl+W",),
            "file.save_project": ("Ctrl+S",), "file.save_project_as": ("Ctrl+Shift+S",), "file.quit": ("Ctrl+Q",),
            "view3d.rotate_left": ("Left", "A", "H"), "view3d.rotate_right": ("Right", "D", "L"),
            "view3d.rotate_up": ("Up", "W", "K"), "view3d.rotate_down": ("Down", "S", "J"),
            "view3d.roll_ccw": ("Shift+Left",), "view3d.roll_cw": ("Shift+Right",),
            "view3d.zoom_in": ("Shift+Up",), "view3d.zoom_out": ("Shift+Down",),
            # the arrow keys turn the view in Avogadro; it documents no keys for stepping frames
            "playback.previous": (), "playback.next": (), "molecules.previous": (), "molecules.next": (),
        }),
    "vmd": Preset(
        "vmd", "VMD",
        "VMD's graphics-window hot keys: R rotate mode, 8 move-molecule mode, Ctrl+R reset view, H/J/K/L turn the "
        "view and G / Shift+G roll it, + / F next frame, - / B previous frame, . / (pause) play and stop. VMD's "
        "Ctrl+A and Ctrl+Z (scale) are left out: they are Select all and Undo here.",
        (("VMD User's Guide: Hot Keys", VMD_URL),),
        {
            "view.mode_rotate": ("R",), "view.mode_move": ("8",), "view.reset": ("Ctrl+R",),
            "view3d.rotate_left": ("H",), "view3d.rotate_right": ("L",), "view3d.rotate_up": ("K",),
            "view3d.rotate_down": ("J",), "view3d.roll_ccw": ("G",), "view3d.roll_cw": ("Shift+G",),
            "playback.next": ("+", "F", "Shift+F"), "playback.previous": ("-", "B", "Shift+B"),
            "playback.play": (".", ">", "/", "?"),
            # +, - and F step frames in VMD: QuantaBench's zoom and fit keys give way
            "view3d.zoom_in": (), "view3d.zoom_out": (), "view2d.zoom_in": (), "view2d.zoom_out": (),
            "view2d.fit": (),
        }),
    "spartan": Preset(
        "spartan", "Spartan",
        "The keys Spartan's Tutorial and User's Guide names in its text: Page Up / Page Down step through the "
        "molecules and Home / End go to the first and last. Its other keys are mouse modifiers (Insert or Delete "
        "held while clicking) or are only shown in the manual's menu pictures, so they are not here.",
        (("Spartan'26 Tutorial and User's Guide (Chapter 1, Keyboard Operations)", SPARTAN_URL),),
        {
            "molecules.previous": ("PgUp",), "molecules.next": ("PgDown",), "molecules.first": ("Home",),
            "molecules.last": ("End",), "edit.paste": ("Ctrl+V",),
            # Home and End are Spartan's first / last molecule
            "playback.first": (), "playback.last": (),
        }),
}
DEFAULT_PRESET = "quantabench"

#: Qt's dedicated-key bindings of the standard keys (the Open, Copy, Help... keys of some keyboards): left out
_HARDWARE_KEYS = {Qt.Key.Key_Open, Qt.Key.Key_Close, Qt.Key.Key_Save, Qt.Key.Key_New, Qt.Key.Key_Copy,
                  Qt.Key.Key_Paste, Qt.Key.Key_Cut, Qt.Key.Key_Undo, Qt.Key.Key_Redo, Qt.Key.Key_Help,
                  Qt.Key.Key_Find, Qt.Key.Key_Cancel}


# ====================================================================== keys


def seq(text) -> QKeySequence:
    """A QKeySequence from portable text, a StandardKey (its platform's first binding) or a QKeySequence."""
    if isinstance(text, QKeySequence):
        return text
    if isinstance(text, SK):
        return QKeySequence(text)
    return QKeySequence(str(text), QKeySequence.SequenceFormat.PortableText)


def portable(s: QKeySequence) -> str:
    return s.toString(QKeySequence.SequenceFormat.PortableText)


def native(s: QKeySequence) -> str:
    return s.toString(QKeySequence.SequenceFormat.NativeText)


def _unique(seqs: Iterable[QKeySequence]) -> list[QKeySequence]:
    out: list[QKeySequence] = []
    for s in seqs:
        if s.isEmpty() or any(s == o for o in out):
            continue
        if s.count() == 1 and s[0].key() in _HARDWARE_KEYS:
            continue
        out.append(s)
    return out


def default_keys(name: str) -> list[QKeySequence]:
    """QuantaBench's default keys of action ``name`` on this platform."""
    spec = BY_NAME.get(name)
    return _unique(seq(k) for k in spec.default) if spec is not None else []


def preset_keys(preset: str, name: str) -> list[QKeySequence]:
    """The keys of ``name`` in ``preset`` (its default where the preset does not set it)."""
    p = PRESETS.get(preset, PRESETS[DEFAULT_PRESET])
    if name in p.keys:
        return _unique(seq(k) for k in p.keys[name])
    return default_keys(name)


@dataclass
class KeymapState:
    """A preset and the user's changes on top of it: action -> keys in portable text (``()`` = no key)."""

    preset: str = DEFAULT_PRESET
    overrides: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def copy(self) -> KeymapState:
        return KeymapState(self.preset, dict(self.overrides))

    def keys(self, name: str) -> list[QKeySequence]:
        if name in self.overrides:
            return _unique(seq(k) for k in self.overrides[name])
        return preset_keys(self.preset, name)

    def set_keys(self, name: str, keys: Iterable) -> None:
        """Give ``name`` these keys; an override equal to the preset's keys is dropped."""
        new = _unique(seq(k) for k in keys)
        base = preset_keys(self.preset, name)
        if [portable(k) for k in new] == [portable(k) for k in base]:
            self.overrides.pop(name, None)
        else:
            self.overrides[name] = tuple(portable(k) for k in new)

    def reset(self, name: str) -> None:
        self.overrides.pop(name, None)

    def is_changed(self, name: str) -> bool:
        return name in self.overrides

    def set_preset(self, preset: str) -> None:
        """Switch preset; the user's changes stay on top of it (those that now equal the preset are dropped)."""
        if preset not in PRESETS:
            raise ValueError(f"Unknown keymap preset {preset!r}")
        self.preset = preset
        for name in list(self.overrides):
            self.set_keys(name, self.overrides[name])

    # -------------------------------------------------------------- JSON
    def to_dict(self) -> dict:
        return {"quantabench_keymap": 1, "preset": self.preset,
                "overrides": {k: list(v) for k, v in sorted(self.overrides.items())},
                "platform": sys.platform,
                "note": "QuantaBench keymap (Settings > Controls): a preset and the changes on top of it. Keys are "
                        "in Qt's portable text; Ctrl is Command and Meta is Control on macOS."}

    @classmethod
    def from_dict(cls, data: Mapping) -> tuple[KeymapState, list[str]]:
        """(state, problems): unknown actions and keys that do not parse are left out and named."""
        if not isinstance(data, Mapping) or "quantabench_keymap" not in data:
            raise ValueError("Not a QuantaBench keymap (no \"quantabench_keymap\" entry)")
        problems = []
        preset = str(data.get("preset") or DEFAULT_PRESET)
        if preset not in PRESETS:
            problems.append(f"unknown preset {preset!r}: using {PRESETS[DEFAULT_PRESET].label}")
            preset = DEFAULT_PRESET
        state = cls(preset)
        raw = data.get("overrides") or {}
        if not isinstance(raw, Mapping):
            raise ValueError("\"overrides\" must be an object (action name -> list of keys)")
        for name, keys in raw.items():
            if name not in BY_NAME:
                problems.append(f"unknown action {name!r} left out")
                continue
            if isinstance(keys, str):
                keys = [keys] if keys else []
            good = []
            for k in keys or []:
                s = seq(k)
                if s.count() != 1 or not portable(s) or s[0].key() == Qt.Key.Key_unknown:
                    problems.append(f"{name}: key {k!r} not understood")
                    continue
                good.append(s)
            state.overrides[name] = tuple(portable(s) for s in _unique(good))
        for name in list(state.overrides):      # drop changes that equal the preset
            state.set_keys(name, state.overrides[name])
        platform = data.get("platform")
        if platform and platform != sys.platform and (platform == "darwin" or MAC):
            problems.append(f"made on {_platform_name(platform)}: Ctrl is Command on macOS and Control elsewhere, "
                            "so check the modifier keys")
        return state, problems


def _platform_name(p: str) -> str:
    return {"darwin": "macOS", "win32": "Windows"}.get(p, "Linux" if p.startswith("linux") else p)


def scopes_overlap(a: str, b: str) -> bool:
    """One scope contains the other (or they are the same): the same key in both is ambiguous."""
    def chain(s):
        out = []
        while s is not None:
            out.append(s)
            s = SCOPES.get(s)
        return out
    return a in chain(b) or b in chain(a)


def conflicts(state: KeymapState, name: str, key: QKeySequence, names: Iterable[str]) -> list[str]:
    """The actions among ``names`` (other than ``name``) whose keys include ``key`` in an overlapping scope."""
    scope = BY_NAME[name].scope if name in BY_NAME else "window"
    out = []
    for other in names:
        if other == name or other not in BY_NAME or not scopes_overlap(scope, BY_NAME[other].scope):
            continue
        if any(k == key for k in state.keys(other)):
            out.append(other)
    return out


def all_conflicts(state: KeymapState, names: Iterable[str]) -> dict[str, list[tuple[str, str]]]:
    """action -> [(key in native text, other action)] for every key two actions share."""
    names = list(names)
    out: dict[str, list[tuple[str, str]]] = {}
    for name in names:
        for k in state.keys(name):
            for other in conflicts(state, name, k, names):
                out.setdefault(name, []).append((native(k), other))
    return out


# ====================================================================== actions in widgets

_CONTEXT = {"window": Qt.ShortcutContext.WindowShortcut}


def context_of(name: str) -> Qt.ShortcutContext:
    spec = BY_NAME.get(name)
    return _CONTEXT.get(spec.scope if spec else "window", Qt.ShortcutContext.WidgetWithChildrenShortcut)


def _set_keys(action: QAction, keys: list[QKeySequence]) -> None:
    action.setShortcuts(keys)
    tip = action.property("qb_tip")
    if tip is None:
        return
    text = ", ".join(native(k) for k in keys)
    base = tip or action.text().replace("&", "")
    action.setToolTip(f"{base} ({text})" if text else base)


def prepare(action: QAction, name: str, tip: str | None = None) -> QAction:
    """Name ``action`` and give it its default keys and the tooltip that shows them (no keymap needed)."""
    if name not in BY_NAME:
        raise KeyError(f"{name!r} is not in keymap.SPECS")
    action.setObjectName(name)
    action.setShortcutContext(context_of(name))
    if tip is not None:
        action.setProperty("qb_tip", tip)
    elif action.property("qb_tip") is None:
        action.setProperty("qb_tip", action.toolTip() if action.toolTip() != action.text().replace("&", "") else "")
    _set_keys(action, default_keys(name))
    return action


def local_action(widget: QWidget, name: str, slot: Callable, text: str = "", tip: str | None = None) -> QAction:
    """An action of a view-local scope (``name``'s spec): made on ``widget`` with its default keys; a window's
    :class:`Keymap` applies the user's keys when it adopts it (``Keymap.bind``)."""
    a = QAction(text or BY_NAME[name].title, widget)
    a.triggered.connect(lambda _checked=False: slot())
    prepare(a, name, tip)
    widget.addAction(a)
    return a


def _candidates(event) -> list[QKeySequence]:
    """The key sequences ``event`` can stand for (Shift dropped for a symbol that needs it, keypad ignored)."""
    combo = event.keyCombination()
    key, mods = combo.key(), combo.keyboardModifiers() & ~Qt.KeyboardModifier.KeypadModifier
    out = [QKeySequence(QKeyCombination(mods, key))]
    is_letter = Qt.Key.Key_A.value <= key.value <= Qt.Key.Key_Z.value
    if mods & Qt.KeyboardModifier.ShiftModifier and not is_letter and key.value < Qt.Key.Key_Escape.value:
        out.append(QKeySequence(QKeyCombination(mods & ~Qt.KeyboardModifier.ShiftModifier, key)))
    return out


def dispatch_key(event, widget: QWidget) -> bool:
    """Run the action a key press stands for among the widget-scoped actions of ``widget`` and its parents (up to
    its window), as Qt's shortcut map would; True when one ran. Views call it from their key handling so their keys
    work even when the shortcut map did not take the key (it only does in the active window)."""
    if event.type() != QEvent.Type.KeyPress:
        return False
    pressed = _candidates(event)
    w = widget
    while w is not None:
        for a in w.actions():
            ctx = a.shortcutContext()
            if ctx == Qt.ShortcutContext.WidgetShortcut and w is not widget:
                continue
            if ctx not in (Qt.ShortcutContext.WidgetShortcut, Qt.ShortcutContext.WidgetWithChildrenShortcut):
                continue
            if not a.isEnabled():
                continue
            if any(k == p for k in a.shortcuts() for p in pressed):
                a.trigger()
                event.accept()
                return True
        if w.isWindow():
            break
        w = w.parentWidget()
    return False


class KeyDispatcher(QObject):
    """Event filter for a widget that eats its own keys (the VTK view): ``dispatch_key`` before the widget sees
    them."""

    def __init__(self, widget: QWidget):
        super().__init__(widget)
        self.widget = widget
        widget.installEventFilter(self)

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 - Qt's name
        if event.type() == QEvent.Type.KeyPress:
            return dispatch_key(event, obj if isinstance(obj, QWidget) else self.widget)
        return False


# ====================================================================== the window's keymap

STORE_PRESET = "keymap/preset"
STORE_OVERRIDES = "keymap/overrides"


class Keymap(QObject):
    """The window's actions by name, the active keymap (a preset and the user's changes) and where it is kept.

    ``store`` is the window settings (QSettings: per computer and user; an INI file in the tests) or None (not kept).
    """

    changed = Signal()

    def __init__(self, parent: QObject | None = None, store: QSettings | None = None):
        super().__init__(parent)
        self.store = store
        self.state = KeymapState()
        self._actions: dict[str, list[QAction]] = {}
        self._followers: list[tuple[QWidget, str, str]] = []
        self.load()

    # -------------------------------------------------------------- actions
    def bind(self, action: QAction, name: str, tip: str | None = None, widget: QWidget | None = None) -> QAction:
        """Name ``action`` ``name``, give it the active keys and keep them up to date. ``widget``: where a
        view-local action lives (added to it); a window action is added to the window when it is in no menu."""
        prepare(action, name, tip)
        lst = self._actions.setdefault(name, [])
        if action not in lst:
            lst.append(action)          # deleted ones are dropped when next seen (_alive): no destroyed handler
        if widget is not None and action not in widget.actions():
            widget.addAction(action)
        elif widget is None and BY_NAME[name].scope == "window" and isinstance(self.parent(), QWidget) \
                and not action.associatedObjects():
            self.parent().addAction(action)
        _set_keys(action, self.state.keys(name))
        return action

    def action(self, name: str, text: str, slot: Callable | None = None, tip: str | None = None,
               widget: QWidget | None = None, parent: QObject | None = None) -> QAction:
        """A new action ``name`` (``slot`` on triggered), bound."""
        a = QAction(text, parent or self.parent())
        if slot is not None:
            a.triggered.connect(slot)
        return self.bind(a, name, tip, widget)

    def _alive(self) -> dict[str, list[QAction]]:
        """The bound actions that still exist (a closed window's or panel's are dropped)."""
        for name in list(self._actions):
            live = [a for a in self._actions[name] if shiboken6.isValid(a)]
            if live:
                self._actions[name] = live
            else:
                del self._actions[name]
        return self._actions

    def actions(self, name: str) -> list[QAction]:
        return list(self._alive().get(name, []))

    def names(self) -> list[str]:
        """The bound actions, in the registry's order."""
        alive = self._alive()
        return [s.name for s in SPECS if s.name in alive]

    def keys(self, name: str) -> list[QKeySequence]:
        return self.state.keys(name)

    def key_text(self, name: str) -> str:
        """The first key of ``name`` as the platform writes it ("" when it has none)."""
        keys = self.keys(name)
        return native(keys[0]) if keys else ""

    def follow_tooltip(self, widget: QWidget, name: str, tip: str) -> None:
        """Keep ``widget``'s tooltip as ``tip`` plus ``name``'s key (a button that does what the action does)."""
        self._followers.append((widget, name, tip))
        self._update_follower(widget, name, tip)

    def _update_follower(self, widget, name, tip) -> None:
        key = self.key_text(name)
        widget.setToolTip(f"{tip} ({key})" if key else tip)

    # -------------------------------------------------------------- state
    def set_state(self, state: KeymapState, save: bool = True) -> None:
        self.state = state.copy()
        self.apply()
        if save:
            self.save()
        self.changed.emit()

    def apply(self) -> None:
        for name, actions in self._alive().items():
            keys = self.state.keys(name)
            for a in actions:
                _set_keys(a, keys)
        self._followers = [f for f in self._followers if shiboken6.isValid(f[0])]
        for widget, name, tip in self._followers:
            self._update_follower(widget, name, tip)

    def conflicts(self) -> dict[str, list[tuple[str, str]]]:
        return all_conflicts(self.state, self.names())

    def load(self) -> None:
        if self.store is None:
            return
        preset = str(self.store.value(STORE_PRESET, DEFAULT_PRESET) or DEFAULT_PRESET)
        raw = self.store.value(STORE_OVERRIDES, "") or ""
        try:
            data = {"quantabench_keymap": 1, "preset": preset, "overrides": json.loads(raw) if raw else {}}
            self.state, _problems = KeymapState.from_dict(data)
        except (ValueError, TypeError):
            self.state = KeymapState()

    def save(self) -> None:
        if self.store is None:
            return
        self.store.setValue(STORE_PRESET, self.state.preset)
        self.store.setValue(STORE_OVERRIDES, json.dumps({k: list(v) for k, v in self.state.overrides.items()}))
        self.store.sync()


def export_json(state: KeymapState, path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(state.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def import_json(path) -> tuple[KeymapState, list[str]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return KeymapState.from_dict(data)


def bind(win, action: QAction, name: str, tip: str | None = None, widget: QWidget | None = None) -> QAction:
    """``win.keymap.bind`` when the window has a keymap, else the action's default keys (a panel installed on a
    stand-in window in a test)."""
    km = getattr(win, "keymap", None)
    if isinstance(km, Keymap):
        return km.bind(action, name, tip, widget)
    prepare(action, name, tip)
    if widget is not None and action not in widget.actions():
        widget.addAction(action)
    return action
