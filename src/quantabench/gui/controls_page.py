"""Settings > Controls (UI-9): the keymap's preset, every named action's keys, rebinding, conflicts, reset,
export and import.

The page edits a copy of the window's :class:`~quantabench.gui.keymap.KeymapState` (a preset and the user's changes
on top of it); ``MainWindow.open_settings`` applies it when the dialog is accepted, like every other tab. It is a
display setting: it never changes a calculation and is kept per computer (``gui.keymap``, docs/CONTROLS.md).

- **Keymap:** QuantaBench (the default) or a preset modelled on another program's documented keys, with its
  summary and sources.
- **The table:** category, action and keys of every action the window has, searchable (an action's name, category or
  key). Changed keys are bold; keys that clash with another action are red, with the other action in the tooltip.
- **Rebinding:** select a row, click the key field and press the key. A key another action has in the same place
  asks: *Swap* (the other action gets this one's old keys), *Clear the other* (it loses the key), or Cancel.
- **Reset this action** (to the keymap's keys), **Clear**, **Reset to QuantaBench defaults** (the default keymap
  and no changes), **Export...** / **Import...** a keymap JSON file.
"""

from __future__ import annotations

import html
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QKeySequenceEdit,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import keymap as km

NAME_ROLE = Qt.ItemDataRole.UserRole
CONFLICT_COLOR = QColor(200, 40, 40)


class ControlsPage(QWidget):
    COLUMNS = ("Category", "Action", "Shortcut")

    def __init__(self, keymap: km.Keymap | None = None, parent=None):
        super().__init__(parent)
        self.keymap = keymap
        self.state = keymap.state.copy() if keymap is not None else km.KeymapState()
        self.names = self._listed_names()
        self.message = ""                  # the last thing the page said (status line; the tests read it)

        lay = QVBoxLayout(self)
        form = QFormLayout()
        self.preset_box = QComboBox()
        for p in km.PRESETS.values():
            self.preset_box.addItem(p.label, p.key)
        self.preset_box.setToolTip("Start from QuantaBench's keys or from a keymap modelled on another program's "
                                   "documented keys; your own changes stay on top of it")
        form.addRow("Keymap", self.preset_box)
        lay.addLayout(form)
        self.preset_info = QLabel()
        self.preset_info.setWordWrap(True)
        self.preset_info.setTextFormat(Qt.TextFormat.RichText)
        self.preset_info.setOpenExternalLinks(True)
        self.preset_info.setStyleSheet("color: gray;")
        lay.addWidget(self.preset_info)

        self.search = QLineEdit()
        self.search.setPlaceholderText("Search actions, categories or keys")
        self.search.setClearButtonEnabled(True)
        lay.addWidget(self.search)
        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setObjectName("controls_table")
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setDefaultSectionSize(self.fontMetrics().height() + 8)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        lay.addWidget(self.table, 1)

        edit_row = QHBoxLayout()
        self.selected_label = QLabel("Select an action")
        edit_row.addWidget(self.selected_label, 1)
        self.capture = QKeySequenceEdit()
        self.capture.setObjectName("key_capture")
        self.capture.setMaximumSequenceLength(1)
        self.capture.setClearButtonEnabled(False)
        self.capture.setToolTip("Click here and press the new key for the selected action")
        self.capture.setMinimumWidth(150)
        edit_row.addWidget(self.capture)
        self.b_clear = QPushButton("Clear")
        self.b_clear.setToolTip("No key for the selected action")
        edit_row.addWidget(self.b_clear)
        self.b_reset_action = QPushButton("Reset this action")
        self.b_reset_action.setToolTip("The selected action's keys in the chosen keymap (drops your change)")
        edit_row.addWidget(self.b_reset_action)
        lay.addLayout(edit_row)
        self.status = QLabel()
        self.status.setWordWrap(True)
        lay.addWidget(self.status)

        bottom = QHBoxLayout()
        self.b_reset_all = QPushButton("Reset to QuantaBench defaults")
        self.b_reset_all.setToolTip("The QuantaBench keymap and none of your changes (applied when you press OK)")
        bottom.addWidget(self.b_reset_all)
        bottom.addStretch(1)
        self.b_import = QPushButton("Import...")
        self.b_import.setToolTip("Read a keymap JSON file (from Export... here or on another computer)")
        self.b_export = QPushButton("Export...")
        self.b_export.setToolTip("Save this keymap (its preset and your changes) as a JSON file")
        bottom.addWidget(self.b_import)
        bottom.addWidget(self.b_export)
        lay.addLayout(bottom)
        note = QLabel(
            "Kept on this computer, like the window layout: Windows and macOS use different modifier keys (Ctrl "
            "here is Command on a Mac) and laptops lack some keys, so each computer has its own keymap; Export and "
            "Import carry one across. Plain keys (letters, arrows, Space) work in the structure views only, so "
            "typing in a field or a list is never taken over. Changes apply when you press OK.")
        note.setWordWrap(True)
        note.setStyleSheet("color: gray;")
        lay.addWidget(note)

        self.preset_box.currentIndexChanged.connect(self._on_preset)
        self.search.textChanged.connect(self.apply_filter)
        self.table.itemSelectionChanged.connect(self._on_row_changed)
        self.capture.editingFinished.connect(self._on_captured)
        self.b_clear.clicked.connect(lambda _c=False: self.clear_selected())
        self.b_reset_action.clicked.connect(lambda _c=False: self.reset_selected())
        self.b_reset_all.clicked.connect(lambda _c=False: self.reset_all())
        self.b_import.clicked.connect(lambda _c=False: self.import_dialog())
        self.b_export.clicked.connect(lambda _c=False: self.export_dialog())
        self._fill()
        self._sync_preset_box()

    # ------------------------------------------------------------------ rows
    def _listed_names(self) -> list[str]:
        """The window's actions (those of this build), with the Methods guide's (a window of its own, made when
        it opens); every action of the build when there is no window."""
        if self.keymap is not None and self.keymap.names():
            have = set(self.keymap.names()) | {s.name for s in km.SPECS if s.scope == "guide"}
        else:
            have = {s.name for s in km.SPECS if not s.note}
        order = {c: i for i, c in enumerate(km.CATEGORIES)}
        specs = [s for s in km.SPECS if s.name in have]
        specs.sort(key=lambda s: order[s.category])          # stable: the registry's order within a category
        return [s.name for s in specs]

    def _fill(self) -> None:
        self.table.setRowCount(len(self.names))
        for r, name in enumerate(self.names):
            spec = km.BY_NAME[name]
            cat = QTableWidgetItem(spec.category)
            cat.setData(NAME_ROLE, name)
            title = QTableWidgetItem(spec.title)
            title.setToolTip(f"{spec.title}: works {km.SCOPE_LABELS[spec.scope]} ({name})")
            self.table.setItem(r, 0, cat)
            self.table.setItem(r, 1, title)
            self.table.setItem(r, 2, QTableWidgetItem())
        self.refresh()

    def row_of(self, name: str) -> int:
        return self.names.index(name) if name in self.names else -1

    def keys_text(self, name: str) -> str:
        return ", ".join(km.native(k) for k in self.state.keys(name))

    def refresh(self) -> None:
        """Every row's keys, bold where changed, red where they clash; the status line for the selection."""
        clashes = km.all_conflicts(self.state, self.names)
        bold = QFont(self.table.font())
        bold.setBold(True)
        for r, name in enumerate(self.names):
            item = self.table.item(r, 2)
            text = self.keys_text(name)
            item.setText(text or "(none)")
            item.setFont(bold if self.state.is_changed(name) else self.table.font())
            tips = []
            if self.state.is_changed(name):
                base = ", ".join(km.native(k) for k in km.preset_keys(self.state.preset, name)) or "none"
                tips.append(f"Changed by you ({km.PRESETS[self.state.preset].label}: {base})")
            if name in clashes:
                item.setForeground(CONFLICT_COLOR)
                tips += [f"{key} is also “{km.BY_NAME[o].title}” ({km.BY_NAME[o].category})" for key, o in clashes[name]]
            else:
                item.setForeground(self.palette().text())
                if not text:
                    item.setForeground(QColor(140, 140, 140))
            item.setToolTip("\n".join(tips))
        self._show_status(clashes)
        self.apply_filter(self.search.text())

    def _show_status(self, clashes=None) -> None:
        clashes = km.all_conflicts(self.state, self.names) if clashes is None else clashes
        name = self.selected()
        parts = []
        if name in clashes:
            parts.append("<span style='color:#c82828'>" + html.escape("; ".join(
                f"{key} is also “{km.BY_NAME[o].title}”" for key, o in clashes[name])) + "</span>")
        n = len(self.state.overrides)
        if n:
            parts.append(f"{n} change{'s' if n != 1 else ''} of yours on top of "
                         f"{html.escape(km.PRESETS[self.state.preset].label)}.")
        if self.message:
            parts.append(html.escape(self.message))
        self.status.setText(" ".join(parts))

    def apply_filter(self, text: str = "") -> None:
        t = text.strip().lower()
        for r, name in enumerate(self.names):
            spec = km.BY_NAME[name]
            hay = " ".join((spec.category, spec.title, name, self.keys_text(name))).lower()
            self.table.setRowHidden(r, bool(t) and t not in hay)

    def visible_names(self) -> list[str]:
        return [n for r, n in enumerate(self.names) if not self.table.isRowHidden(r)]

    # ------------------------------------------------------------------ selection and editing
    def selected(self) -> str | None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return None
        return self.table.item(rows[0].row(), 0).data(NAME_ROLE)

    def select(self, name: str) -> None:
        r = self.row_of(name)
        if r >= 0:
            self.table.doItemsLayout()          # rows the search just hid or showed: positions first
            self.table.selectRow(r)
            self.table.scrollToItem(self.table.item(r, 0))

    def _on_row_changed(self) -> None:
        self.message = ""                    # a new row: the last message was about another one
        self._on_selection()

    def _on_selection(self) -> None:
        """The key field, buttons and status line for the selected action."""
        name = self.selected()
        on = name is not None
        for w in (self.capture, self.b_clear, self.b_reset_action):
            w.setEnabled(on)
        self.capture.blockSignals(True)
        keys = self.state.keys(name) if on else []
        self.capture.setKeySequence(keys[0] if keys else QKeySequence())
        self.capture.blockSignals(False)
        self.selected_label.setText(f"<b>{html.escape(km.BY_NAME[name].title)}</b>" if on else "Select an action")
        self._show_status()

    def _on_captured(self) -> None:
        name = self.selected()
        seq = self.capture.keySequence()
        if name is None or seq.isEmpty():
            return
        key = QKeySequence(seq[0])
        if any(key == k for k in self.state.keys(name)) and len(self.state.keys(name)) == 1:
            return
        self.assign(name, key)

    def assign(self, name: str, key) -> bool:
        """Give ``name`` the one key ``key`` (a QKeySequence or portable text). A clash asks (``ask_conflict``):
        swap, clear the other action's key, or cancel; False when cancelled."""
        key = km.seq(key)
        others = km.conflicts(self.state, name, key, self.names)
        if others:
            choice = self.ask_conflict(name, key, others)
            if choice == "cancel":
                self._on_selection()             # the field shows the action's key again
                return False
            old = self.state.keys(name)
            for o in others:
                keep = [k for k in self.state.keys(o) if k != key]
                if choice == "swap":
                    keep += [k for k in old if k != key and not any(k == x for x in keep)]
                self.state.set_keys(o, keep)
            titles = ", ".join(f"“{km.BY_NAME[o].title}”" for o in others)
            self.message = (f"Swapped: {titles} now has {', '.join(km.native(k) for k in old) or 'no key'}."
                            if choice == "swap" else f"{km.native(key)} taken from {titles}.")
        else:
            self.message = ""
        self.state.set_keys(name, [key])
        self.refresh()
        return True

    def ask_conflict(self, name: str, key: QKeySequence, others: list[str]) -> str:
        """"swap", "clear" or "cancel" for a key that ``others`` already have in the same place."""
        titles = ", ".join(f"“{km.BY_NAME[o].title}” ({km.BY_NAME[o].category})" for o in others)
        old = ", ".join(km.native(k) for k in self.state.keys(name)) or "no key"
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Key already in use")
        box.setText(f"{km.native(key)} is already the key of {titles}.")
        box.setInformativeText(f"Swap gives it the key(s) of “{km.BY_NAME[name].title}” now ({old}); Clear takes "
                               f"{km.native(key)} away from it.")
        swap = box.addButton("Swap", QMessageBox.ButtonRole.AcceptRole)
        clear = box.addButton("Clear the other", QMessageBox.ButtonRole.DestructiveRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        clicked = box.clickedButton()
        return "swap" if clicked is swap else "clear" if clicked is clear else "cancel"

    def clear_selected(self) -> None:
        name = self.selected()
        if name is not None:
            self.state.set_keys(name, [])
            self.message = f"“{km.BY_NAME[name].title}” has no key now."
            self.refresh()
            self._on_selection()

    def reset_selected(self) -> None:
        name = self.selected()
        if name is not None:
            self.state.reset(name)
            self.message = f"“{km.BY_NAME[name].title}” has {km.PRESETS[self.state.preset].label}'s keys again."
            self.refresh()
            self._on_selection()

    def reset_all(self) -> None:
        """The QuantaBench keymap and no changes."""
        self.state = km.KeymapState()
        self.message = "QuantaBench's default keys (applied when you press OK)."
        self._sync_preset_box()
        self.refresh()
        self._on_selection()

    # ------------------------------------------------------------------ presets
    def _sync_preset_box(self) -> None:
        self.preset_box.blockSignals(True)
        self.preset_box.setCurrentIndex(max(0, self.preset_box.findData(self.state.preset)))
        self.preset_box.blockSignals(False)
        self._show_preset_info()

    def _show_preset_info(self) -> None:
        p = km.PRESETS[self.state.preset]
        links = " · ".join(f"<a href='{html.escape(url)}'>{html.escape(title)}</a>" for title, url in p.sources)
        self.preset_info.setText(html.escape(p.summary) + (f"<br>Source: {links}" if links else ""))

    def _on_preset(self, _index: int) -> None:
        self.set_preset(self.preset_box.currentData())

    def set_preset(self, key: str) -> None:
        self.state.set_preset(key)
        self.message = f"Keymap: {km.PRESETS[key].label}."
        self._sync_preset_box()
        self.refresh()
        self._on_selection()

    # ------------------------------------------------------------------ files
    def export_to(self, path) -> Path:
        out = km.export_json(self.state, path)
        self.message = f"Saved {Path(out).name}."
        self._show_status()
        return out

    def import_from(self, path) -> list[str]:
        """Read a keymap file into the page; the problems found (unknown actions, keys) are returned and shown."""
        state, problems = km.import_json(path)
        self.state = state
        self.message = f"Read {Path(path).name}" + (": " + "; ".join(problems) if problems else ".")
        self._sync_preset_box()
        self.refresh()
        self._on_selection()
        return problems

    def export_dialog(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export keymap", "quantabench-keymap.json", "Keymap (*.json)")
        if path:
            try:
                self.export_to(path)
            except OSError as exc:
                QMessageBox.warning(self, "Export keymap", str(exc))

    def import_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import keymap", "", "Keymap (*.json);;All files (*)")
        if not path:
            return
        try:
            problems = self.import_from(path)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Import keymap", f"Could not read {Path(path).name}:\n{exc}")
            return
        if problems:
            QMessageBox.information(self, "Import keymap", "Read with these notes:\n" + "\n".join(problems))

    def result_state(self) -> km.KeymapState:
        return self.state.copy()
