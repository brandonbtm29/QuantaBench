"""Named actions and their default shortcuts, in one place, so a rebinding page (BACKLOG UI-9) can list and change
them without knowing which module made each action.

    bind(action, "edit.undo")      # sets the action's object name and its default shortcuts; returns the action

A name is ``<menu or area>.<what>``. A default is a tuple of key sequences in Qt's portable text ("Ctrl+4"; Ctrl is
Command on macOS) or ``<Undo>``-style names of ``QKeySequence.StandardKey`` (the platform's own keys). None of these
defaults copies another program's: the mouse modes are numbered in toolbar order (Ctrl+1 Rotate ... Ctrl+4 Build),
and undo/redo are each platform's standard keys plus Ctrl+Y.
"""

from __future__ import annotations

from PySide6.QtGui import QAction, QKeySequence

DEFAULTS: dict[str, tuple[str, ...]] = {
    "view.mode_rotate": ("Ctrl+1",),
    "view.mode_select": ("Ctrl+2",),
    "view.mode_move": ("Ctrl+3",),
    "view.mode_build": ("Ctrl+4",),
    "edit.undo": ("<Undo>",),
    "edit.redo": ("<Redo>", "Ctrl+Y"),
    "build.model_kit": (),
}


def shortcuts(name: str) -> list[QKeySequence]:
    """The default key sequences of action ``name`` (duplicates dropped: Redo is Ctrl+Y on Windows already)."""
    out: list[QKeySequence] = []
    for text in DEFAULTS[name]:
        if text.startswith("<") and text.endswith(">"):
            seqs = QKeySequence.keyBindings(getattr(QKeySequence.StandardKey, text[1:-1]))
        else:
            seqs = [QKeySequence(text)]
        out += [s for s in seqs if not s.isEmpty() and s not in out]
    return out


def bind(action: QAction, name: str) -> QAction:
    """Give ``action`` the object name ``name`` and its default shortcuts."""
    action.setObjectName(name)
    action.setShortcuts(shortcuts(name))
    return action
