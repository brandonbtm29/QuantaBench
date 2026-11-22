"""The molecule list's right-click *Look up* menu (SETTINGS-1, Spartan's Preferences > URLs): one entry per template
of Settings > Lookups (``settings.lookups.urls``, expanded by ``quantabench.lookups``), opened in the browser."""

from __future__ import annotations

from PySide6.QtCore import QPoint, Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QMenu

from .. import lookups


def open_url(url: str) -> bool:
    """Open ``url`` in the default browser (a function, so tests can replace it)."""
    return QDesktopServices.openUrl(QUrl(url))


def molecule_values(win, ident: str) -> dict:
    """The placeholder values of the list's molecule ``ident``: its structure as drawn (the input, else the list's
    SMILES), name, id and posted data fields."""
    row = win.model.row(ident)
    if row is None:
        return {}
    inp = row.input
    mol = None
    if inp is not None:
        try:
            mol = inp.to_mol()
        except Exception:  # noqa: BLE001 - an unreadable structure: no structure keys
            mol = None
    smiles = (getattr(inp, "smiles", "") if inp is not None else "") or row.smiles
    name = row.name or (getattr(inp, "name", "") if inp is not None else "")
    return lookups.values_for(mol, smiles=smiles or "", name=name or "", ident=row.id, fields=row.fields or {})


def add_lookups(menu: QMenu, win, ident: str) -> QMenu:
    """A *Look up* submenu of ``menu`` for molecule ``ident``: one action per template, greyed out (with the reason)
    where the molecule lacks a value the template needs; the URL is each action's tooltip."""
    sub = menu.addMenu("Look up")
    sub.setObjectName("lookup_menu")
    sub.setToolTipsVisible(True)
    values = molecule_values(win, ident)
    found = lookups.templates(getattr(getattr(win, "s", None), "lookups", None).urls
                              if getattr(win, "s", None) is not None else None)
    for t in found:
        url, missing = lookups.expand(t.url, values)
        act = sub.addAction(t.name)
        if missing:
            act.setEnabled(False)
            act.setText(f"{t.name} (no {', '.join(lookups.PLACEHOLDER_HELP[m] for m in missing)})")
        else:
            act.setToolTip(url)
            act.setData(url)
            act.triggered.connect(lambda _checked=False, u=url: open_url(u))
    if not found:
        sub.addAction("No lookups (Settings > Lookups)").setEnabled(False)
    if hasattr(win, "open_settings"):
        sub.addSeparator()
        sub.addAction("Edit lookups...").triggered.connect(win.open_settings)
    return sub


def molecule_menu(win, ident: str) -> QMenu:
    """The right-click menu of the molecule list's row ``ident``."""
    menu = QMenu(win)
    menu.setObjectName("molecule_menu")
    add_lookups(menu, win, ident)
    return menu


def install(win) -> None:
    """Give ``win.mol_list`` its right-click menu."""
    view = win.mol_list
    view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

    def show(pos: QPoint) -> None:
        index = view.indexAt(pos)
        row = view.row_at(index.row()) if index.isValid() else None
        if row is None:
            return
        molecule_menu(win, row.id).exec(view.viewport().mapToGlobal(pos))

    view.customContextMenuRequested.connect(show)
