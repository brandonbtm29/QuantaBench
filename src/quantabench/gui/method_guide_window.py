"""Help > Methods guide...: every entry of ``chem.method_guide`` by family, with a search box.

CONTRACT:

    show(win, key="") -> MethodsGuideWindow
        Opens (or raises) the window owned by ``win`` and shows entry ``key`` (an entry key, alias or anchor;
        "" = the index). The pipeline pages' "More…" links call this.

The left side lists the families and their entries; typing in the search box filters it (names and aliases
first, then any word of the text, as ``method_guide.search``) and shows the matches on the right. Links between
entries (``guide:<anchor>``) stay in the window; DOI links open in the web browser.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSplitter,
    QTextBrowser,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..chem import method_guide as guide

TITLE = "Methods guide"
KEY_ROLE = Qt.ItemDataRole.UserRole


class MethodsGuideWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle(f"QuantaBench {TITLE}")
        self.setObjectName("methods_guide")
        self.resize(1000, 700)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search: a name (wB97X-D, GFN2, 6-31G*) or any words (anion, dispersion)")
        self.search.setClearButtonEnabled(True)
        self.count_label = QLabel()
        self.count_label.setStyleSheet("color: gray;")
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setMinimumWidth(240)
        self.items: dict[str, QTreeWidgetItem] = {}
        for family, entries in guide.by_family():
            top = QTreeWidgetItem([family])
            top.setData(0, KEY_ROLE, "")
            font = top.font(0)
            font.setBold(True)
            top.setFont(0, font)
            self.tree.addTopLevelItem(top)
            for e in entries:
                it = QTreeWidgetItem([e.name])
                it.setData(0, KEY_ROLE, e.key)
                it.setToolTip(0, guide.plain(e.summary))
                top.addChild(it)
                self.items[e.key] = it
        self.tree.expandAll()
        self.browser = QTextBrowser()
        self.browser.setOpenLinks(False)          # guide: links stay here, DOIs go to the web browser
        self.browser.anchorClicked.connect(self._on_anchor)
        self.browser.document().setDefaultStyleSheet("h2 { margin-bottom: 2px; } li { margin-bottom: 3px; }")
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self.tree)
        split.addWidget(self.browser)
        split.setStretchFactor(1, 1)
        split.setSizes([280, 720])
        top_row = QHBoxLayout()
        top_row.addWidget(self.search, 1)
        top_row.addWidget(self.count_label)
        lay = QVBoxLayout(self)
        lay.addLayout(top_row)
        lay.addWidget(split, 1)
        self.current_key = ""
        self.tree.currentItemChanged.connect(self._on_item)
        self.search.textChanged.connect(self.apply_filter)
        self.search.returnPressed.connect(self._open_first)
        QShortcut(QKeySequence.StandardKey.Find, self, activated=self.search.setFocus)
        self.apply_filter("")

    # ------------------------------------------------------------------ navigation
    def show_entry(self, key: str) -> bool:
        """Show entry ``key`` (a key, alias, name or anchor); False when there is none (the index is shown)."""
        e = guide.lookup(key) if key else None
        if e is None and key:
            e = guide.by_anchor(key)
        if e is None:
            self.current_key = ""
            self.browser.setHtml(guide.html_index(self._visible_entries()))
            return False
        if self.items[e.key].isHidden():               # an entry the filter hides: clear the filter first
            self.search.blockSignals(True)
            self.search.clear()
            self.search.blockSignals(False)
            self._filter(None)
            self.count_label.setText(f"{len(guide.ENTRIES)} entries")
        self.current_key = e.key
        self.tree.blockSignals(True)
        self.tree.setCurrentItem(self.items[e.key])
        self.tree.blockSignals(False)
        self.tree.scrollToItem(self.items[e.key])
        self.browser.setHtml(guide.html_entry(e))
        return True

    def _on_item(self, item, _previous=None) -> None:
        if item is None:
            return
        key = item.data(0, KEY_ROLE)
        if key:
            self.show_entry(key)
        else:                                          # a family: its entries as an index
            self.current_key = ""
            entries = [guide.ENTRIES[item.child(j).data(0, KEY_ROLE)] for j in range(item.childCount())
                       if not item.child(j).isHidden()]
            self.browser.setHtml(guide.html_index(entries, item.text(0)))

    def _on_anchor(self, url: QUrl) -> None:
        if url.scheme() == "guide":
            self.show_entry(url.path() or url.toString().split(":", 1)[-1])
        elif url.scheme() in ("http", "https"):
            QDesktopServices.openUrl(url)

    def _open_first(self) -> None:
        entries = self._visible_entries()
        if entries:
            self.show_entry(entries[0].key)

    # ------------------------------------------------------------------ search
    def apply_filter(self, text: str = "") -> None:
        """Show the entries matching ``text`` (all when empty) in the list and as an index on the right."""
        text = (text or "").strip()
        self._filter({e.key for e in guide.search(text)} if text else None)
        visible = self._visible_entries()
        n = len(guide.ENTRIES)
        self.count_label.setText(f"{len(visible)} of {n}" if text else f"{n} entries")
        if text and len(visible) == 1:
            self.show_entry(visible[0].key)
        else:
            heading = f"Matching “{text}”" if text else TITLE
            self.current_key = ""
            self.browser.setHtml(guide.html_index(visible, heading))

    def _filter(self, keys: set[str] | None) -> None:
        """Hide every entry not in ``keys`` (None shows all), and the families left empty."""
        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            shown = 0
            for j in range(top.childCount()):
                child = top.child(j)
                hide = keys is not None and child.data(0, KEY_ROLE) not in keys
                child.setHidden(hide)
                shown += not hide
            top.setHidden(shown == 0)

    def _visible_entries(self) -> list[guide.Entry]:
        return [guide.ENTRIES[k] for k, it in self.items.items() if not it.isHidden()]

    def visible_keys(self) -> list[str]:
        return [e.key for e in self._visible_entries()]


def show(win, key: str = "") -> MethodsGuideWindow:
    """Open (or raise) ``win``'s methods guide at entry ``key``."""
    w = getattr(win, "methods_guide", None)
    if w is None:
        w = MethodsGuideWindow(win)
        win.methods_guide = w
    w.show()
    w.raise_()
    w.activateWindow()
    if key:
        w.show_entry(key)
    return w
