"""Collapsible sections and info pop-outs (UI-8): the pieces that keep the docks short.

    Section      a header (an arrow, the title, a gray one-line summary, a status at the right, an optional ⓘ) over a
                 body that folds away; ``toggled`` when it opens or closes
    Accordion    sections in a column where opening one closes the others (the Pipeline dock's stages); the open one
                 is remembered in the window settings
    InfoLabel    a rich-text label that says when its text changes (the method guide's labels are InfoLabels)
    InfoButton   an ⓘ that shows a label's text in a pop-out (links still work) instead of the text taking room in a
                 form; it turns into an orange ⚠ while the text says something is not available
    with_info    a form field and its ⓘ in one row
"""

from __future__ import annotations

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

#: words in an info text that turn its ⓘ into a ⚠
WARN_WORDS = ("Not available", "not found", "not installed", "needs ")
INFO_STYLE = "QToolButton { border: none; color: palette(link); font-weight: bold; padding: 0 3px; }"
WARN_STYLE = "QToolButton { border: none; color: #d35400; font-weight: bold; padding: 0 3px; }"
POPUP_WIDTH = 380


def _setting(key: str):
    try:
        from .main_window import _window_settings

        return _window_settings().value(key)
    except Exception:  # noqa: BLE001 - a preference, never worth an error
        return None


def _set_setting(key: str, value) -> None:
    try:
        from .main_window import _window_settings

        _window_settings().setValue(key, value)
    except Exception:  # noqa: BLE001, S110
        pass


def _plain(text: str) -> str:
    from PySide6.QtGui import QTextDocument

    doc = QTextDocument()
    doc.setHtml(text)
    return doc.toPlainText()


class InfoLabel(QLabel):
    """A QLabel that emits ``textSet`` whenever ``setText`` runs (QLabel itself has no such signal)."""

    textSet = Signal(str)

    def setText(self, text: str) -> None:  # noqa: N802 - Qt's name
        super().setText(text)
        self.textSet.emit(text)


class InfoPopup(QFrame):
    """The pop-out of an InfoButton: the text, word-wrapped, its links forwarded to the source label."""

    def __init__(self, text: str, source: QLabel | None = None, parent=None):
        super().__init__(parent, Qt.WindowType.Popup)
        self.setObjectName("info_popup")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        self.label = QLabel(text)
        self.label.setWordWrap(True)
        self.label.setTextFormat(Qt.TextFormat.RichText)
        self.label.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        self.label.setOpenExternalLinks(False)
        self.label.setFixedWidth(POPUP_WIDTH)
        if source is not None:
            self.label.linkActivated.connect(lambda href: (self.close(), source.linkActivated.emit(href)))
        lay.addWidget(self.label)


class InfoButton(QToolButton):
    """ⓘ: shows ``source``'s text (a label kept hidden) or a fixed ``text`` in a pop-out; hovering shows it as a
    tooltip too. With a source that says something is not available it is an orange ⚠."""

    def __init__(self, source: QLabel | None = None, text: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("info_button")
        self.source = source
        self.text_ = text
        self.popup: InfoPopup | None = None
        self.setAutoRaise(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self.clicked.connect(self.show_popup)
        if isinstance(source, InfoLabel):
            source.textSet.connect(lambda _t: self.sync())
        self.sync()

    def info_text(self) -> str:
        return self.source.text() if self.source is not None else self.text_

    def warns(self) -> bool:
        plain = _plain(self.info_text())
        return any(w in plain for w in WARN_WORDS)

    def sync(self) -> None:
        text = self.info_text()
        warn = self.warns()
        self.setText("⚠" if warn else "ⓘ")
        self.setStyleSheet(WARN_STYLE if warn else INFO_STYLE)
        self.setToolTip(f"<div style='max-width: {POPUP_WIDTH}px'>{text}</div>" if text else "")
        self.setEnabled(bool(text))

    def show_popup(self) -> InfoPopup:
        self.popup = InfoPopup(self.info_text(), self.source, self)
        self.popup.adjustSize()
        pos = self.mapToGlobal(QPoint(0, self.height()))
        width = self.popup.width()
        # inside the window where it fits (an ⓘ sits at a dock's right edge): right-aligned to the button
        right = self.mapToGlobal(QPoint(self.width(), 0)).x()
        bounds = self.window().frameGeometry()
        screen = self.screen().availableGeometry() if self.screen() is not None else None
        if screen is not None:
            bounds = bounds.intersected(screen) if bounds.intersects(screen) else screen
        if pos.x() + width > bounds.right():
            pos.setX(max(bounds.left(), right - width))
        self.popup.move(pos)
        self.popup.show()
        return self.popup


def with_info(field: QWidget, source: QLabel) -> QWidget:
    """``field`` with an ⓘ for ``source`` after it, in one row; ``source`` stays out of sight (parented to the row,
    hidden), its text read by the pop-out."""
    row = QWidget()
    lay = QHBoxLayout(row)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(2)
    lay.addWidget(field, 1)
    btn = InfoButton(source, parent=row)
    lay.addWidget(btn)
    source.setParent(row)
    source.hide()
    row.info_button = btn
    return row


class Section(QWidget):
    """A collapsible section (module docstring)."""

    toggled = Signal(bool)

    def __init__(self, title: str, body: QWidget, key: str = "", info: str = "", summary: str = "",
                 open_: bool = False, settings_key: str = "", parent=None):
        super().__init__(parent)
        self.key = key
        self.settings_key = settings_key       # remembers open/closed in the window settings (an Accordion keeps
        if settings_key:                       # its open section itself)
            saved = _setting(settings_key)
            if saved is not None:
                open_ = str(saved).lower() in ("true", "1")
        self.setObjectName(f"section_{key}" if key else "section")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        head = QWidget()
        head.setObjectName("section_head")
        hl = QHBoxLayout(head)
        hl.setContentsMargins(0, 1, 2, 1)
        hl.setSpacing(4)
        self.toggle = QToolButton()
        self._title = title
        self.toggle.setText(title.replace("&", "&&"))      # a single & would be a mnemonic
        self.toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle.setAutoRaise(True)
        self.toggle.setCheckable(True)
        self.toggle.setStyleSheet("QToolButton { font-weight: bold; }")
        self.toggle.toggled.connect(self.set_open)
        self.summary = QLabel(summary)
        self.summary.setStyleSheet("color: gray;")
        self.summary.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.summary.setTextFormat(Qt.TextFormat.PlainText)
        self.status = QLabel()
        self.status.setStyleSheet("color: gray;")
        hl.addWidget(self.toggle)
        hl.addWidget(self.summary, 1)
        hl.addWidget(self.status)
        self.info_button = InfoButton(text=info, parent=head) if info else None
        if self.info_button is not None:
            hl.addWidget(self.info_button)
        lay.addWidget(head)
        self.body = body
        body.setParent(self)
        lay.addWidget(body)
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: palette(mid);")
        lay.addWidget(line)
        self._open = None
        self.set_open(open_)

    def is_open(self) -> bool:
        return bool(self._open)

    def set_open(self, on: bool) -> None:
        on = bool(on)
        if on == self._open:
            return
        self._open = on
        self.toggle.blockSignals(True)
        self.toggle.setChecked(on)
        self.toggle.blockSignals(False)
        self.toggle.setArrowType(Qt.ArrowType.DownArrow if on else Qt.ArrowType.RightArrow)
        self.body.setVisible(on)
        self.summary.setVisible(not on)
        if self.settings_key:
            _set_setting(self.settings_key, on)
        self.toggled.emit(on)

    def set_summary(self, text: str) -> None:
        self.summary.setText(text)
        self.summary.setToolTip(text)

    def set_status(self, text: str) -> None:
        self.status.setText(text)

    def title(self) -> str:
        return self._title


class Accordion(QWidget):
    """Sections in a column, at most one open (``exclusive``); ``settings_key`` remembers the open one."""

    opened = Signal(str)

    def __init__(self, exclusive: bool = True, settings_key: str = "", parent=None):
        super().__init__(parent)
        self.exclusive = exclusive
        self.settings_key = settings_key
        self.sections: dict[str, Section] = {}
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(0)
        self._lay.addStretch(1)

    def add(self, section: Section) -> Section:
        self.sections[section.key] = section
        self._lay.insertWidget(self._lay.count() - 1, section)
        section.toggled.connect(lambda on, k=section.key: self._on_toggled(k, on))
        return section

    def _on_toggled(self, key: str, on: bool) -> None:
        if not on:
            return
        if self.exclusive:
            for k, s in self.sections.items():
                if k != key and s.is_open():
                    s.set_open(False)
        if self.settings_key:
            _set_setting(self.settings_key, key)
        self.opened.emit(key)

    def open(self, key: str) -> None:
        if key in self.sections:
            self.sections[key].set_open(True)

    def open_key(self) -> str:
        return next((k for k, s in self.sections.items() if s.is_open()), "")

    def restore(self, default: str) -> None:
        key = str(_setting(self.settings_key) or default) if self.settings_key else default
        self.open(key if key in self.sections else default)


class Notes(QWidget):
    """A folded list of notes: one line "⚠ 2 notes" that opens to show them (the Molecules dock's engine and
    fallback notes, which can run to ten lines each)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("notes")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.toggle = QToolButton()
        self.toggle.setObjectName("notes_toggle")
        self.toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle.setAutoRaise(True)
        self.toggle.setCheckable(True)
        self.toggle.setStyleSheet("QToolButton { color: #d35400; }")
        self.toggle.toggled.connect(self._show)
        self.label = QLabel()
        self.label.setWordWrap(True)
        self.label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.label.setStyleSheet("color: gray;")
        lay.addWidget(self.toggle)
        lay.addWidget(self.label)
        self.notes: list[str] = []
        self.set_notes([])

    def set_notes(self, notes) -> None:
        self.notes = [str(n) for n in notes if str(n).strip()]
        n = len(self.notes)
        self.toggle.setText(f"⚠ {n} note{'s' if n != 1 else ''}")
        self.toggle.setToolTip(_plain("<br>".join(self.notes)) if n else "")
        self.label.setText("<br>".join(f"• {x}" for x in self.notes))
        self.setVisible(bool(n))
        self._show(self.toggle.isChecked())

    def _show(self, on: bool) -> None:
        self.toggle.setArrowType(Qt.ArrowType.DownArrow if on else Qt.ArrowType.RightArrow)
        self.label.setVisible(bool(on and self.notes))
