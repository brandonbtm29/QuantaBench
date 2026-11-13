"""Playback bar under the 2D/3D view (UI-1): steps through the selected molecule's conformers, or through the
molecules of the list (UI-2: the Conformers / Molecules box at its left).

First / previous / play-pause / next / last, a slider and a label ("Conformer 3 of 20, ΔE 1.2 kJ/mol"). The bar holds
only an index and a count: the main window tells it how many items there are (``set_count``) and which one is shown
(``set_index``), and moves its views when the bar's ``indexChanged`` fires. So the same bar can later step through
trajectory frames or molecules (UI-2) by giving it another count and label.

Play advances at the chosen rate and wraps from the last item to the first; it stops when the count changes (another
molecule) or when the window asks it to (``stop``). In the Molecules mode the window gives it the list's length and
the selected row, and selects the row the bar moves to (``modeChanged`` tells it the mode changed).
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QSlider, QStyle, QToolButton, QWidget

#: playback rates (items per second) offered in the rate box
RATES = (1, 2, 4, 8)
DEFAULT_RATE = 2
#: what the bar steps through: (kind, the mode box's text)
MODES = (("Conformer", "Conformers"), ("Molecule", "Molecules"))


class PlaybackBar(QWidget):
    #: the index the user (or play) moved to; not emitted for ``set_index(..., emit=False)``
    indexChanged = Signal(int)
    #: the user chose what to step through: "Conformer" or "Molecule"
    modeChanged = Signal(str)

    def __init__(self, parent=None, kind: str = "Conformer"):
        super().__init__(parent)
        self.setObjectName("playback")
        self.kind = kind
        self._count = 0
        self._index = -1
        self._detail: Callable[[int], str] | None = None
        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 2, 6, 2)
        lay.setSpacing(2)
        st = self.style()
        self.mode_box = QComboBox(self)
        self.mode_box.setObjectName("playback_mode")
        for k, text in MODES:
            self.mode_box.addItem(text, k)
        self.mode_box.setCurrentIndex(max(0, self.mode_box.findData(kind)))
        self.mode_box.setToolTip("Step through the selected molecule's conformers, or through the molecules of the "
                                 "list (in its order and filter)")
        self.mode_box.currentIndexChanged.connect(self._on_mode)
        lay.addWidget(self.mode_box)

        def button(icon, tip, slot, name):
            b = QToolButton(self)
            b.setIcon(st.standardIcon(icon))
            b.setToolTip(tip)
            b.setAutoRaise(True)
            b.setObjectName(name)
            b.clicked.connect(slot)
            lay.addWidget(b)
            return b

        P = QStyle.StandardPixmap
        self.first_btn = button(P.SP_MediaSkipBackward, "First", self.first, "playback_first")
        self.prev_btn = button(P.SP_MediaSeekBackward, "Previous", lambda: self.step(-1), "playback_prev")
        self.play_btn = button(P.SP_MediaPlay, "Play / pause", self.toggle_play, "playback_play")
        self.next_btn = button(P.SP_MediaSeekForward, "Next", lambda: self.step(1), "playback_next")
        self.last_btn = button(P.SP_MediaSkipForward, "Last", self.last, "playback_last")
        self.slider = QSlider(Qt.Orientation.Horizontal, self)
        self.slider.setMinimum(0)
        self.slider.setPageStep(1)
        self.slider.setMinimumWidth(120)
        self.slider.valueChanged.connect(self._on_slider)
        lay.addWidget(self.slider, 1)
        self.label = QLabel(self)
        self.label.setMinimumWidth(240)
        self.label.setObjectName("playback_label")
        lay.addWidget(self.label)
        self.rate_box = QComboBox(self)
        for r in RATES:
            self.rate_box.addItem(f"{r}/s", r)
        self.rate_box.setCurrentIndex(RATES.index(DEFAULT_RATE))
        self.rate_box.setToolTip(f"{kind}s per second while playing")
        self.rate_box.currentIndexChanged.connect(self._on_rate)
        lay.addWidget(self.rate_box)
        self.timer = QTimer(self)
        self.timer.setInterval(int(1000 / DEFAULT_RATE))
        self.timer.timeout.connect(self._tick)
        self._refresh()

    # ------------------------------------------------------------------ state
    def count(self) -> int:
        return self._count

    def index(self) -> int:
        return self._index

    def is_playing(self) -> bool:
        return self.timer.isActive()

    def set_count(self, n: int, detail: Callable[[int], str] | None = None) -> None:
        """``n`` items (0: nothing to step through); ``detail(i)`` adds to the label (e.g. the conformer's ΔE). Stops
        playing and goes to the first item without emitting."""
        self.stop()
        self._count = max(0, int(n))
        self._detail = detail
        self._index = 0 if self._count else -1
        self.slider.blockSignals(True)
        self.slider.setMaximum(max(0, self._count - 1))
        self.slider.setValue(max(0, self._index))
        self.slider.blockSignals(False)
        self._refresh()

    def set_index(self, i: int, emit: bool = False) -> None:
        """Show item ``i`` as current (clamped); ``emit`` also fires ``indexChanged``."""
        if not self._count:
            return
        i = min(max(0, int(i)), self._count - 1)
        changed = i != self._index
        self._index = i
        self.slider.blockSignals(True)
        self.slider.setValue(i)
        self.slider.blockSignals(False)
        self._refresh()
        if emit and changed:
            self.indexChanged.emit(i)

    def refresh_label(self) -> None:
        """Rebuild the label (the detail text changed, e.g. Settings > Display's units)."""
        self._refresh()

    # ------------------------------------------------------------------ actions
    def step(self, d: int) -> None:
        self.set_index(self._index + d, emit=True)

    def first(self) -> None:
        self.set_index(0, emit=True)

    def last(self) -> None:
        self.set_index(self._count - 1, emit=True)

    def toggle_play(self) -> None:
        if self.timer.isActive():
            self.stop()
        elif self._count > 1:
            self.timer.start()
            self._refresh()

    def stop(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
        self._refresh()

    def _tick(self) -> None:
        if self._count < 2:
            self.stop()
            return
        self.set_index((self._index + 1) % self._count, emit=True)      # wraps to the first

    def set_mode(self, kind: str) -> None:
        """Step through "Conformer"s or "Molecule"s (emits ``modeChanged`` when it changes)."""
        i = self.mode_box.findData(kind)
        if i >= 0 and i != self.mode_box.currentIndex():
            self.mode_box.setCurrentIndex(i)

    def _on_mode(self, _i: int) -> None:
        kind = self.mode_box.currentData() or "Conformer"
        if kind == self.kind:
            return
        self.stop()
        self.kind = kind
        self.rate_box.setToolTip(f"{kind}s per second while playing")
        self.modeChanged.emit(kind)

    def _on_slider(self, value: int) -> None:
        self.set_index(value, emit=True)

    def _on_rate(self, _i: int) -> None:
        self.timer.setInterval(int(1000 / (self.rate_box.currentData() or DEFAULT_RATE)))

    def _refresh(self) -> None:
        n = self._count
        for b in (self.first_btn, self.prev_btn):
            b.setEnabled(n > 1 and self._index > 0)
        for b in (self.next_btn, self.last_btn):
            b.setEnabled(n > 1 and self._index < n - 1)
        self.play_btn.setEnabled(n > 1)
        self.slider.setEnabled(n > 1)
        self.rate_box.setEnabled(n > 1)
        P = QStyle.StandardPixmap
        self.play_btn.setIcon(self.style().standardIcon(P.SP_MediaPause if self.timer.isActive() else P.SP_MediaPlay))
        if not n:
            self.label.setText(f"No {self.kind.lower()}s")
            return
        text = f"{self.kind} {self._index + 1} of {n}"
        if self._detail is not None:
            try:
                extra = self._detail(self._index)
            except Exception:       # a stale callback (the molecule changed): the plain label
                extra = ""
            if extra:
                text += f", {extra}"
        self.label.setText(text)
