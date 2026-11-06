"""An energy input shown in the display unit (Settings > Display) and stored in a fixed one.

The conformer energy window is a science setting stored in kcal/mol (``BuildDefaults.energy_window_kcal``, in
``jobs.SCIENCE_FIELDS``); the Surfaces panel's ESP colour range is in kcal/mol like the mesh's ``esp_kcal``. Their
spin boxes show kJ/mol (or eV, hartree) when the user wants it, and give back exactly the stored value as long as
the user has not changed the number: 6.0 kcal/mol shown as 25.10 kJ/mol is still 6.0, not 5.9990, so opening and
closing a dialog never makes a resumed run look different.
"""

from __future__ import annotations

from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import QDoubleSpinBox

from .. import units

#: decimals and step of an energy input in each unit (kcal/mol: as the inputs were before units)
INPUT_DECIMALS = {units.KJ: 2, units.KCAL: 2, units.EV: 4, units.HARTREE: 6}
INPUT_STEP = {units.KJ: 1.0, units.KCAL: 1.0, units.EV: 0.05, units.HARTREE: 0.001}


class EnergySpinBox(QDoubleSpinBox):
    """``set_stored(v)`` / ``stored()`` in ``stored_unit``; the box shows ``unit`` (``set_unit``)."""

    def __init__(self, minimum: float, maximum: float, stored_unit: str = units.KCAL, unit: str = units.DEFAULT_UNIT,
                 parent=None):
        super().__init__(parent)
        self.stored_unit = units.normalize(stored_unit)
        self._lo, self._hi = float(minimum), float(maximum)      # in stored_unit
        self._stored: float | None = None    # the value given to set_stored (kept while the shown number is its own)
        self._shown: float | None = None     # the number shown for it
        self._unit = ""
        self.set_unit(unit)

    def unit(self) -> str:
        return self._unit

    def set_unit(self, unit: str) -> None:
        """Show the value in ``unit``; the stored value does not change."""
        unit = units.normalize(unit)
        if unit == self._unit:
            return
        keep = self.stored() if self._unit else None
        with QSignalBlocker(self):
            self._unit = unit
            self.setDecimals(INPUT_DECIMALS[unit])
            self.setSingleStep(INPUT_STEP[unit])
            self.setRange(self._to_shown(self._lo), self._to_shown(self._hi))
            self.setSuffix(f" {unit}")
        self.set_stored(self._lo if keep is None else keep)

    def _to_shown(self, v: float) -> float:
        return units.convert(v, self.stored_unit, self._unit)

    def set_stored(self, value: float) -> None:
        """Show ``value`` (in ``stored_unit``, clamped to the range); ``stored()`` returns it until the number is
        edited."""
        value = min(max(float(value), self._lo), self._hi)
        self._stored = value
        self._shown = None                   # while setValue's signal runs, stored() must not trust _shown
        shown = self._to_shown(value)
        with QSignalBlocker(self):
            self.setValue(shown)
        self._shown = self.value()
        self.valueChanged.emit(self.value())

    def stored(self) -> float:
        """The value in ``stored_unit``: the one given to ``set_stored`` while the box still shows it, else the
        shown number converted."""
        if self._stored is not None and self._shown is not None and self.value() == self._shown:
            return self._stored
        return units.convert(self.value(), self._unit, self.stored_unit)
