"""Energy units and number format for the GUI (Spartan's Preferences > Document; ``Settings.display``).

Display only. Computed values keep the units they are stored in (``rel_kcal``, ``energy_ev``, ``esp_*_kcal``,
``zpe_kcal``, ...); exported CSV/JSON/SDF columns keep their names and units (``_kcal``, ``_eV``), and nothing
here is a science setting (``jobs.SCIENCE_FIELDS``). An input that feeds a science setting (the conformer
energy window, stored in kcal/mol) shows the display unit and converts on read and write
(``gui.energy_widgets.EnergySpinBox``).

    convert(value, src, dst)          value in ``src`` units -> ``dst`` units
    resolution_decimals(d, src, dst)  places in ``dst`` that keep the resolution of ``d`` places in ``src``
    format_number(x, decimals, digits, style)
    EnergyDisplay(settings.display)   what the GUI asks: units per kind, numbers as text, headers

Two kinds of energy, as Spartan's "Energy" and "Relative energy" units:

- ``"absolute"``: total energies (a conformer's E, the best energy in the molecule list, the QM energy, H and G);
- ``"relative"``: differences and energies of a probe charge (ΔE of conformers, the energy window, ZPE, H − E,
  G − E, the electrostatic potential on surfaces).

Conversion factors are the ones the computations use: 1 eV = 23.060548 kcal/mol (``chem.conformers``,
``chem.vibrations``), 1 hartree = 27.211386245988 eV (``chem.engines``), 1 kcal = 4.184 kJ (thermochemical
calorie, exact).
"""

from __future__ import annotations

import math

KCAL_PER_EV = 23.060548              # chem.conformers.KCAL_PER_EV
HARTREE_EV = 27.211386245988         # chem.engines.HARTREE_EV
KJ_PER_KCAL = 4.184                  # thermochemical calorie

KJ, KCAL, EV, HARTREE = "kJ/mol", "kcal/mol", "eV", "hartree"
UNITS = (KJ, KCAL, EV, HARTREE)
DEFAULT_UNIT = KJ
#: one eV in each unit
_PER_EV = {EV: 1.0, KCAL: KCAL_PER_EV, KJ: KCAL_PER_EV * KJ_PER_KCAL, HARTREE: 1.0 / HARTREE_EV}
#: names in the Settings dialog
LABELS = {KJ: "kJ/mol", KCAL: "kcal/mol", EV: "eV", HARTREE: "hartree (Eh)"}
_ALIASES = {"kj/mol": KJ, "kj": KJ, "kjmol": KJ, "kcal/mol": KCAL, "kcal": KCAL, "kcalmol": KCAL, "ev": EV,
            "hartree": HARTREE, "eh": HARTREE, "au": HARTREE, "a.u.": HARTREE, "ha": HARTREE}
#: entropy, in the relative unit per 1000 K (J/(mol·K) with kJ/mol)
ENTROPY_LABELS = {KJ: "J/(mol·K)", KCAL: "cal/(mol·K)", EV: "meV/K", HARTREE: "mEh/K"}

KINDS = ("absolute", "relative")
STYLES = ("auto", "decimal", "scientific")
STYLE_LABELS = {"auto": "Automatic", "decimal": "Decimal", "scientific": "Scientific"}
MAX_DIGITS = 15


def normalize(unit: str | None, default: str = DEFAULT_UNIT) -> str:
    """The canonical name of ``unit`` (case and spelling as in a hand-edited settings.json: "kj", "Eh", "au"), or
    ``default`` for anything else."""
    text = str(unit or "").strip()
    if text in _PER_EV:
        return text
    return _ALIASES.get(text.lower().replace(" ", ""), default)


def normalize_style(style: str | None) -> str:
    text = str(style or "").strip().lower()
    return text if text in STYLES else "auto"


def normalize_digits(digits) -> int:
    """0 (automatic) or 1-15 significant digits; anything else is 0."""
    try:
        n = int(digits)
    except (TypeError, ValueError):
        return 0
    return n if 0 < n <= MAX_DIGITS else 0


def factor(src: str, dst: str) -> float:
    """Multiply a value in ``src`` by this to get it in ``dst``."""
    return _PER_EV[normalize(dst)] / _PER_EV[normalize(src)]


def convert(value: float, src: str, dst: str, power: int = 1) -> float:
    """``value`` in ``src`` (raised to ``power``: 2 for a variance such as σ² in (kcal/mol)²) in ``dst``."""
    if normalize(src) == normalize(dst):
        return float(value)
    return float(value) * factor(src, dst) ** power


def resolution_decimals(decimals: int, src: str, dst: str, power: int = 1) -> int:
    """Decimal places in ``dst`` that show a value at least as finely as ``decimals`` places in ``src``: 0.01
    kcal/mol is 0.042 kJ/mol (2 places), 0.00043 eV (4) or 0.0000159 hartree (5)."""
    f = factor(src, dst) ** power
    return max(0, int(decimals) - math.floor(math.log10(f) + 1e-9))


def _significant(x: float, digits: int, style: str) -> str:
    if style == "decimal":
        if x == 0:
            return f"{0.0:.{digits - 1}f}"
        places = digits - 1 - math.floor(math.log10(abs(x)))
        if places > 0:
            return f"{x:.{places}f}"
        return f"{round(x, places):.0f}"
    text = f"{x:#.{digits}g}"                 # keep trailing zeros: a column of values lines up
    if "e" in text:
        mant, exp = text.split("e")
        return mant.rstrip(".") + "e" + exp
    return text.rstrip(".")


def format_number(x: float | None, decimals: int = 2, digits: int = 0, style: str = "auto") -> str:
    """``x`` as text.

    ``digits`` 0 (automatic) shows ``decimals`` places; 1-15 shows that many significant digits instead.
    ``style``: ``"decimal"`` never uses an exponent; ``"scientific"`` always does (``-2.62550e+06``);
    ``"auto"`` uses one only where the decimal form would hide the value (a nonzero value that would read as
    0.00) or, with significant digits, as the ``g`` format does (exponent below -4 or at least ``digits``)."""
    if x is None:
        return ""
    x = float(x)
    if not math.isfinite(x):
        return str(x)
    style = normalize_style(style)
    digits = normalize_digits(digits)
    decimals = max(0, int(decimals))
    if style == "scientific":
        if digits == 0:          # as many significant digits as the decimal form would show
            digits = max(1, min(MAX_DIGITS, (math.floor(math.log10(abs(x))) + 1 if x else 1) + decimals))
        return f"{x:.{digits - 1}e}"
    if digits:
        return _significant(x, digits, style)
    text = f"{x:.{decimals}f}"
    if style == "auto" and x != 0 and float(text) == 0:
        return f"{x:.2e}"
    return text


class EnergyDisplay:
    """The GUI's view of ``Settings.display``: units per kind and energies as text. A missing or hand-edited
    value falls back to the default (kJ/mol, automatic digits and style)."""

    def __init__(self, display=None):
        self.absolute = normalize(getattr(display, "energy_unit", DEFAULT_UNIT))
        self.relative = normalize(getattr(display, "relative_energy_unit", DEFAULT_UNIT))
        self.digits = normalize_digits(getattr(display, "significant_digits", 0))
        self.style = normalize_style(getattr(display, "number_style", "auto"))

    @classmethod
    def of(cls, obj) -> EnergyDisplay:
        """From a ``Settings``, its ``display``, an ``EnergyDisplay`` or None (defaults)."""
        if isinstance(obj, EnergyDisplay):
            return obj
        return cls(getattr(obj, "display", obj))

    def __eq__(self, other) -> bool:
        return isinstance(other, EnergyDisplay) and self.key() == other.key()

    def __hash__(self) -> int:
        return hash(self.key())

    def key(self) -> tuple:
        return self.absolute, self.relative, self.digits, self.style

    def __repr__(self) -> str:
        return (f"EnergyDisplay(absolute={self.absolute!r}, relative={self.relative!r}, digits={self.digits}, "
                f"style={self.style!r})")

    # ---------------------------------------------------------------- units
    def unit(self, kind: str = "relative") -> str:
        if kind not in KINDS:
            raise ValueError(f"energy kind {kind!r}: one of {', '.join(KINDS)}")
        return self.absolute if kind == "absolute" else self.relative

    def label(self, kind: str = "relative", power: int = 1) -> str:
        """The unit as written after a number or in a header: "kJ/mol", "(kJ/mol)²", "eV²"."""
        u = self.unit(kind)
        if power == 1:
            return u
        return (f"({u})" if "/" in u else u) + {2: "²", 3: "³"}.get(power, f"^{power}")

    def header(self, name: str, kind: str = "relative") -> str:
        """A column header: ``header("ΔE")`` -> "ΔE (kJ/mol)"."""
        return f"{name} ({self.label(kind)})"

    def entropy_label(self) -> str:
        return ENTROPY_LABELS[self.relative]

    # ---------------------------------------------------------------- values
    def value(self, x: float, src: str, kind: str = "relative", power: int = 1) -> float:
        return convert(x, src, self.unit(kind), power)

    def text(self, x: float | None, src: str, kind: str = "relative", decimals: int = 2, power: int = 1) -> str:
        """``x`` (in ``src``) in this kind's unit, as a number only. ``decimals`` is the precision QuantaBench
        shows the value with in ``src``; it is kept (``resolution_decimals``) unless significant digits are set."""
        if x is None:
            return ""
        dst = self.unit(kind)
        places = resolution_decimals(decimals, src, dst, power)
        return format_number(convert(x, src, dst, power), places, self.digits, self.style)

    def with_unit(self, x: float | None, src: str, kind: str = "relative", decimals: int = 2, power: int = 1) -> str:
        """``text`` and the unit: "4.18 kJ/mol"."""
        if x is None:
            return ""
        return f"{self.text(x, src, kind, decimals, power)} {self.label(kind, power)}"

    def entropy_text(self, s_cal_mol_k: float, decimals: int = 1) -> str:
        """An entropy in cal/(mol·K) as "188.8 J/(mol·K)" (or cal/(mol·K), meV/K, mEh/K, after the relative unit)."""
        dst = self.relative
        places = resolution_decimals(decimals, KCAL, dst)
        value = convert(s_cal_mol_k, KCAL, dst)        # cal -> J, meV, mEh: the same factor as kcal -> kJ, eV, Eh
        return f"{format_number(value, places, self.digits, self.style)} {self.entropy_label()}"

