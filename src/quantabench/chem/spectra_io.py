"""Readers for experimental (IR) spectra: CSV/TSV/whitespace tables and JCAMP-DX 4.24 / 5.x.

    read_spectrum(path) -> Spectrum(x, y, x_units, y_units, title)

The result is always in wavenumbers (cm⁻¹, ascending) with absorbance-like ordinates: transmittance
(fraction or %T) becomes absorbance A = -log10(T), with T clipped at 1e-5 so zeros stay finite; micrometer and
nanometer abscissas become cm⁻¹ (10⁴/λ[µm], 10⁷/λ[nm]).

JCAMP-DX (McDonald & Wilks, Appl. Spectrosc. 1988, 42, 151; Lampen et al., Appl. Spectrosc. 1999, 53, 72A):

- ``##XYDATA=(X++(Y..Y))``: each line is an abscissa (in units of XFACTOR) followed by ordinates (times
  YFACTOR). Abscissas come from FIRSTX/LASTX/NPOINTS (DELTAX, or the line abscissas when NPOINTS disagrees).
- Ordinates may be AFFN (free-format numbers) or ASDF compressed: PAC (sign-separated), SQZ (``@A-I``/``a-i``
  replace the leading digit and sign), DIF (``%J-R``/``j-r``: differences from the previous ordinate), DUP
  (``S-Z``, ``s``: the previous value or difference occurs n times in total). In DIF form the last ordinate of
  a line is repeated as the first ordinate of the next line (the Y check value); it is verified and dropped.
- ``##PEAK TABLE=`` and ``##XYPOINTS=`` with ``(XY..XY)``, ``(XYXY)`` or ``(XYW..XYW)``: x,y pairs (sticks).
- Labels compare case-insensitively ignoring spaces, dashes, slashes and underscores; ``$$`` starts a comment.
- Compound files (``##BLOCKS=``, link blocks, nested ``##TITLE=``/``##END=``): the first block with spectral
  data is used.
"""

from __future__ import annotations

import csv
import io
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

T_FLOOR = 1e-5          # transmittance clip for -log10

JCAMP_SUFFIXES = (".jdx", ".dx", ".jcamp", ".jcm")


@dataclass
class Spectrum:
    """A spectrum on ascending wavenumbers."""

    x: np.ndarray
    y: np.ndarray
    x_units: str = "cm-1"
    y_units: str = ""
    title: str = ""
    peaks: bool = False                   # a peak table (sticks), not a continuous trace
    source_format: str = ""               # "csv" | "jcamp"
    original_units: tuple[str, str] = ("", "")
    warnings: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.x)


# ======================================================================================== units
def _norm_units(text: str) -> str:
    return re.sub(r"[\s_\-^()\[\]{}]", "", text or "").upper()


def convert_units(x, y, x_units: str, y_units: str, warnings: list[str] | None = None
                  ) -> tuple[np.ndarray, np.ndarray, str, str]:
    """Abscissa to cm⁻¹ and transmittance to absorbance; returns sorted (x, y, x_units, y_units)."""
    warn = warnings if warnings is not None else []
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xu, yu = _norm_units(x_units), _norm_units(y_units)
    if xu in ("MICROMETERS", "MICROMETER", "MICRONS", "MICRON", "UM", "ΜM", "µM"):
        with np.errstate(divide="ignore"):
            x = 1e4 / x
        x_units = "cm-1"
    elif xu in ("NANOMETERS", "NANOMETER", "NM"):
        with np.errstate(divide="ignore"):
            x = 1e7 / x
        x_units = "cm-1"
    elif xu in ("1/CM", "CM1", "CM-1", "WAVENUMBERS", "WAVENUMBER", "WAVENUMBERCM1", "WAVENUMBERS1/CM", ""):
        x_units = "cm-1"
    else:
        warn.append(f"x units {x_units!r} not recognized: taken as cm-1")
        x_units = "cm-1"
    if "TRANSMITTANCE" in yu or yu in ("%T", "T%", "PERCENTT", "%TRANSMISSION", "TRANSMISSION"):
        percent = "%" in yu or "PERCENT" in yu or (np.nanmax(y) > 1.5 if y.size else False)
        t = y / 100.0 if percent else y.copy()
        if np.any(t <= 0):
            warn.append(f"{int((t <= 0).sum())} transmittance value(s) <= 0 clipped to {T_FLOOR:g}")
        y = -np.log10(np.clip(t, T_FLOOR, None))
        y_units = "ABSORBANCE"
    ok = np.isfinite(x) & np.isfinite(y)
    if not ok.all():
        warn.append(f"{int((~ok).sum())} non-finite point(s) dropped")
    order = np.argsort(x[ok], kind="stable")
    return x[ok][order], y[ok][order], x_units, y_units


# ======================================================================================== tables
_NUM = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


def _is_num(s: str) -> bool:
    return bool(_NUM.match(s.strip()))


def _split(line: str, delim: str | None) -> list[str]:
    if delim is None:
        return line.split()
    return [f.strip() for f in next(csv.reader([line], delimiter=delim))]


def _units_from_header(header: list[str]) -> tuple[str, str]:
    xh = header[0].lower() if header else ""
    yh = header[1].lower() if len(header) > 1 else ""
    xu = "MICROMETERS" if re.search(r"µm|μm|\bum\b|micro", xh) else "NANOMETERS" if re.search(r"\bnm\b|nano", xh) \
        else "1/CM"
    yu = "%T" if re.search(r"%\s*t|percent", yh) else "TRANSMITTANCE" if ("transm" in yh or re.fullmatch(
        r"\s*t\s*(\(.*\))?\s*", yh)) else \
        "ABSORBANCE" if "absorb" in yh else (header[1] if len(header) > 1 else "")
    return xu, yu


def read_table(path: str | Path) -> Spectrum:
    """Two numeric columns (more are ignored), optional header lines, delimiter sniffed among tab, comma,
    semicolon and whitespace. Decimal commas ("1234,5;0,12", tab- or space-separated too) are understood.
    Units come from the header when it names them (cm-1, µm, nm, %T, transmittance, absorbance)."""
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith(("#", "//", "%"))]
    if not lines:
        raise ValueError(f"{path.name}: empty file")
    sample = lines[: min(len(lines), 50)]
    best = None
    # (delimiter, decimal comma); on a tie the earlier (plain decimal point) reading wins
    for delim, decimal_comma in (("\t", False), (",", False), (";", False), (None, False),
                                 (";", True), ("\t", True), (None, True)):
        rows = []
        for ln in sample:
            fields = _split(ln, delim)
            if decimal_comma:
                fields = [f.replace(",", ".") for f in fields]
            rows.append(fields)
        numeric = [r for r in rows if len(r) >= 2 and _is_num(r[0]) and _is_num(r[1])]
        score = len(numeric)
        if score and (best is None or score > best[0]):
            best = (score, delim, decimal_comma)
    if best is None:
        raise ValueError(f"{path.name}: no two numeric columns found")
    _score, delim, decimal_comma = best
    xs, ys, header = [], [], []
    for ln in lines:
        fields = _split(ln, delim)
        if decimal_comma:
            fields = [f.replace(",", ".") for f in fields]
        if len(fields) >= 2 and _is_num(fields[0]) and _is_num(fields[1]):
            xs.append(float(fields[0]))
            ys.append(float(fields[1]))
        elif not xs:
            header = fields
    if len(xs) < 2:
        raise ValueError(f"{path.name}: fewer than two data rows")
    warnings: list[str] = []
    xu, yu = _units_from_header(header)
    x, y, x_units, y_units = convert_units(xs, ys, xu, yu, warnings)
    return Spectrum(x, y, x_units, y_units, title=path.stem, source_format="csv",
                    original_units=(header[0] if header else "", header[1] if len(header) > 1 else ""),
                    warnings=warnings)


# ======================================================================================== JCAMP-DX
_SQZ = {"@": 0, **{c: i + 1 for i, c in enumerate("ABCDEFGHI")}, **{c: -(i + 1) for i, c in enumerate("abcdefghi")}}
_DIF = {"%": 0, **{c: i + 1 for i, c in enumerate("JKLMNOPQR")}, **{c: -(i + 1) for i, c in enumerate("jklmnopqr")}}
_DUP = {**{c: i + 1 for i, c in enumerate("STUVWXYZ")}, "s": 9}
_ASDF_ONLY = re.compile(r"[@A-DF-Za-df-z%?]")       # characters that never occur in AFFN (E/e can: exponents)
_AFFN_NUM = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
_X_AT_START = re.compile(r"^\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]\d+)?)")


def label_key(label: str) -> str:
    """JCAMP label comparison form: upper case, no spaces, dashes, slashes or underscores."""
    return re.sub(r"[\s\-/_]", "", label).upper()


def _asdf_tokens(text: str) -> list[tuple[str, float]]:
    """('abs' | 'dif' | 'dup', value) tokens of the ordinates of one ASDF line (after the abscissa)."""
    tokens: list[tuple[str, float]] = []
    kind: str | None = None
    buf = ""

    def flush():
        nonlocal kind, buf
        if kind is not None:
            if buf in ("", "+", "-"):
                raise ValueError(f"incomplete number in {text!r}")
            tokens.append((kind, float(buf)))
        kind, buf = None, ""

    for ch in text:
        if ch.isdigit() or ch == ".":
            if kind is None:            # a bare number (AFFN/PAC): absolute
                kind = "abs"
            buf += ch
        elif ch in _SQZ:
            flush()
            v = _SQZ[ch]
            kind, buf = "abs", ("-" if v < 0 else "") + str(abs(v))
        elif ch in _DIF:
            flush()
            v = _DIF[ch]
            kind, buf = "dif", ("-" if v < 0 else "") + str(abs(v))
        elif ch in _DUP:
            flush()
            kind, buf = "dup", str(_DUP[ch])
        elif ch in "+-":
            flush()
            kind, buf = "abs", ch
        elif ch == "?":                 # JCAMP 5: missing ordinate
            flush()
            tokens.append(("abs", math.nan))
        elif ch in " \t,;":
            flush()
        else:
            raise ValueError(f"unexpected character {ch!r} in compressed data {text!r}")
    flush()
    return tokens


@dataclass
class DataLine:
    """One decoded ``(X++(Y..Y))`` line: its abscissa (unscaled), ordinates (unscaled) and whether a DIF
    Y check value was dropped from its start (the abscissa then belongs to that check point)."""

    x: float
    values: list[float]
    checked: bool = False


def decode_xydata(lines: list[str], asdf: bool | None = None) -> tuple[list[DataLine], list[str]]:
    """Decode ``(X++(Y..Y))`` data lines into :class:`DataLine` objects plus warnings.

    ``asdf=None`` decides from the characters present: AFFN unless a character occurs that only compressed
    data uses (so that ``1.5E-3`` stays an exponent in AFFN files)."""
    if asdf is None:
        asdf = any(_ASDF_ONLY.search(ln.split("$$", 1)[0]) for ln in lines)
    out: list[DataLine] = []
    warnings: list[str] = []
    last: float | None = None
    last_delta: float | None = None
    ended_in_dif = False
    for raw in lines:
        line = raw.split("$$", 1)[0].strip()
        if not line:
            continue
        if not asdf:
            nums = [float(v) for v in _AFFN_NUM.findall(line)]
            if nums:
                out.append(DataLine(nums[0], nums[1:]))
            continue
        m = _X_AT_START.match(line)
        if not m:
            raise ValueError(f"data line without an abscissa: {raw!r}")
        x = float(m.group(1))
        tokens = _asdf_tokens(line[m.end():])
        values: list[float] = []
        check_pending = ended_in_dif
        checked = False
        prev_kind = None
        line_ended_in_dif = False
        for kind, v in tokens:
            if kind == "abs":
                if check_pending:
                    check_pending = False
                    checked = True
                    if last is not None and not (math.isnan(v) or math.isclose(v, last, rel_tol=0, abs_tol=1e-9)):
                        warnings.append(f"Y check failed at X={x:g}: {v:g} != {last:g}")
                    last = v
                    prev_kind = "check"
                    continue
                values.append(v)
                last, last_delta, prev_kind = v, None, "abs"
                line_ended_in_dif = False
            elif kind == "dif":
                if last is None:
                    raise ValueError(f"difference without a preceding value in {raw!r}")
                check_pending = False
                last = last + v
                last_delta = v
                values.append(last)
                prev_kind = "dif"
                line_ended_in_dif = True
            else:  # dup: the previous token occurs n times in total
                n = int(v)
                if prev_kind == "dif":
                    for _ in range(n - 1):
                        last = last + last_delta
                        values.append(last)
                    line_ended_in_dif = True
                elif prev_kind == "abs":
                    values.extend([last] * (n - 1))
                    line_ended_in_dif = False
                else:
                    raise ValueError(f"DUP count without a preceding value in {raw!r}")
        ended_in_dif = line_ended_in_dif
        out.append(DataLine(x, values, checked))
    return out, warnings


def decode_xy_pairs(lines: list[str], per_point: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """``(XY..XY)`` / ``(XYXY)`` / ``(XYW..XYW)`` data: AFFN numbers grouped per point (x, y[, w])."""
    nums: list[float] = []
    for raw in lines:
        line = raw.split("$$", 1)[0]
        nums += [float(v) for v in _AFFN_NUM.findall(line)]
    n = len(nums) // per_point
    arr = np.asarray(nums[: n * per_point], dtype=float).reshape(n, per_point)
    return arr[:, 0], arr[:, 1]


@dataclass
class _Block:
    labels: dict[str, str] = field(default_factory=dict)
    data_label: str = ""
    data_form: str = ""
    data_lines: list[str] = field(default_factory=list)


def _jcamp_blocks(text: str) -> list[_Block]:
    blocks: list[_Block] = []
    stack: list[_Block] = []
    current_data: _Block | None = None
    for raw in text.splitlines():
        stripped = raw.strip()
        if stripped.startswith("##"):
            current_data = None
            label, _, value = stripped[2:].partition("=")
            key = label_key(label)
            value = value.split("$$", 1)[0].strip()
            if key == "TITLE":
                blk = _Block()
                stack.append(blk)
                blocks.append(blk)
            if not stack:                      # a label before any ##TITLE: start an implicit block
                blk = _Block()
                stack.append(blk)
                blocks.append(blk)
            blk = stack[-1]
            if key == "END":
                stack.pop()
                continue
            if key in ("XYDATA", "PEAKTABLE", "XYPOINTS") and not blk.data_label:
                blk.data_label, blk.data_form = key, value.replace(" ", "").upper()
                current_data = blk
                continue
            blk.labels.setdefault(key, value)
        elif current_data is not None:
            current_data.data_lines.append(raw)
    return blocks


def _num_label(labels: dict, key: str, default: float | None = None) -> float | None:
    v = labels.get(key)
    if v is None or v == "":
        return default
    m = _AFFN_NUM.search(v)
    return float(m.group(0)) if m else default


def read_jcamp(path: str | Path) -> Spectrum:
    """The first spectrum of a JCAMP-DX file (see the module docstring)."""
    path = Path(path)
    text = path.read_text(encoding="latin-1")
    all_blocks = _jcamp_blocks(text)
    blocks = [b for b in all_blocks if b.data_label and b.data_lines]
    if not blocks:
        if any("NTUPLES" in b.labels for b in all_blocks):
            raise ValueError(f"{path.name}: JCAMP-DX NTUPLES files (##DATA TABLE) are not supported; "
                             "export the spectrum as ##XYDATA=(X++(Y..Y)) or CSV")
        raise ValueError(f"{path.name}: no ##XYDATA, ##PEAK TABLE or ##XYPOINTS found")
    blk = blocks[0]
    lab = blk.labels
    warnings: list[str] = []
    xfac = _num_label(lab, "XFACTOR", 1.0)
    yfac = _num_label(lab, "YFACTOR", 1.0)
    form = blk.data_form
    peaks = blk.data_label == "PEAKTABLE"
    if "++" in form:                                        # (X++(Y..Y)) and (X++(R..R)), (X++(I..I))
        data, w = decode_xydata(blk.data_lines)
        warnings += w
        y = np.array([v for d in data for v in d.values], dtype=float) * yfac
        n = len(y)
        first, last = _num_label(lab, "FIRSTX"), _num_label(lab, "LASTX")
        npoints = _num_label(lab, "NPOINTS")
        if npoints is not None and int(npoints) != n:
            warnings.append(f"NPOINTS={int(npoints)} but {n} ordinates decoded")
        if first is not None and last is not None and n > 1 and (npoints is None or int(npoints) == n):
            x = np.linspace(first, last, n)
        else:                                               # from the line abscissas and DELTAX
            delta = _num_label(lab, "DELTAX")
            if delta is None and first is not None and last is not None and npoints and npoints > 1:
                delta = (last - first) / (npoints - 1)
            if delta is None and len(data) > 1 and data[0].values:
                # line 2's abscissa is that of its first point, or of the Y-check point (= line 1's last
                # point) when it starts with a DIF check value
                steps = len(data[0].values) - (1 if data[1].checked else 0)
                if steps > 0:
                    delta = (data[1].x - data[0].x) * xfac / steps
            if delta is None:
                raise ValueError(f"{path.name}: cannot place the abscissas (no FIRSTX/LASTX/DELTAX)")
            # a line that starts with a DIF check value: its abscissa is that of the check point
            x = np.array([d.x * xfac + (j + d.checked) * delta for d in data for j in range(len(d.values))])
    elif (form.startswith("(XY") and ("..XY" in form or form in ("(XYXY)", "(XY)"))) or form.startswith("(XYW"):
        per = 3 if form.startswith("(XYW") else 2
        x, y = decode_xy_pairs(blk.data_lines, per)
        x, y = x * xfac, y * yfac
        if blk.data_label == "XYPOINTS":
            peaks = False
    else:
        raise ValueError(f"{path.name}: unsupported data form {blk.data_label}={form}")
    first_y = _num_label(lab, "FIRSTY")
    if first_y is not None and "++" in form and len(y) and not math.isclose(first_y, y[0], rel_tol=1e-3,
                                                                             abs_tol=abs(yfac)):
        warnings.append(f"FIRSTY={first_y:g} but the first ordinate decodes to {y[0]:g}")
    x_units, y_units = lab.get("XUNITS", ""), lab.get("YUNITS", "")
    xc, yc, xu, yu = convert_units(x, y, x_units, y_units, warnings)
    return Spectrum(xc, yc, xu, yu, title=lab.get("TITLE", path.stem) or path.stem, peaks=peaks,
                    source_format="jcamp", original_units=(x_units, y_units), warnings=warnings)


def looks_like_jcamp(path: Path) -> bool:
    if path.suffix.lower() in JCAMP_SUFFIXES:
        return True
    with path.open("rb") as fh:
        head = fh.read(4096).decode("latin-1")
    return bool(re.search(r"^\s*##\s*TITLE\s*=", head, re.M | re.I))


def read_spectrum(path: str | Path) -> Spectrum:
    """CSV/TSV/text table or JCAMP-DX, chosen by extension or content."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = read_jcamp(path) if looks_like_jcamp(path) else read_table(path)
    if len(spec) == 0:
        raise ValueError(f"{path.name}: no data points")
    return spec


def write_table(path: str | Path, columns: dict[str, np.ndarray]) -> Path:
    """CSV with a header (the first column is the abscissa)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(columns)
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(names)
    for row in zip(*(columns[k] for k in names)):
        w.writerow([f"{v:.6g}" if i else f"{v:.2f}" for i, v in enumerate(row)])
    path.write_text(buf.getvalue(), encoding="utf-8")
    return path
