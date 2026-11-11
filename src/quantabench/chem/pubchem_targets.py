"""One target value per molecule from PubChem's experimental records (``pubchem enrich --records``).

The long-form records keep every value PubChem has, with its temperature, condition class (``pure_liquid``,
``aqueous_solution``, ``unknown``, ...), source and reference. A model needs one value per molecule at one
temperature: docs/SURFACE_TENSION.md, "Data assembly", step 3. This module makes that table:

1. Records of one heading (default "Surface Tension") with a standard value, in the wanted condition classes
   (default ``pure_liquid``). Left out, and counted per molecule: records marked ``needs_review``, estimated
   values ("est", "estimated", "calculated", "predicted" in the text), records without a stated temperature,
   and records outside the temperature window (default 20-30 °C).
   Values outside the heading's plausible range (:data:`PLAUSIBLE`) are repaired when they are a factor 1000 off
   and the text names that unit (N/m written for mN/m), else left out.
2. The temperature slope: the molecule's own when its usable records span at least ``min_span`` K (a least
   squares line through the records within ``fit_span`` K of the target), if that slope lies in the plausible
   range; otherwise the stated default (surface tension -0.1 mN/m per K, density -0.001 g/cm³ per K).
3. Each record in the window is moved to the target temperature: value + slope x (target - T).
4. One value per reference (its record nearest the target), and each reference gets a tier from its text:
   ``compilation`` (CRC Handbook, Riddick's Organic Solvents, Jasper, Daubert & Danner/DIPPR, Lange, Yaws, TRC,
   Landolt-Börnstein, ...), ``other`` (encyclopedias, monographs, dossiers), ``chris`` (the U.S. Coast Guard's
   CHRIS manual: rounded and sometimes estimated values).
5. The target is the median over the best tier present. The spread over every reference in the window is kept,
   and flagged above ``spread_flag`` (default 2 mN/m).

Every molecule with a record of the heading gets a row; ``status`` says ``ok`` or why there is no value. The raw
value, its temperature and the correction stay in the row, so nothing is hidden by the correction.
"""

from __future__ import annotations

import csv
import re
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

#: Typical temperature slopes of liquids, used when a molecule's own records do not give one (per K).
DEFAULT_SLOPES = {"Surface Tension": -0.1, "Density": -0.001}
#: The plausible range of a molecule's own slope (per K); a fit outside it falls back to the default.
SLOPE_RANGES = {"Surface Tension": (-0.3, -0.02), "Density": (-0.003, -0.0002)}
#: Plausible values of a liquid near room temperature (standard units: mN/m, g/cm³). A value outside is a unit slip
#: of the source or of the parser (HSDB: "38.00 N/m at 25 °C" for mN/m; "43.55 dynes/cm; 0.04355 Newtons/meter"
#: read as 0.04355 dyn/cm). Such a value is repaired when the factor 1000 brings it inside and the record's text
#: names that unit (flag ``unit_repaired``), else left out (``excluded_implausible``).
PLAUSIBLE = {"Surface Tension": (5.0, 80.0), "Density": (0.3, 3.5)}

#: Reference tiers, best first: (tier, rank, pattern). A reference matching no pattern is ``other`` (rank 2).
TIERS: tuple[tuple[str, int, str], ...] = (
    ("compilation", 1, r"CRC Handbook|Handbook of Chemistry and Physics|Riddick|Organic Solvents|Jasper|"
                       r"Daubert|DIPPR|Physical and Thermodynamic Properties of Pure Chemicals|Lange'?s|Yaws|"
                       r"\bTRC\b|Thermodynamics Research Center|Landolt|Wohlfarth|Mulero|Vargaftik"),
    ("chris", 3, r"Coast Guard|\bCHRIS\b"),
)
OTHER_TIER = ("other", 2)

_ESTIMATED = re.compile(r"\b(est\.?|estimated|calculated|calcd\.?|predicted)\b", re.IGNORECASE)

COLUMNS = (
    "id", "input_cas", "pubchem_inchikey", "cid", "heading", "status", "target_t_c", "value", "unit", "tier",
    "n_references_tier", "value_all_median", "spread_all", "n_references", "n_records_window", "value_raw",
    "t_raw_c", "correction", "slope", "slope_source", "reference", "references", "flags", "n_records",
    "excluded_review", "excluded_estimated", "excluded_implausible", "excluded_no_temperature",
    "excluded_outside_window", "excluded_class",
)


def _float(v) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    text = str(v).strip()
    if not text or text.lower() in ("nan", "none", "n/a"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _flag(v) -> bool:
    return str(v or "").strip().lower() in ("yes", "true", "1", "y")


def reference_tier(reference: str, tiers: Sequence[tuple[str, int, str]] = TIERS) -> tuple[str, int]:
    """(tier, rank) of a reference text: the first pattern that matches, else ``other`` (rank 2)."""
    for name, rank, pattern in tiers:
        if re.search(pattern, reference or "", re.IGNORECASE):
            return name, rank
    return OTHER_TIER


def is_estimated(raw: str) -> bool:
    """Whether a record's text says the value is estimated, calculated or predicted."""
    return bool(_ESTIMATED.search(raw or ""))


def _ref_key(reference: str, source: str) -> str:
    text = re.sub(r"\s+", " ", (reference or source or "").strip().lower())
    return text[:160] or "(no reference)"


@dataclass
class Settings:
    heading: str = "Surface Tension"
    classes: tuple[str, ...] = ("pure_liquid",)
    target_t: float = 25.0
    window: tuple[float, float] = (20.0, 30.0)
    slope: float | None = None                   # None: DEFAULT_SLOPES[heading] (0 for another heading)
    slope_range: tuple[float, float] | None = None   # None: SLOPE_RANGES[heading]; no own fit for another heading
    fit_span: float = 75.0                       # K around the target that the own-slope fit uses
    min_span: float = 10.0                       # K that the fitted records must span
    include_review: bool = False
    include_estimated: bool = False
    allow_no_temperature: bool = False           # records without a temperature count as at the target
    spread_flag: float = 2.0
    tiers: tuple[tuple[str, int, str], ...] = field(default=TIERS)

    @property
    def default_slope(self) -> float:
        return self.slope if self.slope is not None else DEFAULT_SLOPES.get(self.heading, 0.0)

    @property
    def slope_bounds(self) -> tuple[float, float] | None:
        return self.slope_range if self.slope_range is not None else SLOPE_RANGES.get(self.heading)


_N_PER_M = re.compile(r"\bN\s*/\s*m\b|newtons?\s*/\s*m", re.IGNORECASE)


def plausible_value(record: dict, value: float, heading: str) -> tuple[float | None, bool]:
    """(value, repaired) of a record's standard value: as it is when it lies in :data:`PLAUSIBLE` (or the heading
    has no range), times or divided by 1000 when that brings it inside and the text names N/m (a surface tension
    printed in N/m but meant as mN/m, or the N/m number of a "dyn/cm = N/m" pair read with the dyn/cm unit),
    else None."""
    rng = PLAUSIBLE.get(heading)
    if rng is None or rng[0] <= value <= rng[1]:
        return value, False
    if heading == "Surface Tension" and _N_PER_M.search(record.get("raw", "") or ""):
        for fixed in (value / 1000.0, value * 1000.0):
            if rng[0] <= fixed <= rng[1]:
                return fixed, True
    return None, False


def _fit_slope(points: list[tuple[float, float]]) -> float | None:
    if len({t for t, _ in points}) < 2:
        return None
    n = len(points)
    mt = sum(t for t, _ in points) / n
    mv = sum(v for _, v in points) / n
    sxx = sum((t - mt) ** 2 for t, _ in points)
    if sxx <= 0:
        return None
    return sum((t - mt) * (v - mv) for t, v in points) / sxx


def _fmt(v, nd: int = 4):
    if v is None:
        return ""
    return round(v, nd) if isinstance(v, float) else v


def molecule_target(records: list[dict], s: Settings) -> dict:
    """The target row of one molecule's records (dicts with the ``pubchem enrich --records`` columns)."""
    first = records[0] if records else {}
    row = {c: "" for c in COLUMNS}
    row.update(id=first.get("id", ""), input_cas=first.get("input_cas", ""),
               pubchem_inchikey=first.get("pubchem_inchikey", ""), cid=first.get("cid", ""), heading=s.heading,
               target_t_c=s.target_t)
    recs = [r for r in records if r.get("heading") == s.heading and _float(r.get("value_std")) is not None]
    row["n_records"] = len(recs)
    counts = dict.fromkeys(("review", "estimated", "implausible", "no_temperature", "outside_window", "class"), 0)
    usable: list[tuple[dict, float, float | None]] = []      # (record, value, temperature)
    repaired = 0
    for r in recs:
        if s.classes and r.get("condition_class", "") not in s.classes:
            counts["class"] += 1
            continue
        if _flag(r.get("needs_review")) and not s.include_review:
            counts["review"] += 1
            continue
        if is_estimated(r.get("raw", "")) and not s.include_estimated:
            counts["estimated"] += 1
            continue
        value, fixed = plausible_value(r, _float(r["value_std"]), s.heading)
        if value is None:
            counts["implausible"] += 1
            continue
        repaired += fixed
        usable.append((r, value, _float(r.get("temperature_c"))))
    unit = next((r.get("unit_std", "") for r, _, _ in usable if r.get("unit_std")), "")
    row["unit"] = unit
    # the molecule's own slope, from every usable record with a temperature near the target
    pts = [(t, v) for _, v, t in usable if t is not None and abs(t - s.target_t) <= s.fit_span]
    slope, slope_source = s.default_slope, "default"
    bounds = s.slope_bounds
    if pts and max(t for t, _ in pts) - min(t for t, _ in pts) >= s.min_span and bounds:
        fit = _fit_slope(pts)
        if fit is not None and bounds[0] <= fit <= bounds[1]:
            slope = fit
            slope_source = f"own fit ({len(pts)} records, {min(t for t, _ in pts):g}-{max(t for t, _ in pts):g} °C)"
        elif fit is not None:
            slope_source = f"default (own fit {fit:.3g} outside {bounds[0]:g} to {bounds[1]:g})"
    row["slope"] = _fmt(slope, 5)
    row["slope_source"] = slope_source
    lo, hi = min(s.window), max(s.window)
    window: list[tuple[dict, float, float, bool]] = []       # (record, value, temperature, temperature assumed)
    for r, v, t in usable:
        if t is None:
            if not s.allow_no_temperature:
                counts["no_temperature"] += 1
                continue
            window.append((r, v, s.target_t, True))
        elif lo <= t <= hi:
            window.append((r, v, t, False))
        else:
            counts["outside_window"] += 1
    for k, n in counts.items():
        row[f"excluded_{k}"] = n
    row["n_records_window"] = len(window)
    if not recs:
        row["status"] = "no value"
        return row
    if not window:
        why = max(counts, key=lambda k: counts[k]) if any(counts.values()) else "none usable"
        row["status"] = {"class": "no record in the wanted condition class", "review": "only records to review",
                         "estimated": "only estimated values", "implausible": "only implausible values",
                         "no_temperature": "no stated temperature",
                         "outside_window": f"no record within {lo:g}-{hi:g} °C"}.get(why, why)
        return row
    # one value per reference: its record nearest the target
    # reference -> (|T - target|, record, value, T, T assumed, tier, rank, value at the target)
    per_ref: dict[str, tuple] = {}
    for r, v, t, assumed in window:
        key = _ref_key(r.get("reference", ""), r.get("source_name", ""))
        corrected = v + slope * (s.target_t - t)
        tier, rank = reference_tier(r.get("reference", "") or r.get("source_name", ""), s.tiers)
        cand = (abs(t - s.target_t), r, v, t, assumed, tier, rank, corrected)
        old = per_ref.get(key)
        if old is None or cand[0] < old[0]:
            per_ref[key] = cand
    refs = sorted(per_ref.values(), key=lambda c: (c[6], c[0]))
    best_rank = refs[0][6]
    best = [c for c in refs if c[6] == best_rank]
    rep = best[0]
    all_values = [c[7] for c in refs]
    spread = max(all_values) - min(all_values)
    flags = []
    if spread > s.spread_flag:
        flags.append(f"spread>{s.spread_flag:g}")
    if len(refs) == 1:
        flags.append("single_reference")
    if any(c[4] for c in refs):
        flags.append("temperature_assumed")
    if slope_source.startswith("default"):
        flags.append("default_slope")
    if repaired:
        flags.append("unit_repaired")
    row.update(
        status="ok", value=_fmt(statistics.median(c[7] for c in best)), tier=rep[5], n_references_tier=len(best),
        value_all_median=_fmt(statistics.median(all_values)), spread_all=_fmt(spread), n_references=len(refs),
        value_raw=_fmt(rep[2]), t_raw_c=_fmt(rep[3]), correction=_fmt(slope * (s.target_t - rep[3])),
        reference=(rep[1].get("reference", "") or rep[1].get("source_name", ""))[:300],
        references=" | ".join(f"{c[7]:.2f} [{c[5]}] {(c[1].get('reference', '') or c[1].get('source_name', ''))[:80]}"
                              for c in refs)[:1000],
        flags=";".join(flags))
    return row


def targets(records: Iterable[dict], settings: Settings | None = None) -> list[dict]:
    """One row per molecule (by ``id``, else InChIKey or CID) that has a record of the heading, in input order."""
    s = settings or Settings()
    groups: dict[str, list[dict]] = {}
    for r in records:
        if r.get("heading") != s.heading:
            continue
        key = r.get("id") or r.get("pubchem_inchikey") or str(r.get("cid", ""))
        groups.setdefault(key, []).append(r)
    return [molecule_target(recs, s) for recs in groups.values()]


def read_records(path: str | Path) -> list[dict]:
    """The long-form records CSV of ``pubchem enrich --records``."""
    csv.field_size_limit(1 << 30)
    with Path(path).expanduser().open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: str | Path, rows: list[dict]) -> Path:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(COLUMNS), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    tmp.replace(path)
    return path


def summary(rows: list[dict]) -> str:
    ok = [r for r in rows if r["status"] == "ok"]
    tiers: dict[str, int] = {}
    for r in ok:
        tiers[r["tier"]] = tiers.get(r["tier"], 0) + 1
    flagged = sum(1 for r in ok if "spread>" in r["flags"])
    own = sum(1 for r in ok if str(r["slope_source"]).startswith("own"))
    why: dict[str, int] = {}
    for r in rows:
        if r["status"] != "ok":
            why[r["status"]] = why.get(r["status"], 0) + 1
    parts = [f"{len(ok)} of {len(rows)} molecule(s) with a target value"]
    if tiers:
        parts.append("tiers " + ", ".join(f"{k} {v}" for k, v in sorted(tiers.items())))
    parts.append(f"{own} with their own temperature slope")
    if flagged:
        parts.append(f"{flagged} with a spread above the flag")
    if why:
        parts.append("no value: " + ", ".join(f"{k} {v}" for k, v in sorted(why.items(), key=lambda kv: -kv[1])))
    return "; ".join(parts)
