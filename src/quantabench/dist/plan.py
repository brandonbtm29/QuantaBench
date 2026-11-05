"""Deterministic weighted splitting of molecule ids between targets (machines or lanes).

``weighted_index(mol_id, weights)`` maps the SHA-256 of the id to [0, 1) and picks the target whose
cumulative-weight bin contains it. With equal weights over n targets, the shares match ``jobs.chunk_index``
statistically (not id by id: chunk_index uses a modulo). Every machine computes the same answer, and adding
molecules to a list never moves the existing ones.

A target with weight 0 owns an empty bin, so adding one (say, a machine that cannot run the method) moves
nothing. Changing a positive weight does move molecules: the bins after it shift.

``parse_weights(spec, method, profiles)`` turns a ``--chunk-weights`` value into ``Weights``:

    "3,1"                                   positional: chunk 1 gets 3 parts, chunk 2 gets 1
    "btm-nova=3,brandons-macbook-pro=1"     named: chunk k is the k-th machine named
    "auto"                                  every machine in the registry (WSL profiles only when named),
                                            sorted by name, weighted by its measured throughput for
                                            ``method`` (``dist.profile``)
    "btm-nova=auto,brandons-macbook-pro=2"  mixed: "auto" per machine

A machine without a benchmark for the method is estimated from its core count (``ESTIMATE_PER_CORE``), and
the notes say so. ``explain(weights, n)`` prints the shares; ``resolve_chunk`` combines a weights spec with
``--chunk K/N`` (or finds this machine's chunk in a named spec).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

# Rough benchmark-set molecules/hour per physical core (``profile.benchmark`` protocol), used only when a
# machine has no measurement for the method, so that a measured and an unmeasured machine can be compared.
# Only the ratios between machines matter, and per-core rates vary a lot: measured 2026-11, MMFF gave 9,700/h
# per core on a 2-core cloud Xeon and 2,200 on BTM NOVA (12 cores); GFN1-xTB (dxtb, CPU) gave 250 and 24
# (AMD's Windows ROCm torch has a slow CPU BLAS). A real benchmark is always better than these.
ESTIMATE_PER_CORE: dict[str, float] = {
    "mmff": 3000.0,
    "gfn1-xtb": 150.0,
    "gfn2-xtb": 60.0,
    "aimnet2": 150.0,
    "xtb-gfn2": 100.0,
    "xtb-gfn1": 120.0,
    "gfnff": 1000.0,
    "tblite-gfn2": 60.0,
    "tblite-gfn1": 100.0,
    "mopac-pm7": 200.0,
    "mopac-pm6": 200.0,
}
DEFAULT_PER_CORE = 100.0


def unit_interval(mol_id: str) -> float:
    digest = hashlib.sha256(mol_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2.0**64


def weighted_index(mol_id: str, weights: Sequence[float]) -> int:
    w = [max(0.0, float(x)) for x in weights]
    total = sum(w)
    if not w or total <= 0:
        raise ValueError("weights must contain at least one positive value")
    u = unit_interval(mol_id) * total
    acc = 0.0
    for i, x in enumerate(w):
        acc += x
        if u < acc:
            return i
    return max(i for i, x in enumerate(w) if x > 0)


def split(ids: Sequence[str], weights: dict[str, float]) -> dict[str, list[str]]:
    """{target: [ids]} for every target in ``weights`` (empty lists included), in input order."""
    names = list(weights)
    out: dict[str, list[str]] = {n: [] for n in names}
    for mol_id in ids:
        out[names[weighted_index(mol_id, [weights[n] for n in names])]].append(mol_id)
    return out


# ------------------------------------------------------------------------------------- weights
class Weights(dict):
    """``{target: weight}`` in chunk order (a plain dict for ``split``), plus where each weight came from.

    ``named``: targets are machine names (named or auto spec), not chunk numbers ("1", "2", ...).
    ``sources[target]``: "given", "benchmark gfn1-xtb@cpu", "estimate: 12 cores x 120/h", ...
    ``notes``: things the user should know (estimates, machines that cannot run the method).
    """

    def __init__(self, items=(), *, named: bool = False, sources: dict[str, str] | None = None,
                 notes: list[str] | None = None):
        super().__init__(items)
        self.named = named
        self.sources: dict[str, str] = dict(sources or {})
        self.notes: list[str] = list(notes or [])

    @property
    def values_list(self) -> list[float]:
        return [float(v) for v in self.values()]

    def spec(self) -> str:
        """The explicit form of these weights (what to pass on every machine instead of 'auto')."""
        vals = ",".join(_fmt_weight(v) for v in self.values())
        if not self.named:
            return vals
        return ",".join(f"{k}={_fmt_weight(v)}" for k, v in self.items())


def _fmt_weight(x: float) -> str:
    """Exact (round-trip) text of a weight, without a trailing '.0'."""
    x = float(x)
    if x == int(x) and abs(x) < 1e15:
        return str(int(x))
    return repr(x)


def round_sig(x: float, digits: int = 3) -> float:
    """``x`` to ``digits`` significant digits. Measured weights are rounded so that the explicit spec printed
    for the other machines (``Weights.spec``) reproduces the split exactly."""
    if x <= 0 or not math.isfinite(x):
        return 0.0
    return float(f"{x:.{digits}g}")


def norm_name(name: str) -> str:
    """Machine names compare like Tailscale MagicDNS names: lowercase, spaces and underscores as '-'."""
    return "-".join(name.strip().lower().replace("_", " ").split())


def _parse_number(text: str, what: str) -> float:
    try:
        x = float(text)
    except ValueError:
        raise ValueError(f"--chunk-weights: {what} {text!r} is not a number (or 'auto')") from None
    if not math.isfinite(x) or x < 0:
        raise ValueError(f"--chunk-weights: {what} {text!r} must be a finite number >= 0")
    return x


def _estimate(p, method: str) -> tuple[float, str]:
    cores = int(getattr(p, "physical_cores", 0) or 0) or max(1, int(getattr(p, "logical_cores", 0) or 0) // 2) or 1
    factor = ESTIMATE_PER_CORE.get(method, DEFAULT_PER_CORE)
    return cores * factor, f"estimate: {cores} cores x {factor:g}/h"


def auto_weight(p, method: str) -> tuple[float, str]:
    """(weight, source) for profile ``p`` and ``method``: its best measured throughput, 0 if the method is
    unavailable there, else an estimate from the core count."""
    from .profile import best_throughput

    engines = getattr(p, "engines", {}) or {}
    if method in engines and not engines[method]:
        return 0.0, f"{method} unavailable"
    hit = best_throughput(p, method)
    if hit is not None:
        key, value = hit
        return round_sig(float(value)), f"benchmark {key}"
    value, source = _estimate(p, method)
    return round_sig(value), source


def parse_weights(spec: str, method: str = "", profiles: Sequence | None = None) -> Weights:
    """Parse a ``--chunk-weights`` value; see the module docstring. ``profiles``: the machine registry
    (``profile.load_profiles()``), needed only for "auto"."""
    text = (spec or "").strip()
    if not text:
        raise ValueError("--chunk-weights is empty")
    parts = [x.strip() for x in text.split(",") if x.strip()]
    if not parts:
        raise ValueError(f"--chunk-weights {spec!r}: no weights")
    by_name = {norm_name(p.name): p for p in (profiles or [])}

    def need_method() -> None:
        if not method or method == "auto":
            raise ValueError("--chunk-weights auto needs an explicit --method: method 'auto' picks the engine "
                             "per machine, so throughputs (and the level of theory) would not match")

    if len(parts) == 1 and parts[0].lower() == "auto":
        need_method()
        if not by_name:
            raise ValueError("--chunk-weights auto: the machine registry is empty. Run `quantabench-cli machines "
                             "--refresh --bench <method>` on each machine first (or give weights like 3,1)")
        from .profile import STALE_DAYS, profile_age_s

        w = Weights(named=True)
        skipped = sorted(n for n, p in by_name.items() if getattr(p, "wsl", False))
        for name in sorted(set(by_name) - set(skipped)):
            w[name], w.sources[name] = auto_weight(by_name[name], method)
            age = profile_age_s(by_name[name])
            if age is not None and age > STALE_DAYS * 86400:
                w.notes.append(f"{name}'s profile is {age / 86400:.0f} days old")
        if skipped:  # a WSL instance shares its computer's cores: giving it a share of its own double-counts them
            w.notes.append(f"left out WSL profile(s) {', '.join(skipped)}; name them to include them "
                           f"({skipped[0]}=auto)")
        if not w:
            raise ValueError("--chunk-weights auto: the registry has only WSL profiles; name the machines")
        _auto_notes(w, method)
        return w

    named = ["=" in x for x in parts]
    if any(named) and not all(named):
        raise ValueError(f"--chunk-weights {spec!r}: use either all positional weights (3,1) or all named "
                         "ones (btm-nova=3,brandons-macbook-pro=1)")
    if not any(named):
        w = Weights(named=False)
        for k, x in enumerate(parts, 1):
            if x.lower() == "auto":
                raise ValueError("--chunk-weights: 'auto' is either the whole value or a machine's value "
                                 "(name=auto)")
            w[str(k)] = _parse_number(x, f"weight {k}")
            w.sources[str(k)] = "given"
    else:
        w = Weights(named=True)
        for x in parts:
            name, _, value = x.partition("=")
            name = norm_name(name)
            if not name:
                raise ValueError(f"--chunk-weights: {x!r} has no machine name")
            if name in w:
                raise ValueError(f"--chunk-weights: {name} is named twice")
            if value.strip().lower() == "auto":
                need_method()
                p = by_name.get(name)
                if p is None:
                    raise ValueError(f"--chunk-weights: {name}=auto, but {name} is not in the machine registry "
                                     f"(known: {', '.join(sorted(by_name)) or 'none'})")
                w[name], w.sources[name] = auto_weight(p, method)
            else:
                w[name] = _parse_number(value.strip(), f"weight of {name}")
                w.sources[name] = "given"
        if any(s.startswith(("benchmark", "estimate", f"{method} unavailable")) for s in w.sources.values()):
            _auto_notes(w, method)
    if sum(w.values()) <= 0:
        raise ValueError(f"--chunk-weights {spec!r}: at least one weight must be positive")
    return w


def _auto_notes(w: Weights, method: str) -> None:
    est = [n for n, s in w.sources.items() if s.startswith("estimate")]
    zero = [n for n, s in w.sources.items() if s.endswith("unavailable")]
    measured = [n for n, s in w.sources.items() if s.startswith("benchmark")]
    if est:
        how = "core count x a per-engine factor"
        if measured:
            w.notes.append(f"no {method} benchmark for {', '.join(est)}: weight estimated from the {how}, next to "
                           f"measured {', '.join(measured)}. Run `quantabench-cli machines --refresh --bench "
                           f"{method}` there for a real number.")
        else:
            w.notes.append(f"no machine has a {method} benchmark: weights are estimated from the {how} only. "
                           f"Run `quantabench-cli machines --refresh --bench {method}` on each machine.")
    if zero:
        w.notes.append(f"{', '.join(zero)} cannot run {method} (weight 0, no molecules)")
    if sum(w.values()) <= 0:
        raise ValueError(f"no machine in the registry can run {method}")


def explain(weights: dict[str, float], n: int | Sequence[str] | None, this: str | None = None) -> str:
    """Human-readable shares. ``n``: number of molecules (expected counts, '~'), the ids themselves (exact
    counts from ``split``), or None (shares only). ``this`` marks one target as this machine."""
    names = list(weights)
    total_w = sum(max(0.0, float(v)) for v in weights.values())
    if total_w <= 0:
        raise ValueError("weights must contain at least one positive value")
    exact = None
    if n is None:
        count = None
    elif isinstance(n, int):
        count = n
    else:
        ids = list(n)
        count = len(ids)
        exact = {k: len(v) for k, v in split(ids, {k: float(weights[k]) for k in names}).items()}
    sources = getattr(weights, "sources", {})
    width = max(len(x) for x in names)
    lines = ["Split by weight:" if count is None else f"Split of {count:,} molecule(s) by weight:"]
    for k in names:
        share = max(0.0, float(weights[k])) / total_w
        if count is None:
            got = ""
        else:
            got = f"{exact[k]:,}" if exact is not None else f"~{round(share * count):,}"
        mark = "  <- this machine" if this is not None and k == this else ""
        src = sources.get(k, "")
        src = f"  ({src})" if src and src != "given" else ""
        lines.append(f"  {k:<{width}}  weight {_fmt_weight(weights[k]):>7}  {100 * share:5.1f}%  "
                     f"{got:>8}{src}{mark}")
    for note in getattr(weights, "notes", []):
        lines.append(f"  note: {note}")
    return "\n".join(lines)


def chunk_plan(weights: Weights, chunk: tuple, ids: Sequence[str] | None = None, warnings: Sequence[str] = (),
               this: str | None = None) -> str:
    """What ``build --chunk-weights`` prints before it runs: the shares, this machine's count, and, when a
    weight was measured or estimated, the explicit spec to give the other machines (so that a benchmark
    refreshed in between cannot make them compute a different split)."""
    from .profile import machine_name

    names = list(weights)
    i, n = int(chunk[0]), int(chunk[1])
    me = norm_name(this or machine_name())
    unique = list(dict.fromkeys(ids)) if ids else None
    lines = [explain(weights, unique or None, this=names[i])]
    who = f"{me}, chunk {i + 1}/{n}" if not weights.named or names[i] != me else me
    who += f", weight {_fmt_weight(weights[names[i]])}"
    if unique:
        mine = sum(1 for x in unique if weighted_index(x, weights.values_list) == i)
        lines.append(f"this machine ({who}): {mine:,} of {len(unique):,} molecules")
    else:
        total = sum(weights.values())
        lines.append(f"this machine ({who}): {100 * weights[names[i]] / total:.1f}% of the run's list")
    for w in warnings:
        lines.append(f"warning: {w}")
    if any(s != "given" for s in weights.sources.values()):
        lines.append(f"Give the other machines the same split: --chunk-weights {weights.spec()}")
    return "\n".join(lines)


def resolve_chunk(spec: str, chunk: tuple[int, int] | None, method: str = "", profiles: Sequence | None = None,
                  this: str | None = None) -> tuple[tuple[int, int, tuple[float, ...]], Weights, list[str]]:
    """``--chunk-weights`` + ``--chunk`` -> (weighted chunk ``(i, n, weights)``, Weights, warnings).

    Without ``chunk``, a named or auto spec picks this machine's position (``this``, default
    ``profile.machine_name()``). With both, the counts must agree; a named chunk that is not this machine's
    is allowed (running another machine's share here) but warned about.
    """
    from .profile import machine_name

    w = parse_weights(spec, method, profiles)
    names = list(w)
    me = norm_name(this or machine_name())
    warnings: list[str] = []
    if chunk is None:
        if not w.named:
            raise ValueError(f"--chunk-weights {spec}: positional weights need --chunk K/{len(names)}")
        if me not in names:
            raise ValueError(f"this machine ({me}) is not in --chunk-weights ({', '.join(names)}): add it, or pass "
                             f"--chunk K/{len(names)} to run another machine's share")
        i = names.index(me)
    else:
        i, n = int(chunk[0]), int(chunk[1])
        if n != len(names):
            raise ValueError(f"--chunk {i + 1}/{n} does not match --chunk-weights {spec!r}, which has {len(names)} "
                             f"target(s): {', '.join(names)}")
        if w.named and names[i] != me:
            warnings.append(f"chunk {i + 1}/{n} is {names[i]}'s share, but this machine is {me}")
    if float(w[names[i]]) <= 0:
        raise ValueError(f"chunk {i + 1}/{len(names)} ({names[i]}) has weight 0: it would get no molecules")
    return (i, len(names), tuple(w.values_list)), w, warnings
