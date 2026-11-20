"""Surface tension and densities from a liquid-slab run of ``md.liquid`` (MD-2).

``analyze(folder)`` reads ``slab_box.txt``, ``pressure_tensor.txt``, ``density_profile.txt``, ``bulk_density.txt``
and the logs, and returns a report dict; ``write_report`` saves it as ``surface_tension.json`` and
``surface_tension.md`` in the folder. ``quantabench-cli md surface-tension DIR`` does both.

The estimator (Kirkwood & Buff, J. Chem. Phys. 1949, 17, 338): for a slab with two interfaces normal to z,

    σ = (Lz / 2) · ⟨Pzz − (Pxx + Pyy) / 2⟩

with Lz the whole box (liquid and vacuum). LAMMPS ``real`` units give pressures in atm and lengths in Å, and
1 atm·Å = 101325 Pa × 1e-10 m = 1.01325e-5 N/m = 0.0101325 mN/m (``ATM_A_TO_MN_M``).

Errors: the samples (every 10 fs) are strongly correlated, so the standard error comes from block averages,
1 ns blocks by default as the plan asks (at least 4 blocks: shorter runs get 4 equal blocks and a note). Two
checks go with it: the statistical inefficiency g (Chodera et al. 2007's estimator, the number of correlated
samples per independent one) and its standard error σ_x·√(g/N), and the Flyvbjerg-Petersen blocking curve
(J. Chem. Phys. 1989, 91, 461), whose plateau is the same quantity estimated another way.

Densities: each 10-ps density profile is centred on the slab's centre of mass (a circular mean over the periodic
box, so a slab that drifts or crosses the boundary is still found), the centred profiles of the production run are
averaged, and ρ(z) = ½(ρl+ρv) − ½(ρl−ρv)·tanh(2(|z−zc|−z0)/d) is fitted: ρl the liquid plateau, ρv the vapour,
z0 the Gibbs dividing surface, d the interface width (10-90 % width = 1.0986·d). The plateau is also averaged
directly over the slab's middle (|z−zc| < z0 − 2d), and compared with the bulk NPT density: a slab whose middle is
not at the bulk density is too thin or not equilibrated.
"""

from __future__ import annotations

import json
import math
import re
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from .. import __version__

#: 1 atm·Å in mN/m: 101325 Pa × 1e-10 m = 1.01325e-5 N/m
ATM_A_TO_MN_M = 0.0101325
#: tanh profile: the 10-90 % width is atanh(0.8) · d = 1.0986 d for ρ ∝ tanh(2(z − z0)/d)
TANH_10_90 = math.atanh(0.8)


# ------------------------------------------------------------------------------------------------ readers
def read_slab_box(path: str | Path) -> dict[str, float]:
    """``slab_box.txt`` (``QB_SLAB lx .. ly .. lz ..``) as a dict of numbers."""
    tok = Path(path).read_text(encoding="utf-8").split()
    out = {}
    for k, v in zip(tok[1::2], tok[2::2]):
        try:
            out[k] = float(v)
        except ValueError:
            pass
    return out


def read_pressure_tensor(path: str | Path) -> np.ndarray:
    """Rows (step, Pxx, Pyy, Pzz) of ``pressure_tensor.txt`` (atm)."""
    a = np.loadtxt(path, comments="#", ndmin=2)
    if a.shape[1] < 4:
        raise ValueError(f"{path}: expected 4 columns (step, Pxx, Pyy, Pzz)")
    return a[:, :4]


def read_profiles(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``fix ave/chunk`` output: (steps (nblocks,), bin centres (nbins,), density (nblocks, nbins))."""
    steps, blocks = [], []
    z = None
    rows: list[list[float]] = []
    want = 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        t = s.split()
        if want == 0:
            steps.append(int(float(t[0])))
            want = int(float(t[1]))
            rows = []
            continue
        rows.append([float(x) for x in t])
        want -= 1
        if want == 0:
            r = np.array(rows)
            if z is None:
                z = r[:, 1]
            if len(r) == len(z):
                blocks.append(r[:, -1])
            else:
                steps.pop()
    if z is None or not blocks:
        return np.array([], dtype=int), np.array([]), np.zeros((0, 0))
    n = min(len(steps), len(blocks))
    return np.array(steps[:n]), z, np.array(blocks[:n])


def read_bulk_density(path: str | Path) -> np.ndarray:
    """Rows (step, density g/cm³, volume Å³, lx, ly, lz, pe) of ``bulk_density.txt`` (none yet: shape (0, 0))."""
    with warnings.catch_warnings():          # a run shorter than the first 1-ps average leaves only the header
        warnings.simplefilter("ignore", UserWarning)
        return np.loadtxt(path, comments="#", ndmin=2)


def log_performance(path: str | Path) -> list[float]:
    """The ``Performance: X ns/day`` values of a LAMMPS log, in order."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return [float(m) for m in re.findall(r"Performance:\s*([0-9.eE+-]+)\s*ns/day", text)]


# ----------------------------------------------------------------------------------------------- statistics
@dataclass
class BlockAverage:
    mean: float
    sem: float
    n_blocks: int
    block_size: int                      # samples per block
    block_means: list[float]


def block_average(x: np.ndarray, block_size: int) -> BlockAverage:
    """Mean and its standard error from non-overlapping blocks of ``block_size`` samples (a trailing partial
    block is dropped from the error, not from the mean)."""
    x = np.asarray(x, dtype=float)
    nb = len(x) // max(1, int(block_size))
    if nb < 2:
        return BlockAverage(float(x.mean()) if len(x) else math.nan, math.nan, nb, int(block_size), [])
    means = x[: nb * block_size].reshape(nb, block_size).mean(axis=1)
    return BlockAverage(float(x.mean()), float(means.std(ddof=1) / math.sqrt(nb)), nb, int(block_size),
                        [float(m) for m in means])


def statistical_inefficiency(x: np.ndarray, mintime: int = 3) -> float:
    """g = 1 + 2 Σ_t (1 − t/N) C(t), summed until the normalized autocorrelation C(t) first drops to zero or
    below after ``mintime`` lags (the estimator of Chodera et al., J. Chem. Theory Comput. 2007, 3, 26, also in
    pymbar). Computed with an FFT, so millions of samples are fine. g ≥ 1."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 4:
        return math.nan
    dx = x - x.mean()
    var = float(dx @ dx) / n
    if var == 0:
        return 1.0
    f = np.fft.rfft(dx, n=2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n] / (np.arange(n, 0, -1) * var)
    t = np.arange(1, n - 1)
    stop = np.nonzero((acf[1:n - 1] <= 0) & (t > mintime))[0]
    end = int(stop[0]) if len(stop) else len(t)          # lags 1 .. end (the first non-positive one excluded)
    g = 1.0 + 2.0 * float(np.sum(acf[1:end + 1] * (1.0 - t[:end] / n)))
    return max(1.0, g)


def blocking_curve(x: np.ndarray) -> list[tuple[int, float, float]]:
    """Flyvbjerg-Petersen: (block length in samples, SEM, its uncertainty) as pairs of neighbours are averaged
    repeatedly; the SEM rises and levels off at the true standard error."""
    x = np.asarray(x, dtype=float)
    out = []
    size = 1
    while len(x) >= 4:
        n = len(x)
        s = float(np.sqrt(x.var(ddof=1) / n)) if n > 1 else math.nan
        out.append((size, s, s / math.sqrt(2.0 * (n - 1))))
        x = 0.5 * (x[: n // 2 * 2 : 2] + x[1 : n // 2 * 2 : 2])
        size *= 2
    return out


def sigma_series(p: np.ndarray, lz: float) -> np.ndarray:
    """σ (mN/m) per sample from rows (step, Pxx, Pyy, Pzz) in atm and the box length Lz in Å."""
    return 0.5 * lz * (p[:, 3] - 0.5 * (p[:, 1] + p[:, 2])) * ATM_A_TO_MN_M


# ---------------------------------------------------------------------------------------------- profiles
def profile_centre(z: np.ndarray, rho: np.ndarray, zlo: float, lz: float) -> float:
    """Centre of mass of a profile in a periodic box: the circular mean of z weighted by density."""
    th = 2.0 * math.pi * (np.asarray(z) - zlo) / lz
    a = math.atan2(float(np.sum(rho * np.sin(th))), float(np.sum(rho * np.cos(th))))
    return zlo + (a % (2.0 * math.pi)) * lz / (2.0 * math.pi)


def centred_profiles(z: np.ndarray, rho: np.ndarray, zlo: float, lz: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each block rolled by whole bins so its centre sits at the box middle. Returns (distance of each bin from
    the middle, signed (nbins,), rolled profiles (nblocks, nbins), centres per block (nblocks,))."""
    nb = len(z)
    dz = lz / nb
    mid = zlo + 0.5 * lz
    centres = np.array([profile_centre(z, r, zlo, lz) for r in rho])
    rolled = np.array([np.roll(r, int(round((mid - c) / dz))) for r, c in zip(rho, centres)])
    return z - mid, rolled, centres


def unwrap_centres(c: np.ndarray, lz: float) -> np.ndarray:
    if len(c) == 0:
        return c
    d = np.diff(c)
    d -= lz * np.round(d / lz)
    return np.concatenate([[c[0]], c[0] + np.cumsum(d)])


def fit_tanh(dist: np.ndarray, rho: np.ndarray) -> dict[str, float] | None:
    """Fit ρ(|z|) = ½(ρl+ρv) − ½(ρl−ρv)·tanh(2(|z| − z0)/d) to a centred, symmetric profile."""
    from scipy.optimize import curve_fit

    r = np.abs(dist)
    if rho.max() <= 0:
        return None

    def f(x, rl, rv, z0, d):
        return 0.5 * (rl + rv) - 0.5 * (rl - rv) * np.tanh(2.0 * (x - z0) / d)

    rl0 = float(np.median(rho[r < max(2.0, 0.2 * r.max())])) if np.any(r < max(2.0, 0.2 * r.max())) else float(rho.max())
    half = r[rho > 0.5 * rl0]
    z00 = float(half.max()) if len(half) else 0.25 * float(r.max())
    try:
        p, cov = curve_fit(f, r, rho, p0=[rl0, 0.0, z00, 5.0], maxfev=20000)
    except (RuntimeError, ValueError):
        return None
    err = np.sqrt(np.diag(cov)) if np.all(np.isfinite(cov)) else [math.nan] * 4
    return {"rho_liquid": float(p[0]), "rho_vapour": float(p[1]), "z0": float(p[2]), "d": abs(float(p[3])),
            "width_10_90": abs(float(p[3])) * TANH_10_90,
            "err_rho_liquid": float(err[0]), "err_z0": float(err[2]), "err_d": float(err[3])}


# ------------------------------------------------------------------------------------------------- analysis
def _setup(folder: Path) -> dict:
    p = folder / "liquid_setup.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def analyze(folder: str | Path, *, skip_ns: float | None = None, block_ns: float = 1.0,
            min_blocks: int = 4) -> dict:
    """The surface tension, densities and checks of a finished (or running) slab folder."""
    folder = Path(folder)
    setup = _setup(folder)
    cfg = setup.get("config", {})
    notes: list[str] = []
    checks: list[dict] = []
    box = read_slab_box(folder / "slab_box.txt")
    lz, zlo = box["lz"], box["zlo"]
    dt = box.get("dt", cfg.get("timestep", 1.0))
    settle_steps = box.get("settle_steps", round(cfg.get("settle_ns", 2.0) * 1e6 / dt))
    skip_steps = settle_steps if skip_ns is None else round(skip_ns * 1e6 / dt)
    out: dict = {"folder": str(folder), "generator": f"QuantaBench {__version__}",
                 "analyzed": datetime.now().isoformat(timespec="seconds"),
                 "molecule": setup.get("molecule", {}).get("name"), "smiles": setup.get("molecule", {}).get("smiles"),
                 "forcefield": setup.get("molecule", {}).get("label"), "dispersion": cfg.get("dispersion"),
                 "temperature_K": box.get("T", cfg.get("temperature")), "cutoff_A": box.get("rc", cfg.get("cutoff")),
                 "n_molecules": setup.get("n_molecules"), "box_A": [box.get("lx"), box.get("ly"), lz],
                 "bulk_lz_A": box.get("bulk_lz"), "timestep_fs": dt, "skip_ns": skip_steps * dt / 1e6}

    # --- σ from the pressure tensor ---
    p = read_pressure_tensor(folder / "pressure_tensor.txt")
    prod = p[p[:, 0] >= skip_steps]
    if len(prod) < 8:
        raise ValueError(f"only {len(prod)} pressure samples after the first {skip_steps * dt / 1e6:g} ns: the "
                         "production run has not started (or use --skip-ns)")
    sig = sigma_series(prod, lz)
    every = float(np.median(np.diff(prod[:, 0]))) if len(prod) > 1 else 1.0
    dt_sample_ns = every * dt / 1e6
    span_ns = (prod[-1, 0] - prod[0, 0] + every) * dt / 1e6
    per_block = int(round(block_ns / dt_sample_ns))
    nb = len(sig) // max(1, per_block)
    if nb < min_blocks:
        per_block = max(1, len(sig) // min_blocks)
        notes.append(f"production is {span_ns:.3g} ns: fewer than {min_blocks} blocks of {block_ns:g} ns, so "
                     f"{min_blocks} blocks of {per_block * dt_sample_ns:.3g} ns were used (a rough error bar)")
    ba = block_average(sig, per_block)
    g = statistical_inefficiency(sig)
    sem_g = float(sig.std(ddof=1) * math.sqrt(g / len(sig)))
    curve = blocking_curve(sig)
    half = len(sig) // 2
    out["sigma"] = {
        "mean_mN_m": ba.mean, "sem_mN_m": ba.sem, "n_blocks": ba.n_blocks,
        "block_ns": per_block * dt_sample_ns, "block_means_mN_m": ba.block_means,
        "statistical_inefficiency": g, "correlation_time_ps": 0.5 * (g - 1.0) * dt_sample_ns * 1e3,
        "sem_from_inefficiency_mN_m": sem_g, "n_samples": int(len(sig)), "sample_every_fs": every * dt,
        "production_ns": span_ns, "first_half_mN_m": float(sig[:half].mean()), "second_half_mN_m": float(sig[half:].mean()),
        "blocking_curve": [{"block_ps": s * dt_sample_ns * 1e3, "sem_mN_m": e, "sem_err": u} for s, e, u in curve],
        "pressure_atm": {"Pxx": float(prod[:, 1].mean()), "Pyy": float(prod[:, 2].mean()), "Pzz": float(prod[:, 3].mean())},
    }
    settle = p[p[:, 0] < skip_steps]
    if len(settle):
        out["sigma"]["settle_mean_mN_m"] = float(sigma_series(settle, lz).mean())
    if ba.n_blocks >= 2 and abs(out["sigma"]["first_half_mN_m"] - out["sigma"]["second_half_mN_m"]) > 3 * max(ba.sem, 1e-9) * math.sqrt(2):
        checks.append({"check": "sigma drift", "ok": False,
                       "detail": f"first half {out['sigma']['first_half_mN_m']:.2f}, second half "
                                 f"{out['sigma']['second_half_mN_m']:.2f} mN/m: not converged or not equilibrated"})
    min_ns = 7.0
    checks.append({"check": "production length", "ok": span_ns >= min_ns,
                   "detail": f"{span_ns:.3g} ns (Tsige & Grest 2008 needed at least 7 ns for perfluoroalkanes)"})

    # --- densities from the profiles ---
    prof_path = folder / "density_profile.txt"
    if prof_path.is_file():
        steps, z, rho = read_profiles(prof_path)
        keep = steps > skip_steps
        if keep.sum() >= 1:
            dist, rolled, centres = centred_profiles(z, rho[keep], zlo, lz)
            avg = rolled.mean(axis=0)
            fit = fit_tanh(dist, avg)
            dens: dict = {"n_profiles": int(keep.sum()), "bin_A": float(np.median(np.diff(z))),
                          "profile_z_A": [float(v) for v in dist], "profile_g_cm3": [float(v) for v in avg]}
            if fit:
                dens["tanh_fit"] = fit
                inner = np.abs(dist) < fit["z0"] - 2.0 * fit["d"]
                outer = np.abs(dist) > fit["z0"] + 3.0 * fit["d"]
                if inner.sum() >= 3:
                    per_block_plateau = rolled[:, inner].mean(axis=1)
                    dens["plateau_g_cm3"] = float(per_block_plateau.mean())
                    dens["plateau_sem_g_cm3"] = (float(per_block_plateau.std(ddof=1) / math.sqrt(len(per_block_plateau)))
                                                 if len(per_block_plateau) > 1 else math.nan)
                    dens["plateau_width_A"] = float(2.0 * (fit["z0"] - 2.0 * fit["d"]))
                else:
                    notes.append("no bulk-like plateau: the slab is thinner than its two interfaces")
                if outer.sum() >= 3:
                    dens["vapour_g_cm3"] = float(avg[outer].mean())
                dens["slab_thickness_gibbs_A"] = 2.0 * fit["z0"]
            uc = unwrap_centres(centres, lz)
            dens["centre_drift_A"] = float(uc[-1] - uc[0]) if len(uc) > 1 else 0.0
            dens["centre_max_excursion_A"] = float(np.max(np.abs(uc - uc[0]))) if len(uc) else 0.0
            out["density"] = dens
            checks.append({"check": "slab drift", "ok": dens["centre_max_excursion_A"] < 5.0,
                           "detail": f"centre moved {dens['centre_drift_A']:+.2f} Å (max {dens['centre_max_excursion_A']:.2f} Å)"})
        else:
            notes.append("no density profile after the settle time yet (profiles are written every 10 ps)")
    # --- bulk ---
    bulk_path = folder / "bulk_density.txt"
    if bulk_path.is_file():
        b = read_bulk_density(bulk_path)
        if len(b) >= 4:
            second = b[len(b) // 2:, 1]
            bb = block_average(second, max(1, len(second) // 5))
            out["bulk"] = {"density_g_cm3": float(second.mean()), "sem_g_cm3": bb.sem,
                           "last_half_ns": float((b[-1, 0] - b[len(b) // 2, 0]) * dt / 1e6),
                           "box_A": [float(b[len(b) // 2:, k].mean()) for k in (3, 4, 5)]}
    if "bulk" in out and "density" in out and "plateau_g_cm3" in out["density"]:
        rb, rp = out["bulk"]["density_g_cm3"], out["density"]["plateau_g_cm3"]
        rel = (rp - rb) / rb
        out["density"]["plateau_vs_bulk"] = rel
        checks.append({"check": "slab centre at the bulk density", "ok": abs(rel) < 0.01,
                       "detail": f"plateau {rp:.4f} vs bulk NPT {rb:.4f} g/cm3 ({rel:+.2%})"})
    # --- references and cost ---
    ref_s, ref_d = cfg.get("reference_sigma"), cfg.get("reference_density")
    if ref_s:
        out["reference"] = {"sigma_mN_m": ref_s, "note": cfg.get("reference_note", ""),
                            "sigma_diff_mN_m": ba.mean - ref_s, "sigma_rel": (ba.mean - ref_s) / ref_s}
    if ref_d and "bulk" in out:
        out.setdefault("reference", {})["density_g_cm3"] = ref_d
        out["reference"]["density_rel"] = (out["bulk"]["density_g_cm3"] - ref_d) / ref_d
    perf = {stage: log_performance(folder / f"log.{stage}") for stage in ("bulk", "slab")}
    out["performance_ns_day"] = {k: (v[-1] if v else None) for k, v in perf.items()}
    if (cfg.get("dispersion") or "").startswith("cutoff"):
        notes.append(f"plain LJ cutoff at {out['cutoff_A']:g} Å without a tail correction in the slab: σ is biased "
                     "low (Ismail et al. 2007: up to 20 % at 12 Å for alkanes); compare with the ewald variant")
    out["checks"] = checks
    out["notes"] = notes
    return out


def _jsonable(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return asdict(o)


def write_report(folder: str | Path, result: dict) -> tuple[Path, Path]:
    folder = Path(folder)
    js = folder / "surface_tension.json"
    js.write_text(json.dumps(result, indent=1, default=_jsonable) + "\n", encoding="utf-8")
    md = folder / "surface_tension.md"
    md.write_text(format_report(result), encoding="utf-8")
    return js, md


def _f(x, fmt: str = ".3f") -> str:
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else format(x, fmt)


def format_report(r: dict) -> str:
    s = r["sigma"]
    L = [f"# Surface tension: {r.get('molecule')} ({r.get('dispersion')})", "",
         f"- Force field: {r.get('forcefield')}",
         f"- {r.get('n_molecules')} molecules, T = {r.get('temperature_K')} K, cutoff {r.get('cutoff_A')} Å, box "
         f"{' x '.join(_f(v, '.2f') for v in r['box_A'])} Å",
         f"- Production: {s['production_ns']:.3g} ns after skipping {r['skip_ns']:.3g} ns; "
         f"{s['n_samples']} samples every {s['sample_every_fs']:g} fs", "",
         "| quantity | value |", "|---|---|",
         f"| **σ (mN/m)** | **{s['mean_mN_m']:.2f} ± {_f(s['sem_mN_m'], '.2f')}** ({s['n_blocks']} blocks of "
         f"{s['block_ns']:.3g} ns) |",
         f"| SEM from the statistical inefficiency | {_f(s['sem_from_inefficiency_mN_m'], '.2f')} (g = "
         f"{_f(s['statistical_inefficiency'], '.1f')}, τ = {_f(s['correlation_time_ps'], '.2f')} ps) |",
         f"| first / second half | {s['first_half_mN_m']:.2f} / {s['second_half_mN_m']:.2f} |",
         f"| ⟨Pxx⟩, ⟨Pyy⟩, ⟨Pzz⟩ (atm) | {s['pressure_atm']['Pxx']:.1f}, {s['pressure_atm']['Pyy']:.1f}, "
         f"{s['pressure_atm']['Pzz']:.1f} |"]
    b = r.get("bulk")
    if b:
        L.append(f"| bulk NPT density (g/cm³) | {b['density_g_cm3']:.4f} ± {_f(b['sem_g_cm3'], '.4f')} |")
    d = r.get("density")
    if d:
        if "plateau_g_cm3" in d:
            L.append(f"| slab plateau density (g/cm³) | {d['plateau_g_cm3']:.4f} ({_f(d.get('plateau_vs_bulk'), '+.2%')} "
                     f"vs bulk; plateau {d['plateau_width_A']:.1f} Å wide) |")
        if "vapour_g_cm3" in d:
            L.append(f"| vapour density (g/cm³) | {d['vapour_g_cm3']:.2e} |")
        if "tanh_fit" in d:
            t = d["tanh_fit"]
            L.append(f"| tanh fit: ρl, z0, 10-90 width | {t['rho_liquid']:.4f} g/cm³, {t['z0']:.1f} Å, "
                     f"{t['width_10_90']:.1f} Å |")
        L.append(f"| slab drift (Å) | {d['centre_drift_A']:+.2f} (max {d['centre_max_excursion_A']:.2f}) |")
    ref = r.get("reference")
    if ref and "sigma_mN_m" in ref:
        L.append(f"| reference σ (mN/m) | {ref['sigma_mN_m']} ({ref['sigma_rel']:+.1%}) {ref.get('note', '')} |")
    if ref and "density_g_cm3" in ref:
        L.append(f"| reference density (g/cm³) | {ref['density_g_cm3']} (bulk {ref['density_rel']:+.1%}) |")
    perf = r.get("performance_ns_day", {})
    if any(perf.values()):
        L.append(f"| speed (ns/day) | bulk {_f(perf.get('bulk'), '.2f')}, slab {_f(perf.get('slab'), '.2f')} |")
    L += ["", "## Checks", ""]
    L += [f"- {'ok' if c['ok'] else '**check**'}: {c['check']}: {c['detail']}" for c in r.get("checks", [])]
    if r.get("notes"):
        L += ["", "## Notes", ""] + [f"- {n}" for n in r["notes"]]
    L += ["", f"σ = (Lz/2)·⟨Pzz − (Pxx+Pyy)/2⟩ (Kirkwood & Buff 1949); 1 atm·Å = {ATM_A_TO_MN_M} mN/m. "
          f"{r['generator']}, {r['analyzed']}.", ""]
    return "\n".join(L)
