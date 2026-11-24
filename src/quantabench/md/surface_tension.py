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

LJ tail (cutoff variants): a slab run with a plain LJ cutoff misses the dispersion beyond rc, and LAMMPS' tail
correction assumes a uniform density, so the slab runs without it. ``lj_tail_sigma`` adds it afterwards as
Tsige & Grest (J. Phys. Chem. C 2008, 112, 5029) did, with the mean-field result for a tanh profile
(Blokhuis et al., Mol. Phys. 1995, 85, 665; Chapela et al., J. Chem. Soc. Faraday Trans. 2 1977, 73, 1133):

    σ_tail = 12π Σᵢⱼ εᵢⱼ σᵢⱼ⁶ Δρᵢ Δρⱼ ∫₀¹ ds ∫_rc^∞ dr coth(rs/w) (3s³ − s) r⁻³

over the LJ site types i, j (Δρᵢ: liquid minus vapour number density of type i, from the fitted plateau), with
w = d/2 for the fitted ρ ∝ tanh(2(z − z0)/d), i.e. ρ ∝ tanh((z − z0)/w). A sharp interface (w → 0) gives the
upper bound (3π/2) Σ εσ⁶ΔρΔρ / rc², which is reported too. The correction does not undo the slab's own lower
density (a truncated liquid is less cohesive than the bulk run, which had the uniform tail), so a corrected cutoff σ
is a check against the full-dispersion (ewald) variant, not a replacement for it.
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
#: 1 kcal/mol/Å² in mN/m: 4184 J / N_A / 1e-20 m²
KCAL_MOL_A2_TO_MN_M = 4184.0 / 6.02214076e23 / 1e-20 * 1e3
AVOGADRO = 6.02214076e23


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


def merge_steps(a: np.ndarray) -> np.ndarray:
    """Rows merged by their step (first column): the last occurrence of a step wins, in step order. A file appended
    to by resumed segments (SIM-1) has no repeats once the supervisor cut it, but a hand-made resume may."""
    if a.ndim != 2 or len(a) < 2:
        return a
    steps = a[:, 0]
    if np.all(np.diff(steps) > 0):
        return a
    _u, last = np.unique(steps[::-1], return_index=True)
    return a[len(a) - 1 - last]


def read_pressure_tensor(path: str | Path) -> np.ndarray:
    """Rows (step, Pxx, Pyy, Pzz) of ``pressure_tensor.txt`` (atm), merged by step."""
    a = np.loadtxt(path, comments="#", ndmin=2)
    if a.shape[1] < 4:
        raise ValueError(f"{path}: expected 4 columns (step, Pxx, Pyy, Pzz)")
    return merge_steps(a[:, :4])


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
    st, bl = np.array(steps[:n]), np.array(blocks[:n])
    if n > 1 and not np.all(np.diff(st) > 0):            # merged by step, the last block of a step wins
        _u, last = np.unique(st[::-1], return_index=True)
        keep = n - 1 - last
        st, bl = st[keep], bl[keep]
    return st, z, bl


def read_bulk_density(path: str | Path) -> np.ndarray:
    """Rows (step, density g/cm³, volume Å³, lx, ly, lz, pe) of ``bulk_density.txt`` (none yet: shape (0, 0))."""
    with warnings.catch_warnings():          # a run shorter than the first 1-ps average leaves only the header
        warnings.simplefilter("ignore", UserWarning)
        return merge_steps(np.loadtxt(path, comments="#", ndmin=2))


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


# ------------------------------------------------------------------------------------------- the LJ tail
@dataclass
class LJType:
    """One LAMMPS atom type of the run: its LJ ε (kcal/mol) and σ (Å), mass (g/mol) and number of atoms."""
    index: int
    eps: float
    sigma: float
    mass: float
    count: int
    label: str = ""


def tail_integral(rc: float, w: float, n: int = 96) -> float:
    """I(rc, w) = ∫₀¹ ds ∫_rc^∞ dr coth(rs/w) (3s³ − s) / r³ (Å⁻²), for ρ ∝ tanh((z − z0)/w); w = 0: the sharp
    interface, 1/(8 rc²). With t = rc/r the integrand is smooth on (0, 1]², so Gauss-Legendre in s and t suffices."""
    if rc <= 0:
        raise ValueError("the cutoff must be positive")
    if w <= 0:
        return 1.0 / (8.0 * rc * rc)
    x, wq = np.polynomial.legendre.leggauss(n)
    u, wu = 0.5 * (x + 1.0), 0.5 * wq
    s, t = np.meshgrid(u, u, indexing="ij")
    coth = 1.0 / np.tanh(rc * s / (w * t))
    f = t * coth * (3.0 * s**3 - s)
    return float((wu[:, None] * wu[None, :] * f).sum() / (rc * rc))


def _mixed(a: LJType, b: LJType, mixing: str) -> tuple[float, float]:
    eps = math.sqrt(a.eps * b.eps)
    sig = 0.5 * (a.sigma + b.sigma) if mixing == "arithmetic" else math.sqrt(a.sigma * b.sigma)
    return eps, sig


def lj_tail_sigma(types: list[LJType], rho_liquid: float, rho_vapour: float, rc: float, w: float, *,
                  mixing: str = "geometric", pairs: dict[tuple[int, int], tuple[float, float]] | None = None) -> float:
    """The LJ surface tension beyond the cutoff ``rc`` (Å), in mN/m, for a pure liquid whose atoms are ``types``
    (counts over the whole system: only their ratios matter), at liquid and vapour mass densities ``rho_liquid`` and
    ``rho_vapour`` (g/cm³) and tanh width ``w`` (Å, ρ ∝ tanh((z − z0)/w)). ``pairs``: explicit (i, j) -> (ε, σ)
    coefficients that replace the mixing rule. See the module docstring."""
    mass = sum(t.count * t.mass for t in types)
    if mass <= 0 or rho_liquid <= rho_vapour:
        return 0.0
    per_atom = (rho_liquid - max(rho_vapour, 0.0)) * AVOGADRO * 1e-24 / mass   # Δρ of a type, per atom of it, Å⁻³
    total = 0.0
    for a in types:
        for b in types:
            key = (min(a.index, b.index), max(a.index, b.index))
            eps, sig = (pairs or {}).get(key) or _mixed(a, b, mixing)
            total += eps * sig**6 * (a.count * per_atom) * (b.count * per_atom)
    return 12.0 * math.pi * total * tail_integral(rc, w) * KCAL_MOL_A2_TO_MN_M


def _data_types(path: Path) -> tuple[dict[int, float], dict[int, int]]:
    """Masses and atom counts per type from a LAMMPS data file (Atoms styles full, molecular, charge, atomic)."""
    masses: dict[int, float] = {}
    counts: dict[int, int] = {}
    section, col = "", 2
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.split("#", 1)[0].strip()
        head = line.strip()
        if not text:
            continue
        word = text.split()[0]
        if word.isalpha() or word[0].isalpha():
            section = word
            if section == "Atoms":
                style = head.split("#", 1)[1].strip().split()[0] if "#" in head else "full"
                col = 1 if style in ("atomic", "charge") else 2
            continue
        tok = text.split()
        if section == "Masses" and len(tok) >= 2:
            masses[int(tok[0])] = float(tok[1])
        elif section == "Atoms" and len(tok) > col:
            t = int(tok[col])
            counts[t] = counts.get(t, 0) + 1
    return masses, counts


def read_lj_types(folder: str | Path) -> tuple[list[LJType], str, dict[tuple[int, int], tuple[float, float]]] | None:
    """The LJ types of a slab folder: ``pair_coeff`` lines of ``ff.params``, the mixing rule of ``in.slab``
    (LAMMPS' default: geometric) and the masses and counts of ``bulk_final.data`` (else ``system.data``).
    None when something is missing or the pair style is not plain 12-6 LJ."""
    folder = Path(folder)
    ff, data = folder / "ff.params", next((folder / n for n in ("bulk_final.data", "system.data")
                                           if (folder / n).is_file()), None)
    if not ff.is_file() or data is None:
        return None
    slab = folder / "in.slab"
    mixing = "geometric"
    if slab.is_file():
        text = slab.read_text(encoding="utf-8")
        style = re.search(r"^\s*pair_style\s+(\S+)", text, re.M)
        if style and not style.group(1).startswith("lj/cut"):
            return None
        mix = re.search(r"^\s*pair_modify\s.*\bmix\s+(geometric|arithmetic|sixthpower)", text, re.M)
        if mix:
            mixing = mix.group(1)
    if mixing == "sixthpower":
        return None
    own: dict[int, tuple[float, float, str]] = {}
    pairs: dict[tuple[int, int], tuple[float, float]] = {}
    for line in ff.read_text(encoding="utf-8").splitlines():
        body, _, comment = line.partition("#")
        tok = body.split()
        if len(tok) >= 5 and tok[0] == "pair_coeff":
            try:
                i, j, eps, sig = int(tok[1]), int(tok[2]), float(tok[3]), float(tok[4])
            except ValueError:
                return None                      # wildcards or another style's coefficients
            if i == j:
                own[i] = (eps, sig, comment.strip())
            else:
                pairs[(min(i, j), max(i, j))] = (eps, sig)
    masses, counts = _data_types(data)
    if not own or not counts or any(t not in own or t not in masses for t in counts):
        return None
    types = [LJType(t, own[t][0], own[t][1], masses[t], n, own[t][2]) for t, n in sorted(counts.items())]
    return types, mixing, pairs


# ------------------------------------------------------------------------------------------------- analysis
def _segments(folder: Path) -> list[dict]:
    """The segments of the folder's run.json (SIM-1): part, machine, variant, threads, steps, ns/day, exit."""
    p = folder / "run.json"
    try:
        run = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    keys = ("n", "part", "machine", "variant", "threads", "first_step", "last_step", "ns", "ns_per_day", "status")
    return [{k: s.get(k) for k in keys} for s in run.get("segments", [])]


def _setup(folder: Path) -> dict:
    p = folder / "liquid_setup.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def sim_engine(folder: Path) -> str:
    """The engine of a run folder's run.json ("lammps" when there is none)."""
    try:
        return json.loads((Path(folder) / "run.json").read_text(encoding="utf-8")).get("engine") or "lammps"
    except (OSError, ValueError):
        return "lammps"


def _ta_every_fs(folder: Path, dt: float) -> float:
    cfg = _setup(folder).get("config", {})
    return float(cfg.get("ta_every") or 1) * dt


def _sigma_stats(steps: np.ndarray, sig: np.ndarray, dt: float, skip_steps: int, block_ns: float, min_blocks: int,
                 notes: list[str]) -> dict:
    """Block average, statistical inefficiency and blocking curve of a σ series (steps, mN/m) after ``skip_steps``."""
    keep = steps >= skip_steps
    st, x = steps[keep], sig[keep]
    if len(x) < 8:
        raise ValueError(f"only {len(x)} σ samples after the first {skip_steps * dt / 1e6:g} ns: the production run "
                         "has not started (or use --skip-ns)")
    every = float(np.median(np.diff(st))) if len(st) > 1 else 1.0
    dt_sample_ns = every * dt / 1e6
    span_ns = (st[-1] - st[0] + every) * dt / 1e6
    per_block = int(round(block_ns / dt_sample_ns))
    nb = len(x) // max(1, per_block)
    if nb < min_blocks:
        per_block = max(1, len(x) // min_blocks)
        notes.append(f"production is {span_ns:.3g} ns: fewer than {min_blocks} blocks of {block_ns:g} ns, so "
                     f"{min_blocks} blocks of {per_block * dt_sample_ns:.3g} ns were used (a rough error bar)")
    ba = block_average(x, per_block)
    g = statistical_inefficiency(x)
    sem_g = float(x.std(ddof=1) * math.sqrt(g / len(x)))
    curve = blocking_curve(x)
    half = len(x) // 2
    out = {"mean_mN_m": ba.mean, "sem_mN_m": ba.sem, "n_blocks": ba.n_blocks,
           "block_ns": per_block * dt_sample_ns, "block_means_mN_m": ba.block_means,
           "statistical_inefficiency": g, "correlation_time_ps": 0.5 * (g - 1.0) * dt_sample_ns * 1e3,
           "sem_from_inefficiency_mN_m": sem_g, "n_samples": int(len(x)), "sample_every_fs": every * dt,
           "production_ns": span_ns, "first_half_mN_m": float(x[:half].mean()), "second_half_mN_m": float(x[half:].mean()),
           "blocking_curve": [{"block_ps": s * dt_sample_ns * 1e3, "sem_mN_m": e, "sem_err": u} for s, e, u in curve]}
    settle = sig[steps < skip_steps]
    if len(settle):
        out["settle_mean_mN_m"] = float(settle.mean())
    return out


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
    openmm = (sim_engine(folder) == "openmm")
    out: dict = {"folder": str(folder), "generator": f"QuantaBench {__version__}",
                 "analyzed": datetime.now().isoformat(timespec="seconds"),
                 "molecule": setup.get("molecule", {}).get("name"), "smiles": setup.get("molecule", {}).get("smiles"),
                 "forcefield": setup.get("molecule", {}).get("label"), "dispersion": cfg.get("dispersion"),
                 "temperature_K": box.get("T", cfg.get("temperature")), "cutoff_A": box.get("rc", cfg.get("cutoff")),
                 "n_molecules": setup.get("n_molecules"), "box_A": [box.get("lx"), box.get("ly"), lz],
                 "bulk_lz_A": box.get("bulk_lz"), "timestep_fs": dt, "skip_ns": skip_steps * dt / 1e6}

    # --- σ from the pressure tensor (an OpenMM run: its rerun, and its on-the-fly samples; OPENMM-1) ---
    pt_path = folder / "pressure_tensor.txt"
    kb = None
    if (folder / "sigma_kb.txt").is_file():
        from . import openmm_sigma as OS

        kb = OS.read_kb(folder)
    rerun_stats = None
    p = None
    if pt_path.is_file():
        p = read_pressure_tensor(pt_path)
        if (p[:, 0] >= skip_steps).sum() >= 8 or kb is None:
            rerun_stats = _sigma_stats(p[:, 0], sigma_series(p, lz), dt, skip_steps, block_ns, min_blocks, notes)
            prod = p[p[:, 0] >= skip_steps]
            rerun_stats["pressure_atm"] = {"Pxx": float(prod[:, 1].mean()), "Pyy": float(prod[:, 2].mean()),
                                           "Pzz": float(prod[:, 3].mean())}
    elif kb is None:
        raise FileNotFoundError(f"{pt_path}: no pressure tensor (an OpenMM run: md surface-tension reruns its frames)")
    kb_stats = None
    if kb is not None and (kb[:, 0] >= skip_steps).sum() >= 8:
        kb_stats = _sigma_stats(kb[:, 0], kb[:, 1], dt, skip_steps, block_ns, min_blocks, [])
        kb_stats["source"] = "on the fly, the run's own engine (sigma_kb.txt)"
        if rerun_stats is not None:
            from . import openmm_sigma as OS

            rs = {int(s): v for s, v in zip(p[:, 0], sigma_series(p, lz))}
            kb_stats["check"] = OS.compare_series(kb, rs, skip_steps)
    if openmm:
        out["engine"] = "openmm"
    if kb_stats is not None and (rerun_stats is None or (kb_stats.get("check") or {}).get("ok")):
        main = kb_stats
        if rerun_stats is not None:
            out["sigma_rerun"] = {**rerun_stats, "source": "rerun of the saved frames in double precision"}
            if "pressure_atm" in rerun_stats:
                main["pressure_atm"] = rerun_stats["pressure_atm"]
            chk = kb_stats["check"]
            checks.append({"check": "on-the-fly sigma vs the double-precision rerun", "ok": True,
                           "detail": f"{chk['mean_diff_mN_m']:+.3f} ± {chk['sem_diff_mN_m']:.3f} mN/m over "
                                     f"{chk['n']} frames (rms {chk['rms_diff_mN_m']:.2f})"})
        else:
            notes.append("σ from the run's on-the-fly samples, not yet checked against a double-precision rerun of "
                         "its frames (md surface-tension reruns them)")
    elif rerun_stats is not None:
        main = rerun_stats
        if openmm:
            main["source"] = "rerun of the saved frames in double precision (pressure_tensor.txt)"
        if kb_stats is not None:
            out["sigma_on_the_fly"] = kb_stats
            chk = kb_stats.get("check")
            if chk:
                checks.append({"check": "on-the-fly sigma vs the double-precision rerun", "ok": False,
                               "detail": f"{chk['mean_diff_mN_m']:+.3f} ± {chk['sem_diff_mN_m']:.3f} mN/m over "
                                         f"{chk['n']} frames: the on-the-fly samples are biased; σ is the rerun's"})
    else:
        raise ValueError(f"fewer than 8 σ samples after the first {skip_steps * dt / 1e6:g} ns: the production run "
                         "has not started (or use --skip-ns)")
    out["sigma"] = main
    ba_mean, ba_sem = main["mean_mN_m"], main["sem_mN_m"]
    if main["n_blocks"] >= 2 and abs(main["first_half_mN_m"] - main["second_half_mN_m"]) > 3 * max(ba_sem, 1e-9) * math.sqrt(2):
        checks.append({"check": "sigma drift", "ok": False,
                       "detail": f"first half {main['first_half_mN_m']:.2f}, second half "
                                 f"{main['second_half_mN_m']:.2f} mN/m: not converged or not equilibrated"})
    min_ns = 7.0
    checks.append({"check": "production length", "ok": main["production_ns"] >= min_ns,
                   "detail": f"{main['production_ns']:.3g} ns (Tsige & Grest 2008 needed at least 7 ns for "
                             "perfluoroalkanes)"})
    if openmm and (folder / "test_area.txt").is_file():
        from . import openmm_sigma as OS

        ta = OS.test_area_sigma(folder, float(out.get("temperature_K") or 298.15), skip_steps,
                                block=max(1, int(main["n_samples"] / max(1, main["n_blocks"]) * main["sample_every_fs"]
                                                 / max(1.0, _ta_every_fs(folder, dt)))))
        if ta:
            out["sigma_test_area"] = ta

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
                            "sigma_diff_mN_m": ba_mean - ref_s, "sigma_rel": (ba_mean - ref_s) / ref_s}
    if ref_d and "bulk" in out:
        out.setdefault("reference", {})["density_g_cm3"] = ref_d
        out["reference"]["density_rel"] = (out["bulk"]["density_g_cm3"] - ref_d) / ref_d
    perf = {stage: log_performance(folder / f"log.{stage}") for stage in ("bulk", "slab")}
    out["performance_ns_day"] = {k: (v[-1] if v else None) for k, v in perf.items()}
    segs = _segments(folder)
    if segs:             # a run of segments (SIM-1): each one's speed, and the stage's from its longest segment
        out["segments"] = segs
        for stage in ("bulk", "slab"):
            mine = [s for s in segs if s["part"] == stage and s.get("ns_per_day")]
            if mine:
                out["performance_ns_day"][stage] = max(mine, key=lambda s: s.get("ns") or 0.0)["ns_per_day"]
    if (cfg.get("dispersion") or "").startswith("cutoff"):
        fit = out.get("density", {}).get("tanh_fit")
        lj = read_lj_types(folder) if fit and out.get("cutoff_A") else None
        if lj:
            types, mixing, pairs = lj
            rc, w = float(out["cutoff_A"]), 0.5 * fit["d"]
            rl, rv = fit["rho_liquid"], max(fit["rho_vapour"], 0.0)
            tail = lj_tail_sigma(types, rl, rv, rc, w, mixing=mixing, pairs=pairs)
            sharp = lj_tail_sigma(types, rl, rv, rc, 0.0, mixing=mixing, pairs=pairs)
            out["sigma"]["tail"] = {"mN_m": tail, "sharp_interface_mN_m": sharp, "w_A": w, "rc_A": rc,
                                    "mixing": mixing, "rho_liquid_g_cm3": rl, "rho_vapour_g_cm3": rv,
                                    "method": "mean-field LJ tail for a tanh profile (Blokhuis et al. 1995)"}
            out["sigma"]["corrected_mN_m"] = ba_mean + tail
            if ref_s:
                out["reference"]["sigma_corrected_rel"] = (ba_mean + tail - ref_s) / ref_s
            notes.append(f"plain LJ cutoff at {rc:g} Å, no tail correction in the slab: the LJ beyond rc adds "
                         f"{tail:.2f} mN/m (sharp interface {sharp:.2f}), computed afterwards from the fitted profile. "
                         "It does not undo the slab's lower density, so compare with the ewald variant")
        else:
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
         f"| first / second half | {s['first_half_mN_m']:.2f} / {s['second_half_mN_m']:.2f} |"]
    if s.get("pressure_atm"):
        L.append(f"| ⟨Pxx⟩, ⟨Pyy⟩, ⟨Pzz⟩ (atm) | {s['pressure_atm']['Pxx']:.1f}, {s['pressure_atm']['Pyy']:.1f}, "
                 f"{s['pressure_atm']['Pzz']:.1f} |")
    if r.get("engine") == "openmm":           # OPENMM-1: where σ came from, and the other estimates
        L.insert(L.index("|---|---|") + 1, f"| engine | OpenMM; σ: {s.get('source', '')} |")
        for key, label in (("sigma_rerun", "σ from the double-precision rerun of the frames"),
                           ("sigma_on_the_fly", "σ on the fly (not used: biased against the rerun)"),
                           ("sigma_test_area", "σ by test-area (second estimate)")):
            x = r.get(key)
            if x:
                L.append(f"| {label} | {x['mean_mN_m']:.2f} ± {_f(x['sem_mN_m'], '.2f')} ({x['n_samples']} samples) |")
        chk = s.get("check") or (r.get("sigma_on_the_fly") or {}).get("check")
        if chk:
            L.append(f"| on the fly − rerun, same frames | {chk['mean_diff_mN_m']:+.3f} ± {chk['sem_diff_mN_m']:.3f} mN/m "
                     f"({chk['n']} frames, rms {chk['rms_diff_mN_m']:.2f}) |")
        fr = r.get("frames") or {}
        if fr.get("thinned"):
            L.append(f"| saved frames (slab.dcd) | every {fr['frame_keep_every']}th kept after the check passed: "
                     f"{fr['frames_before']:,} → {fr['frames_after']:,} ({fr['bytes_before'] / 1e6:,.0f} → "
                     f"{fr['bytes_after'] / 1e6:,.0f} MB) |")
    if "tail" in s:
        t = s["tail"]
        L.insert(L.index("|---|---|") + 2,
                 f"| σ + LJ tail beyond {t['rc_A']:g} Å (mN/m) | {s['corrected_mN_m']:.2f} (tail +{t['mN_m']:.2f}, "
                 f"w = {t['w_A']:.1f} Å; sharp interface +{t['sharp_interface_mN_m']:.2f}) |")
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
        corr = f"; with the tail {ref['sigma_corrected_rel']:+.1%}" if "sigma_corrected_rel" in ref else ""
        L.append(f"| reference σ (mN/m) | {ref['sigma_mN_m']} ({ref['sigma_rel']:+.1%}{corr}) {ref.get('note', '')} |")
    if ref and "density_g_cm3" in ref:
        L.append(f"| reference density (g/cm³) | {ref['density_g_cm3']} (bulk {ref['density_rel']:+.1%}) |")
    perf = r.get("performance_ns_day", {})
    if any(perf.values()):
        L.append(f"| speed (ns/day) | bulk {_f(perf.get('bulk'), '.2f')}, slab {_f(perf.get('slab'), '.2f')} |")
    segs = [s for s in r.get("segments", []) if s.get("last_step") is not None]
    if len(segs) > 1:
        L.append("| segments | " + "; ".join(
            f"{s['part']} {s['first_step']:,}-{s['last_step']:,} {s.get('variant') or ''} x {s.get('threads')} on "
            f"{s.get('machine')}: {_f(s.get('ns_per_day'), '.2f')} ns/day" for s in segs) + " |")
    L += ["", "## Checks", ""]
    L += [f"- {'ok' if c['ok'] else '**check**'}: {c['check']}: {c['detail']}" for c in r.get("checks", [])]
    if r.get("notes"):
        L += ["", "## Notes", ""] + [f"- {n}" for n in r["notes"]]
    L += ["", f"σ = (Lz/2)·⟨Pzz − (Pxx+Pyy)/2⟩ (Kirkwood & Buff 1949); 1 atm·Å = {ATM_A_TO_MN_M} mN/m. "
          f"{r['generator']}, {r['analyzed']}.", ""]
    return "\n".join(L)
