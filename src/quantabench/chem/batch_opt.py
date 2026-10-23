"""Batched (GPU-ready) conformer refinement.

Why this module exists
----------------------
``pipeline._refine`` optimizes conformers one at a time with ASE's LBFGS. Each step is one
energy/force call on a 20-60 atom molecule, which keeps a GPU almost idle: the work per call is far
too small to fill the device, and the time goes to kernel launches and Python overhead. Stacking the
B conformers of a molecule into one ``[B, N, 3]`` tensor turns B small calls into one larger call
with the same number of launches, which is how GPU potentials (dxtb, AIMNet2) are meant to be used.

The contract is deliberately small so new engines plug in without touching the optimizers::

    energy_and_forces(positions[B, N, 3] in Å) -> (E[B] in eV, F[B, N, 3] in eV/Å)

Optimizers
----------
- ``fire_batched``: FIRE (Bitzek et al., Phys. Rev. Lett. 97, 170201, 2006) with the paper's
  parameters (N_min = 5, f_inc = 1.1, f_dec = 0.5, alpha_start = 0.1, f_alpha = 0.99) and ASE's
  time-step defaults (dt = 0.1, dt_max = 1.0, unit masses). Every system has its own dt and alpha.
- ``lbfgs_batched``: the same L-BFGS as ``ase.optimize.LBFGS`` (two-loop recursion, Nocedal & Wright
  Algorithm 7.4; memory 100, initial inverse Hessian 1/70 Å²/eV, no line search), vectorized over
  systems. One safeguard differs from ASE: curvature pairs with s·y <= 0 are skipped instead of
  producing a negative rho.

Both use the same convergence test as ASE (largest per-atom force norm < fmax, per system) and the
same step cap (no atom moves more than ``max_step`` = 0.2 Å per step; the whole step is scaled, so
its direction is kept). Converged systems are masked: they stop moving but stay in the batch, so the
tensor shapes stay fixed. Models that set ``supports_subset = True`` are additionally only evaluated
on the still-active systems once at least half of the evaluated batch has converged ("compaction"),
which saves work on long-tailed batches without re-shaping on every step.

L-BFGS is the default (``DEFAULT_ALGORITHM``): on the benchmark below FIRE needed 2.3x more steps
and stopped 3-10 meV higher on the flat torsional surface of a perfluoroalkyl chain, while batched
L-BFGS reproduces the sequential ASE path step for step, so batched and sequential results are
interchangeable. See ``BENCHMARK_NOTES``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .engines import BOHR_ANG, HARTREE_EV
from .optimize import Cancelled

if TYPE_CHECKING:  # torch is optional at import time
    import torch

    from .conformers import ConformerSet
    from .pipeline import BuildConfig, ConformerResult, MoleculeResult

EnergyForcesFn = Callable[..., tuple["torch.Tensor", "torch.Tensor"]]

BATCHED_METHODS = ("gfn1-xtb", "gfn2-xtb", "aimnet2")
DEFAULT_ALGORITHM = "lbfgs"
CPU_MAX_BATCH = 64
MAX_BATCH_CAP = 256

BENCHMARK_NOTES = """\
PFOA anion (C8F15O2-, 25 atoms), 8 MMFF94s conformers, GFN1-xTB via dxtb on CPU (float64),
fmax 0.03 eV/Å, 2 shared vCPUs, no GPU (2026-10-23):

    path                               wall    steps (sum / max)   max |dE| vs ASE LBFGS
    sequential ASE LBFGS (_refine)     42.9 s  297 / 45            0
    batched LBFGS (refine_batched)     17.9 s  297 / 45            9e-13 eV
    batched FIRE  (refine_batched)     36.9 s  679 / 92            1.0e-2 eV (FIRE stops higher)
    sequential ASE FIRE (_refine)      91.7 s  679 / 92            1.0e-2 eV

Batched L-BFGS takes exactly the steps ASE takes per conformer; the 2.4x CPU gain comes from fewer,
larger dxtb calls (batched eigh and einsum instead of eight Python-level SCF loops). On a GPU the
cost of one call is nearly flat in B until the device fills, so the wall time should approach
(max steps) x (one call) instead of (sum of steps) x (one call): at most 297/45 = 6.6x for this
case, more for larger ensembles. Verify on real hardware before quoting GPU numbers.
"""


# ============================================================================ optimizers
@dataclass
class BatchOptResult:
    """Outcome of a batched minimization. All tensors are on the optimizer's device."""

    positions: torch.Tensor     # [B, N, 3] Å, final geometries (the last evaluated ones)
    energies: torch.Tensor      # [B] eV at ``positions``
    forces: torch.Tensor        # [B, N, 3] eV/Å at ``positions``
    converged: torch.Tensor     # [B] bool
    failed: torch.Tensor        # [B] bool: the model returned a non-finite energy or force
    steps: torch.Tensor         # [B] int64, optimizer steps taken (ASE's ``nsteps``)
    fmax: torch.Tensor          # [B] eV/Å, largest per-atom force norm at ``positions``
    n_calls: int = 0            # batched energy/force calls
    n_system_evals: int = 0     # sum over calls of systems evaluated (the work actually done)


def _check(should_stop: Callable[[], bool] | None) -> None:
    if should_stop is not None and should_stop():
        raise Cancelled()


def _cap_step(dr: torch.Tensor, max_step: float) -> torch.Tensor:
    """Scale each system's step so that no atom moves more than ``max_step`` (ASE LBFGS convention)."""
    longest = dr.norm(dim=-1).amax(dim=-1)                       # [B]
    scale = (max_step / longest.clamp_min(1e-300)).clamp(max=1.0)
    return dr * scale[:, None, None]


def _evaluate(fn: EnergyForcesFn, positions: torch.Tensor, index: torch.Tensor | None):
    if index is None:
        out = fn(positions)
    else:
        out = fn(positions[index], index)
    e, f = out[0], out[1]
    return e.detach().to(positions).reshape(-1), f.detach().to(positions).reshape(-1, *positions.shape[1:])


def _minimize(fn: EnergyForcesFn, positions, fmax: float, max_steps: int,
              should_stop: Callable[[], bool] | None, stepper, compact: bool) -> BatchOptResult:
    """Shared driver: evaluate -> test convergence -> step, like ``ase.optimize.Optimizer.irun``.

    The positions reported for each system are always the last ones that were evaluated, so
    energies, forces and any model side outputs (charges) belong to the returned geometry.
    """
    import torch

    pos = positions.detach().clone()
    B = pos.shape[0]
    dev = pos.device
    converged = torch.zeros(B, dtype=torch.bool, device=dev)
    failed = torch.zeros(B, dtype=torch.bool, device=dev)
    steps = torch.zeros(B, dtype=torch.long, device=dev)
    energies = torch.full((B,), float("nan"), dtype=pos.dtype, device=dev)
    forces = torch.zeros_like(pos)
    fnorm = torch.full((B,), float("inf"), dtype=pos.dtype, device=dev)
    subset = compact and bool(getattr(fn, "supports_subset", False))
    eval_idx = torch.arange(B, device=dev)
    n_calls = n_sys = 0
    while True:
        _check(should_stop)
        full = eval_idx.numel() == B
        e, f = _evaluate(fn, pos, None if full else eval_idx)
        n_calls += 1
        n_sys += eval_idx.numel()
        energies[eval_idx] = e
        forces[eval_idx] = f
        bad = ~torch.isfinite(e) | ~torch.isfinite(f).flatten(1).all(dim=1)
        failed[eval_idx] |= bad
        fn_new = f.norm(dim=-1).amax(dim=-1)
        fnorm[eval_idx] = fn_new
        converged[eval_idx] |= (fn_new < fmax) & ~bad   # ASE uses "<"; latched: never un-converges
        active = ~converged & ~failed & (steps < max_steps)
        n_active = int(active.sum())
        if n_active == 0:
            break
        with torch.no_grad():
            dr = stepper(pos, torch.nan_to_num(forces), active, steps)
            pos[active] += dr[active]
        steps += active.long()
        if subset and n_active <= eval_idx.numel() // 2:
            eval_idx = active.nonzero().flatten()
    return BatchOptResult(pos, energies, forces, converged, failed, steps, fnorm, n_calls, n_sys)


def fire_batched(energy_and_forces: EnergyForcesFn, positions, fmax: float = 0.03, max_steps: int = 500,
                 should_stop: Callable[[], bool] | None = None, *, dt: float = 0.1, dt_max: float = 1.0,
                 max_step: float = 0.2, n_min: int = 5, f_inc: float = 1.1, f_dec: float = 0.5,
                 alpha_start: float = 0.1, f_alpha: float = 0.99, compact: bool = True) -> BatchOptResult:
    """Batched FIRE. Mirrors ``ase.optimize.FIRE`` per system, except that the 0.2 Å cap applies to
    the largest atomic displacement rather than to the norm of the whole 3N step (the whole-step norm
    makes FIRE needlessly slow for large molecules)."""
    import torch

    B = positions.shape[0]
    dev, dtype = positions.device, positions.dtype
    v = torch.zeros_like(positions)
    dts = torch.full((B,), dt, dtype=dtype, device=dev)
    alpha = torch.full((B,), alpha_start, dtype=dtype, device=dev)
    n_pos = torch.zeros(B, dtype=torch.long, device=dev)

    def step(pos, f, active, steps):
        nonlocal v, dts, alpha, n_pos
        first = steps == 0                      # ASE starts from v = 0 without judging the "uphill" test
        vf = (f * v).flatten(1).sum(1)
        f_norm = f.flatten(1).norm(dim=1).clamp_min(1e-300)
        v_norm = v.flatten(1).norm(dim=1)
        a = alpha[:, None, None]
        mixed = (1 - a) * v + a * f / f_norm[:, None, None] * v_norm[:, None, None]
        downhill = (vf > 0) & ~first
        uphill = ~downhill & ~first
        speed_up = downhill & (n_pos > n_min)
        v = torch.where(downhill[:, None, None], mixed, torch.zeros_like(v))
        dts = torch.where(speed_up, (dts * f_inc).clamp(max=dt_max), dts)
        alpha = torch.where(speed_up, alpha * f_alpha, alpha)
        n_pos = torch.where(downhill, n_pos + 1, torch.zeros_like(n_pos))
        dts = torch.where(uphill, dts * f_dec, dts)
        alpha = torch.where(uphill, torch.full_like(alpha, alpha_start), alpha)
        v = v + dts[:, None, None] * f
        mask = active[:, None, None]
        v = torch.where(mask, v, torch.zeros_like(v))   # frozen systems carry no momentum
        return _cap_step(dts[:, None, None] * v, max_step)

    return _minimize(energy_and_forces, positions, fmax, max_steps, should_stop, step, compact)


def lbfgs_batched(energy_and_forces: EnergyForcesFn, positions, fmax: float = 0.03, max_steps: int = 500,
                  should_stop: Callable[[], bool] | None = None, *, memory: int = 100, alpha: float = 70.0,
                  damping: float = 1.0, max_step: float = 0.2, compact: bool = True) -> BatchOptResult:
    """Batched L-BFGS with ASE's defaults (memory 100, H0 = 1/alpha with alpha = 70 eV/Å², no line search).

    Every active system steps on every iteration and masked systems contribute zero (s, y) pairs
    with rho = 0, so one shared history length is exact for all systems."""
    import torch

    B, N = positions.shape[:2]
    s_hist: list = []
    y_hist: list = []
    rho_hist: list = []
    state: dict[str, Any] = {"r0": None, "f0": None}

    def step(pos, f, active, steps):
        r = pos.reshape(B, 3 * N)
        g = f.reshape(B, 3 * N)
        if state["r0"] is not None:
            s0 = r - state["r0"]
            y0 = state["f0"] - g
            ys = (y0 * s0).sum(1)
            ok = active & torch.isfinite(ys) & (ys > 1e-12)
            rho0 = torch.where(ok, 1.0 / torch.where(ok, ys, torch.ones_like(ys)), torch.zeros_like(ys))
            s_hist.append(s0)
            y_hist.append(y0)
            rho_hist.append(rho0)
            if len(s_hist) > memory:
                s_hist.pop(0)
                y_hist.pop(0)
                rho_hist.pop(0)
        q = -g.clone()
        a = []
        for s, y, rho in zip(reversed(s_hist), reversed(y_hist), reversed(rho_hist)):
            ai = rho * (s * q).sum(1)
            q -= ai[:, None] * y
            a.append(ai)
        a.reverse()
        z = q / alpha
        for s, y, rho, ai in zip(s_hist, y_hist, rho_hist, a):
            b = rho * (y * z).sum(1)
            z += s * (ai - b)[:, None]
        p = -z.reshape(B, N, 3)
        dr = _cap_step(p, max_step) * damping
        dr = torch.where(active[:, None, None], dr, torch.zeros_like(dr))
        # the next (s, y) pair is measured from here; frozen systems give s = 0 and are skipped
        state["r0"] = r.clone()   # positions before this step (``r`` is a view that moves in place)
        state["f0"] = g.clone()
        return dr

    return _minimize(energy_and_forces, positions, fmax, max_steps, should_stop, step, compact)


def optimize_batched(energy_and_forces: EnergyForcesFn, positions, fmax: float = 0.03, max_steps: int = 500,
                     algorithm: str = DEFAULT_ALGORITHM, should_stop: Callable[[], bool] | None = None,
                     **kwargs) -> BatchOptResult:
    """Dispatch on ``algorithm`` ("lbfgs" or "fire"), the same names ``BuildConfig.optimizer`` uses."""
    if algorithm == "fire":
        return fire_batched(energy_and_forces, positions, fmax, max_steps, should_stop, **kwargs)
    if algorithm == "lbfgs":
        return lbfgs_batched(energy_and_forces, positions, fmax, max_steps, should_stop, **kwargs)
    raise ValueError(f"unknown batched optimizer {algorithm!r} (use 'lbfgs' or 'fire')")


# ============================================================================ models
class BatchedModel:
    """Energies and forces for B geometries of ONE molecule (same atoms, charge and multiplicity).

    Subclasses implement ``_compute(positions, index) -> (E[b] eV, F[b, N, 3] eV/Å, charges[b, N] | None)``.
    ``index`` (or None for the full batch) gives the rows of the original batch being evaluated, so
    models that keep per-system state (and the charge buffer below) stay aligned after compaction.
    """

    supports_subset = True
    provides_charges = False

    def __init__(self) -> None:
        self.charges: torch.Tensor | None = None   # [B, N] e, from each system's last evaluation

    def _compute(self, positions, index):  # pragma: no cover - abstract
        raise NotImplementedError

    def __call__(self, positions, index=None):
        e, f, q = self._compute(positions, index)
        if q is not None:
            q = q.detach()
            if index is None:
                self.charges = q.clone()
            else:
                if self.charges is None:
                    raise RuntimeError("the first evaluation of a batched model must cover the full batch")
                self.charges[index.to(self.charges.device)] = q.to(self.charges)
        return e, f


class HarmonicModel(BatchedModel):
    """Pure-torch test model: E = k/2 * sum |x - x0|^2 per system, minimum at ``reference``.

    With ``reference=None`` the minimum is the first full-batch geometry shifted by ``offset`` Å,
    which mirrors the toy calculator in tests/test_pipeline.py."""

    def __init__(self, reference=None, k: float = 2.0, offset: float = 0.0, charges=None):
        super().__init__()
        self.reference = reference
        self.k = float(k)
        self.offset = float(offset)
        self.fixed_charges = charges
        self.provides_charges = charges is not None

    def _compute(self, positions, index):
        import torch

        if self.reference is None:
            if index is not None:
                raise RuntimeError("HarmonicModel needs a full-batch first call to set its reference")
            self.reference = positions.detach().clone() + self.offset
        ref = torch.as_tensor(self.reference, dtype=positions.dtype, device=positions.device)
        if ref.ndim == 3 and index is not None:
            ref = ref[index]
        d = positions - ref
        e = 0.5 * self.k * (d**2).flatten(1).sum(1)
        q = None
        if self.fixed_charges is not None:
            q = torch.as_tensor(self.fixed_charges, dtype=positions.dtype, device=positions.device)
            q = q.expand(positions.shape[0], -1)
        return e, -self.k * d, q


class DxtbBatched(BatchedModel):
    """GFN1/GFN2-xTB through dxtb's conformer batch mode (``batch_mode=2``: same atoms, no padding).

    Units: dxtb works in Hartree and Bohr; we convert to eV and Å. Forces come from autograd of
    E.sum(): systems are independent, so d(sum E)/dR_b = dE_b/dR_b. As in ``engines.DxtbCalculator``
    the CPU runs in float64 and GPUs in float32. One dxtb Calculator is built (and cached) per batch
    size, which compaction keeps to about log2(B) sizes. dxtb's batch mode rejects B = 1, so a
    single system uses an unbatched Calculator. No charges are returned, matching the sequential
    ``DxtbCalculator`` (the pipeline adds GFN1 charges for the best conformer afterwards).
    """

    def __init__(self, numbers, method: str = "gfn1-xtb", charge: int = 0, multiplicity: int = 1,
                 device: str = "cpu"):
        super().__init__()
        import dxtb
        import torch

        self._dxtb = dxtb
        self.device = torch.device(device)
        self.dtype = torch.float64 if self.device.type == "cpu" else torch.float32
        self._par = dxtb.GFN2_XTB if method == "gfn2-xtb" else dxtb.GFN1_XTB
        self._opts: dict[str, Any] = {"verbosity": 0}
        if method == "gfn2-xtb":
            self._opts["int_driver"] = "libcint"
        self.charge = int(charge)
        self.spin = int(multiplicity) - 1
        self._numbers = torch.as_tensor(np.asarray(numbers), device=self.device)
        self._calcs: dict[int, Any] = {}

    def _calc(self, b: int):
        if b not in self._calcs:
            if b == 1:
                nums, opts = self._numbers, self._opts
            else:
                nums, opts = self._numbers.expand(b, -1).contiguous(), {**self._opts, "batch_mode": 2}
            self._calcs[b] = self._dxtb.Calculator(nums, self._par, opts=opts, device=self.device,
                                                   dtype=self.dtype)
        return self._calcs[b]

    def _compute(self, positions, index):
        import torch

        b = positions.shape[0]
        pos = (positions.detach() / BOHR_ANG).to(self.device, self.dtype)
        if b == 1:
            pos = pos[0]
        pos.requires_grad_(True)
        calc = self._calc(b)
        calc.reset()
        if b == 1:
            kwargs: dict[str, Any] = {"chrg": self.charge}
            if self.spin:
                kwargs["spin"] = self.spin
        else:
            kwargs = {"chrg": torch.full((b,), float(self.charge), device=self.device, dtype=self.dtype)}
            if self.spin:
                kwargs["spin"] = torch.full((b,), float(self.spin), device=self.device, dtype=self.dtype)
        energy = calc.get_energy(pos, **kwargs)
        (grad,) = torch.autograd.grad(energy.sum(), pos)
        e = energy.detach().reshape(b) * HARTREE_EV
        f = -grad.detach().reshape(b, -1, 3) * (HARTREE_EV / BOHR_ANG)
        return e, f, None


class AIMNet2Batched(BatchedModel):
    """AIMNet2 through ``AIMNet2Calculator``'s batched dict input.

    Input:  coord [b, N, 3] float32 Å, numbers [b, N] int, charge [b] float, mult [b] float.
    Output: energy [b] eV, forces [b, N, 3] eV/Å, charges [b, N] e (AIMNet2 predicts its own
    charges, so dipoles come for free). On CPU the calculator flattens the batch internally; on a GPU
    molecules up to ``nb_threshold`` (120) atoms keep the dense [b, N, N] neighbor path.
    """

    provides_charges = True

    def __init__(self, numbers, charge: int, multiplicity: int, calculator):
        super().__init__()
        import torch

        self.calc = calculator
        self.device = torch.device(getattr(calculator, "device", "cpu"))
        self._numbers = torch.as_tensor(np.asarray(numbers), dtype=torch.long, device=self.device)
        self.charge = float(charge)
        self.mult = float(multiplicity)

    def _compute(self, positions, index):
        import torch

        b, n = positions.shape[:2]
        data = {
            "coord": positions.detach().to(self.device, torch.float32),
            "numbers": self._numbers.expand(b, -1),
            "charge": torch.full((b,), self.charge, dtype=torch.float32, device=self.device),
            "mult": torch.full((b,), self.mult, dtype=torch.float32, device=self.device),
        }
        out = self.calc(data, forces=True)
        e = out["energy"].detach().reshape(b)
        f = out["forces"].detach().reshape(b, n, 3)
        q = out.get("charges")
        q = None if q is None else q.detach().reshape(b, n)
        return e, f, q


def dxtb_batched(numbers, method: str = "gfn1-xtb", charge: int = 0, multiplicity: int = 1,
                 device: str = "cpu") -> DxtbBatched:
    return DxtbBatched(numbers, method, charge, multiplicity, device)


def aimnet2_batched(model_name: str, numbers, charge: int = 0, multiplicity: int = 1, device: str = "cpu",
                    calculator=None) -> AIMNet2Batched:
    """``calculator`` injects an ``AIMNet2Calculator``-like object (tests); otherwise the cached one
    from ``engines`` is used, so the sequential and batched paths share loaded weights."""
    if calculator is None:
        from . import engines

        calculator = engines._aimnet_base(model_name, device)
    return AIMNet2Batched(numbers, charge, multiplicity, calculator)


def supports_batched(method: str) -> bool:
    """True when ``method`` has a batched adapter and its packages are importable."""
    if method not in BATCHED_METHODS:
        return False
    from . import engines

    return engines.available(method)[0]


def make_batched_model(key: str, numbers, charge: int = 0, multiplicity: int = 1, device: str = "cpu",
                       model: str = "aimnet2") -> BatchedModel:
    """Batched twin of ``engines.make_calculator`` (same checks, same arguments)."""
    from . import engines

    ok, why = engines.available(key)
    if not ok:
        raise RuntimeError(f"{engines.ENGINES[key].label} unavailable: {why}")
    ok, why = engines.supports(key, numbers)
    if not ok:
        raise RuntimeError(why)
    if key in ("gfn1-xtb", "gfn2-xtb"):
        return dxtb_batched(numbers, key, charge, multiplicity, device)
    if key == "aimnet2":
        return aimnet2_batched(model, numbers, charge, multiplicity, device)
    raise RuntimeError(f"{key} has no batched model")


# ============================================================================ batch sizing
def _orbitals(z: int) -> int:
    """Rough GFN basis size per atom (H: 1-2 s functions; 2nd row: s+p; heavier: s+p+d)."""
    return 2 if z <= 2 else 4 if z <= 10 else 9


def bytes_per_system(method: str, numbers, itemsize: int = 4) -> int:
    """Conservative memory estimate for one conformer's energy+forces evaluation.

    These are heuristics, not measurements: dxtb keeps O(10^3) nao x nao matrices alive through the
    SCF and its backward pass; AIMNet2 stores ~O(neighbors x features x layers) activations per atom.
    ``refine_batched`` halves the batch on an out-of-memory error, so an optimistic guess costs one
    retry, not a crash. Calibrate with ``torch.cuda.max_memory_allocated`` on a real GPU.
    """
    n = len(numbers)
    if method in ("gfn1-xtb", "gfn2-xtb"):
        nao = sum(_orbitals(int(z)) for z in numbers)
        return int(1000 * nao * nao * itemsize + 2_000_000)
    return int(n * min(n, 128) * 256 * 20 * itemsize + 2_000_000)


def estimate_max_batch(method: str, numbers, device: str, memory_fraction: float = 0.8,
                       cpu_default: int = CPU_MAX_BATCH, cap: int = MAX_BATCH_CAP) -> int:
    """How many conformers to evaluate at once. CUDA/ROCm: from free VRAM (``torch.cuda.mem_get_info``,
    which ROCm builds also implement). CPU and MPS: ``cpu_default``."""
    try:
        import torch

        dev = torch.device(device)
        if dev.type != "cuda" or not torch.cuda.is_available():
            return cpu_default
        free, _total = torch.cuda.mem_get_info(dev)
    except Exception:
        return cpu_default
    per = bytes_per_system(method, numbers, itemsize=4)
    return int(max(1, min(cap, free * memory_fraction // per)))


def _is_oom(exc: BaseException) -> bool:
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:  # noqa: S110 - torch missing or old: fall through to the message test
        pass
    return "out of memory" in str(exc).lower()


# ============================================================================ pipeline stage
def refine_batched(result: MoleculeResult, conf_set: ConformerSet, cfg: BuildConfig, device: str,
                   should_stop: Callable[[], bool], log: Callable[[str], None]) -> list[ConformerResult]:
    """Drop-in replacement for ``pipeline._refine`` that optimizes conformers in batches.

    Same contract: refines ``conf_set.conf_ids[:cfg.refine_top]``, writes the optimized coordinates
    back into ``conf_set.mol``, fills charges and dipoles when the model predicts charges, drops to
    the CPU (with a note) when the device fails, and raises ``Cancelled`` when ``should_stop()``.
    Methods without a batched adapter are handed to ``pipeline._refine``.
    """
    import torch

    from . import descriptors as desc
    from . import pipeline

    mol = conf_set.mol
    ids = conf_set.conf_ids[: max(1, cfg.refine_top)]
    ff = conf_set.energies_kcal[: len(ids)]
    if result.method == "mmff" or result.method not in BATCHED_METHODS:
        return pipeline._refine(result, conf_set, cfg, device, should_stop, log)
    numbers = [a.GetAtomicNum() for a in mol.GetAtoms()]
    masses = np.array([a.GetMass() for a in mol.GetAtoms()])
    start = np.stack([np.asarray(mol.GetConformer(cid).GetPositions()) for cid in ids])  # [B, N, 3] Å
    if should_stop():
        raise Cancelled()

    def run_chunk(rows: list[int], dev: str):
        """Optimize start[rows] on ``dev``; halve on out-of-memory. Returns [(row, ...), ...]."""
        size = len(rows)
        while True:
            try:
                out = []
                for k in range(0, len(rows), size):
                    part = rows[k:k + size]
                    model = make_batched_model(result.method, numbers, result.charge, result.multiplicity, dev)
                    tdev = torch.device(dev)
                    dtype = torch.float32 if tdev.type == "mps" else torch.float64  # MPS has no float64
                    x0 = torch.as_tensor(start[part], dtype=dtype, device=tdev)
                    res = optimize_batched(model, x0, cfg.fmax, cfg.max_steps, cfg.optimizer, should_stop)
                    q = model.charges
                    for j, row in enumerate(part):
                        out.append((row, res.positions[j].cpu().double().numpy(), float(res.energies[j]),
                                    bool(res.converged[j]), int(res.steps[j]), bool(res.failed[j]),
                                    None if q is None else q[j].cpu().double().numpy()))
                    log(f"  batched {cfg.optimizer.upper()} on {dev}: {len(part)} conformer(s), "
                        f"{res.n_calls} calls, {int(res.steps.max()) if len(part) else 0} steps max")
                return out
            except Cancelled:
                raise
            except Exception as exc:
                if not _is_oom(exc) or size == 1:
                    raise
                size = max(1, size // 2)
                if torch.device(dev).type == "cuda":
                    torch.cuda.empty_cache()
                log(f"  out of memory on {dev}; retrying with batches of {size}")

    max_batch = estimate_max_batch(result.method, numbers, device,
                                   memory_fraction=getattr(cfg, "gpu_memory_fraction", 0.8) or 0.8)
    rows_all = list(range(len(ids)))
    chunks = [rows_all[k:k + max_batch] for k in range(0, len(rows_all), max_batch)]
    done = []
    dev = device
    for chunk in chunks:
        try:
            done += run_chunk(chunk, dev)
        except Cancelled:
            raise
        except Exception as exc:
            if result.device == "cpu":
                raise
            result.notes.append(f"{result.method} failed on {result.device} ({type(exc).__name__}: {exc}); "
                                "fell back to CPU")
            log(result.notes[-1])
            result.device = dev = "cpu"
            done += run_chunk(chunk, dev)

    ConformerResult = pipeline.ConformerResult
    out: list[ConformerResult] = []
    n_failed = 0
    for row, pos, energy, conv, steps, bad, charges in sorted(done, key=lambda t: t[0]):
        cid, e_ff = ids[row], ff[row]
        if bad:
            n_failed += 1
            log(f"  conformer {row + 1}/{len(ids)}: non-finite energy or forces; dropped")
            continue
        conf = mol.GetConformer(cid)
        for j, p in enumerate(pos):
            conf.SetAtomPosition(j, p.tolist())
        q = None if charges is None else [float(x) for x in charges]
        dip = desc.dipole_debye(pos, np.asarray(q), masses) if q else None
        out.append(ConformerResult(rank=0, conf_id=cid, energy_ev=energy, ff_energy_kcal=e_ff,
                                   converged=conv, steps=steps, charges=q, dipole_debye=dip))
        log(f"  conformer {row + 1}/{len(ids)}: E = {energy:.5f} eV, {steps} steps"
            + ("" if conv else " (not converged)"))
    if n_failed:
        result.notes.append(f"{n_failed} conformer(s) gave non-finite {result.method} energies and were dropped")
    if not out:
        raise RuntimeError(f"{result.method} gave non-finite energies for every conformer")
    return out
