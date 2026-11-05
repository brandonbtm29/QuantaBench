"""dxtb's dense linear algebra on the CPU while the rest of the calculation stays on the GPU.

Why
---
- AMD's Windows ROCm PyTorch (2.9.1+rocm7.2.1, BTM NOVA's RX 9060 XT is ``cuda:0``) is built without
  MAGMA. Probed op by op on that GPU (float32 and float64): ``torch.linalg.cholesky``,
  ``torch.cholesky_solve`` and ``torch.linalg.lstsq`` raise ("requires compiling PyTorch with MAGMA");
  ``eigh``, ``solve``, ``solve_triangular``, ``inv``, ``lu_factor``, ``qr`` and ``svd`` work. dxtb needs a
  Cholesky factorization of the overlap matrix in every SCF diagonalization and in its EEQ starting
  guess, so stock dxtb fails on that GPU (and QuantaBench used to fall back to the CPU).
- Apple's MPS has no float64 and incomplete ``torch.linalg`` coverage, so everything is routed there.

What dxtb 0.4 calls (GFN1/GFN2, default "full" SCF; energies, autograd forces and Hessians)
------------------------------------------------------------------------------------------
- SCF diagonalization, every SCF step: ``tad_mctc.storch.eighb(H, S)`` (called from
  ``dxtb._src.scf.unrolling.base.diagonalize``): ``cholesky(S)``, ``solve(L, I)`` for L^-1, then
  ``eigh(L^-1 H L^-T)`` through tad-mctc's broadened ``_SymEigB`` autograd function (its backward is
  plain matmuls). Counted for PFOA: 14 per evaluation.
- EEQ starting guess, once per evaluation: ``tad_multicharge.get_eeq_charges``: ``cholesky`` and
  ``cholesky_solve`` of the nat x nat Coulomb matrix.
- Anderson mixer (``dxtb._src.scf.mixer.anderson``, the default since Broyden falls back to it),
  most SCF steps: ``torch.linalg.solve`` of an m x m system, m <= the mixing history.

All are ``torch.linalg``/``torch`` calls inside those wrappers, and autograd passes through every one
of them (forces: one backward; Hessians: ``create_graph=True`` and a second backward). No inverse,
SVD, QR or lstsq is called on this path. The implicit SCF modes (``scf_mode="implicit"``, xitorch's
``lsymeig``) add more cholesky/inverse calls; QuantaBench does not use them and they are not routed.

Two more places need float64 on the device, which MPS lacks (pseudo-op ``"float64"``, routed on MPS):
tad-dftd3's ``model.weight_references`` (GFN1's D3 dispersion) evaluates the CN weights in float64 on
purpose, and tad-mctc's ``data.radii.VDW_PAIRWISE`` loads its float64 table with
``map_location=device`` before casting. Both run on the CPU when routed.

How
---
``linalg_routing(device)`` patches only those wrapper functions (``_TARGETS``), and only while the
context is open (reference-counted, restored on exit). A patched function checks the device of its input and
passes it through untouched unless that device is being routed. A routed call copies its inputs to the
CPU (``.to()`` is differentiable), optionally upcasts float32 to float64, runs the original function
there, and copies the result back in the caller's dtype: the dtype cast happens on the CPU before the
copy, so no float64 tensor ever reaches the device (MPS). The autograd graph of the routed part lives on
the CPU, so its backward (and the double backward of a Hessian) runs there too.

The diagonalization is split at the narrowest point the device needs: on NOVA only the Cholesky
factorization of the overlap and the inverse of its factor run on the CPU, while the reduction
``C = L^-1 H L^-T``, the eigensolver and the back-transformation stay on the GPU, where batched ``eigh``
is 10-20x faster than this wheel's CPU LAPACK. When ``eigh`` is routed too (MPS, or ``cpu`` policy),
the whole ``eighb`` call runs on the CPU with one copy in and one copy out.

Policy (``QUANTABENCH_LINALG`` overrides ``set_policy``; ``apply_settings`` sets it from Settings/BuildConfig)
-------------------------------------------------------------------------------------------
- ``auto`` (default): CPU devices are never routed; ``mps`` routes everything; any other device is
  probed once per process (``probe``: float64 support, then a tiny batched cholesky / cholesky_solve /
  eigh / solve in float32 and float64, compared with the CPU) and the ops that raise or disagree are
  routed. NOVA: ``cholesky`` and ``cholesky_solve``.
- ``cpu``: route every op, on every device (including the CPU itself: used by tests).
- ``device``: never route (stock dxtb).

Which device dxtb runs on (``QUANTABENCH_DXTB_GPU`` overrides ``set_dxtb_gpu``)
------------------------------------------------------------------------------
Routing makes dxtb *run* on NOVA's GPU, but it is 5-8x slower there than on the CPU at every size and
batch size measured (``DXTB_GPU_BENCHMARK``). So ``dxtb_device`` (used by every dxtb entry point)
keeps dxtb on the CPU when offered a GPU that needs routing ('auto'); 'on' uses the GPU anyway (with
routing), 'off' always the CPU.
"""

from __future__ import annotations

import contextlib
import importlib
import inspect
import os
import threading
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

ENV_VAR = "QUANTABENCH_LINALG"
POLICIES = ("auto", "cpu", "device")
PROBE_OPS = ("cholesky", "cholesky_solve", "eigh", "solve", "float64")
ALL_OPS = frozenset(PROBE_OPS)
_ORTHO_OPS = frozenset({"cholesky", "solve"})              # the overlap factorization in eighb
_EEQ_OPS = frozenset({"cholesky", "cholesky_solve", "solve"})

_policy = "auto"
_lock = threading.RLock()
_tls = threading.local()
_probes: dict[str, LinalgSupport] = {}
_routes: dict[str, list[_Route]] = {}
_installed: list[tuple[object, str, object]] = []         # (module, attribute, original), for restoring
_install_count = 0
_status: dict[str, str] = {}                               # "module.attr" -> "patched" | why not
_stats: Counter[str] = Counter()


# ============================================================================ policy and devices
def set_policy(policy: str) -> None:
    """Default policy when ``QUANTABENCH_LINALG`` is not set ('auto' | 'cpu' | 'device')."""
    global _policy
    policy = (policy or "auto").strip().lower()
    if policy not in POLICIES:
        raise ValueError(f"linalg policy must be one of {POLICIES}, not {policy!r}")
    _policy = policy


def current_policy() -> str:
    env = os.environ.get(ENV_VAR, "").strip().lower()
    if env:
        if env not in POLICIES:
            raise ValueError(f"{ENV_VAR} must be one of {POLICIES}, not {env!r}")
        return env
    return _policy


def device_type(device) -> str:
    """'cpu' | 'cuda' | 'mps' | ... for a torch.device or a device string, without importing torch."""
    return str(getattr(device, "type", device)).split(":", 1)[0].strip().lower()


def device_key(device) -> str:
    """The device as tensors report it: 'cpu', 'cuda:0', 'mps:0' (an index is filled in)."""
    import torch

    dev = torch.device(device)
    if dev.type != "cpu" and dev.index is None:
        index = torch.cuda.current_device() if dev.type == "cuda" else 0
        dev = torch.device(dev.type, index)
    return str(dev)


def compute_dtype(device):
    """dtype QuantaBench runs dxtb in on ``device``: float64 on the CPU, float32 on GPUs (MPS has no
    float64, and consumer GPUs run float64 at 1/16-1/64 rate)."""
    import torch

    return torch.float64 if device_type(device) == "cpu" else torch.float32


def float64_device(device) -> str:
    """Where a calculation that needs float64 (Hessians) can run: ``device``, except MPS -> 'cpu'."""
    return "cpu" if device_type(device) == "mps" else str(device)


def apply_settings(source, strict: bool = True) -> None:
    """Set the linear-algebra policy and the dxtb GPU mode from ``source``: a ``BuildConfig``, a
    ``settings.ResourceSettings`` or a whole ``Settings`` (its ``resources``), anything with ``linalg`` and
    ``dxtb_gpu``. Entry points (CLI, GUI) call it once with ``strict=False`` (an invalid value, say from a
    hand-edited settings.json, means 'auto'), ``pipeline.run_molecule`` for each molecule run on a GPU
    (an invalid value raises). The environment variables still win (``current_policy``, ``dxtb_gpu_mode``)."""
    src = getattr(source, "resources", source)
    for setter, value in ((set_policy, getattr(src, "linalg", "auto")), (set_dxtb_gpu, getattr(src, "dxtb_gpu", "auto"))):
        try:
            setter(value or "auto")
        except (ValueError, AttributeError, TypeError):
            if strict:
                raise
            setter("auto")


# ============================================================================ capability probe
@dataclass(frozen=True)
class LinalgSupport:
    """Result of probing one device: the ops that raised or gave wrong results, with the reason."""

    device: str
    failed: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.failed

    @property
    def routed_ops(self) -> frozenset[str]:
        return frozenset(self.failed) & ALL_OPS

    def summary(self) -> str:
        if self.ok:
            return f"{self.device}: dense linear algebra OK"
        return f"{self.device}: " + "; ".join(f"{op} unavailable ({why})" for op, why in sorted(self.failed.items()))


def probe(device, refresh: bool = False) -> LinalgSupport:
    """Try each op in ``PROBE_OPS`` on a tiny batched SPD system on ``device``; cached per device."""
    key = device_key(device)
    with _lock:
        if not refresh and key in _probes:
            return _probes[key]
    support = _probe_now(key)
    with _lock:
        _probes[key] = support
    return support


def clear_probe_cache() -> None:
    with _lock:
        _probes.clear()


def _probe_now(key: str) -> LinalgSupport:
    import torch

    dev = torch.device(key)
    failed: dict[str, str] = {}
    try:
        float(torch.ones(2, dtype=torch.float64, device=dev).sum())
        dtypes = (torch.float32, torch.float64)
    except Exception as exc:
        failed["float64"] = f"{type(exc).__name__}: {_first_line(exc)}"
        dtypes = (torch.float32,)
    n = 24
    for dt in dtypes:
        gen = torch.Generator().manual_seed(0)
        m = torch.rand(2, n, n, generator=gen, dtype=dt)
        a_cpu = m @ m.transpose(-1, -2) + n * torch.eye(n, dtype=dt)
        rhs_cpu = torch.rand(2, n, 1, generator=gen, dtype=dt)
        with _routed_region():           # the CPU reference values
            l_cpu = torch.linalg.cholesky(a_cpu)
            refs = {
                "cholesky": (lambda a, r, ll: torch.linalg.cholesky(a), l_cpu),
                "cholesky_solve": (lambda a, r, ll: torch.cholesky_solve(r, ll), torch.cholesky_solve(rhs_cpu, l_cpu)),
                "eigh": (lambda a, r, ll: torch.linalg.eigh(a)[0], torch.linalg.eigh(a_cpu)[0]),
                "solve": (lambda a, r, ll: torch.linalg.solve(a, r), torch.linalg.solve(a_cpu, rhs_cpu)),
            }
        try:
            a, rhs, ll = a_cpu.to(dev), rhs_cpu.to(dev), l_cpu.to(dev)
        except Exception as exc:     # the device itself is unusable: route everything
            return LinalgSupport(key, {op: f"{type(exc).__name__}: {_first_line(exc)}" for op in PROBE_OPS})
        for op, (fn, ref) in refs.items():
            if op in failed:
                continue
            try:
                out = fn(a, rhs, ll).to("cpu")
                if not torch.allclose(out, ref, rtol=1e-3, atol=1e-3):
                    failed[op] = f"wrong result in {str(dt).replace('torch.', '')}"
            except Exception as exc:
                failed[op] = f"{type(exc).__name__}: {_first_line(exc)}"
    return LinalgSupport(key, failed)


def _first_line(exc: BaseException) -> str:
    text = str(exc).strip()
    return text.splitlines()[0][:200] if text else ""


def routed_ops(device, policy: str | None = None) -> frozenset[str]:
    """The ops ``linalg_routing`` sends to the CPU for ``device`` under ``policy`` (default: current)."""
    pol = policy or current_policy()
    if pol not in POLICIES:
        raise ValueError(f"linalg policy must be one of {POLICIES}, not {pol!r}")
    if pol == "device":
        return frozenset()
    if pol == "cpu":
        return ALL_OPS
    kind = device_type(device)
    if kind == "cpu":
        return frozenset()
    if kind == "mps":
        return ALL_OPS
    return probe(device).routed_ops


def describe(device) -> str:
    """One line for logs and `doctor`: what runs where for dxtb on ``device``."""
    ops = routed_ops(device)
    if not ops:
        return f"dxtb linear algebra on {device}"
    return f"dxtb linear algebra on {device}, except {', '.join(sorted(ops))} on the CPU ({current_policy()})"


# ============================================================================ which device dxtb uses
DXTB_GPU_ENV = "QUANTABENCH_DXTB_GPU"
DXTB_GPU_MODES = ("auto", "on", "off")
_dxtb_gpu = "auto"

DXTB_GPU_BENCHMARK = """\
BTM NOVA (RX 9060 XT 16 GB, Windows 11, torch 2.9.1+rocm7.2.1, dxtb 0.4; CPU: Ryzen 9 7900X, 12 threads
of the same wheel's slow CPU BLAS), GFN1-xTB energy + autograd forces, warm, seconds per evaluation
(cholesky/cholesky_solve routed to the CPU; routing eigh and solve as well changes nothing):

    system                       AOs   CPU f64   cuda:0 f32   GPU/CPU
    water                          8    0.05       0.24        4.8x
    CF3COOH                       30    0.13       0.78        6.1x
    PFOA anion                   100    0.29       1.87        6.5x
    C12 perfluorocarboxylate     148    0.47       3.93        8.4x
    C20 perfluorocarboxylate     244    1.29       9.98        7.8x
    PFOA anion, batch of 4       100    1.01       7.23        7.2x
    PFOA anion, batch of 8       100    2.00      13.78        6.9x
    PFOA anion, batch of 20      100    4.94      33.68        6.8x

The GPU time is not the routed linear algebra: 74% of it is ~11,000 host synchronizations per
evaluation (``.item()``, ~150 us each on this driver) in dxtb's PyTorch overlap integrals
(OverlapAG forward + backward: ~1.9 s of 2.1 s for PFOA), which dxtb 0.4 also evaluates once per
system in a Python loop in batch mode, so a batch costs B single evaluations. The SCF itself is
batched and does gain (0.17 s for 1 and 0.20 s for 4 conformers on the GPU, 0.11 s and 0.36 s on
the CPU), but not enough to matter. There is no batch size at which the GPU wins (cost ~linear in B).
"""


def set_dxtb_gpu(mode: str) -> None:
    """Default for ``dxtb_device`` when ``QUANTABENCH_DXTB_GPU`` is not set ('auto' | 'on' | 'off')."""
    global _dxtb_gpu
    mode = (mode or "auto").strip().lower()
    if mode not in DXTB_GPU_MODES:
        raise ValueError(f"dxtb GPU mode must be one of {DXTB_GPU_MODES}, not {mode!r}")
    _dxtb_gpu = mode


def dxtb_gpu_mode() -> str:
    env = os.environ.get(DXTB_GPU_ENV, "").strip().lower()
    if env:
        if env not in DXTB_GPU_MODES:
            raise ValueError(f"{DXTB_GPU_ENV} must be one of {DXTB_GPU_MODES}, not {env!r}")
        return env
    return _dxtb_gpu


def dxtb_device(device) -> tuple[str, str]:
    """(device dxtb should run on when it is offered ``device``, why that differs; '' if it doesn't).

    'auto' keeps a GPU only when it has the dense linear algebra and float64 dxtb needs (CUDA with
    cuSOLVER, ROCm builds with MAGMA). On GPUs that need routing (AMD's Windows ROCm wheel, Apple MPS)
    dxtb runs on the CPU: measured 5-8x slower on NOVA's GPU at every size and batch size tried
    (``DXTB_GPU_BENCHMARK``); MPS is unmeasured and assumed similar. 'on' uses the GPU with routing
    (``linalg_routing``), 'off' always uses the CPU."""
    req = str(device)
    if device_type(req) == "cpu":
        return req, ""
    mode = dxtb_gpu_mode()
    if mode == "on":
        return req, ""
    if mode == "off":
        return "cpu", f"dxtb ran on the CPU instead of {req} ({DXTB_GPU_ENV}=off)"
    if device_type(req) == "mps":
        return "cpu", (f"dxtb ran on the CPU instead of {req}: MPS has no float64 and incomplete linear algebra "
                       f"({DXTB_GPU_ENV}=on tries the GPU with them routed to the CPU)")
    support = probe(req)
    if support.ok:
        return req, ""
    return "cpu", (f"dxtb ran on the CPU instead of {req}: {', '.join(sorted(support.routed_ops))} not available "
                   f"there, and with them on the CPU dxtb 0.4 measured 5-8x slower on such a GPU than on the CPU "
                   f"(BTM NOVA; {DXTB_GPU_ENV}=on uses the GPU anyway)")


# ============================================================================ routing context
@dataclass(frozen=True)
class _Route:
    ops: frozenset[str]
    upcast: bool


@contextlib.contextmanager
def linalg_routing(device, ops=None, upcast: bool = True) -> Iterator[frozenset[str]]:
    """Run dxtb's dense linear algebra for tensors on ``device`` on the CPU inside this block.

    ``ops``: the ops to route (subset of ``ALL_OPS``); default ``routed_ops(device)`` (policy + probe).
    ``upcast``: compute routed float32 ops in float64 and cast the results back.
    Yields the routed ops (empty: nothing is patched and dxtb runs unchanged)."""
    route_ops = routed_ops(device) if ops is None else frozenset(ops)
    unknown = route_ops - ALL_OPS
    if unknown:
        raise ValueError(f"unknown linalg ops {sorted(unknown)} (known: {sorted(ALL_OPS)})")
    if not route_ops:
        yield route_ops
        return
    key = device_key(device)
    route = _Route(route_ops, bool(upcast))
    with _lock:
        _install()
        _routes.setdefault(key, []).append(route)
    try:
        yield route_ops
    finally:
        with _lock:
            stack = _routes.get(key, [])
            for i in range(len(stack) - 1, -1, -1):
                if stack[i] is route:
                    del stack[i]
                    break
            if not stack:
                _routes.pop(key, None)
            _uninstall()


def in_routed_region() -> bool:
    """True while a routed op runs on the CPU (in this thread)."""
    return getattr(_tls, "depth", 0) > 0


@contextlib.contextmanager
def _routed_region() -> Iterator[None]:
    _tls.depth = getattr(_tls, "depth", 0) + 1
    try:
        yield
    finally:
        _tls.depth -= 1


def routing_stats() -> dict[str, int]:
    """Routed calls so far by kind ('eighb', 'eighb_ortho', 'eeq', 'solve'); for logs and benchmarks."""
    return dict(_stats)


def reset_stats() -> None:
    _stats.clear()


def patch_status() -> dict[str, str]:
    """What the last install did, per target: 'patched' or why it was skipped."""
    return dict(_status)


def _route_for(tensor) -> _Route | None:
    if not _routes:
        return None
    device = getattr(tensor, "device", None)
    if device is None:
        return None
    stack = _routes.get(str(device))
    return stack[-1] if stack else None


# ============================================================================ moving tensors
def _work_dtype(dtype, upcast: bool):
    import torch

    if upcast and dtype in (torch.float16, torch.bfloat16, torch.float32):
        return torch.float64
    return None


def _down(x, work):
    import torch

    if isinstance(x, torch.Tensor):
        x = x.to("cpu")
        if work is not None and x.is_floating_point():
            x = x.to(work)
    return x


def _up(x, device, dtype):
    import torch

    if isinstance(x, torch.Tensor):
        if x.is_floating_point() and x.dtype != dtype:
            x = x.to(dtype)          # cast on the CPU first: float64 never reaches the device
        return x.to(device)
    if isinstance(x, tuple):
        items = [_up(t, device, dtype) for t in x]
        if hasattr(x, "_fields"):                       # namedtuple
            return type(x)(*items)
        if type(x).__module__ == "torch.return_types":  # structseq (eigh, slogdet, ...)
            return type(x)(items)
        return tuple(items)
    if isinstance(x, list):
        return [_up(t, device, dtype) for t in x]
    return x


def _ref_tensor(args, kwargs):
    import torch

    for x in (*args, *kwargs.values()):
        if isinstance(x, torch.Tensor) and x.is_floating_point():
            return x
    return None


def _on_cpu(kind: str, fn: Callable, args: tuple, kwargs: dict, ref, upcast: bool):
    """Call ``fn`` with every tensor argument on the CPU, results back on ``ref``'s device and dtype."""
    work = _work_dtype(ref.dtype, upcast)
    _stats[kind] += 1
    with _routed_region():
        cargs = tuple(_down(x, work) for x in args)
        ckwargs = {k: _down(v, work) for k, v in kwargs.items()}
        out = fn(*cargs, **ckwargs)
        return _up(out, ref.device, ref.dtype)


# ============================================================================ wrappers
_EIGHB_PARAMS = ("a", "b", "scheme", "broadening_method", "factor", "sort_out", "aux", "is_posdef")


def _make_eighb(orig: Callable) -> Callable | str:
    """Routed twin of tad-mctc's ``eighb(a, b, scheme, broadening_method, factor, sort_out, aux,
    is_posdef, **kwargs)``; returns why not when the signature is not the one this was written for."""
    try:
        names = list(inspect.signature(orig).parameters)
    except (TypeError, ValueError):
        return "signature not inspectable"
    if tuple(names[: len(_EIGHB_PARAMS)]) != _EIGHB_PARAMS:
        return f"unexpected signature {names}"

    def _chol_inverse(b, direct_inv: bool):
        """L^-1 of the Cholesky factor of ``b``, computed as tad-mctc's 'chol' scheme does."""
        import torch

        low = torch.linalg.cholesky(b)
        if direct_inv:
            return torch.inverse(low)
        identity = torch.zeros_like(low)
        identity.diagonal(dim1=-2, dim2=-1)[:] = 1
        return torch.linalg.solve(low, identity)

    def eighb(a, b=None, scheme="chol", broadening_method="cond", factor=1e-12, sort_out=True, aux=True,
              is_posdef=False, **kwargs):
        route = _route_for(a)
        call = (a, b, scheme, broadening_method, factor, sort_out, aux, is_posdef)
        if route is None:
            return orig(*call, **kwargs)
        need_ortho = b is not None and scheme == "chol" and bool(route.ops & _ORTHO_OPS)
        if "eigh" in route.ops:                 # everything on the CPU: one copy in, one copy out
            return _on_cpu("eighb", orig, call, kwargs, a, route.upcast)
        if not need_ortho:
            return orig(*call, **kwargs)
        if not (is_posdef and aux):
            # tad-mctc then takes its padding mask from b, which the split below does not reproduce
            return _on_cpu("eighb", orig, call, kwargs, a, route.upcast)
        # Split (dxtb's call: generalized, Cholesky, is_posdef, aux): factorize the overlap on the CPU,
        # reduce and diagonalize on the device. The standard problem for C = L^-1 A L^-T is what
        # tad-mctc's 'chol' branch solves: padding rows of A (all zero) are zero rows of C, so the
        # padding mask, the shift of the padding eigenvalues and the sort-out are the same.
        l_inv = _on_cpu("eighb_ortho", _chol_inverse, (b, bool(kwargs.get("direct_inv", False))), {}, a,
                        route.upcast)
        l_inv_t = l_inv.transpose(-1, -2)
        c = l_inv @ a @ l_inv_t
        rest = {k: v for k, v in kwargs.items() if k != "direct_inv"}
        w, v_ = orig(c, None, scheme, broadening_method, factor, sort_out, aux, True, **rest)
        return w, l_inv_t @ v_

    eighb.__wrapped__ = orig       # type: ignore[attr-defined]
    eighb.__doc__ = orig.__doc__
    return eighb


_D3_REFS: dict[int, tuple[object, object]] = {}     # id(device Reference) -> (it, its CPU copy)


def _make_d3_weights(orig: Callable) -> Callable:
    """tad-dftd3's ``weight_references`` computes the Gaussian CN weights in float64 on the
    coordination numbers' device on purpose (underflow); on a device without float64 it runs here."""

    def weight_references(numbers, cn, reference, *args, **kwargs):
        route = _route_for(cn)
        if route is None or "float64" not in route.ops:
            return orig(numbers, cn, reference, *args, **kwargs)
        import torch

        cached = _D3_REFS.get(id(reference))
        if cached is None or cached[0] is not reference:
            if len(_D3_REFS) > 8:
                _D3_REFS.clear()
            cached = _D3_REFS[id(reference)] = (reference, reference.to(torch.device("cpu")))
        return _on_cpu("d3_weights", orig, (numbers, cn, cached[1], *args), kwargs, cn, route.upcast)

    weight_references.__wrapped__ = orig    # type: ignore[attr-defined]
    weight_references.__doc__ = orig.__doc__
    return weight_references


def _make_cpu_loader(orig: Callable) -> Callable:
    """tad-mctc's ``VDW_PAIRWISE(device, dtype)`` loads a float64 table with ``map_location=device``
    and casts afterwards; for a device without float64 it loads on the CPU, casts, then moves."""

    def loader(*args, **kwargs):
        device = kwargs.get("device", args[0] if args else None)
        stack = _routes.get(device_key(device)) if device is not None and _routes else None
        if not stack or "float64" not in stack[-1].ops:
            return orig(*args, **kwargs)
        import torch

        if "device" not in kwargs:
            args = args[1:]
        kwargs = {**kwargs, "device": torch.device("cpu")}
        _stats["cpu_table"] += 1
        with _routed_region():
            table = orig(*args, **kwargs)
        return table.to(device)

    loader.__wrapped__ = orig    # type: ignore[attr-defined]
    loader.__doc__ = orig.__doc__
    return loader


def _make_eeq(orig: Callable) -> Callable:
    def get_eeq_charges(*args, **kwargs):
        ref = _ref_tensor(args, kwargs)
        route = _route_for(ref) if ref is not None else None
        if route is None or not route.ops & _EEQ_OPS:
            return orig(*args, **kwargs)
        return _on_cpu("eeq", orig, args, kwargs, ref, route.upcast)

    get_eeq_charges.__wrapped__ = orig    # type: ignore[attr-defined]
    get_eeq_charges.__doc__ = orig.__doc__
    return get_eeq_charges


class _LinalgShim:
    """``torch.linalg`` for one dxtb module: ``solve`` is routed, everything else is torch's."""

    def __init__(self, real) -> None:
        self._real = real

    def __getattr__(self, name: str):
        return getattr(self._real, name)

    def solve(self, A, B, *args, **kwargs):  # noqa: N803 - torch's argument names
        route = _route_for(A)
        if route is None or "solve" not in route.ops:
            return self._real.solve(A, B, *args, **kwargs)
        return _on_cpu("solve", self._real.solve, (A, B, *args), kwargs, A, route.upcast)


class _TorchShim:
    """Stands in for ``torch`` inside one dxtb module (attribute access is forwarded to torch)."""

    def __init__(self, real) -> None:
        self._real = real
        self.linalg = _LinalgShim(real.linalg)

    def __getattr__(self, name: str):
        return getattr(self._real, name)


def _make_torch_shim(orig) -> object | str:
    import torch

    if orig is not torch:
        return "module's torch is not the torch module"
    return _TorchShim(orig)


# (module, attribute, factory). Factories return the replacement, or a string saying why not.
_TARGETS: tuple[tuple[str, str, Callable[[Any], Any]], ...] = (
    ("tad_mctc.storch", "eighb", _make_eighb),             # what dxtb calls: storch.eighb(...)
    ("tad_mctc.storch.linalg", "eighb", _make_eighb),      # where it is defined
    ("tad_multicharge", "get_eeq_charges", _make_eeq),     # dxtb._src.scf.guess imports it at call time
    ("tad_multicharge.model.eeq", "get_charges", _make_eeq),
    ("tad_dftd3.model", "weight_references", _make_d3_weights),     # dxtb calls d3.model.weight_references
    ("tad_dftd3.model.weights", "weight_references", _make_d3_weights),
    # the dxtb fork (tad-mctc 0.11) carries its own tad-dftd3, binds storch.eighb in its compat layer and
    # wraps tad-multicharge 0.8 (which takes a Structure, so its own get_eeq_charges sees no tensor)
    ("dxtb._src.exlibs.tad_dftd3.model", "weight_references", _make_d3_weights),
    ("dxtb._src.exlibs.tad_dftd3.model.weights", "weight_references", _make_d3_weights),
    ("dxtb._src.exlibs.mctc_compat.storch", "eighb", _make_eighb),
    ("dxtb._src.exlibs.mctc_compat", "get_eeq_charges", _make_eeq),   # imported at call time by the guess
    ("tad_mctc.data.radii", "VDW_PAIRWISE", _make_cpu_loader),      # dxtb's D3 cache: radii.VDW_PAIRWISE(**dd)
    ("dxtb._src.scf.mixer.anderson", "torch", _make_torch_shim),   # private: guarded, see _install
)

# Modules only one dxtb build has: PyPI's dxtb uses PyPI tad-dftd3, the fork its vendored copy (exlibs).
# A missing one is expected, not a failed patch.
_BUILD_SPECIFIC = ("tad_dftd3", "dxtb._src.exlibs")
ABSENT = "absent (not used by this dxtb build)"


def _install() -> None:
    """Patch the targets (first active context only). Missing or changed targets are skipped and
    recorded in ``patch_status()``; the device would then raise as stock dxtb does."""
    global _install_count
    _install_count += 1
    if _install_count > 1:
        return
    _status.clear()
    made: dict[int, Any] = {}
    for mod_name, attr, factory in _TARGETS:
        name = f"{mod_name}.{attr}"
        try:
            mod = importlib.import_module(mod_name)
        except ModuleNotFoundError as exc:
            build_specific = mod_name.startswith(_BUILD_SPECIFIC) and (exc.name or "").startswith(_BUILD_SPECIFIC)
            _status[name] = ABSENT if build_specific else f"not importable ({type(exc).__name__})"
            continue
        except Exception as exc:
            _status[name] = f"not importable ({type(exc).__name__})"
            continue
        orig = getattr(mod, attr, None)
        if orig is None:
            _status[name] = "missing"
            continue
        if id(orig) not in made:
            made[id(orig)] = factory(orig)
        new = made[id(orig)]
        if isinstance(new, str):
            _status[name] = f"skipped: {new}"
            continue
        setattr(mod, attr, new)
        _installed.append((mod, attr, orig))
        _status[name] = "patched"


def _uninstall() -> None:
    global _install_count
    _install_count = max(0, _install_count - 1)
    if _install_count:
        return
    while _installed:
        mod, attr, orig = _installed.pop()
        setattr(mod, attr, orig)
