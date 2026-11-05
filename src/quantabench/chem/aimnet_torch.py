"""AIMNet2 without NVIDIA Warp, for AMD (ROCm) and Apple (MPS) GPUs (GPU-1 / FORK-3).

Where ``aimnet`` 0.2 needs Warp
-------------------------------
The AIMNet2 network is plain PyTorch. Warp enters through ``nvalchemi-toolkit-ops`` and one kernel
of aimnet's own:

1. Neighbor lists: ``AIMNet2Calculator.make_nbmat`` calls ``nvalchemiops.torch.neighbors.neighbor_list``
   for flattened input, which the calculator uses for 2-D (single-molecule) coordinates, on the CPU,
   above ``nb_threshold`` (120) atoms, for Hessians and for periodic cells.
2. Dispersion: the external ``aimnet.modules.DFTD3`` module calls ``nvalchemiops``' ``dftd3`` kernel,
   and in dense mode first builds a 15 Å neighbor list with ``neighbor_list``.
3. ``aimnet.kernels.conv_sv_2d_sp``: a Warp kernel for the vector convolution in ``ConvSV``. It runs
   whenever a neighbor matrix is present and ``device.type == "cuda"``, which includes ROCm.
4. DSF/Ewald/PME Coulomb (periodic systems only); not needed for molecules.

Warp targets NVIDIA CUDA and the CPU: it rejects ROCm devices (BTM NOVA: "Invalid device identifier:
cuda:0", because HIP builds of PyTorch also call the GPU ``cuda:0``) and is CPU-only on macOS.

What this module does instead
-----------------------------
- The model is loaded with aimnet's own loader (``AIMNet2Calculator(..., needs_dispersion=False)``
  on the CPU, then moved), so registry names, metadata and family rules stay aimnet's.
- It is always evaluated in aimnet's dense mode (``nb_mode`` 0: no neighbor matrix, all pairs, cutoff
  functions inside the network): no neighbor list and no Warp convolution. This is the mode the stock
  calculator itself uses on CUDA for batched molecules up to ``nb_threshold`` atoms. Memory grows as
  N² per molecule, which is fine for the molecules QuantaBench refines (< ~200 atoms).
- Coulomb: the stock "simple" all-pairs ``LRCoulomb`` (pure torch), as the stock calculator adds it.
- D3(BJ): ``tad-dftd3`` (pure PyTorch) with aimnet's ``d3_params`` (ωB97M: s6 1.0, s8 0.3908,
  a1 0.5660, a2 3.1280, from the model metadata, the same values as
  ``aimnet/models/aimnet2_dftd3_wb97m.yaml``), aimnet's own reference tables (``aimnet/dftd3_data.pt``:
  C6 references, reference CNs, covalent radii, r4/r2), and the stock calculator's cutoff treatment
  (pairs and coordination numbers within 15 Å, energies switched off by nvalchemiops' C² S5 polynomial
  between 12 and 15 Å, no three-body term).
- Forces and Hessians by autograd (the Hessian as double backward, rows batched with ``torch.func.vmap``).

Precision
---------
The network runs in float32 everywhere, as in the stock calculator. D3 runs in float64 on the CPU and
float32 on GPUs. Apple GPUs have no float64, so with ``allow_float64=False`` (automatic on ``mps``)
the three modules that accumulate in float64 (``AtomicShift`` weights, ``SRCoulomb``, ``LRCoulomb``)
are switched to float32. The per-element reference energies (SAE, ~10⁴ eV per molecule) are then
added on the CPU in float64, and the energy is returned as a float64 CPU tensor. Forces and charges
stay on the device.

Which path runs
---------------
``load_calculator(model, device)`` returns the stock ``AIMNet2Calculator`` where Warp can serve the
device (NVIDIA CUDA, CPU) and this adapter on ROCm, MPS, when Warp cannot see the GPU, or when the
stock calculator fails a one-molecule probe at load. ``QUANTABENCH_AIMNET_BACKEND`` = ``torch`` or
``stock`` forces one or the other (``auto`` is the default).

What still needs Warp
---------------------
``aimnet`` 0.2 imports ``warp`` and ``nvalchemiops`` at import time, so both packages must be
installed (CPU-only Warp is enough, as on macOS and Windows ROCm). Making these imports lazy is the
upstream change proposed to aimnet.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

BACKEND_ENV = "QUANTABENCH_AIMNET_BACKEND"
BACKENDS = ("auto", "torch", "stock")

#: aimnet 0.2 ``AIMNet2Calculator._default_dftd3_cutoff`` / ``_default_dftd3_smoothing``.
D3_CUTOFF_A = 15.0
D3_SMOOTHING_FRACTION = 0.2
#: ωB97M-D3(BJ) as aimnet stores it (``aimnet/models/aimnet2_dftd3_wb97m.yaml``,
#: ``calculators/calculator.py::_WB97M_D3_PARAMS``); the released .pt files carry the same values as
#: ``d3_params`` metadata. Used only when a model asks for dispersion without parameters.
WB97M_D3BJ = {"s6": 1.0, "s8": 0.3908, "a1": 0.5660, "a2": 3.1280}

BOHR_ANG = 0.529177210903
HARTREE_EV = 27.211386245988

_PROBE_NUMBERS = (8, 1, 1)
_PROBE_COORD = ((0.0, 0.0, 0.1173), (0.0, 0.7572, -0.4692), (0.0, -0.7572, -0.4692))


# ============================================================================ backend choice
def requested_backend() -> str:
    """``QUANTABENCH_AIMNET_BACKEND`` (lower case), "auto" when unset or unknown."""
    value = os.environ.get(BACKEND_ENV, "auto").strip().lower()
    return value if value in BACKENDS else "auto"


def supports_float64(device) -> bool:
    """False for Apple's MPS backend, which has no float64 tensors."""
    import torch

    return torch.device(device).type != "mps"


def warp_problem(device) -> str:
    """Why NVIDIA Warp cannot run AIMNet2's D3 and neighbor lists on ``device`` ("" when it can)."""
    import torch

    dev = torch.device(device)
    if dev.type == "mps":
        return "Warp has no Apple GPU backend (it is CPU-only on macOS)"
    if dev.type == "cuda" and getattr(torch.version, "hip", None):
        return "ROCm GPU: Warp supports NVIDIA CUDA only"
    if dev.type not in ("cpu", "cuda"):
        return f"Warp does not support {dev.type!r} devices"
    try:
        import warp
    except Exception as exc:
        return f"NVIDIA Warp cannot be imported ({type(exc).__name__}: {exc})"
    if dev.type == "cuda":
        try:
            ok = bool(warp.is_cuda_available())
        except Exception as exc:
            return f"Warp failed to initialize CUDA ({type(exc).__name__}: {exc})"
        if not ok:
            return "Warp finds no CUDA driver"
    return ""


def choose_backend(device) -> tuple[str, str]:
    """("stock" | "torch", reason) for ``device``, honoring ``QUANTABENCH_AIMNET_BACKEND``."""
    req = requested_backend()
    if req != "auto":
        return req, f"{BACKEND_ENV}={req}"
    why = warp_problem(device)
    return ("torch", why) if why else ("stock", "")


_CHECKED_NBLIST: list[type] = []


def guard_neighbor_lists(calc) -> int:
    """Make a stock ``AIMNet2Calculator``'s adaptive neighbor lists grow when they overflow.

    aimnet 0.2's ``AdaptiveNeighborList`` shrinks its buffer after small molecules (down to 16
    neighbors) and grows it again only when nvalchemiops raises ``NeighborOverflowError``. nvalchemiops
    0.5 does not raise: it returns the true neighbor counts with a truncated matrix. So once a small
    molecule has been evaluated, larger ones silently lose neighbors (0.1-0.2 eV on PFOA after water,
    random-weight model). This re-classes the calculator's lists (in place) to check the counts and
    retry with a larger buffer. Returns the number of lists guarded."""
    from aimnet.calculators.calculator import AdaptiveNeighborList

    if not _CHECKED_NBLIST:
        class CheckedAdaptiveNeighborList(AdaptiveNeighborList):
            def __call__(self, positions, cell=None, pbc=None, batch_idx=None, fill_value=None):
                while True:
                    nbmat, num, shifts = super().__call__(positions, cell, pbc, batch_idx, fill_value)
                    need = int(num.max().item()) if num.numel() else 0
                    if need <= nbmat.shape[1]:
                        return nbmat, num, shifts
                    self.max_neighbors = self._round_to_16(int(need / self.target_utilization) + 1)

        _CHECKED_NBLIST.append(CheckedAdaptiveNeighborList)
    n = 0
    for name in ("_nblist", "_nblist_lr", "_nblist_dftd3", "_nblist_coulomb"):
        nl = getattr(calc, name, None)
        if type(nl) is AdaptiveNeighborList:
            nl.__class__ = _CHECKED_NBLIST[0]
            n += 1
    return n


def _probe(calc) -> None:
    """One water single point with forces, to surface device/kernel failures at load time."""
    import torch

    data = {"coord": torch.tensor([_PROBE_COORD], dtype=torch.float32),
            "numbers": torch.tensor([_PROBE_NUMBERS]),
            "charge": torch.tensor([0.0]), "mult": torch.tensor([1.0])}
    out = calc(data, forces=True)
    if not bool(torch.isfinite(out["forces"]).all()):
        raise RuntimeError("non-finite forces in the AIMNet2 probe")


def load_calculator(model: str = "aimnet2", device: str = "cpu"):
    """AIMNet2 calculator for ``device``: the stock ``AIMNet2Calculator`` where Warp serves the device,
    else ``AIMNet2TorchCalculator``. Both have the same ``eval``/``__call__`` interface. The returned
    object carries ``qb_backend`` ("stock" or "torch") and ``qb_backend_reason``."""
    backend, why = choose_backend(device)
    if backend == "stock":
        from aimnet.calculators import AIMNet2Calculator

        try:
            calc = AIMNet2Calculator(model, device=device)
        except TypeError:  # older aimnet without the device argument
            calc = AIMNet2Calculator(model)
        guard_neighbor_lists(calc)
        if requested_backend() == "stock":
            calc.qb_backend, calc.qb_backend_reason = "stock", why
            return calc
        try:
            _probe(calc)
        except Exception as exc:
            why = f"stock AIMNet2 failed on {device} ({type(exc).__name__}: {exc})"
            del calc
        else:
            calc.qb_backend, calc.qb_backend_reason = "stock", ""
            return calc
    calc = AIMNet2TorchCalculator(model, device=device)
    calc.qb_backend_reason = why
    return calc


# ============================================================================ D3(BJ)
def _aimnet_d3_tables():
    """aimnet's DFT-D3 tables (``aimnet/dftd3_data.pt``) as (cn_ref[Z, ref], c6[Zi, Zj, ref_i, ref_j],
    rcov[Z] Bohr, r4r2[Z]) float64 CPU tensors, or None when aimnet is not installed. The file stores
    C6 and the reference CNs per element pair (old D3 ``pars`` layout, 5 references); missing references
    have C6 = 0 and get reference CN -1 (tad-dftd3's convention)."""
    import torch

    spec = importlib.util.find_spec("aimnet")
    if spec is None or spec.origin is None:
        return None
    path = Path(spec.origin).parent / "dftd3_data.pt"
    if not path.is_file():
        return None
    p = torch.load(path, map_location="cpu", weights_only=True)
    c6ab = p["c6ab"][..., 0].double()
    cnref = p["c6ab"][..., 1].double()
    valid = c6ab != 0
    has_ref = valid.any(dim=(1, 3))                                   # [Z, ref]
    n_valid = valid.sum(dim=(1, 3)).clamp_min(1)
    cn = torch.where(has_ref, (cnref * valid).sum(dim=(1, 3)) / n_valid, torch.full_like(c6ab[:, 0, :, 0], -1.0))
    return cn, c6ab, p["rcov"].double(), p["r4r2"].double()


def _s5_switch(d, r_on: float, r_off: float):
    """nvalchemiops' C² switch: 1 below r_on, 1 - (10t³ - 15t⁴ + 6t⁵) between, 0 above r_off."""
    t = ((d - r_on) / (r_off - r_on)).clamp(0.0, 1.0)
    return 1.0 - t * t * t * (10.0 - 15.0 * t + 6.0 * t * t)


class TorchD3:
    """Two-body DFT-D3(BJ) through ``tad-dftd3``, computed the way aimnet's ``DFTD3`` module computes it.

    Same damping parameters, aimnet's reference tables (``tad-dftd3``'s own tables when aimnet is not
    installed; they agree to float32 precision for every AIMNet2 element), coordination numbers and
    pairs within ``cutoff`` Å, energies switched off by the S5 polynomial over the last
    ``smoothing_fraction`` of the cutoff, no Axilrod-Teller-Muto term.

    With ``allow_float64`` on a CPU or CUDA/ROCm device the reference weights and C6 come from
    ``tad_dftd3.model`` (it normalizes the weights in float64). Without float64 (Apple GPUs),
    or on other devices, the same weights are normalized in the working dtype with a max-shift
    (exp(arg - max), never all-zero), which is what nvalchemiops does in float32.
    """

    def __init__(self, s6: float = 1.0, s8: float = WB97M_D3BJ["s8"], a1: float = WB97M_D3BJ["a1"],
                 a2: float = WB97M_D3BJ["a2"], *, cutoff: float = D3_CUTOFF_A,
                 smoothing_fraction: float = D3_SMOOTHING_FRACTION, device="cpu", dtype=None,
                 allow_float64: bool | None = None):
        import torch

        try:
            import tad_dftd3 as d3
        except ImportError as exc:
            raise ImportError("AIMNet2 without NVIDIA Warp needs tad-dftd3 for its dispersion: "
                              "pip install tad-dftd3 (it also comes with dxtb)") from exc

        self.device = torch.device(device)
        self.allow_float64 = supports_float64(self.device) if allow_float64 is None else bool(allow_float64)
        if dtype is None:
            dtype = torch.float64 if self.device.type == "cpu" and self.allow_float64 else torch.float32
        if dtype == torch.float64 and not self.allow_float64:
            raise ValueError(f"float64 is not available on {self.device}")
        self.dtype = dtype
        dd = {"device": self.device, "dtype": dtype}
        tables = _aimnet_d3_tables()
        if tables is not None:
            cn, c6, rcov, r4r2 = (t.to(**dd) for t in tables)
            self.source = "aimnet dftd3_data.pt"
        else:
            from tad_mctc.data import radii

            ref = d3.reference.Reference(device=self.device, dtype=dtype)
            cn, c6 = ref.cn, ref.c6
            rcov, r4r2 = radii.COV_D3(**dd), d3.data.R4R2(**dd)
            self.source = "tad-dftd3 built-in tables"
        self.ref = d3.reference.Reference(cn=cn, c6=c6, device=self.device, dtype=dtype)
        self.rcov, self.r4r2 = rcov, r4r2
        self.params = {"s6": float(s6), "s8": float(s8), "a1": float(a1), "a2": float(a2)}
        self._param = {k: torch.tensor(v, **dd) for k, v in self.params.items()}
        self.cutoff = float(cutoff)
        self.r_off = self.cutoff / BOHR_ANG
        self.r_on = self.cutoff * (1.0 - float(smoothing_fraction)) / BOHR_ANG
        self._use_tad_model = self.allow_float64 and self.device.type in ("cpu", "cuda")

    def _c6(self, numbers, cn):
        import torch

        if self._use_tad_model:
            import tad_dftd3 as d3

            weights = d3.model.weight_references(numbers, cn, self.ref)
            return d3.model.atomic_c6(numbers, weights, self.ref)
        refcn = self.ref.cn[numbers]                                   # [..., N, R]
        mask = refcn >= 0
        # finite filler (not -inf): padding atoms have no references and must not produce NaN gradients
        arg = torch.where(mask, -4.0 * (refcn - cn.unsqueeze(-1)) ** 2, torch.full_like(refcn, -1.0e4))
        arg = arg - arg.amax(-1, keepdim=True).detach()                # shift: largest weight is exp(0) = 1
        w = torch.where(mask, arg.exp(), torch.zeros_like(arg))
        w = w / w.sum(-1, keepdim=True).clamp_min(torch.finfo(w.dtype).tiny)
        rc6 = self.ref.c6[numbers.unsqueeze(-1), numbers.unsqueeze(-2)]  # [..., N, N, R, R]
        return torch.einsum("...ia,...jb,...ijab->...ij", w, w, rc6)

    def energy(self, numbers, positions):
        """D3 energy in eV, shape [B], for ``numbers`` [B, N] and ``positions`` [B, N, 3] in Å
        (differentiable in ``positions``; padding atoms have number 0)."""
        import tad_dftd3 as d3
        import torch
        from tad_mctc.ncoord import coordination_number, exp_count

        numbers = numbers.to(self.device)
        x = positions.to(self.device, self.dtype) / BOHR_ANG
        x = x - x.mean(dim=-2, keepdim=True).detach()   # smaller |x| -> less cancellation in float32 distances
        cutoff = torch.tensor(self.r_off, device=self.device, dtype=self.dtype)
        cn = coordination_number(numbers, x, counting_function=exp_count, rcov=self.rcov[numbers], cutoff=cutoff)
        c6 = self._c6(numbers, cn)
        r_on, r_off = self.r_on, self.r_off

        def damping(order, distances, qq, param):
            return d3.damping.rational_damping(order, distances, qq, param) * _s5_switch(distances, r_on, r_off)

        e_atom = d3.disp.dispersion2(numbers, x, self._param, c6, self.r4r2[numbers], damping, cutoff)
        return e_atom.sum(-1) * HARTREE_EV


# ============================================================================ float32-only modules
_F32_CLASSES: dict[str, type] = {}


def _float32_classes() -> dict[str, type]:
    """Subclasses of aimnet's Coulomb modules that accumulate in the input dtype instead of float64.
    Same formulas as ``aimnet.modules.lr`` (``coul_simple``, ``_calc_coulomb_sr``)."""
    if _F32_CLASSES:
        return _F32_CLASSES
    from aimnet import nbops, ops
    from aimnet.modules import LRCoulomb, SRCoulomb

    def coulomb_sr(data, rc, envelope, key_in, factor):
        d_ij = data["d_ij"]
        q_i, q_j = nbops.get_ij(data[key_in], data)
        fc = ops.exp_cutoff(d_ij, rc) if envelope == "exp" else ops.cosine_cutoff(d_ij, rc.item())
        e_ij = nbops.mask_ij_(fc * q_i * q_j / d_ij, data, 0.0)
        return factor * nbops.mol_sum(e_ij.sum(-1), data)

    class SRCoulomb32(SRCoulomb):
        def forward(self, data):
            e_sr = coulomb_sr(data, self.rc, self.envelope, self.key_in, self._factor)
            data[self.key_out] = data[self.key_out] - e_sr if self.key_out in data else -e_sr
            return data

    class LRCoulomb32(LRCoulomb):
        def coul_simple(self, data):
            suffix = nbops.resolve_suffix(data, ["_coulomb", "_lr"])
            data = ops.lazy_calc_dij(data, suffix)
            q_i, q_j = nbops.get_ij(data[self.key_in], data, suffix=suffix)
            e_ij = nbops.mask_ij_(q_i * q_j / data[f"d_ij{suffix}"], data, 0.0, suffix=suffix)
            e = self._factor * nbops.mol_sum(e_ij.sum(-1), data)
            if self.subtract_sr:
                e = e - coulomb_sr(data, self.rc, self.envelope, self.key_in, self._factor)
            return e

        def forward(self, data, **kwargs):
            if self.method != "simple":
                raise NotImplementedError(f"Coulomb method {self.method!r} needs nvalchemiops (Warp)")
            e = self.coul_simple(data)
            data[self.key_out] = data[self.key_out] + e if self.key_out in data else e
            return data

    _F32_CLASSES.update(SRCoulomb=SRCoulomb32, LRCoulomb=LRCoulomb32)
    return _F32_CLASSES


def _to_float32_modules(module) -> None:
    """Re-class every SRCoulomb/LRCoulomb inside ``module`` (in place) to its float32 variant."""
    from aimnet.modules import LRCoulomb, SRCoulomb

    classes = _float32_classes()
    for m in module.modules():
        if type(m) is SRCoulomb:
            m.__class__ = classes["SRCoulomb"]
        elif type(m) is LRCoulomb:
            m.__class__ = classes["LRCoulomb"]


def _split_atomic_shift(model):
    """Move the per-element reference energies out of ``model.outputs.atomic_shift`` (float64 in the
    released models): returns them as a float64 CPU table and zeroes the module (float32)."""
    import torch
    from aimnet.modules import AtomicShift

    outputs = getattr(model, "outputs", None)
    if outputs is None:
        return None
    for m in outputs.modules():
        if isinstance(m, AtomicShift) and m.key_in == "energy" and m.key_out == "energy":
            table = m.shifts.weight.detach().to("cpu", torch.float64).reshape(-1).clone()
            m.float()
            with torch.no_grad():
                m.shifts.weight.zero_()
            return table
    return None


def _pop_embedded_dftd3(model) -> list[dict]:
    """Remove nvalchemiops-backed ``DFTD3`` modules from ``model.outputs`` (raw YAML-built models);
    returns their parameters so ``TorchD3`` can replace them. Converted .pt models have none."""
    from aimnet.modules import DFTD3

    outputs = getattr(model, "outputs", None)
    found: list[dict] = []
    if outputs is None:
        return found
    names = list(outputs.keys()) if hasattr(outputs, "keys") else list(reversed(range(len(outputs))))
    for name in names:
        m = outputs[name]
        if isinstance(m, DFTD3):
            found.append({"s6": m.s6, "s8": m.s8, "a1": m.a1, "a2": m.a2, "cutoff": m.smoothing_off,
                          "smoothing_fraction": 1.0 - m.smoothing_on / m.smoothing_off})
            del outputs[name]
    return found


# ============================================================================ calculator
class AIMNet2TorchCalculator:
    """Warp-free stand-in for ``aimnet.calculators.AIMNet2Calculator`` on molecules (no periodic cells).

    Same call contract: ``calc(data, forces=False, stress=False, hessian=False, validate_species=True)``
    with ``data`` = coord [N, 3] or [B, N, 3] Å, numbers, charge [B] (and mult [B]); returns energy
    [B] eV, charges, forces eV/Å and, for one molecule, hessian [N, 3, N, 3] eV/Å². So it works under
    ``AIMNet2ASE``, ``vibrations.aimnet_hessian`` and ``batch_opt.AIMNet2Batched`` unchanged.

    ``dtype`` is the D3 dtype (default float64 on the CPU, float32 on GPUs). ``allow_float64`` (default:
    False on ``mps`` only) keeps aimnet's float64 energy accumulation; with False, no float64 tensor
    touches the device (see the module docstring). ``hessian_chunk`` caps the Hessian rows evaluated
    in one vmapped backward pass.
    """

    qb_backend = "torch"

    def __init__(self, model: Any = "aimnet2", device: str = "cpu", *, dtype=None,
                 allow_float64: bool | None = None, needs_dispersion: bool | None = None,
                 hessian_chunk: int | None = None):
        import torch
        from aimnet.calculators import AIMNet2Calculator

        dev = torch.device(device)
        self.device = str(dev)
        self.allow_float64 = supports_float64(dev) if allow_float64 is None else bool(allow_float64)
        if self.allow_float64 and not supports_float64(dev):
            raise ValueError(f"float64 is not available on {dev}")
        self.hessian_chunk = hessian_chunk
        self.qb_backend_reason = ""
        # aimnet's loader on the CPU: registry, metadata, family defaults, external Coulomb config.
        base = AIMNet2Calculator(model, device="cpu", needs_dispersion=False)
        self.model = base.model
        if isinstance(self.model, torch.jit.ScriptModule) and not self.allow_float64:
            raise NotImplementedError("legacy TorchScript AIMNet2 models cannot run without float64")
        self.cutoff = base.cutoff
        self.keys_in = dict(base.keys_in)
        meta = dict(base.metadata or {})
        self._meta = meta
        self.coulomb = base.external_coulomb
        if self.coulomb is not None and self.coulomb.method != "simple":
            raise NotImplementedError(f"Coulomb method {self.coulomb.method!r} needs nvalchemiops (Warp)")
        embedded = [] if isinstance(self.model, torch.jit.ScriptModule) else _pop_embedded_dftd3(self.model)
        del base

        d3_kwargs: dict[str, Any] | None = None
        need_d3 = bool(meta.get("needs_dispersion", False)) if needs_dispersion is None else bool(needs_dispersion)
        if embedded:
            d3_kwargs = embedded[0]
        elif need_d3:
            p = meta.get("d3_params") or {}
            if not p and needs_dispersion is None:
                raise ValueError("model needs dispersion but its metadata has no d3_params")
            p = {**WB97M_D3BJ, **p}
            d3_kwargs = {"s6": p["s6"], "s8": p["s8"], "a1": p["a1"], "a2": p["a2"]}

        self._sae = None
        if not self.allow_float64:
            self._sae = _split_atomic_shift(self.model)
            _to_float32_modules(self.model)
            if self.coulomb is not None:
                _to_float32_modules(self.coulomb)
            self.model.float()
        self.model.to(dev)
        if self.coulomb is not None:
            self.coulomb.to(dev)
        self.d3 = (TorchD3(**d3_kwargs, device=dev, dtype=dtype, allow_float64=self.allow_float64)
                   if d3_kwargs is not None else None)
        self.dtype = self.d3.dtype if self.d3 is not None else (dtype or torch.float32)

    # ---------------------------------------------------------------- AIMNet2Calculator look-alike
    def __call__(self, *args, **kwargs):
        return self.eval(*args, **kwargs)

    @property
    def metadata(self):
        from types import MappingProxyType

        return MappingProxyType(self._meta)

    @property
    def is_nse(self) -> bool:
        return getattr(self.model, "num_charge_channels", 1) == 2

    @property
    def has_external_coulomb(self) -> bool:
        return self.coulomb is not None

    @property
    def has_external_dftd3(self) -> bool:
        return False

    @property
    def coulomb_method(self) -> str | None:
        return None if self.coulomb is None else "simple"

    def set_lrcoulomb_method(self, method: str, *args, **kwargs) -> None:
        if method != "simple":
            raise NotImplementedError(f"Coulomb method {method!r} needs nvalchemiops (Warp); "
                                      "the Warp-free AIMNet2 path supports molecules only")

    def __repr__(self) -> str:
        d3 = "none" if self.d3 is None else f"tad-dftd3 {self.d3.dtype}".replace("torch.", "")
        return f"AIMNet2TorchCalculator(device={self.device!r}, d3={d3}, float64={self.allow_float64})"

    # ---------------------------------------------------------------- evaluation
    def _validate(self, numbers, charge) -> None:
        impl = self._meta.get("implemented_species") or []
        if impl:
            seen = {int(z) for z in numbers.flatten().tolist() if int(z) > 0}
            bad = sorted(seen - set(impl))
            if bad:
                raise ValueError(f"Atomic numbers {bad} are not in this model's implemented_species {sorted(impl)}. "
                                 "Pass validate_species=False to bypass.")
        if self._meta.get("supports_charged_systems") is False and float(charge.abs().max()) > 1e-6:
            raise ValueError("This model does not support net-charged systems. Pass validate_species=False to bypass.")

    def _hessian(self, forces_flat, x):
        """-d(forces)/dx by vmapped vector-Jacobian products in chunks of rows (a plain loop if vmap
        cannot batch some backward op on this device)."""
        import warnings

        import torch

        n = forces_flat.numel()
        n_atoms = n // 3
        chunk = self.hessian_chunk or max(8, min(n, int(1e6 // max(1, n_atoms * n_atoms))))
        eye = torch.eye(n, dtype=forces_flat.dtype, device=forces_flat.device)

        def vjp(v):
            return torch.autograd.grad(forces_flat, x, v, retain_graph=True)[0]

        rows = []
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for s in range(0, n, chunk):
                    rows.append(torch.func.vmap(vjp)(eye[s:s + chunk]).reshape(-1, n))
        except Exception:
            rows = [vjp(eye[i]).reshape(1, n) for i in range(n)]
        return -torch.cat(rows, 0).reshape(n_atoms, 3, n_atoms, 3)

    def eval(self, data: dict, forces: bool = False, stress: bool = False, hessian: bool = False, *,
             validate_species: bool = True) -> dict:
        import torch

        if stress or data.get("cell") is not None:
            raise NotImplementedError("periodic cells and stress need the stock AIMNet2 calculator (Warp)")
        dev = torch.device(self.device)
        coord = torch.as_tensor(data["coord"], dtype=torch.float32, device=dev)
        numbers = torch.as_tensor(data["numbers"], device=dev).long()
        flat = coord.ndim == 2
        if flat:
            mol_idx = data.get("mol_idx")
            if mol_idx is not None and int(torch.as_tensor(mol_idx).max()) > 0:
                raise NotImplementedError("flattened multi-molecule input: pass coord as [B, N, 3]")
            coord, numbers = coord.unsqueeze(0), numbers.unsqueeze(0)
        b = coord.shape[0]
        if hessian and b != 1:
            raise NotImplementedError("Hessian calculation is not supported for batched inputs with B > 1")

        def per_system(key, default=None):
            v = data.get(key, default)
            if v is None:
                return None
            t = torch.as_tensor(v, dtype=torch.float32, device=dev).reshape(-1)
            return t.expand(b) if t.numel() == 1 else t

        charge = per_system("charge", 0.0)
        mult = per_system("mult")
        if validate_species:
            self._validate(numbers, charge)

        res = self._device_eval(coord, numbers, charge, mult, forces, hessian)
        if self._sae is not None:   # reference energies in float64 on the CPU (see the module docstring)
            res["energy"] = res["energy"].to("cpu", torch.float64) + self._sae[numbers.cpu()].sum(-1)
        if flat:
            for k in ("charges", "spin_charges", "forces"):
                if k in res:
                    res[k] = res[k][0]
        return res

    def _device_eval(self, coord, numbers, charge, mult, forces: bool, hessian: bool) -> dict:
        """Everything that runs on the device: model (dense nb_mode 0: no neighbor list, no Warp
        kernel), Coulomb, D3, and the autograd forces/Hessian. Returns detached tensors."""
        import torch

        need_grad = forces or hessian
        x = coord.detach().requires_grad_(need_grad)
        res: dict = {}
        with torch.set_grad_enabled(need_grad):
            inp = {"coord": x, "numbers": numbers, "charge": charge}
            if mult is not None:
                inp["mult"] = mult
            out = self.model(inp)
            if self.coulomb is not None:
                out = self.coulomb(out)
            energy = out["energy"]
            if self.d3 is not None:
                energy = energy + self.d3.energy(numbers, x).to(energy.dtype)
            if need_grad:
                (g,) = torch.autograd.grad(energy.sum(), x, create_graph=hessian)
                res["forces"] = -g
            if hessian:
                res["hessian"] = self._hessian(res["forces"].reshape(-1), x)
        res = {k: v.detach() for k, v in res.items()}
        res["energy"] = energy.detach()
        res["charges"] = out["charges"].detach()
        if "spin_charges" in out:
            res["spin_charges"] = out["spin_charges"].detach()
        return res
