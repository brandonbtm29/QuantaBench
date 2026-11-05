"""Energy/force engines behind one interface.

Each engine produces an ASE ``Calculator`` for a given molecule (atomic numbers, charge,
multiplicity) on a given torch device. Engines are optional: ``available()`` explains what is
missing instead of failing at import time, and the pipeline falls back from GPU to CPU if an
engine cannot run on the GPU it was given.

Engine        Package        Device        Charged molecules   Notes
-----------   -------------  ------------  ------------------  -----------------------------------
mmff          rdkit          CPU threads   yes                 already applied during conformer search
gfn1-xtb      dxtb           CPU or GPU    yes                 pure PyTorch: runs on ROCm and CUDA
gfn2-xtb      dxtb+libcint   CPU (ints)    yes                 libcint backend: Linux/WSL for now
aimnet2       aimnet         CPU or GPU    yes                 wB97M-D3 quality for H,B,C,N,O,F,Si,P,S,Cl,As,Se,Br,I;
                                                               ROCm/Apple GPUs through ``aimnet_torch`` (no Warp)

More engines register themselves from plugin modules (``PLUGINS``): the xtb program (GFN2/GFN1/GFN-FF
with ALPB solvation), tblite (in-process GFN2/GFN1), MOPAC (PM7/PM6). A plugin calls ``register`` with
an ``EngineInfo``, an ``available`` check and a calculator factory, and optionally a properties
function. Everything else (pipeline, CLI, GUI, frequencies) goes through this module, so a new engine
needs no changes elsewhere.

Calculator factories take ``(numbers, charge, multiplicity, device, **options)``; ``options`` carries
``solvent`` (canonical name from ``chem.solvents``, "" = gas phase) and ``threads``. Properties
functions take ``(numbers, positions_A, charge, multiplicity, **options)`` and return a dict of
JSON-ready scalars named with the engine's ``prop_prefix`` (``<prefix>_dipole_debye``, ...) plus an
optional ``"charges"`` list and ``"_notes"`` list.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from ..resources import has_package

HARTREE_EV = 27.211386245988
BOHR_ANG = 0.529177210903

AIMNET2_ELEMENTS = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 33, 34, 35, 53}


@dataclass(frozen=True)
class EngineInfo:
    key: str
    label: str
    description: str
    gpu_capable: bool
    elements: frozenset[int] | None = None  # None = any
    family: str = "tight binding"           # "force field" | "tight binding" | "semi-empirical" | "ML potential" | "DFT"
    solvation: tuple[str, ...] = ()         # implicit solvent models it supports, e.g. ("alpb", "gbsa")
    external: str = ""                      # external program it runs (chem.external.PROGRAMS key), "" = in-process
    prop_prefix: str = ""                   # descriptor prefix of its properties function
    batched: bool = False                   # has a batch_opt adapter


@dataclass
class _Plugin:
    info: EngineInfo
    available: Callable[[], tuple[bool, str]]
    factory: Callable[..., Calculator]
    properties: Callable[..., dict] | None = None
    native_optimize: Callable[..., object] | None = None   # optional, see register()
    extra: dict = field(default_factory=dict)


ENGINES: dict[str, EngineInfo] = {
    "mmff": EngineInfo("mmff", "MMFF94s (force field)", "RDKit MMFF94s; UFF when MMFF lacks parameters. Fast, CPU.",
                       False, family="force field"),
    "gfn1-xtb": EngineInfo("gfn1-xtb", "GFN1-xTB (dxtb)", "Semi-empirical tight binding in PyTorch. GPU-capable.", True,
                           frozenset(range(1, 87)), batched=True),
    "gfn2-xtb": EngineInfo("gfn2-xtb", "GFN2-xTB (dxtb + libcint)", "GFN2 tight binding; needs tad-libcint (Linux/WSL).",
                           True, frozenset(range(1, 87)), batched=True),
    "aimnet2": EngineInfo("aimnet2", "AIMNet2 (ML, wB97M-D3)", "Neural network potential trained on wB97M-D3/def2-TZVPP; "
                          "neutral and charged molecules. GPU-capable: NVIDIA (stock aimnet), AMD ROCm and "
                          "Apple GPUs (Warp-free path with tad-dftd3 dispersion).", True, frozenset(AIMNET2_ELEMENTS),
                          family="ML potential", batched=True),
}

#: Plugin modules (``quantabench.chem.<name>``) that register more engines when imported.
PLUGINS = ("eng_tblite", "eng_xtb", "eng_mopac")

#: ``method="auto"`` tries these in order (then MMFF): best accuracy per cost first, in-process before external.
AUTO_ORDER = ["aimnet2", "gfn2-xtb", "tblite-gfn2", "xtb-gfn2", "gfn1-xtb"]

_REGISTRY: dict[str, _Plugin] = {}
_PLUGINS_LOADED = False
PLUGIN_ERRORS: dict[str, str] = {}


def register(info: EngineInfo, available: Callable[[], tuple[bool, str]], factory: Callable[..., Calculator],
             properties: Callable[..., dict] | None = None, native_optimize: Callable[..., object] | None = None,
             **extra) -> None:
    """Add an engine. ``available()`` must be cheap (no heavy imports, no subprocess beyond a cached
    version probe). ``native_optimize(atoms, fmax, max_steps, should_stop, log, **options)`` may return an
    ``optimize.OptResult`` from the program's own optimizer (much faster than ASE driving an external
    program step by step); the pipeline uses it when ``BuildConfig.optimizer == "native"``."""
    ENGINES[info.key] = info
    _REGISTRY[info.key] = _Plugin(info, available, factory, properties, native_optimize, dict(extra))


def load_plugins() -> None:
    """Import the plugin modules once. A plugin that fails to import is recorded in PLUGIN_ERRORS."""
    global _PLUGINS_LOADED
    if _PLUGINS_LOADED:
        return
    _PLUGINS_LOADED = True
    for name in PLUGINS:
        try:
            importlib.import_module(f"{__package__}.{name}")
        except ModuleNotFoundError as exc:
            if exc.name != f"{__package__}.{name}":
                PLUGIN_ERRORS[name] = f"{type(exc).__name__}: {exc}"
        except Exception as exc:  # a broken plugin must not break the app
            PLUGIN_ERRORS[name] = f"{type(exc).__name__}: {exc}"


def plugin(key: str) -> _Plugin | None:
    load_plugins()
    return _REGISTRY.get(key)


def all_engines() -> dict[str, EngineInfo]:
    """Every known engine, plugins included (for menus and the CLI)."""
    load_plugins()
    return dict(ENGINES)


FAMILY_ORDER = ("ML potential", "tight binding", "semi-empirical", "force field", "DFT")


def method_keys() -> list[str]:
    """'auto' then every engine grouped by family (ML, tight binding, semi-empirical, force field)."""
    engines = all_engines()
    order = {f: i for i, f in enumerate(FAMILY_ORDER)}
    keys = sorted(engines, key=lambda k: (order.get(engines[k].family, 99), list(engines).index(k)))
    return ["auto", *keys]


def properties_function(key: str) -> Callable[..., dict] | None:
    p = plugin(key)
    return p.properties if p else None


def native_optimizer(key: str) -> Callable[..., object] | None:
    p = plugin(key)
    return p.native_optimize if p else None


def supports_solvent(key: str, solvent: str) -> bool:
    """Whether engine ``key`` can model ``solvent`` ("" always works)."""
    if not solvent:
        return True
    from . import solvents

    info = all_engines().get(key)
    if info is None:
        return False
    return any(solvents.for_engine(solvent, m) is not None for m in info.solvation)


_LIBCINT_OK: bool | None = None


def has_libcint() -> bool:
    """True when dxtb's libcint integral backend is importable *and* compatible.

    It provides the multipole integrals needed for GFN2-xTB and for any dxtb electric-field quantity
    (dipole moment, polarizability, IR intensities). There is no compatible Windows build yet:
    tad-libcint 0.4 (the only release with a Windows wheel) needs a newer tad-mctc than dxtb 0.4 pins.
    """
    global _LIBCINT_OK
    if _LIBCINT_OK is None:
        try:
            import tad_libcint  # noqa: F401

            _LIBCINT_OK = True
        except Exception:
            _LIBCINT_OK = False
    return _LIBCINT_OK


def available(key: str) -> tuple[bool, str]:
    if key == "mmff":
        return True, ""
    p = plugin(key)
    if p is not None:
        try:
            return p.available()
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"
    if not has_package("torch"):
        return False, "PyTorch is not installed (see docs/INSTALL.md)"
    if key in ("gfn1-xtb", "gfn2-xtb"):
        if not has_package("dxtb"):
            return False, "install with: pip install dxtb"
        if key == "gfn2-xtb" and not has_libcint():
            return False, "GFN2 needs tad-libcint, currently Linux/WSL only (pip install 'tad-libcint<0.4')"
        return True, ""
    if key == "aimnet2":
        if not has_package("aimnet"):
            return False, "install with: pip install aimnet"
        if not has_package("warp") or not has_package("nvalchemiops"):
            return False, "aimnet needs warp-lang and nvalchemi-toolkit-ops importable (CPU-only Warp is enough)"
        return True, ""
    return False, f"unknown engine {key!r}"


def supports(key: str, numbers) -> tuple[bool, str]:
    info = all_engines()[key]
    if info.elements is None:
        return True, ""
    missing = sorted({int(z) for z in numbers} - set(info.elements))
    if missing:
        from ase.data import chemical_symbols

        return False, f"{info.label} has no parameters for {', '.join(chemical_symbols[z] for z in missing)}"
    return True, ""


def auto_candidates(numbers, solvent: str = "") -> list[str]:
    """Usable engines for these elements in ``AUTO_ORDER``, then "mmff". With a solvent, engines that
    can model it come first (the gas-phase ones stay as fallbacks)."""
    load_plugins()
    keys = [k for k in AUTO_ORDER if k in ENGINES and available(k)[0] and supports(k, numbers)[0]]
    if solvent:
        keys = [k for k in keys if supports_solvent(k, solvent)] + [k for k in keys if not supports_solvent(k, solvent)]
    return [*keys, "mmff"]


def resolve_auto(numbers, solvent: str = "") -> str:
    """Best available engine for these elements (see ``AUTO_ORDER``), MMFF if nothing else runs."""
    return auto_candidates(numbers, solvent)[0]


# ---------------------------------------------------------------------------- dxtb
class DxtbCalculator(Calculator):
    """ASE calculator around dxtb (energies and autograd forces)."""

    implemented_properties: ClassVar[list[str]] = ["energy", "free_energy", "forces"]

    def __init__(self, numbers, method: str = "gfn1-xtb", charge: int = 0, multiplicity: int = 1,
                 device: str = "cpu", **kwargs):
        super().__init__(**kwargs)
        import dxtb
        import torch

        self._torch = torch
        self.device = torch.device(device)
        self.dtype = torch.float64 if self.device.type == "cpu" else torch.float32
        par = dxtb.GFN2_XTB if method == "gfn2-xtb" else dxtb.GFN1_XTB
        opts = {"verbosity": 0}
        if method == "gfn2-xtb":
            opts["int_driver"] = "libcint"
        self.charge = charge
        self.spin = multiplicity - 1
        self._numbers = torch.as_tensor(np.asarray(numbers), device=self.device)
        self._calc = dxtb.Calculator(self._numbers, par, opts=opts, device=self.device, dtype=self.dtype)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        torch = self._torch
        pos = torch.tensor(self.atoms.get_positions() / BOHR_ANG, device=self.device, dtype=self.dtype,
                           requires_grad=True)
        self._calc.reset()
        kwargs = {"chrg": self.charge}
        if self.spin:
            kwargs["spin"] = self.spin
        energy = self._calc.get_energy(pos, **kwargs)
        (grad,) = torch.autograd.grad(energy, pos)
        e_ev = float(energy.detach().cpu()) * HARTREE_EV
        self.results["energy"] = e_ev
        self.results["free_energy"] = e_ev
        self.results["forces"] = -grad.detach().cpu().double().numpy() * HARTREE_EV / BOHR_ANG


# ---------------------------------------------------------------------------- AIMNet2
_AIMNET_CACHE: dict[tuple[str, str, str], object] = {}


def _aimnet_base(model: str, device: str):
    """Cached AIMNet2 base calculator for ``device``: aimnet's stock ``AIMNet2Calculator`` where NVIDIA
    Warp serves the device (CUDA, CPU), the Warp-free ``aimnet_torch.AIMNet2TorchCalculator`` on ROCm,
    Apple GPUs or when the stock one fails at load (``QUANTABENCH_AIMNET_BACKEND`` forces either). Its
    ``qb_backend`` attribute says which one it is."""
    from . import aimnet_torch

    key = (model, device, aimnet_torch.requested_backend())
    if key not in _AIMNET_CACHE:
        _AIMNET_CACHE[key] = aimnet_torch.load_calculator(model, device)
    return _AIMNET_CACHE[key]


def make_calculator(key: str, numbers, charge: int = 0, multiplicity: int = 1, device: str = "cpu",
                    model: str = "aimnet2", **options) -> Calculator:
    """ASE calculator for engine ``key``. ``options``: ``solvent`` (canonical name, "" = gas phase),
    ``threads`` (for external programs); unknown options are ignored by engines that do not use them."""
    info = all_engines().get(key)
    if info is None:
        raise RuntimeError(f"unknown engine {key!r}")
    ok, why = available(key)
    if not ok:
        raise RuntimeError(f"{info.label} unavailable: {why}")
    ok, why = supports(key, numbers)
    if not ok:
        raise RuntimeError(why)
    p = plugin(key)
    if p is not None:
        return p.factory(numbers, charge, multiplicity, device, **options)
    if key in ("gfn1-xtb", "gfn2-xtb"):
        return DxtbCalculator(numbers, key, charge, multiplicity, device)
    if key == "aimnet2":
        from aimnet.calculators import AIMNet2ASE

        return AIMNet2ASE(_aimnet_base(model, device), charge=charge, mult=multiplicity)
    raise RuntimeError(f"{key} has no ASE calculator (MMFF runs inside RDKit)")


def clear_caches() -> None:
    _AIMNET_CACHE.clear()
