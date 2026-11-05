"""Energy/force engines behind one interface.

Each engine produces an ASE ``Calculator`` for a given molecule (atomic numbers, charge,
multiplicity) on a given torch device. Engines are optional: ``available()`` explains what is
missing instead of failing at import time, and the pipeline falls back from GPU to CPU if an
engine cannot run on the GPU it was given.

Engine        Package        Device        Charged molecules   Notes
-----------   -------------  ------------  ------------------  -----------------------------------
mmff          rdkit          CPU threads   yes                 already applied during conformer search
gfn1-xtb      dxtb           CPU or GPU    yes                 pure PyTorch: ROCm, CUDA, MPS (see torch_compat)
gfn2-xtb      dxtb+libcint   CPU (ints)    yes                 libcint backend (any OS with the dxtb fork)
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
    "gfn2-xtb": EngineInfo("gfn2-xtb", "GFN2-xTB (dxtb + libcint)", "GFN2 tight binding; needs dxtb's libcint backend (the dxtb fork on Windows/macOS).",
                           True, frozenset(range(1, 87)), batched=True),
    "aimnet2": EngineInfo("aimnet2", "AIMNet2 (ML, wB97M-D3)", "Neural network potential trained on wB97M-D3/def2-TZVPP; "
                          "neutral and charged molecules. GPU-capable: NVIDIA (stock aimnet), AMD ROCm and "
                          "Apple GPUs (Warp-free path, D3 from dxtb's tad-dftd3).", True, frozenset(AIMNET2_ELEMENTS),
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


_LIBCINT_STATUS: tuple[bool, str] | None = None

# dxtb 0.4 ported to tad-mctc >= 0.11 / tad-libcint >= 0.4 (FORK-1, docs/PLATFORMS.md section 4)
DXTB_FORK_REQ = "dxtb @ git+https://github.com/brandonbtm29/dxtb@qb-v0.4.0.post1"
LIBCINT_FORK_INSTALL = f'pip install "{DXTB_FORK_REQ}" "tad-libcint>=0.4"'


def _dist_version(name: str) -> str:
    from importlib import metadata

    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return ""


def libcint_status() -> tuple[bool, str]:
    """``(ok, reason)`` for dxtb's libcint integral backend; ``reason`` says what to install when not ok.

    Distinguishes tad-libcint missing from tad-libcint installed but incompatible: tad-libcint 0.4
    (the first release with Windows and macOS arm64 wheels) needs tad-mctc >= 0.11, which PyPI's
    dxtb 0.4 (pinned to tad-mctc 0.7) does not run with; the dxtb fork (``DXTB_FORK_REQ``) does.
    PyPI's dxtb 0.4 with ``tad-libcint<0.4`` (Linux only) stays valid. Cached after the first call.
    """
    global _LIBCINT_STATUS
    if _LIBCINT_STATUS is not None:
        return _LIBCINT_STATUS

    import importlib.util
    import sys

    cint, mctc, dxtb_v = _dist_version("tad-libcint"), _dist_version("tad-mctc"), _dist_version("dxtb")
    linux = sys.platform.startswith("linux")
    status = (True, "")
    if not dxtb_v and importlib.util.find_spec("dxtb") is None:
        status = (False, "dxtb is not installed")
    elif not cint:
        try:  # the fork's tad-mctc compatibility layer (finding it imports dxtb, which may fail)
            fork = importlib.util.find_spec("dxtb._src.exlibs.mctc_compat") is not None
        except Exception:
            fork = False
        if fork:
            hint = 'pip install "tad-libcint>=0.4"'
        elif linux:
            hint = f"pip install 'tad-libcint<0.4' (PyPI dxtb), or the dxtb fork: {LIBCINT_FORK_INSTALL}"
        else:
            hint = LIBCINT_FORK_INSTALL
        status = (False, f"tad-libcint is not installed: {hint}")
    else:
        try:
            import tad_libcint  # noqa: F401
        except Exception as exc:
            status = (False, f"tad-libcint {cint} is installed but does not import with tad-mctc {mctc or '?'} "
                             f"({type(exc).__name__}: {exc}); tad-libcint 0.4 needs dxtb from "
                             f"brandonbtm29/dxtb@qb-v0.4.0.post1: {LIBCINT_FORK_INSTALL}"
                             + (" (or keep PyPI dxtb with 'tad-libcint<0.4')" if linux else ""))
        else:
            try:
                from dxtb._src.exlibs import libcint

                libcint.LibcintWrapper  # noqa: B018 - the lazy import that fails on an incompatible pair
            except Exception as exc:
                status = (False, f"dxtb {dxtb_v or '?'} cannot use tad-libcint {cint} with tad-mctc {mctc or '?'} "
                                 f"({type(exc).__name__}: {exc}); {LIBCINT_FORK_INSTALL}")
    _LIBCINT_STATUS = status
    return status


def has_libcint() -> bool:
    """True when dxtb's libcint integral backend is importable *and* compatible.

    It provides the multipole integrals needed for GFN2-xTB and for any dxtb electric-field quantity
    (dipole moment, polarizability, IR intensities). On Windows and macOS it needs the dxtb fork with
    tad-libcint >= 0.4; ``libcint_status()`` gives the reason when it is False.
    """
    return libcint_status()[0]


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
            return False, f"GFN2 needs dxtb's libcint backend: {libcint_status()[1]}"
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
    """ASE calculator around dxtb (energies and autograd forces).

    float64 on the CPU, float32 on GPUs. ``device`` is a request: ``torch_compat.dxtb_device`` keeps
    dxtb on the CPU for GPUs where it is slower there (``device_note`` says why; ``device`` is where it
    runs). On a GPU without dense linear algebra (``QUANTABENCH_DXTB_GPU=on``) the Cholesky
    factorization / eigensolver run on the CPU inside each evaluation (``torch_compat.linalg_routing``);
    ``linalg_routed`` holds the ops routed in the last evaluation."""

    implemented_properties: ClassVar[list[str]] = ["energy", "free_energy", "forces"]

    def __init__(self, numbers, method: str = "gfn1-xtb", charge: int = 0, multiplicity: int = 1,
                 device: str = "cpu", **kwargs):
        super().__init__(**kwargs)
        import dxtb
        import torch

        from . import torch_compat

        self._torch = torch
        self._compat = torch_compat
        self.requested_device = str(device)
        run_on, self.device_note = torch_compat.dxtb_device(device)
        self.device = torch.device(run_on)
        self.dtype = torch_compat.compute_dtype(self.device)
        self.linalg_routed: frozenset[str] = frozenset()
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
        with self._compat.linalg_routing(self.device) as routed:
            energy = self._calc.get_energy(pos, **kwargs)
            (grad,) = torch.autograd.grad(energy, pos)
        self.linalg_routed = routed
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
