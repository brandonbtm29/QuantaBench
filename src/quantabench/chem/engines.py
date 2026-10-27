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
aimnet2       aimnet         CPU or GPU    yes                 wB97M-D3 quality for H,B,C,N,O,F,Si,P,S,Cl,As,Se,Br,I
"""

from __future__ import annotations

from dataclasses import dataclass
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


ENGINES: dict[str, EngineInfo] = {
    "mmff": EngineInfo("mmff", "MMFF94s (force field)", "RDKit MMFF94s; UFF when MMFF lacks parameters. Fast, CPU.",
                       False),
    "gfn1-xtb": EngineInfo("gfn1-xtb", "GFN1-xTB (dxtb)", "Semi-empirical tight binding in PyTorch. GPU-capable.", True,
                           frozenset(range(1, 87))),
    "gfn2-xtb": EngineInfo("gfn2-xtb", "GFN2-xTB (dxtb + libcint)", "GFN2 tight binding; needs tad-libcint (Linux/WSL).",
                           True, frozenset(range(1, 87))),
    "aimnet2": EngineInfo("aimnet2", "AIMNet2 (ML, wB97M-D3)", "Neural network potential trained on wB97M-D3/def2-TZVPP; "
                          "neutral and charged molecules. GPU-capable.", True, frozenset(AIMNET2_ELEMENTS)),
}


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
        return True, ""
    return False, f"unknown engine {key!r}"


def supports(key: str, numbers) -> tuple[bool, str]:
    info = ENGINES[key]
    if info.elements is None:
        return True, ""
    missing = sorted({int(z) for z in numbers} - set(info.elements))
    if missing:
        from ase.data import chemical_symbols

        return False, f"{info.label} has no parameters for {', '.join(chemical_symbols[z] for z in missing)}"
    return True, ""


def resolve_auto(numbers) -> str:
    """Best available engine for these elements: AIMNet2 > GFN2-xTB > GFN1-xTB > MMFF."""
    for key in ("aimnet2", "gfn2-xtb", "gfn1-xtb"):
        ok, _ = available(key)
        if ok and supports(key, numbers)[0]:
            return key
    return "mmff"


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
_AIMNET_CACHE: dict[tuple[str, str], object] = {}


def _aimnet_base(model: str, device: str):
    key = (model, device)
    if key not in _AIMNET_CACHE:
        from aimnet.calculators import AIMNet2Calculator

        try:
            _AIMNET_CACHE[key] = AIMNet2Calculator(model, device=device)
        except TypeError:  # older aimnet without the device argument
            _AIMNET_CACHE[key] = AIMNet2Calculator(model)
    return _AIMNET_CACHE[key]


def make_calculator(key: str, numbers, charge: int = 0, multiplicity: int = 1, device: str = "cpu",
                    model: str = "aimnet2") -> Calculator:
    ok, why = available(key)
    if not ok:
        raise RuntimeError(f"{ENGINES[key].label} unavailable: {why}")
    ok, why = supports(key, numbers)
    if not ok:
        raise RuntimeError(why)
    if key in ("gfn1-xtb", "gfn2-xtb"):
        return DxtbCalculator(numbers, key, charge, multiplicity, device)
    if key == "aimnet2":
        from aimnet.calculators import AIMNet2ASE

        return AIMNet2ASE(_aimnet_base(model, device), charge=charge, mult=multiplicity)
    raise RuntimeError(f"{key} has no ASE calculator (MMFF runs inside RDKit)")


def clear_caches() -> None:
    _AIMNET_CACHE.clear()
