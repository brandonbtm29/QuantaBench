"""torch_compat: dxtb's dense linear algebra on the CPU for GPUs that lack it (Windows ROCm, MPS).

Everything runs on the CPU here. A "broken device" is simulated with a TorchFunctionMode that makes
the dense ops raise (like AMD's MAGMA-less Windows wheel) unless they run inside a routed region, and
"MPS" by running dxtb in float32 with every op routed while watching for float64 tensors.
Molecules are water and CF3COOH so the whole file takes seconds.
"""

import importlib.util
import re

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule

from quantabench.chem import torch_compat as tc

torch = pytest.importorskip("torch")
from torch.overrides import TorchFunctionMode  # noqa: E402

NOVA_BROKEN = ("cholesky", "cholesky_solve")      # what AMD's Windows ROCm 7.2.1 wheel lacks (probed)
BOHR = 0.529177210903
HARTREE = 27.211386245988


def _dense(names):
    table = {
        "cholesky": (torch.linalg.cholesky, torch.linalg.cholesky_ex),
        "cholesky_solve": (torch.cholesky_solve,),
        "eigh": (torch.linalg.eigh, torch.linalg.eigvalsh),
        "solve": (torch.linalg.solve, torch.linalg.solve_ex, torch.inverse, torch.linalg.inv),
    }
    return {f for n in names for f in table.get(n, ())}      # 'float64' is not a function to break


class BrokenLinalg(TorchFunctionMode):
    """Dense ops raise outside routed regions, as on a GPU without MAGMA."""

    def __init__(self, names=NOVA_BROKEN):
        super().__init__()
        self.funcs = _dense(names)
        self.raised = 0

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if func in self.funcs and not tc.in_routed_region():
            self.raised += 1
            raise RuntimeError(f"simulated: {getattr(func, '__name__', func)} requires compiling PyTorch with MAGMA")
        return func(*args, **(kwargs or {}))


# float64 made on the CPU, never on the device: STO-nG expansion coefficients and tad-dftd3's reference
# C6/CN tables (loaded onto the default device, cast before moving), and tad-libcint 0.4's inputs for the C
# library (x.to(device="cpu", dtype=float64) -> numpy; the dxtb fork's integral driver)
CPU_CONSTANT_SITES = {("dxtb/_src/basis/slater.py", "slater_to_gauss"), ("tad_dftd3/reference.py", "_load_c6"),
                      ("tad_dftd3/reference.py", "_load_cn"), ("tad_libcint/basis.py", "_to_numpy")}


def _package_path(path: str) -> str:
    """'dxtb/...' or 'tad_dftd3/...' from a full path, wherever the package is installed; the dxtb fork's
    vendored copy (dxtb/_src/exlibs/tad_dftd3/...) maps to the same name as PyPI's tad_dftd3."""
    path = path.split("site-packages/")[-1]
    m = re.search(r"(?:^|/)((?:dxtb|tad_\w+)/.*)$", path)
    path = m.group(1) if m else path
    return path.split("dxtb/_src/exlibs/")[-1]


class Float64Watch(TorchFunctionMode):
    """Records where float64 tensors are made outside routed regions (the 'device' path), as
    (file, function) of the innermost caller outside torch."""

    def __init__(self):
        super().__init__()
        self.sites: set[tuple[str, str]] = set()

    def __torch_function__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if not tc.in_routed_region():
            items = out if isinstance(out, (tuple, list)) else (out,)
            if any(isinstance(t, torch.Tensor) and t.dtype == torch.float64 for t in items):
                import traceback

                for frame in reversed(traceback.extract_stack()[:-1]):
                    path = frame.filename.replace("\\", "/")
                    if "/torch/" not in path:
                        self.sites.add((_package_path(path), frame.name))
                        break
        return out

    @property
    def device_hits(self):
        return sorted(self.sites - CPU_CONSTANT_SITES)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(tc.ENV_VAR, raising=False)
    monkeypatch.delenv(tc.DXTB_GPU_ENV, raising=False)
    tc.set_policy("auto")
    tc.set_dxtb_gpu("auto")
    tc.clear_probe_cache()
    tc.reset_stats()
    yield
    tc.set_policy("auto")
    tc.set_dxtb_gpu("auto")
    tc.clear_probe_cache()


def _dxtb():
    pytest.importorskip("dxtb")


def _cf3cooh() -> Atoms:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles("OC(=O)C(F)(F)F"))
    AllChem.EmbedMolecule(mol, randomSeed=3)
    AllChem.MMFFOptimizeMolecule(mol)
    return Atoms([a.GetAtomicNum() for a in mol.GetAtoms()], positions=mol.GetConformer().GetPositions())


def _ef(atoms: Atoms, device: str = "cpu", calc=None):
    from quantabench.chem import engines

    a = atoms.copy()
    a.calc = calc or engines.DxtbCalculator(a.numbers, "gfn1-xtb", 0, 1, device)
    return a.get_potential_energy(), a.get_forces(), a.calc


# ============================================================================ policy and probe
def test_policy_env_overrides_setting(monkeypatch):
    tc.set_policy("device")
    assert tc.current_policy() == "device"
    monkeypatch.setenv(tc.ENV_VAR, "CPU")
    assert tc.current_policy() == "cpu"
    monkeypatch.setenv(tc.ENV_VAR, "gpu")
    with pytest.raises(ValueError, match=tc.ENV_VAR):
        tc.current_policy()
    with pytest.raises(ValueError):
        tc.set_policy("fast")


def test_routed_ops_by_device_and_policy(monkeypatch):
    def no_probe(*_a, **_k):
        raise AssertionError("must not probe")

    monkeypatch.setattr(tc, "probe", no_probe)
    assert tc.routed_ops("cpu") == frozenset()
    assert tc.routed_ops("mps") == tc.ALL_OPS                       # MPS: always, no probe
    assert tc.routed_ops("mps", "device") == frozenset()
    assert tc.routed_ops("cpu", "cpu") == tc.ALL_OPS
    monkeypatch.setattr(tc, "probe", lambda dev, refresh=False: tc.LinalgSupport(str(dev), {"cholesky": "MAGMA"}))
    assert tc.routed_ops("cuda:0") == {"cholesky"}                  # other GPUs: what the probe says
    assert "cholesky on the CPU" in tc.describe("cuda:0")


def test_dtype_helpers():
    assert tc.compute_dtype("cpu") == torch.float64
    assert tc.compute_dtype("cuda:0") == torch.float32
    assert tc.compute_dtype(torch.device("mps")) == torch.float32
    assert tc.float64_device("mps") == "cpu" and tc.float64_device("cuda:0") == "cuda:0"
    assert tc.device_type(torch.device("cuda", 1)) == "cuda"


def test_dxtb_device_policy(monkeypatch):
    """Default: dxtb stays on the CPU for GPUs that need routing (measured slower there)."""
    assert tc.dxtb_device("cpu") == ("cpu", "")
    dev, why = tc.dxtb_device("mps")
    assert dev == "cpu" and "float64" in why
    monkeypatch.setattr(tc, "probe", lambda d, refresh=False: tc.LinalgSupport(str(d), {"cholesky": "MAGMA"}))
    dev, why = tc.dxtb_device("cuda:0")
    assert dev == "cpu" and "cholesky" in why and tc.DXTB_GPU_ENV in why
    monkeypatch.setattr(tc, "probe", lambda d, refresh=False: tc.LinalgSupport(str(d)))
    assert tc.dxtb_device("cuda:0") == ("cuda:0", "")          # full linear algebra: keep the GPU
    tc.set_dxtb_gpu("off")
    assert tc.dxtb_device("cuda:0")[0] == "cpu"
    monkeypatch.setenv(tc.DXTB_GPU_ENV, "on")                   # the variable beats the setting
    assert tc.dxtb_device("mps") == ("mps", "")
    monkeypatch.setenv(tc.DXTB_GPU_ENV, "maybe")
    with pytest.raises(ValueError):
        tc.dxtb_device("cuda:0")


def test_dxtb_classes_follow_dxtb_device():
    """Offered 'mps' (auto), every dxtb entry point runs on the CPU and says so; no MPS needed."""
    _dxtb()
    from quantabench.chem import batch_opt, engines, properties
    from quantabench.chem import vibrations as vib

    atoms = molecule("H2O")
    e0, f0, _ = _ef(atoms)
    e1, f1, calc = _ef(atoms, "mps")
    assert calc.device.type == "cpu" and calc.requested_device == "mps" and "MPS" in calc.device_note
    assert calc.dtype == torch.float64 and abs(e1 - e0) < 1e-10 and np.abs(f1 - f0).max() < 1e-10
    model = batch_opt.dxtb_batched(atoms.numbers, "gfn1-xtb", 0, 1, "mps")
    x = torch.tensor(np.stack([atoms.positions] * 2), dtype=torch.float32)   # an MPS optimizer's dtype
    e, f = model(x)
    assert model.device.type == "cpu" and e.dtype == f.dtype == torch.float32
    assert abs(float(e[0]) - e0) < 1e-4
    p = properties.xtb_properties(atoms.numbers, atoms.positions, atoms.get_masses(), device="mps",
                                  polarizability=False)
    assert np.allclose(p["xtb_charges"], properties.xtb_properties(
        atoms.numbers, atoms.positions, atoms.get_masses(), polarizability=False)["xtb_charges"])
    assert vib.dxtb_run_device("mps") == "cpu" and vib.dxtb_run_device("cpu") == "cpu"
    assert engines.DxtbCalculator(atoms.numbers, device="cpu").device_note == ""


def test_probe_cpu_ok_and_cached():
    sup = tc.probe("cpu", refresh=True)
    assert sup.ok and sup.routed_ops == frozenset() and "OK" in sup.summary()
    assert tc.probe("cpu") is sup


def test_probe_flags_a_broken_device():
    with BrokenLinalg(NOVA_BROKEN):
        sup = tc.probe("cpu", refresh=True)
    assert sup.routed_ops == set(NOVA_BROKEN)
    assert "MAGMA" in sup.failed["cholesky"]
    assert tc.probe("cpu") is sup                     # cached until refreshed
    with BrokenLinalg(tc.ALL_OPS):
        assert tc.probe("cpu", refresh=True).routed_ops == tc.ALL_OPS - {"float64"}


# ============================================================================ the eigensolver wrapper
def _gen_problem(n=6, batch=(), pad=0, dtype=torch.float64, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(*batch, n, n, generator=g, dtype=dtype)
    a = 0.5 * (x + x.transpose(-1, -2))
    s = torch.rand(*batch, n, n, generator=g, dtype=dtype)
    b = 0.1 * (s @ s.transpose(-1, -2)) + torch.eye(n, dtype=dtype)
    if pad:   # zero-padded system, overlap with ones on the padding diagonal (as dxtb's get_overlap)
        a[..., -pad:, :] = 0
        a[..., :, -pad:] = 0
        b[..., -pad:, :] = 0
        b[..., :, -pad:] = 0
        b[..., range(n - pad, n), range(n - pad, n)] = 1
    return a, b


def _dxtb_eighb(a, b):
    from tad_mctc import storch

    return storch.eighb(a=a, b=b, is_posdef=True, factor=1e-12, broadening_method="lorn")


@pytest.mark.parametrize("ops", [tc.ALL_OPS, frozenset(NOVA_BROKEN)], ids=["all", "nova"])
def test_eighb_routed_cpu_to_cpu_is_identical(ops):
    pytest.importorskip("tad_mctc")
    a, b = _gen_problem(n=7, batch=(3,), pad=2)
    w0, v0 = _dxtb_eighb(a, b)
    with tc.linalg_routing("cpu", ops=ops):
        w1, v1 = _dxtb_eighb(a, b)
    assert torch.allclose(w0, w1, rtol=0, atol=1e-13) and torch.allclose(v0, v1, rtol=0, atol=1e-13)
    assert tc.routing_stats().get("eighb" if "eigh" in ops else "eighb_ortho") == 1


def test_eighb_float32_upcast_and_dtype_restored():
    pytest.importorskip("tad_mctc")
    a, b = _gen_problem(n=10)
    w64, _ = _dxtb_eighb(a, b)
    with tc.linalg_routing("cpu", ops=tc.ALL_OPS):
        w32, v32 = _dxtb_eighb(a.float(), b.float())
    assert w32.dtype == v32.dtype == torch.float32
    assert torch.allclose(w32.double(), w64, atol=1e-6)


@pytest.mark.parametrize("ops", [tc.ALL_OPS, frozenset(NOVA_BROKEN)], ids=["all", "nova"])
def test_eighb_gradients_through_routing_match_finite_differences(ops):
    """First and second derivatives (forces, Hessians) through the CPU-routed eigensolver."""
    pytest.importorskip("tad_mctc")
    n, k = 4, 2
    g = torch.Generator().manual_seed(1)
    x0 = torch.rand(n, n, generator=g, dtype=torch.float64, requires_grad=True)
    s0 = torch.rand(n, n, generator=g, dtype=torch.float64, requires_grad=True)
    occ = torch.tensor([2.0, 1.0], dtype=torch.float64)

    def f(x, s):
        a = 0.5 * (x + x.T) + torch.diag(torch.arange(n, dtype=x.dtype))   # well separated eigenvalues
        b = 0.1 * (s @ s.T) + torch.eye(n, dtype=x.dtype)
        w, v = _dxtb_eighb(a, b)
        density = (v[:, :k] * occ) @ v[:, :k].T                            # sign-invariant
        return (w[:k] * occ).sum() + (density * a).sum()

    with tc.linalg_routing("cpu", ops=ops):
        assert torch.autograd.gradcheck(f, (x0, s0))
        assert torch.autograd.gradgradcheck(f, (x0, s0))
    assert tc.routing_stats()


def test_context_patches_and_restores():
    pytest.importorskip("tad_mctc")
    pytest.importorskip("tad_multicharge")
    import tad_mctc.storch
    import tad_multicharge

    eighb, eeq = tad_mctc.storch.eighb, tad_multicharge.get_eeq_charges
    with tc.linalg_routing("cpu", ops=tc.ALL_OPS) as ops:
        assert ops == tc.ALL_OPS
        assert tad_mctc.storch.eighb is not eighb and tad_mctc.storch.eighb.__wrapped__ is eighb
        with tc.linalg_routing("cpu", ops={"cholesky"}):              # nested: still installed once
            assert tad_mctc.storch.eighb.__wrapped__ is eighb
        assert tad_mctc.storch.eighb is not eighb
    assert tad_mctc.storch.eighb is eighb and tad_multicharge.get_eeq_charges is eeq
    with pytest.raises(ZeroDivisionError), tc.linalg_routing("cpu", ops=tc.ALL_OPS):
        _ = 1 / 0
    assert tad_mctc.storch.eighb is eighb
    with tc.linalg_routing("cpu", ops=frozenset()) as ops:            # nothing to route: no patching
        assert ops == frozenset() and tad_mctc.storch.eighb is eighb
    with pytest.raises(ValueError, match="unknown"):
        with tc.linalg_routing("cpu", ops={"svd"}):
            pass


def test_other_devices_pass_through():
    """A route for one device leaves tensors on other devices alone."""
    pytest.importorskip("tad_mctc")
    a, b = _gen_problem()
    with tc.linalg_routing("meta", ops=tc.ALL_OPS):
        _dxtb_eighb(a, b)
    assert tc.routing_stats() == {}


# ============================================================================ dxtb through QuantaBench
def test_dxtb_forced_routing_identical_to_stock(monkeypatch):
    _dxtb()
    atoms = _cf3cooh()
    e0, f0, c0 = _ef(atoms)
    assert c0.linalg_routed == frozenset()                       # CPU, auto: stock dxtb
    monkeypatch.setenv(tc.ENV_VAR, "cpu")
    e1, f1, c1 = _ef(atoms)
    assert c1.linalg_routed == tc.ALL_OPS
    assert abs(e1 - e0) < 1e-10 and np.abs(f1 - f0).max() < 1e-10
    stats = tc.routing_stats()
    assert stats["eighb"] >= 5 and stats["eeq"] == 1 and stats.get("solve", 0) >= 1   # SCF steps, guess, mixer
    status = tc.patch_status()
    assert set(status.values()) <= {"patched", tc.ABSENT}
    try:                                                       # the dxtb fork (tad-mctc 0.11) vendors tad-dftd3
        fork = importlib.util.find_spec("dxtb._src.exlibs.tad_dftd3") is not None
    except ModuleNotFoundError:
        fork = False
    d3 = "dxtb._src.exlibs.tad_dftd3.model" if fork else "tad_dftd3.model"
    assert status[f"{d3}.weight_references"] == "patched"


def test_broken_device_stock_fails_routing_works(monkeypatch):
    """The NOVA case: cholesky and cholesky_solve raise on the device, eigh and solve work."""
    _dxtb()
    from quantabench.chem import engines

    atoms = molecule("H2O")
    e0, f0, _ = _ef(atoms)
    calc = engines.DxtbCalculator(atoms.numbers, "gfn1-xtb", 0, 1, "cpu")   # float64, built first
    monkeypatch.setattr(tc, "device_type", lambda _dev: "cuda")             # the CPU poses as a GPU
    with BrokenLinalg(NOVA_BROKEN) as broken:
        monkeypatch.setenv(tc.ENV_VAR, "device")                            # stock dxtb
        with pytest.raises(RuntimeError, match="MAGMA"):
            _ef(atoms, calc=calc)
        monkeypatch.setenv(tc.ENV_VAR, "auto")                              # probe -> route
        e1, f1, calc = _ef(atoms, calc=calc)
    assert broken.raised >= 2
    assert calc.linalg_routed == set(NOVA_BROKEN)
    stats = tc.routing_stats()
    assert stats["eighb_ortho"] >= 5 and stats["eeq"] == 1 and "eighb" not in stats   # eigh stayed on the device
    assert abs(e1 - e0) < 1e-10 and np.abs(f1 - f0).max() < 1e-10


def test_all_ops_broken_like_mps(monkeypatch):
    _dxtb()
    atoms = molecule("H2O")
    e0, f0, _ = _ef(atoms)
    monkeypatch.setenv(tc.ENV_VAR, "cpu")
    with BrokenLinalg(tc.ALL_OPS):
        e1, f1, _ = _ef(atoms)
    assert abs(e1 - e0) < 1e-10 and np.abs(f1 - f0).max() < 1e-10


def test_float32_with_float64_linalg_close_to_float64(monkeypatch):
    """GPU precision: dxtb in float32 (linear algebra upcast to float64 on the CPU) vs float64."""
    _dxtb()
    atoms = _cf3cooh()
    e64, f64, _ = _ef(atoms)
    monkeypatch.setattr(tc, "compute_dtype", lambda _dev: torch.float32)
    monkeypatch.setenv(tc.ENV_VAR, "cpu")
    e32, f32, calc = _ef(atoms)
    assert calc.dtype == torch.float32
    # measured: 1.0e-6 Eh and 9e-7 Eh/Bohr (PFOA: 8.6e-6 Eh, 2.4e-6 Eh/Bohr); float32 itself limits it
    assert abs(e32 - e64) / HARTREE < 1e-5
    assert np.abs(f32 - f64).max() / (HARTREE / BOHR) < 1e-5


def test_forces_through_routed_ops_match_finite_differences(monkeypatch):
    _dxtb()
    monkeypatch.setenv(tc.ENV_VAR, "cpu")
    atoms = molecule("H2O")
    atoms.positions[0] += [0.02, -0.03, 0.01]          # off the minimum: non-zero forces
    _, forces, calc = _ef(atoms)
    h = 1e-3
    for atom, axis in ((0, 0), (1, 2), (2, 1)):
        e = []
        for sign in (1, -1):
            a = atoms.copy()
            a.positions[atom, axis] += sign * h
            e.append(_ef(a, calc=calc)[0])
        assert abs(-(e[0] - e[1]) / (2 * h) - forces[atom, axis]) < 2e-4      # eV/Å


def test_hessian_through_routed_ops(monkeypatch):
    """Analytic Hessian (double backward through the routed ops) = stock = finite differences."""
    _dxtb()
    from quantabench.chem import vibrations as vib

    atoms = molecule("H2O")
    h0 = vib.dxtb_hessian(atoms.numbers, atoms.positions, "gfn1-xtb")
    monkeypatch.setenv(tc.ENV_VAR, "cpu")
    with BrokenLinalg(tc.ALL_OPS):
        h1 = vib.dxtb_hessian(atoms.numbers, atoms.positions, "gfn1-xtb")
    assert np.abs(h1 - h0).max() < 1e-8
    fd = np.zeros_like(h0)
    from quantabench.chem import engines

    calc = engines.DxtbCalculator(atoms.numbers, "gfn1-xtb", 0, 1, "cpu")
    step = 1e-3
    for i in range(3 * len(atoms)):
        f = []
        for sign in (1, -1):
            a = atoms.copy()
            a.positions.flat[i] += sign * step
            f.append(_ef(a, calc=calc)[1].ravel())
        fd[:, i] = -(f[0] - f[1]) / (2 * step)
    assert np.abs(fd - h1).max() < 0.02 * np.abs(h1).max()


def test_batched_model_routed(monkeypatch):
    _dxtb()
    from quantabench.chem import batch_opt

    atoms = molecule("H2O")
    x = torch.tensor(np.stack([atoms.positions, atoms.positions * 1.02]))
    model = batch_opt.dxtb_batched(atoms.numbers, "gfn1-xtb", 0, 1, "cpu")
    e0, f0 = model(x)
    monkeypatch.setenv(tc.ENV_VAR, "cpu")
    with BrokenLinalg(tc.ALL_OPS):
        e1, f1 = model(x)
        e2, f2 = model(x[:1], torch.tensor([0]))          # b = 1 uses dxtb's unbatched calculator
    assert model.linalg_routed == tc.ALL_OPS
    assert torch.allclose(e0, e1, atol=1e-10) and torch.allclose(f0, f1, atol=1e-10)
    assert torch.allclose(e2, e0[:1], atol=1e-8) and torch.allclose(f2, f0[:1], atol=1e-6)


def test_properties_routed(monkeypatch):
    _dxtb()
    from quantabench.chem import properties

    atoms = molecule("H2O")
    args = (atoms.numbers, atoms.positions, atoms.get_masses())
    p0 = properties.xtb_properties(*args)
    monkeypatch.setenv(tc.ENV_VAR, "cpu")
    with BrokenLinalg(tc.ALL_OPS):
        p1 = properties.xtb_properties(*args)
    assert np.allclose(p0["xtb_charges"], p1["xtb_charges"], atol=1e-10)
    for key in ("xtb_dipole_debye", "xtb_polarizability_iso_A3"):
        if key in p0:
            assert abs(p0[key] - p1[key]) < 1e-8


# ============================================================================ MPS logic on the CPU
@pytest.mark.parametrize("int_driver", ["dxtb", None], ids=["pytorch-ints", "default-ints"])
def test_no_float64_on_the_device_path_in_float32(monkeypatch, int_driver):
    """With dxtb in float32 and every op routed (MPS), no float64 tensor is made on the device path:
    batched and single-system models, Mulliken charges. Without routing, tad-dftd3's D3 weights are
    float64 (MPS would raise). 'dxtb' = PyTorch integrals, as on Windows/macOS without tad-libcint."""
    _dxtb()
    import dxtb

    from quantabench.chem import batch_opt, properties

    monkeypatch.setattr(tc, "compute_dtype", lambda _dev: torch.float32)
    atoms = _cf3cooh()
    x = torch.tensor(np.stack([atoms.positions, atoms.positions * 1.01]), dtype=torch.float32)
    model = batch_opt.dxtb_batched(atoms.numbers, "gfn1-xtb", 0, 1, "cpu")
    if int_driver:
        model._opts["int_driver"] = int_driver
    stock = Float64Watch()
    with stock:
        model(x)
    assert ("tad_dftd3/model/weights.py", "weight_references") in stock.device_hits

    monkeypatch.setenv(tc.ENV_VAR, "cpu")
    watch = Float64Watch()
    with watch:
        e, f = model(x)
        e1, f1 = model(x[:1], torch.tensor([0]))
    assert e.dtype == f.dtype == e1.dtype == torch.float32
    assert watch.device_hits == []
    assert tc.routing_stats()["eighb"] > 0 and tc.routing_stats()["d3_weights"] == 2

    orig = dxtb.Calculator.get_mulliken_charges

    def watched(self, *a, **k):
        with watch:
            return orig(self, *a, **k)

    monkeypatch.setattr(dxtb.Calculator, "get_mulliken_charges", watched)
    out = properties.xtb_properties(atoms.numbers, atoms.positions, atoms.get_masses(), polarizability=False)
    assert len(out["xtb_charges"]) == len(atoms)
    assert watch.device_hits == []


def test_mps_hessian_runs_on_cpu_in_float64(monkeypatch):
    _dxtb()
    from quantabench.chem import vibrations as vib

    seen = {}
    orig = vib._dxtb_setup

    def spy(numbers, positions, method, device, field=False):
        calc, pos = orig(numbers, positions, method, device, field)
        seen["device"], seen["dtype"] = pos().device.type, pos().dtype
        return calc, pos

    monkeypatch.setattr(vib, "_dxtb_setup", spy)
    atoms = molecule("H2O")
    vib.dxtb_hessian(atoms.numbers, atoms.positions, "gfn1-xtb", device="mps")
    assert seen == {"device": "cpu", "dtype": torch.float64}


def test_settings_reach_the_policies_at_every_entry_point(capsys):
    """apply_settings takes a BuildConfig, ResourceSettings or Settings; the CLI applies Settings > Compute
    for every command (freq and doctor build no BuildConfig); invalid stored values mean 'auto' there."""
    from quantabench import cli
    from quantabench import settings as sm
    from quantabench.chem.pipeline import BuildConfig

    tc.apply_settings(BuildConfig(linalg="cpu", dxtb_gpu="on"))
    assert tc.current_policy() == "cpu" and tc.dxtb_gpu_mode() == "on"
    s = sm.Settings()
    s.resources.linalg, s.resources.dxtb_gpu = "device", "off"
    tc.apply_settings(s.resources)
    assert tc.current_policy() == "device" and tc.dxtb_gpu_mode() == "off"
    tc.set_policy("auto")
    tc.set_dxtb_gpu("auto")
    sm.save(s)
    assert cli.main(["settings"]) == 0
    capsys.readouterr()
    assert tc.current_policy() == "device" and tc.dxtb_gpu_mode() == "off"
    s.resources.dxtb_gpu = "sometimes"
    with pytest.raises(ValueError):
        tc.apply_settings(s)
    tc.apply_settings(s, strict=False)
    assert tc.dxtb_gpu_mode() == "auto" and tc.current_policy() == "device"
