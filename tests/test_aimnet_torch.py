"""Warp-free AIMNet2 (``chem.aimnet_torch``): device routing, tad-dftd3 D3 parity with aimnet's
nvalchemiops D3, and full-model parity with the stock calculator on a random-weight AIMNet2.

The released weights cannot be downloaded in CI, so the model is aimnet's own ωB97M-D3 architecture
(``aimnet/models/aimnet2_dftd3_wb97m.yaml``) with random weights, converted to the released v2 file
layout (external Coulomb and D3, SRCoulomb embedded, float64 reference energies) exactly as
``aimnet.models.utils`` does. Tests that need the stock calculator skip when aimnet (and with it
NVIDIA Warp and nvalchemiops) cannot be imported.
"""

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tad_dftd3")

from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

from quantabench.chem import aimnet_torch as at  # noqa: E402

HAS_AIMNET = importlib.util.find_spec("aimnet") is not None
if HAS_AIMNET:
    try:
        import aimnet.calculators  # noqa: F401  (imports warp and nvalchemiops)
    except Exception:  # pragma: no cover - aimnet without its Warp dependencies
        HAS_AIMNET = False
needs_aimnet = pytest.mark.skipif(not HAS_AIMNET, reason="aimnet with warp-lang/nvalchemiops not importable")

MOLS = {
    "acetate-": "CC(=O)[O-]",
    "PFOA-": "[O-]C(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F",
    "mixed": "ClC1=CC(Br)=C(N)C(S[Si](C)(C)C)=C1OP(=O)(O)O",       # B-free mix of AIMNet2 elements
    "C20H42": "C" * 20,                                             # 19 Å: crosses the 12-15 Å D3 switch
}
#: rough ωB97M atomic reference energies (eV), so the float64/float32 bookkeeping is tested at real scale
SAE = {1: -13.6, 5: -673.0, 6: -1029.8, 7: -1484.9, 8: -2043.9, 9: -2715.6, 14: -7880.0, 15: -9283.0,
       16: -10832.0, 17: -12522.0, 33: -60870.0, 34: -65790.0, 35: -70045.0, 53: -8100.0}


def _geoms(smiles, n=1, seed=7):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=n, randomSeed=seed))
    AllChem.MMFFOptimizeMoleculeConfs(mol)
    z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()])
    x = np.stack([mol.GetConformer(i).GetPositions() for i in ids])
    return z, x, Chem.GetFormalCharge(mol)


def _batch(smiles, n=1):
    z, x, q = _geoms(smiles, n)
    return {"coord": torch.tensor(x, dtype=torch.float32), "numbers": torch.tensor(z).expand(len(x), -1),
            "charge": torch.full((len(x),), float(q)), "mult": torch.ones(len(x))}


def _raw_model(seed=0):
    """aimnet's ωB97M-D3 network as built from its YAML (embedded LRCoulomb and nvalchemiops DFTD3)."""
    import aimnet
    import yaml
    from aimnet.config import build_module

    cfg = yaml.safe_load((Path(aimnet.__file__).parent / "models" / "aimnet2_dftd3_wb97m.yaml").read_text())
    torch.manual_seed(seed)
    model = build_module(copy.deepcopy(cfg))
    model.outputs.atomic_shift.double()
    with torch.no_grad():
        w = model.outputs.atomic_shift.shifts.weight
        w.zero_()
        for z, v in SAE.items():
            w[z] = v
    return cfg, model.eval()


@pytest.fixture(scope="module")
def random_model(tmp_path_factory):
    """(path of a v2 .pt file, raw nn.Module with the same weights)."""
    if not HAS_AIMNET:
        pytest.skip("aimnet not importable")
    import yaml
    from aimnet.config import build_module
    from aimnet.models.utils import strip_lr_modules_from_yaml

    cfg, raw = _raw_model()
    core_cfg, coulomb_mode, needs_disp, d3p, rc, env, _ = strip_lr_modules_from_yaml(cfg, {})
    core = build_module(copy.deepcopy(core_cfg))
    core.outputs.atomic_shift.double()
    core.load_state_dict(raw.state_dict(), strict=False)
    assert coulomb_mode == "sr_embedded" and needs_disp and d3p == {**at.WB97M_D3BJ, "a1": 0.566, "a2": 3.128}
    meta = {"format_version": 2, "model_yaml": yaml.dump(core_cfg, default_flow_style=False, sort_keys=False),
            "cutoff": 5.0, "needs_coulomb": True, "needs_dispersion": True, "coulomb_mode": coulomb_mode,
            "coulomb_sr_rc": rc, "coulomb_sr_envelope": env, "d3_params": d3p, "has_embedded_lr": True,
            "has_embedded_d3ts": False, "implemented_species": sorted(SAE)}
    path = tmp_path_factory.mktemp("aimnet") / "random_aimnet2.pt"
    torch.save({**meta, "state_dict": core.state_dict()}, path)
    return str(path), raw


def _stock(path):
    from aimnet.calculators import AIMNet2Calculator

    calc = AIMNet2Calculator(path, device="cpu")
    at.guard_neighbor_lists(calc)
    return calc


# ---------------------------------------------------------------------------- routing
def test_backend_choice_by_device_and_env(monkeypatch):
    monkeypatch.delenv(at.BACKEND_ENV, raising=False)
    assert at.requested_backend() == "auto"
    assert "Apple" in at.warp_problem("mps") and at.choose_backend("mps")[0] == "torch"
    monkeypatch.setattr(torch.version, "hip", "7.2.0", raising=False)
    assert "ROCm" in at.warp_problem("cuda:0") and at.choose_backend("cuda:0")[0] == "torch"
    for value, expected in (("torch", "torch"), ("STOCK", "stock"), ("bogus", "auto")):
        monkeypatch.setenv(at.BACKEND_ENV, value)
        assert at.requested_backend() == expected
    monkeypatch.setenv(at.BACKEND_ENV, "stock")
    assert at.choose_backend("mps") == ("stock", f"{at.BACKEND_ENV}=stock")
    assert at.supports_float64("cpu") and not at.supports_float64("mps")


# ---------------------------------------------------------------------------- D3
def test_torch_d3_is_standard_d3_for_small_molecules():
    """Below the 12 Å switch TorchD3 is plain tad-dftd3 D3(BJ) (2-body) with the ωB97M parameters."""
    import tad_dftd3 as d3

    z, x, _ = _geoms(MOLS["acetate-"])
    numbers = torch.tensor(z)
    pos = torch.tensor(x[0], dtype=torch.float64)
    ref = d3.dftd3(numbers, pos / at.BOHR_ANG, {k: torch.tensor(v, dtype=torch.float64)
                                               for k, v in at.WB97M_D3BJ.items()}).sum() * at.HARTREE_EV
    for kw in ({}, {"dtype": torch.float32, "allow_float64": False}):
        e = at.TorchD3(**at.WB97M_D3BJ, **kw).energy(numbers[None], pos[None])
        assert abs(float(e[0]) - float(ref)) < 1e-6


@needs_aimnet
@pytest.mark.parametrize("name", list(MOLS))
def test_d3_matches_aimnet_nvalchemiops_d3(name):
    """aimnet's DFTD3 module (nvalchemiops/Warp kernel, CPU) vs TorchD3: energies and forces to 2e-6."""
    from aimnet.modules import DFTD3

    z, x, _ = _geoms(MOLS[name])
    coord = torch.tensor(x, dtype=torch.float32)
    stock = DFTD3(**{k: at.WB97M_D3BJ[k] for k in ("s6", "s8", "a1", "a2")})
    out, terms = stock({"coord": coord, "numbers": torch.tensor(z)[None], "_nb_mode": torch.tensor(0)},
                       compute_forces=True)
    e_ref, f_ref = out["energy"].double(), terms.forces.double()
    for kw in ({}, {"dtype": torch.float32, "allow_float64": False}):
        d3 = at.TorchD3(**at.WB97M_D3BJ, **kw)
        assert d3.source == "aimnet dftd3_data.pt"
        pos = coord.clone().requires_grad_(True)
        e = d3.energy(torch.tensor(z)[None], pos)
        (g,) = torch.autograd.grad(e.sum(), pos)
        assert float((e.detach().double() - e_ref).abs().max()) < 2e-6, kw
        assert float((-g.double() - f_ref).abs().max()) < 2e-6, kw


# ---------------------------------------------------------------------------- full model
@pytest.fixture(scope="module")
def calcs(random_model):
    """Shared calculators (each builds the full network): guarded stock, adapter with aimnet's float64
    accumulation, adapter on the float32-only (Apple GPU) code path."""
    path, _ = random_model
    return {"stock": _stock(path), "ad64": at.AIMNet2TorchCalculator(path),
            "ad32": at.AIMNet2TorchCalculator(path, dtype=torch.float32, allow_float64=False)}


@needs_aimnet
def test_full_model_matches_stock_calculator(calcs):
    """Energies, forces and charges of the dense Warp-free path vs aimnet's calculator (sparse
    neighbor lists, nvalchemiops D3), batched, in float64-accumulation and float32-only modes."""
    for smi, n in ((MOLS["PFOA-"], 2), (MOLS["mixed"], 1)):
        data = _batch(smi, n)
        ref = calcs["stock"](dict(data), forces=True)
        for key in ("ad64", "ad32"):
            out = calcs[key](dict(data), forces=True)
            assert out["energy"].dtype == torch.float64 and out["energy"].shape == (n,)
            assert float((out["energy"] - ref["energy"].detach().double()).abs().max()) < 1e-5, key
            assert float((out["forces"].double() - ref["forces"].double()).abs().max()) < 1e-5, key
            assert float((out["charges"].double() - ref["charges"].detach().double()).abs().max()) < 1e-5, key


@needs_aimnet
def test_hessian_matches_stock_calculator(calcs):
    z, x, q = _geoms(MOLS["acetate-"])
    data = {"coord": torch.tensor(x[0], dtype=torch.float32), "numbers": torch.tensor(z),
            "charge": torch.tensor([float(q)]), "mult": torch.tensor([1.0])}
    ref = calcs["stock"](dict(data), forces=True, hessian=True)["hessian"].double()
    n = 3 * len(z)
    for key, chunk in (("ad64", None), ("ad32", 5)):   # 5: several vmapped chunks
        calc = calcs[key]
        calc.hessian_chunk = chunk
        try:
            out = calc(dict(data), forces=True, hessian=True)
        finally:
            calc.hessian_chunk = None
        h = out["hessian"].double()
        assert h.shape == (len(z), 3, len(z), 3) and out["forces"].shape == (len(z), 3)
        assert float((h - ref).abs().max()) < 2e-5, key
        assert float((h.reshape(n, n) - h.reshape(n, n).T).abs().max()) < 2e-5, key


class _Float64Ops(torch.utils._python_dispatch.TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.ops: set[str] = set()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if any(isinstance(t, torch.Tensor) and t.dtype == torch.float64
               for t in torch.utils._pytree.tree_leaves(out)):
            self.ops.add(str(func))
        return out


@needs_aimnet
def test_float32_path_never_makes_float64_tensors(calcs, monkeypatch):
    """The Apple-GPU code path (no float64 on the device) run on the CPU: no op on the device side,
    forward or backward, produces a float64 tensor; energies keep full precision through the
    CPU-side reference energies."""
    calc = calcs["ad32"]
    assert all(t.dtype != torch.float64 for t in (*calc.model.parameters(), *calc.model.buffers()))
    assert calc.d3.dtype == torch.float32 and not calc.d3._use_tad_model
    seen = []
    device_eval = calc._device_eval

    def watched(*args, **kwargs):
        with _Float64Ops() as mode:
            out = device_eval(*args, **kwargs)
        seen.append(mode.ops)
        return out

    monkeypatch.setattr(calc, "_device_eval", watched)
    data = _batch(MOLS["acetate-"])
    out = calc(dict(data), forces=True)
    calc(dict(data, coord=data["coord"][0], numbers=data["numbers"][0]), forces=True, hessian=True)
    assert seen == [set(), set()]
    full = calcs["ad64"](dict(data), forces=True)
    assert abs(float(out["energy"][0]) - float(full["energy"][0])) < 1e-5   # |E| ~ 6e3 eV
    assert out["energy"].device.type == "cpu" and out["forces"].dtype == torch.float32


@needs_aimnet
def test_embedded_nvalchemiops_dftd3_is_replaced(random_model, calcs):
    """A raw YAML-built model (embedded LRCoulomb and nvalchemiops DFTD3) gives the same result as the
    converted file: the DFTD3 module is removed and its parameters go to tad-dftd3."""
    _, raw = random_model
    calc = at.AIMNet2TorchCalculator(copy.deepcopy(raw))
    assert "dftd3" not in calc.model.outputs and calc.d3 is not None and calc.coulomb is None
    assert calc.d3.params == {**at.WB97M_D3BJ, "a1": 0.566, "a2": 3.128}
    data = _batch(MOLS["acetate-"])
    a, b = calc(dict(data), forces=True), calcs["ad64"](dict(data), forces=True)
    assert float((a["energy"] - b["energy"]).abs().max()) < 1e-5
    assert float((a["forces"] - b["forces"]).abs().max()) < 1e-5


@needs_aimnet
def test_guarded_stock_neighbor_lists_grow_after_small_molecule(random_model, calcs):
    """aimnet 0.2 shrinks its neighbor buffer after a small molecule and (with nvalchemiops 0.5, which
    does not raise on overflow) would then truncate larger ones; the guard keeps results identical."""
    path, _ = random_model
    pfoa = _batch(MOLS["PFOA-"])
    calc = _stock(path)
    calc(dict(_batch("O")), forces=True)
    assert calc._nblist.max_neighbors == 16
    out = calc(dict(pfoa), forces=True)
    assert calc._nblist.max_neighbors > 16
    assert float((out["energy"].detach() - calcs["ad64"](dict(pfoa))["energy"]).abs().max()) < 1e-5


# ---------------------------------------------------------------------------- integration
@needs_aimnet
def test_load_calculator_routes_and_falls_back(random_model, monkeypatch):
    from aimnet.calculators import AIMNet2Calculator

    path, _ = random_model
    monkeypatch.setenv(at.BACKEND_ENV, "torch")
    assert isinstance(at.load_calculator(path, "cpu"), at.AIMNet2TorchCalculator)
    monkeypatch.setenv(at.BACKEND_ENV, "stock")
    stock = at.load_calculator(path, "cpu")
    assert isinstance(stock, AIMNet2Calculator) and stock.qb_backend == "stock"
    assert type(stock._nblist).__name__ == "CheckedAdaptiveNeighborList"
    monkeypatch.setenv(at.BACKEND_ENV, "auto")
    monkeypatch.setattr(at, "warp_problem", lambda device: "ROCm GPU: Warp supports NVIDIA CUDA only")
    calc = at.load_calculator(path, "cpu")
    assert calc.qb_backend == "torch" and "ROCm" in calc.qb_backend_reason
    monkeypatch.setattr(at, "warp_problem", lambda device: "")

    def broken(calc):
        raise RuntimeError("Invalid device identifier: cuda:0")

    monkeypatch.setattr(at, "_probe", broken)
    calc = at.load_calculator(path, "cpu")
    assert calc.qb_backend == "torch" and "Invalid device identifier" in calc.qb_backend_reason


@needs_aimnet
def test_engine_ase_hessian_and_refine_batched_use_adapter(random_model, calcs, monkeypatch):
    """make_calculator (ASE single point and optimization), vibrations.aimnet_hessian and
    refine_batched all run through the Warp-free calculator; the ASE optimization follows the stock
    calculator's trajectory."""
    from aimnet.calculators import AIMNet2ASE
    from ase import Atoms

    from quantabench.chem import batch_opt, engines, pipeline, vibrations
    from quantabench.chem.conformers import ConformerSet
    from quantabench.chem.inputs import from_smiles
    from quantabench.chem.optimize import optimize

    path, _ = random_model
    monkeypatch.setenv(at.BACKEND_ENV, "torch")
    engines.clear_caches()
    z, x, q = _geoms(MOLS["acetate-"])
    calc = engines.make_calculator("aimnet2", z, q, 1, "cpu", model=path)
    assert calc.base_calc.qb_backend == "torch"
    runs = []
    for c in (calc, AIMNet2ASE(calcs["stock"], charge=q, mult=1)):
        atoms = Atoms(numbers=z, positions=x[0])
        atoms.calc = c
        e0, f0 = atoms.get_potential_energy(), atoms.get_forces()
        res = optimize(atoms, 0.05, 5, "lbfgs", lambda: False)
        runs.append((e0, f0, res.steps, atoms.get_potential_energy(), atoms.get_positions()))
    (ea, fa, sa, e1a, xa), (eb, fb, sb, e1b, xb) = runs
    assert abs(ea - eb) < 1e-5 and np.abs(fa - fb).max() < 1e-5
    # same trajectory (float32 differences grow a little over L-BFGS steps on a random-weight surface)
    assert sa == sb > 0 and abs(e1a - e1b) < 1e-3 and np.abs(xa - xb).max() < 5e-3
    h = vibrations.aimnet_hessian(calc, atoms)
    assert h.shape == (3 * len(z), 3 * len(z)) and np.isfinite(h).all()

    monkeypatch.setattr(engines, "_aimnet_base", lambda model, device: calcs["ad64"])
    mol = Chem.AddHs(Chem.MolFromSmiles("[O-]C(=O)C(F)(F)F"))   # TFA anion, 2 conformers
    ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=2, randomSeed=7))
    AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=40)
    cs = ConformerSet(mol=mol, conf_ids=ids, energies_kcal=[0.0] * 2, forcefield="MMFF94s", n_embedded=2)
    result = pipeline.MoleculeResult(input=from_smiles("OC(=O)C(F)(F)F", id="TFA"), charge=-1, method="aimnet2",
                                     device="cpu")
    cfg = pipeline.BuildConfig(method="aimnet2", refine_top=2, fmax=0.05, max_steps=10, xtb_properties=False,
                               write_files=False)
    out = batch_opt.refine_batched(result, cs, cfg, "cpu", lambda: False, lambda m: None)
    assert len(out) == 2 and result.device == "cpu" and all(np.isfinite(c.energy_ev) for c in out)
    assert all(c.charges is not None and abs(sum(c.charges) + 1) < 1e-3 for c in out)
    model = batch_opt.aimnet2_batched("aimnet2", [a.GetAtomicNum() for a in mol.GetAtoms()], -1, 1, "cpu")
    pos = torch.tensor(np.stack([mol.GetConformer(c).GetPositions() for c in cs.conf_ids]))
    e, _ = model(pos)
    assert model.energies_ev.dtype == torch.float64 and torch.allclose(model.energies_ev, e.double())
    e_sub, _ = model(pos[1:], torch.tensor([1]))
    assert torch.allclose(model.energies_ev[1:], e_sub.double())
    engines.clear_caches()

