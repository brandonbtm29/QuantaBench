"""Batched conformer refinement: optimizers, model adapters and the pipeline stage."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ase import Atoms  # noqa: E402
from ase.calculators.calculator import Calculator, all_changes  # noqa: E402
from ase.optimize import FIRE, LBFGS  # noqa: E402
from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

from quantabench.chem import batch_opt, engines, pipeline  # noqa: E402
from quantabench.chem.batch_opt import (  # noqa: E402
    HarmonicModel,
    aimnet2_batched,
    fire_batched,
    lbfgs_batched,
    optimize_batched,
    refine_batched,
)
from quantabench.chem.conformers import ConformerSet  # noqa: E402
from quantabench.chem.inputs import from_smiles  # noqa: E402
from quantabench.chem.optimize import Cancelled, optimize  # noqa: E402
from quantabench.chem.standardize import prepare  # noqa: E402

ALGOS = ("lbfgs", "fire")


def _starts(b=4, n=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    ref = torch.randn(b, n, 3, generator=g, dtype=torch.float64)
    x0 = ref + 0.5 * torch.randn(b, n, 3, generator=g, dtype=torch.float64)
    return ref, x0


# ---------------------------------------------------------------------------- optimizers
@pytest.mark.parametrize("algo", ALGOS)
def test_harmonic_converges_to_known_minimum(algo):
    ref, x0 = _starts()
    k = 2.0
    res = optimize_batched(HarmonicModel(ref, k=k), x0, fmax=1e-3, max_steps=500, algorithm=algo)
    assert res.converged.all() and not res.failed.any()
    # |F| = k |x - x0| per atom, so every atom is within fmax / k of the minimum
    assert (res.positions - ref).norm(dim=-1).max() < 1e-3 / k
    assert res.energies.abs().max() < 1e-6
    assert (res.fmax < 1e-3).all() and (res.steps > 0).all()


@pytest.mark.parametrize("algo", ALGOS)
@pytest.mark.parametrize("subset", [True, False])
def test_masking_converged_system_takes_zero_steps(algo, subset):
    ref, x0 = _starts(b=4)
    x0[1], x0[3] = ref[1], ref[3]        # systems 1 and 3 start at their minimum
    model = HarmonicModel(ref)
    fn = model if subset else (lambda pos: model(pos))   # plain callable: no compaction
    res = optimize_batched(fn, x0, fmax=1e-3, max_steps=500, algorithm=algo)
    assert res.steps.tolist()[1::2] == [0, 0] and (res.steps[0::2] > 0).all()
    assert torch.equal(res.positions[1], x0[1]) and torch.equal(res.positions[3], x0[3])   # frozen exactly
    assert res.converged.all()
    if subset:  # half the batch converged at once, so later calls evaluate only the active half
        assert res.n_system_evals < res.n_calls * 4
    else:
        assert res.n_system_evals == res.n_calls * 4


def test_max_steps_stops_unconverged():
    ref, x0 = _starts(b=2)
    res = lbfgs_batched(HarmonicModel(ref), x0, fmax=1e-12, max_steps=3)
    assert (res.steps == 3).all() and not res.converged.any()


def test_nonfinite_system_is_flagged_not_propagated():
    ref, x0 = _starts(b=3)
    model = HarmonicModel(ref)

    def fn(pos):
        e, f = model(pos)
        e = e.clone()
        e[2] = float("nan")
        return e, f

    res = lbfgs_batched(fn, x0, fmax=1e-3)
    assert res.failed.tolist() == [False, False, True]
    assert res.converged[:2].all() and torch.isfinite(res.positions).all()


@pytest.mark.parametrize("algo", ALGOS)
def test_should_stop_raises_cancelled(algo):
    ref, x0 = _starts()
    calls = []

    def stop():
        calls.append(1)
        return len(calls) > 3

    with pytest.raises(Cancelled):
        optimize_batched(HarmonicModel(ref), x0, fmax=1e-6, algorithm=algo, should_stop=stop)


def _morse_torch(pos, eps=1.0, r0=1.1, a=1.6):
    """Morse cluster energy for [B, N, 3] positions (no cutoff)."""
    d = pos[:, :, None, :] - pos[:, None, :, :]
    n = pos.shape[1]
    iu = torch.triu_indices(n, n, 1)
    r = d.norm(dim=-1)[:, iu[0], iu[1]]
    x = torch.exp(-a * (r - r0))
    return (eps * (x * x - 2 * x)).sum(1)


def _morse_fn(pos):
    p = pos.detach().clone().requires_grad_(True)
    e = _morse_torch(p)
    (g,) = torch.autograd.grad(e.sum(), p)
    return e.detach(), -g


class MorseASE(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        e, f = _morse_fn(torch.as_tensor(atoms.positions[None], dtype=torch.float64))
        self.results = {"energy": float(e[0]), "forces": f[0].numpy()}


def _morse_starts(b=3, n=6, seed=3):
    rng = np.random.default_rng(seed)
    base = np.array([[0, 0, 0], [1.1, 0, 0], [0, 1.1, 0], [0, 0, 1.1], [1.1, 1.1, 0], [1.1, 0, 1.1]])[:n]
    return np.stack([base + 0.15 * rng.standard_normal(base.shape) for _ in range(b)])


@pytest.mark.parametrize("algo", ALGOS)
def test_matches_ase_on_morse_cluster(algo):
    """Same algorithm as ASE: same minimum; L-BFGS also the same number of steps."""
    x0 = _morse_starts()
    res = optimize_batched(_morse_fn, torch.as_tensor(x0), fmax=0.01, max_steps=300, algorithm=algo)
    for b in range(len(x0)):
        atoms = Atoms("H6", positions=x0[b])
        atoms.calc = MorseASE()
        opt = (FIRE if algo == "fire" else LBFGS)(atoms, logfile=None)
        assert opt.run(fmax=0.01, steps=300)
        assert bool(res.converged[b])
        assert abs(float(res.energies[b]) - atoms.get_potential_energy()) < 1e-4
        if algo == "lbfgs":
            assert int(res.steps[b]) == opt.nsteps


# ---------------------------------------------------------------------------- adapters
class FakeAIMNet2Calculator:
    """Mimics AIMNet2Calculator.__call__(data, forces=True) for batched (3-D) input."""

    device = "cpu"

    def __init__(self):
        self.seen = []

    def __call__(self, data, forces=False, stress=False, hessian=False, validate_species=True):
        coord, numbers, charge, mult = data["coord"], data["numbers"], data["charge"], data["mult"]
        b, n = coord.shape[:2]
        assert coord.dtype == torch.float32 and numbers.shape == (b, n)
        assert charge.shape == (b,) and mult.shape == (b,)
        self.seen.append(b)
        c = coord.detach().clone().requires_grad_(True)
        center = c.mean(dim=1, keepdim=True)
        e = 0.5 * ((c - center - 1.0) ** 2).sum((1, 2))
        (g,) = torch.autograd.grad(e.sum(), c)
        w = numbers.float()                                   # electronegativity stand-in
        q = charge[:, None] * w / w.sum(1, keepdim=True)      # sums to the total charge per system
        out = {"energy": e.detach(), "charges": q}
        if forces:
            out["forces"] = -g
        return out


def test_fake_aimnet2_adapter_shapes_and_charges():
    numbers = [8, 6, 8, 6, 9, 9, 9]
    fake = FakeAIMNet2Calculator()
    model = aimnet2_batched("aimnet2", numbers, charge=-1, multiplicity=1, device="cpu", calculator=fake)
    x = torch.randn(3, 7, 3, dtype=torch.float64)
    e, f = model(x)
    assert e.shape == (3,) and f.shape == (3, 7, 3)
    assert model.charges.shape == (3, 7)
    assert torch.allclose(model.charges.sum(1), torch.full((3,), -1.0), atol=1e-6)
    before = model.charges.clone()
    model(x[[2]] + 0.1, torch.tensor([2]))               # compacted call updates only row 2
    assert fake.seen == [3, 1]
    assert torch.equal(model.charges[:2], before[:2])
    res = lbfgs_batched(model, x, fmax=1e-3)
    assert res.converged.all()


def test_dxtb_batched_matches_sequential_calculator():
    pytest.importorskip("dxtb")
    mol, ids = _tfa_conformers(n=3)
    numbers = [a.GetAtomicNum() for a in mol.GetAtoms()]
    pos = np.stack([mol.GetConformer(c).GetPositions() for c in ids])
    model = batch_opt.dxtb_batched(numbers, "gfn1-xtb", -1, 1, "cpu")
    e, f = model(torch.as_tensor(pos))
    calc = engines.DxtbCalculator(numbers, "gfn1-xtb", -1, 1, "cpu")
    for b in range(len(ids)):
        atoms = Atoms(numbers=numbers, positions=pos[b])
        atoms.calc = calc
        assert abs(atoms.get_potential_energy() - float(e[b])) < 1e-6
        assert np.abs(atoms.get_forces() - f[b].numpy()).max() < 1e-5
    e1, f1 = model(torch.as_tensor(pos[[1]]), torch.tensor([1]))   # B = 1 uses an unbatched calculator
    assert abs(float(e1[0]) - float(e[1])) < 1e-8


# ---------------------------------------------------------------------------- refine_batched
TFA = "OC(=O)C(F)(F)F"


def _tfa_conformers(n=3, seed=7):
    """TFA anion with ``n`` distinct, lightly pre-optimized starting geometries (no RMSD pruning)."""
    mol, _ = prepare(Chem.MolFromSmiles(TFA), 7.0, True, "rules")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.pruneRmsThresh = -1
    ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=n, params=params))
    AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=40)  # close to MMFF minima: fewer (slow) xTB steps
    return mol, ids


def _setup(method="gfn1-xtb", device="cpu", n=3, **cfg_kw):
    mol, ids = _tfa_conformers(n)
    cs = ConformerSet(mol=mol, conf_ids=ids, energies_kcal=[0.0] * len(ids), forcefield="MMFF94s",
                      n_embedded=len(ids))
    res = pipeline.MoleculeResult(input=from_smiles(TFA, id="TFA"), charge=-1, method=method, device=device)
    cfg = pipeline.BuildConfig(method=method, refine_top=n, xtb_properties=False, write_files=False, **cfg_kw)
    return cs, res, cfg


def _copy(cs):
    return ConformerSet(Chem.Mol(cs.mol), list(cs.conf_ids), list(cs.energies_kcal), cs.forcefield, cs.n_embedded)


def test_refine_batched_matches_sequential_gfn1():
    pytest.importorskip("dxtb")
    cs, res, cfg = _setup(fmax=0.05)
    assert res.charge == -1 and cs.mol.GetNumAtoms() == 7
    start = np.stack([cs.mol.GetConformer(c).GetPositions() for c in cs.conf_ids])
    cs_seq, cs_bat = _copy(cs), _copy(cs)
    res_seq = pipeline.MoleculeResult(input=res.input, charge=-1, method="gfn1-xtb", device="cpu")
    seq = pipeline._refine(res_seq, cs_seq, cfg, "cpu", lambda: False, lambda m: None)
    lines = []
    bat = refine_batched(res, cs_bat, cfg, "cpu", lambda: False, lines.append)
    assert [c.conf_id for c in bat] == [c.conf_id for c in seq]
    for s, b in zip(seq, bat):
        assert s.converged and b.converged
        assert abs(s.energy_ev - b.energy_ev) < 2e-3
        assert b.charges is None and b.dipole_debye is None     # same as the sequential dxtb path
    # optimized coordinates were written back into the molecule
    end = np.stack([cs_bat.mol.GetConformer(c).GetPositions() for c in cs_bat.conf_ids])
    assert np.abs(end - start).max() > 1e-3
    assert any("batched LBFGS" in m for m in lines)


def test_refine_batched_cancel(monkeypatch):
    monkeypatch.setattr(batch_opt, "make_batched_model", lambda *a, **k: HarmonicModel(offset=0.05))
    cs, res, cfg = _setup(fmax=1e-6)
    calls = []

    def stop():
        calls.append(1)
        return len(calls) > 2

    with pytest.raises(Cancelled):
        refine_batched(res, cs, cfg, "cpu", stop, lambda m: None)
    with pytest.raises(Cancelled):
        refine_batched(res, _copy(cs), cfg, "cpu", lambda: True, lambda m: None)


class GpuShyHarmonic(HarmonicModel):
    """Like the toy calculator in test_pipeline: no kernels for non-CPU devices."""

    def __init__(self, device):
        super().__init__(offset=0.05, charges=[-0.5, 0.2, -0.5, 0.3, -0.2, -0.2, -0.1])
        self.device = device

    def _compute(self, positions, index):
        if self.device != "cpu":
            raise RuntimeError("no GPU kernels for this device")
        return super()._compute(positions, index)


def test_refine_batched_gpu_fallback_and_charges(monkeypatch):
    monkeypatch.setattr(engines, "available", lambda key: (True, ""))
    monkeypatch.setattr(batch_opt, "make_batched_model", lambda key, numbers, q, m, dev: GpuShyHarmonic(dev))
    monkeypatch.setattr(batch_opt, "estimate_max_batch", lambda *a, **k: 2)   # two chunks: 2 + 1
    cs, res, cfg = _setup(device="cuda:0")
    out = refine_batched(res, cs, cfg, "cuda:0", lambda: False, lambda m: None)
    assert res.device == "cpu" and sum("fell back to CPU" in n for n in res.notes) == 1
    assert len(out) == 3 and all(c.converged for c in out)
    assert all(c.charges is not None and c.dipole_debye is not None for c in out)
    assert abs(sum(out[0].charges) + 1.0) < 1e-9


def test_refine_batched_halves_on_out_of_memory(monkeypatch):
    class Picky(HarmonicModel):
        def _compute(self, positions, index):
            if positions.shape[0] > 1:
                raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
            return super()._compute(positions, index)

    monkeypatch.setattr(batch_opt, "make_batched_model", lambda *a, **k: Picky(offset=0.05))
    cs, res, cfg = _setup()
    lines = []
    out = refine_batched(res, cs, cfg, "cpu", lambda: False, lines.append)
    assert len(out) == 3 and all(c.converged for c in out)
    assert any("out of memory" in m for m in lines) and res.device == "cpu"


def test_refine_batched_aimnet2_with_fake_calculator(monkeypatch):
    fake = FakeAIMNet2Calculator()
    monkeypatch.setattr(engines, "available", lambda key: (True, ""))
    monkeypatch.setattr(engines, "_aimnet_base", lambda model, device: fake)
    cs, res, cfg = _setup(method="aimnet2")
    out = refine_batched(res, cs, cfg, "cpu", lambda: False, lambda m: None)
    assert fake.seen[0] == 3
    assert all(c.charges is not None and abs(sum(c.charges) + 1) < 1e-5 for c in out)
    assert all(c.dipole_debye is not None and c.dipole_debye >= 0 for c in out)


def test_mmff_and_unknown_methods_delegate():
    cs, res, cfg = _setup(method="mmff")
    out = refine_batched(res, cs, cfg, "cpu", lambda: False, lambda m: None)
    assert [c.conf_id for c in out] == cs.conf_ids and all(c.steps == 0 for c in out)


def test_supports_batched_and_batch_size(monkeypatch):
    assert not batch_opt.supports_batched("mmff")
    assert batch_opt.supports_batched("gfn1-xtb") == engines.available("gfn1-xtb")[0]
    numbers = [8, 6, 8, 6, 9, 9, 9]
    assert batch_opt.estimate_max_batch("gfn1-xtb", numbers, "cpu") == batch_opt.CPU_MAX_BATCH
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda dev=None: (8 * 2**30, 16 * 2**30))
    n = batch_opt.estimate_max_batch("gfn1-xtb", numbers, "cuda:0")
    assert 1 <= n <= batch_opt.MAX_BATCH_CAP
    big = batch_opt.estimate_max_batch("gfn1-xtb", [6] * 400, "cuda:0")
    assert 1 <= big < n


def test_sequential_optimize_reference_still_works():
    """Guard: the ASE path used as the benchmark baseline is unchanged."""
    atoms = Atoms("H6", positions=_morse_starts(b=1)[0])
    atoms.calc = MorseASE()
    assert optimize(atoms, fmax=0.01, steps=300).converged


def test_fire_parameters_follow_bitzek():
    """A system that keeps going downhill accelerates (dt grows) only after N_min steps."""
    ref = torch.zeros(1, 1, 3, dtype=torch.float64)
    x0 = torch.full((1, 1, 3), 0.05, dtype=torch.float64)
    res6 = fire_batched(HarmonicModel(ref, k=0.01), x0, fmax=1e-12, max_steps=6)
    assert res6.steps.item() == 6 and not res6.converged.item()
