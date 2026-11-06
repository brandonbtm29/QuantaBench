"""Conformer comparison (chem.conformer_compare): symmetric RMSD with PFAS symmetry, mirror images and stereo,
torsions, readers (QuantaBench run folders, Spartan documents and imports), coverage, causes, the CLI, and the
GFN2-xTB common level (tblite, skipped without it). Synthetic conformer sets only; no data of Brandon's."""

from __future__ import annotations

import csv
import sys
import time

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign

from quantabench.chem import conformer_compare as cc
from quantabench.chem import io_spartan

PFBA_ANION = "[O-]C(=O)C(F)(F)C(F)(F)C(F)(F)F"
PFHXA_ANION = "[O-]C(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"
PFHXS_ANION = "[O-]S(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"
PFDS_ANION = "[O-]S(=O)(=O)" + "C(F)(F)" * 9 + "C(F)(F)F"     # C10 sulfonate: 6 x 2^9 x 6 = 18,432 automorphisms
PFHXA_ACID = "OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"
QUAT = "C[N+](C)(C)CCCNS(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"


def _embed(smiles: str, n: int = 8, seed: int = 11) -> Chem.Mol:
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMultipleConfs(m, n, randomSeed=seed)
    AllChem.MMFFOptimizeMoleculeConfs(m)
    return m


def _graph(m: Chem.Mol):
    return ([a.GetAtomicNum() for a in m.GetAtoms()], [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in m.GetBonds()])


def _model(m: Chem.Mol) -> cc.SymmetryModel:
    return cc.SymmetryModel(*_graph(m))


def _pos(m: Chem.Mol, k: int) -> np.ndarray:
    return m.GetConformer(m.GetConformers()[k].GetId()).GetPositions()


def _shuffled_copy(model: cc.SymmetryModel, pos: np.ndarray, rng, noise: float = 0.05) -> np.ndarray:
    """The same conformer, noisy, rotated, moved, with the atoms of every sibling group renumbered."""
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    out = (pos + rng.normal(0.0, noise, pos.shape)) @ q + rng.normal(size=3)
    perm = np.arange(len(pos))
    for g in model.groups:
        full = model.heavy[g]
        perm[full] = rng.permutation(full)
    return out[perm]


# ------------------------------------------------------------------------------------------ symmetry model
def test_sibling_groups_are_the_resonance_equivalent_terminal_atoms():
    m = _model(Chem.AddHs(Chem.MolFromSmiles(PFBA_ANION)))
    z = m.numbers[m.heavy]
    sizes = sorted((int(z[g[0]]), len(g)) for g in m.groups)
    assert sizes == [(8, 2), (9, 2), (9, 2), (9, 3)]            # COO-, 2 CF2, CF3
    assert len(m.skeleton) == 4 and m.n_full == 2 * 2 * 2 * 6 and len(m.reduced_auts) == 1
    acid = _model(Chem.AddHs(Chem.MolFromSmiles(PFHXA_ACID)))
    assert not any(int(acid.numbers[acid.heavy][g[0]]) == 8 for g in acid.groups)   # =O and -OH differ by their H
    assert len(acid.polar_h) == 1 and any(t[4] == "polar_h" for t in acid.torsions)
    quat = _model(Chem.AddHs(Chem.MolFromSmiles(QUAT)))
    assert any(int(quat.numbers[quat.heavy][g[0]]) == 6 and len(g) == 3 for g in quat.groups)   # N(CH3)3+
    kinds = {t[4] for t in quat.torsions}
    assert {"backbone", "rotor"} <= kinds
    # a symmetric molecule: perfluorohexane's chain reversal is a reduced automorphism
    assert len(_model(Chem.AddHs(Chem.MolFromSmiles("FC(F)(F)" + "C(F)(F)" * 4 + "C(F)(F)F"))).reduced_auts) == 2


def test_constitution_key_ignores_atom_order_bond_orders_and_charge_placement():
    m = Chem.AddHs(Chem.MolFromSmiles(PFBA_ANION))
    z, b = _graph(m)
    key = io_spartan.constitution_key(z, b, -1)
    r = Chem.RenumberAtoms(m, list(reversed(range(m.GetNumAtoms()))))
    assert io_spartan.constitution_key(*_graph(r), -1) == key
    other = Chem.AddHs(Chem.MolFromSmiles("O=C([O-])C(F)(F)C(F)(F)C(F)(F)F"))     # the charge on the other O
    assert io_spartan.constitution_key(*_graph(other), -1) == key
    assert io_spartan.constitution_key(z, b, 0) != key
    acid = Chem.AddHs(Chem.MolFromSmiles("OC(=O)C(F)(F)C(F)(F)C(F)(F)F"))
    assert io_spartan.constitution_key(*_graph(acid), -1) != key                 # another species


# ------------------------------------------------------------------------------------------ symmetric RMSD
@pytest.mark.parametrize("smiles", [PFBA_ANION, PFHXA_ANION, PFHXS_ANION])
def test_exact_path_equals_rdkit_best_rms(smiles):
    m = _embed(smiles, 6)
    model = _model(m)
    assert model.n_full <= cc.EXACT_MAPS
    heavy = Chem.RemoveHs(m)
    ids = [c.GetId() for c in m.GetConformers()]
    for i in ids[:2]:
        for j in ids[2:5]:
            al = cc.symmetric_rmsd(model, m.GetConformer(i).GetPositions(), m.GetConformer(j).GetPositions(),
                                   mirror=False)
            ref = rdMolAlign.GetBestRMS(heavy, heavy, prbId=j, refId=i, maxMatches=100000)
            assert al.method == "exact" and al.rmsd == pytest.approx(ref, abs=1e-6)


@pytest.mark.parametrize("smiles", [PFHXA_ANION, PFHXS_ANION, QUAT])
def test_reduced_path_is_exact_for_alike_conformers(smiles, monkeypatch):
    """The sibling-group factorization (align on skeleton + group centroids, assign each group, refit) finds the
    exact enumeration's minimum for noisy, rotated copies with every group's atoms renumbered."""
    rng = np.random.default_rng(5)
    m = _embed(smiles, 3)
    model = _model(m)
    perms, _ = model.full_maps()
    for k in range(3):
        a = _pos(m, k)
        b = _shuffled_copy(model, a, rng)
        exact = cc._rmsd_batch(a[model.heavy], b[model.heavy][perms]).min()
        monkeypatch.setattr(cc, "EXACT_MAPS", 1)
        al = cc.symmetric_rmsd(model, a, b, mirror=False)
        monkeypatch.undo()
        assert al.method == "reduced" and al.rmsd == pytest.approx(exact, abs=1e-8) and al.rmsd < 0.15


def test_long_sulfonate_does_not_enumerate_its_automorphisms():
    """C10 sulfonate: 18,432 heavy-atom automorphisms; the reduced graph has one, so a pair costs milliseconds."""
    rng = np.random.default_rng(2)
    m = _embed(PFDS_ANION, 2)
    model = _model(m)
    assert model.n_full == 6 * 2 ** 9 * 6 and len(model.reduced_auts) == 1
    a = _pos(m, 0)
    b = _shuffled_copy(model, a, rng, noise=0.02)
    t = time.perf_counter()
    al = cc.symmetric_rmsd(model, a, b)
    assert time.perf_counter() - t < 1.0
    assert al.method == "reduced" and al.rmsd < 0.06
    pr = cc.compare_pair(model, a, b)
    assert pr.same and pr.torsion_max < 10


def test_collinear_skeleton_takes_the_exact_path():
    m = _embed("[O-]C(=O)C(F)(F)F", 3)                          # TFA: C, C and two centroids on one line
    model = _model(m)
    assert model.degenerate(_pos(m, 0)[model.heavy])
    rng = np.random.default_rng(0)
    b = _shuffled_copy(model, _pos(m, 0), rng, noise=0.01)
    assert cc.symmetric_rmsd(model, _pos(m, 0), b).method == "exact"


# ------------------------------------------------------------------------------------------ mirror images, stereo
def _twisted(m: Chem.Mol, model: cc.SymmetryModel) -> int:
    """A conformer with a backbone torsion far from 0/180 degrees (its mirror image differs from it)."""
    for k in range(m.GetNumConformers()):
        p = _pos(m, k)
        for a, b, c, d, kind in model.torsions:
            if kind == "backbone":
                phi = abs(cc._dihedral(*(p[model.heavy[x]] for x in (a, b, c, d))))
                if 40 < phi < 140:
                    return k
    pytest.skip("no twisted conformer embedded")


def test_mirror_image_is_the_same_conformer_of_an_achiral_molecule():
    m = _embed(PFHXA_ANION, 12)
    model = _model(m)
    k = _twisted(m, model)
    a = _pos(m, k)
    mirror = a * np.array([-1.0, 1.0, 1.0])
    pr = cc.compare_pair(model, a, mirror)
    assert pr.same and pr.mirrored and pr.rmsd < 1e-6
    strict = cc.compare_pair(model, a, mirror, mirror=False)
    assert not strict.same and strict.torsion_max > 60


def test_stereocentres_keep_their_configuration():
    chiral = "[O-]C(=O)[C@H](F)C(F)(F)C(F)(F)C(F)(F)F"            # a CHF stereocentre
    m = _embed(chiral, 4)
    model = _model(m)
    assert len(model.stereocentres) == 1
    a = _pos(m, 0)
    enantiomer = a * np.array([-1.0, 1.0, 1.0])
    assert not cc.symmetric_rmsd(model, a, enantiomer, mirror=False).config_ok
    assert not cc.compare_pair(model, a, enantiomer, mirror=False).same
    pr = cc.compare_pair(model, a, enantiomer)            # opposite enantiomers drawn: accepted as mirror images
    assert pr.same and pr.mirrored
    # diastereomers never match, mirrored or not
    d1 = _embed("[O-]C(=O)[C@H](F)[C@H](F)C(F)(F)F", 6, seed=3)
    d2 = _embed("[O-]C(=O)[C@H](F)[C@@H](F)C(F)(F)F", 6, seed=3)
    m1 = _model(d1)
    iso = cc.atom_map(cc.ConformerSet("k", np.array(_graph(d1)[0]), _graph(d1)[1], -1),
                      cc.ConformerSet("k", np.array(_graph(d2)[0]), _graph(d2)[1], -1))
    assert iso is not None
    for i in range(3):
        for j in range(3):
            assert not cc.compare_pair(m1, _pos(d1, i), _pos(d2, j)[iso]).same


def test_polar_h_torsion_tells_syn_from_anti_acid():
    m = _embed(PFHXA_ACID, 1)
    model = _model(m)
    a = _pos(m, 0)
    o = next(iter(model.polar_h))
    h = model.polar_h[o]
    c = model.heavy[model.hnbrs[o][0]]
    # rotate the acid H by 180 degrees about the C-O bond
    axis = a[model.heavy[o]] - a[c]
    axis /= np.linalg.norm(axis)
    v = a[h] - a[model.heavy[o]]
    v_rot = -v + 2 * (v @ axis) * axis
    b = a.copy()
    b[h] = a[model.heavy[o]] + v_rot
    pr = cc.compare_pair(model, a, b)
    assert pr.rmsd < 1e-6 and pr.polar_h_max == pytest.approx(180.0, abs=1.0) and not pr.same


# ------------------------------------------------------------------------------------------ sets and comparison
def _distinct_conformers(smiles: str, want: int) -> tuple[Chem.Mol, list[np.ndarray]]:
    m = _embed(smiles, 40, seed=7)
    model = _model(m)
    picked: list[np.ndarray] = []
    for k in range(m.GetNumConformers()):
        p = _pos(m, k)
        if all(not cc.compare_pair(model, q, p).same for q in picked):
            picked.append(p)
        if len(picked) == want:
            return m, picked
    pytest.skip("not enough distinct conformers embedded")


def _set(m: Chem.Mol, confs: list[np.ndarray], level: str, energies: list[float], *, program: str, label: str,
         order: list[int] | None = None, charge: int = -1, dipoles: list[float] | None = None) -> cc.ConformerSet:
    mol = Chem.Mol(m) if order is None else Chem.RenumberAtoms(m, order)
    z, bonds = _graph(mol)
    idx = np.arange(m.GetNumAtoms()) if order is None else np.asarray(order)
    cs = cc.ConformerSet(io_spartan.constitution_key(z, bonds, charge), np.array(z), bonds, charge, program=program,
                         label=label, labels=[label])
    for k, (p, e) in enumerate(zip(confs, energies)):
        c = cc.Conformer(np.asarray(p)[idx], {level: e}, source=f"{program} {k}")
        if dipoles:
            c.dipoles[level] = dipoles[k]
        cs.conformers.append(c)
    return cs


def test_compare_coverage_same_best_and_causes():
    m, c = _distinct_conformers(PFHXA_ANION, 5)
    la, lb = "Spartan B3LYP/6-311+G** water", "QuantaBench tblite-gfn2 water"
    rng = np.random.default_rng(3)
    model = _model(m)
    order = list(reversed(range(m.GetNumAtoms())))
    a = _set(m, c[:4], la, [0.0, 3.0, 8.0, 20.0], program="Spartan", label="X", dipoles=[10.0, 9.0, 8.0, 7.0])
    near = [_shuffled_copy(model, c[1], rng, 0.03), _shuffled_copy(model, c[0], rng, 0.03), c[4]]
    b = _set(m, near, lb, [0.0, 2.0, 1.0], program="QuantaBench", label="X", order=order, dipoles=[4.0, 9.5, 5.0])
    comp = cc.compare([a], [b], a_name="Spartan", b_name="QuantaBench")
    r = comp.molecules[0].row
    assert (r["level_a"], r["level_b"]) == (la, lb)
    assert (r["a_best"], r["b_best"], r["same_best"], r["a_best_found"]) == (0, 0, False, True)
    assert r["a_best_match_b"] == 1 and r["a_best_match_rel_b_kj"] == pytest.approx(2.0)
    assert (r["a_n_win5"], r["a_found_win5"], r["a_n_win10"], r["a_found_win10"]) == (2, 2, 3, 2)
    assert (r["b_n_win5"], r["b_found_win5"]) == (3, 2)
    assert r["cause"] == "ranking: found, ranked higher"
    assert r["dip_a_ref"] == 10.0 and r["dip_b_best"] == 4.0 and r["dip_b_match_a_best"] == 9.5
    s = comp.summary[0]
    assert s["same_best"] == 0 and s["a_best_found"] == 1 and s["ranking"] == 1 and s["search"] == 0
    assert s["dipole_mae_b_best_vs_a_ref"] == pytest.approx(6.0)

    # B never produced A's best: the common level decides search vs ranking
    common = cc.common_level("water", relaxed=True)
    for e_a_best, cause in ((-5.0, "search: missed, lower at common level"),
                            (+5.0, "ranking: missed, higher at common level")):
        a2 = _set(m, c[:4], la, [0.0, 3.0, 8.0, 20.0], program="Spartan", label="X")
        b2 = _set(m, near[:1] + near[2:], lb, [0.0, 1.0], program="QuantaBench", label="X", order=order)
        a2.conformers[0].energies[common] = 100.0 + e_a_best
        for k, conf in enumerate(b2.conformers):
            conf.energies[common] = 100.0 + k
        a2.conformers[0].dipoles[cc.common_level("water")] = 9.8
        a2.conformers[0].dipoles[common], b2.conformers[0].dipoles[common] = 11.0, 4.5
        r2 = cc.compare([a2], [b2], phase="water").molecules[0].row
        assert r2["dip_common_union_best"] == (11.0 if e_a_best < 0 else 4.5)
        assert not r2["a_best_found"] and r2["cause"] == cause and r2["common_level"] == common
        assert r2["de_common_a_best_minus_b_best_kj"] == pytest.approx(e_a_best)
        assert r2["union_best"] == ("A only" if e_a_best < 0 else "both")
        assert r2["dip_common_a_best_sp"] == 9.8


def test_same_best_and_one_side_only():
    m, c = _distinct_conformers(PFBA_ANION, 2)
    lv = "QuantaBench tblite-gfn2 gas"
    a = _set(m, c, lv, [0.0, 1.0], program="QuantaBench", label="P")
    b = _set(m, [c[0] + 0.01], lv, [0.0], program="QuantaBench", label="P")
    other = _set(_embed("[O-]C(=O)C(F)(F)F", 1), [_pos(_embed("[O-]C(=O)C(F)(F)F", 1), 0)], lv, [0.0],
                 program="QuantaBench", label="TFA")
    comp = cc.compare([a, other], [b])
    assert len(comp.molecules) == 1 and comp.molecules[0].row["cause"] == "same best"
    assert [s.label for s in comp.only_a] == ["TFA"] and not comp.only_b


def test_merge_sets_maps_atom_order_and_merges_duplicate_geometries():
    m, c = _distinct_conformers(PFHXA_ANION, 3)
    order = [int(i) for i in np.random.default_rng(1).permutation(m.GetNumAtoms())]
    top30 = _set(m, c, "Spartan MMFF94 (search)", [0.0, 1.0, 2.0], program="Spartan", label="X")
    dft = _set(m, c[1:2], "Spartan B3LYP/6-311+G** water", [-5.0], program="Spartan", label="", order=order)
    merged = cc.merge_sets([top30, dft])
    assert len(merged) == 1 and len(merged[0].conformers) == 3
    both = merged[0].conformers[1]
    assert set(both.energies) == {"Spartan MMFF94 (search)", "Spartan B3LYP/6-311+G** water"}
    assert cc.pick_level(merged[0]) == "Spartan B3LYP/6-311+G** water"
    assert cc.pick_level(merged[0], "mmff") == "Spartan MMFF94 (search)"


def test_level_quality_order():
    levels = ["Spartan MMFF94 (search)", "Spartan PM6", "Spartan WB97X-D/6-31G(d)", "Spartan DLXD2XV",
              "QuantaBench tblite-gfn2 water", cc.common_level("water", True)]
    q = [cc.level_quality(lv) for lv in levels]
    assert q == [10, 30, 40, 50, 30, 0]


def test_spartan_values_by_level():
    b3 = {"spartan_task": "SOPT", "spartan_method": "B3LYP", "spartan_basis": "6-311+G**", "spartan_solvent": "water",
          "spartan_status": "completed", "spartan_spenergy": "-1616.5", "spartan_spenergyat": "B3LYP/6-311+G**",
          "dipole_debye": "7.7", "spartan_dc_list_energy_kcal": "113.15", "spartan_optat": "MMFF94"}
    e, d, notes = cc._spartan_values(b3)
    assert e == {"Spartan B3LYP/6-311+G** water": pytest.approx(-1616.5 * cc.KJ_PER_HARTREE),
                 "Spartan MMFF94 (search)": pytest.approx(113.15 * 4.184)}
    assert d == {"Spartan B3LYP/6-311+G** water": 7.7} and not notes
    est = {"spartan_task": "SOPT", "spartan_method": "DLXD2XV", "spartan_status": "completed",
           "spartan_spenergy": "-500.1", "spartan_spenergyat": "WB97X-D/6-31G(d)", "spartan_postenergy": "-500.6",
           "spartan_postenergyat": "DLXD2XV", "dipole_debye": "3.0", "qm_dipole_level": "WB97X-D/6-31G(d)"}
    e, d, _n = cc._spartan_values(est)
    assert set(e) == {"Spartan DLXD2XV", "Spartan WB97X-D/6-31G(d)"} and d == {"Spartan WB97X-D/6-31G(d)": 3.0}
    opt = {"spartan_task": "OPT", "spartan_method": "PM6", "spartan_status": "completed", "spartan_e_kcal": "-900"}
    e, _d, notes = cc._spartan_values(opt)
    assert not e and "optimized geometry" in notes[0]
    failed = {**b3, "spartan_status": "failed: SCF"}
    e, d, notes = cc._spartan_values(failed)
    assert set(e) == {"Spartan MMFF94 (search)"} and not d and "failed" in notes[0]


# ------------------------------------------------------------------------------------------ readers and CLI
def _commands(task: str, task_name: str, method: str, method_class: str, basis: str = "") -> str:
    return (f"# version : SPARTAN'24 1.3.0  1.3.0 Win/64b\r\n# Task    :   {task} '{task_name}'\r\n"
            f"# Method  :   {method} '{method_class}'\r\n# Basis   : {basis}\r\n")


def _calculations(keywords: str) -> bytes:
    return b"\xe0\x01\x00\xbd\x08\x00\x00\x00\xb0\x00\x00\xbd%\x00\x00\x00" + keywords.encode() + b"\x00\x00\x00\x01"


JOBLOG_OK = b"\x00\x03\x00\xbd\x93\x00\x00\x00\x00\x00\x00\xaeNormal exit.\x00No Message Returned  \x00C:/WFQM.exe\x00"
_ORDER = {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2, Chem.BondType.TRIPLE: 3}


def _input(m: Chem.Mol, pos: np.ndarray, label: str, keywords: str, charge: int) -> str:
    lines = [f" {keywords}", label, f"{charge} 1"]
    lines += [f"{a.GetAtomicNum():>2d} {x:14.9f} {y:14.9f} {z:14.9f}" for a, (x, y, z) in zip(m.GetAtoms(), pos)]
    lines += ["ENDCART", "HESSIAN", " ".join("1" for _ in m.GetAtoms())]
    lines += [f"{b.GetBeginAtomIdx() + 1:5d}{b.GetEndAtomIdx() + 1:5d}{_ORDER[b.GetBondType()]:5d}" for b in m.GetBonds()]
    lines += ["ENDHESS", ""]
    return "\r\n".join(lines)


def _doc(molecules: list[dict]) -> io_spartan.DictContainer:
    streams = {"Document": b"\x00"}
    for k, parts in enumerate(molecules, 1):
        for name, data in parts.items():
            streams[f"Molecules/M{k:04d}/{name}"] = data.encode("latin-1") if isinstance(data, str) else data
    return io_spartan.DictContainer(streams)


def _spartan_docs(m: Chem.Mol, confs: list[np.ndarray]) -> dict[str, io_spartan.DictContainer]:
    """A base document (the CAS label), an All_Conformers document (MMFF search energies, default labels) and a
    DFT_Top5 document (B3LYP C-PCM single points of two of them, with dipoles)."""
    kekule = Chem.Mol(m)
    Chem.Kekulize(kekule, clearAromaticFlags=True)
    base = [{"Input": _input(kekule, confs[0], "999-99-9", "OPT MMFF94", -1)}]
    allc = [{"Input": _input(kekule, p, f"M{k + 1:04d}", "OPT MMFF94", -1),
             "PropertyArchive": f"PROP VALUE DC_LIST_ENERGY_KCAL = {100 + 2 * k:.4f}\r\nPROP STRING OPTAT = \"MMFF94\"\r\n"}
            for k, p in enumerate(confs)]
    kw = "SOPT B3LYP 6-311+G** SOLVENT=WASSER"
    dft = [{"Input": _input(kekule, confs[k], f"M{n + 1:04d}", kw, -1),
            "PropertyArchive": (f"PROP VALUE SPENERGY = {-1000 - 0.004 * (1 - n):.6f}\r\nPROP STRING SPENERGYAT = "
                                f"\"B3LYP/6-311+G**\"\r\nPROP VALUE DIPOLE_MAG = 5.0\r\nPROP VALUE DIPOLE_VEC 3 BEGIN\r\n"
                                f" 0.0 0.0 {5.0 + n:.3f}\r\nEND\r\nPROP VALUE DC_LIST_ENERGY_KCAL = {100 + 2 * k:.4f}\r\n"
                                "PROP STRING OPTAT = \"MMFF94\"\r\n"),
            "Commands": _commands("SOPT", "Single Point Energy", "B3LYP", "Density-Functional", "6-311+G**"),
            "Calculations": _calculations(kw), "JobLog": JOBLOG_OK}
           for n, k in enumerate((2, 0))]
    return {"X_P1.spartan": _doc(base), "X_P1_All_Conformers_Top30.spartan": _doc(allc),
            "X_P1_DFT_Top5.spartan": _doc(dft)}


def _write_run(tmp_path, m: Chem.Mol, confs: list[np.ndarray], energies: list[float], dipoles: list[float]):
    from quantabench.chem import export
    from quantabench.chem.inputs import MoleculeInput
    from quantabench.chem.pipeline import ConformerResult, MoleculeResult

    mol = Chem.Mol(m)
    mol.RemoveAllConformers()
    rows = []
    for k, (p, e, d) in enumerate(zip(confs, energies, dipoles)):
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i, xyz in enumerate(p):
            conf.SetAtomPosition(i, xyz.tolist())
        cid = mol.AddConformer(conf, assignId=True)
        rows.append(ConformerResult(rank=k, conf_id=cid, energy_ev=e, rel_kcal=(e - energies[0]) * 23.06,
                                    dipole_debye=d))
    res = MoleculeResult(input=MoleculeInput(id="999-99-9", smiles=Chem.MolToSmiles(Chem.RemoveHs(m)), name="X acid"),
                         status="ok", smiles=Chem.MolToSmiles(Chem.RemoveHs(m)), charge=-1, method="tblite-gfn2",
                         conformers=rows, mol=mol, solvent="water", phase="water", conformer_search="rdkit")
    run = tmp_path / "run"
    export.write_molecule(res, run / "999-99-9")
    return run


def test_readers_and_cli_end_to_end(tmp_path, monkeypatch):
    m, c = _distinct_conformers(PFHXA_ANION, 3)
    docs = _spartan_docs(m, c)
    monkeypatch.setattr(io_spartan, "open_container", lambda path: docs[str(path).replace("\\", "/").split("/")[-1]])
    sets = cc.read_spartan_documents(list(docs), include=["*_DFT_Top5.spartan", "*_All_Conformers_Top30.spartan"])
    merged = cc.merge_sets(sets)
    assert len(merged) == 1
    sp = merged[0]
    assert sp.label == "999-99-9" and sp.group == "X" and len(sp.conformers) == 3       # DFT copies merged
    assert cc.pick_level(sp) == "Spartan B3LYP/6-311+G** water"
    best = sp.best("Spartan B3LYP/6-311+G** water")
    assert sp.conformers[best].source.startswith("X_P1_All_Conformers_Top30.spartan M0003")   # conformer 3 is lower
    assert len([x for x in sp.conformers if "Spartan B3LYP/6-311+G** water" in x.dipoles]) == 2

    # QuantaBench found conformers 1 and 2 of Spartan's (another atom order is fine), its best is Spartan's 2nd
    order = list(reversed(range(m.GetNumAtoms())))
    rm = Chem.RenumberAtoms(m, order)
    run = _write_run(tmp_path, rm, [c[1][order], c[0][order]], [-100.0, -99.99], [6.0, 5.5])
    qb = cc.read_sources([run])
    assert len(qb) == 1 and qb[0].label == "999-99-9" and qb[0].phase == "water"
    assert set(qb[0].levels()) == {"QuantaBench tblite-gfn2 water"} and qb[0].key == sp.key

    from quantabench.cli import main

    out = tmp_path / "cmp"
    rc = main(["conformers", "compare", "--a", *docs, "--include", "*_DFT_Top5.spartan",
               "*_All_Conformers_Top30.spartan", "--b", str(run), "--out", str(out), "--a-name", "Spartan"])
    assert rc == 0
    rows = list(csv.DictReader((out / "molecules.csv").open(encoding="utf-8")))
    assert len(rows) == 1
    r = rows[0]
    assert r["label"] == "999-99-9" and r["a_best_found"] == "False" and r["b_best_found"] == "True"
    assert r["cause"] == "missed (no common energies)" and r["n_a"] == "3" and r["n_b"] == "2"
    assert (out / "report.md").read_text(encoding="utf-8").startswith("# Conformers: Spartan vs QuantaBench")
    levels = list(csv.DictReader((out / "levels.csv").open(encoding="utf-8")))
    assert {x["level"] for x in levels} == {"Spartan B3LYP/6-311+G** water", "Spartan MMFF94 (search)",
                                            "QuantaBench tblite-gfn2 water"}
    confs = list(csv.DictReader((out / "conformers.csv").open(encoding="utf-8")))
    assert len(confs) == 5 and sum(x["found"] == "True" for x in confs) == 4

    # the same through a Spartan import (long table + SDF)
    table, sdf = tmp_path / "long.csv", tmp_path / "conf.sdf"
    io_spartan.write_table(list(docs), table, sdf=sdf, conformers="all")
    imp = cc.read_sources([table, sdf], include=["*_DFT_Top5.spartan", "*_All_Conformers_Top30.spartan"])
    assert len(imp) == 1 and len(imp[0].conformers) == 3 and imp[0].key == sp.key
    assert imp[0].label == "999-99-9"


def test_read_sources_rejects_unknown_inputs(tmp_path):
    (tmp_path / "x.txt").write_text("x")
    with pytest.raises(ValueError):
        cc.read_sources([tmp_path / "x.txt"])
    with pytest.raises(ValueError):
        cc.read_sources([tmp_path / "a.csv"])          # a table without its SDF
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError):
        cc.read_sources([empty])


def test_an_optimization_that_moves_a_proton_is_set_aside(tmp_path):
    """Cached common-level results: a relaxed geometry whose bonds differ from the start keeps its single points but
    gives no optimized energy, and is flagged."""
    import json

    m = _embed(PFHXA_ACID, 1)
    z, bonds = _graph(m)
    cs = cc.ConformerSet(io_spartan.constitution_key(z, bonds, 0), np.array(z), bonds, 0)
    pos = _pos(m, 0)
    cs.conformers = [cc.Conformer(pos.copy()), cc.Conformer(pos + 0.001)]
    model = _model(m)
    o = next(iter(model.polar_h))
    h = model.polar_h[o]
    carbonyl = next(model.heavy[k] for k in model.hnbrs[model.hnbrs[o][0]] if k != o and model.numbers[model.heavy[k]] == 8)
    moved = pos.copy()
    moved[h] = pos[carbonyl] + (pos[h] - pos[model.heavy[o]])      # the acid H now sits on the carbonyl O
    phases, relax, fmax, steps = ("gas", "water"), "water", 0.03, 300
    opt, sp = cc.common_level("water", True), cc.common_level("water")
    lines = []
    for c, rel in zip(cs.conformers, (moved, pos + 0.002)):
        key = cc._rescore_key(cs.numbers, c.positions, 0, 1, phases, relax, fmax, steps)
        lines.append(json.dumps({"key": key, f"E|{sp}": -1.0, f"E|{opt}": -2.0, f"D|{opt}": 1.0,
                                 "relaxed": rel.tolist(), "relax_converged": True}))
    cache = tmp_path / "c.jsonl"
    cache.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert cc.rescore([cs], phases=phases, relax=relax, cache=cache) == 0
    broken, fine = cs.conformers
    assert sp in broken.energies and opt not in broken.energies and opt not in broken.dipoles
    assert broken.extra["relax_topology_changed"] and "relaxed" not in broken.extra
    assert fine.energies[opt] == -2.0 and "relaxed" in fine.extra


# ------------------------------------------------------------------------------------------ common level (tblite)
def _tblite():
    if "tblite" not in sys.modules:
        import os

        os.environ.setdefault("OMP_NUM_THREADS", "1")
    return pytest.importorskip("tblite")


def test_rescore_gfn2_single_points_optimization_and_cache(tmp_path):
    _tblite()
    from quantabench.chem import eng_tblite

    m = _embed("[O-]C(=O)C(F)(F)F", 2)
    z, bonds = _graph(m)
    cs = cc.ConformerSet(io_spartan.constitution_key(z, bonds, -1), np.array(z), bonds, -1, program="Spartan")
    for k in range(2):
        cs.conformers.append(cc.Conformer(_pos(m, k)))
    cache = tmp_path / "cache.jsonl"
    n = cc.rescore([cs], phases=("gas", "water"), relax="water", steps=60, cache=cache)
    assert n == 2
    c0 = cs.conformers[0]
    for lv in (cc.common_level("gas"), cc.common_level("water"), cc.common_level("water", True)):
        assert lv in c0.energies and lv in c0.dipoles
    assert c0.energies[cc.common_level("water", True)] < c0.energies[cc.common_level("water")]   # relaxed lower
    assert c0.extra["relaxed"].shape == c0.positions.shape
    # the dipole is the engine's own (about the centre of mass), as QuantaBench's refinement reports it
    props = eng_tblite._properties("GFN2-xTB", "tblite_gfn2")(z, c0.positions, -1, solvent="water", polarizability=False)
    assert c0.dipoles[cc.common_level("water")] == pytest.approx(props["tblite_gfn2_dipole_debye"], abs=1e-4)
    # a rerun reads the cache
    fresh = cc.ConformerSet(cs.key, cs.numbers, cs.bonds, -1, conformers=[cc.Conformer(_pos(m, k)) for k in range(2)])
    assert cc.rescore([fresh], phases=("gas", "water"), relax="water", steps=60, cache=cache) == 0
    assert fresh.conformers[1].energies == pytest.approx(cs.conformers[1].energies)
