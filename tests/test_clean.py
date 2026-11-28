"""Clean up (BUILD-CLEAN, chem.clean and chem.edit.CleanUp): a messy 2D drawing and a distorted 3D geometry come back
clean with the connectivity and the stereo unchanged; held atoms stay put; Undo puts the old geometry back; a clean-up
that would change the stereo or the bonds is refused and changes nothing."""

import math

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from quantabench.chem import clean
from quantabench.chem.edit import EditableMolecule, EditError

PFOA = "OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"
GENX_R = "OC(=O)[C@@](F)(C(F)(F)F)OC(F)(F)C(F)(F)C(F)(F)F"         # one stereocentre
E_HFO = "F/C=C/C(F)(F)F"                                          # trans-1,3,3,3-tetrafluoropropene (HFO-1234ze(E))
Z_HFO = "F/C=C\\C(F)(F)F"
STEREO_2D = ("C/C=C/[C@H](F)Cl", GENX_R, "C[C@H]1CC[C@@H](C)CC1", "F/C(Cl)=C(\\F)C(F)(F)F", "N[C@@H](C)C(=O)O",
             "OC(=O)/C=C\\C(=O)O")


def _canon(smiles: str) -> str:
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


def _messy_block(smiles: str, noise: float = 0.35) -> str:
    """A hand drawing: the molecule laid out, then every atom nudged at random (bonds 20-40 % uneven), wedges and
    double bonds as RDKit draws them for those coordinates. The first seed whose drawing still says ``smiles``."""
    for seed in range(1, 50):
        m = Chem.MolFromSmiles(smiles)
        clean.layout_2d(m)
        conf = m.GetConformer()
        rng = np.random.default_rng(seed)
        for i in range(m.GetNumAtoms()):
            p = conf.GetAtomPosition(i)
            dx, dy = rng.normal(0.0, noise, 2)
            conf.SetAtomPosition(i, (p.x + dx, p.y + dy, 0.0))
        block = Chem.MolToV3KMolBlock(m)
        if Chem.MolToSmiles(Chem.MolFromMolBlock(block)) == _canon(smiles):
            return block
    raise AssertionError(f"no messy drawing of {smiles} keeps its stereo")


def _bond_spread(mol: Chem.Mol) -> float:
    """Relative spread (std / mean) of the drawn bond lengths."""
    pos = mol.GetConformer().GetPositions()
    lengths = [np.linalg.norm(pos[b.GetBeginAtomIdx()] - pos[b.GetEndAtomIdx()]) for b in mol.GetBonds()]
    return float(np.std(lengths) / np.mean(lengths))


# ====================================================================== 2D
@pytest.mark.parametrize("smiles", STEREO_2D)
def test_messy_drawing_comes_back_clean_with_connectivity_and_stereo_unchanged(smiles):
    block = _messy_block(smiles)
    drawing = clean.read_drawing(block)
    assert Chem.MolToSmiles(drawing) == _canon(smiles)            # the messy drawing still says the same molecule
    assert _bond_spread(drawing) > 0.15
    new_block, res = clean.clean_molblock(block)
    back = Chem.MolFromMolBlock(new_block, removeHs=False)
    assert Chem.MolToSmiles(back) == _canon(smiles)               # stereo included
    assert clean.connectivity_changes(drawing, back) == [] and clean.stereo_changes(drawing, back) == []
    assert [a.GetSymbol() for a in back.GetAtoms()] == [a.GetSymbol() for a in drawing.GetAtoms()]   # same order
    assert clean.layout_problems(back) == []
    assert _bond_spread(back) < 0.05
    assert res.layout in ("CoordGen", "zig-zag")


def test_long_chain_keeps_the_zigzag_rule():
    _block, res = clean.clean_molblock(_messy_block(PFOA, noise=0.5))
    assert res.layout == "zig-zag"
    m = res.mol
    path = clean.longest_acyclic_path(m)
    assert len(path) >= clean.ZIGZAG_MIN_ATOMS
    pos = m.GetConformer().GetPositions()[:, :2]
    turns = []
    for a, b, c in zip(path, path[1:], path[2:], strict=False):
        u, v = pos[b] - pos[a], pos[c] - pos[b]
        angle = math.degrees(math.acos(np.dot(-u, v) / np.linalg.norm(u) / np.linalg.norm(v)))
        assert abs(angle - 120.0) < 1.0
        turns.append(np.sign(u[0] * v[1] - u[1] * v[0]))
    assert all(t1 == -t2 for t1, t2 in zip(turns, turns[1:], strict=False))     # left, right, left: a zig-zag


def test_zigzag_that_squashes_bonds_falls_back_to_coordgen():
    """GenX: the depictor could not place the C=O and one C-F around the pinned chain (half-length bonds); the rule
    now takes CoordGen's layout there, in the 2D view as in Clean up."""
    m = Chem.MolFromSmiles(GENX_R)
    assert clean.layout_2d(m) == "CoordGen" and clean.layout_problems(m) == []
    pfoa = Chem.MolFromSmiles(PFOA)
    assert clean.layout_2d(pfoa) == "zig-zag" and clean.layout_problems(pfoa) == []


def _flip_first_double_bond(mol: Chem.Mol) -> str:
    """A stand-in layout that draws the first double bond the other way round (one substituent mirrored across it)."""
    clean.layout_2d(mol)
    b = next(b for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE)
    i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
    k = next(n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() != j)
    conf = mol.GetConformer()
    p, q, r = (np.array(conf.GetAtomPosition(x))[:2] for x in (i, j, k))
    axis = (q - p) / np.linalg.norm(q - p)
    d = r - p
    mirrored = p + 2 * np.dot(d, axis) * axis - d
    conf.SetAtomPosition(k, (float(mirrored[0]), float(mirrored[1]), 0.0))
    return "flipped"


def test_layout_that_would_change_a_double_bond_is_refused(monkeypatch):
    block = _messy_block(E_HFO)
    monkeypatch.setattr(clean, "LAYOUTS", (_flip_first_double_bond,))
    with pytest.raises(clean.CleanError) as err:
        clean.clean_molblock(block)
    assert "refused" in str(err.value) and "would turn from E to Z" in str(err.value)
    assert "atom 2 (C)=atom 3 (C)" in str(err.value) or "atom 3 (C)=atom 2 (C)" in str(err.value)


def test_a_bad_layout_is_passed_over_for_the_next(monkeypatch):
    monkeypatch.setattr(clean, "LAYOUTS", (_flip_first_double_bond, clean._coordgen))
    _block, res = clean.clean_molblock(_messy_block(E_HFO))
    assert res.layout == "CoordGen" and "flipped" in res.notes[0]
    assert Chem.MolToSmiles(Chem.MolFromMolBlock(_block)) == _canon(E_HFO)


def test_stereo_left_open_is_not_compared():
    """2-butene drawn as "either" (crossed) and a centre without a wedge: nothing to keep, nothing refused."""
    m = Chem.MolFromSmiles("CC=CC(F)Cl")
    m.GetBondWithIdx(1).SetStereo(Chem.BondStereo.STEREOANY)
    clean.layout_2d(m)
    _block, res = clean.clean_molblock(Chem.MolToV3KMolBlock(m))
    assert Chem.MolToSmiles(Chem.MolFromMolBlock(_block)) == "CC=CC(F)Cl"
    z = Chem.MolFromSmiles(Z_HFO)
    e = Chem.MolFromSmiles(E_HFO)
    assert clean.stereo_changes(Chem.MolFromSmiles("FC=CC(F)(F)F"), e) == []      # open in the reference
    assert clean.new_stereo(Chem.MolFromSmiles("FC=CC(F)(F)F"), e) == ["the C1=C2 double bond is now E"]
    assert clean.stereo_changes(e, z) == ["the C1=C2 double bond would turn from E to Z"]


@pytest.mark.parametrize("cx", ("a:1", "&1:1", "o1:1"))
def test_enhanced_stereo_groups_are_kept(cx):
    """The sketcher exports a centre's group (abs, and = racemic, or) as a V3000 collection: it comes back."""
    m = Chem.MolFromSmiles(f"C[C@H](N)C(=O)O |{cx}|")
    clean.layout_2d(m)
    block, _res = clean.clean_molblock(Chem.MolToV3KMolBlock(m))
    assert clean._groups(Chem.MolFromMolBlock(block)) == clean._groups(m) != []
    assert clean.stereo_changes(m, Chem.MolFromSmiles("C[C@H](N)C(=O)O |&1:1|" if cx == "a:1" else
                                                      "C[C@H](N)C(=O)O |a:1|")) == [
        "the enhanced stereo groups (abs, and, or) would change"]


def test_connectivity_changes_name_what_differs():
    a, b = Chem.MolFromSmiles("CC(=O)O"), Chem.MolFromSmiles("CC(O)O")
    assert clean.connectivity_changes(a, a) == []
    out = clean.connectivity_changes(a, b)
    assert "O2: hydrogens 0 → 1" in out and "the C1-O2 bond: double → single" in out
    assert clean.connectivity_changes(a, Chem.MolFromSmiles("CCO")) == ["4 atoms became 3"]


def test_unreadable_or_empty_drawing_is_refused():
    with pytest.raises(clean.CleanError, match="no structure"):
        clean.clean_molblock("")
    rw = Chem.RWMol()                                  # a carbon with five bonds, as a sketcher can export it
    for _ in range(6):
        rw.AddAtom(Chem.Atom(6))
    for k in range(1, 6):
        rw.AddBond(0, k, Chem.BondType.SINGLE)
    rw.UpdatePropertyCache(strict=False)
    from rdkit.Chem import rdDepictor

    rdDepictor.Compute2DCoords(rw)
    with pytest.raises(clean.CleanError, match="more than C allows"):
        clean.clean_molblock(Chem.MolToV3KMolBlock(rw))


# ====================================================================== 3D
def _distorted(smiles: str, seed: int = 3, noise: float = 0.15) -> EditableMolecule:
    """The molecule embedded, then every atom moved at random and one bond stretched by 0.5 Å (as one Move)."""
    m = EditableMolecule.from_smiles(smiles)
    pos = m.positions + np.random.default_rng(seed).normal(0.0, noise, m.positions.shape)
    pos[1] += np.array([0.5, 0.0, 0.0])
    m.set_positions(pos)
    m.history.clear()
    return m


def _bond_errors(m: EditableMolecule) -> np.ndarray:
    """|bond length - MMFF94's fully optimized bond length| for every bond (Å)."""
    ref = Chem.Mol(m.to_mol())
    AllChem.MMFFOptimizeMolecule(ref, maxIters=20000)
    p, q = m.to_mol().GetConformer().GetPositions(), ref.GetConformer().GetPositions()
    return np.array([abs(np.linalg.norm(p[b.GetBeginAtomIdx()] - p[b.GetEndAtomIdx()])
                         - np.linalg.norm(q[b.GetBeginAtomIdx()] - q[b.GetEndAtomIdx()])) for b in ref.GetBonds()])


@pytest.mark.parametrize("smiles", (GENX_R, E_HFO, "C[C@H]1CC[C@@H](C)CC1", "N[C@@H](C)C(=O)O"))
def test_distorted_3d_comes_back_clean_with_connectivity_and_stereo_unchanged(smiles):
    m = _distorted(smiles)
    graph, stereo = m._graph(), m.to_smiles()
    assert stereo == _canon(smiles)
    assert _bond_errors(m).max() > 0.3
    res = m.clean_up()
    assert res.forcefield == "MMFF94" and res.converged and res.energy < res.energy_before - 100
    assert res.defined == () and res.held == 0
    assert m._graph() == graph and m.to_smiles() == stereo
    assert _bond_errors(m).max() < 0.01
    assert clean.geometry_bond_problems(m.mol) == []


def test_held_atoms_do_not_move_and_undo_restores_the_old_geometry():
    m = _distorted(GENX_R)
    before = m.positions.copy()
    held = [0, 1, 2, 3]
    res = m.clean_up(frozen=held)
    after = m.positions.copy()
    assert res.held == 4
    assert np.array_equal(after[held], before[held])
    free = [i for i in range(m.n_atoms) if i not in held]
    assert np.abs(after[free] - before[free]).max() > 0.1
    assert m.history.undo_text == "Clean up" and len(m.history) == 1
    assert m.history.undo()
    assert np.array_equal(m.positions, before)                 # exactly the old geometry
    assert m.history.redo()
    assert np.array_equal(m.positions, after)                  # and the same clean-up again


def test_flat_centre_gets_a_configuration_and_says_so():
    m = EditableMolecule.from_smiles("N[C@@H](C)C(=O)O")
    pos = m.positions.copy()
    for k, n in enumerate(m.neighbors(1)):                     # C1's four neighbours in one plane: no configuration
        ang = 2 * math.pi * k / 4
        pos[n] = pos[1] + 1.5 * np.array([math.cos(ang), math.sin(ang), 0.0])
    m.set_positions(pos)
    assert "@" not in m.to_smiles()
    res = m.clean_up()
    assert len(res.defined) == 1 and res.defined[0].startswith("C1 is now ")
    assert "@" in m.to_smiles()


def test_uff_where_mmff_has_no_parameters():
    m = _distorted("CB(C)C")
    res = m.clean_up()
    assert res.forcefield == "UFF" and "B" in res.note and res.energy < res.energy_before


def test_clean_up_that_would_invert_a_centre_is_refused(monkeypatch):
    m = _distorted("N[C@@H](C)C(=O)O")
    before = m.positions.copy()
    minimize = EditableMolecule._op_minimize

    def mirrored(self, frozen, max_iters, movable=None):        # a minimizer that lands on the mirror image
        out = minimize(self, frozen, max_iters, movable)
        p = self.positions.copy()
        p[:, 0] *= -1.0
        self._set_positions(p)
        return out

    monkeypatch.setattr(EditableMolecule, "_op_minimize", mirrored)
    with pytest.raises(EditError, match="C1 would turn from S to R") as err:
        m.clean_up()
    assert "Nothing was changed" in str(err.value)
    assert np.array_equal(m.positions, before) and len(m.history) == 0


def test_clean_up_that_would_put_atoms_into_bonding_distance_is_refused():
    """Two hydrogens of different carbons held on top of each other: the result would read as an H-H bond."""
    m = EditableMolecule.from_smiles("CC")
    pos = m.positions.copy()
    a, b = m.hydrogens(m.neighbors(0))[0], m.hydrogens(m.neighbors(1))[0]
    pos[b] = pos[a] + np.array([0.3, 0.0, 0.0])
    m.set_positions(pos)
    m.history.clear()
    before = m.positions.copy()
    with pytest.raises(EditError, match=f"H{min(a, b)} and H{max(a, b)} would be 0.30 Å apart"):
        m.clean_up(frozen=[a, b])
    assert np.array_equal(m.positions, before) and len(m.history) == 0


def test_geometry_bond_problems_reads_broken_and_new_bonds():
    m = EditableMolecule.from_smiles("CCO").mol
    assert clean.geometry_bond_problems(m) == []
    conf = m.GetConformer()
    p = conf.GetAtomPosition(2)
    conf.SetAtomPosition(2, (p.x + 3.0, p.y, p.z))
    out = clean.geometry_bond_problems(m)
    assert any("C1-O2 bond would be" in s and "broken" in s for s in out)


def test_nothing_to_clean_up():
    m = EditableMolecule.from_atom("C sp3", fill_h=False)
    res = m.clean_up()
    assert res.forcefield == "none" and res.note
