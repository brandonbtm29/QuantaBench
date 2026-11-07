"""The model kit's edit model (chem/edit.py): free valences, attach/replace/delete, bond orders and charges,
geometry setters, minimize, undo/redo, import/export, and four PFAS built with kit operations only."""

import math

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from quantabench.chem import edit, inputs
from quantabench.chem import fragments as fr
from quantabench.chem.edit import EditableMolecule, EditError, History

PFOA = "OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"
PFOS = "OS(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"
GENX = "OC(=O)C(F)(C(F)(F)F)OC(F)(F)C(F)(F)C(F)(F)F"            # HFPO-DA
FTS62 = "OS(=O)(=O)CCC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"   # 6:2 fluorotelomer sulfonic acid


def canonical(smiles: str) -> str:
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


def hs(model: EditableMolecule, atom: int) -> list[int]:
    return [n for n in model.neighbors(atom) if model.symbol(n) == "H"]


def angle(u, v) -> float:
    return math.degrees(math.acos(np.clip(np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v), -1, 1)))


def state(model: EditableMolecule):
    return model.to_smiles(), model.positions.copy()


# ====================================================================== the four PFAS, kit operations only
def build_pfoa(model_kw=None) -> EditableMolecule:
    """Atom by atom: an sp2 carbon, =O, an sp3 O, then seven sp3 carbons with fluorines."""
    m = EditableMolecule.from_atom("C sp2", **(model_kw or {}))
    m.attach_atom(m.free_valences(0)[0], "=O")
    m.attach_atom(m.free_valences(0)[0], "O sp3")
    cur = 0
    for _ in range(7):
        nxt = m.attach_atom(hs(m, cur)[0], "C sp3")      # a new atom's first hydrogen is anti to the chain
        for h in hs(m, cur):
            m.attach_atom(h, "F")
        cur = nxt
    for h in hs(m, cur):
        m.attach_atom(h, "F")
    return m


def build_pfos() -> EditableMolecule:
    m = EditableMolecule.from_fragment("sulfonic acid")
    m.attach_fragment(m.history._undo[-1].result.entry, "perfluorooctyl")
    return m


def build_genx() -> EditableMolecule:
    m = EditableMolecule.from_fragment("carboxylic acid")
    c = m.attach_atom(m.history._undo[-1].result.entry, "C sp3")
    h = hs(m, c)
    m.attach_atom(h[0], "F")
    m.attach_fragment(h[1], "CF3")
    ether = m.attach_fragment(h[2], "ether O")
    m.attach_fragment(ether.exit, "perfluoropropyl")
    return m


def build_fts62() -> EditableMolecule:
    m = EditableMolecule.from_fragment("sulfonic acid")
    linker = m.attach_fragment(m.history._undo[-1].result.entry, "fluorotelomer linker")
    chain = m.attach_fragment(linker.exit, fr.cf2_chain(5))
    m.attach_fragment(chain.exit, "CF3")
    return m


def mmff_reference(smiles: str) -> Chem.Mol:
    ref = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(ref, randomSeed=7)
    AllChem.MMFFOptimizeMolecule(ref, mmffVariant="MMFF94", maxIters=5000)
    return ref


def max_bond_deviation(model: EditableMolecule, ref: Chem.Mol) -> float:
    mol = model.to_mol()
    match = mol.GetSubstructMatch(ref)
    assert len(match) == ref.GetNumAtoms() == mol.GetNumAtoms()
    rp, mp = ref.GetConformer().GetPositions(), mol.GetConformer().GetPositions()
    return max(abs(np.linalg.norm(rp[b.GetBeginAtomIdx()] - rp[b.GetEndAtomIdx()])
                   - np.linalg.norm(mp[match[b.GetBeginAtomIdx()]] - mp[match[b.GetEndAtomIdx()]]))
               for b in ref.GetBonds())


def min_non_bonded(model: EditableMolecule) -> float:
    mol = model.to_mol()
    pos = mol.GetConformer().GetPositions()
    d = np.linalg.norm(pos[:, None] - pos[None], axis=2)
    bonded = np.eye(len(pos), dtype=bool)
    for b in mol.GetBonds():
        bonded[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] = bonded[b.GetEndAtomIdx(), b.GetBeginAtomIdx()] = True
    return float(d[~bonded].min())


@pytest.mark.parametrize("build, smiles", [(build_pfoa, PFOA), (build_pfos, PFOS), (build_genx, GENX),
                                           (build_fts62, FTS62)], ids=["PFOA", "PFOS", "GenX", "6:2 FTS"])
def test_pfas_built_with_the_kit(build, smiles):
    model = build()
    assert model.to_smiles(isomeric=False) == canonical(smiles)
    assert model.problems() == []
    assert min_non_bonded(model) > 1.0                 # sensible before minimizing, too
    result = model.minimize()
    assert result.forcefield == "MMFF94" and result.converged
    assert model.to_smiles(isomeric=False) == canonical(smiles)
    assert max_bond_deviation(model, mmff_reference(smiles)) < 0.05
    assert min_non_bonded(model) > 1.0


def test_genx_keeps_its_stereocentre_from_3d():
    model = build_genx()
    assert "@" in model.to_smiles()
    assert "@" not in model.to_smiles(isomeric=False)


# ====================================================================== palette and free valences
def test_palette():
    assert edit.ORGANIC == ("C sp3", "C sp2", "C sp", "N sp3", "N sp2", "N sp", "P", "O sp3", "O sp2", "S", "Si",
                            "H", "F", "Cl", "Br", "I", "=O", "=S", "≡N")
    assert edit.atom_type("C") is edit.PALETTE["C sp3"]
    assert edit.atom_type("#N") is edit.atom_type("≡N") is edit.PALETTE["≡N"]
    assert edit.atom_type(" c SP2 ").key == "C sp2"
    with pytest.raises(EditError, match="Unknown atom type"):
        edit.atom_type("Xx sp9")


@pytest.mark.parametrize("kind, n, ideal", [
    ("C sp3", 4, 109.47), ("C sp2", 3, 120.0), ("C sp", 2, 180.0), ("N sp3", 3, 109.47), ("N sp2", 2, 120.0),
    ("N sp", 1, None), ("P", 3, 109.47), ("O sp3", 2, 109.47), ("O sp2", 1, None), ("S", 2, 109.47),
    ("Si", 4, 109.47), ("F", 1, None), ("Cl", 1, None), ("=O", 1, None), ("≡N", 1, None),
    ("N+ sp3", 4, 109.47), ("S(VI)", 4, 109.47), ("P(V)", 4, 109.47),
])
def test_free_valences_per_hybridization(kind, n, ideal):
    bare = EditableMolecule.from_atom(kind, fill_h=False)
    open_ = bare.free_valences(0)
    assert len(open_) == n and all(v.h is None for v in open_)
    if ideal is not None:
        assert all(abs(angle(a.vector, b.vector) - ideal) < 0.1 for k, a in enumerate(open_) for b in open_[k + 1:])
    filled = EditableMolecule.from_atom(kind)
    assert len(filled.free_valences(0)) == n and all(v.h is not None for v in filled.free_valences(0))
    assert filled.free_valences(0, include_h=False) == []


def test_lone_pairs_take_domains():
    m = EditableMolecule.from_atom("C sp3", fill_h=False)
    n = m.attach_atom(m.free_valences(0)[0], "N sp3", fill_h=False)
    o = m.attach_atom(m.free_valences(0)[0], "O sp3", fill_h=False)
    assert len(m.free_valences(n)) == 2 and len(m.free_valences(o)) == 1
    for v in m.free_valences(n):     # still tetrahedral around N
        assert abs(angle(v.vector, m.positions[0] - m.positions[n]) - 109.47) < 0.1
    assert m.atom_state(n).lone_pairs == 1 and m.atom_state(o).lone_pairs == 2
    nitrile = EditableMolecule.from_atom("C sp")
    nn = nitrile.attach_atom(nitrile.free_valences(0)[0], "≡N")
    assert nitrile.free_valences(nn) == [] and nitrile.to_smiles() == "C#N"


def test_double_free_valences_and_automatic_bond_orders():
    m = EditableMolecule.from_atom("C sp2")
    assert {v.max_order for v in m.free_valences(0)} == {2}
    assert m.problems() == ["C0 (sp2) has 1 unfilled double-bond valence"]
    c2 = m.attach_atom(m.free_valences(0)[0], "C sp2")         # two double valences join: C=C
    assert m.bond_order(0, c2) == 2 and m.to_smiles() == "C=C" and m.problems() == []
    c3 = m.attach_atom(hs(m, c2)[0], "C sp2")                   # no double valence left on c2: single
    assert m.bond_order(c2, c3) == 1
    m.attach_atom(hs(m, c3)[0], "C sp2")
    assert m.to_smiles() == "C=CC=C"
    yne = EditableMolecule.from_atom("C sp")
    yne.attach_atom(yne.free_valences(0)[0], "C sp")
    assert yne.to_smiles() == "C#C"
    with pytest.raises(EditError, match="no free double valence"):
        sp3 = EditableMolecule.from_atom("C sp3")
        sp3.attach_atom(sp3.free_valences(0)[0], "=O")


def test_attach_places_the_atom_on_the_valence():
    m = EditableMolecule.from_atom("C sp3")
    v = m.free_valences(0)[0]
    c = m.attach_atom(v, "C sp3")
    assert c == v.h                                             # the new atom takes over the hydrogen's index
    assert abs(m.distance(0, c) - edit.bond_length("C", "C")) < 1e-9
    assert angle(m.positions[c] - m.positions[0], v.vector) < 1e-6
    torsions = sorted(round(abs(m.dihedral(a, 0, c, b))) for a in hs(m, 0) for b in hs(m, c))
    assert torsions == [60] * 6 + [180] * 3                     # staggered
    assert m.atom_state(c).hyb == "sp3" and len(hs(m, c)) == 3


@pytest.mark.parametrize("kind, smiles", [
    ("C sp3", "CC"), ("N sp3", "CN"), ("O sp3", "CO"), ("S", "CS"), ("P", "CP"), ("Si", "C[SiH3]"), ("F", "CF"),
    ("Cl", "CCl"), ("Br", "CBr"), ("I", "CI"), ("H", "C"), ("N+ sp3", "C[NH3+]"), ("O-", "C[O-]"),
    ("S(VI)", None),
])
def test_attach_each_palette_type_to_methane(kind, smiles):
    m = EditableMolecule.from_atom("C sp3")
    m.attach_atom(m.free_valences(0)[0], kind)
    if smiles is None:               # sulfonyl S: four sigma bonds, two double valences still to fill
        assert len(m.problems()) == 1
    else:
        assert m.to_smiles() == smiles and m.problems() == []


def test_oxo_raises_the_valence_of_sulfur():
    m = EditableMolecule.from_atom("C sp3")
    s = m.attach_atom(hs(m, 0)[0], "S")
    m.attach_atom(hs(m, s)[0], "=O")
    assert m.to_smiles() == "C[SH]=O" and m.atom_state(s).valence == 4
    m.attach_atom(hs(m, s)[0], "=O")
    m.attach_atom(hs(m, s)[0], "O sp3")
    assert m.to_smiles() == "CS(=O)(=O)O" and m.atom_state(s).valence == 6


def test_open_valences_without_hydrogens():
    m = EditableMolecule.from_atom("C sp3", fill_h=False)
    c = m.attach_atom(m.free_valences(0)[0], "C sp3", fill_h=False)
    o = m.attach_atom(m.free_valences(c)[0], "O sp3", fill_h=False)
    assert m.n_atoms == 3 and len(m.problems()) == 3
    assert m.to_smiles() == "CCO"                               # export fills open valences
    assert m.to_mol(fill_open=False).GetAtomWithIdx(o).GetNumRadicalElectrons() == 1
    added = m.fill_hydrogens()
    assert len(added) == 6 and m.n_atoms == 9 and m.problems() == []
    with pytest.raises(EditError, match="not a hydrogen"):
        m.valence_of(0)


# ====================================================================== replace, delete, bonds, charges
def test_replace_atom_keeps_its_position_and_rebuilds_hydrogens():
    m = EditableMolecule.from_smiles("CC")
    where = m.positions[0].copy()
    si = m.replace_atom(0, "Si")
    assert np.allclose(m.positions[si], where) and m.to_smiles() == "C[SiH3]"
    assert all(abs(m.distance(si, h) - 1.48) < 1e-9 for h in hs(m, si))
    n = m.replace_atom(si, "N sp3")
    assert np.allclose(m.positions[n], where) and m.to_smiles() == "CN" and len(hs(m, n)) == 2
    h = hs(m, 1)[0]
    f = m.replace_atom(h, "F")                                  # on a hydrogen: attach there
    assert f == h and m.to_smiles() == "NCF"
    enol = EditableMolecule.from_smiles("C=CO")
    oxygen = next(i for i in range(enol.n_atoms) if enol.symbol(i) == "O")
    with pytest.raises(EditError, match="no free double valence"):
        enol.replace_atom(oxygen, "=O")                         # its carbon has no double valence left


def test_delete_atoms_leaves_hydrogens_behind():
    m = EditableMolecule.from_smiles("CCO")
    o = next(i for i in range(m.n_atoms) if m.symbol(i) == "O")
    c = m.neighbors(o)[0] if m.symbol(m.neighbors(o)[0]) == "C" else m.neighbors(o)[1]
    old_dir = m.positions[o] - m.positions[c]
    mapping = m.delete_atoms([o])
    assert m.to_smiles() == "CC" and m.n_atoms == 8 and o not in mapping
    new_h = m.n_atoms - 1
    assert m.neighbors(new_h) == [mapping[c]] and angle(m.positions[new_h] - m.positions[mapping[c]], old_dir) < 1e-6
    two = EditableMolecule.from_smiles("CC.O")
    two.delete_fragment(next(i for i in range(two.n_atoms) if two.symbol(i) == "O"))
    assert two.to_smiles() == "CC"


def test_bond_order_changes_rehybridize():
    m = EditableMolecule.from_smiles("CC")
    m.set_bond_order(0, 1, 2)
    assert m.to_smiles() == "C=C" and m.n_atoms == 6 and m.atom_state(0).hyb == "sp2"
    assert abs(m.angle(hs(m, 0)[0], 0, 1) - 120) < 1e-6
    m.set_bond_order(0, 1, 3)
    assert m.to_smiles() == "C#C" and m.atom_state(1).hyb == "sp"
    m.set_bond_order(0, 1, 1)
    assert m.to_smiles() == "CC" and m.n_atoms == 8
    m.set_bond_order(0, 1, 0)
    assert m.to_smiles() == "C.C"
    with pytest.raises(EditError, match="not bonded"):
        m.set_bond_order(0, 1, 2)
    neo = EditableMolecule.from_smiles("CC(C)(C)C")
    with pytest.raises(EditError):
        neo.set_bond_order(0, 1, 2)                             # the central carbon has four heavy neighbours


def test_formal_charges():
    m = EditableMolecule.from_smiles("CN")
    n = next(i for i in range(m.n_atoms) if m.symbol(i) == "N")
    m.set_charge(n, 1)
    assert m.to_smiles() == "C[NH3+]"
    acid = EditableMolecule.from_smiles("CC(=O)O")
    oh = next(i for i in range(acid.n_atoms) if acid.symbol(i) == "O" and hs(acid, i))
    acid.set_charge(oh, -1)
    assert acid.to_smiles() == "CC(=O)[O-]"
    carbonyl = next(i for i in range(acid.n_atoms) if acid.symbol(i) == "O" and acid.atom_state(i).hyb == "sp2")
    with pytest.raises(EditError):
        acid.set_charge(carbonyl, -1)


def test_make_and_break_bonds():
    m = EditableMolecule.from_smiles("CCCC")
    m.make_bond(0, 3)
    assert m.to_smiles() == "C1CCC1" and m.n_atoms == 12
    m.minimize()
    m.break_bond(0, 3)
    assert m.to_smiles() == "CCCC" and m.n_atoms == 14
    with pytest.raises(EditError, match="already bonded"):
        m.make_bond(0, 1)
    full = EditableMolecule.from_smiles("C(F)(F)(F)F.C")
    with pytest.raises(EditError, match="no free valence"):
        full.make_bond(0, 5)


# ====================================================================== geometry
def test_set_dihedral_moves_only_the_smaller_side():
    m = EditableMolecule.from_smiles("CCCCC")                    # pentane: C0-C1 | C2-C3-C4
    before = m.positions.copy()
    m.set_dihedral(0, 1, 2, 3, 65.0)
    assert abs(m.dihedral(0, 1, 2, 3) - 65.0) < 1e-6
    moved = np.linalg.norm(m.positions - before, axis=1) > 1e-9
    side_small = m._bfs(1, cut=(1, 2))                           # C0, C1 and their hydrogens
    assert set(np.where(moved)[0]) <= side_small and moved.any()
    assert not moved[[2, 3, 4]].any()
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in m.mol.GetBonds()]
    assert all(abs(m.distance(i, j) - np.linalg.norm(before[i] - before[j])) < 1e-9 for i, j in bonds)
    m.set_dihedral(3, 2, 1, 0, -170.0)                           # either way round
    assert abs(m.dihedral(0, 1, 2, 3) + 170.0) < 1e-6
    ring = EditableMolecule.from_smiles("C1CCCCC1")
    with pytest.raises(EditError, match="in a ring"):
        ring.set_dihedral(0, 1, 2, 3, 0.0)


def test_set_angle_and_distance():
    m = EditableMolecule.from_smiles("CCO")
    o = next(i for i in range(m.n_atoms) if m.symbol(i) == "O")
    m.set_angle(0, 1, o, 120.0)
    assert abs(m.angle(0, 1, o) - 120.0) < 1e-6
    before = m.positions.copy()
    m.set_distance(1, o, 1.60)
    assert abs(m.distance(1, o) - 1.60) < 1e-9
    moved = set(np.where(np.linalg.norm(m.positions - before, axis=1) > 1e-9)[0])
    assert moved == {o, *hs(m, o)}                              # the smaller side: O and its H
    with pytest.raises(EditError, match="not bonded"):
        m.set_distance(0, o, 3.0)
    pair = EditableMolecule.from_smiles("C.C")
    pair.set_distance(0, 1, 6.0)                                # two molecules: one moves
    assert abs(pair.distance(0, 1) - 6.0) < 1e-9
    ring = EditableMolecule.from_smiles("C1CCCCC1")
    with pytest.raises(EditError, match="in a ring"):
        ring.set_distance(0, 1, 1.7)
    with pytest.raises(EditError, match="in a ring"):
        ring.set_angle(0, 1, 2, 100.0)


def test_domain_directions():
    tet = edit.domain_directions([], 4)
    for n in range(4):
        rest = edit.domain_directions(tet[:n], 4)
        assert len(rest) == 4 - n
        allv = tet[:n] + rest
        assert all(abs(angle(a, b) - 109.47) < 0.1 for k, a in enumerate(allv) for b in allv[k + 1:])
    u, ref = np.array([0.0, 0, 1]), np.array([1.0, 0, 0])
    first = edit.domain_directions([u], 4, reference=ref)[0]
    assert first[0] > 0 and abs(first[1]) < 1e-9                # leans toward the reference
    assert len(edit.domain_directions([u], 3)) == 2 and len(edit.domain_directions([u], 2)) == 1


# ====================================================================== minimize and relax
def test_minimize_with_frozen_atoms_and_uff_fallback():
    m = EditableMolecule.from_smiles("CCCC")
    m.set_dihedral(0, 1, 2, 3, 10.0)
    before = m.positions.copy()
    result = m.minimize(frozen=[0, 1])
    assert result.forcefield == "MMFF94" and result.converged and result.moved == m.n_atoms - 2
    assert np.array_equal(m.positions[[0, 1]], before[[0, 1]])
    assert abs(m.dihedral(0, 1, 2, 3)) > 30
    assert EditableMolecule.from_smiles("CB(C)C").minimize().forcefield == "UFF"    # MMFF has no boron


def test_local_relax_moves_only_atoms_near_the_edit():
    m = EditableMolecule.from_smiles("CCCCCCCC")
    end = hs(m, 7)[0]
    f = m.attach_atom(end, "Cl")
    assert m.last_edit and f in m.last_edit
    before = m.positions.copy()
    result = m.relax(radius=1)
    assert result.moved < m.n_atoms
    moved = set(np.where(np.linalg.norm(m.positions - before, axis=1) > 1e-9)[0])
    assert moved and moved <= {f, 7, *m.neighbors(7), *(h for n in m.neighbors(7) for h in hs(m, n))}
    auto = EditableMolecule.from_smiles("CCCCCCCC", auto_relax=True)
    before = auto.positions.copy()
    auto.attach_atom(hs(auto, 7)[0], "Br")
    assert np.array_equal(auto.positions[[0, 1, 2]], before[[0, 1, 2]])


# ====================================================================== commands and history
def test_undo_redo_round_trips_restore_smiles_and_coordinates():
    m = EditableMolecule.from_fragment("carboxylic acid")
    states = [state(m)]

    def step(result):
        states.append(state(m))
        return result

    c = step(m.attach_atom(m.history._undo[-1].result.entry, "C sp3"))
    h = hs(m, c)
    step(m.attach_atom(h[0], "F"))
    step(m.attach_fragment(h[1], "CF3"))
    o = step(m.attach_fragment(h[2], "ether O")).anchor
    step(m.attach_fragment(hs(m, o)[0], "perfluoropropyl"))
    step(m.minimize())
    step(m.set_dihedral(0, c, o, next(n for n in m.neighbors(o) if n != c), 175.0))
    assert len(m.history) == len(states)
    for expected in reversed(states[:-1]):
        assert m.history.undo()
        smiles, pos = state(m)
        assert smiles == expected[0] and np.array_equal(pos, expected[1])
    assert m.history.undo() and m.n_atoms == 0 and not m.history.can_undo
    for expected in states:
        assert m.history.redo()
        smiles, pos = state(m)
        assert smiles == expected[0] and np.array_equal(pos, expected[1])
    assert not m.history.can_redo and m.to_smiles(isomeric=False) == canonical(GENX)


def test_failed_edit_leaves_the_model_and_history_unchanged():
    m = EditableMolecule.from_atom("C sp3")
    before, n = state(m), len(m.history)
    with pytest.raises(EditError):
        m.attach_atom(m.free_valences(0)[0], "=O")
    assert len(m.history) == n and state(m)[0] == before[0] and np.array_equal(state(m)[1], before[1])


def test_history_texts_listeners_and_external_stack():
    calls = []
    m = EditableMolecule.from_atom("C sp3")
    m.listeners.append(lambda model: calls.append(model.n_atoms))
    m.attach_atom(hs(m, 0)[0], "Cl")
    assert m.history.undo_text == "Attach Cl" and calls == [5]
    m.history.undo()
    assert m.history.redo_text == "Attach Cl" and calls == [5, 5]
    assert m.history.redo() and calls[-1] == 5

    class Stack:                       # what a QUndoStack adaptor provides
        def __init__(self, model):
            self.model, self.done = model, []

        def push(self, command):
            self.done.append(command)
            return command.apply(self.model)

    m2 = EditableMolecule.from_atom("C sp3")
    m2.history = Stack(m2)
    m2.attach_atom(hs(m2, 0)[0], "F")
    cmd = m2.history.done[-1]
    cmd.undo(m2)
    assert m2.to_smiles() == "C"
    cmd.apply(m2)
    assert m2.to_smiles() == "CF"
    own = History(m2, limit=2)
    for kind in ("F", "F", "F"):
        own.push(edit.AttachAtom(hs(m2, 0)[0], kind))
    assert len(own) == 2


# ====================================================================== import and export
def test_import_and_export():
    m = EditableMolecule.from_smiles("c1ccccc1O")
    assert m.n_atoms == 13 and m.to_smiles() == "Oc1ccccc1"
    assert all(m.bond_order(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) in (1, 2) for b in m.mol.GetBonds())  # kekulized
    block = m.to_molblock()
    again = EditableMolecule.from_molblock(block)
    assert again.to_smiles() == "Oc1ccccc1" and np.allclose(again.positions, m.positions, atol=1e-4)
    flat = EditableMolecule.from_molblock(Chem.MolToMolBlock(Chem.MolFromSmiles("CCO")))   # 2D: embedded
    assert flat.n_atoms == 9 and np.ptp(flat.positions[:, 2]) > 0.1
    item = m.to_input(name="phenol")
    assert isinstance(item, inputs.MoleculeInput) and item.id == "phenol" and item.source == "model kit"
    assert item.smiles == "Oc1ccccc1" and item.mol.GetNumConformers() == 1 and item.mol.GetNumAtoms() == 13
    assert EditableMolecule.from_input(item).to_smiles() == "Oc1ccccc1"
    assert EditableMolecule.from_input(inputs.from_smiles("CCO")).to_smiles() == "CCO"
    with pytest.raises(EditError):
        EditableMolecule.from_smiles("C(C")
