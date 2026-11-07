"""The model kit's fragment library (chem/fragments.py) and how each fragment attaches (chem/edit.py)."""

import numpy as np
import pytest
from rdkit import Chem

from quantabench.chem import fragments as fr
from quantabench.chem.edit import EditableMolecule, EditError

REQUIRED = [
    "methyl", "ethyl", "hydroxyl", "amine", "carbonyl", "carboxylic acid", "carboxylate", "ester", "amide",
    "nitro", "nitrile", "sulfonic acid", "sulfonate", "sulfonamide", "phosphate", "phenyl", "cyclohexyl",
    "2-pyridyl", "3-pyridyl", "4-pyridyl", "cyclopentyl", "2-furanyl", "2-thiophenyl", "trifluoromethyl",
    "ether o", "fluorotelomer linker", "pfca head", "pfsa head", "pfca head anion", "pfsa head anion",
]


def canonical(smiles: str) -> str:
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


def filled(fragment: fr.Fragment, attach: str) -> str:
    """The fragment's SMILES with [*:1] replaced by ``attach`` and the chain end [*:2] by a hydrogen."""
    return canonical(fragment.smiles.replace("[*:1]", attach).replace("[*:2]", "[H]"))


def test_library_is_complete_and_valid():
    names = [f.name for f in fr.FRAGMENTS.values()]
    assert len(names) == len(set(names))
    for name in REQUIRED:
        assert name in fr.FRAGMENTS, name
    for f in fr.FRAGMENTS.values():
        assert f.category in fr.CATEGORIES
        mol, d1, d2 = fr.attachment_points(f)
        assert mol.GetAtomWithIdx(d1).GetAtomMapNum() == 1
        assert d2 is None or mol.GetAtomWithIdx(d2).GetAtomMapNum() == 2
    assert {c for c in fr.CATEGORIES} == {f.category for f in fr.FRAGMENTS.values()}
    assert all(f.category == "ring" for f in fr.by_category("ring"))
    with pytest.raises(ValueError):
        fr.by_category("protein")


def test_lookup_by_name_alias_label_and_builder_text():
    assert fr.get("Carboxylic Acid").name == "carboxylic acid"
    assert fr.get("COOH") is fr.get("carboxylic acid")
    assert fr.get("cf3").smiles == "[*:1]C(F)(F)F"
    assert fr.get("SO3-").name == "sulfonate"
    assert fr.get("(CF2)6") == fr.cf2_chain(6)
    assert fr.get("C6F13").smiles == fr.perfluoroalkyl(6).smiles      # the library's perfluorohexyl
    assert fr.get("C7F15") == fr.perfluoroalkyl(7)                     # built on the fly
    assert fr.get(fr.FRAGMENTS["phenyl"]) is fr.FRAGMENTS["phenyl"]
    with pytest.raises(KeyError):
        fr.get("unobtainium")
    with pytest.raises(KeyError):
        fr.get("C6F12")              # not CnF(2n+1)


@pytest.mark.parametrize("n", [1, 2, 4, 8])
def test_chain_builders(n):
    chain = Chem.MolFromSmiles(fr.cf2_chain(n).smiles)
    assert sum(a.GetSymbol() == "C" for a in chain.GetAtoms()) == n
    assert sum(a.GetSymbol() == "F" for a in chain.GetAtoms()) == 2 * n
    assert sorted(a.GetAtomMapNum() for a in chain.GetAtoms() if a.GetAtomicNum() == 0) == [1, 2]
    rf = Chem.MolFromSmiles(fr.perfluoroalkyl(n).smiles)
    assert sum(a.GetSymbol() == "C" for a in rf.GetAtoms()) == n
    assert sum(a.GetSymbol() == "F" for a in rf.GetAtoms()) == 2 * n + 1
    assert fr.perfluoroalkyl(n).label == f"C{n}F{2 * n + 1}"
    with pytest.raises(ValueError):
        fr.cf2_chain(0)
    with pytest.raises(ValueError):
        fr.perfluoroalkyl(0)


def test_fragment_from_user_smiles():
    marked = fr.from_smiles("[*:1]C(=O)OCC", name="ethyl ester")
    assert marked.name == "ethyl ester" and marked.caption == "ethyl ester"
    first = fr.from_smiles("OCC")            # no *: the first atom is the attachment atom
    mol, d1, _ = fr.attachment_points(first)
    assert mol.GetAtomWithIdx(d1).GetNeighbors()[0].GetSymbol() == "O"
    assert filled(first, "C") == canonical("COCC")
    assert filled(fr.from_smiles("c1ccncc1"), "C") == canonical("Cc1ccncc1")
    with pytest.raises(ValueError, match="no hydrogen"):
        fr.from_smiles("FC(F)(F)F")
    with pytest.raises(ValueError, match="valid SMILES"):
        fr.from_smiles("C(C")


@pytest.mark.parametrize("smiles, why", [
    ("CC(=O)O", "no attachment point"),
    ("[*:1]CC[*:1]", "two dummy atoms"),
    ("[*:1]CC[*:3]", "only"),
    ("C([*:1])([*:1])", "two dummy atoms"),
    ("[*:1]C=[*:2]", "single bond"),
])
def test_invalid_fragments_are_refused(smiles, why):
    with pytest.raises(ValueError, match=why):
        fr.attachment_points(fr.Fragment("bad", smiles, "group"))


def _non_bonded_min(model: EditableMolecule, min_separation: int = 3) -> float:
    """Smallest distance between atoms at least ``min_separation`` bonds apart."""
    mol = model.to_mol()
    topo = Chem.GetDistanceMatrix(mol)
    pos = mol.GetConformer().GetPositions()
    d = np.linalg.norm(pos[:, None] - pos[None], axis=2)
    mask = topo >= min_separation
    return float(d[mask].min()) if mask.any() else np.inf


@pytest.mark.parametrize("name", sorted(fr.FRAGMENTS))
def test_every_fragment_attaches_and_stands_alone(name):
    frag = fr.FRAGMENTS[name]
    model = EditableMolecule.from_atom("C sp3")
    placed = model.attach_fragment(model.free_valences(0)[0], frag)
    assert model.to_smiles(isomeric=False) == filled(frag, "C"), name
    assert model.problems() == []
    assert model.bond_order(0, placed.anchor) >= 1
    assert _non_bonded_min(model) > 1.3, name             # the torsion scan avoids clashes
    if placed.exit is not None:
        assert model.symbol(placed.exit) == "H"
    alone = EditableMolecule.from_fragment(frag)
    assert alone.to_smiles(isomeric=False) == filled(frag, "[H]"), name
    entry = alone.history._undo[-1].result.entry
    assert entry is not None and alone.symbol(entry) == "H"


def test_cyclohexyl_is_a_chair_and_chains_start_extended():
    model = EditableMolecule.from_fragment("cyclohexyl")
    mol = model.to_mol()
    ring = mol.GetRingInfo().AtomRings()[0]
    torsions = [model.dihedral(*(ring[(k + s) % 6] for s in range(4))) for k in range(6)]
    assert all(45 < abs(t) < 65 for t in torsions)
    assert all(a * b < 0 for a, b in zip(torsions, torsions[1:]))      # alternating: chair, not boat
    chain = EditableMolecule.from_fragment(fr.perfluoroalkyl(8))
    carbons = [i for i in range(chain.n_atoms) if chain.symbol(i) == "C"]
    backbone = [carbons[k:k + 4] for k in range(len(carbons) - 3)]
    assert all(abs(chain.dihedral(*q)) > 150 for q in backbone)        # anti, with the usual helical twist


def test_unknown_fragment_text_is_an_edit_error():
    model = EditableMolecule.from_atom("C sp3")
    with pytest.raises(EditError, match="No fragment named"):
        model.attach_fragment(model.free_valences(0)[0], "unobtainium(")
    placed = model.attach_fragment(model.free_valences(0)[0], "[*:1]C#N")   # SMILES text works
    assert model.to_smiles() == "CC#N" and model.symbol(placed.anchor) == "C"
