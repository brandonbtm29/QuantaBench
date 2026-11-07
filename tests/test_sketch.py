"""The 2D sketcher's hand-off (quantabench.chem.sketch): MOL blocks to MoleculeInput, checks, Edit selected."""

import pytest
from conftest import PFOA
from rdkit import Chem
from rdkit.Chem import AllChem

from quantabench.chem import sketch
from quantabench.chem.inputs import MoleculeInput, from_smiles

# what Schrödinger's Sketcher exports (MDL_MOLV3000 ends with an SD record separator)
SKETCHER_ETHANOL = """
     RDKit          2D

  0  0  0  0  0  0  0  0  0  0999 V3000
M  V30 BEGIN CTAB
M  V30 COUNTS 3 2 0 0 0
M  V30 BEGIN ATOM
M  V30 1 C -1.299038 -0.750000 0.000000 0
M  V30 2 C 0.000000 0.000000 0.000000 0
M  V30 3 O 1.299038 -0.750000 0.000000 0
M  V30 END ATOM
M  V30 BEGIN BOND
M  V30 1 1 1 2
M  V30 2 1 2 3
M  V30 END BOND
M  V30 END CTAB
M  END
$$$$
"""


def v3000(atoms, bonds):
    """A MOL V3000 block from (symbol, extra) atoms and (order, a, b) bonds (1-based), 2D coordinates."""
    lines = ["", "     test          2D", "", "  0  0  0  0  0  0  0  0  0  0999 V3000", "M  V30 BEGIN CTAB",
             f"M  V30 COUNTS {len(atoms)} {len(bonds)} 0 0 0", "M  V30 BEGIN ATOM"]
    for i, (sym, extra) in enumerate(atoms, 1):
        lines.append(f"M  V30 {i} {sym} {i * 1.3:.4f} {(i % 2) * 0.75:.4f} 0.0000 0{(' ' + extra) if extra else ''}")
    lines += ["M  V30 END ATOM", "M  V30 BEGIN BOND"]
    lines += [f"M  V30 {i} {order} {a} {b}" for i, (order, a, b) in enumerate(bonds, 1)]
    lines += ["M  V30 END BOND", "M  V30 END CTAB", "M  END"]
    return "\n".join(lines) + "\n"


def canon(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


def test_sketcher_block_becomes_an_input():
    res = sketch.input_from_molblock(SKETCHER_ETHANOL)
    assert res.input.smiles == "CCO" and res.input.source == "sketch"
    assert res.input.id == "sketch_C2H6O" and res.formula == "C2H6O" and not res.notes
    assert res.input.mol is None                        # like a typed SMILES: the pipeline embeds it in 3D
    assert res.input.to_mol().GetNumAtoms() == 3


def test_round_trip_keeps_charges_isotopes_and_stereo():
    for smiles in (PFOA, "[13CH3][C@H](N)C(=O)[O-].[Na+]", "F/C=C/Cl", "C[N+](C)(C)C.FC(F)(F)S(=O)(=O)[O-]",
                   "OC(=O)C(F)(C(F)(F)F)OC(F)(F)C(F)(F)C(F)(F)F"):
        block = sketch.molblock_for_mol(Chem.MolFromSmiles(smiles))
        assert "V3000" in block
        assert sketch.input_from_molblock(block).input.smiles == canon(smiles), smiles


def test_id_and_name_for_replace():
    res = sketch.input_from_molblock(SKETCHER_ETHANOL, id="PFOA", name="perfluorooctanoic acid")
    assert (res.input.id, res.input.name) == ("PFOA", "perfluorooctanoic acid")


def test_valence_problem_is_reported_with_the_atom():
    block = v3000([("C", "")] * 6, [(1, 1, 2), (1, 2, 3), (1, 2, 4), (1, 2, 5), (1, 2, 6)])
    with pytest.raises(sketch.SketchError, match=r"atom 2 \(C\) has 5 bonds, more than C allows"):
        sketch.input_from_molblock(block)


def test_charged_atom_label():
    block = v3000([("N", "CHG=1")] + [("C", "")] * 5, [(1, 1, k) for k in range(2, 7)])
    with pytest.raises(sketch.SketchError, match=r"atom 1 \(N\+\) has 5 bonds"):
        sketch.input_from_molblock(block)


def test_aromatic_ring_that_cannot_be_kekulized():
    block = v3000([("C", "")] * 5, [(4, 1, 2), (4, 2, 3), (4, 3, 4), (4, 4, 5), (4, 5, 1)])
    with pytest.raises(sketch.SketchError, match="aromatic ring cannot be written"):
        sketch.input_from_molblock(block)


def test_query_atoms_and_bonds_are_refused():
    with pytest.raises(sketch.SketchError, match="query atoms, R-groups or attachment points"):
        sketch.input_from_molblock(v3000([("C", ""), ("A", "")], [(1, 1, 2)]))
    with pytest.raises(sketch.SketchError, match="query atoms"):
        sketch.input_from_molblock(v3000([("C", ""), ("R#", "RGROUPS=(1 1)")], [(1, 1, 2)]))
    with pytest.raises(sketch.SketchError, match="query bonds"):
        sketch.input_from_molblock(v3000([("C", ""), ("C", "")], [(8, 1, 2)]))


@pytest.mark.parametrize("block, message", [("", "no structure"), ("   \n", "no structure"),
                                            ("not a molecule", "could not read"),
                                            (v3000([], []), "empty")])
def test_empty_or_unreadable(block, message):
    with pytest.raises(sketch.SketchError, match=message):
        sketch.input_from_molblock(block)


def test_notes_for_pieces_and_radicals():
    res = sketch.input_from_molblock(sketch.molblock_for_mol(Chem.MolFromSmiles("CCO.O")))
    assert any("2 separate pieces" in n for n in res.notes)
    res = sketch.input_from_molblock(v3000([("C", ""), ("C", "RAD=2"), ("C", "")], [(1, 1, 2), (1, 2, 3)]))
    assert res.input.smiles == "C[CH]C" and any("unpaired" in n for n in res.notes)


def test_molblock_for_typed_input_is_2d_without_hydrogens():
    block = sketch.molblock_for_input(from_smiles("C[C@H](N)C(=O)O", id="L-Ala"))
    mol = Chem.MolFromMolBlock(block)
    assert not mol.GetConformer().Is3D() and all(a.GetAtomicNum() != 1 for a in mol.GetAtoms())
    assert Chem.MolToSmiles(mol) == canon("C[C@H](N)C(=O)O")


def test_molblock_for_3d_input_keeps_its_stereo():
    mol = Chem.AddHs(Chem.MolFromSmiles("C[C@@H](O)C(F)(F)F"))
    AllChem.EmbedMolecule(mol, randomSeed=7)
    mol = Chem.MolFromMolBlock(Chem.MolToMolBlock(mol), removeHs=False)   # as read from an SD file
    inp = MoleculeInput(id="x", smiles="", mol=mol, source="x.sdf#1")
    out = sketch.input_from_molblock(sketch.molblock_for_input(inp)).input
    assert out.smiles == canon("C[C@@H](O)C(F)(F)F")


def test_molblock_for_bad_input():
    with pytest.raises(sketch.SketchError):
        sketch.molblock_for_input(MoleculeInput(id="bad", smiles="C1CC"))


def test_fragments_parse_and_can_be_bonded():
    labels = [label for label, _s in sketch.FRAGMENTS]
    assert len(labels) == len(set(labels)) and any("(CF2)n" in label for label in labels)
    for label, smiles in sketch.FRAGMENTS:
        if smiles is None:
            continue
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, label
        assert any(a.GetTotalNumHs() for a in mol.GetAtoms()), label    # an H to give up for the new bond


def test_cf2_chain():
    mol = Chem.MolFromSmiles(sketch.cf2_chain(3))
    from rdkit.Chem import rdMolDescriptors

    assert rdMolDescriptors.CalcMolFormula(mol) == "C3H2F6"
    for n in (0, sketch.MAX_CHAIN + 1):
        with pytest.raises(ValueError):
            sketch.cf2_chain(n)
