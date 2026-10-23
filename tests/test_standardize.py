import pytest
from conftest import FBSA, K_PFOS, PFOA
from rdkit import Chem

from quantabench.chem.standardize import formal_charge, prepare


def charge_of(smiles, ph=7.0):
    mol, notes = prepare(Chem.MolFromSmiles(smiles), ph)
    return formal_charge(mol), Chem.MolToSmiles(Chem.RemoveHs(mol)), notes


def test_pfoa_is_anion_at_ph7():
    q, smi, notes = charge_of(PFOA)
    assert q == -1 and "[O-]" in smi
    assert any("carboxylic" in n for n in notes)


def test_pfoa_neutral_at_low_ph():
    q, _, _ = charge_of(PFOA, ph=2.0)
    assert q == 0


def test_salt_is_stripped():
    q, smi, notes = charge_of(K_PFOS)
    assert q == -1 and "K" not in smi
    assert any("standardized" in n for n in notes)


def test_perfluoroalkyl_sulfonamide_is_acidic():
    q, smi, _ = charge_of(FBSA)
    assert q == -1 and "[NH-]" in smi


def test_amine_protonated_and_ftoh_neutral():
    assert charge_of("CCCN")[0] == 1
    assert charge_of("OCCC(F)(F)C(F)(F)C(F)(F)C(F)(F)F")[0] == 0  # 4:2 FTOH


def test_no_ph_keeps_input_state():
    mol, _ = prepare(Chem.MolFromSmiles(PFOA), None)
    assert formal_charge(mol) == 0


@pytest.mark.parametrize("smiles, out, charge", [("[Cl-]", "[Cl-]", -1), ("[Br-]", "[Br-]", -1), ("[I-]", "[I-]", -1),
                                                 ("[F-]", "[F-]", -1), ("Cl", "[Cl-]", -1), ("[NH4+]", "[NH4+]", 1),
                                                 ("N", "[NH4+]", 1)])
def test_halides_and_ammonium_at_ph7(smiles, out, charge):
    """The Uncharger neutralizes Cl- to HCl and NH4+ to NH3; the pH rules must restore the ions."""
    mol, _ = prepare(Chem.MolFromSmiles(smiles), 7.0)
    assert Chem.MolToSmiles(Chem.RemoveHs(mol)) == out and formal_charge(mol) == charge


def test_hf_is_neutral_below_its_pka():
    mol, _ = prepare(Chem.MolFromSmiles("F"), 2.0)
    assert formal_charge(mol) == 0
