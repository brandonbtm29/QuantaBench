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


# ---------------------------------------------------------------- salts, metals, charges, radicals
def test_counter_ion_removal_is_named_and_the_charge_change_explained():
    """Sodium trifluoroacetate: Na+ is removed by name, and the notes say the charge went 0 -> -1."""
    info = {}
    mol, notes = prepare(Chem.MolFromSmiles("[Na+].[O-]C(=O)C(F)(F)F"), 7.0, info=info)
    assert formal_charge(mol) == -1 and "Na" not in Chem.MolToSmiles(mol)
    assert info["removed_fragments"] == ["[Na+]"] and info["input_charge"] == 0 and info["n_fragments"] == 2
    assert any("removed [Na+]" in n for n in notes)
    assert any(n.startswith("charge +0 as drawn, -1 computed") for n in notes)


def test_covalently_drawn_sodium_is_disconnected_like_an_ion():
    mol, notes = prepare(Chem.MolFromSmiles("CC(=O)O[Na]"), 7.0)
    assert Chem.MolToSmiles(Chem.RemoveHs(mol)) == "CC(=O)[O-]" and any("removed [Na+]" in n for n in notes)


def test_repeated_counter_ions_are_counted():
    _, notes = prepare(Chem.MolFromSmiles("C[N+](C)(C)CCCCCC[N+](C)(C)C.[Br-].[Br-]"), 7.0)
    assert any("removed 2 x [Br-]" in n for n in notes)


@pytest.mark.parametrize("smiles", ["CCCC[Sn](Cl)(CCCC)CCCC",          # tributyltin chloride
                                    "COCC[Hg]OC(C)=O",                  # methoxyethylmercury acetate
                                    "CCCC[Sn](CCCC)(CCCC)O[Sn](CCCC)(CCCC)CCCC"])   # TBTO
def test_bonds_to_tin_and_mercury_are_kept(smiles):
    """RDKit's default MetalDisconnector splits Hg-O and Sn-O, which turned R-Hg-OAc into R-Hg+ and TBTO into
    half a molecule (Bu3Sn+). The drawn molecule is computed whole."""
    info = {}
    mol, _ = prepare(Chem.MolFromSmiles(smiles), 7.0, info=info)
    ref = Chem.MolFromSmiles(smiles)
    assert mol.GetNumHeavyAtoms() == ref.GetNumHeavyAtoms() and formal_charge(mol) == 0
    assert info["removed_fragments"] == []


@pytest.mark.parametrize("smiles, metal", [
    ("[Fe+2].c1cc[cH-]c1.c1cc[cH-]c1", "Fe"),                              # ferrocene as ions: carbanion ligands
    ("[C-]#[O+].[C-]#[O+].[C-]#[O+].[C-]#[O+].[Co]", "Co"),                # metal carbonyl
    ("[C-]#[O+].[C-]#[O+].[Cl-].[Cl-].[Rh].[Rh]", "Rh"),                   # rhodium carbonyl chloride
    ("[Cu+2].c1ccc(P(c2ccccc2)c2ccccc2)cc1.CC(=O)[O-].CC(=O)[O-]", "Cu"),  # a phosphine next to the metal
    ("[Zn+2].[C-]#N.[C-]#N", "Zn"),                                        # cyanide: charge on carbon
    ("[Al+3].CCOP(=O)[O-].CCOP(=O)[O-].CCOP(=O)[O-].C=O", "Al"),           # a neutral organic fragment too
])
def test_a_metal_is_never_dropped_silently(smiles, metal):
    """Keeping the largest fragment of a complex drawn as ions computed cyclopentadiene for ferrocene and CO
    for a metal carbonyl. Now the molecule is refused with a reason naming the metal."""
    from quantabench.chem.standardize import UnsupportedStructure

    with pytest.raises(UnsupportedStructure, match=f"metal complex.*{metal}"):
        prepare(Chem.MolFromSmiles(smiles), 7.0)


@pytest.mark.parametrize("smiles, kept, removed", [
    # fosetyl-Al: one ethyl phosphonate anion, the Al3+ and the other two anions are the salt
    ("[Al+3].CCO[PH](=O)[O-].CCO[PH](=O)[O-].CCO[PH](=O)[O-]", "CCO[PH](=O)[O-]",
     ["[Al+3]", "CCO[PH](=O)[O-]", "CCO[PH](=O)[O-]"]),
    ("[Zn+2].CC(=O)[O-].CC(=O)[O-].O.O", "CC(=O)[O-]", ["O", "[Zn+2]", "CC(=O)[O-]", "O"]),   # with hydrate water
    ("[Ca+2].[O-]C(=O)C(F)(F)C(F)(F)F.[O-]C(=O)C(F)(F)C(F)(F)F", "O=C([O-])C(F)(F)C(F)(F)F",
     ["[Ca+2]", "O=C([O-])C(F)(F)C(F)(F)F"]),
    ("[Mg+2].[O-]S(=O)(=O)C(F)(F)F.[O-]S(=O)(=O)C(F)(F)F", "O=S(=O)([O-])C(F)(F)F",
     ["[Mg+2]", "O=S(=O)([O-])C(F)(F)F"]),
    ("[Ba+2].[O-]c1ccccc1.[O-]c1ccccc1", "[O-]c1ccccc1", ["[Ba+2]", "[O-]c1ccccc1"]),
    ("[Sr+2].CC(=O)[O-].CC(=O)[O-]", "CC(=O)[O-]", ["[Sr+2]", "CC(=O)[O-]"]),
    ("[Fe+3].[O-]C(=O)C(F)(F)F.[O-]C(=O)C(F)(F)F.[O-]C(=O)C(F)(F)F", "O=C([O-])C(F)(F)F",
     ["[Fe+3]", "O=C([O-])C(F)(F)F", "O=C([O-])C(F)(F)F"]),
    ("[Fe+2].[O-]C(=O)CC.[O-]C(=O)CC", "CCC(=O)[O-]", ["[Fe+2]", "CCC(=O)[O-]"]),
    ("[Cu+2].[O-]S(=O)(=O)c1ccccc1.[O-]S(=O)(=O)c1ccccc1", "O=S(=O)([O-])c1ccccc1",
     ["[Cu+2]", "O=S(=O)([O-])c1ccccc1"]),
    ("[Zn+2].[Cl-].[Cl-].[Cl-].[Cl-].C[N+](C)(C)CCCCCCCCCCCC.C[N+](C)(C)CCCCCCCCCCCC",   # ZnCl4(2-) salt
     "CCCCCCCCCCCC[N+](C)(C)C", ["[Zn+2]", "[Cl-]", "[Cl-]", "[Cl-]", "[Cl-]", "CCCCCCCCCCCC[N+](C)(C)C"]),
])
def test_bondless_metal_cations_of_organic_salts_are_counter_ions(smiles, kept, removed):
    """Al3+, Zn2+, Ca2+, Mg2+, Ba2+, Sr2+, Fe2+/3+ and Cu2+ drawn as separate ions next to an organic ion are a
    salt: the organic ion is computed with its charge as drawn (no pH model), the rest is named in
    removed_fragments."""
    info = {}
    mol, notes = prepare(Chem.MolFromSmiles(smiles), None, info=info)
    assert Chem.MolToSmiles(Chem.RemoveHs(mol)) == kept
    assert sorted(info["removed_fragments"]) == sorted(removed)
    assert formal_charge(mol) == Chem.GetFormalCharge(Chem.MolFromSmiles(kept))
    assert any("kept the largest of" in n for n in notes)


def test_covalently_bonded_salt_metals_stay_whole():
    """Only a bondless metal cation is a counter-ion: an Al-O or Zn-C bond drawn in the record is kept."""
    for smiles in ("CCO[Al](OCC)OCC", "CC[Zn]CC"):
        info = {}
        mol, _ = prepare(Chem.MolFromSmiles(smiles), None, info=info)
        assert mol.GetNumHeavyAtoms() == Chem.MolFromSmiles(smiles).GetNumHeavyAtoms()
        assert info["removed_fragments"] == []


@pytest.mark.parametrize("smiles, charge", [("C1CNCCN1", 1),          # piperazine: pKa2 5.6
                                            ("CN1CCN(C)CC1", 1),
                                            ("NCCN", 1),              # ethylenediamine: pKa2 6.9
                                            ("NC(=N)N", 1),           # guanidine: one proton, not one per NH2
                                            ("C[N+](C)(C)CCN", 1),    # an amine next to a quaternary ammonium
                                            ("NCCCN", 2),             # 1,3-diamine (4 bonds): both
                                            ("NCCCCCCN", 2)])
def test_close_basic_sites_get_one_proton_at_ph7(smiles, charge):
    """The pH-7 rules protonated every aliphatic amine: piperazine and 1,2-diamines came out as dications. Two basic
    sites within three bonds now get one proton."""
    mol, notes = prepare(Chem.MolFromSmiles(smiles), 7.0)
    assert formal_charge(mol) == charge
    if charge == 1 and "+" not in smiles:
        assert any(n.startswith("left ") and "within 3 bonds" in n for n in notes)


def test_inchikey_of_ions_and_unrepresentable_species():
    from quantabench.chem.standardize import inchikey

    acid, _ = inchikey(Chem.MolFromSmiles("CC(=O)O"))
    anion, _ = inchikey(Chem.MolFromSmiles("CC(=O)[O-]"))
    assert acid == "QTBSBXVTEAMEQO-UHFFFAOYSA-N" and anion == "QTBSBXVTEAMEQO-UHFFFAOYSA-M"   # -M: one H+ removed
    assert inchikey(Chem.AddHs(Chem.MolFromSmiles("CC(=O)[O-]")))[0] == anion                 # explicit H: same key
    assert inchikey(None) == ("", "no structure")
    key, why = inchikey(Chem.MolFromSmiles("*C", sanitize=False))      # a dummy atom: InChI has no key for it
    assert key == "" and why


def test_whole_record_when_fragments_are_kept():
    info = {}
    mol, notes = prepare(Chem.MolFromSmiles("[Na+].[O-]C(=O)C(F)(F)F"), None, keep_largest_fragment=False, info=info)
    assert formal_charge(mol) == 0 and "Na" in Chem.MolToSmiles(mol) and info["removed_fragments"] == []
    assert any("2 fragments computed together" in n for n in notes)


def test_phosphonate_dianion_charge_is_reported():
    info = {}
    mol, notes = prepare(Chem.MolFromSmiles("[O-]P([O-])(=O)CCC(F)(F)C(F)(F)F"), 7.0, info=info)
    assert info["input_charge"] == -2 and formal_charge(mol) == -2
    assert not any("as drawn" in n for n in notes)       # unchanged charge: no charge note
    mol, notes = prepare(Chem.MolFromSmiles("[O-]P([O-])(=O)CCC(F)(F)C(F)(F)F"), 0.0)
    assert formal_charge(mol) == 0 and any("charge -2 as drawn, +0 computed" in n for n in notes)


@pytest.mark.parametrize("smiles, charge, mult", [("CCO", 0, 1), ("CC[CH2]", 0, 2), ("[CH2]", 0, 3),
                                                  ("CC(C)(C)[N]", 0, 3), ("[O-][N+](=O)C", 0, 1),
                                                  ("C[N+](C)(C)C", 1, 1)])
def test_spin_multiplicity_from_drawn_radicals(smiles, charge, mult):
    from quantabench.chem.standardize import spin_multiplicity

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    m, note = spin_multiplicity(mol, charge)
    assert m == mult and (bool(note) == (mult != 1))


def test_spin_multiplicity_checks_an_explicit_request():
    from quantabench.chem.standardize import UnsupportedStructure, spin_multiplicity

    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    assert spin_multiplicity(mol, 0, 3) == (3, "")
    with pytest.raises(UnsupportedStructure, match="impossible"):
        spin_multiplicity(mol, 0, 2)
    assert spin_multiplicity(mol, 1)[0] == 2       # a radical cation without drawn radicals: doublet


@pytest.mark.parametrize("smiles", ["*C(F)(F)C(F)(F)F", "[1*]CC(=O)O"])
def test_placeholder_atoms_are_refused(smiles):
    """R groups and '*' (and molfile atom lists) are not a structure: ETKDG aborted on them with an RDKit
    invariant violation and a C++ stack trace."""
    from quantabench.chem.standardize import UnsupportedStructure

    with pytest.raises(UnsupportedStructure, match="not a definite structure"):
        prepare(Chem.MolFromSmiles(smiles), 7.0)


def test_unphysical_polarizability_is_flagged():
    from quantabench.chem.properties import unphysical_polarizability

    numbers = [8, 15, 8, 8, 8] + [6] * 10 + [9] * 21
    assert unphysical_polarizability([20.0, 25.0, 40.0], numbers) == ""
    assert "unbound" in unphysical_polarizability([4.0e5, 4.2e5, 4.3e5], numbers, -2)
    assert unphysical_polarizability([-3.0, 5.0, 9.0], numbers)
