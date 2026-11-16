import time

import numpy as np
import pytest
from conftest import PFOA
from rdkit import Chem

from quantabench.chem import conformers as confs


def test_generate_and_prune():
    mol = Chem.AddHs(Chem.MolFromSmiles(PFOA))
    cs = confs.generate(mol, max_conformers=8, window_kcal=10, rmsd_threshold=0.5, seed=1)
    assert cs.forcefield == "MMFF94s"
    assert 1 <= len(cs.conf_ids) <= cs.n_embedded <= 8
    assert cs.energies_kcal == sorted(cs.energies_kcal)


def test_prune_removes_duplicates():
    mol = Chem.AddHs(Chem.MolFromSmiles("CCCC"))
    cs = confs.generate(mol, max_conformers=20, window_kcal=20, rmsd_threshold=0.3, seed=1)
    # butane has two distinct conformer families (anti, gauche +/-): far fewer than 20 survive
    assert len(cs.conf_ids) <= 4


def test_boltzmann():
    w = confs.boltzmann_weights([0.0, 0.0, 10.0])
    assert np.isclose(w.sum(), 1) and np.isclose(w[0], w[1]) and w[2] < 1e-6


@pytest.mark.parametrize("smiles", ["C1CCCC1", "O[C@@H]1CCC[C@@H]1O"])
def test_cyclopentanes_embed(smiles):
    """RDKit 2026.09.1's small-ring torsions abort plain cyclopentanes with an invariant violation (every
    seed); the embedding is repeated without them, with a note. 2026.09.2 embeds them (no note)."""
    from quantabench.chem.standardize import prepare

    mol, _ = prepare(Chem.MolFromSmiles(smiles), 7.0)
    cs = confs.generate(mol, max_conformers=4)
    assert cs.n_embedded >= 1 and cs.forcefield == "MMFF94s"
    from rdkit import rdBase

    if rdBase.rdkitVersion == "2026.09.1":
        assert any("small-ring torsions" in n for n in cs.notes)


# ----------------------------------------------------------------------------- time limit of the embedding
#: a 60-heavy-atom cyclic peptide, cyclo(AFLG AFLG SG): ~0.4 s per ETKDG conformer on one core
CYCLIC_PEPTIDE = ("N1[C@@H](C)C(=O)N[C@@H](Cc2ccccc2)C(=O)N[C@@H](CC(C)C)C(=O)NCC(=O)N[C@@H](C)C(=O)"
                  "N[C@@H](Cc2ccccc2)C(=O)N[C@@H](CC(C)C)C(=O)NCC(=O)N[C@@H](CO)C(=O)NCC1=O")


def test_embed_time_limit_is_a_share_of_the_molecule_limit():
    from quantabench import resources

    assert confs.embed_time_limit(1800) == confs.EMBED_TIME_FRACTION * 1800
    assert confs.embed_time_limit(0) == 0.0
    try:
        resources.set_molecule_limits(molecule_timeout_s=400)
        assert confs.embed_time_limit() == confs.EMBED_TIME_FRACTION * 400
        resources.set_molecule_limits()
        assert confs.embed_time_limit() == 0.0          # no run recorded a limit: none
    finally:
        resources.set_molecule_limits()


def test_a_limit_that_is_not_reached_changes_nothing():
    mol = Chem.AddHs(Chem.MolFromSmiles(PFOA))
    plain, limited = Chem.Mol(mol), Chem.Mol(mol)
    notes: list[str] = []
    a = confs.embed(plain, 10, seed=3)
    b = confs.embed(limited, 10, seed=3, notes=notes, timeout_s=600)
    assert a == b and not notes
    for cid in a:
        assert np.allclose(plain.GetConformer(cid).GetPositions(), limited.GetConformer(cid).GetPositions())


def test_embedding_stops_at_its_time_limit():
    mol = Chem.AddHs(Chem.MolFromSmiles(CYCLIC_PEPTIDE))
    notes: list[str] = []
    t0 = time.monotonic()
    ids = confs.embed(mol, 50, seed=1, notes=notes, timeout_s=1.5)
    elapsed = time.monotonic() - t0
    assert 1 <= len(ids) < 50 and mol.GetNumConformers() == len(ids)
    assert elapsed < 6.0                                    # without the limit: ~20 s
    assert any(n.startswith(("conformer embedding limited by the time limit",
                             "conformer embedding stopped by the time limit")) for n in notes), notes


def test_an_etkdg_start_that_finds_no_conformer_gives_way(monkeypatch):
    """FAIL-1: with RDKit 2026.09's small-ring torsions, ETKDG found no conformer for digitoxin, digoxin, gitoxin,
    glycyrrhizin and maltopentaose within the 450 s embedding limit; RDKit's timeout then returned the id -1, which
    QuantaBench took for a conformer ("Bad Conformer Id"). Now -1 is dropped, the first conformer may take
    FIRST_CONFORMER_S, and the next way of embedding (without small-ring torsions) follows, with a note."""
    from rdkit.Chem import AllChem

    real = AllChem.EmbedMultipleConfs
    calls = []

    def embed(mol, numConfs=1, params=None):
        calls.append((params.useSmallRingTorsions, params.useRandomCoords, params.timeout))
        if params.useSmallRingTorsions:
            return [-1]                                       # what a call stopped by RDKit's timeout returns
        return real(mol, numConfs=numConfs, params=params)

    monkeypatch.setattr(confs.AllChem, "EmbedMultipleConfs", embed)
    mol = Chem.AddHs(Chem.MolFromSmiles("OCC(O)CO"))
    notes = []
    ids = confs.embed(mol, 6, seed=3, notes=notes)
    assert ids and all(i >= 0 for i in ids) and mol.GetNumConformers() == len(ids)
    assert calls[0] == (True, False, int(confs.FIRST_CONFORMER_S))   # the first conformer, capped
    assert any("with small-ring torsions found no conformer" in n for n in notes)
    assert any("RDKit ETKDG found none; embedded without small-ring torsions" in n for n in notes)
    cs = confs.generate(Chem.AddHs(Chem.MolFromSmiles("OCC(O)CO")), max_conformers=6, timeout_s=0)
    assert cs.n_embedded >= 1                                 # the whole conformer search goes on from there


def test_no_way_of_embedding_finds_one_is_a_clear_error(monkeypatch):
    monkeypatch.setattr(confs.AllChem, "EmbedMultipleConfs", lambda mol, numConfs=1, params=None: [-1])
    with pytest.raises(RuntimeError, match="could not embed"):
        confs.embed(Chem.AddHs(Chem.MolFromSmiles("CCO")), 4)
