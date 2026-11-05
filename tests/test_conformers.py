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
    """RDKit 2026.09's small-ring torsions abort plain cyclopentanes with an invariant violation (every
    seed); the embedding is repeated without them, with a note."""
    from quantabench.chem.standardize import prepare

    mol, _ = prepare(Chem.MolFromSmiles(smiles), 7.0)
    cs = confs.generate(mol, max_conformers=4)
    assert cs.n_embedded >= 1 and cs.forcefield == "MMFF94s"
    from rdkit import rdBase

    if rdBase.rdkitVersion.startswith("2026.09"):
        assert any("small-ring torsions" in n for n in cs.notes)
