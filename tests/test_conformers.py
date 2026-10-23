import numpy as np
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
