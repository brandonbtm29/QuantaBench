import numpy as np
from ase import Atoms

from quantabench.build import surfaces


def test_graphite_geometry():
    slab = surfaces.graphite_slab(4, 3, layers=2, vacuum=20)
    assert len(slab) == 4 * 4 * 3 * 2
    d = slab.get_all_distances(mic=True)
    np.fill_diagonal(d, 99)
    assert np.isclose(d.min(), surfaces.CC_BOND, atol=1e-3)
    # every atom has exactly three in-plane neighbours at 1.42 Å
    assert ((np.abs(d - surfaces.CC_BOND) < 0.01).sum(axis=1) == 3).all()
    z = sorted(set(np.round(slab.positions[:, 2], 3)))
    assert np.isclose(z[1] - z[0], surfaces.INTERLAYER)
    assert not surfaces.min_distance_report(slab, 1.0)


def test_place_molecule():
    slab = surfaces.graphite_slab(8, 5, layers=2)
    mol = Atoms("CO2", positions=[[0, 0, 0], [1.16, 0, 0], [-1.16, 0, 0]])
    system = surfaces.place_on_surface(slab, mol, height=3.0, orientation="flat")
    assert len(system) == len(slab) + 3
    tags = system.get_tags()
    vertical = system.positions[tags == 1, 2].min() - system.positions[tags == 0, 2].max()
    assert np.isclose(vertical, 3.0)
    assert surfaces.surface_adsorbate_gap(system) >= 3.0 - 1e-6
    assert (system.get_tags() == 1).sum() == 3
