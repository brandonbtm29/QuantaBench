"""Graphite slabs (build.surfaces) and molecular surfaces / GIPF descriptors (chem.surfaces)."""

import os
import subprocess
import time

import numpy as np
import pytest
from ase import Atoms

from quantabench.build import surfaces
from quantabench.chem import cube as cm
from quantabench.chem import surfaces as msurf

B = cm.BOHR_A
K = msurf.COULOMB_KCAL


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


# ====================================================================== molecular surfaces (chem.surfaces)
def _sphere_cubes(r0_A=2.0, h=0.2, decay=0.6, iso=0.002, axes=None, pad=2.5, charges=((0.0, 0.0, 0.0, 1.0),)):
    """Density A·exp(-r/a) (bohr units) that equals ``iso`` at r0, and the ESP (hartree/e) of point charges
    (x, y, z in Å, q in e), on the same grid."""
    r0 = r0_A / B
    amp = iso * np.exp(r0 / decay)
    axes = np.eye(3) * h if axes is None else np.asarray(axes, float)
    n = int(np.ceil(2 * (r0 + pad) / h)) + 1
    origin = -0.5 * (n - 1) * axes.sum(axis=0) + np.array([0.013, -0.021, 0.007])  # break grid symmetry
    proto = cm.Cube(origin, axes, np.zeros((n, n, n)), [1], [[0.0, 0.0, 0.0]], "synthetic\nsphere")
    p = proto.points_A() / B
    dens = proto.with_data(amp * np.exp(-np.linalg.norm(p, axis=1) / decay))
    pot = np.zeros(len(p))
    for x, y, z, q in charges:
        pot += q / np.maximum(np.linalg.norm(p - np.array([x, y, z]) / B, axis=1), 1e-9)
    return dens, dens.with_data(pot)


def _outward(surf, center=(0.0, 0.0, 0.0)):
    radial = surf.points - np.asarray(center)
    radial /= np.linalg.norm(radial, axis=1)[:, None]
    return np.einsum("ij,ij->i", surf.point_normals, radial)


def test_density_sphere_area_and_volume():
    dens, _ = _sphere_cubes()
    assert dens.shape[0] < 100
    surf = msurf.density_surface(dens, 0.002)
    d = msurf.gipf_descriptors(surf)
    assert set(d) == {"surf_area_A2", "surf_volume_A3"}
    assert d["surf_area_A2"] == pytest.approx(4 * np.pi * 2.0**2, rel=0.01)
    assert d["surf_volume_A3"] == pytest.approx(4 / 3 * np.pi * 2.0**3, rel=0.01)
    assert surf.is_all_triangles
    dots = _outward(surf)
    assert dots.mean() > 0.99 and dots.min() > 0.5           # normals point outward (to lower density)
    np.testing.assert_allclose(msurf.vertex_areas(surf).sum(), d["surf_area_A2"])


def test_density_sphere_on_skewed_grid():
    """Non-orthogonal cube axes go through a StructuredGrid and the general contour filter."""
    h = 0.25
    axes = [[h, 0.0, 0.0], [0.06, h, 0.0], [0.03, -0.04, h]]
    dens, esp = _sphere_cubes(h=h, axes=axes, pad=3.0)
    assert not dens.is_orthogonal and dens.shape[0] < 100
    surf, v = msurf.qm_esp_surface(dens, esp, 0.002)
    d = msurf.gipf_descriptors(surf, v)
    assert d["surf_area_A2"] == pytest.approx(4 * np.pi * 2.0**2, rel=0.02)
    assert d["surf_volume_A3"] == pytest.approx(4 / 3 * np.pi * 2.0**3, rel=0.02)
    assert d["esp_mean_kcal"] == pytest.approx(K / 2.0, rel=0.01)
    assert _outward(surf).mean() > 0.99


def test_point_charge_esp_is_constant_on_the_sphere():
    """A +1 e point charge at the center: V = kq/r0 = 166.03 kcal/mol everywhere on the r0 = 2 Å sphere."""
    dens, esp = _sphere_cubes()
    d = msurf.surface_descriptors(dens, esp, iso=0.002)
    target = K * 1.0 / 2.0
    for key in ("esp_mean_kcal", "esp_min_kcal", "esp_max_kcal", "esp_mean_pos_kcal"):
        assert d[key] == pytest.approx(target, rel=0.005), key
    assert d["esp_sigma2_pos"] < 0.05 and d["esp_sigma2_neg"] == 0.0 and d["esp_nu"] == 0.0
    assert d["esp_pi"] < 0.2
    assert d["esp_area_neg_A2"] == 0.0 and d["esp_mean_neg_kcal"] == 0.0
    assert d["esp_area_pos_A2"] == pytest.approx(d["surf_area_A2"])
    assert set(d) == set(msurf.DESCRIPTOR_KEYS) and all(isinstance(x, float) for x in d.values())


def test_dipole_gives_balanced_potential():
    """±0.5 e at z = ±0.5 Å: mirror-symmetric ESP, so σ²+ = σ²-, ν = 0.25, V̄ = 0, A+ = A-."""
    dens, esp = _sphere_cubes(charges=((0, 0, 0.5, 0.5), (0, 0, -0.5, -0.5)))
    d = msurf.surface_descriptors(dens, esp)
    assert d["esp_nu"] == pytest.approx(0.25, abs=0.002)
    assert d["esp_sigma2_pos"] == pytest.approx(d["esp_sigma2_neg"], rel=0.03)
    assert d["esp_sigma2_tot"] == pytest.approx(d["esp_sigma2_pos"] + d["esp_sigma2_neg"])
    assert d["esp_max_kcal"] == pytest.approx(-d["esp_min_kcal"], rel=0.02)
    assert d["esp_mean_pos_kcal"] == pytest.approx(-d["esp_mean_neg_kcal"], rel=0.02)
    assert abs(d["esp_mean_kcal"]) < 0.02 * d["esp_max_kcal"]
    assert d["esp_area_pos_A2"] == pytest.approx(d["esp_area_neg_A2"], rel=0.02)
    # analytic extremes on the poles: k (q/(r0-d) - q/(r0+d)) with r0 = 2, d = 0.5
    assert d["esp_max_kcal"] == pytest.approx(K * 0.5 * (1 / 1.5 - 1 / 2.5), rel=0.02)
    assert d["esp_pi"] > 0


def test_gipf_formulas_on_a_hand_checked_mesh():
    """Irregular tetrahedron: vertex weights differ, so area weighting matters; compare with a direct sum."""
    import pyvista as pv

    pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    faces = np.array([[3, 0, 2, 1], [3, 0, 1, 3], [3, 0, 3, 2], [3, 1, 2, 3]]).ravel()
    mesh = pv.PolyData(pts, faces)
    v = np.array([3.0, -1.0, -2.0, 0.5])
    big = np.sqrt(3) / 2                                     # area of the slanted face
    w = np.array([1.5, 1.0 + big, 1.0 + big, 1.0 + big]) / 3  # 1/3 of each adjacent triangle's area
    np.testing.assert_allclose(msurf.vertex_areas(mesh), w)
    d = msurf.gipf_descriptors(mesh, v)
    assert d["surf_area_A2"] == pytest.approx(1.5 + big)
    assert d["surf_volume_A3"] == pytest.approx(1 / 6)
    mean = (w * v).sum() / w.sum()
    p, n = v > 0, v < 0
    mp, mn = (w[p] * v[p]).sum() / w[p].sum(), (w[n] * v[n]).sum() / w[n].sum()
    s2p = (w[p] * (v[p] - mp) ** 2).sum() / w[p].sum()
    s2n = (w[n] * (v[n] - mn) ** 2).sum() / w[n].sum()
    expect = {"esp_mean_kcal": mean, "esp_mean_pos_kcal": mp, "esp_mean_neg_kcal": mn, "esp_sigma2_pos": s2p,
              "esp_sigma2_neg": s2n, "esp_sigma2_tot": s2p + s2n, "esp_nu": s2p * s2n / (s2p + s2n) ** 2,
              "esp_pi": (w * np.abs(v - mean)).sum() / w.sum(), "esp_area_pos_A2": w[p].sum(),
              "esp_area_neg_A2": w[n].sum(), "esp_min_kcal": -2.0, "esp_max_kcal": 3.0}
    for key, val in expect.items():
        assert d[key] == pytest.approx(val), key
    assert d["esp_mean_kcal"] != pytest.approx(v.mean())     # not the plain vertex average


def test_open_isosurface_is_reported():
    dens, _ = _sphere_cubes(pad=-0.3)                        # the box cuts through the 0.002 au sphere
    with pytest.raises(msurf.OpenSurfaceError, match="edge of the cube grid"):
        msurf.density_surface(dens, 0.002)


def test_surface_descriptors_from_cube_files(tmp_path):
    dens, esp = _sphere_cubes(h=0.3)
    pd_, pe = cm.write_cube(tmp_path / "Dt.cube", dens), cm.write_cube(tmp_path / "ESP.cube", esp)
    d = msurf.surface_descriptors(pd_, pe, iso=0.002)
    assert d["surf_area_A2"] == pytest.approx(4 * np.pi * 4.0, rel=0.02)
    assert d["esp_mean_kcal"] == pytest.approx(K / 2.0, rel=0.01)
    only = msurf.surface_descriptors(str(pd_))
    assert set(only) == {"surf_area_A2", "surf_volume_A3"}
    assert only["surf_volume_A3"] == pytest.approx(d["surf_volume_A3"])
    # a lower isovalue encloses more
    assert msurf.surface_descriptors(pd_, iso=0.001)["surf_volume_A3"] > d["surf_volume_A3"]


def test_charge_surface_single_atom_vdw_and_sas():
    r = 1.52                                                  # Bondi O
    for kind, radius in (("vdw", r), ("sas", r + 1.4)):
        surf, v = msurf.charge_esp_surface([8], [[0.3, -0.2, 0.1]], [-0.5], kind=kind, spacing=0.2)
        d = msurf.gipf_descriptors(surf, v)
        assert d["surf_area_A2"] == pytest.approx(4 * np.pi * radius**2, rel=0.015), kind
        assert d["surf_volume_A3"] == pytest.approx(4 / 3 * np.pi * radius**3, rel=0.015), kind
        np.testing.assert_allclose(v, -0.5 * K / radius, rtol=0.01)
        assert _outward(surf, [0.3, -0.2, 0.1]).mean() > 0.99
        np.testing.assert_allclose(surf.point_data["esp_kcal"], v)
    s1, _ = msurf.charge_esp_surface([8], [[0, 0, 0]], [0.0], scale=1.2, spacing=0.2)
    assert msurf.gipf_descriptors(s1)["surf_area_A2"] == pytest.approx(4 * np.pi * (1.2 * r) ** 2, rel=0.015)


def test_charge_surface_union_of_two_spheres():
    """Two overlapping spheres: the smooth union encloses at least the hard union, and only a little more."""
    r1, r2, dist = 1.70, 1.52, 1.25                           # Bondi C and O, a C=O bond
    surf, v = msurf.charge_esp_surface([6, 8], [[0, 0, 0], [dist, 0, 0]], [0.4, -0.4], spacing=0.15)
    vol = msurf.gipf_descriptors(surf, v)["surf_volume_A3"]
    lens = np.pi * (r1 + r2 - dist) ** 2 * (dist**2 + 2 * dist * (r1 + r2) - 3 * (r1 - r2) ** 2) / (12 * dist)
    hard = 4 / 3 * np.pi * (r1**3 + r2**3) - lens
    assert hard * 0.995 < vol < hard * 1.03
    assert v.min() < 0 < v.max()
    assert surf.n_open_edges == 0


def test_charge_surface_pfoa_is_fast():
    from rdkit import Chem
    from rdkit.Chem import AllChem

    pfoa = "OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"
    mol = Chem.AddHs(Chem.MolFromSmiles(pfoa))
    assert AllChem.EmbedMolecule(mol, randomSeed=7) == 0
    AllChem.ComputeGasteigerCharges(mol)
    q = [a.GetDoubleProp("_GasteigerCharge") for a in mol.GetAtoms()]
    z = [a.GetAtomicNum() for a in mol.GetAtoms()]
    pos = mol.GetConformer().GetPositions()
    msurf.charge_esp_surface(z[:1], pos[:1], q[:1])           # imports and VTK start-up out of the timing
    t = time.perf_counter()
    surf, v = msurf.charge_esp_surface(z, pos, q)
    d = msurf.gipf_descriptors(surf, v)
    dt = time.perf_counter() - t
    assert dt < 1.0, f"{dt:.2f} s"
    assert surf.n_open_edges == 0 and surf.n_points > 2000
    assert 180 < d["surf_volume_A3"] < 280 and d["esp_min_kcal"] < 0 < d["esp_max_kcal"]
    sas, _ = msurf.charge_esp_surface(z, pos, q, kind="sas")
    assert msurf.gipf_descriptors(sas)["surf_area_A2"] > 1.4 * d["surf_area_A2"]


def test_charge_surface_rejects_bad_input():
    with pytest.raises(ValueError, match="kind"):
        msurf.charge_esp_surface([1], [[0, 0, 0]], [0.0], kind="ses")
    with pytest.raises(ValueError, match="charges"):
        msurf.charge_esp_surface([1, 1], [[0, 0, 0], [0.74, 0, 0]], [0.0])


_PSI4_WATER = """
import psi4
psi4.set_memory("1 GB")
psi4.set_num_threads(1)
psi4.core.set_output_file("psi4.out", False)
psi4.geometry('''0 1
O 0.000000 0.000000 0.000000
H 0.756950 0.000000 0.585882
H -0.756950 0.000000 0.585882
units angstrom
symmetry c1
no_com
no_reorient''')
e, wfn = psi4.energy("wb97x-d/def2-svp", return_wfn=True)
psi4.set_options({"cubeprop_tasks": ["density", "esp"], "cubic_grid_spacing": [0.2] * 3,
                  "cubic_grid_overage": [4.0] * 3})
psi4.cubeprop(wfn)
"""


@pytest.mark.external
def test_psi4_water_density_and_esp_surface(tmp_path):
    """Real data: wB97X-D/def2-SVP water (about 5 s). Set QUANTABENCH_PSI4_PYTHON to a Python with psi4."""
    from quantabench.chem import external

    py = external.find_python_with("psi4", "psi4_python")
    if py is None:
        pytest.skip("Psi4 not found (set QUANTABENCH_PSI4_PYTHON to a Python that can import psi4)")
    script = tmp_path / "water_cubes.py"
    script.write_text(_PSI4_WATER, encoding="utf-8")
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PSI_SCRATCH=str(tmp_path))
    proc = subprocess.run([py, "-I", script.name], cwd=tmp_path, env=env, capture_output=True, text=True,
                          timeout=600, check=False)
    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-3000:]
    dens, esp = cm.read_cube(tmp_path / "Dt.cube"), cm.read_cube(tmp_path / "ESP.cube")
    assert list(dens.numbers) == [8, 1, 1]
    np.testing.assert_allclose(dens.positions_A[1], [0.75695, 0, 0.585882], atol=1e-4)   # input frame kept
    d = msurf.surface_descriptors(dens, esp, iso=0.002)
    assert 18.0 < d["surf_volume_A3"] < 23.0 and 33.0 < d["surf_area_A2"] < 40.0
    assert d["esp_min_kcal"] < -35 and d["esp_max_kcal"] > 40 and d["esp_nu"] > 0.2
    assert msurf.surface_descriptors(dens, esp, iso=0.001)["surf_volume_A3"] > d["surf_volume_A3"]
    surf, v = msurf.qm_esp_surface(dens, esp, 0.002)
    o_dist = np.linalg.norm(surf.points[np.argmin(v)] - dens.positions_A[0])
    h_dist = np.linalg.norm(surf.points[np.argmax(v)] - dens.positions_A[1:], axis=1).min()
    assert o_dist < 2.0 and h_dist < 1.5                      # minimum at the O lone pairs, maximum on an H
