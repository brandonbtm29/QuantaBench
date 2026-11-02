"""Gaussian cube files: read/write round trips, units, MO cubes, skewed grids, sampling, VTK ordering."""

import numpy as np
import pytest

from quantabench.chem import cube as cm

B = cm.BOHR_A


def _cube(shape=(5, 4, 3), axes=None, nval=1, seed=0):
    rng = np.random.default_rng(seed)
    axes = np.diag([0.3, 0.25, 0.2]) if axes is None else np.asarray(axes, float)
    data = rng.normal(size=(*shape, nval)) * 10.0 ** rng.integers(-6, 2, size=(*shape, nval))
    return cm.Cube([-1.0, 0.5, 2.0], axes, data[..., 0], [8, 1, 1],
                   [[0.0, 0.0, 0.0], [1.43, 0.0, 1.1], [-1.43, 0.0, 1.1]], "title line\nsecond line",
                   values=data if nval > 1 else None)


def test_round_trip(tmp_path):
    c = _cube()
    p = cm.write_cube(tmp_path / "a.cube", c)
    lines = p.read_text().splitlines()
    assert lines[0] == "title line" and lines[1] == "second line"
    assert len(lines) == 2 + 4 + 3 + 5 * 4          # header, atoms, one line per z row (3 values each)
    r = cm.read_cube(p)
    assert r.shape == (5, 4, 3)
    np.testing.assert_allclose(r.data, c.data, rtol=2e-5, atol=1e-30)
    np.testing.assert_allclose(r.origin_bohr, c.origin_bohr)
    np.testing.assert_allclose(r.axes_bohr, c.axes_bohr)
    np.testing.assert_allclose(r.positions_bohr, c.positions_bohr)
    assert list(r.numbers) == [8, 1, 1] and r.comment == c.comment
    np.testing.assert_allclose(r.positions_A, c.positions_bohr * B)


def test_z_fastest_and_line_wrapping(tmp_path):
    """Values are written with z fastest, 6 per line, and a new line for every z row."""
    data = np.arange(2 * 1 * 8, dtype=float).reshape(2, 1, 8)
    c = cm.Cube(np.zeros(3), np.eye(3), data, [1], [[0, 0, 0]], "a\nb")
    lines = cm.write_cube(tmp_path / "z.cube", c).read_text().splitlines()[7:]
    assert [len(x.split()) for x in lines] == [6, 2, 6, 2]
    assert float(lines[0].split()[1]) == 1.0 and float(lines[2].split()[0]) == 8.0   # data[1, 0, 0]


def test_angstrom_units(tmp_path):
    """A negative voxel count means the origin, steps and atom coordinates are in Å."""
    text = ("ang\ncube\n    1   -1.000000    0.000000    0.000000\n   -2    0.500000    0.000000    0.000000\n"
            "   -2    0.000000    0.500000    0.000000\n   -2    0.000000    0.000000    0.500000\n"
            "    1    1.000000    0.200000    0.000000    0.000000\n" + " 1.0 2.0\n 3.0 4.0\n 5.0 6.0\n 7.0 8.0\n")
    p = tmp_path / "ang.cube"
    p.write_text(text)
    c = cm.read_cube(p)
    np.testing.assert_allclose(c.origin_A, [-1.0, 0, 0])
    np.testing.assert_allclose(c.axes_A, np.eye(3) * 0.5)
    np.testing.assert_allclose(c.positions_A, [[0.2, 0, 0]])
    assert c.data[1, 1, 1] == 8.0 and c.data[0, 0, 1] == 2.0
    again = cm.read_cube(cm.write_cube(tmp_path / "bohr.cube", c))     # written back in bohr
    np.testing.assert_allclose(again.origin_A, c.origin_A, atol=1e-6)
    np.testing.assert_allclose(again.data, c.data)


def test_mo_cube_negative_atom_count(tmp_path):
    """Negative atom count: an MO-list line follows the atoms, and every point has one value per MO."""
    c = _cube(nval=3)
    c.mo_ids = [5, 6, 9]
    p = cm.write_cube(tmp_path / "mo.cube", c)
    lines = p.read_text().splitlines()
    assert int(lines[2].split()[0]) == -3 and lines[9].split() == ["3", "5", "6", "9"]
    r = cm.read_cube(p)
    assert r.mo_ids == [5, 6, 9] and r.values.shape == (5, 4, 3, 3)
    np.testing.assert_allclose(r.values, c.values, rtol=2e-5, atol=1e-30)
    np.testing.assert_allclose(r.data, c.values[..., 0], rtol=2e-5, atol=1e-30)
    grid = r.to_pyvista()
    assert {"values", "values_5", "values_6", "values_9"} <= set(grid.point_data.keys())


def test_mo_list_wrapped_over_lines(tmp_path):
    """Gaussian wraps long MO lists; the reader keeps reading until it has NMO numbers."""
    head = ("mo\ncube\n   -1    0.0 0.0 0.0\n    1    1.0 0.0 0.0\n    1    0.0 1.0 0.0\n    2    0.0 0.0 1.0\n"
            "    6    6.0 0.0 0.0 0.0\n    4    1    2\n    3    4\n")
    p = tmp_path / "wrap.cube"
    p.write_text(head + " 1 2 3 4 5 6 7 8\n")
    r = cm.read_cube(p)
    assert r.mo_ids == [1, 2, 3, 4] and r.values.shape == (1, 1, 2, 4)
    assert list(r.values[0, 0, 1]) == [5, 6, 7, 8]


def test_several_values_per_point_from_header(tmp_path):
    c = _cube(nval=2)
    r = cm.read_cube(cm.write_cube(tmp_path / "nval.cube", c))
    assert r.values.shape[-1] == 2 and not r.mo_ids
    np.testing.assert_allclose(r.values, c.values, rtol=2e-5, atol=1e-30)


def test_fortran_exponents(tmp_path):
    p = tmp_path / "f.cube"
    p.write_text("x\ny\n    1 0 0 0\n    1 1 0 0\n    1 0 1 0\n    3 0 0 1\n    1 1.0 0 0 0\n"
                 "  0.12345-100  1.0D+00  2.5E-01\n")
    np.testing.assert_allclose(cm.read_cube(p).data.ravel(), [0.12345e-100, 1.0, 0.25])


def test_sampling_is_trilinear_and_nan_outside():
    """A linear field is reproduced exactly inside the grid, also on a skewed (non-orthogonal) grid."""
    for axes in (np.diag([0.3, 0.25, 0.2]), [[0.3, 0.0, 0.0], [0.1, 0.25, 0.0], [0.05, -0.04, 0.2]]):
        c = _cube(shape=(7, 6, 5), axes=axes)
        pts = c.points_A()
        f = lambda p: 1.0 + 2.0 * p[:, 0] - 3.0 * p[:, 1] + 0.5 * p[:, 2]   # noqa: E731
        c.data = f(pts).reshape(c.shape)
        rng = np.random.default_rng(3)
        idx = rng.uniform(0, 1, size=(200, 3)) * (np.array(c.shape) - 1)
        q = (c.origin_bohr + idx @ c.axes_bohr) * B
        np.testing.assert_allclose(c.sample(q), f(q), atol=1e-10)
        np.testing.assert_allclose(c.sample(pts[[0, -1]]), c.data.ravel()[[0, -1]])   # corners exactly
        outside = (c.origin_bohr + np.array([[-0.5, 1, 1], [1, 1, c.shape[2] - 0.5]]) @ c.axes_bohr) * B
        assert np.isnan(c.sample(outside)).all()


@pytest.mark.parametrize("skewed", [False, True])
def test_to_pyvista_ordering(skewed):
    """VTK wants x fastest; the cube stores z fastest. Every point and value must land where it belongs."""
    pytest.importorskip("pyvista")
    import pyvista as pv

    axes = [[0.3, 0.0, 0.0], [0.1, 0.25, 0.0], [0.0, 0.05, 0.2]] if skewed else np.diag([0.3, 0.25, 0.2])
    c = _cube(shape=(5, 4, 3), axes=axes)
    grid = c.to_pyvista()
    assert isinstance(grid, pv.StructuredGrid if skewed else pv.ImageData)
    assert c.is_orthogonal is (not skewed)
    nx, ny, nz = c.shape
    assert tuple(grid.dimensions) == (nx, ny, nz)
    vals = grid.point_data["values"]
    for i, j, k in [(0, 0, 0), (4, 0, 0), (0, 3, 0), (0, 0, 2), (3, 2, 1), (4, 3, 2)]:
        pid = i + nx * (j + ny * k)
        assert vals[pid] == c.data[i, j, k]
        np.testing.assert_allclose(grid.points[pid], (c.origin_bohr + np.array([i, j, k]) @ c.axes_bohr) * B,
                                   atol=1e-6)


def test_rejects_truncated_data(tmp_path):
    p = tmp_path / "bad.cube"
    p.write_text("x\ny\n    1 0 0 0\n    2 1 0 0\n    2 0 1 0\n    2 0 0 1\n    1 1.0 0 0 0\n 1 2 3\n")
    with pytest.raises(ValueError, match="expected 8 values"):
        cm.read_cube(p)
