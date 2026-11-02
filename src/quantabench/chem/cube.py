"""Gaussian cube files: read, write, sample.

CONTRACT (implemented by the surfaces builder):

    Cube(origin_bohr (3,), axes_bohr (3,3) rows = step vectors, data (nx,ny,nz) float64,
         numbers (N,), positions_bohr (N,3), comment)
    read_cube(path) -> Cube            # handles negative atom count (MO cubes) and >1 value per point
    write_cube(path, cube) -> Path
    Cube.points_A() / Cube.sample(points_A) -> trilinear interpolation (outside the grid -> nan)
    Cube.to_pyvista() -> pyvista.ImageData (or StructuredGrid for non-orthogonal axes), point data "values",
                         coordinates in Å
"""

from __future__ import annotations
