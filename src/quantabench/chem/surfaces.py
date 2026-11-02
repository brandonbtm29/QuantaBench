"""Molecular surfaces: density isosurfaces, ESP mapping and surface descriptors.

CONTRACT (implemented by the surfaces builder):

    density_surface(density: Cube, iso=0.002) -> pyvista.PolyData   (Å, triangulated, outward normals)
    map_esp(surface, esp: Cube) -> np.ndarray                        (ESP in kcal/mol at each vertex)
    surface_descriptors(density_path_or_cube, esp_path_or_cube=None, iso=0.002) -> dict
        keys (all floats): surf_area_A2, surf_volume_A3 (enclosed by the isosurface),
        esp_min_kcal, esp_max_kcal, esp_mean_kcal, esp_mean_pos_kcal, esp_mean_neg_kcal,
        esp_sigma2_pos, esp_sigma2_neg, esp_sigma2_tot, esp_nu (balance), esp_pi (local polarity),
        esp_area_pos_A2, esp_area_neg_A2     (Murray-Politzer GIPF quantities, area-weighted)
    charge_esp_surface(numbers, positions_A, charges, kind="vdw", scale=1.0, resolution=...) ->
        (pyvista.PolyData, esp_kcal per vertex)
        Fallback when no QM cube exists: a vdW (or SAS) surface from atom spheres (marching cubes on a
        smooth distance field), ESP from point charges (Coulomb, e/Å -> kcal/mol/e).
"""

from __future__ import annotations


def surface_descriptors(density, esp=None, iso: float = 0.002) -> dict:
    raise NotImplementedError("surface descriptors are not implemented yet")
