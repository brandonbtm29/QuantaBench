"""Molecular surfaces: density isosurfaces, ESP mapping and surface descriptors.

CONTRACT (implemented by the surfaces builder):

    density_surface(density: Cube, iso=0.002) -> pyvista.PolyData   (Å, triangulated, outward normals)
    map_esp(surface, esp: Cube) -> np.ndarray                        (ESP in kcal/mol at each vertex)
    surface_descriptors(density_path_or_cube, esp_path_or_cube=None, iso=0.002) -> dict
        keys (all floats): surf_area_A2, surf_volume_A3 (enclosed by the isosurface),
        esp_min_kcal, esp_max_kcal, esp_mean_kcal, esp_mean_pos_kcal, esp_mean_neg_kcal,
        esp_sigma2_pos, esp_sigma2_neg, esp_sigma2_tot, esp_nu (balance), esp_pi (local polarity),
        esp_area_pos_A2, esp_area_neg_A2     (Murray-Politzer GIPF quantities, area-weighted)
        esp_polar_area_75_A2, _100_, _125_   (area where |V| > 75/100/125 kJ/mol, as Spartan's polar areas)
        esp_area_below_m250_A2, esp_area_m250_m225_A2, ..., esp_area_p225_p250_A2, esp_area_above_p250_A2
                                             (the ESP distribution: surface area per 25 kJ/mol of V, ESP_BINS_KJMOL)
    esp_profile(surface, esp_kcal, bin_kjmol=5.0) -> dict      (the finer distribution, saved with the results)
    profile_area(profile, lo_kjmol=None, hi_kjmol=None), profile_polar_area(profile, t_kjmol)
    charge_esp_surface(numbers, positions_A, charges, kind="vdw", scale=1.0, probe=1.4, spacing=0.25) ->
        (pyvista.PolyData, esp_kcal per vertex)
        Fallback when no QM cube exists: a vdW (or SAS) surface from atom spheres (contour of a smooth union of
        spheres), ESP from point charges (Coulomb, k = 332.0637 kcal·Å/(mol·e²)).

Surfaces
--------
The molecular surface of Bader et al. is an isosurface of the electron density, usually 0.001 au (the GIPF
papers) or 0.002 au (closer to crystal-packing volumes); ``density_surface`` contours a density cube with VTK
(flying edges on axis-aligned grids, ``vtkContourFilter`` on skewed ones), triangulates, merges duplicate
points, orients the triangles so that the enclosed volume is positive and computes point normals from that
winding, so they point outward (towards lower density). An isosurface that reaches the edge of the grid is not
closed: :class:`OpenSurfaceError` says so instead of returning a wrong area and volume.

The charge-based fallback contours ``ln(sum_i exp(-k (|r - r_i| - R_i))) / k = 0``, a smooth minimum of the
atoms' signed sphere distances (``k`` = :data:`SMOOTH_K` per Å, so seams between spheres are filled by at most
``ln(2)/k`` ~ 0.07 Å). Radii are Bondi's van der Waals radii (*J. Phys. Chem.* 1964, 68, 441) as tabulated in
``ase.data.vdw_radii`` (completed by Mantina et al., *J. Phys. Chem. A* 2009, 113, 5806), the same table the
3D viewer's space-filling style draws; ``kind="sas"`` adds the probe radius (1.4 Å, water) to every radius.

Descriptors (Murray-Politzer GIPF)
----------------------------------
With surface potential V(r) and area elements as weights (each vertex carries one third of the area of its
triangles, so the statistics do not depend on how evenly the mesh samples the surface; Bulat et al. 2010)::

    V̄_S   = Σ w V / Σ w                       esp_mean_kcal
    V̄_S+  = Σ+ w V / A+,  V̄_S- = Σ- w V / A-   esp_mean_pos_kcal, esp_mean_neg_kcal (A± = Σ± w)
    σ²+   = Σ+ w (V - V̄_S+)² / A+              esp_sigma2_pos   (kcal/mol)²
    σ²-   = Σ- w (V - V̄_S-)² / A-              esp_sigma2_neg   (kcal/mol)²
    σ²tot = σ²+ + σ²-                          esp_sigma2_tot
    ν     = σ²+ σ²- / (σ²tot)²                 esp_nu  (0 ... 0.25; 0.25 when σ²+ = σ²-)
    Π     = Σ w |V - V̄_S| / Σ w                esp_pi  (local polarity), kcal/mol

Σ+ and Σ- run over vertices with V > 0 and V < 0. An empty sign class gives mean and variance 0, and ν = 0
when σ²tot = 0.

The ESP distribution
--------------------
The polar areas keep three numbers of the distribution of V over the surface. The whole distribution is kept as
well, as the surface area in each interval of V (kJ/mol, like Spartan's polar areas; the area elements as above,
so the intervals add up to ``surf_area_A2``):

- ``esp_area_<lo>_<hi>_A2``: 25 kJ/mol intervals from -250 to +250 kJ/mol (``m``/``p`` = minus/plus, so
  ``esp_area_m025_p000_A2`` is the area with -25 <= V < 0 kJ/mol), plus ``esp_area_below_m250_A2`` and
  ``esp_area_above_p250_A2`` for the tails (anions and cations reach past ±250). 22 columns, the same for every
  molecule, for QSAR tables. A neutral organic's surface is mostly within ±100 kJ/mol.
- :func:`esp_profile`: the same in 5 kJ/mol intervals over the molecule's own range, saved with each result
  (``qm["surface"]["esp_profile"]`` in the summary and ``<id>_esp_profile.csv``), so another binning or the polar
  area at any multiple of 5 kJ/mol (:func:`profile_polar_area`) can be had later without the cubes.

This is the ESP analogue of COSMO-RS's σ-profile (the area per interval of screening charge density; Klamt,
*J. Phys. Chem.* 1995, 99, 2224), which is what such distributions are usually fed to QSPR models as.

References: J. S. Murray, P. Lane, T. Brinck, K. Paulsen, M. E. Grice, P. Politzer, *J. Phys. Chem.* 1993, 97,
9369 (σ²tot, ν); J. S. Murray, T. Brinck, P. Lane, K. Paulsen, P. Politzer, *J. Mol. Struct. THEOCHEM* 1994,
307, 55 (the GIPF); J. S. Murray, Z. Peralta-Inga, P. Politzer, *Int. J. Quantum Chem.* 1999, 75, 267 (Π and
the definitions above, Eqs. 2-9); F. A. Bulat, A. Toro-Labbé, T. Brinck, J. S. Murray, P. Politzer,
*J. Mol. Model.* 2010, 16, 1679 (surface quadrature, areas, volumes); T. Lu, F. Chen, *J. Mol. Graph. Model.*
2012, 38, 314 (the area-weighted implementation in Multiwfn); R. F. W. Bader, M. T. Carroll, J. R. Cheeseman,
C. Chang, *J. Am. Chem. Soc.* 1987, 109, 7968 (isodensity envelopes).
"""

from __future__ import annotations

import numpy as np

from .cube import Cube, read_any

HARTREE_KCAL = 627.5095           # kcal/mol per hartree (ESP cubes are in hartree/e)
COULOMB_KCAL = 332.0637           # kcal·Å/(mol·e²)
SMOOTH_K = 10.0                   # 1/Å, sharpness of the smooth sphere union
PROBE_A = 1.4                     # Å, solvent probe (water) for kind="sas"
#: |ESP| thresholds (kJ/mol) of the polar areas, as Spartan's QSAR_ELPOT_POLAR_AREA_75/_100/_125 (its Settings
#: tab: "Polar Area Range 100 kJ/mol"); ``esp_polar_area_<T>_A2`` = area of the surface where |V| > T
POLAR_ESP_KJMOL = (75, 100, 125)
KJ_PER_KCAL = 4.184
#: edges (kJ/mol) of the ESP distribution's descriptor intervals; below the first and above the last are the tails
ESP_BINS_KJMOL = tuple(range(-250, 251, 25))
#: interval (kJ/mol) of the saved ESP profile (``esp_profile``)
PROFILE_BIN_KJMOL = 5.0


def _edge_label(x: float) -> str:
    return f"{'m' if x < 0 else 'p'}{abs(int(round(x))):03d}"


def esp_bin_keys(edges=ESP_BINS_KJMOL) -> list[str]:
    """Descriptor names of the ESP distribution: the low tail, one per interval, the high tail."""
    e = list(edges)
    return ([f"esp_area_below_{_edge_label(e[0])}_A2"]
            + [f"esp_area_{_edge_label(a)}_{_edge_label(b)}_A2" for a, b in zip(e[:-1], e[1:])]
            + [f"esp_area_above_{_edge_label(e[-1])}_A2"])


DESCRIPTOR_KEYS = ("surf_area_A2", "surf_volume_A3", "esp_min_kcal", "esp_max_kcal", "esp_mean_kcal",
                   "esp_mean_pos_kcal", "esp_mean_neg_kcal", "esp_sigma2_pos", "esp_sigma2_neg", "esp_sigma2_tot",
                   "esp_nu", "esp_pi", "esp_area_pos_A2", "esp_area_neg_A2",
                   *(f"esp_polar_area_{t}_A2" for t in POLAR_ESP_KJMOL), *esp_bin_keys())


class OpenSurfaceError(ValueError):
    """The isosurface reaches the edge of the grid, so it is not closed (area and volume would be wrong)."""


# ====================================================================== mesh helpers
def triangles(mesh) -> np.ndarray:
    """(M, 3) vertex indices of the triangles of a PolyData (polygons are triangulated first; same points)."""
    if not mesh.is_all_triangles:
        mesh = mesh.triangulate()
    faces = getattr(mesh, "regular_faces", None)
    if faces is not None:
        return np.asarray(faces, int).reshape(-1, 3)
    return np.asarray(mesh.faces, int).reshape(-1, 4)[:, 1:]


def _tri_geometry(mesh) -> tuple[np.ndarray, np.ndarray, float]:
    """Triangle areas, vertex weights (1/3 of each adjacent triangle's area) and the signed enclosed volume."""
    pts = np.asarray(mesh.points, float)
    tri = triangles(mesh)
    if len(tri) == 0:
        return np.zeros(0), np.zeros(len(pts)), 0.0
    a, b, c = pts[tri[:, 0]], pts[tri[:, 1]], pts[tri[:, 2]]
    cross = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(cross, axis=1)
    # divergence theorem; translating to the centroid keeps the sum well conditioned far from the origin
    o = pts.mean(axis=0)
    vol = float(np.einsum("ij,ij->i", a - o, np.cross(b - o, c - o)).sum() / 6.0)
    w = np.bincount(tri.ravel(), weights=np.repeat(area / 3.0, 3), minlength=len(pts))
    return area, w, vol


def vertex_areas(mesh) -> np.ndarray:
    """Area (Å²) carried by each vertex: one third of the area of every triangle that uses it."""
    return _tri_geometry(mesh)[1]


def _finish_surface(surf):
    """Triangulate, merge points, orient outward (positive enclosed volume) and add point normals."""
    if surf.n_points == 0:
        raise ValueError("the isosurface is empty (isovalue outside the data range?)")
    surf = surf.triangulate().clean()
    _, _, vol = _tri_geometry(surf)
    if vol < 0:
        surf = surf.flip_faces() if hasattr(surf, "flip_faces") else surf.flip_normals()
    surf = surf.compute_normals(cell_normals=False, point_normals=True, split_vertices=False,
                                consistent_normals=False, auto_orient_normals=False)
    return surf


def _contour(grid, scalars: str, iso: float):
    import pyvista as pv

    method = "flying_edges" if isinstance(grid, pv.ImageData) else "contour"
    surf = grid.contour([float(iso)], scalars=scalars, method=method, compute_normals=False,
                        compute_scalars=False)
    return _finish_surface(surf)


# ====================================================================== QM density surfaces
def density_surface(density: Cube, iso: float = 0.002, check_closed: bool = True):
    """The ``iso`` (e/bohr³) isosurface of a density cube: PolyData in Å, triangles, outward point normals.

    Raises :class:`OpenSurfaceError` if the isosurface reaches the edge of the grid (``check_closed``)."""
    density = read_any(density)
    if check_closed:
        edge = density.boundary_max()
        if edge >= iso:
            raise OpenSurfaceError(
                f"the {iso:g} au density isosurface reaches the edge of the cube grid (density on the boundary up "
                f"to {edge:.3g} au), so it is not closed and its area and volume would be wrong. Use a larger "
                "grid (Psi4: cubic_grid_overage, e.g. [6.0, 6.0, 6.0] bohr) or a higher isovalue.")
    if float(np.max(density.data)) < iso:
        raise ValueError(f"the density never reaches {iso:g} au (max {np.max(density.data):.3g}): is it a density cube?")
    return _contour(density.to_pyvista(), "values", iso)


def map_esp(surface, esp: Cube) -> np.ndarray:
    """ESP (kcal/mol) at each vertex of ``surface``: trilinear interpolation of a hartree/e cube."""
    esp = read_any(esp)
    v = esp.sample(np.asarray(surface.points)) * HARTREE_KCAL
    bad = ~np.isfinite(v)
    if bad.any():
        raise ValueError(f"{int(bad.sum())} of {len(v)} surface points lie outside the ESP cube's grid")
    return v


def qm_esp_surface(density, esp=None, iso: float = 0.002):
    """``(surface, esp_kcal or None)`` from density (and ESP) cubes or their paths; ESP also stored as point
    data ``"esp_kcal"``."""
    surf = density_surface(read_any(density), iso)
    values = None
    if esp is not None:
        values = map_esp(surf, read_any(esp))
        surf.point_data["esp_kcal"] = values
    return surf, values


# ====================================================================== descriptors
def gipf_descriptors(surface, esp_kcal: np.ndarray | None = None) -> dict:
    """Area, enclosed volume and (with ``esp_kcal`` per vertex) the area-weighted Murray-Politzer statistics.

    Without an ESP only ``surf_area_A2`` and ``surf_volume_A3`` are returned."""
    area, w, vol = _tri_geometry(surface)
    out = {"surf_area_A2": float(area.sum()), "surf_volume_A3": float(abs(vol))}
    if esp_kcal is None:
        return out
    v = np.asarray(esp_kcal, float).reshape(-1)
    if len(v) != len(w):
        raise ValueError(f"{len(v)} ESP values for a surface with {len(w)} vertices")
    total = float(w.sum())
    if total <= 0:
        raise ValueError("the surface has no area")
    mean = float(np.dot(w, v) / total)
    pos, neg = v > 0, v < 0

    def stats(mask):
        a = float(w[mask].sum())
        if a <= 0:
            return a, 0.0, 0.0
        m = float(np.dot(w[mask], v[mask]) / a)
        return a, m, float(np.dot(w[mask], (v[mask] - m) ** 2) / a)

    a_pos, m_pos, s2_pos = stats(pos)
    a_neg, m_neg, s2_neg = stats(neg)
    s2_tot = s2_pos + s2_neg
    out.update({
        "esp_min_kcal": float(v.min()), "esp_max_kcal": float(v.max()), "esp_mean_kcal": mean,
        "esp_mean_pos_kcal": m_pos, "esp_mean_neg_kcal": m_neg,
        "esp_sigma2_pos": s2_pos, "esp_sigma2_neg": s2_neg, "esp_sigma2_tot": s2_tot,
        "esp_nu": float(s2_pos * s2_neg / s2_tot ** 2) if s2_tot > 0 else 0.0,
        "esp_pi": float(np.dot(w, np.abs(v - mean)) / total),
        "esp_area_pos_A2": a_pos, "esp_area_neg_A2": a_neg,
        **{f"esp_polar_area_{t}_A2": float(w[np.abs(v) > t / KJ_PER_KCAL].sum()) for t in POLAR_ESP_KJMOL},
        **esp_bin_areas(w, v),
    })
    return out


def esp_bin_areas(w: np.ndarray, esp_kcal: np.ndarray, edges=ESP_BINS_KJMOL) -> dict:
    """Surface area (Å²) per interval of V (kJ/mol) for vertex areas ``w``: the descriptors of
    :func:`esp_bin_keys` (intervals [lo, hi), the tails V < first edge and V >= last edge)."""
    v = np.asarray(esp_kcal, float).reshape(-1) * KJ_PER_KCAL
    e = np.asarray(edges, float)
    idx = np.searchsorted(e, v, side="right")          # 0 = below the first edge, len(e) = at or above the last
    areas = np.bincount(idx, weights=np.asarray(w, float).reshape(-1), minlength=len(e) + 1)
    return {k: float(a) for k, a in zip(esp_bin_keys(edges), areas)}


def esp_profile(surface, esp_kcal: np.ndarray, bin_kjmol: float = PROFILE_BIN_KJMOL) -> dict:
    """The distribution of V over ``surface``: area (Å²) per ``bin_kjmol`` interval of V over the molecule's
    own range. ``{"unit": "kJ/mol", "bin_kjmol", "lo_kjmol" (first edge, a multiple of the bin), "area_A2" [...],
    "total_area_A2"}``; interval i is [lo + i·bin, lo + (i + 1)·bin)."""
    w = vertex_areas(surface)
    v = np.asarray(esp_kcal, float).reshape(-1) * KJ_PER_KCAL
    if len(v) != len(w):
        raise ValueError(f"{len(v)} ESP values for a surface with {len(w)} vertices")
    b = float(bin_kjmol)
    if b <= 0:
        raise ValueError("bin_kjmol must be positive")
    lo = float(np.floor(v.min() / b) * b) if len(v) else 0.0
    idx = np.floor((v - lo) / b).astype(int) if len(v) else np.zeros(0, int)
    areas = np.bincount(idx, weights=w, minlength=int(idx.max()) + 1 if len(idx) else 0)
    return {"unit": "kJ/mol", "bin_kjmol": b, "lo_kjmol": lo, "area_A2": [round(float(a), 6) for a in areas],
            "total_area_A2": float(w.sum())}


def profile_edges(profile: dict) -> np.ndarray:
    """The interval edges (kJ/mol) of an :func:`esp_profile`, one more than its areas."""
    n = len(profile["area_A2"])
    return profile["lo_kjmol"] + profile["bin_kjmol"] * np.arange(n + 1)


def profile_area(profile: dict, lo_kjmol: float | None = None, hi_kjmol: float | None = None) -> float:
    """Area (Å²) of the profile's intervals that lie within [lo, hi) (None = open). Exact when lo and hi are
    interval edges (multiples of the profile's bin); otherwise an interval counts when its centre is inside."""
    e = profile_edges(profile)
    mid = 0.5 * (e[:-1] + e[1:])
    a = np.asarray(profile["area_A2"], float)
    keep = np.ones(len(a), bool)
    if lo_kjmol is not None:
        keep &= mid >= lo_kjmol
    if hi_kjmol is not None:
        keep &= mid < hi_kjmol
    return float(a[keep].sum())


def profile_polar_area(profile: dict, t_kjmol: float) -> float:
    """Area (Å²) where |V| > t (kJ/mol), from a saved profile (for t a multiple of its bin this is the polar area
    of the surface up to the vertices that lie exactly on ±t)."""
    return profile_area(profile, None, -t_kjmol) + profile_area(profile, t_kjmol, None)


def profile_csv(profile: dict) -> str:
    """The profile as CSV text: lo_kjmol, hi_kjmol, area_A2."""
    e = profile_edges(profile)
    rows = ["lo_kjmol,hi_kjmol,area_A2"]
    rows += [f"{e[i]:g},{e[i + 1]:g},{a:.6g}" for i, a in enumerate(profile["area_A2"])]
    return "\n".join(rows) + "\n"


def surface_descriptors(density, esp=None, iso: float = 0.002, *, profile: bool = False) -> dict:
    """GIPF descriptors of the ``iso`` density isosurface, ESP from ``esp`` (cubes or cube paths).

    Without an ESP cube only the area and volume are returned (see :func:`gipf_descriptors`). ``profile``: also
    the ESP distribution of :func:`esp_profile` under ``"esp_profile"`` (a dict, not a number)."""
    surf, values = qm_esp_surface(density, esp, iso)
    out = gipf_descriptors(surf, values)
    if profile and values is not None:
        out["esp_profile"] = esp_profile(surf, values)
    return out


# ====================================================================== charge-based fallback
def vdw_radii_A(numbers) -> np.ndarray:
    """Bondi van der Waals radii (Å), Mantina et al. where Bondi has none, 2.0 Å for anything else
    (``radii.radii(numbers, "bondi")``)."""
    from .radii import radii

    return radii(np.asarray(numbers, int).reshape(-1), "bondi")


def point_charge_esp(points_A, positions_A, charges, chunk: int = 4096) -> np.ndarray:
    """Coulomb potential (kcal/mol per unit positive charge) of point charges (e) at ``points_A``."""
    pts = np.asarray(points_A, float).reshape(-1, 3)
    pos = np.asarray(positions_A, float).reshape(-1, 3)
    q = np.asarray(charges, float).reshape(-1)
    if len(q) != len(pos):
        raise ValueError(f"{len(q)} charges for {len(pos)} atoms")
    out = np.empty(len(pts))
    for s in range(0, len(pts), chunk):
        d = np.linalg.norm(pts[s:s + chunk, None, :] - pos[None, :, :], axis=2)
        out[s:s + chunk] = (q / np.maximum(d, 1e-6)).sum(axis=1)
    return COULOMB_KCAL * out


def sphere_union_surface(positions_A, radii_A, spacing: float = 0.25, smoothness: float = SMOOTH_K):
    """Closed surface of a smooth union of spheres (PolyData in Å, outward normals)."""
    import pyvista as pv

    pos = np.asarray(positions_A, float).reshape(-1, 3)
    rad = np.asarray(radii_A, float).reshape(-1)
    if len(pos) == 0:
        raise ValueError("no atoms")
    h = float(spacing)
    k = float(smoothness)
    cut = 15.0 / k                                   # contributions below exp(-15) ~ 3e-7 are dropped
    pad = 3 * h
    lo = pos.min(axis=0) - rad.max() - pad
    hi = pos.max(axis=0) + rad.max() + pad
    n = np.ceil((hi - lo) / h).astype(int) + 1
    axes = [lo[d] + h * np.arange(n[d]) for d in range(3)]
    s = np.full(tuple(n), np.exp(-k * cut))
    for p, r in zip(pos, rad):
        reach = r + cut
        sl = []
        for d in range(3):
            i0 = max(int(np.floor((p[d] - reach - lo[d]) / h)), 0)
            i1 = min(int(np.ceil((p[d] + reach - lo[d]) / h)) + 1, n[d])
            sl.append(slice(i0, i1))
        dx = (axes[0][sl[0]] - p[0])[:, None, None]
        dy = (axes[1][sl[1]] - p[1])[None, :, None]
        dz = (axes[2][sl[2]] - p[2])[None, None, :]
        dist = np.sqrt(dx * dx + dy * dy + dz * dz)
        s[sl[0], sl[1], sl[2]] += np.exp(np.minimum(-k * (dist - r), 700.0))
    field = np.log(s) / k                            # > 0 inside, ~ distance below the surface
    grid = pv.ImageData(dimensions=tuple(int(x) for x in n), spacing=(h, h, h), origin=tuple(lo))
    grid.point_data["field"] = field.ravel(order="F")
    return _contour(grid, "field", 0.0)


def charge_esp_surface(numbers, positions_A, charges, kind: str = "vdw", scale: float = 1.0,
                       probe: float = PROBE_A, spacing: float = 0.25):
    """``(surface, esp_kcal)``: a vdW (``kind="vdw"``, radii x ``scale``) or solvent-accessible (``"sas"``,
    radii x ``scale`` + ``probe``) surface with the point-charge ESP (kcal/mol) at each vertex, also stored as
    point data ``"esp_kcal"``."""
    kind = kind.lower()
    if kind not in ("vdw", "sas"):
        raise ValueError("kind must be 'vdw' or 'sas'")
    radii = vdw_radii_A(numbers) * float(scale) + (float(probe) if kind == "sas" else 0.0)
    surf = sphere_union_surface(positions_A, radii, spacing)
    esp = point_charge_esp(surf.points, positions_A, charges)
    surf.point_data["esp_kcal"] = esp
    return surf, esp
