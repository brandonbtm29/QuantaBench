"""Long-range settings for liquid slabs: the options, their slab-safe defaults, and a recommendation (MD-LR).

The choices a liquid-slab setup (``md.liquid``) makes about the interactions beyond the real-space cutoff, as one
light module (no numpy, no LAMMPS) that the CLI, ``md.liquid`` and the Simulate panel (SIM-1) all use:

``method``
    ``ewald``: full long-range dispersion. ``lj/long/coul/long long long rc`` + ``kspace_style pppm/disp``: r⁻⁶ is
    Ewald-summed like the charges (in 't Veld, Ismail & Grest 2007; Isele-Holder et al. 2012), so nothing is cut
    off and no tail correction is needed. CPU only: LAMMPS' GPU package has no ``lj/long/coul/long`` or
    ``pppm/disp`` (their accelerated versions are OPENMP, INTEL and OPT).
    ``cutoff``: the LJ truncated at rc + the analytic tail. ``lj/cut/coul/long rc`` + ``kspace_style pppm``; the bulk
    NPT has LAMMPS' homogeneous tail (``pair_modify tail yes``), the slab none (it assumes a uniform density), and
    ``md surface-tension`` adds the planar-interface tail to σ afterwards (Blokhuis et al. 1995, PR #37). The GPU
    package runs it.
``cutoff`` (the "screening length")
    The LJ (and real-space Coulomb) cutoff rc in Å, 8-30. Defaults: 12 for ``ewald``, 16 for ``cutoff``.
Coulomb accuracy (``kspace_accuracy``)
    PPPM's relative force accuracy. Defaults: 1e-6 for ``ewald`` (converged), 1e-5 for ``cutoff`` (pilot B's, which
    reproduced Tsige & Grest; PPPM runs on the CPU threads of a GPU run and sets its speed).
Dispersion k-space (``ewald`` only)
    ``disp_grid = "spacing"`` (default): ``kspace_modify gewald/disp G`` with G = ``DISP_G_RC`` / rc, and
    ``mesh/disp`` computed in the input from the box so that no spacing exceeds h = ``DISP_H_G`` / G (the inputs
    redo it after the slab's box is opened). ``disp_grid = "accuracy"``: LAMMPS picks G and the mesh from
    ``force/disp/real`` and ``force/disp/kspace`` (defaults 1e-5 and 2e-5, A-fine's).
Coulomb mesh (``cutoff`` only: plain ``pppm``)
    ``coul_grid = "spacing"`` (default): G is LAMMPS' own first estimate for the Coulomb accuracy (``coulomb_gewald``),
    pinned with ``kspace_modify gewald``, and the mesh has one spacing h ≤ ``COUL_H_G`` / G in all three directions,
    computed in the input from the box (again after the slab's box is opened). ``coul_grid = "accuracy"``: LAMMPS
    picks the mesh (the folders written before MD-LR-2). ``pppm/disp`` (``ewald``) already picks an isotropic
    Coulomb mesh.

Why the cutoff runs' Coulomb mesh is set (2026-11-24, MD-LR-2)
--------------------------------------------------------------
Plain ``pppm`` estimates its error per direction for a uniform density, and an edge five times longer tolerates a
coarser spacing: for the pilots' 40 x 40 x 200 Å slab boxes LAMMPS chose meshes about 1.5 times coarser along z than
along x and y (pilot C at 1e-5: 18 x 18 x 60, h·G 0.47 / 0.71; pilot B: 12 x 12 x 45). In a slab, half the box is
vacuum and that error goes into Pzz. Re-evaluating the pilots' 101 production frames (``Runs\\md-pilot-2026-11-20\\
followup_checks_2026-11-23``) showed C's σ 2.37 ± 0.03 mN/m too high and B's 0.30 ± 0.02, and Coulomb 1e-6 left C
0.32 high (h·G 0.54 along z). At C's own G an isotropic mesh at its x/y spacing (18 x 18 x 90, h·G 0.475) gave the
converged value within 0.003 mN/m, B's at h·G 0.52 within 0.004, at no extra cost. ``COUL_H_G`` = 0.47 is that bound.
The rule itself on the same frames (G from the accuracy, ``coulomb_gewald``; ``coulomb_mesh``): at 1e-6, C's
24 x 24 x 108 is within 0.013 ± 0.004 mN/m of converged and B's 16 x 16 x 80 within 0.002 ± 0.004 (LAMMPS' own 1e-6
meshes: 0.32 and 0.025), with fewer mesh points than LAMMPS' 27 x 27 x 90 and 20 x 20 x 72.

Why the defaults changed (2026-11-23)
-------------------------------------
QuantaBench used LAMMPS Howto_dispersion's accuracies (``force/disp/real 1e-4``, ``force/disp/kspace 2e-3``, Coulomb
1e-5) until MD-LR. For pilot A's C6F14 slab (40.3 x 40.3 x 200.9 Å, half of it vacuum) LAMMPS turned those into G 0.171
1/Å and a **2 x 2 x 8** dispersion mesh: its error estimate assumes a uniform density, and the vacuum dilutes it.
Re-evaluating A's 101 production frames with finer settings (``Runs\\md-pilot-2026-11-20\\mesh_check_2026-11-22``)
showed A's σ (10.70 mN/m) 1.36 ± 0.03 mN/m too low for numerical reasons alone: the dispersion mesh −1.19, the
real-space truncation of r⁻⁶ −0.30 (at G·rc = 2.05, 21 % of r⁻⁶ is left at rc and dropped beyond it), Coulomb PPPM
at 1e-5 +0.15. Converged within 0.02-0.03 mN/m: ``pppm/disp 1e-6``, ``force/disp/real 1e-5``,
``force/disp/kspace 2e-5`` (G 0.2305 1/Å, mesh 18 x 18 x 90, h 2.24 Å), the settings of the A-fine run.

The spacing rule is those settings written so they don't depend on LAMMPS' estimate: G·rc = 2.77 (1.8 % of r⁻⁶ left at
rc; G·rc 3.26 changed σ by only 0.02-0.03 more) and h·G ≤ 0.52, which reproduces A-fine's grid for A's box exactly. In
the mesh check h·G 0.69 (at A's G) and 0.57 (at G 0.28) were still converged and 1.17 lost 0.19 mN/m. The rule is the
default because it is the more robust of the two: it gives the same G in the bulk and the slab (LAMMPS' estimate gave
A-fine 0.2375 in the bulk and 0.2305 in the slab), the mesh follows the box whatever the vacuum, and the slab's
grid no longer depends on how much of the box is empty.

The recommendation
------------------
``recommend_long_range(geometry, box, n_molecules, n_atoms, machine, purpose)`` returns a method, cutoff, accelerator
and accuracies with a one-line reason and a speed from the pilots (``PILOT_TABLE``). A choice the user makes always
wins; ``differences(...)`` says where it departs from the recommendation, and ``check_settings(...)`` warns about a
choice the pilots showed to be biased. The table was provisional until A-fine had run; since 2026-11-24 it holds
A-fine's measured σ (12.62 ± 0.40 mN/m for C6F14, on Freire 2006's 12.33) and B's and C's Coulomb-corrected values,
and ``PROVISIONAL`` is False.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field

METHODS = ("ewald", "cutoff")
METHOD_LABELS = {
    "ewald": "Full long-range dispersion (pppm/disp + lj/long/coul/long; CPU only)",
    "cutoff": "LJ cutoff + analytic tail (lj/cut/coul/long + pppm; the tail is added to σ; GPU or CPU)",
}
METHOD_ALIASES = {"ewald": "ewald", "full": "ewald", "pppm/disp": "ewald", "long": "ewald",
                  "cutoff": "cutoff", "tail": "cutoff", "cutoff+tail": "cutoff"}
CUTOFF_LABEL = "LJ cutoff (screening length)"
CUTOFF_UNIT = "Å"
CUTOFF_RANGE = (8.0, 30.0)
DEFAULT_CUTOFF = {"ewald": 12.0, "cutoff": 16.0}
#: Coulomb PPPM relative accuracy per method when none is given
COULOMB_ACCURACY = {"ewald": 1.0e-6, "cutoff": 1.0e-5}
PURPOSES = {"screen": "Quick screen (σ to a few %, fastest on this machine)",
            "reference": "Reference value (numerically converged, no tail approximation)"}
GEOMETRIES = ("slab", "bulk")
DISP_GRIDS = {"spacing": "G and mesh from the cutoff and the box (slab-safe)",
              "accuracy": "LAMMPS picks G and the mesh from two accuracies (its estimate assumes a uniform density)"}

#: pppm/disp's spacing rule (see the module docstring): G = DISP_G_RC / rc and mesh spacing h <= DISP_H_G / G
DISP_G_RC = 2.77            # A-fine's slab: G 0.2305 1/Å at rc 12 Å
DISP_H_G = 0.52             # A-fine's slab: 18 x 18 x 90 on 40.33 x 40.33 x 200.9 Å, h 2.24 Å
DISP_ORDER = 5
#: accuracy mode's defaults (kcal/mol/Å; A-fine's) and the old ones (LAMMPS Howto_dispersion, real units)
DISP_REAL_ACCURACY = 1.0e-5
DISP_KSPACE_ACCURACY = 2.0e-5
HOWTO_DISP_ACCURACY = (1.0e-4, 2.0e-3)
#: what the mesh check validated (above these the warnings start)
VALIDATED_HG = 0.6
VALIDATED_GRC = 2.7
#: plain pppm's Coulomb mesh (MD-LR-2): one spacing h <= COUL_H_G / G in every direction, G from the accuracy
COUL_H_G = 0.47             # pilot C's 18 x 18 x 90 at its G (h·G 0.475) was converged; LAMMPS' 18 x 18 x 60 was not
COUL_GRIDS = {"spacing": "an isotropic mesh, h·G ≤ 0.47 in every direction (G from the accuracy; slab-safe)",
              "accuracy": "LAMMPS picks the mesh from the accuracy (about 1.5 times coarser along a slab box's z)"}

PROVISIONAL = False         # True until A-fine had run (2026-11-24)
PROVISIONAL_NOTE = ("no longer provisional: A-fine (pilot A with converged pppm/disp) measured σ 12.62 ± 0.40 mN/m "
                    "on 2026-11-24, confirming the mesh check's estimate on pilot A's frames (12.06 ± 0.70)")


# ------------------------------------------------------------------------------------------- the pilot table
@dataclass(frozen=True)
class PilotRow:
    """One pilot run (or benchmark) the recommendations rest on. σ in mN/m, densities in g/cm³."""

    setup: str                          # what ran
    method: str                         # "ewald" | "cutoff"
    cutoff: float                       # Å
    accelerator: str                    # "cpu" | "gpu"
    threads: int                        # CPU (OpenMP) threads; with a GPU the threads PPPM and the bonds ran on
    ns_per_day: float | None
    stage: str = "slab"                 # "slab" | "bulk"
    atoms: int = 9780
    sigma_raw: float | None = None
    sigma_tail: float | None = None     # with the analytic tail (cutoff runs)
    sigma_err: float | None = None
    plateau_vs_bulk: float | None = None   # slab plateau density relative to the bulk NPT density
    bulk_density: float | None = None
    source: str = ""


#: The pilots of 2026-11-20/24 (C6F14, OPLS-AA with the Watkins-Jorgensen types, 489 molecules = 9,780 atoms, 40 x 40 Å
#: slab with 5 nm of vacuum on each side, 298.15 K, 1 fs, 2 ns settle + 10 ns production; the workstation's LAMMPS
#: 4 Jul 2026, Ryzen 9 7900X, RX 9060 XT on OpenCL). Reports: Runs\md-pilot-2026-11-20\pilot_{A,B,C,A_fine}_report.md,
#: pilots_summary.md, mesh_check_2026-11-22\mesh_check.md, followup_checks_2026-11-23\. Speeds are the production
#: runs'. B's and C's σ are corrected for their Coulomb PPPM mesh (MD-LR-2; the runs as analysed in each ``source``).
PILOT_TABLE: dict[str, PilotRow] = {
    "A": PilotRow("pilot A: ewald, LAMMPS Howto_dispersion accuracies (the old default), 2 x 2 x 8 dispersion mesh",
                  "ewald", 12.0, "cpu", 8, 7.6, sigma_raw=10.70, sigma_err=0.70, plateau_vs_bulk=+0.0005,
                  bulk_density=1.6645,
                  source="pilot_A_report.md (production speed); 1.36 ± 0.03 mN/m low numerically (mesh_check.md); "
                         "superseded by A-fine"),
    "A-fine": PilotRow("A-fine: ewald, converged (pppm/disp 1e-6, force/disp/real 1e-5, force/disp/kspace 2e-5; "
                       "= the spacing rule)", "ewald", 12.0, "cpu", 8, 5.85, sigma_raw=12.62, sigma_err=0.40,
                       plateau_vs_bulk=-0.0044, bulk_density=1.6716,
                       source="pilot_A_fine_report.md (2026-11-24): σ from 10 blocks of 1 ns (± 0.65 allowing for "
                              "their correlation: a step at 7 ns in the tangential pressures); it confirms pilot A + "
                              "the mesh check's offset (12.06 ± 0.70) within 0.7 σ; production speed on the shared "
                              "workstation (its idle 2,000-step benchmark: 6.6 ns/day)"),
    "B": PilotRow("pilot B: 16 Å cutoff + tail (Tsige & Grest's setup), GPU + CPU", "cutoff", 16.0, "gpu", 6, 21.0,
                  sigma_raw=9.30, sigma_tail=11.41, sigma_err=0.30, plateau_vs_bulk=-0.020, bulk_density=1.6719,
                  source="pilot_B_report.md: slab 3-12 ns on the GPU (pair style) + 6 OpenMP threads (PPPM, bonds); "
                         "σ as run 9.60 raw, 11.71 + tail, 0.30 too high from its Coulomb PPPM mesh "
                         "(followup_checks_2026-11-23)"),
    "B-cpu": PilotRow("pilot B's first 3 ns: 16 Å cutoff on the CPU", "cutoff", 16.0, "cpu", 4, 3.9,
                      source="pilot_B_report.md: slab 0-3 ns, -sf omp -pk omp 4"),
    "C": PilotRow("pilot C: 12 Å cutoff + tail, GPU + CPU", "cutoff", 12.0, "gpu", 4, 16.1, sigma_raw=5.60,
                  sigma_tail=8.76, sigma_err=0.32, plateau_vs_bulk=-0.047, bulk_density=1.6714,
                  source="pilot_C_report.md (tail from PR #37's analysis); σ as run 7.96 raw, 11.12 + tail, 2.37 too "
                         "high from its Coulomb PPPM mesh (followup_checks_2026-11-23)"),
    "C-bulk": PilotRow("pilot C's bulk NPT: 12 Å cutoff + homogeneous tail, GPU + CPU", "cutoff", 12.0, "gpu", 4,
                       13.4, stage="bulk", bulk_density=1.6714, source="pilot_C_report.md"),
    "A-fine-bulk": PilotRow("A-fine's bulk NPT: ewald, converged", "ewald", 12.0, "cpu", 8, 5.58, stage="bulk",
                            bulk_density=1.6716,
                            source="A-fine's log.bulk: 1 ns NPT, density over its last 0.5 ns ± 0.0022 (as the "
                                   "analysis reports it); the workstation was shared with test runs at the time"),
}
#: published and measured σ of C6F14 at 298 K the table is read against (mN/m)
REFERENCES = {"Tsige & Grest 2008 (OPLS-AA W&J, 16 Å + tail)": 11.25,
              "Freire et al. 2006 (experiment; the target)": 12.33,
              "Freire et al. 2006 (the measured point at 298.15 K, ± 0.14)": 12.23}


def pilot_gap(row_id: str, ref_id: str = "A-fine") -> float:
    """σ of a pilot (with the tail for a cutoff run) relative to another's, e.g. pilot B + tail against A-fine:
    −0.096 (16 Å + tail reads 9.6 % low for C6F14)."""
    a, b = PILOT_TABLE[row_id], PILOT_TABLE[ref_id]
    sa = a.sigma_tail if a.sigma_tail is not None else a.sigma_raw
    sb = b.sigma_tail if b.sigma_tail is not None else b.sigma_raw
    if sa is None or sb is None:
        raise ValueError(f"pilot {row_id!r} or {ref_id!r} has no σ")
    return sa / sb - 1.0


# ------------------------------------------------------------------------------------------- names and rules
def normalize_method(name: str) -> str:
    m = METHOD_ALIASES.get(str(name).strip().lower())
    if m is None:
        raise ValueError(f"long-range method {name!r}: choose {' or '.join(METHODS)} "
                         f"({'; '.join(f'{k}: {v}' for k, v in METHOD_LABELS.items())})")
    return m


def check_cutoff(rc: float) -> float:
    rc = float(rc)
    lo, hi = CUTOFF_RANGE
    if not (lo <= rc <= hi) or not math.isfinite(rc):
        raise ValueError(f"{CUTOFF_LABEL} {rc:g} {CUTOFF_UNIT}: choose {lo:g}-{hi:g} {CUTOFF_UNIT}")
    return rc


def dispersion_name(method: str, cutoff: float | None = None) -> str:
    """The folder's ``dispersion`` label: ``"ewald"`` or ``"cutoff16"`` (``md.liquid.parse_dispersion``)."""
    m = normalize_method(method)
    if m == "ewald":
        return "ewald"
    return f"cutoff{check_cutoff(cutoff if cutoff is not None else DEFAULT_CUTOFF['cutoff']):g}"


def method_of(dispersion: str) -> str:
    return "ewald" if str(dispersion).strip().lower() == "ewald" else "cutoff"


def rule_gewald(cutoff: float) -> float:
    """The spacing rule's pppm/disp G (1/Å) for a real-space cutoff (Å): ``DISP_G_RC`` / rc (0.2308 at 12 Å)."""
    return DISP_G_RC / check_cutoff(cutoff)


def rule_spacing(gewald: float) -> float:
    """The spacing rule's largest dispersion-mesh spacing (Å) for G (1/Å): ``DISP_H_G`` / G (2.25 Å at G 0.2308)."""
    if not gewald or gewald <= 0:
        raise ValueError("gewald/disp must be positive")
    return DISP_H_G / float(gewald)


def _smooth(n: int) -> int:
    """The smallest m >= n whose prime factors are 2, 3 and 5 (LAMMPS' PPPM rounds its meshes up that way)."""
    m = max(2, int(n))
    while True:
        k = m
        for p in (2, 3, 5):
            while k % p == 0:
                k //= p
        if k == 1:
            return m
        m += 1


def disp_mesh(box: Sequence[float], spacing: float) -> tuple[int, int, int]:
    """The dispersion mesh LAMMPS ends up with for a box (Å): the inputs ask for ceil(L / h) per direction
    (``mesh/disp $(ceil(lx/v_disp_h))``) and pppm/disp rounds each up to a product of 2, 3 and 5 (53 -> 54)."""
    if not spacing or spacing <= 0:
        raise ValueError("the mesh spacing must be positive")
    return tuple(_smooth(math.ceil(float(x) / float(spacing) - 1e-9)) for x in box)   # type: ignore[return-value]


def real_space_left(g_rc: float) -> float:
    """Fraction of r⁻⁶ the real-space part still carries at rc (and drops beyond it) for G·rc:
    exp(-x²)(1 + x² + x⁴/2). 0.21 at pilot A's 2.05, 0.018 at 2.77."""
    x2 = float(g_rc) ** 2
    return math.exp(-x2) * (1.0 + x2 + 0.5 * x2 * x2)


def coulomb_gewald(accuracy: float, cutoff: float, n_atoms: int, volume: float, sum_q2: float) -> float:
    """Plain pppm's G (1/Å) for a relative Coulomb ``accuracy``: LAMMPS' own first estimate (``PPPM::set_grid_global``,
    the real-space error at rc equal to the accuracy), G = sqrt(−ln(acc·√(N·rc·V) / (2 Σq²))) / rc, with ``sum_q2``
    the sum of the squared charges (e²) and ``volume`` the box's (Å³). LAMMPS then adjusts G a little for its own mesh
    (pilot C's slab, 1e-5: 0.2014 here, 0.2115 in its log); the inputs pin this one so that h·G is what the rule says.
    The inputs compute it from the box they run in (``md.liquid.coul_mesh_lines``); this is the same in Python."""
    rc = check_cutoff(cutoff)
    if not (0.0 < float(accuracy) < 1.0e-3):
        raise ValueError(f"Coulomb accuracy {accuracy:g}: the mesh rule needs a relative accuracy below 1e-3")
    if sum_q2 <= 0 or n_atoms <= 0 or volume <= 0:
        raise ValueError("coulomb_gewald needs charges, atoms and a box")
    x = float(accuracy) * math.sqrt(n_atoms * rc * volume) / (2.0 * float(sum_q2))
    if x >= 1.0:
        raise ValueError(f"Coulomb accuracy {accuracy:g} is too loose for this system (LAMMPS' estimate breaks down)")
    return math.sqrt(-math.log(x)) / rc


def coulomb_spacing(gewald: float) -> float:
    """The rule's largest Coulomb-mesh spacing (Å) for G (1/Å): ``COUL_H_G`` / G (2.33 Å at pilot C's 1e-5 G)."""
    if not gewald or gewald <= 0:
        raise ValueError("gewald must be positive")
    return COUL_H_G / float(gewald)


def coulomb_mesh(box: Sequence[float], gewald: float) -> tuple[int, int, int]:
    """The isotropic Coulomb mesh the inputs give LAMMPS for a box (Å) and G: ceil(L / h) per direction with
    h = ``COUL_H_G`` / G, rounded up to products of 2, 3 and 5 as PPPM does (pilot C's slab at 1e-6: 24 x 24 x 108)."""
    return disp_mesh(box, coulomb_spacing(gewald))


def settings(method: str, cutoff: float | None = None, *, kspace_accuracy: float | None = None,
             disp_grid: str = "spacing", disp_gewald: float | None = None, disp_spacing: float | None = None,
             disp_real_accuracy: float = DISP_REAL_ACCURACY, disp_kspace_accuracy: float = DISP_KSPACE_ACCURACY,
             coul_grid: str = "spacing") -> dict:
    """Every long-range setting resolved, as ``md.liquid.LiquidSlabConfig`` fields (``config_kwargs``) plus the
    derived ones: what the panel shows and what the inputs get. ``coul_grid`` (``cutoff`` only) appears only when it
    is not the default "spacing"."""
    m = normalize_method(method)
    rc = check_cutoff(cutoff if cutoff is not None else DEFAULT_CUTOFF[m])
    out = {"method": m, "dispersion": dispersion_name(m, rc), "cutoff": rc,
           "kspace_accuracy": float(kspace_accuracy or COULOMB_ACCURACY[m])}
    if coul_grid not in COUL_GRIDS:
        raise ValueError(f"coul_grid {coul_grid!r}: choose {', '.join(COUL_GRIDS)}")
    if m == "cutoff" and coul_grid != "spacing":
        out["coul_grid"] = coul_grid
    if m == "ewald":
        if disp_grid not in DISP_GRIDS:
            raise ValueError(f"disp_grid {disp_grid!r}: choose {', '.join(DISP_GRIDS)}")
        out["disp_grid"] = disp_grid
        if disp_grid == "spacing":
            g = float(disp_gewald or rule_gewald(rc))
            out.update(disp_gewald=g, disp_spacing=float(disp_spacing or rule_spacing(g)))
        else:
            out.update(disp_real_accuracy=float(disp_real_accuracy), disp_kspace_accuracy=float(disp_kspace_accuracy))
    return out


def config_kwargs(s: Mapping) -> dict:
    """``LiquidSlabConfig`` keyword arguments from ``settings(...)`` or a recommendation's ``settings``."""
    keys = ("dispersion", "cutoff", "kspace_accuracy", "disp_grid", "disp_gewald", "disp_spacing",
            "disp_real_accuracy", "disp_kspace_accuracy", "coul_grid")
    return {k: s[k] for k in keys if k in s}


def settings_of(cfg) -> dict:
    """``settings(...)`` of a (resolved or not) ``LiquidSlabConfig``-like object."""
    m = method_of(getattr(cfg, "dispersion", "ewald"))
    rc = getattr(cfg, "cutoff", None)
    if rc is None:
        disp = str(getattr(cfg, "dispersion", "ewald"))
        rc = float(disp[6:]) if m == "cutoff" and disp[6:] else DEFAULT_CUTOFF[m]
    return settings(m, rc, kspace_accuracy=getattr(cfg, "kspace_accuracy", None),
                    disp_grid=getattr(cfg, "disp_grid", "spacing") or "spacing",
                    disp_gewald=getattr(cfg, "disp_gewald", None), disp_spacing=getattr(cfg, "disp_spacing", None),
                    disp_real_accuracy=getattr(cfg, "disp_real_accuracy", DISP_REAL_ACCURACY),
                    disp_kspace_accuracy=getattr(cfg, "disp_kspace_accuracy", DISP_KSPACE_ACCURACY),
                    coul_grid=getattr(cfg, "coul_grid", "spacing") or "spacing")


def describe(s: Mapping, box: Sequence[float] | None = None, coul_gewald: float | None = None) -> str:
    """One line: 'full long-range dispersion: lj/long/coul/long 12 Å + pppm/disp, Coulomb 1e-06, dispersion G 0.2308
    1/Å, mesh spacing ≤ 2.25 Å (18 x 18 x 90 for this box)'. For a cutoff run with ``box`` and ``coul_gewald``
    (``coulomb_gewald``) the Coulomb mesh for that box too."""
    rc = float(s["cutoff"])
    if s["method"] == "cutoff":
        if s.get("coul_grid", "spacing") == "spacing":
            mesh = f", isotropic mesh h·G ≤ {COUL_H_G:g} (G from the accuracy"
            if box and coul_gewald:
                n = coulomb_mesh(box, coul_gewald)
                mesh += (f": {coul_gewald:.4g} 1/Å, {' x '.join(str(k) for k in n)} for a "
                         f"{' x '.join(f'{x:.1f}' for x in box)} Å box")
            mesh += ")"
        else:
            mesh = ", mesh chosen by LAMMPS (coarser along a slab box's z)"
        return (f"{rc:g} Å LJ cutoff + tail: lj/cut/coul/long {rc:g} + pppm, Coulomb {s['kspace_accuracy']:g}{mesh} "
                "(bulk NPT with LAMMPS' homogeneous tail; the planar tail is added to σ in the analysis)")
    head = f"full long-range dispersion: lj/long/coul/long {rc:g} Å + pppm/disp, Coulomb {s['kspace_accuracy']:g}"
    if s.get("disp_grid") == "accuracy":
        return (f"{head}, dispersion force/disp/real {s['disp_real_accuracy']:g} force/disp/kspace "
                f"{s['disp_kspace_accuracy']:g} (LAMMPS picks G and the mesh)")
    txt = f"{head}, dispersion G {s['disp_gewald']:.4g} 1/Å, mesh spacing ≤ {s['disp_spacing']:.3g} Å"
    if box:
        txt += f" ({' x '.join(str(n) for n in disp_mesh(box, s['disp_spacing']))} for a {' x '.join(f'{x:.1f}' for x in box)} Å box)"
    return txt


def method_text(method: str, cutoff: float, accelerator: str = "") -> str:
    where = {"gpu": " on the GPU", "cpu": " on the CPU"}.get(accelerator, "")
    if method == "ewald":
        return f"full long-range dispersion (pppm/disp, {cutoff:g} Å real space){where}"
    return f"{cutoff:g} Å LJ cutoff + tail correction{where}"


# ------------------------------------------------------------------------------------------- the machine
@dataclass
class MachineInfo:
    """What the recommendation needs to know about the machine that will run the folder. ``benchmark``: that
    machine's MD benchmark (``MachineProfile.md_throughput``, ``md.bench``), whose measured ns/day replaces the pilot
    table's speeds; ``detect_machine`` fills it in for this computer, a stated machine (``--gpu yes|no``) has none."""

    gpu: bool = False                   # LAMMPS there has the GPU package and found a GPU it can use
    note: str = ""                      # how that was decided
    threads: int | None = None
    benchmark: dict = field(default_factory=dict, repr=False, compare=False)

    def to_dict(self) -> dict:
        d = asdict(self)
        md = d.pop("benchmark") or {}
        if md:
            from . import bench

            d["benchmark"] = {"date": md.get("date", ""), "lammps": md.get("lammps", ""),
                              "variants": sorted(bench.rated(md))}
        return d


def detect_machine(refresh: bool = False) -> MachineInfo:
    """This computer: does its LAMMPS (``md.bench.find_lammps``: ``lmp -h``) have the GPU package with a GPU, and
    did a GPU variant of this machine's MD benchmark (if any) pass the CPU/GPU pressure check?"""
    from . import bench

    md = _this_benchmark()
    info, why = bench.find_lammps(refresh)
    if info is None:
        return MachineInfo(False, f"{why}; recommending for the CPU", benchmark=md)
    ok, why = bench.can_run(info, bench.Variant("gpu", "cutoff16"))
    if not ok:
        return MachineInfo(False, f"LAMMPS {info.version or info.path}: {why}", benchmark=md)
    api = info.accelerators.get("GPU", {}).get("api", "")
    note = f"LAMMPS {info.version} has the GPU package{f' ({api})' if api else ''} and found a GPU"
    if _gpu_check_failed():
        return MachineInfo(False, f"{note}, but its GPU variants failed the CPU/GPU pressure check of "
                                  "`machines --bench md`; recommending for the CPU", benchmark=md)
    return MachineInfo(True, note, benchmark=md)


def _this_benchmark() -> dict:
    """This machine's MD benchmark from the registry (``md_throughput``), {} when it has none."""
    try:
        from ..dist import profile as profile_mod

        me = profile_mod.machine_name()
        md = next((p.md_throughput for p in profile_mod.load_profiles() if p.name == me), None) or {}
    except Exception:  # noqa: BLE001 - no registry: the pilot table stands
        return {}
    return dict(md) if isinstance(md, dict) else {}


def _gpu_check_failed() -> bool:
    """True when this machine's MD benchmark has GPU variants and none of them passed the CPU/GPU check."""
    from . import bench

    md = _this_benchmark()
    recs = [r for k, r in (md.get("variants") or {}).items() if k.startswith("gpu-") and isinstance(r, dict)
            and r.get("check")]
    return bool(recs) and not any(bench.verified(r) for r in recs)


def machine_for(answer: str | bool | None) -> MachineInfo:
    """``--gpu yes|no|auto``: a stated machine, or this computer's (``detect_machine``)."""
    if isinstance(answer, bool):
        return MachineInfo(answer, "as stated")
    a = str(answer or "auto").strip().lower()
    if a in ("yes", "y", "true", "1", "gpu"):
        return MachineInfo(True, "stated: LAMMPS can use a GPU on the machine that runs it")
    if a in ("no", "n", "false", "0", "cpu"):
        return MachineInfo(False, "stated: CPU only")
    if a != "auto":
        raise ValueError(f"--gpu {answer!r}: yes, no or auto")
    return detect_machine()


# ------------------------------------------------------------------------------------------- the recommendation
@dataclass
class LongRangeRecommendation:
    method: str
    cutoff: float
    accelerator: str                    # "gpu" | "cpu"
    settings: dict                      # ``settings(...)``: every long-range value, resolved
    reason: str                         # one line
    ns_per_day: float | None = None     # for this setup's atoms: this machine's MD benchmark, else the pilot table
    speed_basis: str = ""               # where ns_per_day comes from; starts with its source
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    inputs: dict = field(default_factory=dict)
    provisional: bool = PROVISIONAL
    speed_source: str = ""              # "benchmark" (this machine's, md.bench), "pilot table" or "" (no speed)

    @property
    def dispersion(self) -> str:
        return self.settings["dispersion"]

    def config_kwargs(self) -> dict:
        return config_kwargs(self.settings)

    def headline(self) -> str:
        return method_text(self.method, self.cutoff, self.accelerator)

    def summary(self) -> str:
        """'Recommended for this setup: 16 Å LJ cutoff + tail correction on the GPU because ...'"""
        speed = f"; about {self.ns_per_day:.3g} ns/day ({self.speed_basis})" if self.ns_per_day else ""
        return (f"Recommended for this setup: {self.headline()} because {self.reason}{speed}"
                + (" [provisional]" if self.provisional else ""))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["dispersion"] = self.dispersion
        d["summary"] = self.summary()
        if self.provisional:
            d["provisional_note"] = PROVISIONAL_NOTE
        return d


#: atoms per molecule the speeds assume when only the number of molecules is given (C6F14, the pilots' liquid)
PILOT_ATOMS_PER_MOLECULE = 20


def _speed(row_id: str, n_atoms: int | None, n_molecules: int | None) -> tuple[float | None, str]:
    """(ns/day for this many atoms, its basis) from a ``PILOT_TABLE`` row: scaled by the atom ratio, as
    ``md.bench.estimate_hours`` does (PPPM MD costs about linearly in the atoms at a fixed density)."""
    row = PILOT_TABLE[row_id]
    if not row.ns_per_day:
        return None, ""
    where = f"GPU + {row.threads} CPU threads" if row.accelerator == "gpu" else f"{row.threads} CPU threads"
    basis = f"pilot table: pilot {row_id}, {where}"
    scale = 1.0
    if n_atoms:
        scale = row.atoms / float(n_atoms)
        if abs(scale - 1.0) > 0.005:
            basis += f", scaled from {row.atoms:,} to {n_atoms:,} atoms"
    elif n_molecules:
        scale = 489.0 / float(n_molecules)
        if abs(scale - 1.0) > 0.005:
            basis += f", scaled from 489 to {n_molecules:,} molecules of C6F14's size"
    return row.ns_per_day * scale, basis


def _benchmark_speed(md: Mapping | None, method: str, cutoff: float, accelerator: str, geometry: str,
                     n_atoms: int | None, n_molecules: int | None, box: Sequence[float] | None
                     ) -> tuple[float | None, str]:
    """(ns/day for this setup, its basis) from a machine's MD benchmark (``md.bench``): the record of the same
    accelerator and dispersion (a cutoff not measured is scaled from the nearest one), preferring the setup's geometry,
    scaled to its atoms by ``md.bench.scaled_rate`` (and its box). Unverified GPU records are not used. (None, "")
    when the benchmark has nothing that fits."""
    if not md:
        return None, ""
    from . import bench

    atoms = n_atoms or (n_molecules * PILOT_ATOMS_PER_MOLECULE if n_molecules else None)
    dispersion = dispersion_name(method, cutoff)
    # an ewald record from before MD-LR (no ``long_range``) timed the old, coarser pppm/disp: not this setup's
    usable = {k: r for k, r in bench.rated(md).items()
              if not (bench.split_key(k)[0].endswith("-ewald") and not r.get("long_range"))}
    probe = atoms or max((int(r.get("atoms") or 0) for r in usable.values()), default=0)
    if probe <= 0:
        return None, ""
    e = bench.best_estimate({"variants": usable}, probe, 1.0, dispersion=dispersion, accelerator=accelerator,
                            geometry=geometry, box=box if atoms else None)
    if e is None:
        return None, ""
    rec = (md.get("variants") or {}).get(e.key) or {}
    threads = int(rec.get("threads") or 1)
    where = f"GPU + {threads} CPU threads" if accelerator == "gpu" else f"{threads} CPU thread{'s' * (threads != 1)}"
    when = str(rec.get("date") or md.get("date") or "")[:10]
    basis = (f"this machine's MD benchmark: {e.key} {bench.fmt_ns(e.measured_ns_per_day)} ns/day on "
             f"{e.measured_atoms:,} atoms, {where}" + (f", {when}" if when else ""))
    if atoms and atoms != e.measured_atoms:
        basis += (f", scaled to {atoms:,} atoms" if n_atoms else
                  f", scaled to {n_molecules:,} molecules of C6F14's size")
    if e.note:
        basis += f"; {e.note}"
    return e.ns_per_day, basis


def _setup_speed(row_id: str | None, machine: MachineInfo, method: str, cutoff: float, accelerator: str,
                 geometry: str, n_atoms: int | None, n_molecules: int | None, box: Sequence[float] | None
                 ) -> tuple[float | None, str, str]:
    """(ns/day, basis, source): this machine's MD benchmark when it has a fitting record, else the pilot table's row
    ``row_id`` (None: no row fits)."""
    v, basis = _benchmark_speed(machine.benchmark, method, cutoff, accelerator, geometry, n_atoms, n_molecules, box)
    if v:
        return v, basis, "benchmark"
    if row_id is None:
        return None, "", ""
    v, basis = _speed(row_id, n_atoms, n_molecules)
    return v, basis, ("pilot table" if v else "")


def recommend_long_range(geometry: str = "slab", box: Sequence[float] | None = None, n_molecules: int | None = None,
                         n_atoms: int | None = None, machine: MachineInfo | bool | None = None,
                         purpose: str = "screen") -> LongRangeRecommendation:
    """The long-range method, cutoff, accelerator and accuracies for a setup, with a one-line reason and a speed.

    ``geometry``: "slab" (a liquid-vapour interface; "interface" works too) or "bulk". ``box``: (lx, ly, lz) in Å (the
    slab's box with its vacuum); its lateral edges limit the cutoff (``md.liquid.plan_box`` needs > 2 rc + 1).
    ``n_atoms`` (else ``n_molecules``) scales the speed. ``machine``: ``MachineInfo``, a bool (GPU usable) or None
    (CPU). ``purpose``: "screen" or "reference".

    The speed (``ns_per_day``, ``speed_basis``, ``speed_source``) is this machine's MD benchmark when
    ``machine.benchmark`` has a record of the recommended accelerator and dispersion (``detect_machine`` fills it in;
    ``md.bench.scaled_rate``: ns/day x measured atoms / atoms, the k-space share by the box volume when ``box`` is
    given), preferring the setup's geometry; else the pilot table's. The summary line names the source.

    Rules (``PILOT_TABLE``):

    * reference → full long-range dispersion, converged settings, on the CPU (no tail approximation, the slab at the
      bulk density);
    * slab, GPU usable, lateral box > 33 Å → 16 Å cutoff + tail on the GPU (pilot B reproduced Tsige & Grest; for C6F14
      it reads 9.6 % below A-fine's converged σ);
    * slab, CPU only (or a box too narrow for 16 Å) → full long-range dispersion (a cutoff's speed comes from the GPU);
    * bulk → 12 Å cutoff with LAMMPS' homogeneous tail (valid in a uniform liquid), on the GPU when there is one.
    """
    geo = {"slab": "slab", "interface": "slab", "bulk": "bulk"}.get(str(geometry).strip().lower())
    if geo is None:
        raise ValueError(f"geometry {geometry!r}: slab (interface) or bulk")
    if purpose not in PURPOSES:
        raise ValueError(f"purpose {purpose!r}: choose {', '.join(PURPOSES)}")
    if isinstance(machine, bool) or machine is None:
        machine = MachineInfo(bool(machine), "as given" if machine is not None else "not given: the CPU")
    gpu = bool(machine.gpu)
    lateral = min(float(box[0]), float(box[1])) if box else None
    max_rc = (lateral - 1.0) / 2.0 if lateral else math.inf
    inputs = {"geometry": geo, "box": [float(x) for x in box] if box else None, "n_molecules": n_molecules,
              "n_atoms": n_atoms, "machine": machine.to_dict(), "purpose": purpose}
    warnings: list[str] = []
    notes: list[str] = []

    def ewald(reason: str, row: str = "A-fine") -> LongRangeRecommendation:
        rc = DEFAULT_CUTOFF["ewald"]
        if rc > max_rc:
            rc = max(CUTOFF_RANGE[0], math.floor(max_rc))
            warnings.append(f"the lateral box ({lateral:.1f} Å) allows at most a {max_rc:.1f} Å cutoff; pppm/disp's "
                            f"real-space cutoff is {rc:g} Å (G and the mesh follow it)")
        s = settings("ewald", rc)
        notes.extend(notes_for(s, geo, machine))
        v, basis, src = _setup_speed(row, machine, "ewald", rc, "cpu", geo, n_atoms, n_molecules, box)
        return LongRangeRecommendation("ewald", rc, "cpu", s, reason, v, basis, warnings, notes, inputs,
                                       speed_source=src)

    if purpose == "reference":
        rec = ewald("a reference value needs converged full long-range dispersion: no tail approximation, the slab "
                    "stays within 0.5 % of the bulk density (A-fine: −0.44 %; 16 Å + tail: −2.0 %), and these "
                    "settings are converged to ~0.03 mN/m (mesh check; A-fine gave C6F14 12.62 ± 0.40 mN/m, on "
                    "Freire 2006's 12.33)", "A-fine-bulk" if geo == "bulk" else "A-fine")
        return rec
    if geo == "bulk":
        rc = 12.0 if max_rc >= 12.0 else max(CUTOFF_RANGE[0], math.floor(max_rc))
        s = settings("cutoff", rc)
        accel = "gpu" if gpu else "cpu"
        v, basis, src = _setup_speed("C-bulk" if gpu else None, machine, "cutoff", rc, accel, geo, n_atoms,
                                     n_molecules, box)
        if not v:
            basis = ("no pilot ran a CPU-only cutoff bulk, and this machine has no MD benchmark of it "
                     "(quantabench-cli machines --bench md --md-size small)")
        return LongRangeRecommendation(
            "cutoff", rc, accel, s,
            "a uniform liquid needs no interface treatment: a 12 Å cutoff with LAMMPS' homogeneous tail gave the same "
            "bulk density as converged pppm/disp (pilots C and B 1.6714 and 1.6719, A-fine 1.6716 g/cm³)"
            + (", and the GPU runs it" if gpu else ""), v, basis or "", warnings, notes, inputs, speed_source=src)
    if gpu and max_rc >= 16.0:
        s = settings("cutoff", 16.0)
        v, basis, src = _setup_speed("B", machine, "cutoff", 16.0, "gpu", geo, n_atoms, n_molecules, box)
        notes.extend(notes_for(s, geo, machine))
        if src == "benchmark":
            ew, _b, ew_src = _setup_speed(None, machine, "ewald", DEFAULT_CUTOFF["ewald"], "cpu", geo, n_atoms,
                                          n_molecules, box)
            if ew_src == "benchmark" and ew:
                notes.append(f"this machine's MD benchmark: {v:.3g} ns/day for this, {ew:.3g} for full long-range "
                             f"dispersion on the CPU ({v / ew:.1f} x)")
        a_fine = PILOT_TABLE["A-fine"]
        return LongRangeRecommendation(
            "cutoff", 16.0, "gpu", s,
            "LAMMPS can use a GPU here, and 16 Å + tail reproduced Tsige & Grest's C6F14 value (pilot B: "
            f"{PILOT_TABLE['B'].sigma_tail:.2f} ± 0.30 vs 11.25 mN/m) at about "
            f"{PILOT_TABLE['B'].ns_per_day / a_fine.ns_per_day:.1f} times the speed of full long-range dispersion on "
            f"the CPU ({a_fine.ns_per_day:g} ns/day on {a_fine.threads} threads); as a screen it reads "
            f"{-100 * pilot_gap('B'):.1f} % below converged full long-range dispersion (A-fine "
            f"{a_fine.sigma_raw:.2f} mN/m)", v, basis, warnings, notes, inputs, speed_source=src)
    if gpu:
        return ewald(f"the lateral box ({lateral:.1f} Å) is too narrow for a 16 Å cutoff (needs more than 33 Å), and a "
                     "shorter cutoff biases a slab (pilot C at 12 Å: plateau 4.7 % below the bulk, σ + tail 31 % "
                     "low), so full long-range dispersion on the CPU")
    b_cpu, a_fine = PILOT_TABLE["B-cpu"], PILOT_TABLE["A-fine"]
    return ewald("LAMMPS has no GPU to use here and a cutoff's speed comes from the GPU (16 Å: "
                 f"{b_cpu.ns_per_day:g} ns/day on {b_cpu.threads} CPU threads; converged pppm/disp: "
                 f"{a_fine.ns_per_day:g} on {a_fine.threads}), while pppm/disp needs no tail correction and keeps the "
                 "slab within 0.5 % of the bulk density")


# ------------------------------------------------------------------------------------------- choice vs recommendation
FIELD_LABELS = {"kspace_accuracy": "Coulomb PPPM accuracy", "disp_grid": "dispersion k-space",
                "disp_gewald": "dispersion G (1/Å)", "disp_spacing": "dispersion mesh spacing (Å)",
                "disp_real_accuracy": "force/disp/real", "disp_kspace_accuracy": "force/disp/kspace",
                "coul_grid": "Coulomb mesh"}


def _fmt(x) -> str:
    return "none" if x is None else (x if isinstance(x, str) else f"{x:.4g}")


def differences(chosen: Mapping, rec: LongRangeRecommendation) -> list[str]:
    """Where the chosen ``settings`` depart from the recommendation: ["method: cutoff (recommended: ewald)", ...]."""
    out = []
    r = rec.settings
    if chosen["method"] != r["method"]:
        out.append(f"method: {chosen['method']} (recommended: {r['method']})")
    if abs(float(chosen["cutoff"]) - float(r["cutoff"])) > 1e-9:
        out.append(f"{CUTOFF_LABEL}: {chosen['cutoff']:g} Å (recommended: {r['cutoff']:g} Å)")
    if chosen["method"] == r["method"]:
        for k in ("kspace_accuracy", "disp_grid", "disp_gewald", "disp_spacing", "disp_real_accuracy",
                  "disp_kspace_accuracy"):
            a, b = chosen.get(k), r.get(k)
            if a == b or (isinstance(a, float) and isinstance(b, float) and math.isclose(a, b, rel_tol=1e-6)):
                continue
            out.append(f"{FIELD_LABELS[k]}: {_fmt(a)} (recommended: {_fmt(b)})")
        if chosen["method"] == "cutoff":
            a, b = chosen.get("coul_grid", "spacing"), r.get("coul_grid", "spacing")
            if a != b:
                out.append(f"{FIELD_LABELS['coul_grid']}: {a} (recommended: {b})")
    return out


def check_settings(s: Mapping, geometry: str = "slab", box: Sequence[float] | None = None,
                   machine: MachineInfo | bool | None = None, purpose: str = "screen") -> list[str]:
    """Warnings for a set of long-range ``settings`` that the pilots and the mesh check showed to be biased or
    unvalidated. They never block a choice."""
    geo = "bulk" if str(geometry).lower() == "bulk" else "slab"
    rc = float(s["cutoff"])
    out: list[str] = []
    if box and min(float(box[0]), float(box[1])) < 2.0 * rc + 1.0:
        out.append(f"the lateral box ({min(box[0], box[1]):.1f} Å) is too small for a {rc:g} Å cutoff (needs more "
                   f"than {2 * rc + 1:g} Å)")
    if s["method"] == "cutoff":
        if geo == "slab" and rc < 16.0 - 1e-9:
            out.append(f"a {rc:g} Å cutoff in a slab: pilot C (12 Å) settled 4.7 % below its bulk density, and its σ "
                       "with the tail, 8.76 mN/m (Coulomb-corrected), was 31 % below converged full long-range "
                       "dispersion (A-fine 12.62): the tail does not restore the density. 16 Å (pilot B: −9.6 %) or "
                       "full long-range dispersion is safer")
        if geo == "slab" and s.get("coul_grid", "spacing") == "accuracy":
            out.append("plain pppm with the mesh LAMMPS picks: for a slab box it chooses a spacing about 1.5 times "
                       "coarser along z than along x and y, and the error goes into Pzz (pilot C, 12 Å, 1e-5: σ 2.37 "
                       "mN/m high, still 0.32 at 1e-6; pilot B, 16 Å: 0.30). The default isotropic mesh (h·G ≤ "
                       f"{COUL_H_G:g}) removes it at no extra cost")
        if purpose == "reference":
            out.append("a cutoff + tail σ keeps a residual low bias (pilot B, 16 Å + tail, Coulomb-corrected: 11.41 "
                       "against A-fine's 12.62 mN/m for converged pppm/disp, −9.6 %, from the slab's 2.0 % lower "
                       "density; −5 % for hexane at 16 Å, Isele-Holder et al. 2012): for a reference value use full "
                       "long-range dispersion")
    else:
        if s.get("disp_grid") == "accuracy":
            loose = []
            if float(s["disp_real_accuracy"]) > DISP_REAL_ACCURACY * (1 + 1e-6):
                loose.append(f"force/disp/real {s['disp_real_accuracy']:g} (> {DISP_REAL_ACCURACY:g}: real-space "
                             "truncation, −0.30 mN/m at 1e-4)")
            if float(s["disp_kspace_accuracy"]) > 10 * DISP_KSPACE_ACCURACY * (1 + 1e-6):
                loose.append(f"force/disp/kspace {s['disp_kspace_accuracy']:g} (at 0.002 LAMMPS chose a 2 x 2 x 8 "
                             "mesh for pilot A's slab: −1.19 mN/m)")
            if loose and geo == "slab":
                out.append("pppm/disp looser than the mesh check's converged settings: " + "; ".join(loose)
                           + ". LAMMPS' estimate assumes a uniform density; the spacing rule is the slab-safe default")
        else:
            g, h = float(s["disp_gewald"]), float(s["disp_spacing"])
            if g * rc < VALIDATED_GRC:
                out.append(f"dispersion G·rc = {g * rc:.2f}: {real_space_left(g * rc):.0%} of r⁻⁶ is left at rc and "
                           "dropped beyond it (pilot A's 2.05, 21 %, lost 0.30 mN/m; 2.77 is converged)")
            if h * g > VALIDATED_HG:
                out.append(f"dispersion mesh spacing × G = {h * g:.2f} is coarser than the mesh check validated at "
                           "this G (converged: 0.52 at G 0.23, 0.57 at G 0.28; 1.17 lost 0.19 mN/m)")
    if float(s["kspace_accuracy"]) > 1.0e-5 * (1 + 1e-6):
        out.append(f"Coulomb PPPM {s['kspace_accuracy']:g} is looser than any pilot ran (1e-5 already shifts σ by "
                   "+0.15 mN/m against 1e-6, mesh check)")
    elif purpose == "reference" and float(s["kspace_accuracy"]) > 1.0e-6 * (1 + 1e-6):
        out.append(f"Coulomb PPPM {s['kspace_accuracy']:g}: +0.15 mN/m against 1e-6 at 1e-5 (mesh check); a "
                   "reference value uses 1e-6")
    return out


def notes_for(s: Mapping, geometry: str = "slab", machine: MachineInfo | bool | None = None) -> list[str]:
    """What to know about a set of ``settings`` that is not a bias: where it runs, what the analysis adds."""
    gpu = machine.gpu if isinstance(machine, MachineInfo) else bool(machine)
    out = []
    if s["method"] == "ewald" and gpu:
        out.append("the GPU stays idle: LAMMPS' GPU package has no lj/long/coul/long or pppm/disp")
    if s["method"] == "cutoff" and str(geometry).lower() != "bulk":
        out.append("the tail correction is added to σ by `md surface-tension`; it does not restore the slab's density "
                   "(pilot B: plateau 2.0 % below the bulk, σ + tail 9.6 % below converged full long-range dispersion)")
    return out


def options() -> dict:
    """The choices a form (the Simulate panel's physics nodes) offers, with labels, defaults and ranges."""
    return {
        "method": {"label": "Long-range treatment", "choices": dict(METHOD_LABELS), "default": "ewald"},
        "cutoff": {"label": CUTOFF_LABEL, "unit": CUTOFF_UNIT, "range": CUTOFF_RANGE,
                   "default": dict(DEFAULT_CUTOFF),
                   "help": "the real-space cutoff of the LJ and Coulomb terms; with 'cutoff' the LJ ends there and "
                           "the tail beyond it is added to σ"},
        "kspace_accuracy": {"label": "Coulomb PPPM accuracy (relative)", "default": dict(COULOMB_ACCURACY)},
        "disp_grid": {"label": "Dispersion k-space (full long-range only)", "choices": dict(DISP_GRIDS),
                      "default": "spacing"},
        "disp_gewald": {"label": "Dispersion G (1/Å)", "default": f"{DISP_G_RC:g} / cutoff", "applies": "spacing"},
        "disp_spacing": {"label": "Largest dispersion-mesh spacing (Å)", "default": f"{DISP_H_G:g} / G",
                         "applies": "spacing"},
        "disp_real_accuracy": {"label": "force/disp/real (kcal/mol/Å)", "default": DISP_REAL_ACCURACY,
                               "applies": "accuracy"},
        "disp_kspace_accuracy": {"label": "force/disp/kspace (kcal/mol/Å)", "default": DISP_KSPACE_ACCURACY,
                                 "applies": "accuracy"},
        "coul_grid": {"label": "Coulomb mesh (LJ cutoff only; plain pppm)", "choices": dict(COUL_GRIDS),
                      "default": "spacing", "applies": "cutoff"},
        "purpose": {"label": "Purpose", "choices": dict(PURPOSES), "default": "screen"},
        "provisional": PROVISIONAL, "provisional_note": PROVISIONAL_NOTE,
    }
