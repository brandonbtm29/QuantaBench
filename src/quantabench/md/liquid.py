"""Liquid-vapour slab MD of a pure liquid in LAMMPS, for its surface tension (MD-2).

``write_liquid_slab(typed_molecule, out_dir, config)`` writes a folder that runs with the plain ``lmp``
executable, no Python needed at run time::

    system.data        the packed bulk box: atom_style full, bonds, angles, dihedrals, impropers
    ff.params          pair/bond/angle/dihedral/improper coefficients (included by both inputs)
    in.bulk            minimize -> warm-up -> [compress] -> NPT at T, P  -> bulk_final.data, bulk_density.txt
    in.slab            reads bulk_final.data, opens the box along z with vacuum on both sides, NVT at T
                       -> pressure_tensor.txt (every 10 fs), density_profile.txt, slab_box.txt, slab_final.data
    in.run             both in one go (include in.bulk, clear, include in.slab)
    README.txt         the model, the exact commands (CPU, OpenMP, MPI, GPU where the variant allows it)
    liquid_setup.json  every setting and every force-field parameter with its source

``quantabench-cli md surface-tension DIR`` (``md.surface_tension``) turns the outputs into σ ± error.

The protocol, and why
---------------------
* **Bulk first.** The box is packed at an estimated density (Packmol if found, else molecules on a lattice and an
  NVT compression to that density), minimized, warmed at 0.5 fs, then equilibrated NPT (Nosé-Hoover thermostat
  and barostat, 1 fs, isotropic so the box keeps its shape). The bulk density of the force field is a result
  of its own (``bulk_density.txt``) and the reference the slab's centre is checked against.
* **Then the slab.** ``in.slab`` makes every molecule whole across the z boundary (unwrapped coordinates,
  molecule centres of mass put back into the bulk box), adds ``vacuum_A`` of empty space on each side, and runs
  NVT. The box stays periodic in z (two interfaces, no walls, no slab correction): with 2 x 50 Å of vacuum
  the periodic images of a neutral slab interact negligibly, and a molecule that evaporates crosses the boundary
  instead of being lost.
* **Long-range dispersion** (``dispersion``) is the main systematic choice:
  ``ewald``: ``lj/long/coul/long long long 12`` + ``pppm/disp`` (Ewald sums for r⁻⁶ as well as for charges; in 't
  Veld, Ismail & Grest 2007; Isele-Holder et al. 2012), the Tsige/Grest recommendation. CPU only (no GPU styles).
  ``cutoff12``: ``lj/cut/coul/long 12`` + ``pppm``: plain truncation of the LJ terms, which biases σ low (up to
  20 % for alkanes at 12 Å, Ismail et al. 2007). ``cutoff16``/``cutoff18``: the same with a longer cutoff, which
  the GPU package runs. The bulk NPT of the cutoff variants uses ``pair_modify tail yes`` (valid in a
  homogeneous liquid, so the bulk density is right); the slab never does (the tail correction assumes a
  uniform density).
* **Outputs for σ** (Kirkwood & Buff 1949): σ = (Lz/2)·⟨Pzz − (Pxx + Pyy)/2⟩ over the production run, the pressure
  tensor written every ``pressure_every`` steps; the mass density along z in ``profile_bin_A`` bins, averaged
  every 10 ps, for the plateau density, the vapour density, the interface width and the slab's drift.
  ``fix momentum`` removes the centre-of-mass motion every 1000 steps (PPPM does not conserve momentum
  exactly, and a drifting slab would smear the profile).

Units are LAMMPS ``real``: Å, fs, kcal/mol, g/mol, e, atm.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path

import numpy as np

from .. import __version__
from .forcefield import AVOGADRO, ForceFieldError, TypedMolecule

SIZES: dict[str, tuple[float, float]] = {
    # lateral box edge (Å) and liquid slab thickness (Å) at the estimated density
    "pilot": (40.0, 100.0),
    "full": (50.0, 100.0),
}
DISPERSION_HELP = ("ewald (lj/long/coul/long + pppm/disp), cutoff12 / cutoff16 / cutoff18 (lj/cut/coul/long + "
                   "pppm, plain LJ truncation at 12/16/18 Å)")


def parse_dispersion(name: str) -> tuple[str, float]:
    """``"ewald"`` -> ("ewald", 12.0); ``"cutoff16"`` -> ("cutoff", 16.0)."""
    s = name.strip().lower()
    if s == "ewald":
        return "ewald", 12.0
    m = re.fullmatch(r"cutoff(\d+(?:\.\d+)?)", s)
    if m and 8.0 <= float(m.group(1)) <= 30.0:
        return "cutoff", float(m.group(1))
    raise ValueError(f"dispersion {name!r}: choose {DISPERSION_HELP}")


@dataclass
class LiquidSlabConfig:
    """Everything that shapes the inputs. Times in fs or ns, lengths in Å, T in K, P in atm."""

    temperature: float = 298.15
    pressure_atm: float = 1.0
    timestep: float = 1.0
    dispersion: str = "ewald"
    cutoff: float | None = None              # None: from ``dispersion`` (ewald: 12 Å real-space cutoff)
    kspace_accuracy: float = 1.0e-5          # relative force accuracy of pppm and of pppm/disp's Coulomb part
    disp_real_accuracy: float = 1.0e-4       # pppm/disp, kcal/mol/Å: LAMMPS Howto_dispersion's values for real units
    disp_kspace_accuracy: float = 2.0e-3
    # box
    size: str = "pilot"
    lateral: float | None = None             # Å; None: from ``size``
    slab_thickness: float | None = None      # Å at the estimated density; None: from ``size``
    n_molecules: int | None = None           # overrides the thickness (the lateral size stays)
    vacuum: float = 50.0                     # Å of vacuum on each side of the slab
    density: float | None = None             # g/cm³ to pack at; None: QuantaBench's estimate
    packing: str = "auto"                    # auto (packmol if found) | packmol | lattice
    pack_tolerance: float = 2.0              # Å, Packmol's tolerance and the lattice margin
    packmol_density_factor: float = 0.95     # pack slightly below the estimate; NPT closes the gap
    # protocol
    warmup_ps: float = 10.0                  # NVT at half the timestep after the minimization
    compress_ps: float = 50.0                # lattice start only: NVT + fix deform to the estimated density
    bulk_ns: float = 1.0
    settle_ns: float = 2.0
    production_ns: float = 10.0
    tdamp: float = 100.0                     # fs, Nosé-Hoover thermostat
    pdamp: float = 1000.0                    # fs, Nosé-Hoover barostat
    # output
    pressure_every: int = 10                 # steps between pressure-tensor samples (10 fs at 1 fs)
    profile_bin: float = 0.5                 # Å
    profile_every: int = 100                 # steps between density-profile samples ...
    profile_repeat: int = 100                # ... averaged over this many ...
    profile_freq: int = 10_000               # ... and written every this many steps (10 ps)
    thermo_every: int = 1000
    dump_every: int = 100_000                # slab trajectory frames (0 = none)
    restart_every: int = 1_000_000
    momentum_every: int = 1000
    seed: int = 20261120
    # what the analysis compares against (optional, any source)
    reference_sigma: float | None = None     # mN/m
    reference_density: float | None = None   # g/cm³
    reference_note: str = ""

    def resolved(self) -> LiquidSlabConfig:
        kind, rc = parse_dispersion(self.dispersion)
        if self.size not in SIZES:
            raise ValueError(f"size {self.size!r}: choose {', '.join(SIZES)}")
        lat, thick = SIZES[self.size]
        return replace(self, cutoff=self.cutoff or rc, lateral=self.lateral or lat,
                       slab_thickness=self.slab_thickness or thick)


@dataclass
class LiquidSetup:
    out_dir: Path
    files: list[Path]
    config: LiquidSlabConfig
    n_molecules: int
    n_atoms: int
    box: tuple[float, float, float]           # as packed (Å)
    target_box: tuple[float, float, float]    # at the estimated density (Å)
    density_estimate: float
    density_source: str
    packing: str
    min_contact: float
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------------------------- the box
def plan_box(tm: TypedMolecule, density: float, cfg: LiquidSlabConfig) -> tuple[int, tuple[float, float, float]]:
    """(molecules, box at ``density``): lateral L x L, thickness from ``slab_thickness`` or ``n_molecules``."""
    v = tm.mass / density / AVOGADRO * 1e24            # Å³ per molecule
    lat = float(cfg.lateral)
    if cfg.n_molecules:
        n = int(cfg.n_molecules)
    else:
        n = max(2, int(round(lat * lat * float(cfg.slab_thickness) / v)))
    t = n * v / (lat * lat)
    if lat < 2.0 * float(cfg.cutoff) + 1.0:
        raise ValueError(f"lateral box {lat:.1f} Å is too small for a {cfg.cutoff:g} Å cutoff (needs > 2 x cutoff)")
    return n, (lat, lat, t)


def _aligned(pos: np.ndarray) -> np.ndarray:
    """Coordinates centred and rotated onto their principal axes (longest along x)."""
    c = pos - pos.mean(axis=0)
    _u, _s, vt = np.linalg.svd(c, full_matrices=False)
    r = c @ vt.T
    if np.linalg.det(vt) < 0:
        r[:, 2] *= -1
    return r


def pack_lattice(tm: TypedMolecule, n: int, box: Sequence[float], tolerance: float, seed: int
                 ) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Molecules on a rectangular lattice, all on their principal axes (every other one turned by 180° about z so
    the lattice has no net dipole), in the smallest box of ``box``'s shape whose cells clear each molecule by
    ``tolerance``. Returns (positions (n * atoms, 3), that box). The box is larger than ``box``; the bulk input
    then compresses it (NVT + fix deform) before the NPT stage."""
    ref = _aligned(tm.positions)
    ext = ref.max(axis=0) - ref.min(axis=0) + tolerance
    b = np.asarray(box, dtype=float)
    s = 1.0
    while True:
        counts = np.maximum(1, np.floor(s * b / ext)).astype(int)
        if int(np.prod(counts)) >= n:
            break
        s *= 1.01
    big = s * b
    spacing = big / counts
    rng = np.random.default_rng(seed)
    sites = np.sort(rng.choice(int(np.prod(counts)), size=n, replace=False))
    flip = ref * np.array([-1.0, -1.0, 1.0])
    out = np.empty((n, len(ref), 3))
    for m, site in enumerate(sites):
        ix, iy, iz = np.unravel_index(site, counts)
        centre = (np.array([ix, iy, iz]) + 0.5) * spacing
        out[m] = (flip if (ix + iy + iz) % 2 else ref) + centre
    return out.reshape(-1, 3), (float(big[0]), float(big[1]), float(big[2]))


def find_packmol() -> str | None:
    from ..chem import external

    info = external.find_program("packmol", version=False)
    return None if info is None or info.via_wsl else info.path


def pack_packmol(tm: TypedMolecule, n: int, box: Sequence[float], tolerance: float, seed: int, work: Path,
                 packmol: str, timeout: float = 3600.0) -> np.ndarray:
    """``n`` copies packed by Packmol (Martínez et al., J. Comput. Chem. 2009, 30, 2157) inside ``box``, kept
    ``tolerance``/2 from every face so that images across the periodic boundaries are also ``tolerance`` apart."""
    work.mkdir(parents=True, exist_ok=True)
    lines = [str(tm.n_atoms), tm.name]
    lines += [f"{e} {x:.6f} {y:.6f} {z:.6f}" for e, (x, y, z) in zip(tm.elements, tm.positions)]
    (work / "molecule.xyz").write_text("\n".join(lines) + "\n", encoding="utf-8")
    m = tolerance / 2.0
    inp = (f"tolerance {tolerance:g}\nfiletype xyz\noutput packed.xyz\nseed {seed}\n\n"
           f"structure molecule.xyz\n  number {n}\n"
           f"  inside box {m:.4f} {m:.4f} {m:.4f} {box[0] - m:.4f} {box[1] - m:.4f} {box[2] - m:.4f}\nend structure\n")
    (work / "packmol.inp").write_text(inp, encoding="utf-8")
    with open(work / "packmol.inp", encoding="utf-8") as fin, open(work / "packmol.log", "w", encoding="utf-8") as log:
        cp = subprocess.run([packmol], stdin=fin, stdout=log, stderr=subprocess.STDOUT, cwd=work,  # noqa: S603
                            timeout=timeout)
    out = work / "packed.xyz"
    text = (work / "packmol.log").read_text(encoding="utf-8", errors="replace")
    if not out.is_file() or "Success!" not in text:
        tail = "\n".join(text.strip().splitlines()[-8:])
        raise RuntimeError(f"Packmol did not converge (exit {cp.returncode}); see {work / 'packmol.log'}:\n{tail}")
    rows = out.read_text(encoding="utf-8").splitlines()[2:2 + n * tm.n_atoms]
    return np.array([[float(v) for v in r.split()[1:4]] for r in rows])


def min_contact(positions: np.ndarray, n_atoms_per_mol: int, box: Sequence[float]) -> float:
    """Closest distance (Å) between atoms of different molecules, periodic in all three directions."""
    from scipy.spatial import cKDTree

    b = np.asarray(box, dtype=float)
    p = np.mod(positions, b)
    tree = cKDTree(p, boxsize=b)
    mol = np.arange(len(p)) // n_atoms_per_mol
    pairs = tree.query_pairs(4.0, output_type="ndarray")
    if len(pairs) == 0:
        return math.inf
    pairs = pairs[mol[pairs[:, 0]] != mol[pairs[:, 1]]]
    if len(pairs) == 0:
        return math.inf
    d = p[pairs[:, 0]] - p[pairs[:, 1]]
    d -= b * np.round(d / b)
    return float(np.sqrt((d * d).sum(axis=1)).min())


# ------------------------------------------------------------------------------------------------ writers
def _g(x: float) -> str:
    return f"{float(x):.10g}"


def _coeff_text(style: str, coeffs: Sequence[float]) -> str:
    """Coefficients as LAMMPS reads them: integers where the style wants integers."""
    c = list(coeffs)
    if style == "fourier":                       # m K n d
        return f"1 {_g(c[0])} {int(round(c[1]))} {_g(c[2])}"
    if style == "cvff":                          # K d n
        return f"{_g(c[0])} {int(round(c[1]))} {int(round(c[2]))}"
    return " ".join(_g(v) for v in c)


def _type_maps(tm: TypedMolecule) -> dict[str, dict[str, int]]:
    return {"atom": {k: i + 1 for i, k in enumerate(tm.lj_types)},
            "bond": {k: i + 1 for i, k in enumerate(tm.bond_types)},
            "angle": {k: i + 1 for i, k in enumerate(tm.angle_types)},
            "dihedral": {k: i + 1 for i, k in enumerate(tm.dihedral_types)},
            "improper": {k: i + 1 for i, k in enumerate(tm.improper_types)}}


def write_data(path: Path, tm: TypedMolecule, positions: np.ndarray, n_mol: int, box: Sequence[float],
               title: str) -> dict[str, int]:
    """LAMMPS data file (atom_style full) of ``n_mol`` copies; returns the section counts."""
    maps = _type_maps(tm)
    na = tm.n_atoms
    counts = {"atoms": n_mol * na, "bonds": n_mol * len(tm.bonds), "angles": n_mol * len(tm.angles),
              "dihedrals": n_mol * len(tm.dihedrals), "impropers": n_mol * len(tm.impropers)}
    head = [f"{title} (QuantaBench {__version__}; atom_style full, units real)", ""]
    head += [f"{counts[k]} {k}" for k in ("atoms", "bonds", "angles", "dihedrals", "impropers")]
    head += [""]
    for kind in ("atom", "bond", "angle", "dihedral", "improper"):
        if maps[kind]:
            head.append(f"{len(maps[kind])} {kind} types")
    head += ["", f"0.0 {box[0]:.8f} xlo xhi", f"0.0 {box[1]:.8f} ylo yhi", f"0.0 {box[2]:.8f} zlo zhi", "",
             "Masses", ""]
    head += [f"{maps['atom'][k]} {t.mass:.5f} # {k}" for k, t in tm.lj_types.items()]
    out = ["\n".join(head), "", "Atoms # full", ""]
    rows = []
    q = tm.charges
    types = [maps["atom"][t] for t in tm.atom_types]
    for m in range(n_mol):
        for a in range(na):
            i = m * na + a
            x, y, z = positions[i]
            rows.append(f"{i + 1} {m + 1} {types[a]} {q[a]:.8f} {x:.6f} {y:.6f} {z:.6f} 0 0 0")
    out.append("\n".join(rows))
    for sec, terms, kind, k in (("Bonds", tm.bonds, "bond", 2), ("Angles", tm.angles, "angle", 3),
                                ("Dihedrals", tm.dihedrals, "dihedral", 4), ("Impropers", tm.impropers, "improper", 4)):
        if not terms:
            continue
        rows = []
        c = 0
        for m in range(n_mol):
            off = m * na + 1
            for t in terms:
                c += 1
                rows.append(f"{c} {maps[kind][t[k]]} " + " ".join(str(off + i) for i in t[:k]))
        out += ["", sec, "", "\n".join(rows)]
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return counts


def write_ff_params(path: Path, tm: TypedMolecule) -> None:
    maps = _type_maps(tm)
    L = [f"# Force-field coefficients for {tm.name}: {tm.label}",
         f"# QuantaBench {__version__}. Units real (kcal/mol, Å, degrees). Sources: README.txt and liquid_setup.json.",
         f"# Mixing: {tm.mixing} (pair_modify in the inputs); 1-4 scaling LJ {tm.special_lj[2]:g}, "
         f"Coulomb {tm.special_coul[2]:.6g} (special_bonds).", ""]
    for k, t in tm.lj_types.items():
        L.append(f"pair_coeff {maps['atom'][k]} {maps['atom'][k]} {_g(t.epsilon)} {_g(t.sigma)}  # {k} {t.element}")
    for kind, types in (("bond", tm.bond_types), ("angle", tm.angle_types), ("dihedral", tm.dihedral_types),
                        ("improper", tm.improper_types)):
        if types:
            L.append("")
        for k, t in types.items():
            L.append(f"{kind}_coeff {maps[kind][k]} {_coeff_text(t.style, t.coeffs)}  # {k}")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def _styles(tm: TypedMolecule, cfg: LiquidSlabConfig, stage: str) -> list[str]:
    kind, _rc = parse_dispersion(cfg.dispersion)
    L = ["units           real", "atom_style      full", "boundary        p p p"]
    for kind_t, terms in (("bond", tm.bonds), ("angle", tm.angles), ("dihedral", tm.dihedrals),
                          ("improper", tm.impropers)):
        if terms:
            L.append(f"{kind_t}_style{' ' * (10 - len(kind_t))}{tm.styles[kind_t]}")
    lj, cl = tm.special_lj, tm.special_coul
    L.append(f"special_bonds   lj {lj[0]:g} {lj[1]:g} {lj[2]:.10g} coul {cl[0]:g} {cl[1]:g} {cl[2]:.10g}")
    if kind == "ewald":
        L.append("pair_style      lj/long/coul/long long long ${rc}")
        L.append(f"pair_modify     mix {tm.mixing}")
    else:
        L.append("pair_style      lj/cut/coul/long ${rc}")
        if stage == "bulk":
            L.append(f"pair_modify     mix {tm.mixing} tail yes    # homogeneous bulk: the LJ tail correction is valid")
        else:
            L.append(f"pair_modify     mix {tm.mixing}             # NO tail correction: it assumes a uniform density")
    return L


def _kspace(tm: TypedMolecule, cfg: LiquidSlabConfig) -> list[str]:
    kind, _rc = parse_dispersion(cfg.dispersion)
    if kind == "ewald":
        L = ["kspace_style    pppm/disp ${kacc}",
             "# pppm/disp's dispersion accuracies set explicitly (LAMMPS Howto_dispersion: 0.0001 / 0.002 for real",
             "# units); without them LAMMPS stops, and its automatic choice is called inaccurate or slow",
             f"kspace_modify   force/disp/real {cfg.disp_real_accuracy:g} force/disp/kspace {cfg.disp_kspace_accuracy:g}"]
        if tm.mixing == "arithmetic":
            L.append("kspace_modify   mix/disp geom    # Lorentz-Berthelot in real space, geometric in k-space (LAMMPS "
                     "Howto_dispersion)")
        return L
    return ["kspace_style    pppm ${kacc}"]


def _header(tm: TypedMolecule, cfg: LiquidSlabConfig, what: str) -> list[str]:
    return [f"# QuantaBench {__version__}: {what}",
            f"# {tm.name} ({tm.smiles}), {tm.label}",
            f"# dispersion variant '{cfg.dispersion}'. Generated {datetime.now():%Y-%m-%d %H:%M}; see README.txt.",
            "# Override any 'index' variable on the command line, e.g.  -var production_ns 2", ""]


def _variables(cfg: LiquidSlabConfig) -> list[str]:
    return [f"variable T              index {cfg.temperature:g}       # K",
            f"variable P              index {cfg.pressure_atm:g}          # atm",
            f"variable dt             index {cfg.timestep:g}          # fs",
            f"variable rc             index {cfg.cutoff:g}         # Å, real-space cutoff",
            f"variable kacc           index {cfg.kspace_accuracy:g}",
            f"variable seed           index {cfg.seed}",
            f"variable tdamp          index {cfg.tdamp:g}",
            f"variable pdamp          index {cfg.pdamp:g}",
            f"variable thermo_every   index {cfg.thermo_every}"]


def bulk_input(tm: TypedMolecule, cfg: LiquidSlabConfig, compress_to: Sequence[float] | None) -> str:
    L = _header(tm, cfg, "bulk liquid, NPT")
    L += ["log             log.bulk", ""]
    L += _variables(cfg)
    L += [f"variable warmup_ps      index {cfg.warmup_ps:g}",
          f"variable compress_ps    index {cfg.compress_ps:g}" if compress_to else "",
          f"variable bulk_ns        index {cfg.bulk_ns:g}",
          "variable warmup_steps   equal round(v_warmup_ps*1000/(0.5*v_dt))",
          "variable compress_steps equal round(v_compress_ps*1000/v_dt)" if compress_to else "",
          "variable bulk_steps     equal round(v_bulk_ns*1000000/v_dt)", ""]
    L += _styles(tm, cfg, "bulk")
    L += ["read_data       system.data", "include         ff.params"]
    L += _kspace(tm, cfg)
    L += ["neighbor        2.0 bin", "neigh_modify    delay 0 every 1 check yes", ""]
    L += pair_energy_guard()
    L += [
          "variable        rho equal density",
          "thermo_style    custom step time temp press pe ke etotal density lx ly lz",
          "thermo          ${thermo_every}", "thermo_modify   flush yes", "",
          "# --- 1. energy minimization: removes the close contacts of the packing ---",
          "min_style       cg",
          "minimize        1.0e-4 1.0e-6 2000 20000",
          "reset_timestep  0", "",
          "# --- 2. warm-up: NVT at half the time step ---",
          "velocity        all create ${T} ${seed} dist gaussian mom yes rot yes",
          "timestep        $(0.5*v_dt)",
          "fix             warm all nvt temp ${T} ${T} 50.0",
          "run             ${warmup_steps}",
          "unfix           warm",
          "timestep        ${dt}", ""]
    if compress_to:
        L += ["# --- 3. lattice start: NVT compression to the estimated liquid density (fix deform) ---",
              "fix             nvt all nvt temp ${T} ${T} ${tdamp}",
              f"fix             squeeze all deform 1 x final 0.0 {compress_to[0]:.6f} y final 0.0 "
              f"{compress_to[1]:.6f} z final 0.0 {compress_to[2]:.6f} remap x",
              "run             ${compress_steps}",
              "unfix           squeeze", "unfix           nvt", ""]
    L += ["# --- NPT: Nosé-Hoover thermostat and barostat, isotropic (the box keeps its shape) ---",
          "fix             npt all npt temp ${T} ${T} ${tdamp} iso ${P} ${P} ${pdamp}",
          "variable        vol equal vol",
          "variable        lxv equal lx",
          "variable        lyv equal ly",
          "variable        lzv equal lz",
          "variable        pev equal pe",
          "# every 1 ps: averages of 10 samples 100 steps apart (density g/cm3, volume A3, box A, pe kcal/mol)",
          "fix             rhoavg all ave/time 100 10 1000 v_rho v_vol v_lxv v_lyv v_lzv v_pev file bulk_density.txt",
          f"restart         {cfg.restart_every} bulk.restart.a bulk.restart.b",
          "run             ${bulk_steps}",
          "unfix           rhoavg",
          "write_data      bulk_final.data nocoeff",
          'print           "QB_BULK done: bulk_final.data written (box $(lx) $(ly) $(lz))"', ""]
    return "\n".join(x for x in L if x is not None) + "\n"


def slab_input(tm: TypedMolecule, cfg: LiquidSlabConfig, elements_by_type: Sequence[str]) -> str:
    L = _header(tm, cfg, "liquid-vapour slab, NVT, pressure tensor and density profile")
    L += ["log             log.slab", ""]
    L += _variables(cfg)
    L += [f"variable vacuum         index {cfg.vacuum:g}         # Å on each side",
          f"variable settle_ns      index {cfg.settle_ns:g}",
          f"variable production_ns  index {cfg.production_ns:g}",
          f"variable p_every        index {cfg.pressure_every}         # steps between pressure-tensor samples",
          f"variable bin            index {cfg.profile_bin:g}        # Å, density-profile bins",
          f"variable prof_every     index {cfg.profile_every}",
          f"variable prof_repeat    index {cfg.profile_repeat}",
          f"variable prof_freq      index {cfg.profile_freq}",
          f"variable dump_every     index {cfg.dump_every}",
          "variable settle_steps   equal round(v_settle_ns*1000000/v_dt)",
          "variable production_steps equal round(v_production_ns*1000000/v_dt)",
          "variable slab_steps     equal v_settle_steps+v_production_steps", ""]
    L += _styles(tm, cfg, "slab")
    L += ["read_data       bulk_final.data", "include         ff.params"]
    L += _kspace(tm, cfg)
    L += ["neighbor        2.0 bin", "neigh_modify    delay 0 every 1 check yes", ""]
    L += pair_energy_guard()
    L += [
          "# --- open the box along z ---",
          "# Store every atom's unwrapped z and its molecule's centre of mass, shift each molecule by whole box",
          "# lengths so that its centre lies in the bulk box (molecules stay whole), then add the vacuum.",
          "variable        zlo0 equal $(zlo)",
          "variable        lz0 equal $(lz)",
          "compute         molchunk all chunk/atom molecule",
          "compute         molcom all com/chunk molchunk",
          "compute         molcomz all chunk/spread/atom molchunk c_molcom[3]",
          "fix             zsave all store/state 0 zu c_molcomz",
          "thermo_style    custom step temp press pe",
          "run             0 post no",
          "variable        kshift atom floor((f_zsave[2]-v_zlo0)/v_lz0)",
          "variable        znew atom f_zsave[1]-v_kshift*v_lz0",
          "change_box      all z final $(v_zlo0-v_vacuum) $(v_zlo0+v_lz0+v_vacuum) units box",
          "set             group all image NULL NULL 0",
          "set             group all z v_znew",
          "unfix           zsave",
          "uncompute       molcomz", "uncompute       molcom", "uncompute       molchunk",
          "reset_timestep  0",
          "timestep        ${dt}",
          'print           "QB_SLAB lx $(lx:%.6f) ly $(ly:%.6f) lz $(lz:%.6f) zlo $(zlo:%.6f) zhi $(zhi:%.6f) '
          'dt ${dt} p_every ${p_every} settle_steps ${settle_steps} production_steps ${production_steps} '
          'T ${T} rc ${rc} bulk_lz $(v_lz0:%.6f)" file slab_box.txt',
          "",
          "# --- NVT (Nosé-Hoover); no drift of the slab's centre of mass ---",
          "fix             nvt all nvt temp ${T} ${T} ${tdamp}",
          f"fix             nodrift all momentum {cfg.momentum_every} linear 1 1 1",
          "variable        comz equal xcm(all,z)",
          "thermo_style    custom step time temp press pe ke etotal pxx pyy pzz v_comz cpuremain",
          "thermo          ${thermo_every}", "thermo_modify   flush yes", "",
          "# pressure tensor (atm) every p_every steps: sigma = (Lz/2) <Pzz - (Pxx+Pyy)/2>",
          "fix             ptens all ave/time ${p_every} 1 ${p_every} c_thermo_press[1] c_thermo_press[2] "
          'c_thermo_press[3] file pressure_tensor.txt format " %.8g"',
          "# mass density along z (g/cm3), bins fixed in the box: averages written every prof_freq steps",
          "compute         zbins all chunk/atom bin/1d z lower ${bin} units box",
          "fix             dprof all ave/chunk ${prof_every} ${prof_repeat} ${prof_freq} zbins density/mass "
          "file density_profile.txt",
          f"restart         {cfg.restart_every} slab.restart.a slab.restart.b"]
    if cfg.dump_every:
        L += ["if \"${dump_every} > 0\" then &",
              '  "dump traj all custom ${dump_every} slab.lammpstrj id mol type element x y z ix iy iz" &',
              f'  "dump_modify traj sort id element {" ".join(elements_by_type)}"']
    L += ["run             ${slab_steps}",
          "write_data      slab_final.data nocoeff",
          'print           "QB_SLAB done"', ""]
    return "\n".join(L) + "\n"


RUN_INPUT = """# QuantaBench: bulk NPT and then the slab, in one LAMMPS run (the same as in.bulk followed by in.slab)
include in.bulk
clear
include in.slab
"""


def gpu_switches(opencl: bool = True, omp_threads: int | None = 4) -> list[str]:
    """LAMMPS command-line switches for a GPU-package run.

    On OpenCL builds the GPU package picks per-vendor kernel tuning. On the workstation's AMD RX 9060 XT (gfx1200,
    RDNA4) the "AMD_GPU" tuning built empty neighbor lists without any error: the pair style and pppm/gpu then
    returned no force, energy or virial, so an NPT box of C6F14 expanded to half its density, and ``clear`` hung
    while the device shut down (2026-11-20, report in Runs\\md-gpu-check-2026-11-20). ``device_type generic``
    fixes both and matches the CPU forces to 1e-6. ``pair/only on`` keeps PPPM and the bonded terms on the CPU
    (OPENMP threads), which was also the fastest setup measured there (15.2 vs 10.2 ns/day for 9,780 atoms).
    ``opencl=False`` (CUDA/HIP builds) leaves the device tuning to LAMMPS."""
    pk = ["-pk", "gpu", "1"] + (["device_type", "generic"] if opencl else [])
    if omp_threads:
        return ["-sf", "hybrid", "gpu", "omp", *pk, "pair/only", "on", "-pk", "omp", str(omp_threads)]
    return ["-sf", "gpu", *pk]


def pair_energy_guard() -> list[str]:
    """Input lines that stop LAMMPS (exit code 3) when the pair style returns no energy at all: what a GPU
    neighbor-list failure looks like (see ``gpu_switches``). Placed after read_data and the force field."""
    return ["# --- guard: nonbonded energy must be non-zero (a GPU neighbor-list failure returns none) ---",
            "run             0 post no",
            "variable        qb_epair equal abs(ecoul)+abs(evdwl-etail)",
            'if "${qb_epair} < 1.0e-6" then "print \'QB_ERROR: the pair style returned no energy; nonbonded '
            'forces are missing (GPU neighbor-list failure?)\'" "quit 3"', ""]


def _commands(cfg: LiquidSlabConfig) -> list[str]:
    kind, _rc = parse_dispersion(cfg.dispersion)
    L = ["HOW TO RUN (from this folder; the two stages one after the other, or both with in.run)",
         "  Windows (native, the LAMMPS installer's lmp.exe; backlog MD-1: the PyPI wheel failed 5-10 % of",
         "  identical runs on Windows, so prefer the installer's binary there):",
         "    lmp.exe -sf omp -pk omp 12 -in in.bulk",
         "    lmp.exe -sf omp -pk omp 12 -in in.slab",
         "    (or MPI:  mpiexec -np 12 lmp.exe -in in.bulk   then   mpiexec -np 12 lmp.exe -in in.slab)",
         "  Linux / WSL / macOS:",
         "    lmp -sf omp -pk omp 12 -in in.bulk && lmp -sf omp -pk omp 12 -in in.slab",
         "    mpirun -np 12 lmp -in in.bulk && mpirun -np 12 lmp -in in.slab",
         "  Both stages in one command:  lmp -sf omp -pk omp 12 -in in.run"]
    if kind == "ewald":
        L += ["  GPU: not for this variant. LAMMPS has no GPU version of lj/long/coul/long or pppm/disp (their",
              "  accelerated versions are OPENMP, INTEL and OPT), so run it on the CPU as above. The GPU-friendly",
              "  variants are cutoff16 / cutoff18 (quantabench-cli md liquid-slab ... --dispersion cutoff16)."]
    else:
        g = " ".join(gpu_switches(True, 4))
        L += ["  GPU (GPU package; the pair style on the GPU, PPPM and the bonded terms on 4 OpenMP threads):",
              f"    lmp.exe {g} -in in.bulk",
              f"    lmp.exe {g} -in in.slab",
              f"    (Linux: lmp {g} -in in.bulk && lmp {g} -in in.slab)",
              "  'device_type generic' is needed on OpenCL builds: with LAMMPS' AMD tuning an RX 9060 XT (gfx1200)",
              "  built empty neighbor lists and the run had no nonbonded forces (the inputs now stop with QB_ERROR",
              "  if that happens). CUDA/HIP builds: drop 'device_type generic'. Run the two stages as separate",
              "  processes on a GPU (in.run, with clear between them, hung there without the fix)."]
    L += ["  Shorter runs: add for example  -var bulk_ns 0.2  to in.bulk and  -var settle_ns 0.5 -var production_ns 2",
          "  to in.slab. A smoke test: -var bulk_ns 0.02 (bulk), -var settle_ns 0.01 -var production_ns 0.01 (slab).",
          "  Python (the lammps module, for short checks): quantabench.md.liquid.run_liquid_slab(<this folder>)"]
    return L


def _readme(tm: TypedMolecule, cfg: LiquidSlabConfig, s: LiquidSetup, counts: dict[str, int]) -> str:
    kind, rc = parse_dispersion(cfg.dispersion)
    lx, ly, lz = s.target_box
    disp = {"ewald": f"lj/long/coul/long long long {rc:g} + kspace pppm/disp {cfg.kspace_accuracy:g} (dispersion "
                     f"Ewald; force/disp/real {cfg.disp_real_accuracy:g}, force/disp/kspace {cfg.disp_kspace_accuracy:g})",
            "cutoff": f"lj/cut/coul/long {rc:g} + kspace pppm {cfg.kspace_accuracy:g}: LJ truncated at {rc:g} Å "
                      "(bulk: with the tail correction; slab: without, so sigma is biased low)"}[kind]
    L = [f"QuantaBench {__version__}: liquid-vapour slab of {tm.name}", "=" * 60, "",
         "WHAT THIS SIMULATES",
         f"  {s.n_molecules} molecules of {tm.name} ({tm.smiles}), {s.n_atoms} atoms, M = {tm.mass:.3f} g/mol.",
         f"  Packed ({s.packing}) at {s.density_estimate:.4f} g/cm3 ({s.density_source}); at that density the box is "
         f"{lx:.2f} x {ly:.2f} x {lz:.2f} A.",
         f"  Bulk (in.bulk): minimize, {cfg.warmup_ps:g} ps NVT warm-up at {cfg.timestep / 2:g} fs"
         + (f", {cfg.compress_ps:g} ps NVT compression to that box" if s.packing == "lattice" else "")
         + f", then {cfg.bulk_ns:g} ns NPT at {cfg.temperature:g} K and {cfg.pressure_atm:g} atm (Nose-Hoover,",
         f"  Tdamp {cfg.tdamp:g} fs, Pdamp {cfg.pdamp:g} fs, isotropic), time step {cfg.timestep:g} fs.",
         f"  Slab (in.slab): the bulk box opened along z with {cfg.vacuum:g} A of vacuum on each side, periodic in all",
         f"  three directions; NVT at {cfg.temperature:g} K for {cfg.settle_ns:g} ns to settle + {cfg.production_ns:g} ns of "
         "production (one run;", "  the analysis skips the settle time).",
         f"  Long-range interactions: {disp}.",
         "",
         "FORCE FIELD",
         f"  {tm.label}.",
         f"  Mixing: {tm.mixing}. 1-4 scaling: LJ {tm.special_lj[2]:g}, Coulomb {tm.special_coul[2]:.6g}.",
         f"  Bonded styles: {', '.join(f'{k} {v}' for k, v in tm.styles.items())}.",
         "  Atom types (epsilon kcal/mol, sigma A, charge e):"]
    charges = {}
    for t, q in zip(tm.atom_types, tm.charges):
        charges.setdefault(t, []).append(float(q))
    for k, t in tm.lj_types.items():
        qs = charges.get(k, [0.0])
        qtxt = f"{qs[0]:+.4f}" if max(qs) - min(qs) < 1e-6 else f"{min(qs):+.4f}..{max(qs):+.4f}"
        L.append(f"    {k:12s} {t.element:2s} eps {t.epsilon:8.5f} sigma {t.sigma:7.4f} q {qtxt}")
        L.append(f"      {t.source}")
    for title, types in (("Bonds", tm.bond_types), ("Angles", tm.angle_types), ("Dihedrals", tm.dihedral_types),
                         ("Impropers", tm.improper_types)):
        if not types:
            continue
        L.append(f"  {title}:")
        for k, t in types.items():
            L.append(f"    {k:28s} {t.style:9s} {_coeff_text(t.style, t.coeffs)}")
            L.append(f"      {t.source}")
    if tm.overrides:
        L.append(f"  Overridden LJ parameters: {', '.join(tm.overrides)} (see liquid_setup.json).")
    L += ["  References:"] + [f"    - {r}" for r in tm.references]
    if tm.notes:
        L += ["  Notes:"] + [f"    - {n}" for n in tm.notes]
    L += ["", *_commands(cfg), "",
          "OUTPUTS",
          "  log.bulk, log.slab        LAMMPS logs (thermo every 1 ps; Performance: ns/day at the end of each run)",
          "  bulk_density.txt          1-ps averages over the NPT run: density (g/cm3), volume, box, pe",
          "  bulk_final.data           the equilibrated bulk box (with velocities): the slab's start",
          "  slab_box.txt              the slab box and the run lengths the analysis needs",
          f"  pressure_tensor.txt       Pxx Pyy Pzz (atm) every {cfg.pressure_every} steps "
          f"({cfg.pressure_every * cfg.timestep:g} fs)",
          f"  density_profile.txt       mass density (g/cm3) in {cfg.profile_bin:g} A bins along z, averaged over "
          f"{cfg.profile_freq * cfg.timestep / 1000:g} ps blocks",
          "  slab.lammpstrj            trajectory for OVITO (every 100 ps)" if cfg.dump_every else "",
          "  slab_final.data, *.restart.a/b   snapshots and restart files", "",
          "ANALYSIS",
          "  quantabench-cli md surface-tension <this folder>    (writes surface_tension.json and .md here)",
          "  sigma = (Lz/2) <Pzz - (Pxx+Pyy)/2>,  1 atm*A = 0.0101325 mN/m; error by 1-ns block averages.",
          "  Also: the slab's plateau density vs the bulk NPT density, the vapour density, the interface width",
          "  and the slab's drift.", "",
          "EXPECTED WARNINGS",
          "  'Using 'neigh_modify every 1 delay 0 check yes' setting during minimization': harmless.",
          "  'Increasing communication cutoff ...' / 'Bond/angle/dihedral extent > half of periodic box length' must",
          "  NOT appear for the slab (it would mean a molecule was split when the box was opened).",
          ]
    if cfg.reference_sigma or cfg.reference_density:
        L += ["", "REFERENCE VALUES (for the analysis)",
              f"  sigma {cfg.reference_sigma} mN/m, density {cfg.reference_density} g/cm3. {cfg.reference_note}"]
    L += ["", f"Counts: {counts}"]
    return "\n".join(x for x in L if x is not None) + "\n"


def write_liquid_slab(tm: TypedMolecule, out_dir: str | Path, config: LiquidSlabConfig | None = None, *,
                      density_estimate: float | None = None, density_source: str = "") -> LiquidSetup:
    """Write a runnable liquid-slab folder for the typed molecule ``tm`` (see the module docstring)."""
    cfg = (config or LiquidSlabConfig()).resolved()
    tm.check()
    if abs(tm.net_charge) > 1e-6:
        raise ForceFieldError("the molecule is not neutral")
    rho = cfg.density or density_estimate
    if not rho:
        raise ValueError("no density to pack at: pass config.density or density_estimate")
    how = "given" if cfg.density else (density_source or "estimate")
    n, target = plan_box(tm, rho, cfg)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []
    packmol = find_packmol() if cfg.packing in ("auto", "packmol") else None
    if cfg.packing == "packmol" and packmol is None:
        raise RuntimeError("Packmol not found (conda install -c conda-forge packmol), or use packing='lattice'")
    if packmol:
        f = (1.0 / cfg.packmol_density_factor) ** (1.0 / 3.0)
        box = (target[0] * f, target[1] * f, target[2] * f)
        pos = pack_packmol(tm, n, box, cfg.pack_tolerance, cfg.seed, out / "packing", packmol)
        packing, compress_to = "packmol", None
        notes.append(f"packed by Packmol at {cfg.packmol_density_factor:g} x the estimated density; NPT closes the gap")
    else:
        pos, box = pack_lattice(tm, n, target, cfg.pack_tolerance, cfg.seed)
        packing, compress_to = "lattice", target
        notes.append(f"lattice start at {rho * np.prod(target) / np.prod(box):.3f} g/cm3, compressed to "
                     f"{rho:.3f} g/cm3 in {cfg.compress_ps:g} ps before the NPT stage")
    contact = min_contact(pos, tm.n_atoms, box)
    if contact < 0.8:
        raise RuntimeError(f"packing left atoms of different molecules {contact:.2f} Å apart")
    title = f"{tm.name} liquid ({tm.forcefield})"
    counts = write_data(out / "system.data", tm, pos, n, box, title)
    write_ff_params(out / "ff.params", tm)
    elements_by_type = [t.element for t in tm.lj_types.values()]
    (out / "in.bulk").write_text(bulk_input(tm, cfg, compress_to), encoding="utf-8")
    (out / "in.slab").write_text(slab_input(tm, cfg, elements_by_type), encoding="utf-8")
    (out / "in.run").write_text(RUN_INPUT, encoding="utf-8")
    setup = LiquidSetup(out_dir=out, files=[], config=cfg, n_molecules=n, n_atoms=n * tm.n_atoms, box=tuple(box),
                        target_box=tuple(target), density_estimate=float(rho), density_source=how, packing=packing,
                        min_contact=contact, notes=notes)
    (out / "README.txt").write_text(_readme(tm, cfg, setup, counts), encoding="utf-8")
    meta = {"generator": f"QuantaBench {__version__}", "created": datetime.now().isoformat(timespec="seconds"),
            "kind": "liquid-slab", "config": asdict(cfg), "n_molecules": n, "n_atoms": n * tm.n_atoms,
            "counts": counts, "box_packed_A": list(box), "box_target_A": list(target),
            "density_pack_g_cm3": float(rho), "density_source": how, "packing": packing,
            "min_intermolecular_contact_A": contact, "notes": notes, "molecule": tm.to_dict(),
            "type_ids": _type_maps(tm), "commands": _commands(cfg)}
    (out / "liquid_setup.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    setup.files = [out / f for f in ("system.data", "ff.params", "in.bulk", "in.slab", "in.run", "README.txt",
                                     "liquid_setup.json")]
    return setup


def read_setup(folder: str | Path) -> dict:
    return json.loads((Path(folder) / "liquid_setup.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------------ running
def run_liquid_slab(folder: str | Path, stages: Sequence[str] = ("bulk", "slab"),
                    variables: Mapping[str, object] | None = None, cmdargs: Sequence[str] = (),
                    screen: bool = False) -> list[Path]:
    """Run ``in.bulk`` and/or ``in.slab`` in ``folder`` with the LAMMPS Python module (short checks and tests;
    production runs belong to the ``lmp`` executable, README.txt). Returns the log paths."""
    import lammps  # lazy: optional dependency (quantabench[md])

    folder = Path(folder).resolve()
    logs = []
    cwd = Path.cwd()
    os.chdir(folder)        # the inputs use relative paths so the folder can be moved or synced as a unit
    try:
        for stage in stages:
            args = ["-nocite", "-log", "none"]
            if not screen:
                args += ["-screen", "none"]
            for k, v in (variables or {}).items():
                args += ["-var", str(k), str(v)]
            args += list(cmdargs)
            lmp = lammps.lammps(cmdargs=args)
            try:
                lmp.file(f"in.{stage}")
            finally:
                lmp.close()
            logs.append(folder / f"log.{stage}")
    finally:
        os.chdir(cwd)
    return logs


def lammps_executable() -> str | None:
    """``lmp`` (or the installer's names) on PATH, for the README's commands and ``--run``."""
    for name in ("lmp", "lmp_serial", "lmp_mpi", "lmp.exe"):
        p = shutil.which(name)
        if p:
            return p
    return None
