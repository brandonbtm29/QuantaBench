"""Steered-MD (SMD) setup for LAMMPS: pull a rigid adsorbate off a frozen surface.

This turns a QuantaBench system (an ASE ``Atoms`` from ``build.surfaces.place_on_surface``: tag 0 =
surface, tag >= 1 = adsorbate) into a ready-to-run LAMMPS folder::

    system.data    atom_style full, one molecule ID per body, per-element types, charges, no bonds
    in.smd         soft start -> NVT equilibration -> constant-velocity SMD along +z
    README.txt     the model, its approximations, and the exact run commands (Windows / Linux / GPU)
    smd_setup.json every setting and parameter, for reproducibility

The model, and why
------------------
The goal is a *physically sensible* detachment run without a bonded force field, because typing
PFAS for OPLS/GAFF by hand is where the earlier runs failed ("exploded" with guessed charges, "Lost
atoms" from overlaps, missing ``extra/special/per/atom`` for bonded topologies).

1. **Rigid adsorbate.** Each adsorbate molecule is one ``fix rigid/small`` body. Its internal
   geometry is the ML/xTB-optimized one from the QuantaBench pipeline and its partial charges are the
   AIMNet2/xTB charges, so no bonds, angles or dihedrals are needed. Intra-molecular pairs are excluded
   from the neighbor list (``neigh_modify exclude molecule/intra``): they could not move a rigid body
   anyway. What is lost: conformational relaxation on the surface (e.g. a PFAS tail flattening or
   a head group rotating independently). Single-atom adsorbates (counter-ions) are integrated as
   point particles (``nve/limit`` + Langevin) with Joung-Cheatham ion LJ parameters.
2. **Frozen surface.** The slab atoms belong to no integrator and start with zero velocity, so they
   never move; surface-surface pairs are excluded. This is the standard rigid-surface approximation
   for adsorption on graphite (stiff lattice, ~0.1 Å thermal motion) and it removes the need for a
   bonded carbon force field. ``fix setforce 0`` on the surface is deliberately *not* used: without an
   integrator the forces are never applied, and keeping them leaves the per-atom surface forces
   available for analysis (e.g. ``compute group/group``).
3. **Nonbonded model.** Lennard-Jones 12-6: UFF for the adsorbate, AMBER/Steele carbon for graphite,
   geometric mixing (see ``quantabench.md.params``), plus long-range Coulomb (PPPM with the
   Yeh-Berkowitz slab correction, ``kspace_modify slab 3.0``) when any charge is non-zero.
   The graphite carries no charges and is **not polarizable**: the image-charge attraction a charged
   head group feels near a conducting graphite sheet is missing.
4. **Vacuum.** There is no solvent. The work measured is a gas-phase detachment work, not an aqueous
   adsorption free energy; for PFAS in water both solvation of the head group and hydrophobic
   dewetting matter. Use this to rank adsorbates and check setups, then add water before quoting
   numbers.
5. **Non-equilibrium work.** A single constant-velocity pull gives W >= ΔF (second law; Jarzynski,
   *Phys. Rev. Lett.* 1997, 78, 2690). A PMF needs several pulls (Jarzynski average or cumulant
   expansion; Park and Schulten, *J. Chem. Phys.* 2004, 120, 5946) or umbrella sampling.

Stages in ``in.smd``
--------------------
* **soft start:** 0.5 fs, Langevin-thermostatted rigid bodies with a short damping time (10 fs) and an
  optional ``fix viscous``, so any residual overlap is dissipated instead of launching the molecule.
* **NVT equilibration:** 1 fs, rigid-body Langevin (100 fs). Langevin rather than Nosé-Hoover because a
  single rigid body has only 6 degrees of freedom, too few for a deterministic thermostat to be ergodic.
* **SMD:** ``fix smd cvel K v tether NULL NULL z_surface 0.0`` on the pulled molecule's center of mass:
  a spring (K = 10 kcal/mol/Å², stiff-spring regime) whose end moves away from the surface plane at
  v = 1e-4 Å/fs (10 m/s) until it has travelled ``pull_distance``. The thermostat is weakened
  (1000 fs) because Langevin friction on the pulled body adds a drag work of about
  M v d / τ (see ``langevin_drag_work``).

Units are LAMMPS ``real``: Å, fs, kcal/mol, g/mol, elementary charges.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.data import atomic_masses, atomic_numbers
from ase.neighborlist import natural_cutoffs, neighbor_list

from .. import __version__
from . import params as P

TAG_SURFACE = 0          # same convention as build.surfaces (kept local so md/ does not import build/)
SURFACE_MOL_ID = 1
FTM2V_REAL = 1.0 / 48.88821291 / 48.88821291   # LAMMPS real units: (kcal/mol/Å) -> g/mol * Å/fs^2
KCAL_MOL_A_TO_PN = 69.4786                       # 1 kcal/mol/Å in pN (4184 J / N_A / 1e-10 m)


@dataclass
class SMDConfig:
    """Everything that shapes ``in.smd``. Times in fs, lengths in Å, energies in kcal/mol."""

    temperature: float = 300.0
    # stage 1: soft start
    soft_steps: int = 2000
    soft_timestep: float = 0.5
    soft_damp: float = 10.0               # Langevin damping time of the rigid bodies (fs)
    # Free counter-ions: strong Langevin friction in every stage. In vacuum an ion falls onto the
    # oppositely charged head group and releases tens of kcal/mol; the weak rigid-body thermostats
    # cannot remove that, the contact pair rings at thousands of K, and the run eventually fails.
    # Ions are not pulled, so their friction does not enter the SMD work.
    ion_damp: float = 20.0
    viscous_gamma: float = 0.0            # optional extra fix viscous (kcal/mol/Å per Å/fs); 0 = off
    # stage 2: NVT equilibration
    equil_steps: int = 50_000             # 50 ps
    timestep: float = 1.0
    equil_damp: float = 100.0
    # stage 3: SMD
    pull_velocity: float = 1.0e-4         # Å/fs (= 10 m/s)
    spring_k: float = 10.0                # kcal/mol/Å^2
    pull_distance: float = 15.0           # Å travelled by the spring end (sets the SMD step count)
    pull_damp: float = 1000.0             # weak thermostat during the pull (drag bias, see README)
    pull_molecule: int = 0                # which adsorbate molecule to pull (0 = first)
    # interactions
    cutoff: float = 12.0
    kspace_accuracy: float = 1.0e-4
    slab_factor: float = 3.0
    graphite: str = P.DEFAULT_GRAPHITE    # "amber" (ε = 0.086) or "steele" (ε = 0.0556)
    interaction_energy: bool = True       # compute group/group pulled-surface energy in thermo/pull file
    # output
    thermo_every: int = 1000
    dump_every: int = 1000
    pull_output_every: int = 100
    # box / safety
    z_padding: float = 10.0               # vacuum above the end point of the pull
    z_below: float = 2.0                  # space under the lowest slab atom
    min_gap: float = 1.5                  # refuse to write systems with closer contacts (Lost atoms)
    seed: int = 4928459
    # GPU package (OpenCL/CUDA/HIP builds): adds "package gpu N neigh no" + "suffix gpu"
    gpu: bool = False
    gpu_count: int = 1

    @property
    def smd_steps(self) -> int:
        return int(math.ceil(self.pull_distance / (self.pull_velocity * self.timestep)))


@dataclass
class SMDSetup:
    """What ``write_smd_setup`` produced."""

    out_dir: Path
    data: Path
    input: Path
    readme: Path
    metadata: Path
    config: SMDConfig
    type_labels: list[str]
    lj: dict[str, P.LJ]
    n_surface: int
    molecules: list[list[int]]            # 0-based atom indices of each adsorbate molecule
    net_charge: float
    box: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    notes: list[str] = field(default_factory=list)

    @property
    def paths(self) -> list[Path]:
        return [self.data, self.input, self.readme, self.metadata]


# ----------------------------------------------------------------------------------------------- system


def _adsorbate_molecules(atoms: Atoms, ads_idx: np.ndarray, tags: np.ndarray) -> list[list[int]]:
    """Split adsorbate atoms into molecules: covalent connectivity, never across different tags."""
    if len(ads_idx) == 0:
        return []
    sub = atoms[ads_idx]
    sub.pbc = atoms.pbc
    i, j = neighbor_list("ij", sub, natural_cutoffs(sub, mult=1.15))
    parent = list(range(len(sub)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    sub_tags = tags[ads_idx]
    for a, b in zip(i, j):
        if sub_tags[a] == sub_tags[b]:
            parent[find(a)] = find(b)
    groups: dict[int, list[int]] = {}
    for k in range(len(sub)):
        groups.setdefault(find(k), []).append(int(ads_idx[k]))
    return sorted(groups.values(), key=lambda g: g[0])


def _mic_min_distance(a: np.ndarray, b: np.ndarray, lx: float, ly: float) -> float:
    if len(a) == 0 or len(b) == 0:
        return math.inf
    d = a[:, None, :] - b[None, :, :]
    d[..., 0] -= lx * np.round(d[..., 0] / lx)
    d[..., 1] -= ly * np.round(d[..., 1] / ly)
    return float(np.sqrt((d ** 2).sum(-1)).min())


def langevin_drag_work(mass_g_mol: float, velocity_a_fs: float, distance_a: float, damp_fs: float) -> float:
    """Extra work (kcal/mol) a Langevin thermostat's friction adds to a body dragged at constant
    speed: F = M v / τ (converted from g/mol·Å/fs² to kcal/mol/Å), times the distance."""
    return mass_g_mol * velocity_a_fs / damp_fs / FTM2V_REAL * distance_a


def _resolve_charges(atoms: Atoms, charges, surface: np.ndarray, ads_idx: np.ndarray) -> np.ndarray:
    n = len(atoms)
    if charges is None:
        q = np.asarray(atoms.get_initial_charges(), dtype=float)
    else:
        c = np.asarray(charges, dtype=float).ravel()
        if len(c) == n:
            q = c.copy()
        elif len(c) == len(ads_idx):
            q = np.zeros(n)
            q[ads_idx] = c
        else:
            raise ValueError(f"charges has {len(c)} entries; expected {n} (whole system) or "
                             f"{len(ads_idx)} (adsorbate atoms only)")
    if not np.all(np.isfinite(q)):
        raise ValueError("charges contain NaN/inf")
    return q


def write_smd_setup(system_atoms: Atoms, out_dir: str | Path, temperature: float | None = None, *,
                    soft_steps: int | None = None, equil_steps: int | None = None,
                    smd_steps: int | None = None, pull_distance: float | None = None,
                    pull_velocity: float | None = None, spring_k: float | None = None,
                    gpu: bool | None = None, params: Mapping[str, tuple[float, float]] | None = None,
                    charges: Sequence[float] | None = None, config: SMDConfig | None = None,
                    title: str = "") -> SMDSetup:
    """Write ``system.data``, ``in.smd``, ``README.txt`` and ``smd_setup.json`` into ``out_dir``.

    ``system_atoms``: tag 0 = surface, tag >= 1 = adsorbate (``place_on_surface`` output). Charges come
    from ``charges`` (whole system or adsorbate only) or else the atoms' initial charges; the surface
    stays neutral unless charges are given for it. ``params`` overrides LJ parameters per type label
    (``"C_surf"``, ``"F"``) or adsorbate element: ``{"F": (0.061, 2.94)}`` (ε kcal/mol, σ Å).
    ``smd_steps`` overrides the step count derived from ``pull_distance / (v dt)`` (for quick tests).
    Keyword arguments override the matching ``config`` fields.
    """
    cfg = config or SMDConfig()
    overrides = {k: v for k, v in dict(temperature=temperature, soft_steps=soft_steps, equil_steps=equil_steps,
                                        pull_distance=pull_distance, pull_velocity=pull_velocity,
                                        spring_k=spring_k, gpu=gpu).items() if v is not None}
    cfg = replace(cfg, **overrides)
    n_smd = int(smd_steps) if smd_steps is not None else cfg.smd_steps
    if cfg.pull_velocity <= 0 or cfg.spring_k <= 0 or cfg.temperature <= 0:
        raise ValueError("pull_velocity, spring_k and temperature must be positive (pulling is along +z)")

    atoms = system_atoms.copy()
    n = len(atoms)
    if n == 0:
        raise ValueError("empty system")
    tags = atoms.get_tags() if atoms.has("tags") else np.zeros(n, dtype=int)
    surface = np.where(tags == TAG_SURFACE)[0]
    ads_idx = np.where(tags != TAG_SURFACE)[0]
    if len(ads_idx) == 0 or len(surface) == 0:
        raise ValueError("need both surface (tag 0) and adsorbate (tag >= 1) atoms; build the system with "
                         "quantabench.build.surfaces.place_on_surface")

    # --- box: orthogonal, periodic in x/y from the ASE cell, padded in z ---
    cell = np.asarray(atoms.cell[:])
    if abs(cell[0, 1]) > 1e-6 or abs(cell[0, 2]) > 1e-6 or abs(cell[1, 0]) > 1e-6 or abs(cell[1, 2]) > 1e-6:
        raise ValueError("the SMD bridge needs an orthogonal x/y cell (graphite_slab builds one); got a "
                         "triclinic cell")
    lx, ly = float(cell[0, 0]), float(cell[1, 1])
    if lx <= 0 or ly <= 0:
        raise ValueError("the system needs a periodic x/y cell (slab repeat lengths)")
    pos = atoms.get_positions()
    image = np.zeros((n, 3), dtype=int)
    image[:, 0] = np.floor(pos[:, 0] / lx).astype(int)
    image[:, 1] = np.floor(pos[:, 1] / ly).astype(int)
    wrapped = pos.copy()
    wrapped[:, 0] -= image[:, 0] * lx
    wrapped[:, 1] -= image[:, 1] * ly

    molecules = _adsorbate_molecules(atoms, ads_idx, tags)
    if not 0 <= cfg.pull_molecule < len(molecules):
        raise ValueError(f"pull_molecule={cfg.pull_molecule} but the system has {len(molecules)} adsorbate molecule(s)")
    # keep each molecule whole: unwrap relative to its first atom so rigid bodies see intact geometry
    for mol in molecules:
        ref = pos[mol[0]]
        for k in mol:
            d = pos[k] - ref
            shift = np.array([lx * round(d[0] / lx), ly * round(d[1] / ly), 0.0])
            if np.any(shift):
                pos[k] = pos[k] - shift
        img = np.floor(pos[mol][:, :2] / [lx, ly]).astype(int)
        image[mol, :2] = img
        wrapped[mol, 0] = pos[mol, 0] - img[:, 0] * lx
        wrapped[mol, 1] = pos[mol, 1] - img[:, 1] * ly

    # --- overlap guard: the cause of "Lost atoms" / exploding starts ---
    notes: list[str] = []
    gap = _mic_min_distance(pos[ads_idx], pos[surface], lx, ly)
    if gap < cfg.min_gap:
        raise ValueError(f"adsorbate-surface contact of {gap:.2f} Å (< {cfg.min_gap} Å): overlapping atoms would "
                         "blow up the first steps. Place the molecule higher.")
    for a in range(len(molecules)):
        for b in range(a + 1, len(molecules)):
            dmin = _mic_min_distance(pos[molecules[a]], pos[molecules[b]], lx, ly)
            if dmin < cfg.min_gap:
                raise ValueError(f"adsorbate molecules {a} and {b} are {dmin:.2f} Å apart (< {cfg.min_gap} Å)")
    notes.append(f"closest adsorbate-surface contact {gap:.2f} Å")

    q = _resolve_charges(atoms, charges, surface, ads_idx)
    if np.any(np.abs(q[surface]) > 1e-12):
        notes.append("surface atoms carry user-supplied charges")
    has_charge = bool(np.any(np.abs(q) > 1e-12))
    net = float(q.sum())
    if has_charge and abs(net) > 1e-3:
        notes.append(f"net charge {net:+.3f} e: PPPM adds a uniform neutralizing background "
                     "(add a counter-ion for a neutral cell)")

    # --- types: surface elements first ("C_surf"), then adsorbate elements in order of appearance ---
    symbols = atoms.get_chemical_symbols()
    labels: list[str] = []
    label_of_atom: list[str] = [""] * n
    for k in surface:
        lab = f"{symbols[k]}_surf"
        label_of_atom[k] = lab
        if lab not in labels:
            labels.append(lab)
    for mol in molecules:
        for k in mol:
            # charged single atoms (counter-ions) get their own type with ion LJ parameters
            lab = f"{symbols[k]}_ion" if len(mol) == 1 and abs(q[k]) >= 0.5 else symbols[k]
            label_of_atom[k] = lab
            if lab not in labels:
                labels.append(lab)
    type_id = {lab: i + 1 for i, lab in enumerate(labels)}
    element_of = {lab: lab.removesuffix("_surf").removesuffix("_ion") for lab in labels}
    surface_labels = {lab for lab in labels if lab.endswith("_surf")}
    lj = P.resolve_type_params(element_of, surface_labels, cfg.graphite, params)

    mol_id = np.full(n, SURFACE_MOL_ID, dtype=int)
    for m, mol in enumerate(molecules):
        mol_id[mol] = SURFACE_MOL_ID + 1 + m

    # --- box z ---
    zmin = float(pos[:, 2].min())
    z_surface_top = float(pos[surface, 2].max())
    zhi = max(float(cell[2, 2]) if cell[2, 2] > pos[:, 2].max() else 0.0,
              float(pos[ads_idx, 2].max()) + cfg.pull_distance + cfg.z_padding)
    box = ((0.0, lx), (0.0, ly), (zmin - cfg.z_below, zhi))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    masses = {lab: float(atomic_masses[atomic_numbers[element_of[lab]]]) for lab in labels}
    data_path = out / "system.data"
    _write_data(data_path, title or "QuantaBench SMD system", labels, masses, box, mol_id,
                [type_id[label_of_atom[k]] for k in range(n)], q, wrapped, image)

    pulled = molecules[cfg.pull_molecule]
    pulled_mass = float(sum(atomic_masses[atoms.numbers[k]] for k in pulled))
    rigid_ids = [SURFACE_MOL_ID + 1 + m for m, mol in enumerate(molecules) if len(mol) > 1]
    ion_ids = [SURFACE_MOL_ID + 1 + m for m, mol in enumerate(molecules) if len(mol) == 1]
    drag = langevin_drag_work(pulled_mass, cfg.pull_velocity, cfg.pull_distance, cfg.pull_damp)
    ctx = dict(cfg=cfg, n_smd=n_smd, labels=labels, lj=lj, has_charge=has_charge, net=net,
               n_mol=len(molecules), rigid_ids=rigid_ids, ion_ids=ion_ids,
               pull_id=SURFACE_MOL_ID + 1 + cfg.pull_molecule, z_tether=z_surface_top, drag=drag,
               pulled_mass=pulled_mass, element_of=element_of)
    in_path = out / "in.smd"
    in_path.write_text(_input_script(ctx), encoding="utf-8")
    readme = out / "README.txt"
    readme.write_text(_readme(ctx, n, len(surface), notes, box), encoding="utf-8")

    meta = out / "smd_setup.json"
    meta.write_text(json.dumps({
        "generator": f"QuantaBench {__version__}", "created": datetime.now().isoformat(timespec="seconds"),
        "config": asdict(cfg), "smd_steps": n_smd, "types": {lab: {"id": type_id[lab], "element": element_of[lab],
                                                                    "mass": masses[lab], "epsilon_kcal_mol": lj[lab].epsilon,
                                                                    "sigma_A": lj[lab].sigma, "source": lj[lab].source}
                                                              for lab in labels},
        "n_atoms": n, "n_surface": int(len(surface)), "molecules": molecules, "net_charge": net,
        "pulled_molecule_mass_g_mol": pulled_mass, "langevin_drag_work_estimate_kcal_mol": drag,
        "langevin_drag_force_kcal_mol_A": drag / cfg.pull_distance if cfg.pull_distance else None,
        "box": box, "z_tether": z_surface_top, "notes": notes,
    }, indent=2) + "\n", encoding="utf-8")
    return SMDSetup(out_dir=out, data=data_path, input=in_path, readme=readme, metadata=meta, config=cfg,
                    type_labels=labels, lj=lj, n_surface=int(len(surface)), molecules=molecules, net_charge=net,
                    box=box, notes=notes)


def _write_data(path: Path, title: str, labels: list[str], masses: dict[str, float], box, mol_id, types, q,
                xyz: np.ndarray, image: np.ndarray) -> None:
    lines = [f"{title} (written by QuantaBench {__version__}; atom_style full, units real, no bonds)", "",
             f"{len(types)} atoms", f"{len(labels)} atom types", "",
             f"{box[0][0]:.8f} {box[0][1]:.8f} xlo xhi", f"{box[1][0]:.8f} {box[1][1]:.8f} ylo yhi",
             f"{box[2][0]:.8f} {box[2][1]:.8f} zlo zhi", "", "Masses", ""]
    lines += [f"{i + 1} {masses[lab]:.5f} # {lab}" for i, lab in enumerate(labels)]
    lines += ["", "Atoms # full", ""]
    for k in range(len(types)):
        x, y, z = xyz[k]
        ix, iy, iz = image[k]
        lines.append(f"{k + 1} {mol_id[k]} {types[k]} {q[k]:.6f} {x:.8f} {y:.8f} {z:.8f} {ix} {iy} {iz}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _ranges(ids: list[int]) -> str:
    return " ".join(str(i) for i in ids)


def _input_script(c: dict) -> str:
    cfg: SMDConfig = c["cfg"]
    labels: list[str] = c["labels"]
    lj: dict[str, P.LJ] = c["lj"]
    L: list[str] = []
    add = L.append
    add(f"# QuantaBench {__version__}: steered MD of a rigid adsorbate pulled off a frozen surface")
    add(f"# Generated {datetime.now():%Y-%m-%d %H:%M}. Model and approximations: see README.txt.")
    add("# Override any 'index' variable from the command line, e.g.  -var smd_steps 50000")
    add("")
    add(f"variable T           index {cfg.temperature:g}          # K")
    add(f"variable seed        index {cfg.seed}")
    add(f"variable soft_steps  index {cfg.soft_steps}")
    add(f"variable equil_steps index {cfg.equil_steps}")
    add(f"variable smd_steps   index {c['n_smd']}")
    add(f"variable vpull       index {cfg.pull_velocity:g}        # Å/fs (1e-4 Å/fs = 10 m/s)")
    add(f"variable kspring     index {cfg.spring_k:g}          # kcal/mol/Å^2")
    add(f"variable thermo_every index {cfg.thermo_every}")
    add(f"variable dump_every  index {cfg.dump_every}")
    add(f"variable pull_every  index {cfg.pull_output_every}")
    add("")
    if cfg.gpu:
        add("# GPU package (OpenCL/CUDA/HIP). Neighbor lists stay on the CPU because GPU-built lists")
        add("# do not support neigh_modify exclude; the pair and PPPM kernels run on the GPU.")
        add(f"package gpu {cfg.gpu_count} neigh no")
        add("suffix gpu")
        add("")
    add("units           real")
    add("atom_style      full")
    add("boundary        p p f")
    add("read_data       system.data")
    add("")
    add("# --- nonbonded model: UFF (adsorbate), graphite carbon preset (surface), geometric mixing ---")
    if c["has_charge"]:
        add(f"pair_style      lj/cut/coul/long {cfg.cutoff:g}")
        add(f"kspace_style    pppm {cfg.kspace_accuracy:g}")
        add(f"kspace_modify   slab {cfg.slab_factor:g}        # Yeh-Berkowitz correction for the p p f slab")
    else:
        add(f"pair_style      lj/cut {cfg.cutoff:g}           # no charges: no long-range electrostatics")
    add("pair_modify     mix geometric")
    for i, lab in enumerate(labels):
        p = lj[lab]
        add(f"pair_coeff      {i + 1} {i + 1} {p.epsilon:.5f} {p.sigma:.5f}    # {lab}: {p.source}")
    add("")
    add("group           surface molecule 1")
    add(f"group           adsorbate molecule 2:{1 + c['n_mol']}")
    if c["rigid_ids"]:
        add(f"group           rigidmols molecule {_ranges(c['rigid_ids'])}")
    if c["ion_ids"]:
        add(f"group           ions molecule {_ranges(c['ion_ids'])}")
    add(f"group           pulled molecule {c['pull_id']}")
    add("")
    add("neighbor        2.0 bin")
    add("neigh_modify    every 1 delay 0 check yes")
    add("neigh_modify    exclude molecule/intra adsorbate   # rigid bodies: internal forces cannot move them")
    add("neigh_modify    exclude molecule/intra surface     # frozen slab: surface-surface pairs are irrelevant")
    add("")
    add("# The surface is never integrated (no fix on it) and starts at rest, so it cannot move.")
    add("velocity        all set 0.0 0.0 0.0")
    add("")
    add("# Temperature of the mobile atoms only; rigid-body constraints are removed automatically,")
    add("# and no COM degrees are subtracted because thermostat and spring do not conserve momentum.")
    add("compute         tAds adsorbate temp")
    add("compute_modify  tAds extra/dof 0")
    add("variable        comz equal xcm(pulled,z)")
    thermo_cols = "step time temp pe ke etotal v_comz"
    if cfg.interaction_energy:
        add("compute         eint pulled group/group surface   # pair (real-space) pulled-surface energy")
        thermo_cols += " c_eint"
    add(f"thermo_style    custom {thermo_cols}")
    add("thermo_modify   temp tAds flush yes")
    add("thermo          ${thermo_every}")
    elements = " ".join(c["element_of"][lab] for lab in labels)
    add("dump            traj all custom ${dump_every} smd.lammpstrj id mol type element q x y z ix iy iz")
    add(f"dump_modify     traj element {elements} sort id")
    add("")

    def integrators(stage: str, damp: float, seed_offset: int, limit: bool) -> None:
        if c["rigid_ids"]:
            add(f"fix             rbody rigidmols rigid/small molecule langevin ${{T}} ${{T}} {damp:g} "
                f"$(v_seed+{seed_offset})")
        if c["ion_ids"]:
            # nve/limit in every stage: at 300 K an ion moves ~0.005 Å/fs, so the cap only bites if a
            # close contact goes wrong, and then it stops the run from blowing up
            add("fix             ionint ions nve/limit " + ("0.05" if limit else "0.1"))
            add(f"fix             ionlgv ions langevin ${{T}} ${{T}} {min(damp, cfg.ion_damp):g} "
                f"$(v_seed+{seed_offset + 100})")

    def unfix_integrators() -> None:
        if c["rigid_ids"]:
            add("unfix           rbody")
        if c["ion_ids"]:
            add("unfix           ionint")
            add("unfix           ionlgv")

    add("# === Stage 1: soft start (short timestep, strong rigid-body Langevin damping) ===")
    add(f"timestep        {cfg.soft_timestep:g}")
    integrators("soft", cfg.soft_damp, 0, True)
    if cfg.viscous_gamma > 0:
        add(f"fix             visc adsorbate viscous {cfg.viscous_gamma:g}")
    add("run             ${soft_steps}")
    if cfg.viscous_gamma > 0:
        add("unfix           visc")
    unfix_integrators()
    add("")
    add(f"# === Stage 2: NVT equilibration at T (rigid-body Langevin, damp {cfg.equil_damp:g} fs) ===")
    add(f"timestep        {cfg.timestep:g}")
    integrators("equil", cfg.equil_damp, 1, False)
    add("run             ${equil_steps}")
    add("write_data      equilibrated.data")
    unfix_integrators()
    add("")
    add("# === Stage 3: constant-velocity SMD of the pulled molecule's COM along +z ===")
    add(f"# Spring tethered to the surface plane z = {c['z_tether']:.4f} Å (x and y free); its end moves away")
    add(f"# at v_pull. Weak thermostat ({cfg.pull_damp:g} fs): Langevin drag adds ~{c['drag']:.2f} kcal/mol "
        "over the pull.")
    integrators("smd", cfg.pull_damp, 2, False)
    add("variable        s0 equal $(step)")
    add("variable        z0 equal $(xcm(pulled,z))")
    add(f"fix             pull pulled smd cvel ${{kspring}} ${{vpull}} tether NULL NULL {c['z_tether']:.6f} 0.0")
    add("variable        tpull equal (step-v_s0)*dt")
    add("variable        dz equal v_comz-v_z0")
    add("variable        fz equal f_pull[3]       # spring force on the COM along z (kcal/mol/Å)")
    add("variable        work equal f_pull[7]     # accumulated external work (kcal/mol)")
    cols = "${tpull} ${comz} ${dz} ${fz} ${work}"
    title = "# time_fs com_z_A displacement_A force_z_kcal_mol_A work_kcal_mol"
    if cfg.interaction_energy:
        add("variable        eint equal c_eint")
        cols += " ${eint}"
        title += " e_int_kcal_mol"
    add(f'fix             pullout all print ${{pull_every}} "{cols}" file pull_force.txt screen no '
        f'title "{title}"')
    add(f"thermo_style    custom {thermo_cols} f_pull[3] f_pull[7]")
    add("thermo_modify   temp tAds flush yes")
    add("run             ${smd_steps}")
    add("write_data      final.data")
    add("")
    return "\n".join(L) + "\n"


def _readme(c: dict, n_atoms: int, n_surface: int, notes: list[str], box) -> str:
    cfg: SMDConfig = c["cfg"]
    lj: dict[str, P.LJ] = c["lj"]
    ps = cfg.timestep * 1e-3
    gpu_line = ("lmp.exe -in in.smd            (this in.smd already contains 'package gpu ... neigh no' and "
                "'suffix gpu')" if cfg.gpu else
                "lmp.exe -sf gpu -pk gpu 1 neigh no -in in.smd")
    lines = [
        f"QuantaBench {__version__} steered-MD setup", "=" * 40, "",
        "WHAT THIS SIMULATES",
        f"  {n_atoms} atoms: {n_surface} frozen surface atoms + {n_atoms - n_surface} adsorbate atoms in "
        f"{c['n_mol']} molecule(s).",
        f"  Molecule {c['pull_id']} (M = {c['pulled_mass']:.2f} g/mol) is pulled along +z by a spring "
        f"(K = {cfg.spring_k:g} kcal/mol/A^2) whose end moves at {cfg.pull_velocity:g} A/fs "
        f"({cfg.pull_velocity * 1e5:g} m/s).",
        f"  Stages: soft start {cfg.soft_steps} x {cfg.soft_timestep:g} fs, NVT {cfg.equil_steps} x "
        f"{cfg.timestep:g} fs ({cfg.equil_steps * ps:g} ps) at {cfg.temperature:g} K, SMD {c['n_smd']} x "
        f"{cfg.timestep:g} fs ({c['n_smd'] * ps:g} ps, {c['n_smd'] * cfg.timestep * cfg.pull_velocity:.2f} A of spring travel).",
        f"  Box: x {box[0][1]:.3f} A, y {box[1][1]:.3f} A (periodic), z {box[2][0]:.2f} .. {box[2][1]:.2f} A "
        "(fixed walls, 'boundary p p f').",
        "",
        "MODEL AND APPROXIMATIONS (read before quoting numbers)",
        "  * Adsorbate = rigid body(ies) (fix rigid/small). Geometry and partial charges come from the",
        "    QuantaBench ML/xTB optimization; no bonded force field is used. Conformational change on the",
        "    surface is not sampled.",
        "  * Surface = frozen: never integrated, zero velocity, uncharged unless charges were supplied, and",
        "    NOT polarizable (no image-charge attraction for charged head groups near graphite).",
        "  * Lennard-Jones 12-6, units real. Adsorbate: UFF (Rappe et al., JACS 1992, 114, 10024),",
        "    eps = D_i, sigma = x_i / 2^(1/6). Graphite: " + P.GRAPHITE_LJ[cfg.graphite][2] + ".",
        "    Geometric mixing (pair_modify mix geometric).",
        (f"  * Long-range Coulomb: PPPM (accuracy {cfg.kspace_accuracy:g}) with kspace_modify slab "
         f"{cfg.slab_factor:g}.")
        if c["has_charge"] else "  * No charges: plain lj/cut, no long-range electrostatics.",
        "  * Vacuum, no solvent: the work is a gas-phase detachment work, not an aqueous free energy.",
        "  * One constant-velocity pull gives W >= dF (Jarzynski, PRL 1997, 78, 2690). For a PMF run",
        "    several pulls (different seeds: -var seed N) and use the Jarzynski average or umbrella sampling.",
        f"  * The weak SMD thermostat (damp {cfg.pull_damp:g} fs) adds a drag work of about {c['drag']:.2f} kcal/mol "
        "over the pull.",
        "",
        "LJ PARAMETERS (epsilon kcal/mol, sigma A)",
    ]
    lines += [f"  type {i + 1:2d} {lab:8s} {lj[lab].epsilon:8.4f} {lj[lab].sigma:8.4f}   {lj[lab].source}"
              for i, lab in enumerate(c["labels"])]
    lines += ["", "NOTES"] + [f"  * {s}" for s in notes] + [
        "",
        "HOW TO RUN",
        "  Windows (native, LAMMPS installer: the binary is lmp.exe; older installers call it lmp_serial.exe):",
        "    cd <this folder>",
        "    lmp.exe -in in.smd",
        "    lmp.exe -sf omp -pk omp 12 -in in.smd          (OpenMP, 12 threads)",
        "    mpiexec -np 12 lmp.exe -in in.smd             (MPI build, 12 ranks)",
        "  Windows GPU (OpenCL GPU package; about 9x over 12 CPU cores on an RX 9060 XT in earlier tests):",
        "    " + gpu_line,
        "    'neigh no' is required: GPU-built neighbor lists do not support the neigh_modify exclude lines.",
        "    If compute group/group complains with the GPU package, regenerate with interaction_energy=False.",
        "  Linux / WSL:",
        "    lmp -in in.smd",
        "    mpirun -np 4 lmp -in in.smd",
        "    lmp -sf gpu -pk gpu 1 neigh no -in in.smd      (GPU build)",
        "  Python (the lammps module): quantabench.md.lammps.run_lammps(<this folder>)",
        "  Shorter test run: lmp -in in.smd -var soft_steps 200 -var equil_steps 500 -var smd_steps 1000",
        "",
        "OUTPUTS",
        "  log.lammps         thermo blocks of the three stages (temp = adsorbate temperature)",
        "  pull_force.txt     time (fs), COM z (A), COM displacement (A), spring force z (kcal/mol/A),",
        "                     accumulated work (kcal/mol)" + (", pulled-surface pair energy (kcal/mol)"
                                                            if cfg.interaction_energy else ""),
        "  smd.lammpstrj      trajectory for OVITO (element column; ix iy iz image flags for unwrapping)",
        "  equilibrated.data, final.data   snapshots for restarts and inspection",
        "  1 kcal/mol/A = 69.48 pN;  1 kcal/mol = 4.184 kJ/mol.",
        "",
        "ANALYSIS",
        "  from quantabench.md.lammps_log import read_pull_force, estimate_detachment, parse_log",
        "  est = estimate_detachment(read_pull_force('pull_force.txt'))",
        "  print(est.summary())",
        "",
        "EXPECTED WARNINGS",
        "  'Neighbor exclusions used with KSpace solver may give inconsistent Coulombic energies':",
        "    harmless here; the excluded pairs are inside rigid bodies or the frozen slab, so the",
        "    inconsistency is a constant energy offset that exerts no net force or torque.",
        "  'Temperature for thermo pressure is not for group all': the thermo temperature is the",
        "    adsorbate's (the frozen slab would dilute it); pressure is meaningless for this system anyway.",
        "  'Fix smd is unmaintained': it still works as documented; fix colvars is the maintained alternative.",
    ]
    if c["has_charge"] and abs(c["net"]) > 1e-3:
        lines.append("  'System is not charge neutral' / 'Using kspace solver on system with net charge': PPPM adds a")
        lines.append(f"    uniform neutralizing background for the {c['net']:+.3f} e; add a counter-ion to avoid it.")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------------------------ running


def run_lammps(setup_dir: str | Path, log_name: str = "log.lammps", variables: Mapping[str, object] | None = None,
               cmdargs: Sequence[str] = (), screen: bool = False) -> Path:
    """Run ``in.smd`` in ``setup_dir`` with the LAMMPS Python module (``import lammps``).

    For long production runs prefer the ``lmp`` executable (README.txt); this is for short checks,
    tests and notebooks. ``variables`` become ``-var name value`` (e.g. ``{"smd_steps": 1000}``).
    Raises whatever the module raises on a LAMMPS ERROR. Returns the log path.
    """
    import lammps  # lazy: optional dependency

    setup_dir = Path(setup_dir).resolve()
    args = ["-nocite", "-log", log_name]
    if not screen:
        args += ["-screen", "none"]
    for k, v in (variables or {}).items():
        args += ["-var", str(k), str(v)]
    args += list(cmdargs)
    cwd = Path.cwd()
    os.chdir(setup_dir)   # in.smd uses relative paths so the folder can be moved/synced as a unit
    try:
        lmp = lammps.lammps(cmdargs=args)
        try:
            lmp.file("in.smd")
        finally:
            lmp.close()
    finally:
        os.chdir(cwd)
    return setup_dir / log_name


def read_data_atoms(path: str | Path) -> dict[str, np.ndarray]:
    """Read the Atoms section of an ``atom_style full`` data file (ours or LAMMPS' ``write_data``).

    Returns arrays sorted by atom id: ``id, mol, type, q, x`` (wrapped, Å), ``image`` and ``unwrapped``.
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    box = {}
    rows = []
    i = 0
    while i < len(lines):
        s = lines[i].split("#")[0].strip()
        for ax in ("x", "y", "z"):
            if s.endswith(f"{ax}lo {ax}hi"):
                lo, hi = map(float, s.split()[:2])
                box[ax] = hi - lo
        if s == "Atoms" or lines[i].strip().startswith("Atoms"):
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            while i < len(lines) and lines[i].strip() and lines[i].strip()[0].isdigit():
                rows.append(lines[i].split("#")[0].split())
                i += 1
            break
        i += 1
    arr = np.array(rows, dtype=float)
    arr = arr[np.argsort(arr[:, 0])]
    x = arr[:, 4:7]
    image = arr[:, 7:10].astype(int) if arr.shape[1] >= 10 else np.zeros((len(arr), 3), dtype=int)
    lengths = np.array([box.get("x", 0.0), box.get("y", 0.0), box.get("z", 0.0)])
    return {"id": arr[:, 0].astype(int), "mol": arr[:, 1].astype(int), "type": arr[:, 2].astype(int),
            "q": arr[:, 3], "x": x, "image": image, "unwrapped": x + image * lengths}
