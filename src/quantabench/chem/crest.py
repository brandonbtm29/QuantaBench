"""CREST conformer search (iMTD-GC metadynamics with GFN2-xTB / GFN-FF), run as an external program.

CONTRACT (implemented by the engines-xtb builder):

    generate(mol, cfg, *, threads, solvent="", charge=0, multiplicity=1, should_stop, log) -> ConformerSet

- ``mol``: RDKit molecule with explicit hydrogens (no conformers needed). Embed one ETKDGv3 geometry,
  MMFF-optimize it, write ``input.xyz`` in RDKit atom order, run CREST in ``external.work_dir``.
- Options from ``cfg`` (``pipeline.BuildConfig``): ``crest_method`` ("gfn2" | "gfnff" | "gfn2//gfnff" |
  "gfn1"), ``crest_quick`` (``--quick``), ``energy_window_kcal`` (``--ewin``), solvent via ``--alpb``
  (``solvents.for_engine(solvent, "alpb")``), charge ``--chrg``, unpaired electrons ``--uhf``, ``-T threads``.
- Parse ``crest_conformers.xyz`` (energies in Eh on the comment lines); keep RDKit atom order (CREST keeps
  input order; verify element sequence and raise ExternalProgramError if it differs); add the geometries as
  conformers of a copy of ``mol``; return ``ConformerSet(mol, conf_ids sorted by energy, energies_kcal
  relative or absolute (kcal/mol), forcefield=f"CREST/{method}", n_embedded=number of CREST conformers)``.
- Topology changes: CREST can find tautomers/reactive structures; drop conformers whose bond graph (from
  ``rdDetermineBonds`` or distance-based connectivity) differs from ``mol`` and note it in ``log``.
- Cancel through ``should_stop`` (passed to ``external.run_program``).

Implementation notes:

- ``energies_kcal`` are absolute CREST energies (Eh x 627.509) at the CREST level, lowest first. At most
  ``cfg.max_conformers`` are kept (the lowest ones).
- Connectivity: a pair is bonded when its distance is below 1.25 x the sum of RDKit's covalent radii.
  The reference is ``mol``'s bond graph (or the MMFF input geometry's distance graph if that does not
  reproduce ``mol``'s bonds, e.g. for unusual valences). Atoms without bonds in ``mol`` (counter-ions
  such as K+ or Na+) are left out of the comparison, so an ion moving around is not a "topology change".
- CREST 3 does its xtb work in-process (tblite/gfnff), except in ``--legacy`` mode, which calls the
  ``xtb`` binary. CREST 3.0.2 only offers ``gfn2//gfnff`` there ("not yet available with new
  calculator"), so that method adds ``--legacy`` and needs xtb. The folder of the xtb QuantaBench found
  (if native, not WSL) is put first on the child's PATH, so CREST uses that xtb and not an older one
  elsewhere on PATH.
- Windows: CREST has no Windows build. Configure it as ``wsl:crest`` (Settings > Engines); the command
  is built with ``ProgramInfo.command()`` and relative file names, and ``run_program`` translates the
  job folder to ``/mnt/c/...``. CREST then uses the xtb inside WSL (install both in the same conda env).
- Run time grows quickly with size and flexibility: GFN-FF ``--quick`` on ethanol takes ~2 s, GFN2
  ``--quick`` on PFBA ~2 min on 2 cores; a C8 perfluoroalkyl acid with GFN2 is tens of minutes.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
from rdkit import Chem

from .conformers import ConformerSet

KCAL_PER_HARTREE = 627.509474
BOND_SCALE = 1.25

#: cfg.crest_method -> CREST flag
METHOD_FLAGS = {"gfn2": "--gfn2", "gfn1": "--gfn1", "gfnff": "--gfnff", "gff": "--gfnff",
                "gfn2//gfnff": "--gfn2//gfnff"}
#: CREST 3.0.2 stops with "-gfn2//gfnff option not yet available with new calculator": these methods
#: need ``--legacy``, which runs the xtb binary for every calculation instead of CREST's built-in tblite
LEGACY_METHODS = {"gfn2//gfnff"}


def available() -> tuple[bool, str]:
    from . import external

    info = external.find_program("crest")
    if info is None:
        return False, "CREST not found (" + external.PROGRAMS["crest"].install_hint + ")"
    return True, ""


# ----------------------------------------------------------------------------- helpers
def _method(cfg) -> str:
    method = str(getattr(cfg, "crest_method", "gfn2") or "gfn2").strip().lower()
    if method not in METHOD_FLAGS:
        raise ValueError(f"crest_method must be one of gfn2, gfnff, gfn2//gfnff, gfn1, not {method!r}")
    return "gfnff" if method == "gff" else method


def command(program, cfg, *, threads: int = 1, solvent: str = "", charge: int = 0, multiplicity: int = 1,
            input_name: str = "input.xyz") -> list[str]:
    """CREST command line (``program`` is an ``external.ProgramInfo``; file names relative to the job folder)."""
    from . import solvents

    method = _method(cfg)
    cmd = [*program.command(), input_name, METHOD_FLAGS[method]]
    if method in LEGACY_METHODS:
        cmd.append("--legacy")
    if getattr(cfg, "crest_quick", True):
        cmd.append("--quick")
    cmd += ["--ewin", f"{float(getattr(cfg, 'energy_window_kcal', 6.0)):g}"]
    if solvent:
        kw = solvents.for_engine(solvent, "alpb")
        if kw is None:
            raise ValueError(f"CREST (ALPB) has no parameters for {solvent}")
        cmd += ["--alpb", str(kw)]
    cmd += ["--chrg", str(int(charge)), "--uhf", str(max(0, int(multiplicity) - 1)), "-T", str(max(1, int(threads)))]
    return cmd


def child_env(program, threads: int) -> dict[str, str]:
    """Thread caps etc. for CREST, with the found xtb's folder first on PATH (native runs only)."""
    from . import external

    extra: dict[str, str] = {}
    if not program.via_wsl:
        xtb = external.find_program("xtb")
        if xtb is not None and not xtb.via_wsl:
            folder = str(Path(xtb.path).parent)
            extra["PATH"] = os.pathsep.join([folder, os.environ.get("PATH", "")])
    return external.program_env(threads, extra, via_wsl=program.via_wsl)


def parse_conformers(text: str) -> list[tuple[list[str], np.ndarray, float]]:
    """``crest_conformers.xyz`` -> [(symbols, positions Å, energy Eh)] in file order."""
    from .eng_xtb import read_xyz_frames

    out = []
    for symbols, pos, comment in read_xyz_frames(text):
        tokens = comment.split()
        energy = None
        for tok in tokens:
            try:
                energy = float(tok)
                break
            except ValueError:
                continue
        if energy is None:
            raise ValueError(f"no energy on the CREST comment line {comment!r}")
        out.append((symbols, pos, energy))
    return out


def _radii(mol: Chem.Mol) -> np.ndarray:
    pt = Chem.GetPeriodicTable()
    return np.array([pt.GetRcovalent(a.GetAtomicNum()) for a in mol.GetAtoms()], dtype=float)


def distance_bonds(positions: np.ndarray, radii: np.ndarray, atoms: set[int] | None = None,
                   scale: float = BOND_SCALE) -> set[tuple[int, int]]:
    """Bonded pairs (i < j) by distance: d < scale * (r_i + r_j), restricted to ``atoms`` if given."""
    pos = np.asarray(positions, dtype=float)
    d = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1))
    cut = scale * (radii[:, None] + radii[None, :])
    i, j = np.nonzero(np.triu(d < cut, k=1))
    pairs = {(int(a), int(b)) for a, b in zip(i, j)}
    if atoms is not None:
        pairs = {p for p in pairs if p[0] in atoms and p[1] in atoms}
    return pairs


def mol_bonds(mol: Chem.Mol, atoms: set[int] | None = None) -> set[tuple[int, int]]:
    pairs = {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in mol.GetBonds()}
    if atoms is not None:
        pairs = {p for p in pairs if p[0] in atoms and p[1] in atoms}
    return pairs


def _embed_input(mol: Chem.Mol, seed: int, threads: int) -> tuple[Chem.Mol, str]:
    from .conformers import embed, forcefield_optimize

    work = Chem.Mol(mol)
    work.RemoveAllConformers()
    embed(work, 1, seed=seed, threads=threads)
    _e, ff = forcefield_optimize(work, threads=threads)
    return work, ff


# ----------------------------------------------------------------------------- the search
def generate(mol: Chem.Mol, cfg, *, threads: int = 1, solvent: str = "", charge: int = 0, multiplicity: int = 1,
             should_stop: Callable[[], bool] = lambda: False,
             log: Callable[[str], None] = lambda _m: None) -> ConformerSet:
    from . import external
    from .eng_xtb import write_xyz
    from .optimize import Cancelled

    method = _method(cfg)
    program = external.require_program("crest")
    if method in LEGACY_METHODS and not program.via_wsl:
        xtb = external.find_program("xtb")
        if xtb is None or xtb.via_wsl:
            raise external.ExternalProgramError(
                f"CREST {method} runs in legacy mode, which needs the xtb program next to CREST "
                f"({external.PROGRAMS['xtb'].install_hint})")
    mol = Chem.Mol(mol)
    if mol.GetNumAtoms() == Chem.RemoveHs(mol).GetNumAtoms() and any(a.GetTotalNumHs() for a in mol.GetAtoms()):
        mol = Chem.AddHs(mol)
    start, ff = _embed_input(mol, int(getattr(cfg, "seed", 42)), threads)
    numbers = [a.GetAtomicNum() for a in mol.GetAtoms()]
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    cmd = command(program, cfg, threads=threads, solvent=solvent, charge=charge, multiplicity=multiplicity)
    env = child_env(program, threads)
    timeout = float(getattr(external._engine_settings(), "timeout_s", 0.0) or 0.0) or None
    if should_stop():
        raise Cancelled()
    log(f"  CREST {method}{' --quick' if getattr(cfg, 'crest_quick', True) else ''} "
        f"(start: ETKDGv3 + {ff}){' in ' + solvent if solvent else ''}, {threads} thread(s)")
    with external.work_dir("crest") as d:
        write_xyz(d / "input.xyz", numbers, start.GetConformer().GetPositions(), "QuantaBench CREST input")
        run = external.run_program(cmd, d, threads=threads, env=env, timeout=timeout, log_name="crest.log",
                                   should_stop=should_stop, log=log)
        out_file = d / "crest_conformers.xyz"
        if not out_file.is_file():
            raise external.ExternalProgramError(f"CREST wrote no crest_conformers.xyz\n{run.tail(30)}")
        frames = parse_conformers(out_file.read_text(encoding="utf-8", errors="replace"))
        if not frames:
            raise external.ExternalProgramError(f"crest_conformers.xyz is empty\n{run.tail(30)}")
        for k, (syms, _pos, _e) in enumerate(frames):
            if syms != symbols:
                raise external.ExternalProgramError(
                    f"CREST conformer {k + 1} has a different atom order than the input "
                    f"({''.join(syms[:12])}... vs {''.join(symbols[:12])}...)")
    log(f"  CREST: {len(frames)} conformers in {run.seconds:.1f} s")

    # topology check against mol's bond graph (counter-ions left out)
    bonded_atoms = {a.GetIdx() for a in mol.GetAtoms() if a.GetDegree() > 0}
    radii = _radii(mol)
    reference = mol_bonds(mol, bonded_atoms)
    start_graph = distance_bonds(start.GetConformer().GetPositions(), radii, bonded_atoms)
    if start_graph != reference:
        log("  CREST: the distance criterion does not reproduce the input bonds; comparing with the "
            "input geometry instead")
        reference = start_graph
    kept, dropped = [], 0
    for _syms, pos, e in frames:
        if distance_bonds(pos, radii, bonded_atoms) == reference:
            kept.append((pos, e))
        else:
            dropped += 1
    if dropped:
        log(f"  CREST: dropped {dropped} of {len(frames)} conformers whose bonds changed "
            "(tautomers or reactive structures)")
    if not kept:
        raise external.ExternalProgramError(f"all {len(frames)} CREST conformers changed the bonding of the input")
    kept.sort(key=lambda t: t[1])
    cap = int(getattr(cfg, "max_conformers", 0) or 0)
    if cap and len(kept) > cap:
        log(f"  CREST: keeping the lowest {cap} of {len(kept)} conformers (max_conformers)")
        kept = kept[:cap]

    out = Chem.Mol(mol)
    out.RemoveAllConformers()
    ids, energies = [], []
    for pos, e in kept:
        conf = Chem.Conformer(out.GetNumAtoms())
        for i, p in enumerate(pos):
            conf.SetAtomPosition(i, p.tolist())
        ids.append(out.AddConformer(conf, assignId=True))
        energies.append(e * KCAL_PER_HARTREE)
    return ConformerSet(mol=out, conf_ids=ids, energies_kcal=energies, forcefield=f"CREST/{method}",
                        n_embedded=len(frames))
