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
"""

from __future__ import annotations

from collections.abc import Callable

from rdkit import Chem

from .conformers import ConformerSet


def available() -> tuple[bool, str]:
    from . import external

    info = external.find_program("crest")
    if info is None:
        return False, "CREST not found (" + external.PROGRAMS["crest"].install_hint + ")"
    return True, ""


def generate(mol: Chem.Mol, cfg, *, threads: int = 1, solvent: str = "", charge: int = 0, multiplicity: int = 1,
             should_stop: Callable[[], bool] = lambda: False,
             log: Callable[[str], None] = lambda _m: None) -> ConformerSet:
    raise NotImplementedError("CREST conformer search is not implemented yet")
