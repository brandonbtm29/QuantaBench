"""QM stage: DFT (or HF) on the lowest conformers with Psi4, run in its own Python environment.

CONTRACT (implemented by the qm builder):

    run_stage(result, cs, cfg, *, threads, should_stop, log) -> None

Fills ``result.qm`` (a JSON-ready dict) and adds ``qm_*`` descriptors to ``result.descriptors``.
Called by ``pipeline.run_molecule`` after refinement when ``cfg.qm_method`` is set; raising is fine (the
pipeline turns it into a note). ``cfg`` fields: ``qm_method`` ("functional/basis", e.g. "wb97x-d/def2-svp",
"b3lyp-d3bj/def2-tzvp", "hf/6-31g*"), ``qm_task`` ("energy" | "optimize"), ``qm_top`` (conformers, lowest
first), ``qm_properties``, ``qm_frequencies`` (best QM conformer only), ``qm_cubes`` (best only), ``solvent``
(PCM via ``solvents.for_engine(s, "pcm")`` when Psi4 has PCMSolver; otherwise note "gas phase").

How Psi4 runs: ``external.find_python_with("psi4", "psi4_python")`` gives an interpreter that can import
psi4 (often a separate conda env). ``chem/psi4_driver.py`` (stdlib + numpy + psi4 only, no quantabench
imports) is run with that interpreter as ``python -I psi4_driver.py task.json`` inside ``external.work_dir``;
it writes ``result.json`` (and cube files). Threads via ``psi4.set_num_threads`` and the environment,
memory from ``settings.engines.qm_memory_gb``, scratch via ``PSI_SCRATCH``.

``result.qm`` keys (QM_RESULT_KEYS): level, program, program_version, task, solvent, conformers (list of
{conf_id, energy_ev, rel_kcal, converged, positions_A (optimize only)}), energy_ev (lowest), properties
(best conformer: dipole_debye, dipole_vector_debye, quadrupole_debye_A, homo_ev, lumo_ev, gap_ev,
charges {mulliken, lowdin, mbis}, mbis_volumes_bohr3, polarizability_iso_A3 when computed), files
({"density": path, "esp": path, "log": path} in a scratch folder: ``export.write_molecule`` copies them next to
the other results), surface (dict from ``surfaces.surface_descriptors`` when cubes were made), timings,
warnings. Frequencies: ``result.vibrations``-compatible dict under ``qm["vibrations"]``.
"""

from __future__ import annotations

from collections.abc import Callable

QM_RESULT_KEYS = ("level", "program", "program_version", "task", "solvent", "conformers", "energy_ev", "properties",
                  "files", "surface", "vibrations", "timings", "warnings")


def available() -> tuple[bool, str]:
    from . import external

    if external.find_python_with("psi4", "psi4_python") is None:
        return False, "Psi4 not found (" + external.PROGRAMS["psi4"].install_hint + ")"
    return True, ""


def run_stage(result, cs, cfg, *, threads: int = 1, should_stop: Callable[[], bool] = lambda: False,
              log: Callable[[str], None] = lambda _m: None) -> None:
    raise NotImplementedError("the QM (Psi4) stage is not implemented yet")
