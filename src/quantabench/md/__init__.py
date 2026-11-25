"""Molecular-dynamics bridges.

``md.lammps`` writes a runnable LAMMPS steered-MD folder (rigid adsorbate, frozen slab, UFF/graphite
Lennard-Jones, PPPM) from a QuantaBench surface system; ``md.lammps_log`` reads ``log.lammps`` and
``pull_force.txt`` back and estimates the detachment work; ``md.params`` holds the LJ tables.

Liquids (MD-2): ``md.forcefield`` types a molecule (``md.oplsaa``: OPLS-AA with the Watkins-Jorgensen
perfluoroalkane types; GAFF2 and OpenFF Sage through ``md/ff_driver.py`` in an AmberTools environment),
``md.liquid`` writes a liquid-vapour slab folder (bulk NPT, then an NVT slab), and ``md.surface_tension``
turns its pressure tensor and density profiles into σ ± error and densities.

``md.long_range`` (MD-LR) holds the long-range options of a liquid slab (full long-range dispersion or an LJ
cutoff + tail, the cutoff or "screening length", the k-space accuracies), their slab-safe defaults, and
``recommend_long_range``, which picks them for a setup (geometry, box, size, machine, purpose) from the pilot runs.
It is pure Python, so the CLI, ``md.liquid`` and the GUI import it freely.

``md.bench`` (MD-BENCH-1) times this machine's LAMMPS on a liquid C6F14 box per accelerator and dispersion variant
(``machines --bench md``), checks GPU variants against the CPU at ``run 0``, and estimates MD run times from that.

MD runs as jobs (SIM-1): ``md.sim`` is the run of a written folder as data (``run.json``: system, parameterization,
protocol with its parts and stages, segments), with the checkpoint and output-cut rules; ``md.supervisor`` runs a
folder's parts as segments, one LAMMPS process at a time, with pause through a STOP file and resume from a checkpoint
(``quantabench-cli md run/status/pause/resume``); ``md.telemetry`` reads the live progress from the engine's own log.
All three are pure Python.

OpenMM as a second engine (OPENMM-1): ``md.openmm_engine`` builds a folder's LAMMPS model as an OpenMM System (and
finds the OpenMM environment, ``envs/openmm.yml``), ``md.openmm_run`` is the process the supervisor starts for an
OpenMM segment (in that environment), ``md.openmm_sigma`` evaluates its saved frames (``md.dcd``) in double precision
for σ. Only ``build_system`` and the run itself import ``openmm``.

None of them import the LAMMPS Python module (or OpenFF, ParmEd, OpenMM) at import time.
"""

from .long_range import recommend_long_range

__all__ = ["recommend_long_range"]
