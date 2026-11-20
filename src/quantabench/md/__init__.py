"""Molecular-dynamics bridges.

``md.lammps`` writes a runnable LAMMPS steered-MD folder (rigid adsorbate, frozen slab, UFF/graphite
Lennard-Jones, PPPM) from a QuantaBench surface system; ``md.lammps_log`` reads ``log.lammps`` and
``pull_force.txt`` back and estimates the detachment work; ``md.params`` holds the LJ tables.

Liquids (MD-2): ``md.forcefield`` types a molecule (``md.oplsaa``: OPLS-AA with the Watkins-Jorgensen
perfluoroalkane types; GAFF2 and OpenFF Sage through ``md/ff_driver.py`` in an AmberTools environment),
``md.liquid`` writes a liquid-vapour slab folder (bulk NPT, then an NVT slab), and ``md.surface_tension``
turns its pressure tensor and density profiles into σ ± error and densities.

``md.bench`` (MD-BENCH-1) times this machine's LAMMPS on a liquid C6F14 box per accelerator and dispersion variant
(``machines --bench md``), checks GPU variants against the CPU at ``run 0``, and estimates MD run times from that.

None of them import the LAMMPS Python module (or OpenFF, ParmEd) at import time.
"""
