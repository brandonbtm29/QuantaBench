"""Molecular-dynamics bridges.

``md.lammps`` writes a runnable LAMMPS steered-MD folder (rigid adsorbate, frozen slab, UFF/graphite
Lennard-Jones, PPPM) from a QuantaBench surface system; ``md.lammps_log`` reads ``log.lammps`` and
``pull_force.txt`` back and estimates the detachment work; ``md.params`` holds the LJ tables. None of
them import the LAMMPS Python module at import time.
"""
