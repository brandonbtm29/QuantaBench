"""Geometry optimization with ASE optimizers, cancellable."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ase import Atoms
from ase.optimize import FIRE, LBFGS


class Cancelled(Exception):
    """Raised inside an optimization when the user cancels."""


@dataclass
class OptResult:
    energy_ev: float
    converged: bool
    steps: int
    fmax: float


def optimize(atoms: Atoms, fmax: float = 0.03, steps: int = 500, algorithm: str = "lbfgs",
             should_stop: Callable[[], bool] | None = None) -> OptResult:
    opt_cls = FIRE if algorithm == "fire" else LBFGS
    opt = opt_cls(atoms, logfile=None)

    def check():
        if should_stop and should_stop():
            raise Cancelled()

    opt.attach(check, interval=1)
    converged = bool(opt.run(fmax=fmax, steps=steps))
    forces = atoms.get_forces()
    return OptResult(float(atoms.get_potential_energy()), converged, int(opt.nsteps),
                     float((forces**2).sum(axis=1).max() ** 0.5))
