"""dxtb's libcint backend (FORK-1): the status and reason behind ``engines.has_libcint()``, and, where the
backend runs, GFN2 / polarizability / IR numbers that agree between the PyPI stack (dxtb 0.4, tad-mctc 0.7,
tad-libcint 0.3, Linux) and the dxtb fork (brandonbtm29/dxtb@qb-v0.4.0.post1 with tad-mctc 0.11 and
tad-libcint 0.4, Linux/Windows/macOS)."""

import importlib.util
import sys

import numpy as np
import pytest
from ase.build import molecule

from quantabench.chem import engines


@pytest.fixture(autouse=True)
def fresh_status(monkeypatch):
    monkeypatch.setattr(engines, "_LIBCINT_STATUS", None)


def _versions(monkeypatch, **dists):
    real = engines._dist_version
    monkeypatch.setattr(engines, "_dist_version", lambda name: dists.get(name, real(name)))


def _fork(monkeypatch, present: bool):
    real = importlib.util.find_spec

    def find_spec(name, *args, **kwargs):
        if name == "dxtb._src.exlibs.mctc_compat":
            return object() if present else None
        if name == "dxtb" and present:
            return real(name) or object()
        return real(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)


def test_status_is_consistent_and_cached():
    ok, reason = engines.libcint_status()
    assert ok is engines.has_libcint()
    assert (reason == "") is ok
    assert engines.libcint_status() is engines.libcint_status()


def test_missing_tad_libcint_names_the_install_line(monkeypatch):
    _versions(monkeypatch, **{"tad-libcint": "", "dxtb": "0.4.0"})
    _fork(monkeypatch, present=False)
    ok, reason = engines.libcint_status()
    assert not ok and reason.startswith("tad-libcint is not installed")
    assert engines.DXTB_FORK_REQ in reason and "tad-libcint>=0.4" in reason
    assert ("tad-libcint<0.4" in reason) is sys.platform.startswith("linux")   # PyPI route: Linux only


def test_missing_tad_libcint_with_the_fork(monkeypatch):
    _versions(monkeypatch, **{"tad-libcint": "", "dxtb": "0.4.0.post1"})
    _fork(monkeypatch, present=True)
    ok, reason = engines.libcint_status()
    assert not ok and reason == 'tad-libcint is not installed: pip install "tad-libcint>=0.4"'


def test_installed_but_incompatible_points_to_the_fork(monkeypatch):
    # PyPI dxtb 0.4 pins tad-mctc 0.7; tad-libcint 0.4 then fails to import (tad_mctc.tree is missing)
    _versions(monkeypatch, **{"tad-libcint": "0.4.0", "tad-mctc": "0.7.0", "dxtb": "0.4.0"})
    monkeypatch.setitem(sys.modules, "tad_libcint", None)    # makes `import tad_libcint` raise ImportError
    ok, reason = engines.libcint_status()
    assert not ok and not engines.has_libcint()
    assert reason.startswith("tad-libcint 0.4.0 is installed but does not import with tad-mctc 0.7.0")
    assert "tad-libcint 0.4 needs dxtb from brandonbtm29/dxtb@qb-v0.4.0.post1" in reason
    assert engines.LIBCINT_FORK_INSTALL in reason
    if engines.has_package("torch") and engines.has_package("dxtb"):
        ok2, why = engines.available("gfn2-xtb")
        assert not ok2 and why == f"GFN2 needs dxtb's libcint backend: {reason}"


def test_fork_install_line_is_documented():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    assert engines.DXTB_FORK_REQ == "dxtb @ git+https://github.com/brandonbtm29/dxtb@qb-v0.4.0.post1"
    for doc in ("docs/INSTALL.md", "docs/ENGINES.md"):
        assert engines.DXTB_FORK_REQ in (root / doc).read_text(encoding="utf-8"), doc


# Water (ase.build.molecule) at its fixed geometry. Reference values from both stacks on Linux:
# PyPI dxtb 0.4.0 + tad-mctc 0.7.0 + tad-libcint 0.3.0, and the fork 0.4.0.post1 + tad-mctc 0.11.3 +
# tad-libcint 0.4.0. They differ by ~1e-8 (relative) only, from tad-mctc's CODATA 2014 -> 2018 constants.
WATER = {"gfn2_energy_eV": -137.9677715, "gfn1_dipole_debye": 1.90633672, "gfn1_alpha_iso_A3": 0.28986969,
         "gfn2_freq_cm1": [1587.53000, 3511.74941, 3526.71305], "gfn2_ir_km_mol": [128.764650, 13.8141574, 5.1422068]}


@pytest.mark.skipif(not engines.has_libcint(), reason=engines.libcint_status()[1] or "libcint backend unavailable")
def test_gfn2_polarizability_and_ir_match_both_stacks():
    from quantabench.chem import properties
    from quantabench.chem import vibrations as vib

    assert engines.available("gfn2-xtb") == (True, "")
    atoms = molecule("H2O")
    a = atoms.copy()
    a.calc = engines.make_calculator("gfn2-xtb", atoms.numbers, 0, 1, "cpu")
    assert a.get_potential_energy() == pytest.approx(WATER["gfn2_energy_eV"], abs=5e-6)

    p = properties.xtb_properties(atoms.numbers, atoms.positions, atoms.get_masses())
    assert "_notes" not in p
    assert p["xtb_dipole_debye"] == pytest.approx(WATER["gfn1_dipole_debye"], abs=1e-6)
    assert p["xtb_polarizability_iso_A3"] == pytest.approx(WATER["gfn1_alpha_iso_A3"], abs=1e-6)

    r = vib.analyze(atoms.copy(), "gfn2-xtb", symmetry_number=1, hessian="analytic")
    assert r.ir_intensities_km_mol is not None and "analytic" in r.ir_source
    assert np.allclose(r.frequencies_cm1, WATER["gfn2_freq_cm1"], atol=1e-3)
    assert np.allclose(r.ir_intensities_km_mol, WATER["gfn2_ir_km_mol"], atol=1e-4)
