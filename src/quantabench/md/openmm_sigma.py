"""σ of an OpenMM liquid-slab run in double precision (OPENMM-1): a rerun of its saved frames.

An OpenMM run on an Apple GPU integrates in single precision, and the pressure tensor such a GPU gives by finite
differences is off by up to ~80 bar per sample (19 mN/m in one sample's σ: Runs/mac-gpu-2026-11-23). So the
pressure tensor σ comes from is not taken from the run: ``rerun`` evaluates the run's saved frames (``slab.dcd``)
again on the CPU in double precision and writes ``pressure_tensor.txt`` (step, Pxx, Pyy, Pzz in atm), the file
``md surface-tension`` reads for a LAMMPS run, with the Kirkwood-Buff estimator unchanged:

``lammps`` (where QuantaBench finds an ``lmp``; the default)
    LAMMPS ``rerun`` of the frames with the run's own model (``in.rerun``: the slab's styles, ``slab_start.data``,
    ``ff.params``, its k-space), ``compute pressure NULL virial``: what ``in.slab`` averages, minus the kinetic part,
    with the Coulomb k-space at least as accurate as ``RERUN_KSPACE_ACCURACY`` (1e-6; the cutoff variants' 1e-5 biases
    σ by a few tenths of a mN/m). Frames go to LAMMPS as text dumps in chunks of ``CHUNK`` frames, deleted afterwards.
``openmm`` / ``openmm-reference``
    OpenMM's CPU platform (single-precision arithmetic with double accumulation; it agrees with the Reference
    platform's double precision to ~0.1 bar here) or its Reference platform (double, slow): each diagonal component
    as a central difference of the energy over an atomic strain of that axis, with the cutoff shift of
    ``md.openmm_engine.add_cutoff_shift`` (the mechanical virial, as LAMMPS'). Runs in the OpenMM environment.

The kinetic part of the pressure is the same in the three directions on average and cancels in σ; the file adds
its equipartition value (N − 1)·kT/V to each diagonal component, so the averages printed match a LAMMPS run's.

How many frames: σ's samples are nearly uncorrelated beyond ~20 fs (bond vibrations dominate the virial), so the
error bar shrinks with the number of frames. Pilot B's own 10-fs series, thinned (``thinning_table``), gives 0.30
mN/m from every 10 fs, 0.38 every 50 fs, 0.81 every 100 fs and 2.9 every 1 ps over 10 ns. Frames every 1 ps (117 KB
each for 9,780 atoms) are cheap to keep but 10 times less precise than pilot B; the run's on-the-fly samples
(``sigma_kb.txt``, every 20 fs by default, ``md.openmm_run``) carry the precision, and this rerun checks them for
bias at the steps both have (``compare_series``).
"""

from __future__ import annotations

import contextlib
import math
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from . import dcd

CHUNK = 200                                        # frames per LAMMPS rerun process
#: the rerun's Coulomb k-space accuracy: never coarser than this. At the cutoff variants' 1e-5, LAMMPS' PPPM virial
#: was off by +0.36 mN/m in σ on a frame of pilot B (C6F14, 16 Å) and +0.9 mN/m on a small hexane slab (8 Å), against
#: 1e-6 / 1e-7 (within 0.01 of each other) and OpenMM's PME, which agree; 1e-6 is converged there.
RERUN_KSPACE_ACCURACY = 1e-6
KCAL_MOL_A3_TO_ATM = 4184.0 / 6.02214076e23 / 1e-30 / 101325.0
BOLTZ_KCAL = 0.0019872043
PT_FILE = "pressure_tensor.txt"
FRAMES = "slab.dcd"
METHODS = ("auto", "lammps", "openmm", "openmm-reference", "none")


def _noop(_msg: str) -> None:
    pass


def is_openmm_run(folder: str | Path) -> bool:
    folder = Path(folder)
    try:
        import json

        return json.loads((folder / "run.json").read_text(encoding="utf-8")).get("engine") == "openmm"
    except (OSError, ValueError):
        return (folder / FRAMES).is_file() and (folder / "slab_start.data").is_file()


def done_steps(folder: str | Path) -> list[int]:
    p = Path(folder) / PT_FILE
    if not p.is_file():
        return []
    out = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.split()
        if s and s[0].lstrip("-").isdigit():
            out.append(int(s[0]))
    return out


def pending_frames(folder: str | Path) -> int:
    folder = Path(folder)
    if not (folder / FRAMES).is_file():
        return 0
    have = set(done_steps(folder))
    return sum(1 for s in dcd.frame_steps(folder / FRAMES) if s not in have)


def _header(folder: Path, how: str) -> None:
    p = folder / PT_FILE
    if not p.is_file() or p.stat().st_size == 0:
        p.write_text(f"# Pressure tensor of the saved frames ({FRAMES}), {how}: virial in double precision plus the "
                     "equipartition kinetic part (N-1)kT/V on each diagonal component (QuantaBench OpenMM engine)\n"
                     "# TimeStep c_thermo_press[1] c_thermo_press[2] c_thermo_press[3]\n", encoding="utf-8")


def _kinetic_atm(n_atoms: int, T: float, box: Sequence[float]) -> float:
    return (n_atoms - 1) * BOLTZ_KCAL * T / float(np.prod(box)) * KCAL_MOL_A3_TO_ATM


def _cut_after(folder: Path, step: int) -> None:
    """Drop rows past ``step`` (a rerun of a run that was cut back on resume)."""
    from . import sim

    sim.cut_at(folder / PT_FILE, "ave/time", step, keep_step=True)


def rerun(folder: str | Path, method: str = "auto", *, threads: int | None = None,
          echo: Callable[[str], None] = _noop, lammps: str | None = None,
          kspace_accuracy: float | None = None) -> dict:
    """Evaluate the frames not yet in ``pressure_tensor.txt`` (``method``: see the module docstring) and append
    them. Returns {"method", "frames", "done", "seconds"}; "method" is "none" when nothing was to do."""
    import time

    folder = Path(folder)
    if method not in METHODS:
        raise ValueError(f"rerun method {method!r}: {', '.join(METHODS)}")
    if not (folder / FRAMES).is_file():
        raise FileNotFoundError(f"{folder}: no {FRAMES} (the run saved no frames: frame_every 0?)")
    steps = dcd.frame_steps(folder / FRAMES)
    if steps:
        _cut_after(folder, steps[-1])
    have = set(done_steps(folder))
    todo = [s for s in steps if s not in have]
    if not todo or method == "none":
        return {"method": "none", "frames": 0, "done": len(have), "seconds": 0.0}
    if method == "auto":
        from . import bench

        info, _why = bench.find_lammps() if lammps is None else (lammps, "")
        method = "lammps" if info else "openmm"
    t0 = time.time()
    if method == "lammps":
        _rerun_lammps(folder, todo, threads=threads, echo=echo, lammps=lammps, kspace_accuracy=kspace_accuracy)
    else:
        _rerun_openmm_process(folder, "Reference" if method == "openmm-reference" else "CPU", threads=threads,
                              echo=echo)
    _sort_rows(folder)
    return {"method": method, "frames": len(todo), "done": len(done_steps(folder)), "seconds": time.time() - t0}


def _sort_rows(folder: Path) -> None:
    p = folder / PT_FILE
    lines = p.read_text(encoding="utf-8").splitlines()
    head = [x for x in lines if x.startswith("#")]
    rows = {}
    for x in lines:
        if x and not x.startswith("#"):
            rows[int(x.split()[0])] = x
    p.write_text("\n".join([*head, *(rows[k] for k in sorted(rows))]) + "\n", encoding="utf-8")


# ------------------------------------------------------------------------------------------- LAMMPS
def rerun_input(folder: str | Path, dump_name: str) -> str:
    """The LAMMPS input that evaluates the frames of ``dump_name`` with the slab's own model."""
    from . import liquid as L

    tm, cfg, _meta = L.load_folder(folder)
    lines = ["# QuantaBench (OPENMM-1): the virial pressure tensor of an OpenMM run's frames, in double precision",
             *L._variables(cfg), *L._styles(tm, cfg, "slab"), "read_data       slab_start.data",
             "include         ff.params", *L._kspace(tm, cfg),
             "neighbor        2.0 bin", "neigh_modify    delay 0 every 1 check yes",
             "compute         qb_pvir all pressure NULL virial",
             "thermo_style    custom step c_qb_pvir[1] c_qb_pvir[2] c_qb_pvir[3] pe",
             "thermo_modify   format float %.12g norm no",
             "thermo          1",
             f"rerun           {dump_name} dump x y z box yes"]
    return "\n".join(lines) + "\n"


def _write_dump(path: Path, frames, ids: np.ndarray) -> None:
    with open(path, "w", encoding="ascii") as fh:
        for step, x, box in frames:
            fh.write(f"ITEM: TIMESTEP\n{step}\nITEM: NUMBER OF ATOMS\n{len(ids)}\nITEM: BOX BOUNDS pp pp pp\n"
                     f"0 {box[0]:.8f}\n0 {box[1]:.8f}\n0 {box[2]:.8f}\nITEM: ATOMS id x y z\n")
            np.savetxt(fh, np.column_stack([ids, x]), fmt=["%d", "%.5f", "%.5f", "%.5f"])


def _thermo_rows(log: Path) -> list[list[float]]:
    rows, on = [], False
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        w = line.split()
        if w[:1] == ["Step"]:
            on = True
            continue
        if line.startswith("Loop time"):
            on = False
        if on and len(w) == 5 and w[0].isdigit():
            try:
                rows.append([float(v) for v in w])
            except ValueError:
                continue
    return rows


def _rerun_lammps(folder: Path, todo: list[int], *, threads: int | None, echo, lammps: str | None,
                  kspace_accuracy: float | None = None) -> None:
    from ..chem import external
    from . import bench
    from . import openmm_engine as OE

    info = bench.find_lammps()[0] if lammps is None else None
    exe = lammps or (info.path if info else None)
    if not exe:
        raise RuntimeError("no LAMMPS found for the rerun (or use --rerun openmm)")
    n = max(1, int(threads or 2))
    args = ["-sf", "omp", "-pk", "omp", str(n)] if info is not None and info.has("OPENMP") else []
    model = OE.load_model(folder / "slab_start.data", folder / "ff.params")
    cfg = OE.read_setup(folder).get("config") or {}
    T = float(cfg.get("temperature", 298.15))
    kacc = (float(kspace_accuracy) if kspace_accuracy else
            min(float(cfg.get("kspace_accuracy") or 1e-5), RERUN_KSPACE_ACCURACY))
    args += ["-var", "kacc", f"{kacc:g}"]
    _header(folder, f"LAMMPS rerun, Coulomb k-space accuracy {kacc:g}")
    want = set(todo)
    frames = [f for f in dcd.read_frames(folder / FRAMES) if f[0] in want]
    work = folder / "rerun.tmp"
    work.mkdir(exist_ok=True)
    (work / "in.rerun").write_text(rerun_input(folder, "frames.lammpstrj"), encoding="utf-8")
    for name in ("slab_start.data", "ff.params"):
        (work / name).write_bytes((folder / name).read_bytes())
    boxes = {s: b for s, _x, b in frames}
    try:
        for k in range(0, len(frames), CHUNK):
            part = frames[k:k + CHUNK]
            _write_dump(work / "frames.lammpstrj", part, model.ids)
            res = external.run_program([exe, "-nocite", "-log", "log.rerun", "-screen", "none", *args, "-in",
                                        "in.rerun"], work, threads=n, log_name="screen.rerun.txt", check=False,
                                       timeout=24 * 3600.0)
            rows = _thermo_rows(work / "log.rerun")
            text = (work / "log.rerun").read_text(encoding="utf-8", errors="replace")
            # the rows decide, not the exit code alone: an MPI build of LAMMPS can end with a signal after it has
            # finished and written everything (seen on macOS: exit -15 after "Total wall time")
            if len(rows) < len(part) or (res.returncode != 0 and "Total wall time" not in text):
                tail = text.splitlines()[-5:]
                raise RuntimeError(f"LAMMPS rerun failed (exit {res.returncode}): {' | '.join(tail)}")
            with open(folder / PT_FILE, "a", encoding="utf-8") as fh:
                for r in rows:
                    s = int(r[0])
                    if s not in want:
                        continue
                    pk = _kinetic_atm(model.n_atoms, T, boxes[s])
                    fh.write(f"{s} {r[1] + pk:.10g} {r[2] + pk:.10g} {r[3] + pk:.10g}\n")
            echo(f"  rerun: {min(k + CHUNK, len(frames))} / {len(frames)} frames (LAMMPS, {n} threads)")
    finally:
        for p in sorted(work.glob("*")):
            with contextlib.suppress(OSError):
                p.unlink()
        with contextlib.suppress(OSError):
            work.rmdir()


# ------------------------------------------------------------------------------------------- OpenMM
def _rerun_openmm_process(folder: Path, platform: str, *, threads: int | None, echo) -> None:
    from . import openmm_engine as OE

    py, why = OE.find_openmm_python()
    if py is None:
        raise RuntimeError(why)
    n = max(1, int(threads or 2))
    cmd = [py, "-m", "quantabench.md.openmm_sigma", "rerun-openmm", str(folder), "--platform", platform,
           "--threads", str(n)]
    echo(f"  rerun: OpenMM {platform} platform, {n} threads ({py})")
    out = subprocess.run(cmd, env=OE.openmm_env(py, n), capture_output=True, text=True,  # noqa: S603
                         timeout=7 * 86400)
    if out.returncode != 0:
        raise RuntimeError(f"OpenMM rerun failed: {(out.stderr or out.stdout).strip().splitlines()[-1:]}")


def rerun_openmm(folder: str | Path, platform: str = "CPU", threads: int = 2, strain: float = 1e-4,
                 echo: Callable[[str], None] = print) -> int:
    """In the OpenMM environment: the diagonal virial of each pending frame by central differences of the energy
    (with the cutoff shift), on ``platform``; appends to ``pressure_tensor.txt``. Returns the frames done."""
    import openmm as mm

    from . import openmm_engine as OE

    folder = Path(folder)
    model = OE.load_model(folder / "slab_start.data", folder / "ff.params")
    meta = OE.read_setup(folder)
    T = float((meta.get("config") or {}).get("temperature", 298.15))
    phys = OE.physics_from_setup(meta, "slab")
    system = OE.build_system(model, phys)
    OE.add_cutoff_shift(system, model, phys)
    ctx = OE.make_context(system, mm.VerletIntegrator(0.001), platform, threads=threads,
                          precision="double" if platform != "CPU" else None)
    have = set(done_steps(folder))
    _header(folder, f"OpenMM {platform} platform, central differences")
    n = 0
    with open(folder / PT_FILE, "a", encoding="utf-8") as fh:
        for step, x, box in dcd.read_frames(folder / FRAMES):
            if step in have:
                continue
            xn, bn = x / 10.0, box / 10.0
            p = []
            for axis in range(3):
                e = []
                for sgn in (1.0, -1.0):
                    s = np.ones(3)
                    s[axis] += sgn * strain
                    b = bn * s
                    ctx.setPeriodicBoxVectors(mm.Vec3(b[0], 0, 0), mm.Vec3(0, b[1], 0), mm.Vec3(0, 0, b[2]))
                    ctx.setPositions(xn * s)
                    e.append(ctx.getState(getEnergy=True).getPotentialEnergy()._value / OE.KCAL)
                dudeps = (e[0] - e[1]) / (2.0 * strain)               # kcal/mol
                p.append(-dudeps / float(np.prod(box)) * KCAL_MOL_A3_TO_ATM)
            pk = _kinetic_atm(model.n_atoms, T, box)
            fh.write(f"{step} {p[0] + pk:.10g} {p[1] + pk:.10g} {p[2] + pk:.10g}\n")
            n += 1
            if n % 200 == 0:
                fh.flush()
                echo(f"  rerun: {n} frames (OpenMM {platform})")
    return n


# ------------------------------------------------------------------------------------------- analysis helpers
def read_kb(folder: str | Path) -> np.ndarray | None:
    """Rows (step, σ_inst mN/m) of ``sigma_kb.txt``, merged by step, or None."""
    p = Path(folder) / "sigma_kb.txt"
    if not p.is_file():
        return None
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        a = np.loadtxt(p, comments="#", ndmin=2)
    if a.size == 0 or a.shape[1] < 2:
        return None
    from .surface_tension import merge_steps

    return merge_steps(a[:, :2])


def compare_series(kb: np.ndarray, rerun_sigma: dict[int, float], skip_steps: int) -> dict | None:
    """The on-the-fly samples against the double-precision rerun at the steps both have (production only):
    {"n", "mean_diff_mN_m" (on-the-fly − rerun), "sem_diff_mN_m", "rms_diff_mN_m", "max_abs_diff_mN_m", "ok"}.
    ok: the mean difference is within 0.1 mN/m plus twice its standard error."""
    d = [(row[1] - rerun_sigma[int(row[0])]) for row in kb if int(row[0]) in rerun_sigma and row[0] >= skip_steps]
    if len(d) < 10:
        return None
    x = np.array(d)
    mean, sem = float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x)))
    return {"n": len(x), "mean_diff_mN_m": mean, "sem_diff_mN_m": sem, "rms_diff_mN_m": float(np.sqrt((x * x).mean())),
            "max_abs_diff_mN_m": float(np.abs(x).max()), "ok": abs(mean) <= 0.1 + 2.0 * sem}


def test_area_sigma(folder: str | Path, T: float, skip_steps: int, block: int) -> dict | None:
    """σ from ``test_area.txt`` (Gloor et al. 2005): σ± = −kT ln⟨exp(−ΔU±/kT)⟩ / (±ε·2A), the mean of the two,
    with block errors over ``block`` samples (at least 4 blocks)."""
    p = Path(folder) / "test_area.txt"
    if not p.is_file():
        return None
    eps = None
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines()[:3]:
        if "eps = +-" in line:
            with contextlib.suppress(ValueError):
                eps = float(line.split("eps = +-")[1].split(";")[0])
    a = np.loadtxt(p, comments="#", ndmin=2)
    if eps is None or a.size == 0:
        return None
    a = a[a[:, 0] >= skip_steps]
    if len(a) < 8:
        return None
    kT = BOLTZ_KCAL * T
    from .surface_tension import KCAL_MOL_A2_TO_MN_M

    def est(rows: np.ndarray) -> float:
        out = []
        for col, sgn in ((1, 1.0), (2, -1.0)):
            du = rows[:, col]
            shift = du.min()
            lnav = -shift / kT + math.log(np.mean(np.exp(-(du - shift) / kT)))
            out.append(-kT * lnav / (sgn * eps * 2.0 * rows[:, 3].mean()) * KCAL_MOL_A2_TO_MN_M)
        return 0.5 * (out[0] + out[1])

    nb = max(4, len(a) // max(1, block))
    nb = min(nb, len(a))
    blocks = np.array_split(a, nb)
    vals = np.array([est(b) for b in blocks])
    return {"mean_mN_m": est(a), "sem_mN_m": float(vals.std(ddof=1) / math.sqrt(len(vals))), "n_blocks": len(vals),
            "n_samples": int(len(a)), "strain": eps,
            "method": "test-area (Gloor et al. 2005), molecules moved as wholes, nonbonded energy"}


def thinning_table(steps: np.ndarray, sigma: np.ndarray, dt_fs: float, factors: Sequence[int] = (1, 2, 5, 10, 20,
                                                                                                  50, 100)
                   ) -> list[dict]:
    """Block errors (10 blocks) of a σ series thinned by each factor: how often frames must be saved."""
    every = float(np.median(np.diff(steps))) * dt_fs if len(steps) > 1 else dt_fs
    out = []
    for f in factors:
        x = sigma[::f]
        if len(x) < 20:
            break
        m = len(x) // 10
        b = x[:m * 10].reshape(10, m).mean(axis=1)
        out.append({"every_fs": every * f, "n": int(len(x)), "block_sem_mN_m": float(b.std(ddof=1) / math.sqrt(10))})
    return out


def _main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m quantabench.md.openmm_sigma")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rerun-openmm")
    r.add_argument("folder")
    r.add_argument("--platform", default="CPU")
    r.add_argument("--threads", type=int, default=2)
    args = ap.parse_args(argv)
    os.environ.setdefault("OPENMM_CPU_THREADS", str(args.threads))
    rerun_openmm(args.folder, args.platform, args.threads)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
