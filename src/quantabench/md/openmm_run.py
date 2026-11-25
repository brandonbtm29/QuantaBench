"""One OpenMM segment of a liquid-slab run (OPENMM-1): the process the run supervisor starts for an OpenMM run.

    <OpenMM env python> -m quantabench.md.openmm_run FOLDER --part bulk|slab --log log.bulk [--checkpoint FILE]
        [--platform auto|OpenCL|CUDA|HIP|CPU] [--precision auto|single|mixed|double] [--device N] [--threads N]

It runs the part's stages from ``run.json`` (``md.sim``) the way ``in.bulk`` and ``in.slab`` run them in LAMMPS, on
the model ``md.openmm_engine`` builds from the folder's own LAMMPS files, and writes the same files, so ``md status``,
the Jobs dock and ``md surface-tension`` read an OpenMM run like a LAMMPS one:

* **bulk**: minimize, NVT warm-up at half the time step, [a lattice start's NVT compression to the target box,
  molecules moved as wholes], NPT (Monte Carlo barostat, isotropic) -> ``bulk_density.txt`` (1-ps averages of 10
  samples: step, density, volume, lx, ly, lz, pe), ``bulk_final.data`` (LAMMPS data with velocities) and
  ``bulk_final.xml``.
* **slab**: the bulk box opened along z with ``vacuum`` Å on each side (molecules whole, the slab in the middle of the
  new box, which starts at 0), ``slab_start.data`` (the opened box, for the σ rerun), ``slab_box.txt``, then NVT:
  ``density_profile.txt`` (fix ave/chunk's layout), ``slab.dcd`` (frames every ``frame_every`` steps, molecules
  whole and their centres in the box), ``sigma_kb.txt`` (on-the-fly Kirkwood-Buff samples, below), optionally
  ``test_area.txt``, and ``slab_final.data`` / ``slab_final.xml``.
* the log (``log.bulk``, ``log.slab``, ``log.<part>.segN``) has LAMMPS' shape: a ``Step ...`` header and thermo rows,
  ``Loop time ...`` and ``Performance: X ns/day ...`` after each stage, and the tags ``QB_PAUSED <part> step N``,
  ``QB_BULK done``, ``QB_SLAB done``, ``QB_ERROR: ...`` (``md.telemetry`` reads it as it grows).

Thermostat and barostat: Langevin (``LangevinMiddleIntegrator``, friction ``friction`` 1/ps) and OpenMM's Monte Carlo
barostat every ``barostat_every`` steps, where LAMMPS uses Nosé-Hoover: the same ensembles, other dynamics. The
centre-of-mass motion is removed every ``momentum_every`` steps (``CMMotionRemover``).

Checkpoints: OpenMM state XML (positions, velocities, box; portable between machines, platforms and OpenMM builds,
unlike OpenMM's binary checkpoints), with the step in ``stepCount``: ``<part>.pause.xml`` when a STOP file pauses the
run at a multiple of ``pause_every`` steps, and every ``restart_every`` steps ``<part>.state.a.xml`` /
``<part>.state.b.xml``. A segment resumes from one of them (``--checkpoint``) and appends to the outputs, which the
supervisor cut at its step first.

σ on the fly (``kb_every`` > 0): every ``kb_every`` steps the slab's potential energy is evaluated at two strained
boxes, the area A = Lx·Ly scaled by (1 ± ε) at constant volume (x and y by √(1 ± ε), z by 1/(1 ± ε)) with every atom
moved with the box, and σ_inst = [U(+ε) − U(−ε)] / (2ε · 2A): the Kirkwood-Buff virial σ = (Lz/2)·(Pzz − (Pxx+Pyy)/2)
of that configuration (the kinetic part averages to zero and is left out), by a central difference. It runs on the
run's own platform (single precision on an Apple GPU); ``md.openmm_sigma`` checks it against the double-precision
rerun of the saved frames at the steps both have. With ``ta_every`` > 0 the test-area method (Gloor et al. 2005)
samples U(±ε) − U(0) of the nonbonded groups with each molecule moved as a whole (``test_area.txt``).

Only ``numpy``, ``openmm`` and QuantaBench's pure-Python ``md.sim`` / ``md.openmm_engine`` / ``md.dcd`` are imported:
it runs in the OpenMM environment (``envs/openmm.yml``) with QuantaBench's source on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import re
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from . import dcd, sim
from . import openmm_engine as OE

AMU_A3_TO_G_CM3 = 1.66053906660        # g/mol per Å³ -> g/cm³
KJ_MOL_NM3_TO_ATM = 16.6053906660 / 1.01325    # kJ/mol/nm³ -> bar -> atm
KCAL_MOL_A2_TO_MN_M = 4184.0 / 6.02214076e23 / 1e-20 * 1e3
WARMUP_FRICTION = 20.0                 # 1/ps (50 fs) for the warm-up and the compression
PAUSE_XML = "{part}.pause.xml"
STATE_XML = ("{part}.state.a.xml", "{part}.state.b.xml")
FINAL_XML = "{part}_final.xml"


class RunFailed(RuntimeError):
    pass


def pause_file(part: str) -> str:
    return PAUSE_XML.format(part=part)


def state_files(part: str) -> list[str]:
    return [s.format(part=part) for s in STATE_XML]


def xml_step(path: str | Path) -> int | None:
    """The ``stepCount`` of an OpenMM state XML file (read from its first lines, nothing parsed), or None."""
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            head = fh.read(2048)
    except OSError:
        return None
    m = re.search(r'stepCount="(\d+)"', head)
    if m is None or "</State>" not in Path(path).read_text(encoding="utf-8", errors="replace")[-200:]:
        return None                    # unfinished (a crash while writing it)
    return int(m.group(1))


# ------------------------------------------------------------------------------------------- the log
class Log:
    """A LAMMPS-shaped log (``md.telemetry`` reads it)."""

    def __init__(self, path: Path, append: bool = False):
        self.fh = open(path, "a" if append else "w", encoding="utf-8", buffering=1)  # noqa: SIM115

    def line(self, text: str = "") -> None:
        self.fh.write(text + "\n")
        self.fh.flush()

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.fh.close()


# ------------------------------------------------------------------------------------------- the engine wrapper
class Engine:
    """One OpenMM context for one stage kind (NVT or NPT) of a part, with the helpers the runner needs."""

    def __init__(self, model: OE.LammpsModel, phys: OE.Physics, cfg: dict, *, platform: str, precision: str,
                 device: str | None, threads: int | None, box_A: Sequence[float], barostat: bool,
                 shift: bool = False):
        import openmm as mm

        self.mm = mm
        self.model = model
        self.system = OE.build_system(model, phys, box_A=box_A)
        self.shift_group = OE.add_cutoff_shift(self.system, model, phys) if shift else None
        self.n = model.n_atoms
        self.dof = 3 * self.n - 3
        self.system.addForce(mm.CMMotionRemover(int(cfg.get("momentum_every") or 1000)))
        T = float(cfg.get("temperature", 298.15))
        self.T = T
        self.barostat = None
        if barostat:
            p_bar = float(cfg.get("pressure_atm", 1.0)) * OE.ATM_TO_BAR
            self.barostat = mm.MonteCarloBarostat(p_bar, T, int(cfg.get("barostat_every") or 25))
            self.barostat.setRandomNumberSeed(int(cfg.get("seed", 1)) % 2**31)
            self.system.addForce(self.barostat)
        self.dt_fs = float(cfg.get("timestep", 1.0))
        self.integrator = mm.LangevinMiddleIntegrator(T, float(cfg.get("friction") or 1.0), self.dt_fs / 1000.0)
        self.integrator.setRandomNumberSeed(int(cfg.get("seed", 1)) % 2**31)
        groups = {self.system.getForce(i).getForceGroup() for i in range(self.system.getNumForces())}
        self.dyn_groups = groups - {OE.SHIFT_GROUP} if self.shift_group is not None else groups
        self.integrator.setIntegrationForceGroups(self.dyn_groups)
        self.context = OE.make_context(self.system, self.integrator, platform, precision=precision, threads=threads,
                                       device=device)
        self.platform = platform
        self.precision = precision

    def set_box(self, box_A: Sequence[float]) -> None:
        b = [float(x) / 10.0 for x in box_A]
        mm = self.mm
        self.context.setPeriodicBoxVectors(mm.Vec3(b[0], 0, 0), mm.Vec3(0, b[1], 0), mm.Vec3(0, 0, b[2]))

    def set_positions(self, x_A: np.ndarray) -> None:
        self.context.setPositions(np.asarray(x_A, dtype=float) / 10.0)

    def set_velocities(self, v_A_fs: np.ndarray) -> None:
        self.context.setVelocities(np.asarray(v_A_fs, dtype=float) * 100.0)

    def timestep(self, dt_fs: float) -> None:
        self.integrator.setStepSize(float(dt_fs) / 1000.0)

    def step(self, n: int) -> None:
        if n > 0:
            self.integrator.step(int(n))

    def state(self, positions: bool = False, velocities: bool = False, energy: bool = False, groups=None):
        kw = {"getPositions": positions, "getVelocities": velocities, "getEnergy": energy}
        if energy:
            kw["groups"] = set(groups) if groups is not None else set(self.dyn_groups)
        return self.context.getState(**kw)

    def strain_groups(self, groups=None) -> set:
        """The groups whose energy a σ sample differentiates: ``groups`` (default: the dynamics') plus the cutoff
        shift when the run has one."""
        g = set(groups) if groups is not None else set(self.dyn_groups)
        return g | ({self.shift_group} if self.shift_group is not None else set())

    @staticmethod
    def box_A(state) -> np.ndarray:
        v = state.getPeriodicBoxVectors(asNumpy=True)._value
        return np.array([v[0][0], v[1][1], v[2][2]]) * 10.0

    @staticmethod
    def pos_A(state) -> np.ndarray:
        return state.getPositions(asNumpy=True)._value * 10.0

    def thermo(self, state) -> dict:
        ke = state.getKineticEnergy()._value / OE.KCAL
        pe = state.getPotentialEnergy()._value / OE.KCAL
        box = self.box_A(state)
        vol = float(np.prod(box))
        return {"ke": ke, "pe": pe, "temp": 2.0 * ke / (self.dof * OE.BOLTZ_KCAL), "box": box, "vol": vol,
                "density": self.model.total_mass / vol * AMU_A3_TO_G_CM3}

    def energy_at(self, x_nm: np.ndarray, box_nm: Sequence[float], groups=None) -> float:
        """kJ/mol of the configuration ``x_nm`` in ``box_nm`` (the context's state is changed: ``restore`` it)."""
        mm = self.mm
        self.context.setPeriodicBoxVectors(mm.Vec3(float(box_nm[0]), 0, 0), mm.Vec3(0, float(box_nm[1]), 0),
                                           mm.Vec3(0, 0, float(box_nm[2])))
        self.context.setPositions(x_nm)
        st = self.state(energy=True, groups=groups)
        return st.getPotentialEnergy()._value

    def restore(self, state) -> None:
        """Back to ``state``'s box and positions, exactly (the platform's own numbers, never rescaled)."""
        self.context.setPeriodicBoxVectors(*state.getPeriodicBoxVectors())
        self.context.setPositions(state.getPositions(asNumpy=True))

    @staticmethod
    def raw(state) -> tuple[np.ndarray, np.ndarray]:
        """(positions, box lengths) in nm as the platform has them."""
        v = state.getPeriodicBoxVectors(asNumpy=True)._value
        return state.getPositions(asNumpy=True)._value, np.array([v[0][0], v[1][1], v[2][2]])

    def save_xml(self, path: Path, step: int) -> None:
        mm = self.mm
        self.context.setStepCount(int(step))
        st = self.context.getState(getPositions=True, getVelocities=True, getParameters=True,
                                   enforcePeriodicBox=False)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(mm.XmlSerializer.serialize(st), encoding="utf-8")
        os.replace(tmp, path)

    def load_xml(self, path: Path) -> int:
        st = self.mm.XmlSerializer.deserialize(Path(path).read_text(encoding="utf-8"))
        self.context.setPeriodicBoxVectors(*st.getPeriodicBoxVectors())
        self.context.setPositions(st.getPositions())
        self.context.setVelocities(st.getVelocities())
        step = int(st.getStepCount())
        self.context.setStepCount(step)
        return step


# ------------------------------------------------------------------------------------------- strain samples
def strained(x: np.ndarray, box: np.ndarray, eps: float, com: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Positions and box with the area scaled by (1 + eps) at constant volume: x, y by sqrt(1 + eps), z by
    1 / (1 + eps). ``com``: per-atom molecular centre of mass (molecules move as wholes), else every atom moves."""
    s = np.array([math.sqrt(1.0 + eps), math.sqrt(1.0 + eps), 1.0 / (1.0 + eps)])
    if com is None:
        return x * s, box * s
    return x + com * (s - 1.0), box * s


def molecule_com(x: np.ndarray, masses: np.ndarray, mols: list[np.ndarray]) -> np.ndarray:
    """Per-atom centre of mass of its molecule (molecules whole in ``x``)."""
    out = np.empty_like(x)
    for idx in mols:
        m = masses[idx]
        out[idx] = (x[idx] * m[:, None]).sum(axis=0) / m.sum()
    return out


def wrap_molecules(x: np.ndarray, box: np.ndarray, masses: np.ndarray, mols: list[np.ndarray]) -> np.ndarray:
    """Each molecule shifted by whole box lengths so that its centre of mass lies in [0, L)."""
    com = molecule_com(x, masses, mols)
    return x - np.floor(com / box) * box


# ------------------------------------------------------------------------------------------- LAMMPS data out
def write_data_like(template: Path, out: Path, x_A: np.ndarray, v_A_fs: np.ndarray | None, box_A: Sequence[float],
                    step: int, model: OE.LammpsModel, note: str) -> None:
    """A LAMMPS data file like ``template`` (its masses, topology and coefficients as they stand) with new
    positions (box from 0, wrapped, with image flags), velocities and box; the first line says "timestep = step"
    (``md.sim.data_step``)."""
    box = np.asarray(box_A, dtype=float)
    img = np.floor(x_A / box).astype(int)
    xw = x_A - img * box
    lines = Path(template).read_text(encoding="utf-8", errors="replace").splitlines()
    head: list[str] = []
    i = 0
    first = f"LAMMPS data file via QuantaBench OpenMM engine, timestep = {int(step)}, units = real ({note})"
    while i < len(lines):
        s = lines[i].split("#", 1)[0].strip()
        if s in ("Masses", "Atoms", "Velocities", "Bonds", "Angles", "Dihedrals", "Impropers", "Pair Coeffs",
                 "PairIJ Coeffs", "Bond Coeffs", "Angle Coeffs", "Dihedral Coeffs", "Improper Coeffs"):
            break
        if i > 0:
            w = s.split()
            if len(w) == 4 and w[2] in ("xlo", "ylo", "zlo"):
                k = "xyz".index(w[2][0])
                lines[i] = f"0.0 {box[k]:.10f} {w[2]} {w[3]}"
            head.append(lines[i])
        i += 1
    sections: dict[str, list[str]] = {}
    order: list[str] = []
    cur = None
    for raw in lines[i:]:
        s = raw.split("#", 1)[0].strip()
        if s and not s[0].isdigit() and not s[0] == "-" and s[0].isalpha():
            cur = raw.strip()
            order.append(cur)
            sections[cur] = []
            continue
        if cur is not None and s:
            sections[cur].append(raw)
    index = {int(a): k for k, a in enumerate(model.ids)}
    out_lines = [first, *head]
    for name in order:
        key = name.split("#", 1)[0].strip()
        if key == "Velocities":
            continue
        if key == "Atoms":
            rows = []
            for r in sections[name]:
                w = r.split("#", 1)[0].split()
                k = index[int(w[0])]
                rows.append(f"{w[0]} {w[1]} {w[2]} {w[3]} {xw[k, 0]:.8f} {xw[k, 1]:.8f} {xw[k, 2]:.8f} "
                            f"{img[k, 0]} {img[k, 1]} {img[k, 2]}")
            out_lines += ["Atoms # full", "", *rows, ""]
            if v_A_fs is not None:
                out_lines += ["Velocities", "", *(f"{int(a)} {v_A_fs[k, 0]:.10g} {v_A_fs[k, 1]:.10g} "
                                                 f"{v_A_fs[k, 2]:.10g}" for k, a in enumerate(model.ids)), ""]
            continue
        out_lines += [name, "", *sections[name], ""]
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    os.replace(tmp, out)


# ------------------------------------------------------------------------------------------- the runner
class Runner:
    def __init__(self, folder: Path, part: str, log_name: str, *, checkpoint: str | None, platform: str,
                 precision: str, device: str | None, threads: int | None):
        self.folder = folder
        self.run = sim.Run.load(folder)
        self.part = self.run.protocol.part(part)
        self.meta = OE.read_setup(folder)
        self.cfg = dict(self.meta.get("config") or {})
        self.checkpoint = checkpoint
        self.log = Log(folder / log_name)
        self.platform, self.precision = platform, precision
        self.device, self.threads = device, threads
        self.pause_every = int(self.run.protocol.pause_every or 10_000)
        self.restart_every = int(self.run.protocol.restart_every or 100_000)
        self.restart_toggle = 0
        self.t_wall0 = time.time()

    # ------------------------------------------------------------------ helpers
    def stop_requested(self) -> bool:
        return (self.folder / sim.STOP_FILE).exists()

    def resolve_platform(self) -> tuple[str, str]:
        plat, prec = self.platform, self.precision
        if plat == "auto" or prec == "auto":
            desc = OE.describe_platforms()
            best, best_prec = OE.best_platform(desc)
            if plat == "auto":
                plat = best
            if prec == "auto":
                usable = next((p["precisions"] for p in desc["platforms"] if p["name"] == plat), [])
                prec = "mixed" if "mixed" in usable else (usable[0] if usable else "single")
        return plat, prec

    def engine(self, model: OE.LammpsModel, stage: str, box: Sequence[float], barostat: bool,
               shift: bool = False) -> Engine:
        phys = OE.physics_from_setup(self.meta, stage)
        return Engine(model, phys, self.cfg, platform=self.platform, precision=self.precision, device=self.device,
                      threads=self.threads, box_A=box, barostat=barostat, shift=shift)

    def header(self, eng: Engine, what: str) -> None:
        self.log.line(f"OpenMM {eng.mm.__version__} on {eng.platform} ({eng.precision} precision)"
                      + (f", {self.threads} CPU threads" if eng.platform == "CPU" and self.threads else "")
                      + f": {what}")

    def thermo_header(self, slab: bool) -> None:
        self.log.line("Step Time Temp PotEng KinEng TotEng Density Lx Ly Lz" if not slab else
                      "Step Time Temp PotEng KinEng TotEng Lx Ly Lz")

    def thermo_row(self, eng: Engine, step: int, t_ps: float, slab: bool) -> dict:
        th = eng.thermo(eng.state(energy=True))
        if not (math.isfinite(th["pe"]) and math.isfinite(th["ke"])):
            raise RunFailed(f"the energy is not finite at step {step} (the run blew up)")
        b = th["box"]
        base = (f"{step} {t_ps * 1000:.1f} {th['temp']:.4f} {th['pe']:.4f} {th['ke']:.4f} {th['pe'] + th['ke']:.4f}")
        self.log.line(base + (f" {th['density']:.6f}" if not slab else "") + f" {b[0]:.5f} {b[1]:.5f} {b[2]:.5f}")
        return th

    def performance(self, steps: int, wall: float, dt_fs: float) -> None:
        if steps <= 0 or wall <= 0:
            return
        sps = steps / wall
        nsday = sps * dt_fs * 86400.0 / 1e6
        self.log.line(f"Loop time of {wall:.4f} on 1 procs for {steps} steps with {self.model_atoms} atoms")
        self.log.line(f"Performance: {nsday:.3f} ns/day, {24.0 / nsday if nsday else 0.0:.3f} hours/ns, "
                      f"{sps:.3f} timesteps/s")

    def periodic_restart(self, eng: Engine, step: int) -> None:
        name = state_files(self.part.name)[self.restart_toggle]
        eng.save_xml(self.folder / name, step)
        self.restart_toggle ^= 1

    def pause(self, eng: Engine, step: int) -> None:
        eng.save_xml(self.folder / pause_file(self.part.name), step)
        self.log.line(f"QB_PAUSED {self.part.name} step {step}")

    def start_state(self) -> int | None:
        """The checkpoint's step (None: from the part's start)."""
        if not self.checkpoint:
            return None
        step = xml_step(self.folder / self.checkpoint)
        if step is None:
            raise RunFailed(f"{self.checkpoint}: not a complete OpenMM state file")
        return step

    # ------------------------------------------------------------------ the stage loop
    def run_stage(self, eng: Engine, stage: sim.Stage, step: int, outputs: list, slab: bool) -> tuple[int, bool]:
        """Run ``stage`` from ``step`` to its end in chunks, writing the outputs; (step, paused)."""
        dt = float(stage.timestep_fs)
        eng.timestep(dt)
        # the warm-up and a lattice start's compression are thermostatted tightly (LAMMPS: Tdamp 50 fs), as the
        # minimized lattice turns kinetic energy into potential; the NPT and production use ``friction``
        warm = stage.name in ("warmup", "compress")
        eng.integrator.setFriction(WARMUP_FRICTION if warm else float(self.cfg.get("friction") or 1.0))
        periods = [self.pause_every, int(self.cfg.get("thermo_every") or 1000)]
        periods += [o.every for o in outputs if o.every]
        if stage.kind == "nvt" and stage.name == "compress":
            periods.append(100)
        if self.restart_every:
            periods.append(self.restart_every)
        chunk = 0
        for p in periods:
            chunk = p if chunk == 0 else math.gcd(chunk, p)
        thermo_every = int(self.cfg.get("thermo_every") or 1000)
        self.thermo_header(slab)
        t_ps = lambda s: (self.part.ns_at(s)) * 1000.0          # noqa: E731
        self.thermo_row(eng, step, t_ps(step), slab)
        for o in outputs:
            o.start(eng, step)
        t0, s0 = time.time(), step
        while step < stage.end:
            n = min(chunk - step % chunk, stage.end - step)
            eng.step(n)
            step += n
            for o in outputs:
                o.after(eng, step)
            if step % thermo_every == 0 or step == stage.end:
                self.thermo_row(eng, step, t_ps(step), slab)
            if self.restart_every and step % self.restart_every == 0 and step < stage.end:
                self.periodic_restart(eng, step)
            if step % self.pause_every == 0 and step < stage.end and self.stop_requested():
                self.performance(step - s0, time.time() - t0, dt)
                for o in outputs:
                    o.close()
                self.pause(eng, step)
                return step, True
        self.performance(step - s0, time.time() - t0, dt)
        for o in outputs:
            o.close()
        return step, False

    # ------------------------------------------------------------------ bulk
    def run_bulk(self) -> int:
        folder, cfg = self.folder, self.cfg
        model = OE.load_model(folder / "system.data", folder / "ff.params")
        self.model_atoms = model.n_atoms
        start = self.start_state()
        box0 = model.box_len
        eng = self.engine(model, "bulk", box0, barostat=True)
        eng.barostat.setFrequency(0)
        eng.context.reinitialize(preserveState=False)
        self.header(eng, f"bulk liquid, {model.n_atoms} atoms")
        stages = [s for s in self.part.stages]
        if start is None:
            OE.set_state(eng.context, model, velocities=False)
            st = eng.state(energy=True)
            self.log.line(f"QB_MIN start: PE {st.getPotentialEnergy()._value / OE.KCAL:.4f} kcal/mol")
            eng.mm.LocalEnergyMinimizer.minimize(eng.context, 10.0, 2000)
            st = eng.state(energy=True)
            self.log.line(f"QB_MIN end: PE {st.getPotentialEnergy()._value / OE.KCAL:.4f} kcal/mol")
            eng.context.setVelocitiesToTemperature(float(cfg.get("temperature", 298.15)), int(cfg.get("seed", 1)))
            step = 0
        else:
            step = eng.load_xml(folder / self.checkpoint)
            self.log.line(f"QB_RESUME bulk from {self.checkpoint} at step {step}")
        density_out = BulkDensity(folder / "bulk_density.txt", model, fresh=start is None)
        for stage in stages:
            if not stage.steps or step >= stage.end:
                continue
            if stage.kind == "npt":
                eng.barostat.setFrequency(int(cfg.get("barostat_every") or 25))
                eng.context.reinitialize(preserveState=True)
            outputs: list = [density_out] if stage.kind == "npt" else []
            if stage.name == "compress":
                outputs.append(Compress(eng, model, np.array(self.meta.get("box_target_A") or box0, dtype=float),
                                        stage))
            self.log.line(f"QB_STAGE bulk {stage.name} steps {max(step, stage.start)}-{stage.end} "
                          f"dt {stage.timestep_fs:g} fs")
            step, paused = self.run_stage(eng, stage, max(step, stage.start), outputs, slab=False)
            if paused:
                return 0
            if stage.kind == "npt":
                eng.barostat.setFrequency(0)
                eng.context.reinitialize(preserveState=True)
        st = eng.state(positions=True, velocities=True)
        box = eng.box_A(st)
        x = wrap_molecules(eng.pos_A(st), box, model.masses, model.molecules())
        v = st.getVelocities(asNumpy=True)._value / 100.0
        write_data_like(folder / "system.data", folder / "bulk_final.data", x, v, box, step, model, "bulk NPT end")
        eng.save_xml(folder / FINAL_XML.format(part="bulk"), step)
        self.log.line(f"QB_BULK done: bulk_final.data written (box {box[0]:.6f} {box[1]:.6f} {box[2]:.6f})")
        return 0

    # ------------------------------------------------------------------ slab
    def run_slab(self) -> int:
        folder, cfg = self.folder, self.cfg
        start = self.start_state()
        vac = float(cfg.get("vacuum", 50.0))
        if start is None:
            model = OE.load_model(folder / "bulk_final.data", folder / "ff.params")
            mols = model.molecules()
            box_bulk = model.box_len.copy()
            x = model.positions - model.box_lo
            x = wrap_molecules(x, box_bulk, model.masses, mols)
            x[:, 2] += vac
            box = box_bulk.copy()
            box[2] += 2.0 * vac
            write_data_like(folder / "bulk_final.data", folder / "slab_start.data", x, model.velocities, box, 0,
                            model, "slab: the bulk box opened along z")
        model = OE.load_model(folder / "slab_start.data", folder / "ff.params")
        self.model_atoms = model.n_atoms
        box = model.box_len
        mols = model.molecules()
        eng = self.engine(model, "slab", box, barostat=False,
                          shift=bool(int(cfg.get("kb_every") or 0) or int(cfg.get("ta_every") or 0)))
        self.header(eng, f"liquid-vapour slab, {model.n_atoms} atoms")
        if start is None:
            OE.set_state(eng.context, model, velocities=True)
            step = 0
            bulk_lz = box[2] - 2.0 * vac
            st_ = self.part.stages[-1]
            settle = int(round(float(cfg.get("settle_ns", 2.0)) * 1e6 / float(cfg.get("timestep", 1.0))))
            pe = int(cfg.get("kb_every") or 0) or int(cfg.get("frame_every") or 0) or int(cfg.get("pressure_every") or 10)
            (folder / "slab_box.txt").write_text(
                f"QB_SLAB lx {box[0]:.6f} ly {box[1]:.6f} lz {box[2]:.6f} zlo {0.0:.6f} zhi {box[2]:.6f} "
                f"dt {float(cfg.get('timestep', 1.0)):g} p_every {pe} settle_steps {settle} "
                f"production_steps {st_.end - settle} T {float(cfg.get('temperature', 298.15)):g} "
                f"rc {float(cfg.get('cutoff') or 12.0):g} bulk_lz {bulk_lz:.6f} engine openmm\n", encoding="utf-8")
        else:
            step = eng.load_xml(folder / self.checkpoint)
            self.log.line(f"QB_RESUME slab from {self.checkpoint} at step {step}")
        fresh = start is None
        outputs: list = [Profile(folder / "density_profile.txt", model, cfg, fresh=fresh)]
        fe = int(cfg.get("frame_every") or 0)
        if fe:
            outputs.append(Frames(folder / "slab.dcd", model, mols, fe, float(cfg.get("timestep", 1.0))))
        ke = int(cfg.get("kb_every") or 0)
        if ke:
            outputs.append(KBSampler(folder / "sigma_kb.txt", eng, ke, float(cfg.get("kb_strain") or 1e-4),
                                     fresh=fresh))
        te = int(cfg.get("ta_every") or 0)
        if te:
            outputs.append(TestArea(folder / "test_area.txt", eng, model, mols, te,
                                    float(cfg.get("ta_strain") or 5e-4), fresh=fresh))
        for stage in self.part.stages:
            if not stage.steps or step >= stage.end:
                continue
            self.log.line(f"QB_STAGE slab {stage.name} steps {max(step, stage.start)}-{stage.end} "
                          f"dt {stage.timestep_fs:g} fs")
            step, paused = self.run_stage(eng, stage, max(step, stage.start), outputs, slab=True)
            if paused:
                return 0
        st = eng.state(positions=True, velocities=True)
        x = wrap_molecules(eng.pos_A(st), eng.box_A(st), model.masses, mols)
        v = st.getVelocities(asNumpy=True)._value / 100.0
        write_data_like(folder / "slab_start.data", folder / "slab_final.data", x, v, eng.box_A(st), step, model,
                        "slab NVT end")
        eng.save_xml(folder / FINAL_XML.format(part="slab"), step)
        self.log.line("QB_SLAB done")
        return 0

    def go(self) -> int:
        try:
            self.platform, self.precision = self.resolve_platform()
            if self.part.name == "bulk":
                return self.run_bulk()
            if self.part.name == "slab":
                return self.run_slab()
            raise RunFailed(f"part {self.part.name!r}: the OpenMM engine runs liquid slabs (bulk, slab)")
        except Exception as exc:  # noqa: BLE001 - every failure ends as QB_ERROR in the log
            self.log.line(f"QB_ERROR: {type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"[:300])
            import traceback

            for line in traceback.format_exc().splitlines():
                self.log.line("# " + line)
            return 1
        finally:
            self.log.close()


# ------------------------------------------------------------------------------------------- outputs
class Output:
    every = 0

    def start(self, eng: Engine, step: int) -> None:
        pass

    def after(self, eng: Engine, step: int) -> None:
        pass

    def close(self) -> None:
        pass


class BulkDensity(Output):
    """``fix ave/time 100 10 1000 v_rho v_vol v_lxv v_lyv v_lzv v_pev``: written at multiples of 1000 steps."""

    every = 100

    def __init__(self, path: Path, model: OE.LammpsModel, fresh: bool):
        self.path, self.model = path, model
        if fresh or not path.is_file():
            path.write_text("# Time-averaged data for fix rhoavg (QuantaBench OpenMM engine)\n"
                            "# TimeStep v_rho v_vol v_lxv v_lyv v_lzv v_pev\n", encoding="utf-8")
        self.acc: list[np.ndarray] = []

    def after(self, eng: Engine, step: int) -> None:
        if step % 100:
            return
        th = eng.thermo(eng.state(energy=True))
        b = th["box"]
        self.acc.append(np.array([th["density"], th["vol"], b[0], b[1], b[2], th["pe"]]))
        if step % 1000 == 0:
            m = np.mean(self.acc[-10:], axis=0)
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(f"{step} " + " ".join(f"{v:.8g}" for v in m) + "\n")
            self.acc = []


class Compress(Output):
    """A lattice start's NVT compression to the target box (LAMMPS' ``fix deform ... remap x``, with molecules moved
    as wholes): every 100 steps the box lengths move linearly toward the target."""

    every = 100

    def __init__(self, eng: Engine, model: OE.LammpsModel, target: np.ndarray, stage: sim.Stage):
        self.target, self.stage = target, stage
        self.mols, self.masses = model.molecules(), model.masses
        self.box0: np.ndarray | None = None

    def start(self, eng: Engine, step: int) -> None:
        self.box0 = eng.box_A(eng.state())
        self.s0 = step

    def after(self, eng: Engine, step: int) -> None:
        if step % 100 and step != self.stage.end:
            return
        frac = (step - self.s0) / max(1, self.stage.end - self.s0)
        new = self.box0 + (self.target - self.box0) * frac
        st = eng.state(positions=True)
        old = eng.box_A(st)
        x = eng.pos_A(st)
        com = molecule_com(x, self.masses, self.mols)
        eng.set_box(new)
        eng.set_positions(x + com * (new / old - 1.0))


class Profile(Output):
    """``fix ave/chunk prof_every prof_repeat prof_freq`` of the mass density in ``bin`` Å slices along z from z = 0."""

    def __init__(self, path: Path, model: OE.LammpsModel, cfg: dict, fresh: bool):
        self.path, self.model = path, model
        self.every = int(cfg.get("profile_every") or 100)
        self.freq = int(cfg.get("profile_freq") or 10_000)
        self.repeat = int(cfg.get("profile_repeat") or 100)
        self.bin = float(cfg.get("profile_bin") or 0.5)
        if fresh or not path.is_file():
            path.write_text("# Chunk-averaged data for fix dprof and group all (QuantaBench OpenMM engine)\n"
                            "# Timestep Number-of-chunks Total-count\n# Chunk Coord1 Ncount density/mass\n",
                            encoding="utf-8")
        self.sum: np.ndarray | None = None
        self.cnt: np.ndarray | None = None
        self.n = 0

    def after(self, eng: Engine, step: int) -> None:
        if step % self.every:
            return
        if (step % self.freq) and (step % self.freq) <= self.freq - self.every * self.repeat:
            return
        st = eng.state(positions=True)
        box = eng.box_A(st)
        z = eng.pos_A(st)[:, 2] % box[2]
        nb = int(math.ceil(box[2] / self.bin - 1e-9))
        k = np.minimum((z / self.bin).astype(int), nb - 1)
        mass = np.bincount(k, weights=self.model.masses, minlength=nb)
        cnt = np.bincount(k, minlength=nb).astype(float)
        dens = mass / (box[0] * box[1] * self.bin) * AMU_A3_TO_G_CM3
        if self.sum is None or len(self.sum) != nb:
            self.sum, self.cnt, self.n = np.zeros(nb), np.zeros(nb), 0
        self.sum += dens
        self.cnt += cnt
        self.n += 1
        if step % self.freq == 0:
            d, c = self.sum / self.n, self.cnt / self.n
            rows = [f"{step} {nb} {self.model.n_atoms}"]
            rows += [f"  {i + 1} {(i + 0.5) * self.bin:.4f} {c[i]:.6g} {d[i]:.8g}" for i in range(nb)]
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write("\n".join(rows) + "\n")
            self.sum, self.cnt, self.n = None, None, 0


class Frames(Output):
    """Frames for OVITO and the σ rerun: ``slab.dcd`` every ``every`` steps (molecules whole, centres in the box)."""

    def __init__(self, path: Path, model: OE.LammpsModel, mols, every: int, dt_fs: float):
        self.writer = dcd.DCDWriter(path, model.n_atoms, every, dt_fs * every / 1000.0,
                                    title=f"QuantaBench OpenMM engine: slab frames every {every} steps")
        self.model, self.mols, self.every = model, mols, every

    def _write(self, eng: Engine, step: int) -> None:
        nxt = self.writer.next_step()
        if nxt is not None and step != nxt:
            return
        st = eng.state(positions=True)
        box = eng.box_A(st)
        self.writer.write(step, wrap_molecules(eng.pos_A(st), box, self.model.masses, self.mols), box)

    def start(self, eng: Engine, step: int) -> None:
        if step % self.every == 0:
            self._write(eng, step)

    def after(self, eng: Engine, step: int) -> None:
        if step % self.every == 0:
            self._write(eng, step)


class KBSampler(Output):
    """σ_inst = [U(+ε) − U(−ε)] / (2ε · 2A) every ``every`` steps (atoms moved with the box; the module docstring)."""

    def __init__(self, path: Path, eng: Engine, every: int, strain: float, fresh: bool):
        self.path, self.every, self.eps = path, int(every), float(strain)
        if fresh or not path.is_file():
            path.write_text(f"# Kirkwood-Buff sigma on the fly (QuantaBench OpenMM engine, {eng.platform} "
                            f"{eng.precision}): central difference of U over the area at constant volume, "
                            f"strain {self.eps:g}, atoms moved with the box\n"
                            "# TimeStep sigma_mN_m dU_kcal (U(+eps) - U(-eps))\n", encoding="utf-8")
        self.fh = open(path, "a", encoding="utf-8")  # noqa: SIM115

    def sample(self, eng: Engine, step: int) -> None:
        st = eng.state(positions=True)
        x, box = eng.raw(st)
        g = eng.strain_groups()
        up = eng.energy_at(*strained(x, box, self.eps, None), groups=g)
        um = eng.energy_at(*strained(x, box, -self.eps, None), groups=g)
        eng.restore(st)
        area2 = 2.0 * box[0] * box[1] * 100.0                          # nm² -> Å², both interfaces
        du = (up - um) / OE.KCAL
        sig = du / (2.0 * self.eps * area2) * KCAL_MOL_A2_TO_MN_M
        self.fh.write(f"{step} {sig:.8g} {du:.8g}\n")

    def start(self, eng: Engine, step: int) -> None:
        if step % self.every == 0:
            self.sample(eng, step)

    def after(self, eng: Engine, step: int) -> None:
        if step % self.every == 0:
            self.sample(eng, step)
            if step % (self.every * 50) == 0:
                self.fh.flush()

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.fh.flush()


class TestArea(Output):
    """Test-area samples (Gloor et al. 2005): ΔU(±ε) of the nonbonded groups with molecules moved as wholes."""

    def __init__(self, path: Path, eng: Engine, model: OE.LammpsModel, mols, every: int, strain: float, fresh: bool):
        self.path, self.every, self.eps = path, int(every), float(strain)
        self.mols, self.masses = mols, model.masses
        self.groups = eng.strain_groups(OE.NONBONDED & set(eng.dyn_groups))
        if fresh or not path.is_file():
            path.write_text(f"# Test-area samples (QuantaBench OpenMM engine, {eng.platform} {eng.precision}): "
                            f"nonbonded energy change when the area grows by a factor (1 + eps) at constant volume, "
                            f"molecules moved as wholes, eps = +-{self.eps:g}; area A = Lx*Ly of one interface\n"
                            "# TimeStep dU_plus_kcal dU_minus_kcal area_A2\n", encoding="utf-8")
        self.fh = open(path, "a", encoding="utf-8")  # noqa: SIM115

    def sample(self, eng: Engine, step: int) -> None:
        st = eng.state(positions=True, energy=True, groups=self.groups)
        x, box = eng.raw(st)
        u0 = st.getPotentialEnergy()._value
        com = molecule_com(x, self.masses, self.mols)
        up = eng.energy_at(*strained(x, box, self.eps, com), groups=self.groups)
        um = eng.energy_at(*strained(x, box, -self.eps, com), groups=self.groups)
        eng.restore(st)
        self.fh.write(f"{step} {(up - u0) / OE.KCAL:.8g} {(um - u0) / OE.KCAL:.8g} {box[0] * box[1] * 100:.6f}\n")

    def start(self, eng: Engine, step: int) -> None:
        if step % self.every == 0:
            self.sample(eng, step)

    def after(self, eng: Engine, step: int) -> None:
        if step % self.every == 0:
            self.sample(eng, step)

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.fh.flush()


# ------------------------------------------------------------------------------------------- command line
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m quantabench.md.openmm_run")
    ap.add_argument("folder")
    ap.add_argument("--part", required=True, choices=["bulk", "slab"])
    ap.add_argument("--log", required=True)
    ap.add_argument("--checkpoint")
    ap.add_argument("--platform", default="auto")
    ap.add_argument("--precision", default="auto")
    ap.add_argument("--device")
    ap.add_argument("--threads", type=int)
    args = ap.parse_args(argv)
    if args.threads:
        os.environ["OPENMM_CPU_THREADS"] = str(args.threads)
    r = Runner(Path(args.folder).resolve(), args.part, args.log, checkpoint=args.checkpoint, platform=args.platform,
               precision=args.precision, device=args.device, threads=args.threads)
    return r.go()


if __name__ == "__main__":
    sys.exit(main())
