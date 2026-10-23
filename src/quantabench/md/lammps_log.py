"""Read LAMMPS output back: thermo blocks from ``log.lammps``, the SMD ``pull_force.txt``, and a
detachment-work estimate.

Pure NumPy (no pandas), so it runs in the core install and in notebooks alike.

How the detachment work is estimated
------------------------------------
``fix smd`` accumulates the external work W(t) = ∫ F_spring · v dt. Once the adsorbate is beyond the
pair cutoff of every surface atom the spring force falls to thermal noise around zero and W(t)
levels off: that plateau is the work of detachment for this pull (in kcal/mol). We average W over the
last ``plateau_fraction`` of the samples and call the plateau *reached* only when the mean force in that
window is small compared with the peak force (default < 10 %) and the work drifts by less than
``drift_tol`` kcal/mol across the window. Otherwise the pull was too short and the number is a lower
bound of this pull's work, which the ``note`` says.

Remember that a single non-equilibrium pull gives W >= ΔF (Jarzynski, Phys. Rev. Lett. 1997, 78,
2690); the plateau is not an adsorption free energy unless averaged over many pulls.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

KCAL_TO_KJ = 4.184
KCAL_MOL_A_TO_PN = 69.4786   # 1 kcal/mol/Å = 4184 J / 6.02214076e23 / 1e-10 m = 69.48 pN

_NUM = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$|^[-+]?(nan|inf)$", re.IGNORECASE)


def _is_number(tok: str) -> bool:
    return bool(_NUM.match(tok))


@dataclass
class ThermoBlock:
    """One ``run``'s thermo output: column names and an (n_rows, n_cols) float array."""

    columns: list[str]
    data: np.ndarray
    loop_time: float | None = None

    def __getitem__(self, name: str) -> np.ndarray:
        return self.data[:, self.columns.index(name)]

    def __len__(self) -> int:
        return len(self.data)

    def as_dict(self) -> dict[str, np.ndarray]:
        return {c: self.data[:, i] for i, c in enumerate(self.columns)}


@dataclass
class LammpsLog:
    blocks: list[ThermoBlock] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    version: str = ""

    @property
    def lost_atoms(self) -> bool:
        return any("Lost atoms" in e for e in self.errors + self.warnings)

    @property
    def ok(self) -> bool:
        return not self.errors

    def concatenated(self) -> dict[str, np.ndarray]:
        """Columns present in every block, joined in run order (the first row of each later block
        repeats the previous run's last step, so it is dropped)."""
        if not self.blocks:
            return {}
        common = [c for c in self.blocks[0].columns if all(c in b.columns for b in self.blocks)]
        parts: dict[str, list[np.ndarray]] = {c: [self.blocks[0][c]] for c in common}
        for prev, b in zip(self.blocks, self.blocks[1:]):
            skip = int("Step" in common and len(prev) > 0 and len(b) > 0 and b["Step"][0] == prev["Step"][-1])
            for c in common:
                parts[c].append(b[c][skip:])
        out = {c: np.concatenate(v) for c, v in parts.items()}
        return out


def parse_log(path: str | Path) -> LammpsLog:
    """Parse every thermo block of a ``log.lammps`` (several ``run`` commands give several blocks).

    Thermo lines interrupted by WARNING lines are handled; a block ends at ``Loop time of`` or at an
    ERROR (in which case the partial block is kept).
    """
    log = LammpsLog()
    header: list[str] | None = None
    rows: list[list[float]] = []

    def close(loop_time: float | None = None) -> None:
        nonlocal header, rows
        if header is not None:
            data = np.array(rows, dtype=float) if rows else np.empty((0, len(header)))
            log.blocks.append(ThermoBlock(header, data, loop_time))
        header, rows = None, []

    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("LAMMPS (") and not log.version:
            log.version = line[len("LAMMPS ("):].rstrip(")")
        if line.startswith("WARNING"):
            log.warnings.append(line)
            continue
        if line.startswith("ERROR"):
            log.errors.append(line)
            close()
            continue
        toks = line.split()
        if toks[0] == "Step" and len(toks) > 1:
            close()
            header = toks
            continue
        if header is not None:
            if line.startswith("Loop time of"):
                m = re.search(r"Loop time of\s+([0-9.eE+-]+)", line)
                close(float(m.group(1)) if m else None)
            elif len(toks) == len(header) and all(_is_number(t) for t in toks):
                rows.append([float(t) for t in toks])
    close()
    return log


@dataclass
class PullData:
    """``pull_force.txt`` columns (see the title line written by ``in.smd``)."""

    columns: list[str]
    data: np.ndarray

    def __getitem__(self, name: str) -> np.ndarray:
        return self.data[:, self.columns.index(name)]

    @property
    def time_fs(self) -> np.ndarray:
        return self["time_fs"]

    @property
    def force(self) -> np.ndarray:
        return self["force_z_kcal_mol_A"]

    @property
    def work(self) -> np.ndarray:
        return self["work_kcal_mol"]

    @property
    def displacement(self) -> np.ndarray:
        return self["displacement_A"]


def read_pull_force(path: str | Path) -> PullData:
    """Read the ``fix print`` output written by ``in.smd`` (``# name name ...`` title, then numbers)."""
    columns: list[str] = []
    rows: list[list[float]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            if not columns:
                columns = s.lstrip("#").split()
            continue
        toks = s.split()
        if all(_is_number(t) for t in toks):
            rows.append([float(t) for t in toks])
    if not columns:
        columns = ["time_fs", "com_z_A", "displacement_A", "force_z_kcal_mol_A", "work_kcal_mol"][: len(rows[0])]
    data = np.array(rows, dtype=float) if rows else np.empty((0, len(columns)))
    if data.ndim == 2 and data.shape[1] != len(columns):
        raise ValueError(f"{path}: {data.shape[1]} data columns but header names {len(columns)}")
    return PullData(columns, data)


def running_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Centered running mean (window shrinks at the ends), to read a peak force through thermal noise."""
    window = max(1, int(window))
    if window == 1 or len(x) < 2:
        return np.asarray(x, dtype=float).copy()
    kernel = np.ones(window)
    num = np.convolve(x, kernel, mode="same")
    den = np.convolve(np.ones_like(x, dtype=float), kernel, mode="same")
    return num / den


@dataclass
class DetachmentEstimate:
    work_kcal_mol: float              # plateau of the accumulated work (mean over the plateau window)
    work_std_kcal_mol: float          # spread of W in that window
    peak_force_kcal_mol_A: float      # maximum of the smoothed spring force along +z (rupture force)
    peak_force_raw_kcal_mol_A: float  # maximum of the unsmoothed force (inflated by thermal noise)
    peak_time_fs: float
    peak_displacement_A: float
    plateau_reached: bool
    plateau_mean_force_kcal_mol_A: float
    n_samples: int
    note: str = ""
    drag_work_kcal_mol: float | None = None   # Langevin friction work over the plateau window (None: unknown)
    raw_work_kcal_mol: float | None = None    # plateau of W as printed by LAMMPS, drag included

    @property
    def work_kj_mol(self) -> float:
        return self.work_kcal_mol * KCAL_TO_KJ

    @property
    def peak_force_pN(self) -> float:
        return self.peak_force_kcal_mol_A * KCAL_MOL_A_TO_PN

    def summary(self) -> str:
        state = "plateau reached" if self.plateau_reached else "NO plateau (pull longer)"
        drag = ""
        if self.drag_work_kcal_mol is not None:
            drag = (f" (W {self.raw_work_kcal_mol:.2f} kcal/mol minus the thermostat's Langevin drag "
                    f"{self.drag_work_kcal_mol:.2f})")
        return (f"Detachment work {self.work_kcal_mol:.2f} ± {self.work_std_kcal_mol:.2f} kcal/mol "
                f"({self.work_kj_mol:.1f} kJ/mol){drag}, peak force {self.peak_force_kcal_mol_A:.2f} kcal/mol/Å "
                f"({self.peak_force_pN:.0f} pN) at t = {self.peak_time_fs:.0f} fs, {state}."
                + (f" {self.note}" if self.note else ""))


def drag_force_from_setup(folder: str | Path) -> float | None:
    """Constant Langevin drag force M v / τ (kcal/mol/Å) on the pulled body, from ``smd_setup.json``."""
    path = Path(folder) / "smd_setup.json"
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if meta.get("langevin_drag_force_kcal_mol_A") is not None:
        return float(meta["langevin_drag_force_kcal_mol_A"])
    work, dist = meta.get("langevin_drag_work_estimate_kcal_mol"), (meta.get("config") or {}).get("pull_distance")
    if work is None or not dist:
        return None
    return float(work) / float(dist)


def estimate_detachment(pull: PullData, plateau_fraction: float = 0.1, smooth: int = 11,
                        force_ratio_tol: float = 0.15, drift_tol: float = 1.0,
                        drag_force_kcal_mol_A: float | None = None) -> DetachmentEstimate:
    """Work of detachment (plateau of W) and peak pulling force from one SMD pull.

    With a stiff spring the instantaneous force is noisy (thermal COM fluctuations give a force spread
    of about sqrt(K kT), 2.4 kcal/mol/Å for K = 10 kcal/mol/Å² at 300 K), so the peak is read from a
    ``smooth``-sample running mean. The plateau is the mean W over the last ``plateau_fraction`` of
    the samples; it counts as reached when the mean force there is below ``force_ratio_tol`` x peak
    and W drifts by less than ``drift_tol`` kcal/mol across the window.

    ``drag_force_kcal_mol_A``: the thermostat's friction on the pulled body, M v / τ (``smd_setup.json``,
    see ``drag_force_from_setup``). It adds a work that grows linearly with the displacement and never
    plateaus, large for fast pulls (4 kcal/mol for TFA at 1e-3 Å/fs over 15 Å). When given, W - F_drag·d
    and f - F_drag are used for the plateau and the reported work, and the raw W is reported next to it.
    """
    if len(pull.data) < 3:
        raise ValueError("pull_force.txt has fewer than 3 samples")
    w, f, t = pull.work, pull.force, pull.time_fs
    disp = pull.displacement if "displacement_A" in pull.columns else np.full_like(t, math.nan)
    raw_w = w
    drag_w = None
    if drag_force_kcal_mol_A and np.all(np.isfinite(disp)):
        drag_w = drag_force_kcal_mol_A * (disp - disp[0])
        w = w - drag_w
        f = f - drag_force_kcal_mol_A
    n = len(w)
    k = max(2, int(math.ceil(plateau_fraction * n)))
    win_w, win_f = w[-k:], f[-k:]
    fs = running_mean(f, min(smooth, max(1, n // 3)))
    i_peak = int(np.argmax(fs))
    peak = float(fs[i_peak])
    mean_f = float(np.mean(win_f))
    drift = float(win_w[-1] - win_w[0])
    reached = bool(peak > 0 and abs(mean_f) < force_ratio_tol * peak and abs(drift) < drift_tol
                   and i_peak < n - k)
    note = ""
    if not reached:
        note = (f"Mean force in the last {k} samples is {mean_f:.2f} kcal/mol/Å (peak {peak:.2f}) and W drifts "
                f"{drift:.2f} kcal/mol: the adsorbate is probably still interacting, so W is a lower bound.")
    if not np.all(np.isfinite(w)):
        note += " Non-finite work values present."
    return DetachmentEstimate(work_kcal_mol=float(np.mean(win_w)), work_std_kcal_mol=float(np.std(win_w)),
                              peak_force_kcal_mol_A=peak, peak_force_raw_kcal_mol_A=float(np.max(f)),
                              peak_time_fs=float(t[i_peak]), peak_displacement_A=float(disp[i_peak]),
                              plateau_reached=reached, plateau_mean_force_kcal_mol_A=mean_f, n_samples=n,
                              note=note.strip(),
                              drag_work_kcal_mol=None if drag_w is None else float(np.mean(drag_w[-k:])),
                              raw_work_kcal_mol=None if drag_w is None else float(np.mean(raw_w[-k:])))
