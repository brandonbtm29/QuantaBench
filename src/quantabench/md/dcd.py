"""DCD trajectories (CHARMM/NAMD layout, as OpenMM, VMD, MDTraj and OVITO read them), numpy only (OPENMM-1).

The OpenMM runs of ``md.openmm_run`` save their slab frames here (``slab.dcd``): positions in Å as float32 and an
orthogonal unit cell per frame. Unlike OpenMM's own ``DCDFile`` this writer can append to a file that a resumed
segment continues, and ``truncate`` cuts it at a step (``md.sim.cut_at``, kind "dcd"), so the frames stay one series
across pauses. ``read_frames`` yields (step, positions, box) for the σ rerun (``md.openmm_sigma``); ``thin`` keeps
the frames at multiples of a longer interval once that rerun is done.

Layout: a header record (``CORD``, 20 int32 control words: frames, first step, steps between frames, last step,
..., Δt as float32, the unit-cell flag 1, CHARMM version 24), a title record, the atom count, then per frame a
48-byte unit-cell record (a, γ, b, β, α, c with angles in degrees) and the x, y and z records. Every record is
framed by its length in int32, little-endian.
"""

from __future__ import annotations

import os
import struct
from collections.abc import Iterator
from pathlib import Path

import numpy as np

_HDR = struct.Struct("<i4s9if10ii")         # 84, "CORD", 9 ints, delta, 10 ints (unit cell flag first), 84


def _header(n_frames: int, first_step: int, every: int, dt_ps: float) -> bytes:
    last = first_step + every * max(0, n_frames - 1)
    icntrl = [n_frames, first_step, every, last, 0, 0, 0, 0, 0]
    tail = [1, 0, 0, 0, 0, 0, 0, 0, 0, 24]
    return _HDR.pack(84, b"CORD", *icntrl, float(dt_ps), *tail, 84)


def _title(text: str) -> bytes:
    lines = [text[i:i + 80].ljust(80) for i in range(0, max(1, len(text)), 80)][:4]
    body = struct.pack("<i", len(lines)) + "".join(lines).encode("ascii", "replace")
    return struct.pack("<i", len(body)) + body + struct.pack("<i", len(body))


class Header:
    def __init__(self, n_frames: int, first_step: int, every: int, dt_ps: float, n_atoms: int, offset: int,
                 cell: bool = True):
        self.n_frames, self.first_step, self.every = n_frames, first_step, every
        self.dt_ps, self.n_atoms, self.offset = dt_ps, n_atoms, offset    # offset: bytes before the first frame
        self.cell = cell

    @property
    def frame_bytes(self) -> int:
        return (56 if self.cell else 0) + 3 * (4 + 4 * self.n_atoms + 4)

    def step(self, k: int) -> int:
        return self.first_step + k * self.every


def read_header(path: str | Path) -> Header:
    with open(path, "rb") as fh:
        raw = fh.read(_HDR.size)
        if len(raw) < _HDR.size:
            raise ValueError(f"{path}: not a DCD file (too short)")
        v = _HDR.unpack(raw)
        if v[0] != 84 or v[1] != b"CORD":
            raise ValueError(f"{path}: not a little-endian CHARMM DCD file")
        n_frames, first, every = v[2], v[3], v[4]
        dt = v[11]
        cell = v[12] == 1
        (tlen,) = struct.unpack("<i", fh.read(4))
        fh.seek(tlen + 4, os.SEEK_CUR)
        (four, natoms, four2) = struct.unpack("<3i", fh.read(12))
        if four != 4 or four2 != 4:
            raise ValueError(f"{path}: bad atom-count record")
        offset = fh.tell()
    h = Header(n_frames, first, every, dt, natoms, offset, cell)
    size = Path(path).stat().st_size
    complete = (size - offset) // h.frame_bytes
    h.n_frames = min(h.n_frames, complete) if h.n_frames else complete      # a frame cut off by a crash is ignored
    return h


class DCDWriter:
    """Append frames to ``path`` (created with its header when missing). ``every``: steps between frames; a file
    continued by a later segment must have the same interval, and its next frame the step after its last."""

    def __init__(self, path: str | Path, n_atoms: int, every: int, dt_ps: float, title: str = "QuantaBench"):
        self.path = Path(path)
        self.n_atoms, self.every, self.dt_ps = int(n_atoms), int(every), float(dt_ps)
        if self.path.is_file() and self.path.stat().st_size > 0:
            h = read_header(self.path)
            if h.n_atoms != self.n_atoms:
                raise ValueError(f"{path}: {h.n_atoms} atoms in the file, {n_atoms} in the run")
            if not h.cell:
                raise ValueError(f"{path}: frames without a unit cell")
            if h.n_frames and h.every != self.every:
                raise ValueError(f"{path}: frames every {h.every} steps in the file, {every} in the run")
            self.header = h
            with open(self.path, "r+b") as fh:          # drop a partial frame left by a crash
                fh.truncate(h.offset + h.n_frames * h.frame_bytes)
        else:
            with open(self.path, "wb") as fh:
                fh.write(_header(0, 0, self.every, self.dt_ps))
                fh.write(_title(title))
                fh.write(struct.pack("<3i", 4, self.n_atoms, 4))
                offset = fh.tell()
            self.header = Header(0, 0, self.every, self.dt_ps, self.n_atoms, offset)

    def next_step(self) -> int | None:
        h = self.header
        return h.step(h.n_frames) if h.n_frames else None

    def write(self, step: int, positions_A: np.ndarray, box_A) -> None:
        h = self.header
        if h.n_frames == 0:
            h.first_step = int(step)
        elif int(step) != h.step(h.n_frames):
            raise ValueError(f"{self.path}: frame at step {step}, expected {h.step(h.n_frames)}")
        a, b, c = (float(x) for x in box_A)
        x = np.asarray(positions_A, dtype="<f4")
        rec = 4 * self.n_atoms
        parts = [struct.pack("<i6di", 48, a, 90.0, b, 90.0, 90.0, c, 48)]
        for k in range(3):
            parts += [struct.pack("<i", rec), np.ascontiguousarray(x[:, k]).tobytes(), struct.pack("<i", rec)]
        with open(self.path, "r+b") as fh:
            fh.seek(h.offset + h.n_frames * h.frame_bytes)
            fh.write(b"".join(parts))
            h.n_frames += 1
            fh.seek(0)
            fh.write(_header(h.n_frames, h.first_step, h.every, h.dt_ps))


def read_frames(path: str | Path, *, after: int | None = None, start: int = 0, stride: int = 1
                ) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """(step, positions (n, 3) Å as float64, box lengths (3,) Å) per frame; ``after``: only frames whose step is
    greater."""
    h = read_header(path)
    n = h.n_atoms
    with open(path, "rb") as fh:
        for k in range(start, h.n_frames, max(1, stride)):
            step = h.step(k)
            if after is not None and step <= after:
                continue
            fh.seek(h.offset + k * h.frame_bytes)
            buf = fh.read(h.frame_bytes)
            cell = struct.unpack_from("<6d", buf, 4) if h.cell else (0.0,) * 6
            off = 56 if h.cell else 0
            xyz = np.empty((n, 3))
            for d in range(3):
                xyz[:, d] = np.frombuffer(buf, dtype="<f4", count=n, offset=off + 4)
                off += 8 + 4 * n
            yield step, xyz, np.array([cell[0], cell[2], cell[5]])


def frame_steps(path: str | Path) -> list[int]:
    h = read_header(path)
    return [h.step(k) for k in range(h.n_frames)]


def truncate(path: str | Path, step: int, keep_step: bool) -> int:
    """Drop the frames after ``step`` (and the one at ``step`` unless ``keep_step``); returns how many."""
    path = Path(path)
    if not path.is_file():
        return 0
    h = read_header(path)
    keep = 0
    while keep < h.n_frames and (h.step(keep) < step or (keep_step and h.step(keep) == step)):
        keep += 1
    dropped = h.n_frames - keep
    with open(path, "r+b") as fh:
        fh.truncate(h.offset + keep * h.frame_bytes)
        fh.seek(0)
        fh.write(_header(keep, h.first_step if keep else 0, h.every, h.dt_ps))
    return dropped


def thin(path: str | Path, every: int) -> tuple[int, int]:
    """Keep only the frames whose step is a multiple of ``every`` (a multiple of the file's interval) and delete the
    rest: the file is rewritten beside itself and then replaces it, so a crash leaves the old one. Returns (frames
    before, frames after); a file already at that interval is left alone."""
    path = Path(path)
    h = read_header(path)
    every = int(every)
    if every <= 0 or every % h.every:
        raise ValueError(f"{path}: frames every {h.every} steps; {every} is not a multiple of that")
    if not h.n_frames or every == h.every:
        return h.n_frames, h.n_frames
    keep = [k for k in range(h.n_frames) if h.step(k) % every == 0]
    tmp = path.with_name(path.name + ".thinning")
    with open(path, "rb") as src, open(tmp, "wb") as out:
        src.seek(_HDR.size)
        out.write(_header(len(keep), h.step(keep[0]) if keep else 0, every, h.dt_ps * every / h.every))
        out.write(src.read(h.offset - _HDR.size))           # title and atom count as they were
        for k in keep:
            src.seek(h.offset + k * h.frame_bytes)
            out.write(src.read(h.frame_bytes))
    os.replace(tmp, path)
    return h.n_frames, len(keep)
