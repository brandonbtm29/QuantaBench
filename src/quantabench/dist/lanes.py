"""Lanes: worker processes that share one machine's CPU cores and GPUs.

CONTRACT (implemented by the lanes builder; ``Lane``/``parse_lanes`` signatures are fixed):

    Lane(name, device, threads)            one worker process; device "cpu" | "cuda:N" | "mps"
    parse_lanes(spec, cfg, report=None) -> list[Lane]
        spec: "1" (today's behaviour: one lane, all threads), "auto" (from the hardware, the method and
        settings.resources.parallel_molecules), "N" (N CPU lanes), or an explicit list such as
        "gpu:1x4,cpu:5x2" (1 GPU lane with 4 threads + 5 CPU lanes with 2 threads).
    jobs.run_resumable(..., lanes=list[Lane] | None)
        None or one lane: the current single-process loop. More: a process pool (spawn start method on
        every OS), the parent stays the only writer of jobs.sqlite, each worker writes its molecule's
        own result folder, logs/progress come back through a queue, cancel reaches every lane, a lane
        that crashes is restarted and its molecule recorded as crashed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Lane:
    name: str            # "gpu0", "cpu1", ...
    device: str = "cpu"  # torch device string
    threads: int = 1     # CPU threads for this lane (BLAS, OpenMP, RDKit, external programs)

    def to_dict(self) -> dict:
        return {"name": self.name, "device": self.device, "threads": self.threads}


def parse_lanes(spec: str | int | None, cfg=None, report=None) -> list[Lane]:
    """Placeholder until the lanes builder lands: everything runs in one lane (today's behaviour)."""
    from .. import resources

    threads = resources.resolve_threads(getattr(cfg, "threads", 0) if cfg is not None else 0)
    return [Lane("main", "auto", threads)]
