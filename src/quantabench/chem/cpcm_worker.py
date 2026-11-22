"""The rank step's GFN1-xTB CPCM term in a helper process (FAIL-1).

tblite 0.8.0's ddX CPCM solver corrupts the process's heap at some geometries: on Windows the process ends with
0xC0000374 (STATUS_HEAP_CORRUPTION), on Linux glibc aborts ("malloc(): invalid size"). It is the geometry, not the
threads or the platform: 2447-54-3's GFN2-xTB water minimum crashes a fresh process on both. In the QSAR tranches
(b03-b15, water) it took down the lane six times, and each of those molecules was marked failed after two crashes.

So ``rank.CpcmTerm`` asks a helper process for its terms: one per process (per lane), started at the first call,
kept for the molecules that follow and started again after a crash. A crash loses that conformer's CPCM term only:
the rank step notes it and ranks the molecule's other conformers (or falls back to the refinement's ranking when
none has a term), and the lane goes on.

Protocol: one JSON object per line on the helper's stdin, one reply per line on its stdout.

    request  {"token": "...", "numbers": [...], "positions": [[x, y, z], ...], "charge": 0, "multiplicity": 1,
              "solvent": "water"}
    reply    {"ok": true, "gas": E_gas, "cpcm": E_cpcm}  (Eh)   or   {"ok": false, "error": "..."}

``token`` names the molecule: the helper keeps that molecule's two tblite calculators, so each conformer's SCF
restarts from the previous one's wavefunction, as in-process. ``QUANTABENCH_CPCM_IN_PROCESS=1`` computes in this
process instead (no helper).
"""

from __future__ import annotations

import atexit
import contextlib
import json
import os
import subprocess
import sys
import threading

IN_PROCESS_ENV = "QUANTABENCH_CPCM_IN_PROCESS"


class HelperCrashed(RuntimeError):
    """The helper process ended while computing a term (tblite's ddX CPCM crashed at that geometry)."""


def in_process() -> bool:
    return os.environ.get(IN_PROCESS_ENV, "").strip().lower() in ("1", "true", "yes")


# ------------------------------------------------------------------------------------------- the helper's side
def serve(lines, write) -> None:
    """Answer requests (``lines``: an iterable of JSON lines) with ``write(reply_line)``, until the input ends."""
    import numpy as np

    from .rank import CpcmTerm

    token, term = None, None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            if req.get("token") != token or term is None:     # a new molecule: fresh calculators
                token = req.get("token")
                term = CpcmTerm(req["numbers"], int(req["charge"]), int(req["multiplicity"]), req["solvent"],
                                in_process=True)
            gas, cpcm = term(np.asarray(req["positions"], float))
            reply = {"ok": True, "gas": float(gas), "cpcm": float(cpcm)}
        except Exception as exc:  # noqa: BLE001 - an SCF that does not converge: the caller notes it
            reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        write(json.dumps(reply) + "\n")


def main(argv: list[str] | None = None) -> int:
    """``python -m quantabench.chem.cpcm_worker THREADS PRIORITY``: the helper process."""
    argv = list(sys.argv[1:] if argv is None else argv)
    threads = int(argv[0]) if argv else 1
    level = argv[1] if len(argv) > 1 else "normal"
    from .. import priority, resources

    resources.apply_thread_limits(max(1, threads))
    with contextlib.suppress(Exception):
        priority.apply_to_process(level)
    # replies go to the original stdout; anything a library prints goes to stderr instead of into the protocol
    out = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", newline="\n")
    with contextlib.suppress(OSError):
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())

    def write(text: str) -> None:
        out.write(text)
        out.flush()

    serve(sys.stdin, write)
    return 0


# ------------------------------------------------------------------------------------------- the caller's side
class Helper:
    """One helper process (started on demand, again after a crash)."""

    def __init__(self, threads: int = 1):
        self.threads = max(1, int(threads))
        self.proc: subprocess.Popen | None = None
        self.lock = threading.Lock()
        self.crashes = 0

    def _start(self) -> subprocess.Popen:
        from .. import priority, proctree
        from . import external

        env = dict(os.environ)
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            env[k] = str(self.threads)
        env.pop(IN_PROCESS_ENV, None)
        level = priority.lower_of(priority.in_force(), priority.current_level())
        # Windows: in a kill-on-close job, so a crash of this process cannot leave the helper in an SCF (ENG-3); on
        # POSIX it ends at the EOF of its stdin
        return proctree.start(
            [sys.executable, "-m", "quantabench.chem.cpcm_worker", str(self.threads), level],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding="utf-8",
            bufsize=1, env=env, creationflags=external._NO_WINDOW)

    def term(self, token: str, numbers, positions, charge: int, multiplicity: int, solvent: str) -> tuple[float, float]:
        """(E(GFN1-xTB gas), E(GFN1-xTB CPCM)) in Eh. Raises ``HelperCrashed`` when the helper dies on it and
        ``RuntimeError`` with tblite's message when the SCF fails."""
        request = json.dumps({"token": token, "numbers": [int(z) for z in numbers],
                              "positions": [[float(x) for x in p] for p in positions], "charge": int(charge),
                              "multiplicity": int(multiplicity), "solvent": solvent}) + "\n"
        with self.lock:
            if self.proc is None or self.proc.poll() is not None:
                self.proc = self._start()
            proc = self.proc
            try:
                proc.stdin.write(request)
                proc.stdin.flush()
                line = proc.stdout.readline()
            except (OSError, ValueError):
                line = ""
            if not line:
                try:
                    code = proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    code = proc.wait()
                _release(proc)
                self.proc = None
                self.crashes += 1
                raise HelperCrashed(f"tblite's ddX CPCM crashed its process at this geometry (exit code {code})")
        reply = json.loads(line)
        if not reply.get("ok"):
            raise RuntimeError(reply.get("error") or "CPCM term failed")
        return float(reply["gas"]), float(reply["cpcm"])

    def close(self) -> None:
        with self.lock:
            proc, self.proc = self.proc, None
        if proc is None:
            return
        with contextlib.suppress(Exception):
            proc.stdin.close()
        try:
            proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            with contextlib.suppress(Exception):
                proc.kill()
        _release(proc)


def _release(proc) -> None:
    from .. import proctree

    with contextlib.suppress(Exception):
        proctree.release(proc)


_helpers: dict[int, Helper] = {}
_helpers_lock = threading.Lock()


def helper(threads: int = 1) -> Helper:
    """This process's helper for ``threads`` threads."""
    threads = max(1, int(threads))
    with _helpers_lock:
        h = _helpers.get(threads)
        if h is None:
            h = _helpers[threads] = Helper(threads)
        return h


@atexit.register
def close_all() -> None:
    with _helpers_lock:
        hs = list(_helpers.values())
        _helpers.clear()
    for h in hs:
        h.close()


if __name__ == "__main__":
    raise SystemExit(main())
