"""Check QuantaBench's OPLS-AA/PF tables (src/quantabench/md/oplsaa_pf.py) against their sources, line by line.

    python scripts/check_oplsaa_pf_params.py --gromacs PATH/oplsaa.ff --clp PATH/il.ff

``--gromacs``: GROMACS' share/top/oplsaa.ff folder (ffnonbonded.itp, ffbonded.itp) at ``md.oplsaa.GROMACS_COMMIT``;
``--clp``: CL&P's il.ff at ``md.oplsaa_pf.CLP_COMMIT``. Every atom type, bond, angle, torsion and improper of the
tables is compared with the line it cites (names, every number). Prints one line per mismatch and a summary, and
exits 1 on any mismatch. The files are not part of this repository; get them with

    curl -LO https://raw.githubusercontent.com/gromacs/gromacs/<GROMACS_COMMIT>/share/top/oplsaa.ff/ffnonbonded.itp
    curl -LO https://raw.githubusercontent.com/gromacs/gromacs/<GROMACS_COMMIT>/share/top/oplsaa.ff/ffbonded.itp
    curl -LO https://raw.githubusercontent.com/paduagroup/clandp/<CLP_COMMIT>/il.ff
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantabench.md import oplsaa_pf as PF  # noqa: E402


def _close(a: float, b: float, tol: float = 1e-9) -> bool:
    return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)


def _fields(line: str) -> list[str]:
    return line.split(";")[0].split("#")[0].split()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gromacs", required=True, type=Path)
    ap.add_argument("--clp", required=True, type=Path)
    a = ap.parse_args(argv)
    nb = (a.gromacs / "ffnonbonded.itp").read_text(encoding="utf-8").splitlines()
    bd = (a.gromacs / "ffbonded.itp").read_text(encoding="utf-8").splitlines()
    clp = a.clp.read_text(encoding="utf-8").splitlines()
    bad = checked = 0

    def fail(msg: str) -> None:
        nonlocal bad
        bad += 1
        print("MISMATCH", msg)

    for t in PF.TYPES.values():
        checked += 1
        line = int(t.source.split("line ")[1].split()[0].rstrip(";"))
        if t.source.startswith("GROMACS"):
            f = _fields(nb[line - 1])
            want_name = "opls_161" if t.name == PF.SILVA_CF2 else t.name
            ok = (f[0] == want_name and _close(f[3], t.mass) and _close(f[6], t.sigma_A / 10.0)
                  and _close(f[7], t.epsilon_kj) and (t.charge is None or _close(f[4], t.charge))
                  and (not t.btype or f[1] == t.btype))
        else:
            f = _fields(clp[line - 1])
            ok = (f[0] == t.name.removeprefix("clp_") and f[1] == t.clp and _close(f[2], t.mass)
                  and _close(f[3], t.charge) and _close(f[5], t.sigma_A) and _close(f[6], t.epsilon_kj))
        if not ok:
            fail(f"type {t.name}: {t.source}: {f}")
    for key, (b0, kb, line, _c) in PF.GMX_BONDS.items():
        checked += 1
        f = _fields(bd[line - 1])
        if not (tuple(f[:2]) == key and f[2] == "1" and _close(f[3], b0) and _close(f[4], kb)):
            fail(f"bond {key} line {line}: {f}")
    for key, (th, k, line, _c) in PF.GMX_ANGLES.items():
        checked += 1
        f = _fields(bd[line - 1])
        if not (tuple(f[:3]) == key and f[3] == "1" and _close(f[4], th) and _close(f[5], k)):
            fail(f"angle {key} line {line}: {f}")
    for key, (rb, line, _c) in PF.GMX_DIHEDRALS.items():
        checked += 1
        f = _fields(bd[line - 1])
        if not (tuple(f[:4]) == key and f[4] == "3" and all(_close(x, y) for x, y in zip(f[5:11], rb))):
            fail(f"dihedral {key} line {line}: {f}")
    for table, n in ((PF.CLP_BONDS, 2), (PF.CLP_ANGLES, 3)):
        for key, (x0, k, line) in table.items():
            checked += 1
            f = _fields(clp[line - 1])
            if not (tuple(f[:n]) == key and f[n] == "harm" and _close(f[n + 1], x0) and _close(f[n + 2], k)):
                fail(f"CL&P {key} line {line}: {f}")
    for table in (PF.CLP_DIHEDRALS, PF.CLP_IMPROPERS):
        for key, (v, line) in table.items():
            checked += 1
            f = _fields(clp[line - 1])
            if not (tuple(f[:4]) == key and f[4] == "opls" and all(_close(x, y) for x, y in zip(f[5:9], v))):
                fail(f"CL&P {key} line {line}: {f}")
    checked += 1
    head = clp[0] if clp else ""
    print(f"CL&P file header: {head}")
    print(f"{checked} entries checked, {bad} mismatches")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
