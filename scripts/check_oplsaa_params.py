"""Check QuantaBench's OPLS-AA table (src/quantabench/md/oplsaa.py) against its two sources.

    python scripts/check_oplsaa_params.py --gromacs PATH/oplsaa.ff --foyer PATH/oplsaa.xml

``--gromacs``: GROMACS' share/top/oplsaa.ff folder (ffnonbonded.itp, ffbonded.itp) at the commit oplsaa.py names;
``--foyer``: foyer's foyer/forcefields/xml/oplsaa.xml. Each table entry is compared with the line it cites in
GROMACS and with foyer's entry for the same types or classes (foyer stores angles in radians, GROMACS in degrees).
Prints one line per check and exits 1 on any mismatch. The files are not part of this repository; get them with

    git clone --depth 1 https://github.com/gromacs/gromacs   (or the commit's archive)
    git clone --depth 1 https://github.com/mosdef-hub/foyer
"""

from __future__ import annotations

import argparse
import math
import sys
import xml.etree.ElementTree as ET  # noqa: S405 - a local file the user chose
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quantabench.md import oplsaa as O  # noqa: E402


def _close(a: float, b: float, tol: float = 1e-6) -> bool:
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gromacs", required=True, type=Path)
    ap.add_argument("--foyer", required=True, type=Path)
    a = ap.parse_args(argv)
    nb = (a.gromacs / "ffnonbonded.itp").read_text(encoding="utf-8").splitlines()
    bd = (a.gromacs / "ffbonded.itp").read_text(encoding="utf-8").splitlines()
    root = ET.parse(a.foyer).getroot()  # noqa: S314
    bad = 0

    def report(ok: bool, what: str) -> None:
        nonlocal bad
        bad += not ok
        print(("ok    " if ok else "DIFF  ") + what)

    foyer_nb = {e.get("type"): e for e in root.iter("Atom")}
    foyer_types = {e.get("name"): e for e in root.iter("Type")}
    for t in O.ATOM_TYPES.values():
        f = nb[t.line - 1].split()
        report(f[0] == t.name and f[1] == t.bond_type and _close(float(f[3]), t.mass) and _close(float(f[4]), t.charge)
               and _close(float(f[6]), t.sigma_nm) and _close(float(f[7]), t.epsilon_kj),
               f"{t.name} vs ffnonbonded.itp:{t.line}: {' '.join(f[:8])}")
        e = foyer_nb.get(t.name)
        ft = foyer_types.get(t.name)
        report(e is not None and _close(float(e.get("charge")), t.charge) and _close(float(e.get("sigma")), t.sigma_nm)
               and _close(float(e.get("epsilon")), t.epsilon_kj),
               f"{t.name} vs foyer <Atom>: {None if e is None else dict(e.attrib)}")
        # foyer leaves opls_963/964 without a class (no SMARTS definition): compare the class where it has one
        if ft is not None and ft.get("def"):
            report(ft.get("class") == t.bond_type, f"{t.name} class {t.bond_type} vs foyer class {ft.get('class')}")

    def foyer_find(tag: str, classes: tuple[str, ...]):
        for e in root.iter(tag):
            cl = tuple(e.get(f"class{i + 1}", "") for i in range(len(classes)))
            want = tuple("" if c == "X" else c for c in classes)
            if cl in (want, want[::-1]):
                return e
        return None

    for key, (b0, kb, line, _c) in O.BONDS.items():
        f = bd[line - 1].split()
        report({f[0], f[1]} == set(key) and _close(float(f[3]), b0) and _close(float(f[4]), kb),
               f"bond {key} vs ffbonded.itp:{line}: {' '.join(f[:5])}")
        e = foyer_find("Bond", key)
        report(e is not None and _close(float(e.get("length")), b0) and _close(float(e.get("k")), kb),
               f"bond {key} vs foyer: {None if e is None else dict(e.attrib)}")
    for key, (th, k, line, _c) in O.ANGLES.items():
        f = bd[line - 1].split()
        report(tuple(f[:3]) in (key, key[::-1]) and _close(float(f[4]), th) and _close(float(f[5]), k),
               f"angle {key} vs ffbonded.itp:{line}: {' '.join(f[:6])}")
        e = foyer_find("Angle", key)
        report(e is not None and _close(float(e.get("angle")), math.radians(th), 1e-8) and _close(float(e.get("k")), k),
               f"angle {key} vs foyer: {None if e is None else dict(e.attrib)}")
    for key, (rb, line, _c) in O.DIHEDRALS_RB.items():
        f = bd[line - 1].split()
        report(tuple(f[:4]) in (key, key[::-1]) and all(_close(float(x), y) for x, y in zip(f[5:11], rb)),
               f"dihedral {key} vs ffbonded.itp:{line}: {' '.join(f[:11])}")
        e = foyer_find("Proper", key)
        report(e is not None and all(_close(float(e.get(f"c{i}")), rb[i]) for i in range(6)),
               f"dihedral {key} vs foyer: {None if e is None else dict(e.attrib)}")
    f = bd[O.PERFLUORO_CCCC_LINE - 1].split()
    report(f[1] == "dih_perfluoroalkane_CT_CT_CT_CT" and all(_close(float(x), y) for x, y in zip(f[2:8], O.PERFLUORO_CCCC_RB)),
           f"perfluoro C-C-C-C vs ffbonded.itp:{O.PERFLUORO_CCCC_LINE}: {' '.join(f)}")
    f = bd[O.IMPROPER_CA_LINE - 1].split()
    report(f[1] == "improper_Z_CA_X_Y" and [float(x) for x in f[2:5]] == [float(x) for x in O.IMPROPER_CA],
           f"aromatic improper vs ffbonded.itp:{O.IMPROPER_CA_LINE}: {' '.join(f)}")
    print(f"{bad} difference(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
