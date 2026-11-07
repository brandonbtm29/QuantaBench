"""Write docs/METHODS.md from the methods guide's data (src/quantabench/chem/method_guide.py).

    python scripts/write_methods_doc.py           # rewrite docs/METHODS.md
    python scripts/write_methods_doc.py --check   # exit 1 if docs/METHODS.md differs from the data

tests/test_method_guide.py fails when the committed file is out of date: run this after editing the data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "METHODS.md"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--check", action="store_true", help="only check that docs/METHODS.md is up to date")
    args = p.parse_args(argv)
    sys.path.insert(0, str(ROOT / "src"))
    from quantabench.chem import method_guide

    text = method_guide.markdown_document()
    current = DOC.read_text(encoding="utf-8") if DOC.is_file() else ""
    if args.check:
        if current != text:
            print(f"{DOC.relative_to(ROOT)} is out of date: run python scripts/write_methods_doc.py", file=sys.stderr)
            return 1
        print(f"{DOC.relative_to(ROOT)} is up to date ({len(method_guide.ENTRIES)} entries)")
        return 0
    if current == text:
        print(f"{DOC.relative_to(ROOT)} unchanged")
        return 0
    DOC.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {DOC.relative_to(ROOT)} ({len(method_guide.ENTRIES)} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
