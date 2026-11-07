"""``quantabench-cli methods [NAME]``: the methods guide (``chem.method_guide``) on the command line.

    quantabench-cli methods                   # every entry by family, one line each
    quantabench-cli methods wb97x-d           # one entry (case, hyphens and ω/w do not matter: ωB97X-D, WB97XD)
    quantabench-cli methods gfn2              # GFN2-xTB, with how dxtb, tblite and the xtb program differ
    quantabench-cli methods wb97x-d/def2-svp  # a QM level: the functional's and the basis set's entries
    quantabench-cli methods --search anion    # entries whose text mentions every word

``register`` adds the command to the parser of ``quantabench.cli``.
"""

from __future__ import annotations

import sys

DESCRIPTION = ("What each method, conformer search, functional, basis set and solvent model is, what it is good and "
               "bad at, and what it costs. The same text as the GUI's Help > Methods guide and docs/METHODS.md.")


def register(sub) -> None:
    p = sub.add_parser("methods", help="explain the methods, basis sets and solvent models (the methods guide)",
                       description=DESCRIPTION)
    p.add_argument("name", nargs="?", default="",
                   help="an entry: engine key, method, functional, basis set, model or concept (e.g. wb97x-d, gfn2, "
                        "6-31g*, alpb, dispersion); a 'functional/basis' level shows both")
    p.add_argument("--search", metavar="TEXT", help="list the entries whose text contains every word of TEXT")
    p.add_argument("--width", type=int, default=100, help="line width (default 100)")
    p.set_defaults(func=_methods)


def _methods(args) -> int:
    from .chem import method_guide as guide

    width = max(40, int(args.width))
    if args.search:
        hits = guide.search(args.search)
        if not hits:
            print(f"No entry mentions {args.search!r}.", file=sys.stderr)
            return 1
        for e in hits:
            print(f"  {e.key:<20}{guide.plain(e.summary)}")
        return 0
    name = (args.name or "").strip()
    if not name:
        print(guide.text_list(width))
        return 0
    entries = [guide.lookup(name)] if guide.lookup(name) is not None else guide.lookup_level(name)
    if not entries:
        hits = guide.search(name)
        if len(hits) == 1:
            entries = hits
        else:
            close = guide.suggestions(name)
            print(f"No entry named {name!r}.", file=sys.stderr)
            if hits:
                print("Entries that mention it: " + ", ".join(e.key for e in hits[:12])
                      + (" ..." if len(hits) > 12 else ""), file=sys.stderr)
            elif close:
                print("Did you mean: " + ", ".join(close), file=sys.stderr)
            print("quantabench-cli methods lists every entry.", file=sys.stderr)
            return 1
    print("\n\n".join(guide.text_entry(e, width) for e in entries))
    return 0
