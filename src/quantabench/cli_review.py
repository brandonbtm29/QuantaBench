"""``quantabench-cli review``: the model advisor's review of a batch before a run (``chem.advisor``).

    quantabench-cli review --file cohort.csv --phase water              # profile, coverage, findings, presets
    quantabench-cli review "OC(=O)C(F)(F)C(F)(F)C(F)(F)F PFBA" --phase gas --method aimnet2
    quantabench-cli review --file cohort.csv --qm b3lyp/def2-svp --json  # the whole review as JSON

``review`` takes the inputs and science options of ``build`` (the same argument groups), so a review and the build
that follows it see the same settings. ``build`` and ``grid build`` print a short review header before they start
(``print_header``); ``--no-review`` skips it. The header never stops a build: a review that fails says so in one
line.

``register`` adds the command to the parser of ``quantabench.cli``.
"""

from __future__ import annotations

import argparse
import json
import sys

DESCRIPTION = ("Reviews the molecules before a run: what is in the batch (elements, charge classes at the run's pH, "
               "PFAS, functional groups, sizes, salts, stereo, radicals), which methods can take them, findings tied "
               "to QuantaBench's measured results (docs/VALIDATION.md) with the setting to change and the methods-"
               "guide entry to read, and three presets (Fast screen, Recommended, Careful) with estimated times. "
               "Takes the same inputs and options as build.")


def add_review_flag(q: argparse.ArgumentParser) -> None:
    """``--no-review`` for ``build`` and ``grid build``."""
    q.add_argument("--no-review", action="store_true",
                   help="skip the short review of the molecules printed before the run (quantabench-cli review "
                        "shows the full one)")


def register(sub) -> None:
    from . import cli as cli_mod

    p = sub.add_parser("review", help="review the molecules before a run: what is in the batch, which methods fit, "
                                      "what to change (the model advisor)", description=DESCRIPTION)
    cli_mod.add_input_args(p)
    cli_mod.add_engine_args(p)
    cli_mod.add_build_science_args(p)
    p.add_argument("--phase", choices=["gas", "water"],
                   help="the phase to review for: gas (no solvent) or water (overrides --solvent)")
    p.add_argument("--json", action="store_true", help="print the whole review as JSON")
    p.add_argument("--width", type=int, default=100, help="line width of the report (default 100)")
    p.add_argument("--ids", type=int, default=8, help="molecule ids listed per finding (default 8)")
    p.set_defaults(func=_review)


def config_for(args):
    """The BuildConfig ``build`` would use for these options, with ``--phase``."""
    from dataclasses import replace

    from . import cli as cli_mod

    cfg = cli_mod.config_from_args(args)
    phase = getattr(args, "phase", None)
    if phase:
        cfg = replace(cfg, solvent="" if phase == "gas" else "water")
    return cfg


def _review(args) -> int:
    from . import cli as cli_mod
    from .chem import advisor

    try:
        items = cli_mod.read_inputs(args)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    if not items:
        print("Nothing to review: give SMILES, names or --file.", file=sys.stderr)
        return 2
    try:
        cfg = config_for(args)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    r = advisor.review(items, cfg, max_ids=max(args.ids, advisor.MAX_IDS))
    if args.json:
        print(json.dumps(r.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(r.text(width=max(60, args.width), max_ids=max(1, args.ids)))
    return 0


def print_header(items, cfg, out=print) -> None:
    """The short review before a build (``build``, ``grid build``). Never raises: a failed review is one line."""
    if not items:
        return
    try:
        from .chem import advisor

        out(advisor.review(items, cfg).header() + "\n")
    except Exception as exc:  # noqa: BLE001 - the review must never stop a build
        out(f"Review skipped ({type(exc).__name__}: {exc})\n")
