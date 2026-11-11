"""Command line for PubChem (``chem.pubchem``, ``chem.pubchem_enrich``, ``chem.pubchem_targets``): ``pubchem enrich``,
``pubchem targets`` and ``pubchem lookup``.

    quantabench-cli pubchem enrich list.csv --out enriched.csv                     # CAS and/or SMILES columns
    quantabench-cli pubchem enrich RUN_DIR --validate --experimental --out enriched.csv --records experimental.csv
    quantabench-cli pubchem enrich list.csv --offline --out enriched.csv          # cache only, no network
    quantabench-cli pubchem lookup 335-67-1 --experimental                        # one identifier, printed
    quantabench-cli pubchem targets experimental.csv --out st25.csv --window 20,25  # one sigma(25 °C) per molecule

``register`` adds the ``pubchem`` command to the parser of ``quantabench.cli``.
"""

from __future__ import annotations

import sys
import time

ENRICH_DESCRIPTION = (
    "Look every molecule of a list up in PubChem and write one row per molecule with pubchem_* columns: CID(s), "
    "title, CAS RNs (synonyms + registry numbers, check digit verified), formula, weights, SMILES, InChI(Key), "
    "XLogP, TPSA, complexity, H-bond counts, charge. --validate compares QuantaBench's structure with PubChem's "
    "for the molecule's CAS RN (match, stereo_differs, charge_or_protonation_differs, component_of_mixture, "
    "different_structure, "
    "cas_not_in_pubchem, ambiguous_cas). --experimental adds PubChem's experimental values (PUG View: surface "
    "tension, density, boiling and melting points, vapor pressure, viscosity, solubility, logP, pKa, ...) as a "
    "summary per molecule and, with --records, every value in long form with its source, reference, temperature "
    "and condition (pure_liquid, aqueous_solution, unknown). Input: a CSV/Excel table (CAS and/or SMILES "
    "columns), an SD file, or a QuantaBench run folder / merged tables (summary.csv). Requests stay within "
    "PubChem's policy (<= 5/s, <= 400/min, slower when its throttling header says Yellow/Red) and every answer is "
    "cached on disk, so an interrupted run resumes and a rerun makes no new requests. See docs/SURFACE_TENSION.md.")


TARGETS_DESCRIPTION = (
    "Reads the long-form records of `pubchem enrich --experimental --records` and writes one row per molecule: the "
    "value at --t (default 25 °C) of one heading (default Surface Tension) in the wanted condition class (default "
    "pure_liquid), from records with a stated temperature within --window. Each record is moved to --t with the "
    "molecule's own temperature slope when its records span >= 10 K and the slope is plausible, else the default "
    "(-0.1 mN/m per K). One value per reference; references are tiered (compilation: CRC Handbook, Riddick, Jasper, "
    "Daubert & Danner/DIPPR, Lange, Yaws; other; chris: the U.S. Coast Guard manual) and the target is the median "
    "of the best tier present. The raw value, its temperature, the correction, the spread over all references and "
    "what was left out (review, estimated, no temperature, outside the window) stay in the row. "
    "See docs/SURFACE_TENSION.md, Data assembly.")


def register(sub) -> None:
    p = sub.add_parser("pubchem", help="PubChem: identifiers, computed properties, structure check, experimental "
                                       "values (surface tension, density, ...)")
    ps = p.add_subparsers(dest="pubchem_command", required=True)
    e = ps.add_parser("enrich", help="one row per molecule with pubchem_* columns (+ experimental records)",
                      description=ENRICH_DESCRIPTION)
    e.add_argument("input", help="CSV/Excel table, SD file, or QuantaBench run folder / summary.csv")
    e.add_argument("--id-col", help="column with the identifier (CAS RN, name or DTXSID; default: a CAS column, "
                                    "else id/name when it is a CAS RN)")
    e.add_argument("--smiles-col", help="column with the SMILES (default: input_smiles, smiles, qsar_ready_smiles...)")
    e.add_argument("--sheet", help="Excel sheet (name or 0-based index)")
    e.add_argument("--experimental", action="store_true", help="also fetch PubChem's experimental values")
    e.add_argument("--validate", action="store_true", help="compare QuantaBench's structure with PubChem's for the "
                                                           "CAS RN")
    e.add_argument("--out", required=True, help="enriched CSV to write")
    e.add_argument("--records", help="with --experimental: CSV of every experimental value (long form)")
    e.add_argument("--cache", metavar="DIR", help="response cache folder or .sqlite file (default: the per-machine "
                                                  "cache, QUANTABENCH_CACHE_DIR/pubchem)")
    e.add_argument("--offline", action="store_true", help="only read the cache; nothing is sent to PubChem")
    e.add_argument("--rate", type=float, default=4.0, help="requests per second (default 4; at most 5)")
    e.add_argument("--limit", type=int, help="only the first N molecules (a trial run)")
    e.add_argument("-q", "--quiet", action="store_true", help="no progress lines")
    e.set_defaults(func=enrich_command)

    tg = ps.add_parser("targets", help="one target value per molecule (e.g. pure-liquid surface tension at 25 °C) "
                                       "from the --records table",
                       description=TARGETS_DESCRIPTION)
    tg.add_argument("records", help="the long-form records CSV of `pubchem enrich --experimental --records`")
    tg.add_argument("--out", required=True, help="target table to write (CSV, one row per molecule)")
    tg.add_argument("--heading", default="Surface Tension", help="the property (default: Surface Tension)")
    tg.add_argument("--class", dest="classes", default="pure_liquid",
                    help="condition classes to use, comma-separated (default pure_liquid; 'any' for all)")
    tg.add_argument("--t", dest="target_t", type=float, default=25.0, help="target temperature, °C (default 25)")
    tg.add_argument("--window", default="20,30", help="temperatures of the records to use, °C (default 20,30)")
    tg.add_argument("--slope", type=float, help="the temperature slope per K when a molecule's own records do not "
                                                "give one (default: surface tension -0.1, density -0.001)")
    tg.add_argument("--slope-range", help="plausible own slopes, per K, as MIN,MAX (default: surface tension "
                                          "-0.3,-0.02; density -0.003,-0.0002)")
    tg.add_argument("--include-review", action="store_true", help="also use records marked needs_review")
    tg.add_argument("--include-estimated", action="store_true", help="also use estimated/calculated values")
    tg.add_argument("--allow-no-temperature", action="store_true",
                    help="use records without a stated temperature as if at the target (flagged)")
    tg.add_argument("--spread-flag", type=float, default=2.0,
                    help="flag molecules whose references differ by more than this (default 2)")
    tg.set_defaults(func=targets_command)

    lk = ps.add_parser("lookup", help="look one identifier up and print what PubChem has")
    lk.add_argument("identifier", help="CAS RN, name, InChIKey, SMILES or CID:<n>")
    lk.add_argument("--experimental", action="store_true", help="also print the experimental records")
    lk.add_argument("--cache", metavar="DIR")
    lk.add_argument("--offline", action="store_true")
    lk.set_defaults(func=lookup_command)


def _client(args):
    from .chem import pubchem as pc

    rate = float(getattr(args, "rate", pc.DEFAULT_RATE) or pc.DEFAULT_RATE)
    if rate > pc.MAX_RATE:
        print(f"--rate {rate:g} is above PubChem's limit; using {pc.MAX_RATE:g}/s", file=sys.stderr)
    return pc.PubChemClient(cache=args.cache or None, offline=args.offline, rate=rate,
                            log=lambda m: print(m, file=sys.stderr))


def enrich_command(args) -> int:
    from .chem import pubchem as pc
    from .chem import pubchem_enrich as pe

    try:
        queries = pe.read_queries(args.input, id_col=args.id_col, smiles_col=args.smiles_col, sheet=args.sheet,
                                  log=lambda m: print(m, file=sys.stderr))
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.limit:
        queries = queries[:args.limit]
    if not queries:
        print("error: no molecules in the input", file=sys.stderr)
        return 2
    try:
        client = _client(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    n_cas = sum(1 for q in queries if q.cas)
    print(f"{len(queries)} molecule(s), {n_cas} with a CAS RN; cache: "
          f"{client.cache.path if client.cache is not None else 'none'}{' (offline)' if args.offline else ''}")
    if not args.offline:
        est = pe.estimate_requests(len(queries), args.experimental)
        print(f"at most {est} request(s) with an empty cache: about {est / client.limiter.rate / 60:.0f} min at "
              f"{client.limiter.rate:g}/s while PubChem reports Green (up to 4x longer when it reports Red)")
    last = [0.0]

    def progress(i: int, n: int, stage: str) -> None:
        now = time.monotonic()
        if args.quiet or (now - last[0] < 30 and i != n):
            return
        last[0] = now
        print(f"  {stage}: {i}/{n}  ({client.requests} requests, {client.cache_hits} from the cache)", flush=True)

    try:
        res = pe.enrich(queries, client, experimental=args.experimental, validate=args.validate, progress=progress)
    except KeyboardInterrupt:
        print("interrupted: what was fetched is cached; rerun the same command to resume", file=sys.stderr)
        return 130
    out = pe.write_csv(args.out, res.rows, pe.output_columns(res.rows, args.experimental, args.validate))
    print(f"wrote {out}")
    if args.experimental and args.records:
        rec = pe.write_csv(args.records, res.records, ["id", "input_cas", "pubchem_inchikey", *pc.RECORD_COLUMNS])
        print(f"wrote {rec} ({len(res.records)} record(s))")
    print(res.summary())
    return 1 if res.stopped else 0


def _pair(text: str, what: str) -> tuple[float, float]:
    parts = [p for p in (text or "").replace(";", ",").split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"{what} needs two numbers, e.g. 20,30 (got {text!r})")
    a, b = (float(p) for p in parts)
    return (min(a, b), max(a, b))


def targets_command(args) -> int:
    from .chem import pubchem_targets as pt

    try:
        classes = tuple(c.strip() for c in args.classes.split(",") if c.strip())
        s = pt.Settings(heading=args.heading, classes=() if classes == ("any",) else classes,
                        target_t=args.target_t, window=_pair(args.window, "--window"), slope=args.slope,
                        slope_range=_pair(args.slope_range, "--slope-range") if args.slope_range else None,
                        include_review=args.include_review, include_estimated=args.include_estimated,
                        allow_no_temperature=args.allow_no_temperature, spread_flag=args.spread_flag)
        records = pt.read_records(args.records)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    rows = pt.targets(records, s)
    if not rows:
        print(f"error: no {args.heading!r} records in {args.records}", file=sys.stderr)
        return 1
    out = pt.write_csv(args.out, rows)
    print(f"wrote {out}")
    print(pt.summary(rows))
    return 0


def lookup_command(args) -> int:
    from .chem import pubchem as pc
    from .chem import pubchem_enrich as pe

    ident = args.identifier.strip()
    q = pe.Query(id=ident)
    if ident.upper().startswith("CID:") and ident[4:].strip().isdigit():
        q.cid = int(ident[4:])
    elif pc.is_cas(ident):
        q.cas = pc.normalize_cas(ident)
    elif len(ident) == 27 and ident[14] == "-" and ident[25] == "-":
        q.inchikey = ident
    else:
        from .chem.inputs import looks_like_smiles

        if looks_like_smiles(ident):
            q.smiles = ident
        else:
            q.name = ident
    try:
        client = pc.PubChemClient(cache=args.cache or None, offline=args.offline)
        res = pe.enrich([q], client, experimental=args.experimental)
    except (ValueError, pc.PubChemError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    row = res.rows[0]
    for k in pe.PUBCHEM_COLUMNS + (pe.EXPERIMENTAL_COLUMNS if args.experimental else ()):
        if row.get(k, "") != "":
            print(f"{k:32s} {row[k]}")
    for r in res.records:
        if r.get("value_std") != "" or r["heading"] in ("Surface Tension",):
            print(f"  {r['heading']:22s} {r['raw'][:60]:60s} -> {r['value_std']} {r['unit_std']}"
                  f"{' at ' + str(r['temperature_c']) + ' °C' if r['temperature_c'] != '' else ''}"
                  f"  [{r['condition_class'] or '-'}] {r['source_name']}")
    return 0 if row["pubchem_status"] in ("ok", "ambiguous") else 1

