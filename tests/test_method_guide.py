"""The methods guide (chem.method_guide): coverage of every choice the GUI and the pipeline offer, name matching,
rendering, the CLI and docs/METHODS.md."""

import re
import subprocess
import sys
from pathlib import Path

import pytest

from quantabench.chem import engines, solvents
from quantabench.chem import method_guide as guide

ROOT = Path(__file__).resolve().parents[1]


def _gui_lists():
    pytest.importorskip("PySide6")
    from quantabench.gui import main_window

    return main_window


# ------------------------------------------------------------------ coverage
def test_every_engine_has_an_entry_and_an_implementation_line():
    """A new engine without an explanation fails here: add its entry (or map it in ENGINE_ENTRIES) and its line."""
    for key in engines.method_keys():                  # "auto" and every registered engine, plugins included
        e = guide.for_engine(key)
        assert e is not None, f"engine {key!r} has no methods-guide entry (method_guide.ENGINE_ENTRIES)"
        if key != "auto":
            assert guide.engine_note(key), f"engine {key!r} has no line in {e.key}.implementations"
    assert set(guide.ENGINE_ENTRIES) >= set(engines.method_keys())


def test_every_gui_choice_has_an_entry():
    mw = _gui_lists()
    for k, _label in mw.CREST_METHODS:
        assert k in guide.CREST_LEVEL_ENTRIES and guide.CREST_LEVEL_ENTRIES[k] in guide.ENTRIES, k
    for k, _label in mw.OPTIMIZERS:
        assert guide.OPTIMIZER_ENTRIES.get(k) in guide.ENTRIES, k
    for k, _label in mw.CONFORMER_SEARCHES:
        assert guide.CONFORMER_SEARCH_ENTRIES.get(k) in guide.ENTRIES, k
    for name in (*mw.QM_FUNCTIONALS, *mw.QM_BASES):
        e = guide.lookup(name)
        assert e is not None, f"QM choice {name!r} has no methods-guide entry"
        assert e.family in ("DFT functionals and HF", "Basis sets"), (name, e.family)
    # the pipeline's own option values too, not only what the GUI lists
    from quantabench.chem import crest

    for k in crest.METHOD_FLAGS:
        assert k in guide.CREST_LEVEL_ENTRIES, k


def test_every_solvation_model_has_an_entry():
    models = {m for info in engines.all_engines().values() for m in info.solvation} | {"pcm", ""}
    for m in models:
        assert guide.SOLVATION_MODEL_ENTRIES.get(m) in guide.ENTRIES, m
    # what the Solvent box shows: the engine's default model, gas phase when it has none, the overview for auto
    assert guide.solvation_entry("tblite-gfn2", "water", ("alpb", "gbsa", "cpcm")).key == "alpb"
    assert guide.solvation_entry("mopac-pm7", "water", ("cosmo",)).key == "cosmo"
    assert guide.solvation_entry("aimnet2", "water", ()).key == "gas-phase"
    assert guide.solvation_entry("auto", "water").key == "solvation"
    assert guide.solvation_entry("xtb-gfn2", "", ("alpb",)).key == "gas-phase"
    assert set(solvents.SOLVENTS)          # every solvent maps through the same models (no per-solvent entries)


def test_spartan_levels_and_concepts_are_there():
    for key in ("spartan-b3lyp", "spartan-wb97x-d", "spartan-pm6", "dispersion", "basis-sets", "solvation",
                "single-point", "boltzmann", "homo-lumo", "b3lyp", "crest"):
        assert key in guide.ENTRIES


# ------------------------------------------------------------------ data quality
def test_entries_are_complete_and_links_resolve():
    anchors = [e.anchor for e in guide.ENTRIES.values()]
    assert len(anchors) == len(set(anchors))
    assert all(re.fullmatch(r"[a-z0-9-]+", a) for a in anchors)
    for e in guide.ENTRIES.values():
        assert e.family in guide.FAMILIES
        for field in ("name", "summary", "what", "good_for", "watch_out", "cost", "dispersion", "solvation"):
            assert getattr(e, field).strip(), (e.key, field)
        summary = guide.plain(e.summary)
        assert len(summary) <= 120 and "\n" not in summary, (e.key, len(summary))    # one line in the GUI
        assert len(e.references) <= 3, e.key
        for r in e.references:
            assert r.doi.startswith("10.") and r.year > 1900 and r.journal and r.authors, (e.key, r)
            assert r.url.startswith("https://doi.org/10.")
        texts = (e.summary, e.what, e.good_for, e.watch_out, e.cost, e.dispersion, e.solvation,
                 *(t for _k, t in e.implementations))
        for t in texts:
            for m in re.finditer(r"\[\[([^\]|]+)", t):
                assert m.group(1) in guide.ENTRIES, (e.key, m.group(1))
            assert t.count("`") % 2 == 0, (e.key, t)
        for k in e.see_also:
            assert k in guide.ENTRIES and k != e.key, (e.key, k)
        for k, _note in e.implementations:
            assert guide.ENGINE_ENTRIES.get(k) == e.key, (e.key, k)


def test_names_do_not_collide():
    """Every spelling (key, anchor, name, alias) of an entry finds that entry, not another one."""
    for e in guide.ENTRIES.values():
        for name in (e.key, e.anchor, e.name, *e.aliases):
            assert guide.lookup(name) is e, (name, e.key, getattr(guide.lookup(name), "key", None))


# ------------------------------------------------------------------ matching
@pytest.mark.parametrize("name, key", [
    ("wb97x-d", "wb97x-d"), ("ωB97X-D", "wb97x-d"), ("WB97XD", "wb97x-d"), ("wB97X-D", "wb97x-d"),
    ("gfn2", "gfn2-xtb"), ("GFN2-xTB", "gfn2-xtb"), ("tblite-gfn2", "gfn2-xtb"), ("xtb-gfn1", "gfn1-xtb"),
    ("mopac-pm6", "pm6-d3h4x"), ("PM7", "pm7"), ("mmff", "mmff94s"), ("gfnff", "gfn-ff"), ("GFN-FF", "gfn-ff"),
    ("B3LYP-D3(BJ)", "b3lyp-d3bj"), ("b3lyp", "b3lyp"), ("PBE0", "pbe0-d3bj"), ("M06-2X", "m06-2x"),
    ("6-31G(d)", "6-31g*"), ("6-31g*", "6-31g*"), ("6-31+G*", "6-31+g*"), ("6-311+G(d,p)", "6-311+g**"),
    ("def2svp", "def2-svp"), ("DEF2-TZVP", "def2-tzvp"), ("aug-cc-pVDZ", "aug-cc-pvdz"),
    ("gfn2//gfnff", "crest-gfn2//gfnff"), ("crest", "crest"), ("rdkit", "rdkit-etkdg"), ("ETKDGv3", "rdkit-etkdg"),
    ("D3(BJ)", "dispersion"), ("D4", "dispersion"), ("ALPB", "alpb"), ("C-PCM", "pcm"), ("cpcm", "pcm"),
    ("IEF-PCM", "pcm"), ("smd", "smd"), ("native", "native-optimizer"), ("L-BFGS", "lbfgs"),
    ("Boltzmann", "boltzmann"), ("HOMO", "homo-lumo"), ("spartan pm6", "spartan-pm6"), ("6-311pgss", "6-311+g**"),
    ("ωB97M-V", "wb97m"), ("auto", "auto"),
])
def test_loose_name_matching(name, key):
    e = guide.lookup(name)
    assert e is not None and e.key == key, (name, getattr(e, "key", None))


def test_unknown_names_levels_and_search():
    assert guide.lookup("not-a-method") is None
    assert guide.lookup("6-31g") is None                       # 6-31G is not 6-31G*: '*' is not loose
    assert [e.key for e in guide.lookup_level("wB97X-D/def2-SVP")] == ["wb97x-d", "def2-svp"]
    assert [e.key for e in guide.lookup_level("b3lyp-d3bj/6-311+G**")] == ["b3lyp-d3bj", "6-311+g**"]
    assert "anion" in guide.plain(guide.get("def2-svpd").good_for).lower()
    hits = [e.key for e in guide.search("anion")]
    assert "def2-svpd" in hits and "basis-sets" in hits
    assert [e.key for e in guide.search("wb97x-d")] == ["wb97x-d"]
    assert guide.search("") == list(guide.ENTRIES.values())
    assert "wb97x-d" in guide.suggestions("wb97x")


# ------------------------------------------------------------------ rendering
def test_rendering_links_code_and_escapes():
    e = guide.get("gfn2-xtb")
    html = guide.html_entry(e)
    assert '<a href="guide:dispersion">' in html and "https://doi.org/10.1021/acs.jctc.8b01176" in html
    assert "[[" not in html and "`" not in html
    text = guide.text_entry(e)
    assert "[[" not in text and "tblite-gfn2:" in text and "xtb-gfn2:" in text and "gfn2-xtb:" in text
    assert max(len(line) for line in text.splitlines()) <= 100
    md = guide.markdown_entry(guide.get("6-31g*"))
    assert "### 6-31G\\*" in md and '<a id="6-31gs"></a>' in md
    mmff = guide.markdown_entry(guide.get("mmff94s"))      # a DOI with < > ( ): percent-encoded in the link
    assert "https://doi.org/10.1002/%28SICI%291096-987X%28199604%2917%3A5/6%3C490" in mmff
    s = guide.summary_html(guide.get("b3lyp"))
    assert s.startswith("<b>B3LYP (no dispersion)</b>: ") and s.endswith('<a href="guide:b3lyp">More…</a>')


def test_methods_doc_is_up_to_date():
    doc = (ROOT / "docs" / "METHODS.md").read_text(encoding="utf-8")
    assert doc == guide.markdown_document(), "docs/METHODS.md is out of date: run python scripts/write_methods_doc.py"
    for e in guide.ENTRIES.values():
        assert f'<a id="{e.anchor}"></a>' in doc
    for anchor in set(re.findall(r"\]\(#([a-z0-9-]+)\)", doc)):
        assert guide.by_anchor(anchor) is not None, anchor


def test_write_methods_doc_check_mode():
    cp = subprocess.run([sys.executable, str(ROOT / "scripts" / "write_methods_doc.py"), "--check"],
                        capture_output=True, text=True, timeout=120)
    assert cp.returncode == 0, cp.stderr
    assert "up to date" in cp.stdout


def test_readme_and_engines_link_the_guide():
    assert "docs/METHODS.md" in (ROOT / "README.md").read_text(encoding="utf-8")
    assert "METHODS.md" in (ROOT / "docs" / "ENGINES.md").read_text(encoding="utf-8")


# ------------------------------------------------------------------ CLI
def test_cli_methods_list_and_entries(capsys):
    from quantabench import cli

    assert cli.main(["methods"]) == 0
    out = capsys.readouterr().out
    for family in guide.FAMILIES:
        assert family in out
    assert "wb97x-d" in out and "crest-gfn2//gfnff" in out

    assert cli.main(["methods", "wb97x-d"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("ωB97X-D  [DFT functionals and HF]") and "Chai, Head-Gordon" in out
    assert "https://doi.org/10.1039/B810189B" in out

    assert cli.main(["methods", "GFN2"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("GFN2-xTB") and "tblite-gfn2:" in out and "dxtb" in out

    assert cli.main(["methods", "ωB97X-D/def2-SVP"]) == 0
    out = capsys.readouterr().out
    assert "ωB97X-D  [" in out and "def2-SVP  [" in out

    assert cli.main(["methods", "--search", "anion"]) == 0
    assert "def2-svpd" in capsys.readouterr().out


def test_cli_methods_unknown_name(capsys):
    from quantabench import cli

    assert cli.main(["methods", "wb97"]) in (0, 1)            # several entries mention it: listed, not guessed
    captured = capsys.readouterr()
    assert "wb97x-d" in captured.out + captured.err
    assert cli.main(["methods", "zzz-not-a-method"]) == 1
    assert "No entry named" in capsys.readouterr().err


# ------------------------------------------------------------------ what each method corrects (ADV-1)
def test_decision_lines_render_everywhere():
    assert set(guide.DECISIONS) <= set(guide.ENTRIES)
    for key in ("dispersion", "gcp", "def2-svpd", "pcm", "alpb", "smd", "r2scan-3c", "pm6-d3h4x", "gfn2-xtb"):
        e = guide.get(key)
        assert e.corrects, key
        assert f"Corrects: {guide.plain(e.corrects)[:30]}" in guide.text_entry(e, width=400), key
        assert "<b>Corrects:</b>" in guide.html_entry(e) and "- **Corrects:**" in guide.markdown_entry(e)
    assert "cavity or dispersion" in guide.get("pcm").corrects and "non-polar" in guide.get("alpb").corrects
    assert "superposition" in guide.get("gcp").corrects
    assert guide.get("aimnet2").avoid_for.startswith("Open shells")
    assert [e.key for e in guide.search("superposition")][:1] == ["gcp"]


def test_new_entries_and_moved_aliases():
    for name, key in (("r2SCAN-3c", "r2scan-3c"), ("HF-3c", "r2scan-3c"), ("g-xTB", "g-xtb"), ("gxtb", "g-xtb"),
                      ("BSSE", "gcp"), ("counterpoise", "gcp"), ("pKa", "protonation"), ("rank step", "conformer-ranking"),
                      ("r2scan", "r2scan"), ("hf", "hf")):
        assert guide.lookup(name).key == key, name
    assert guide.R_GXTB.citation().endswith("ChemRxiv (preprint) (2025).")      # no volume or pages: no stray commas
    assert "Chem. Sci. 16, 10228-10244 (2025)." in guide.R_AIMNET2.citation()
