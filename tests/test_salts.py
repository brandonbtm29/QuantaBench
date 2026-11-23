"""Salts as separate entries (SETTINGS-1 follow-up, ``BuildConfig.counterion_entries``, ``chem.salts``): the
counter-ions a build removes become molecules of their own in the run, linked to their records by ``counterion_of``,
with the records' posted fields, one entry per InChIKey. A science setting."""

import csv
import dataclasses
import json
import sqlite3

import pytest
from rdkit import Chem

from quantabench import cli
from quantabench.chem import import_options as imp
from quantabench.chem import jobs, pipeline, salts
from quantabench.chem.inputs import from_smiles
from quantabench.dist import worker as worker_mod

K_PFOS = "[K+].[O-]S(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"
PFOS_ANION = "O=S(=O)([O-])C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"
K_PFHXS = "[K+].[O-]S(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"
NA_PFOA_HYDRATE = "O.[Na+].[O-]C(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"
CA_2PFOA = "[Ca+2].[O-]C(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F.[O-]C(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"


def _cfg(**kw):
    base = dict(method="mmff", max_conformers=2, refine_top=1, xtb_properties=False, rank_method="refine",
                threads=1, gpu="off", counterion_entries=True)
    return pipeline.BuildConfig(**{**base, **kw})


def test_which_fragments_are_counterions():
    def ions(smiles, keep=True, **strip):
        from quantabench.chem import standardize

        return salts.counterions_of(Chem.MolFromSmiles(smiles), keep, standardize.FragmentStrip(**strip))

    assert ions(K_PFOS) == [("[K+]", "potassium")]
    assert ions(NA_PFOA_HYDRATE) == [("[Na+]", "sodium")]                     # never water
    assert ions(CA_2PFOA) == [("[Ca+2]", "calcium")]                          # not the second copy of the anion
    assert ions("Cl.NCCc1ccccc1") == [("Cl", "chloride")]                      # HCl: the chloride of a hydrochloride
    assert ions("CCO.OC(=O)c1ccccc1") == []                                   # a neutral co-solvent is not an ion
    assert ions(K_PFOS, keep=False) == []                                     # nothing removed, nothing to add
    assert ions(NA_PFOA_HYDRATE, keep=False, counterions=("sodium",)) == [("[Na+]", "sodium")]   # stripped by name
    assert ions("CCO") == []


def test_k_pfos_gives_two_entries_and_ions_are_deduplicated():
    fields = {"CASRN": "2795-39-3", "cohort": "C4A"}
    kpfos = from_smiles(K_PFOS, id="kpfos", name="potassium PFOS")
    kpfos.meta[imp.FIELDS] = dict(fields)
    kpfhxs = from_smiles(K_PFHXS, id="kpfhxs")
    logs: list[str] = []
    out = salts.expand([kpfos, kpfhxs, from_smiles("CCO", id="ethanol")], _cfg(), logs.append)
    assert [x.id for x in out] == ["kpfos", "kpfhxs", "ethanol", "kpfos_K+"]   # one K+ for both salts
    k = out[-1]
    assert (k.smiles, k.name) == ("[K+]", "potassium") and k.meta[salts.COUNTERION_OF] == ["kpfos", "kpfhxs"]
    assert k.meta[imp.FIELDS] == fields and k.meta[imp.FIELDS] is not kpfos.meta[imp.FIELDS]
    assert logs == ["Counter-ions as separate entries: 1 added (kpfos_K+)"]
    assert salts.expand([kpfos], _cfg(counterion_entries=False)) == [kpfos]                    # off: unchanged
    given = salts.expand([kpfos, from_smiles("[K+]", id="potassium")], _cfg())                 # K+ given already
    assert [x.id for x in given] == ["kpfos", "potassium"]
    again = salts.expand(out, _cfg())                                          # an expanded list stays as it is
    assert [x.id for x in again] == [x.id for x in out]
    taken = salts.expand([kpfos, from_smiles("CC", id="kpfos_K+")], _cfg())   # the id is taken: _2
    assert [x.id for x in taken] == ["kpfos", "kpfos_K+", "kpfos_K+_2"]
    long = salts.expand([from_smiles(K_PFOS, id="x" * 70)], _cfg())
    assert len(long[-1].id) <= salts.MAX_ID and long[-1].id.endswith("_K+")


def test_a_run_computes_both_entries_and_links_them(tmp_path):
    kpfos = from_smiles(K_PFOS, id="kpfos")
    kpfos.meta[imp.FIELDS] = {"cohort": "C4A"}
    cfg = _cfg(output_dir=str(tmp_path / "runs"), formats=("xyz",))
    results, run_dir = jobs.run_resumable([kpfos], cfg, tmp_path / "run")
    by_id = {r.input.id: r for r in results}
    assert set(by_id) == {"kpfos", "kpfos_K+"} and all(r.status == "ok" for r in results), [r.error for r in results]
    assert by_id["kpfos"].smiles == PFOS_ANION and by_id["kpfos_K+"].smiles == "[K+]"
    with open(run_dir / "summary.csv", newline="", encoding="utf-8") as f:
        rows = {r["id"]: r for r in csv.DictReader(f)}
    assert rows["kpfos_K+"]["counterion_of"] == "kpfos" and rows["kpfos"]["counterion_of"] == ""
    assert rows["kpfos_K+"]["cohort"] == "C4A" and rows["kpfos_K+"]["charge"] == "1"
    with sqlite3.connect(run_dir / jobs.DB_NAME) as con:
        assert json.loads(con.execute("SELECT config FROM run").fetchone()[0])["counterion_entries"] is True
    # resumed with the same list: nothing new; a run without the setting refuses to resume it
    results, _ = jobs.run_resumable([kpfos], cfg, tmp_path / "run")
    assert jobs.open_run(tmp_path / "run").counts["ok"] == 2
    with pytest.raises(jobs.ConfigMismatchError) as err:
        jobs.run_resumable([kpfos], replace_cfg(cfg, counterion_entries=False), tmp_path / "run")
    assert set(err.value.differences) == {"counterion_entries"}
    # one process (pipeline.run) expands the same way
    results, _ = pipeline.run([from_smiles(K_PFOS, id="kpfos")], _cfg(write_files=False))
    assert [r.input.id for r in results] == ["kpfos", "kpfos_K+"]


def replace_cfg(cfg, **kw):
    return dataclasses.replace(cfg, **kw)


def test_a_chunked_grid_agrees_on_the_entries(tmp_path):
    """Each machine of a grid expands the whole list before taking its chunk: the entries are the same everywhere and
    each lands in exactly one chunk."""
    items = [from_smiles(K_PFOS, id="kpfos"), from_smiles(K_PFHXS, id="kpfhxs"), from_smiles(NA_PFOA_HYDRATE, id="napfoa")]
    expanded = [x.id for x in salts.expand(items, _cfg())]
    assert expanded[3:] == ["kpfos_K+", "napfoa_Na+"]
    chunks = [[x for x in expanded if jobs.in_chunk(x, (i, 2))] for i in range(2)]
    assert sorted(chunks[0] + chunks[1]) == sorted(expanded)


def test_science_setting_and_worker_checks(tmp_path, monkeypatch):
    from quantabench import settings as settings_mod

    assert "counterion_entries" in jobs.SCIENCE_FIELDS and pipeline.BuildConfig().counterion_entries is False
    s = settings_mod.Settings()
    s.build.counterion_entries = True
    assert pipeline.BuildConfig.from_settings(s).counterion_entries is True
    cfg = cli.config_from_args(cli.build_parser().parse_args(["build", "CCO", "--counterion-entries"]))
    assert cfg.counterion_entries is True
    # a run stored before the setting has none: it ran without counter-ion entries
    old = {k: v for k, v in dataclasses.asdict(pipeline.BuildConfig()).items() if k != "counterion_entries"}
    assert jobs.config_differences(old, pipeline.BuildConfig()) == {}
    # a grid worker takes true and false, refuses anything else; one from before the setting refuses the field
    assert worker_mod.validate_config({"counterion_entries": True}) == {"counterion_entries": True}
    assert worker_mod.validate_config({"counterion_entries": False}) == {"counterion_entries": False}
    for bad in ("yes", 1, None):
        with pytest.raises(worker_mod.RequestError, match="counterion_entries"):
            worker_mod.validate_config({"counterion_entries": bad})
    built = worker_mod.build_config({"counterion_entries": True, "method": "mmff"}, tmp_path)
    assert built.counterion_entries is True

    def copy(f):
        if f.default_factory is not dataclasses.MISSING:
            return dataclasses.field(default_factory=f.default_factory)
        return dataclasses.field(default=f.default)

    Old = dataclasses.make_dataclass("BuildConfig", [(f.name, f.type, copy(f)) for f in dataclasses.fields(  # noqa: N806
        pipeline.BuildConfig) if f.name != "counterion_entries"])
    monkeypatch.setattr(pipeline, "BuildConfig", Old)
    with pytest.raises(worker_mod.RequestError, match="unknown config field.*counterion_entries"):
        worker_mod.validate_config({"counterion_entries": True})
