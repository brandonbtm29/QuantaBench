"""Build defaults (SETTINGS-1): Spartan's Import Filters as build filters (water, acids, named counter-ions stripped
before the largest-fragment rule) and its Conformer Rules (twist-boat ring conformers). Both are science settings:
in ``jobs.SCIENCE_FIELDS``, stored with each run, checked by ``dist.worker.validate_config``."""

import dataclasses
import json
import sqlite3
from dataclasses import replace

import pytest
from rdkit import Chem
from rdkit.Chem import rdMolTransforms

from quantabench import cli
from quantabench import settings as settings_mod
from quantabench.chem import conformers, jobs, pipeline, standardize
from quantabench.chem.inputs import from_smiles
from quantabench.dist import worker as worker_mod

#: sodium perfluorooctanoate monohydrate, as a supplier's record draws it
NA_PFOA_HYDRATE = "O.[Na+].[O-]C(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"
PFOA_ANION = "O=C([O-])C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"
#: tetrabutylammonium trifluoroacetate: the counter-ion is the larger ion
TBA_TFA = "CCCC[N+](CCCC)(CCCC)CCCC.O=C([O-])C(F)(F)F"


def _prepared(smiles, keep_largest=True, **strip):
    info: dict = {}
    mol, notes = standardize.prepare(Chem.MolFromSmiles(smiles), 7.0, keep_largest, info=info,
                                     strip=standardize.FragmentStrip(**strip))
    return Chem.MolToSmiles(Chem.RemoveHs(mol)), info, notes


def _cfg(**kw):
    """MMFF only, ranked by the refinement (no AIMNet2 or xTB step), no files."""
    base = dict(method="mmff", max_conformers=3, refine_top=1, xtb_properties=False, rank_method="refine",
                write_files=False, threads=1, gpu="off")
    return pipeline.BuildConfig(**{**base, **kw})


# ------------------------------------------------------------------------------------------- build filters
def test_counterion_names_are_checked_and_sorted():
    assert standardize.counterion_names("Sodium, potassium;sodium") == ("potassium", "sodium")
    assert standardize.canonical_counterions(["tetrabutylammonium", "chloride"]) == "chloride,tetrabutylammonium"
    assert standardize.canonical_counterions("") == ""
    with pytest.raises(ValueError, match="unknown counter-ion.*natrium"):
        standardize.counterion_names("sodium,natrium")
    assert standardize.counterion_names("sodium,natrium", strict=False) == ("sodium",)
    s = standardize.FragmentStrip.of(_cfg(strip_water=True, strip_counterions="sodium"))
    assert s and s.label() == "water, sodium" and not standardize.FragmentStrip()


def test_sodium_pfoa_hydrate_with_and_without_the_filters():
    """With keep largest fragment on, the hydrate's PFOA anion is computed either way, and the notes say which rule
    removed what. With it off, the filters decide what is computed with the anion."""
    smi, info, notes = _prepared(NA_PFOA_HYDRATE)
    assert smi == PFOA_ANION and info["removed_fragments"] == ["O", "[Na+]"] and info["stripped_fragments"] == []
    assert "kept the largest of 3 fragments; removed O, [Na+]" in notes
    smi, info, notes = _prepared(NA_PFOA_HYDRATE, water=True)
    assert smi == PFOA_ANION and info["removed_fragments"] == ["O", "[Na+]"]
    assert info["stripped_fragments"] == [("O", "water")] and info["n_fragments"] == 3
    assert notes[:2] == ["stripped before the largest-fragment rule (build filters): O (water)",
                         "kept the largest of 2 fragments; removed [Na+]"]

    smi, info, notes = _prepared(NA_PFOA_HYDRATE, keep_largest=False)
    assert smi == "O." + PFOA_ANION + ".[Na+]" and "3 fragments computed together" in notes[0]
    smi, info, notes = _prepared(NA_PFOA_HYDRATE, keep_largest=False, water=True)
    assert smi == PFOA_ANION + ".[Na+]" and info["removed_fragments"] == ["O"]           # the ion pair, dry
    assert "2 fragments computed together (keep largest fragment is off)" in notes
    smi, info, notes = _prepared(NA_PFOA_HYDRATE, keep_largest=False, water=True, counterions=("sodium",))
    assert smi == PFOA_ANION and info["removed_fragments"] == ["O", "[Na+]"]
    assert "O (water), [Na+] (sodium)" in notes[0]


def test_a_named_counterion_is_stripped_even_when_it_is_the_larger_ion():
    smi, info, _notes = _prepared(TBA_TFA)
    assert smi == "CCCC[N+](CCCC)(CCCC)CCCC"                 # the largest-fragment rule alone keeps the cation
    smi, info, notes = _prepared(TBA_TFA, counterions=("tetrabutylammonium",))
    assert smi == "O=C([O-])C(F)(F)F" and info["removed_fragments"] == ["CCCC[N+](CCCC)(CCCC)CCCC"]
    # a salt drawn as two neutral molecules: the named ion matches its neutral form (triethylamine)
    smi, *_ = _prepared("CCN(CC)CC.OC(=O)C(F)(F)F", keep_largest=False, counterions=("triethylammonium",))
    assert smi == "O=C([O-])C(F)(F)F"


def test_acids_and_never_everything():
    smi, info, _notes = _prepared("Cl.NCCc1ccccc1", keep_largest=False, acids=True)
    assert smi == "[NH3+]CCc1ccccc1" and info["stripped_fragments"] == [("Cl", "acid")]
    smi, info, notes = _prepared("[Na+].[Cl-]", keep_largest=False, counterions=("chloride", "sodium"))
    assert len(info["stripped_fragments"]) == 1 and "every fragment matched the strip filters" in " ".join(notes)
    smi, info, _notes = _prepared("O", water=True)                # one fragment: nothing to strip
    assert smi == "O" and info["stripped_fragments"] == []


def test_the_pipeline_uses_the_filters():
    off = pipeline.run_molecule(from_smiles(NA_PFOA_HYDRATE, id="napfoa"), _cfg(keep_largest_fragment=False))
    assert off.status == "ok", off.error
    assert off.smiles == "O." + PFOA_ANION + ".[Na+]" and off.charge == 0 and off.removed_fragments == []
    on = pipeline.run_molecule(from_smiles(NA_PFOA_HYDRATE, id="napfoa"),
                               _cfg(keep_largest_fragment=False, strip_water=True, strip_counterions="sodium"))
    assert on.status == "ok", on.error
    assert on.smiles == PFOA_ANION and on.charge == -1 and on.removed_fragments == ["O", "[Na+]"]
    assert any(n.startswith("stripped before the largest-fragment rule") for n in on.notes)
    bad = pipeline.run_molecule(from_smiles(NA_PFOA_HYDRATE, id="napfoa"), _cfg(strip_counterions="natrium"))
    assert bad.status == "failed" and "unknown counter-ion" in bad.error


# ------------------------------------------------------------------------------------------- ring conformers
def _ring_torsions(mol, cid, ring):
    conf = mol.GetConformer(cid)
    return [rdMolTransforms.GetDihedralDeg(conf, *(ring[(k + j) % 6] for j in range(4))) for k in range(6)]


def _is_chair(torsions) -> bool:
    signs = [t > 0 for t in torsions]
    return all(40 < abs(t) < 75 for t in torsions) and all(signs[k] != signs[k - 1] for k in range(6))


def test_twist_boats_of_cyclohexane():
    mol = Chem.AddHs(Chem.MolFromSmiles("C1CCCCC1"))
    cs = conformers.generate(mol, 5, 10.0, 0.5, seed=42, timeout_s=0)
    ring = conformers.flexible_six_rings(cs.mol)[0]
    chair = next(c for c in cs.conf_ids if _is_chair(_ring_torsions(cs.mol, c, ring)))
    m = Chem.Mol(cs.mol)
    notes: list[str] = []
    ids, energies = conformers.twist_boats(m, [chair], [0.0], "MMFF94s", notes)
    assert len(ids) == 6 and "6 twist-boat start(s) of 1 six-membered ring(s)" in notes[0]   # 3 rotations, mirrored
    chair_e = cs.energies_kcal[cs.conf_ids.index(chair)]
    for cid, e in zip(ids, energies):
        tors = _ring_torsions(m, cid, ring)
        assert not _is_chair(tors) and 4.5 < e - chair_e < 7.5          # a twist-boat: about +5.9 kcal/mol in MMFF94s
        assert sorted(round(abs(t) / 5) * 5 for t in tors)[-2:] in ([60, 60], [60, 65], [65, 65])
    assert conformers.flexible_six_rings(Chem.AddHs(Chem.MolFromSmiles("c1ccccc1C1CCC=CC1"))) == []
    assert conformers.twist_boats(Chem.AddHs(Chem.MolFromSmiles("CCCCO")), [0], [0.0], "MMFF94s") == ([], [])


def test_thorough_ring_conformers_add_twist_boats():
    """trans-Decalin-1-ol: the thorough search finds twist-boat conformers the normal one does not; an acyclic
    molecule gets the same conformers either way."""
    mol = Chem.AddHs(Chem.MolFromSmiles("O[C@@H]1CCC[C@@H]2CCCC[C@H]12"))

    def non_chairs(cs):
        rings = conformers.flexible_six_rings(cs.mol)
        return sum(1 for c in cs.conf_ids for r in rings if not _is_chair(_ring_torsions(cs.mol, c, r)))

    normal = conformers.generate(mol, 50, 10.0, 0.5, seed=42, timeout_s=0)
    thorough = conformers.generate(mol, 50, 10.0, 0.5, seed=42, timeout_s=0, ring_conformers="thorough")
    assert non_chairs(thorough) > non_chairs(normal) and len(thorough.conf_ids) >= len(normal.conf_ids)
    assert any("twist-boat" in n and "kept after pruning" in n for n in thorough.notes)
    assert not any("twist-boat" in n for n in normal.notes)
    capped = conformers.generate(mol, 3, 10.0, 0.5, seed=42, timeout_s=0, ring_conformers="thorough")
    assert len(capped.conf_ids) <= 3                         # Max conformers still caps the pool
    chain = Chem.AddHs(Chem.MolFromSmiles("OCCCCCC"))
    a = conformers.generate(chain, 10, 10.0, 0.5, seed=42, timeout_s=0)
    b = conformers.generate(chain, 10, 10.0, 0.5, seed=42, timeout_s=0, ring_conformers="thorough")
    assert a.energies_kcal == b.energies_kcal
    with pytest.raises(ValueError, match="ring_conformers"):
        conformers.generate(chain, 10, ring_conformers="exhaustive")


def test_the_pipeline_passes_the_ring_option(monkeypatch):
    seen = []
    real = conformers.generate

    def spy(*args, **kw):
        seen.append(kw.get("ring_conformers"))
        return real(*args, **kw)

    monkeypatch.setattr(conformers, "generate", spy)
    r = pipeline.run_molecule(from_smiles("OC1CCCCC1", id="cyclohexanol"), _cfg(ring_conformers="thorough"))
    assert r.status == "ok", r.error
    assert seen == ["thorough"]
    assert any("twist-boat" in n for n in r.notes)


# ------------------------------------------------------------------------------------------- science settings
def test_science_fields_settings_and_cli(monkeypatch):
    new = {"strip_water", "strip_acids", "strip_counterions", "ring_conformers"}
    assert new <= set(jobs.SCIENCE_FIELDS)
    d = pipeline.BuildConfig()
    assert (d.strip_water, d.strip_acids, d.strip_counterions, d.ring_conformers) == (False, False, "", "normal")
    s = settings_mod.Settings()
    s.build.strip_water, s.build.strip_counterions, s.build.ring_conformers = True, "Sodium, potassium, natrium", "x"
    cfg = pipeline.BuildConfig.from_settings(s)                # unknown names and choices fall back, sorted
    assert (cfg.strip_water, cfg.strip_counterions, cfg.ring_conformers) == (True, "potassium,sodium", "normal")
    settings_mod.save(s)
    again = settings_mod.load()
    assert (again.build.strip_water, again.build.strip_counterions) == (True, "Sodium, potassium, natrium")

    args = cli.build_parser().parse_args(["build", "CCO", "--strip", "water,HCl,tetrabutylammonium",
                                          "--ring-conformers", "thorough"])
    cfg = cli.config_from_args(args)
    assert (cfg.strip_water, cfg.strip_acids, cfg.strip_counterions, cfg.ring_conformers) == \
        (True, True, "tetrabutylammonium", "thorough")
    cfg = cli.config_from_args(cli.build_parser().parse_args(["build", "CCO", "--strip", "none"]))
    assert (cfg.strip_water, cfg.strip_acids, cfg.strip_counterions) == (False, False, "")
    with pytest.raises(SystemExit, match="unknown counter-ion"):
        cli.config_from_args(cli.build_parser().parse_args(["build", "CCO", "--strip", "natrium"]))


def test_a_run_records_the_filters_and_resumes_only_with_them(tmp_path):
    run_dir = tmp_path / "run"
    first = _cfg(output_dir=str(tmp_path / "runs"), formats=("xyz",), write_files=True, keep_largest_fragment=False,
                 strip_water=True, strip_counterions="sodium")
    results, _ = jobs.run_resumable([from_smiles(NA_PFOA_HYDRATE, id="napfoa")], first, run_dir)
    assert results[0].smiles == PFOA_ANION
    with sqlite3.connect(run_dir / jobs.DB_NAME) as con:
        stored = json.loads(con.execute("SELECT config FROM run").fetchone()[0])
    assert (stored["strip_water"], stored["strip_acids"], stored["strip_counterions"], stored["ring_conformers"]) == \
        (True, False, "sodium", "normal")
    assert jobs.stored_config(stored).strip_counterions == "sodium"
    for change in ({"strip_water": False}, {"strip_counterions": "potassium,sodium"}, {"ring_conformers": "thorough"}):
        with pytest.raises(jobs.ConfigMismatchError) as err:
            jobs.run_resumable([from_smiles("CCO", id="ethanol")], replace(first, **change), run_dir)
        assert set(err.value.differences) == set(change)
    # a run stored before SETTINGS-1 has none of the fields: it ran without filters, with normal ring conformers
    old = {k: v for k, v in dataclasses.asdict(pipeline.BuildConfig()).items()
           if k not in ("strip_water", "strip_acids", "strip_counterions", "ring_conformers")}
    assert jobs.config_differences(old, pipeline.BuildConfig()) == {}
    assert set(jobs.config_differences(old, pipeline.BuildConfig(strip_acids=True))) == {"strip_acids"}


def test_a_grid_worker_checks_the_filters(tmp_path):
    """validate_config accepts the client's filters as BuildConfig.from_settings stores them, and refuses a value
    that would make the worker compute something else than the client asked for."""
    cfg = dataclasses.asdict(pipeline.BuildConfig(strip_water=True, strip_acids=True,
                                                  strip_counterions="potassium,sodium", ring_conformers="thorough"))
    cfg.pop("output_dir")
    out = worker_mod.validate_config(cfg)
    assert (out["strip_water"], out["strip_acids"], out["strip_counterions"], out["ring_conformers"]) == \
        (True, True, "potassium,sodium", "thorough")
    assert worker_mod.validate_config({"strip_counterions": ""})["strip_counterions"] == ""
    for bad, match in (({"ring_conformers": "exhaustive"}, "one of normal, thorough"),
                       ({"strip_counterions": "sodium,natrium"}, "unknown counter-ion"),
                       ({"strip_counterions": "sodium,potassium"}, "sorted"),
                       ({"strip_counterions": ["sodium"]}, "string"),
                       ({"strip_water": "yes"}, "strip_water")):
        with pytest.raises(worker_mod.RequestError, match=match):
            worker_mod.validate_config(bad)
    # an older client's config has none of the fields: the worker builds with the defaults, as that client did
    built = worker_mod.build_config({"method": "mmff"}, tmp_path)
    assert (built.strip_water, built.strip_counterions, built.ring_conformers) == (False, "", "normal")
    built = worker_mod.build_config(out, tmp_path)
    assert (built.strip_water, built.strip_counterions, built.ring_conformers) == (True, "potassium,sodium", "thorough")


def test_a_worker_from_before_settings_1_refuses_the_filters(monkeypatch):
    """A worker whose BuildConfig has no build filters refuses a config that sets them (unknown fields) instead of
    computing without them: one grid never mixes filtered and unfiltered molecules."""
    def copy(f):
        if f.default_factory is not dataclasses.MISSING:
            return dataclasses.field(default_factory=f.default_factory)
        return dataclasses.field(default=f.default)

    old_fields = [(f.name, f.type, copy(f)) for f in dataclasses.fields(pipeline.BuildConfig)
                  if f.name not in ("strip_water", "strip_acids", "strip_counterions", "ring_conformers")]
    Old = dataclasses.make_dataclass("BuildConfig", old_fields)          # noqa: N806
    monkeypatch.setattr(pipeline, "BuildConfig", Old)
    with pytest.raises(worker_mod.RequestError, match="unknown config field.*strip_counterions.*strip_water"):
        worker_mod.validate_config({"method": "mmff", "strip_water": True, "strip_counterions": "sodium"})
    assert worker_mod.validate_config({"method": "mmff"}) == {"method": "mmff"}
