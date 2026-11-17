"""CONF-AUTO: the conformer search "auto" (CREST for flexible neutral molecules, RDKit for the rest) and the merged
CREST + RDKit pool (chem.search_rule), with what uses them: the CLI options, resuming an older run folder, the advisor
and the job recipes. The pipeline runs with a fake CREST are in tests/test_crest.py."""

import dataclasses

import pytest
from rdkit import Chem

from quantabench import cli
from quantabench.chem import advisor, jobs, search_rule
from quantabench.chem import conformers as confs
from quantabench.chem.inputs import from_smiles
from quantabench.chem.pipeline import BuildConfig

DODECANOL = "CCCCCCCCCCCCO"           # 10 rotatable bonds
DECANOL = "CCCCCCCCCCO"               # 8
ANION = "[O-]C(=O)CCCCCCCCCCC"        # 10, charged
ZWITTERION = "[NH3+]CCCCCCCCCCCC([O-])=O"
NITRO = "[O-][N+](=O)CCCCCCCCCCC"     # a +/- pair: neutral


def _m(smi):
    return Chem.AddHs(Chem.MolFromSmiles(smi))


def test_charge_class_and_rotors_match_the_advisor_and_the_checks():
    assert [search_rule.charge_class(_m(s)) for s in (DODECANOL, ANION, ZWITTERION, NITRO, "C[NH3+]")] == \
        ["neutral", "anion", "zwitterion", "neutral", "cation"]
    assert search_rule.charge_class(_m(DODECANOL), charge=-1) == "anion"      # the input's own charge wins
    assert [search_rule.rotatable_bonds(_m(s)) for s in (DODECANOL, DECANOL, "CCO", "CCCCCCCCCCCC")] == [10, 8, 0, 9]
    # the advisor profiles molecules the same way (its rule for the cost and the findings)
    mols = dict(advisor.molecule_features([from_smiles(s, id=i) for i, s in
                                           (("dod", DODECANOL), ("zw", ZWITTERION), ("nitro", NITRO))],
                                          BuildConfig(protonate=False)))
    assert (mols["dod"].rot, mols["dod"].charge_class) == (10, "neutral")
    assert mols["zw"].charge_class == "zwitterion" and mols["nitro"].charge_class == "neutral"


def test_the_auto_rule():
    auto = BuildConfig(conformer_search="auto")
    assert search_rule.decide(auto, _m(DODECANOL)) == (True, "conformer search auto: CREST (neutral, 10 rotatable "
                                                              "bonds >= 9)")
    assert search_rule.decide(auto, _m(DECANOL)) == (False, "conformer search auto: RDKit (8 rotatable bonds < 9)")
    assert search_rule.decide(auto, _m(ANION)) == (False, "conformer search auto: RDKit (anion)")
    assert search_rule.decide(auto, _m(ZWITTERION)) == (False, "conformer search auto: RDKit (zwitterion)")
    assert search_rule.crest_for(auto, _m(NITRO))
    assert search_rule.crest_for(dataclasses.replace(auto, crest_min_rotors=8), _m(DECANOL))
    assert search_rule.crest_for(BuildConfig(conformer_search="crest"), _m(ANION))       # "crest": every molecule
    assert not search_rule.crest_for(BuildConfig(), _m(DODECANOL))                        # "rdkit": none
    assert [search_rule.may_use_crest(BuildConfig(conformer_search=s)) for s in search_rule.SEARCHES] == \
        [False, True, True]
    assert search_rule.used_crest("crest+rdkit") and search_rule.used_crest("crest")
    assert not search_rule.used_crest("rdkit") and not search_rule.used_crest("imported")
    with pytest.raises(ValueError, match="conformer_search must be one of rdkit, crest, auto"):
        search_rule.decide(BuildConfig(conformer_search="spartan"), _m(DODECANOL))
    assert "RDKit for the rest" in search_rule.describe(auto)


def test_merged_pools_keep_each_sources_lowest_and_drop_duplicates():
    mol = _m(DODECANOL)
    a = confs.generate(mol, 12, 50.0, 0.3, seed=1)
    b = confs.generate(mol, 12, 50.0, 0.3, seed=7)
    both = search_rule.merge_pools(a, b, refine_top=4)
    n_b = len(both.conf_ids) - 4
    assert 0 < n_b <= 4 and both.refine_count == len(both.conf_ids)
    assert both.refine_ids(4) == both.conf_ids                              # every conformer is refined
    assert both.mol.GetNumConformers() == len(both.conf_ids) and both.forcefield == "MMFF94s + MMFF94s"
    assert both.n_embedded == a.n_embedded + b.n_embedded and "conformer pools merged" in both.notes[-1]
    same = search_rule.merge_pools(a, a, refine_top=4)                     # RDKit's duplicates CREST's: all dropped
    assert len(same.conf_ids) == 4 and "4 duplicates of a CREST conformer dropped" in same.notes[-1]
    assert a.refine_ids(3) == a.conf_ids[:3] and a.refine_count == 0      # an ordinary set: the lowest refine_top


def test_cli_options():
    args = cli.build_parser().parse_args(["build", "CCO", "--conformers", "auto", "--crest-min-rotors", "7",
                                          "--crest-with-rdkit"])
    cfg = cli._config(args)
    assert (cfg.conformer_search, cfg.crest_min_rotors, cfg.crest_with_rdkit) == ("auto", 7, True)
    cfg = cli._config(cli.build_parser().parse_args(["build", "CCO", "--no-crest-with-rdkit"]))
    assert not cfg.crest_with_rdkit and cfg.conformer_search == "rdkit" and cfg.crest_min_rotors == 9
    with pytest.raises(SystemExit):
        cli._config(cli.build_parser().parse_args(["build", "CCO", "--crest-min-rotors", "-1"]))
    assert advisor.cli_options({"conformer_search": "auto", "crest_min_rotors": 9, "crest_with_rdkit": True}) == \
        "--conformers auto --crest-min-rotors 9 --crest-with-rdkit"


def test_an_older_run_folder_resumes_and_the_new_fields_matter_only_where_they_act():
    """A stored config from before CONF-AUTO has neither field: it resumes as it was (no difference), and the
    threshold is no difference while no run uses "auto", the merge none while CREST never runs."""
    old = {k: v for k, v in dataclasses.asdict(BuildConfig(conformer_search="crest")).items()
           if k not in ("crest_min_rotors", "crest_with_rdkit")}
    assert {"crest_min_rotors", "crest_with_rdkit"} <= set(jobs.SCIENCE_FIELDS)
    assert jobs.config_differences(old, BuildConfig(conformer_search="crest")) == {}
    assert jobs.config_differences(old, BuildConfig(conformer_search="crest", crest_min_rotors=5)) == {}
    assert "crest_with_rdkit" in jobs.config_differences(old, BuildConfig(conformer_search="crest",
                                                                         crest_with_rdkit=True))
    rd = dataclasses.asdict(BuildConfig())
    assert jobs.config_differences(rd, BuildConfig(crest_with_rdkit=True, crest_min_rotors=3)) == {}
    auto = dataclasses.asdict(BuildConfig(conformer_search="auto"))
    assert set(jobs.config_differences(auto, BuildConfig(conformer_search="auto", crest_min_rotors=8))) == \
        {"crest_min_rotors"}
    cfg = jobs.stored_config(old, BuildConfig(threads=3))
    assert (cfg.conformer_search, cfg.crest_min_rotors, cfg.crest_with_rdkit) == ("crest", 9, False)


def _review(smiles: dict, **cfg):
    inputs = [from_smiles(s, id=i) for i, s in smiles.items()]
    return advisor.review(inputs, BuildConfig(**cfg), available={"mmff", "tblite-gfn2", "crest"})


def test_the_advisor_knows_the_rule():
    mols = {"dod": DODECANOL, "acid": "OC(=O)CCCCCCCCCCC", "eth": "CCO"}
    r = _review(mols)
    f = next(f for f in r.findings if f.key == "search.flexible-neutrals")
    assert f.ids == ["dod"] and f.changes == {"conformer_search": "auto", "crest_method": "gfnff",
                                              "crest_quick": True}      # the acid is an anion at pH 7
    assert "search.flexible-neutrals" not in {x.key for x in _review(mols, conformer_search="auto").findings}
    # per-molecule cost: only the flexible neutral one pays for CREST; the merged pool adds RDKit's
    feats = dict(advisor.molecule_features([from_smiles(s, id=i) for i, s in mols.items()], BuildConfig()))
    auto = BuildConfig(conformer_search="auto")
    rd = BuildConfig()
    assert advisor.uses_crest(auto, feats["dod"]) and not advisor.uses_crest(auto, feats["acid"])
    assert advisor.molecule_seconds(feats["eth"], auto, "tblite-gfn2") == \
        advisor.molecule_seconds(feats["eth"], rd, "tblite-gfn2")
    one = advisor.molecule_seconds(feats["dod"], auto, "tblite-gfn2")
    two = advisor.molecule_seconds(feats["dod"], dataclasses.replace(auto, crest_with_rdkit=True), "tblite-gfn2")
    assert one["conformers"] > 10 * advisor.molecule_seconds(feats["dod"], rd, "tblite-gfn2")["conformers"]
    assert two["conformers"] > one["conformers"] and two["refine"] > one["refine"]
    steps = dict(_review(mols, conformer_search="auto", solvent="water").steps)
    assert steps["conformer search"].startswith("ALPB(water) (CREST) for 1 flexible neutral molecule(s)")
    assert "for neutral molecules with >= 9 rotatable bonds" in advisor.settings_line(
        _review(mols, conformer_search="auto"))
