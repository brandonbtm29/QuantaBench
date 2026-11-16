"""UI-5: the job dialog's core (chem.recipes): tasks, the built-in recipes, the user's recipes in the settings, and the
job spec that the dialog submits."""

from dataclasses import replace

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from quantabench import settings as sm
from quantabench.chem import import_options as imp
from quantabench.chem import recipes
from quantabench.chem.inputs import MoleculeInput, from_smiles
from quantabench.chem.pipeline import BuildConfig

BASE = BuildConfig(threads=6, priority="background", solvent="water", qm_memory_gb=12.0, molecule_timeout_s=900.0)


def test_each_builtin_recipe_makes_its_build_config():
    here = {"mmff", "tblite-gfn2", "aimnet2"}
    screen = recipes.BY_KEY["screen"].config(BASE, available=here)
    assert (screen.conformer_search, screen.max_conformers, screen.method, screen.refine_top, screen.rank_method,
            screen.qm_method, screen.frequencies) == ("rdkit", 30, "tblite-gfn2", 10, "auto", "", False)
    std = recipes.BY_KEY["standard"].config(BASE, available=here)
    assert (std.conformer_search, std.max_conformers, std.method, std.refine_top, std.rank_method, std.qm_method) == \
        ("rdkit", 50, "auto", 20, "auto", "")
    careful = recipes.BY_KEY["careful"].config(BASE, available=here)
    assert (careful.conformer_search, careful.crest_method, careful.crest_quick, careful.method, careful.refine_top,
            careful.qm_method, careful.qm_task, careful.qm_top) == \
        ("crest", "gfnff", True, "auto", 20, "wb97x-d/def2-svp", "energy", 3)
    # a recipe leaves the resources, the solvent and the files alone
    for cfg in (screen, std, careful):
        assert (cfg.threads, cfg.priority, cfg.solvent, cfg.qm_memory_gb, cfg.molecule_timeout_s, cfg.formats) == \
            (6, "background", "water", 12.0, 900.0, BASE.formats)
    assert [r.key for r in recipes.BUILTIN] == ["screen", "standard", "careful"]
    assert all(r.task == "conformers" and r.steps for r in recipes.BUILTIN)


def test_the_gfn2_alias_takes_the_engine_installed_here():
    assert recipes.resolve_method("gfn2", {"gfn2-xtb", "tblite-gfn2"}) == ("gfn2-xtb", "")    # AUTO_ORDER's order
    assert recipes.resolve_method("gfn2", {"xtb-gfn2"}) == ("xtb-gfn2", "")
    key, note = recipes.resolve_method("gfn2", {"mmff"})
    assert key == "auto" and "no GFN2-xTB engine" in note
    assert recipes.resolve_method("aimnet2", set()) == ("aimnet2", "")                        # not an alias
    changes, notes = recipes.BY_KEY["screen"].resolved({"mmff"})
    assert changes["method"] == "auto" and notes


def test_a_recipe_says_which_programs_it_needs():
    assert recipes.BY_KEY["careful"].needs(set()) == ["crest", "psi4"]
    assert recipes.BY_KEY["careful"].needs({"crest", "psi4"}) == []
    assert recipes.BY_KEY["screen"].needs(set()) == []


def test_tasks_set_their_fields():
    cfg = recipes.BY_KEY["careful"].config(BASE, available={"mmff"})
    opt = recipes.TASKS["optimize"].apply(cfg)
    assert (opt.max_conformers, opt.refine_top, opt.conformer_search, opt.rank_method, opt.frequencies) == \
        (1, 1, "rdkit", "refine", False)
    assert opt.qm_method == "wb97x-d/def2-svp"                  # the DFT stage stays: it runs on the one structure
    sp = recipes.TASKS["single_point"].apply(cfg)
    assert (sp.max_conformers, sp.max_steps, sp.optimizer, sp.batched) == (1, 0, "lbfgs", "off")
    freq = recipes.TASKS["frequencies"].apply(cfg)
    assert freq.frequencies and freq.max_conformers == 50 and freq.conformer_search == "crest"
    assert recipes.TASKS["conformers"].apply(cfg) == cfg
    assert not recipes.TASKS["scan"].available and recipes.TASKS["scan"].why
    assert recipes.TASKS["optimize"].keeps_coordinates and not recipes.TASKS["conformers"].keeps_coordinates


def test_the_job_spec():
    spec = recipes.JobSpec(name="j", recipe="screen", task="frequencies",
                           cfg=recipes.BY_KEY["screen"].config(BASE, available={"tblite-gfn2"}))
    assert spec.problems() == []
    cfg = spec.config()
    assert cfg.frequencies and cfg.method == "tblite-gfn2"
    assert spec.summary().startswith("GFN2-xTB · Water · frequencies + thermochemistry (30 conformers → lowest 10)")
    scan = replace(spec, task="scan")
    with pytest.raises(ValueError, match="cannot run"):
        scan.config()
    assert any("Scan" in p for p in scan.problems())
    remote = replace(spec, run_on="brandons-macbook-pro")
    assert any("DIST-3" in p for p in remote.problems())
    assert "charge -1" in replace(spec, charge=-1).summary()
    assert replace(spec, multiplicity=0).problems()


def test_job_inputs_override_charge_and_keep_file_coordinates():
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    from_file = MoleculeInput("ethanol_3d", smiles="CCO", mol=mol, meta={"x": 1})
    plain = from_smiles("CC(=O)O", id="acetic")
    spec = recipes.JobSpec(task="optimize", charge=0, multiplicity=1)
    a, b = recipes.job_inputs([from_file, plain], spec)
    assert a.meta == {"x": 1, imp.KEEP_COORDS: True} and from_file.meta == {"x": 1}     # copies, originals kept
    assert imp.KEEP_COORDS not in b.meta and b.charge == 0 and b.multiplicity == 1
    c, = recipes.job_inputs([from_file], recipes.JobSpec(task="conformers"))
    assert c is from_file                                                              # nothing to change
    d, = recipes.job_inputs([plain], recipes.JobSpec(charge=-1))
    assert d.charge == -1 and plain.charge is None


def test_user_recipes_are_saved_in_the_settings_and_read_back():
    s = sm.Settings()
    cfg = replace(recipes.BY_KEY["screen"].config(BuildConfig(), {"tblite-gfn2"}), solvent="octanol", refine_top=7)
    r = recipes.save_recipe(s, "PFAS screen", cfg, task="frequencies", description="mine")
    assert not r.builtin and r.key == "user:PFAS screen" and r.task == "frequencies"
    assert r.changes == recipes.recipe_changes(cfg) and set(r.changes) == set(recipes.RECIPE_FIELDS)
    assert "tblite" not in r.steps[1] and "lowest 7" in r.steps[1]
    again = recipes.save_recipe(s, "pfas screen", replace(cfg, refine_top=9))          # same name: replaced
    assert len(s.recipes) == 1 and again.changes["refine_top"] == 9
    with pytest.raises(ValueError, match="built-in"):
        recipes.save_recipe(s, "Screen", cfg)
    with pytest.raises(ValueError, match="name"):
        recipes.save_recipe(s, "  ", cfg)
    # the settings file keeps them; entries that are not recipes are skipped, unknown or mistyped fields dropped
    s2 = sm.Settings.from_dict(s.to_dict())
    s2.recipes += ["junk", {"name": ""}, {"name": "x", "changes": "no"},
                   {"name": "Odd", "task": "nonsense", "changes": {"refine_top": "7", "max_conformers": 12, "ph": 3.0,
                                                                   "from_the_future": 1}}]
    got = recipes.user_recipes(s2)
    assert [g.name for g in got] == ["pfas screen", "Odd"]
    odd = got[1]
    assert odd.changes == {"max_conformers": 12} and odd.task == "conformers"
    assert [r.name for r in recipes.all_recipes(s2)][:3] == ["Screen", "Standard", "Careful"]
    assert recipes.find("user:Odd", s2).changes == odd.changes
    assert recipes.delete_recipe(s2, "ODD") and [g.name for g in recipes.user_recipes(s2)] == ["pfas screen"]
    assert not recipes.delete_recipe(s2, "nope")


def test_levels_list_every_engine_by_family():
    lv = recipes.levels()
    assert lv[0].key == "auto" and lv[0].available
    fams = [x.family for x in lv[1:]]
    assert fams == sorted(fams, key=recipes.FAMILIES.index)
    mmff = next(x for x in lv if x.key == "mmff")
    assert mmff.available and mmff.family == "force field"
    assert all(x.why for x in lv if not x.available)
