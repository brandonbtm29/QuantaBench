"""The model advisor (chem.advisor, ADV-1): the batch profile, method coverage, the findings and when each fires,
the presets, the CLI (review, the build header) and a 14,000-molecule profile."""

import json
import time
from dataclasses import replace

import pytest

from quantabench.chem import advisor, engines
from quantabench.chem import method_guide as guide
from quantabench.chem.inputs import MoleculeInput
from quantabench.chem.pipeline import BuildConfig

PFOA = "OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"
MOLS = {
    "PFOA": PFOA,
    "perfluorohexane": "FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F",
    "octane": "CCCCCCCC",
    "ethanol": "CCO",
    "FBSA": "NS(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F",
    "K_PFOS": "[K+].[O-]S(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F",
    "ethyl_radical": "C[CH2]",
    "SiB": "C[Si](C)(C)c1ccc(B(O)O)cc1",
    "ferrocene_ions": "[Fe+2].c1cc[cH-]c1.c1cc[cH-]c1",
    "tributyltin": "CCCC[Sn](CCCC)(CCCC)Cl",
}
#: what the tests pretend is installed (the cloud test machine has only tblite of the engines)
ALL = {"mmff", "aimnet2", "gfn2-xtb", "gfn1-xtb", "tblite-gfn2", "tblite-gfn1", "xtb-gfn2", "xtb-gfn1", "gfnff",
       "mopac-pm7", "mopac-pm6", "psi4", "crest"}
TBLITE = {"mmff", "tblite-gfn2", "tblite-gfn1", "psi4"}


def inputs(*names, extra: dict | None = None):
    table = {**MOLS, **(extra or {})}
    return [MoleculeInput(id=n, smiles=table[n]) for n in names]


def cfg_for(solvent="", **kw):
    return replace(BuildConfig(solvent=solvent, threads=4), **kw)


def review(names, solvent="", available=ALL, extra=None, **kw):
    return advisor.review(inputs(*names, extra=extra), cfg_for(solvent, **kw), available=available)


def keys(r):
    return {f.key: f for f in r.findings}


def features(smiles, **cfg):
    advisor.clear_cache()
    return advisor.molecule_features([MoleculeInput(id="x", smiles=smiles)], cfg_for(**cfg))[0][1]


# ------------------------------------------------------------------ profile and SMARTS
def test_profile_of_the_example_molecules():
    advisor.clear_cache()
    r = review(list(MOLS), "water", available=TBLITE)
    p = r.profile
    assert p.n == len(MOLS)
    assert p.group("charge", "anion").ids == ["PFOA", "FBSA", "K_PFOS"]          # at pH 7 (FBSA: pKa ~6.5 rule)
    assert set(p.group("charge", "neutral").ids) == {"perfluorohexane", "octane", "ethanol", "ethyl_radical", "SiB",
                                                     "tributyltin"}
    assert p.group("pfas", "pfas").ids == ["PFOA", "perfluorohexane", "FBSA", "K_PFOS"]
    assert p.group("flags", "hydrocarbon").ids == ["octane", "ethyl_radical"]
    assert p.group("flags", "open_shell").ids == ["ethyl_radical"]
    assert p.group("flags", "multi_fragment").ids == ["K_PFOS"]        # a record that cannot be built is only that
    assert p.group("flags", "not_computable").ids == ["ferrocene_ions"]          # a complex drawn as ions
    assert p.group("removed", "[K+]").ids == ["K_PFOS"]
    assert p.group("flags", "metal").ids == ["tributyltin"]
    assert set(p.group("flags", "unusual").ids) == {"SiB"}                         # B, Si (Sn is a metal)
    assert p.group("groups", "carboxylic").ids == ["PFOA"]
    assert p.group("groups", "sulfonic").ids == ["K_PFOS"]
    assert p.group("groups", "sulfonamide").ids == ["FBSA"]
    assert p.group("groups", "alcohol").ids == ["ethanol"]
    assert p.group("groups", "aromatic").ids == ["SiB"]
    assert p.group("groups", "halogen").ids == ["tributyltin"]
    assert p.group("elements", "Sn").ids == ["tributyltin"] and p.group("elements", "B").ids == ["SiB"]
    assert p.stats["heavy"]["max"] == 29 and p.stats["heavy"]["min"] == 2      # PFOS anion, ethyl


def test_pfas_rule_and_chain_length():
    assert features(PFOA).pf_chain == 7                                            # C7F15-COOH
    assert features(MOLS["K_PFOS"]).pf_chain == 8
    assert features("OC(=O)C(F)(OC(F)(F)C(F)(F)C(F)(F)F)C(F)(F)F").pf_chain == 3   # GenX: the ether O breaks it
    assert features("OS(=O)(=O)CCC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F").pf_chain == 6   # 6:2 FTS
    assert features("FC(F)(F)CO").pfas                                             # a CF3 group: PFAS
    assert not features("FC(F)CO").pfas                                            # CHF2: has an H
    assert not features("ClC(F)(F)CO").pfas                                        # CF2Cl: has a Cl
    assert not features("Fc1ccccc1").pfas                                          # aromatic F
    assert not features("CCCCCCCC").pfas


@pytest.mark.parametrize("smiles, group", [
    ("CC(=O)OCC", "ester"), ("CCOCC", "ether"), ("CC(=O)NC", "amide"), ("CCN(C)C", "amine"),
    ("C[N+](C)(C)CC", "quaternary"), ("CCC[N+]([O-])=O", "nitro"), ("CCSC", "thio"), ("CCS", "thio"),
    ("COP(=O)(O)OC", "phosphate"), ("c1ccccc1", "aromatic"), ("CCBr", "halogen"), ("CCI", "halogen"),
    ("COS(=O)(=O)O", "sulfate"),
])
def test_functional_groups(smiles, group):
    m = features(smiles, protonate=False)
    assert group in m.tags, (smiles, sorted(m.tags))


def test_groups_stay_silent_where_they_should():
    assert features("CC(=O)OCC", protonate=False).tags.isdisjoint({"ether", "alcohol", "amide"})   # ester only
    assert "amine" not in features("CC(=O)NC", protonate=False).tags                              # amide, not amine
    assert "amine" not in features(MOLS["FBSA"], protonate=False).tags                            # sulfonamide N
    assert "ammonium" in features("CCN").tags                                                     # pH 7: RNH3+
    assert "alcohol" not in features("c1ccccc1O").tags                                            # a phenol


def test_charge_classes_stereo_macrocycles_isotopes():
    gly = features("NCC(=O)O")
    assert gly.charge_class == "zwitterion" and gly.charge == 0
    assert features("NCC(=O)O", protonate=False).charge_class == "neutral"
    assert features("CCC[N+]([O-])=O").charge_class == "neutral"                 # a nitro pair is not a zwitterion
    assert features("OC(=O)CC(=O)O").charge == -2
    assert features("CC(O)CC").undef_atoms == 1
    assert features("C[C@H](O)CC").undef_atoms == 0
    assert features("CC(O)C(Cl)C").undef_atoms == 2
    assert features("CC=CC").undef_bonds == 1 and features("C/C=C/C").undef_bonds == 0
    assert "macrocycle" in features("C1CCCCCCCCCCC1").tags and "macrocycle" not in features("C1CCCCC1").tags
    assert "isotope" in features("[13CH3]CO").tags
    assert features("C[CH2]").mult == 2 and features("C[CH2]").radicals == 1
    assert features("[CH2]").mult == 3                                            # two unpaired electrons
    sib = features(MOLS["SiB"])
    assert sib.z == {1, 5, 6, 8, 14} and sib.mmff is False                        # MMFF94 has no boron


def test_fast_path_agrees_with_prepare():
    """Neutral single-fragment records skip RDKit's normalization: the species must be the one prepare computes."""
    from rdkit import Chem

    from quantabench.chem import standardize

    smiles = [PFOA, MOLS["FBSA"], "NCC(=O)O", "CCN", "CC(=O)OCC", "OS(=O)(=O)CCC(F)(F)C(F)(F)F", "NCCCN",
              "C1CNCCN1", "NC(=N)NCCC", "OP(=O)(O)CC(F)(F)F", "CCCC[N+](C)(C)C", "Oc1ccccc1", "CCC[N+]([O-])=O"]
    for s in smiles:
        m = features(s)
        ref, _notes = standardize.prepare(Chem.MolFromSmiles(s), 7.0, True, "rules")
        assert m.charge == Chem.GetFormalCharge(ref), s
        ref_counts = advisor._formula_counts(Chem.RemoveHs(ref))
        assert sum(ref_counts.values()) == m.atoms, s


def test_unreadable_records_are_listed():
    r = advisor.review([MoleculeInput(id="bad", smiles="C1CC"), MoleculeInput(id="ok", smiles="CCO")], cfg_for(),
                       available=TBLITE)
    f = keys(r)["input.not-computable"]
    assert f.level == "problem" and f.ids == ["bad"]


# ------------------------------------------------------------------ coverage
def test_coverage_table_agrees_with_the_engines():
    assert guide.COVERAGE["aimnet2"].elements == engines.AIMNET2_ELEMENTS
    from quantabench.chem import eng_mopac

    assert guide.COVERAGE["pm7"].elements == eng_mopac.ELEMENTS == guide.COVERAGE["pm6-d3h4x"].elements
    for key, info in engines.all_engines().items():
        cov = guide.coverage_for_engine(key)
        assert cov is not None, key
        if info.elements is not None:
            assert cov.elements == info.elements, key
    for level in ("gfn2", "gfn1", "gfnff", "gfn2//gfnff"):
        assert guide.coverage_for_crest(level) is not None
    for basis in ("def2-svp", "def2-svpd", "def2-tzvp", "6-31g*", "6-31+g*", "6-311+g**", "aug-cc-pvdz"):
        assert guide.coverage_for_basis(basis) is not None, basis
    assert guide.coverage_for_basis("6-31G(d)").key == "6-31g*"
    assert guide.coverage_for_basis("not-a-basis") is None
    for c in guide.COVERAGE.values():
        assert c.key in guide.ENTRIES and c.source and c.open_shell in ("yes", "no", "ignored")


def test_coverage_excludes_the_right_methods():
    aim, gfn2, mmff, gfnff = (guide.COVERAGE[k] for k in ("aimnet2", "gfn2-xtb", "mmff94s", "gfn-ff"))
    assert aim.handles({1, 6, 8}) == (True, "")
    assert aim.handles({1, 6, 50})[0] is False and "Sn" in aim.handles({1, 6, 50})[1]
    assert aim.handles({1, 6}, 0, 2)[0] is False and "closed-shell" in aim.handles({1, 6}, 0, 2)[1]
    assert gfn2.handles({1, 6, 50}, -1, 2) == (True, "")
    assert gfn2.handles({1, 92})[0] is False and guide.COVERAGE["g-xtb"].handles({1, 92})[0]
    assert mmff.handles({1, 6, 5})[0] is False and mmff.handles({11})[0]           # B no; Na+ as a free ion yes
    assert gfnff.handles({1, 6}, 0, 2)[0] and gfnff.open_shell == "ignored"
    assert guide.COVERAGE["6-31g*"].missing({1, 6, 53}) == [53]                     # no iodine in Psi4's 6-31G*
    assert not guide.COVERAGE["def2-svp"].missing({1, 6, 53, 50})
    assert guide.COVERAGE["def2-svp"].missing({1, 60}) == [60]                      # Nd: no def2 set in Psi4
    rows = {c.key: c for c in review(["ethyl_radical", "tributyltin", "octane"]).coverage}
    assert rows["aimnet2"].n_ok == 1 and rows["aimnet2"].n_spin == 1 and rows["aimnet2"].n_elements == 1
    assert rows["gfn2-xtb"].n_ok == 3 and rows["gfn-ff"].n_spin_ignored == 1
    assert rows["mmff94s"].n_elements == 1 and rows["g-xtb"].installed is None
    assert rows["6-31g*"].ids_excluded == ["tributyltin"]


def test_coverage_table_in_the_methods_doc():
    text = guide.markdown_document()
    assert "## Which molecules each method takes" in text
    assert "| [AIMNet2](#aimnet2) | H, B, C, N, O, F, Si, P, S, Cl, As, Se, Br, I |" in text
    assert "H-La, Lu-Bi" in text and "H-La, Hf-Rn" in text
    assert guide.element_text([1, 2, 3, 5, 6, 7, 9]) == "H, He, Li, B, C, N, F"
    assert guide.element_text(range(1, 37)) == "H-Kr"


# ------------------------------------------------------------------ the conformer-ranking rules
def test_pfas_ions_ranked_by_gfn2_are_a_problem_and_the_fix_is_measured():
    r = review(["PFOA", "K_PFOS", "FBSA"], "water", available=TBLITE)
    f = keys(r)["rank.gfn2.pfas_ion"]
    assert f.level == "problem" and f.n == 3 and "random pick" in f.message
    assert "0 of 13" in f.why and "RANK-1b" in f.evidence
    assert f.changes == {"qm_method": "wb97x-d/def2-svpd", "qm_top": 8, "energy_window_kcal": 15.0,
                         "refine_top": 20}
    assert "conformer-ranking" in f.guide and "RANK-2" in f.recommendation
    # with the recommended DFT re-rank configured, it is a note
    done = advisor.review(inputs("PFOA", "K_PFOS"), replace(cfg_for("water"), **f.changes), available=TBLITE)
    g = keys(done)["rank.gfn2.pfas_ion"]
    assert g.level == "info" and not g.changes and "re-ranks the lowest 8" in g.message
    assert done.counts()["problem"] == 0
    # an explicit QM level keeps its functional (when it has dispersion) and gains the diffuse basis
    r2 = review(["PFOA"], "water", available=TBLITE, qm_method="b3lyp-d3bj/def2-svp")
    assert keys(r2)["rank.gfn2.pfas_ion"].changes["qm_method"] == "b3lyp-d3bj/def2-svpd"


def test_ranking_rules_by_class_and_phase():
    water = keys(review(["ethanol", "perfluorohexane", "octane"], "water", available=TBLITE))
    assert water["rank.gfn2.polar"].level == "warning" and water["rank.gfn2.polar"].n == 2
    assert water["rank.gfn2.polar"].changes["qm_method"] == "wb97x-d/def2-svp"     # no anions: no diffuse basis
    assert water["rank.gfn2.hydrocarbon"].level == "info"
    assert "rank.gfn2.pfas_ion" not in water
    gas = keys(review(["ethanol", "octane"], "", available=TBLITE))
    assert gas["rank.gfn2.polar"].level == "info" and not gas["rank.gfn2.polar"].changes   # AIMNet2 not installed
    gas2 = keys(review(["ethanol", "octane"], "", available=ALL, method="tblite-gfn2"))
    assert gas2["rank.gfn2.polar"].changes == {"method": "aimnet2"}
    aim = keys(review(["ethanol", "octane"], "", available=ALL))                    # auto -> AIMNet2
    assert aim["rank.aimnet2.polar"].level == "info" and "39 of 40" in aim["rank.aimnet2.polar"].why
    hydro = review(["octane"], "water", available=TBLITE)
    assert hydro.counts() == {"problem": 0, "warning": 0, "info": hydro.counts()["info"]}
    assert advisor.rank_verdict("gfn2", "ion", "gas") is None                      # not measured
    assert advisor.rank_verdict("gfn2", "ion", "water").level == "warning"
    assert advisor.rank_verdict("aimnet2", "pfas_ion", "water").level == "warning"
    assert advisor.rank_verdict("mmff", "pfas_ion", "water").level == "problem"


# ------------------------------------------------------------------ elements, charge, spin
def test_element_and_spin_rules():
    r = keys(review(["ethyl_radical", "ethanol"], "", available=ALL))
    f = r["elements.aimnet2-open-shell"]
    assert f.level == "problem" and f.ids == ["ethyl_radical"] and f.changes == {"method": "gfn2-xtb"}
    assert "elements.aimnet2-open-shell" not in keys(review(["ethanol"], "", available=ALL))
    mixed = keys(review(["tributyltin", "ethanol"], "", available=ALL))
    assert mixed["elements.mixed-auto"].level == "warning" and mixed["elements.mixed-auto"].ids == ["tributyltin"]
    assert mixed["elements.mixed-auto"].changes == {"method": "gfn2-xtb"}
    assert "elements.mixed-auto" not in keys(review(["octane", "ethanol"], "", available=ALL))
    explicit = keys(review(["tributyltin", "ethanol"], "", available=ALL, method="aimnet2"))
    assert explicit["elements.fallback"].level == "problem" and explicit["elements.fallback"].ids == ["tributyltin"]
    missing = keys(review(["ethanol"], "", available=TBLITE, method="aimnet2"))
    assert missing["elements.fallback"].level == "warning" and "not installed" in missing["elements.fallback"].message
    uranium = keys(review(["U"], "", available=TBLITE, extra={"U": "[U]"}))
    assert uranium["elements.beyond-rn"].level == "problem"
    assert "elements.beyond-rn" not in keys(review(["tributyltin"], "", available=TBLITE))
    assert keys(review(["SiB"], "", available=TBLITE))["elements.uff"].ids == ["SiB"]
    unusual = keys(review(["SiB"], "", available=TBLITE))["elements.unusual"]
    assert "B, Si" in unusual.message and "AIMNet2" in unusual.message


def test_qm_level_rules():
    plain = keys(review(["PFOA", "ethanol"], "water", available=TBLITE, qm_method="b3lyp/def2-svp"))
    assert plain["qm.dispersion"].level == "warning"
    assert plain["qm.dispersion"].changes == {"qm_method": "b3lyp-d3bj/def2-svpd"}   # both fixes in one level
    assert plain["qm.diffuse"].changes == {"qm_method": "b3lyp-d3bj/def2-svpd"}
    assert "+15.0" in plain["qm.diffuse"].why and "def2-svpd" in plain["qm.diffuse"].guide
    good = keys(review(["PFOA"], "water", available=TBLITE, qm_method="wb97x-d/def2-svpd"))
    assert "qm.dispersion" not in good and "qm.diffuse" not in good
    neutral = keys(review(["ethanol"], "", available=TBLITE, qm_method="wb97x-d/def2-svp"))
    assert neutral["qm.bsse"].level == "info" and "gcp" in neutral["qm.bsse"].guide
    assert "qm.diffuse" not in neutral
    iodine = keys(review(["MeI"], "", available=TBLITE, qm_method="wb97x-d/6-31g*", extra={"MeI": "CI"}))
    assert iodine["qm.basis-elements"].level == "problem" and iodine["qm.basis-elements"].changes == {
        "qm_method": "wb97x-d/def2-svp"}
    assert "qm.basis-elements" not in keys(review(["ethanol"], "", available=TBLITE, qm_method="wb97x-d/6-31g*"))
    three_c = keys(review(["ethanol"], "", available=TBLITE, qm_method="r2scan3c/def2-mtzvpp"))
    assert three_c["qm.3c"].level == "warning"
    for f in ("wb97x-d", "b3lyp-d3bj", "r2scan-d4", "wb97m-v", "b97-d3", "pbe0-d3bj"):
        assert advisor.has_dispersion(f), f
    for f in ("b3lyp", "pbe0", "hf", "m06-2x", "r2scan", "wb97x"):
        assert not advisor.has_dispersion(f), f
    assert advisor.has_diffuse("def2-svpd") and advisor.has_diffuse("6-31+g*") and advisor.has_diffuse("aug-cc-pvdz")
    assert not advisor.has_diffuse("def2-svp") and not advisor.has_diffuse("6-31g*")


def test_solvent_rules():
    gas = keys(review(["PFOA", "ethanol"], "", available=TBLITE))
    assert gas["solvent.gas-ions"].level == "warning" and gas["solvent.gas-ions"].changes == {"solvent": "water"}
    di = keys(review(["malonic"], "", available=TBLITE, extra={"malonic": "OC(=O)CC(=O)O"}))
    assert di["solvent.gas-ions"].level == "problem"                          # a multiply charged anion
    assert "solvent.gas-ions" not in keys(review(["PFOA"], "water", available=TBLITE))
    mm = keys(review(["ethanol"], "water", available=TBLITE, method="mmff"))
    assert mm["solvent.gas-refinement"].changes == {"method": "tblite-gfn2"}
    assert "solvent.gas-refinement" not in keys(review(["ethanol"], "water", available=TBLITE))
    alkyl = keys(review(["laurate"], "water", available=TBLITE, qm_method="wb97x-d/def2-svpd",
                        extra={"laurate": "OC(=O)CCCCCCCCCCC"}))
    assert alkyl["qm.pcm-electrostatic"].level == "warning"
    assert "qm.pcm-electrostatic" not in keys(review(["PFOA"], "water", available=TBLITE, qm_method="wb97x-d/def2-svpd"))


def test_input_rules():
    salt = keys(review(["K_PFOS"], "water", available=TBLITE))
    assert salt["input.fragments"].level == "info" and "[K+]" in salt["input.fragments"].message
    together = keys(review(["K_PFOS"], "water", available=TBLITE, keep_largest_fragment=False))
    assert together["input.fragments-together"].changes == {"keep_largest_fragment": True}
    stereo = keys(review(["two", "one", "defined"], "", available=TBLITE,
                         extra={"two": "CC(O)C(Cl)C", "one": "CC(O)CC", "defined": "C[C@H](O)CC"}))
    assert stereo["input.stereo"].ids == ["two"] and stereo["input.stereo-one"].ids == ["one"]
    assert keys(review(["ethyl_radical"], "", available=TBLITE))["input.radicals"].level == "warning"
    assert keys(review(["FBSA"], "water", available=TBLITE))["input.pf-sulfonamide"].level == "info"
    nplus = {"drawn": "C[NH+](C)CCCNS(=O)(=O)C(F)(F)F", "wrong": "CN(C)CCC[NH2+]S(=O)(=O)C(F)(F)F"}
    off = keys(review(["wrong"], "water", available=TBLITE, extra=nplus, protonate=False))
    assert off["input.sulfonamide-nplus"].level == "problem" and off["input.sulfonamide-nplus"].changes == {
        "protonate": True}
    on = keys(review(["wrong"], "water", available=TBLITE, extra=nplus))
    assert "input.sulfonamide-nplus" not in on and on["input.sulfonamide-nplus-fixed"].level == "info"
    assert "input.sulfonamide-nplus" not in keys(review(["drawn"], "water", available=TBLITE, extra=nplus))
    noprot = keys(review(["PFOA"], "water", available=TBLITE, protonate=False))
    assert noprot["input.no-protonation"].changes == {"protonate": True}


def test_search_and_cost_rules():
    flex = {"flex": "OC(=O)CCCCCCCCCC(F)(F)F", "ring": "C1CCCCCCCCCCC1"}
    few = keys(review(["flex"], "water", available=ALL, extra=flex, refine_top=5))
    assert few["search.refine-top"].changes == {"refine_top": 20}
    assert few["search.crest-ions"].changes == {"conformer_search": "crest", "crest_method": "gfnff"}
    assert "search.refine-top" not in keys(review(["flex"], "water", available=ALL, extra=flex))
    assert keys(review(["ring"], "", available=TBLITE, extra=flex))["search.macrocycle"].level == "info"
    big = {"big": "OC(=O)" + "C(F)(F)" * 45 + "F"}
    crest = keys(review(["big"], "water", available=ALL, extra=big, conformer_search="crest", crest_method="gfn2"))
    assert crest["cost.crest-gfn2"].changes == {"crest_method": "gfnff"}
    assert "cost.crest-gfn2" not in keys(review(["big"], "water", available=ALL, extra=big,
                                                conformer_search="crest", crest_method="gfnff"))
    total = keys(review(["PFOA"], "water", available=TBLITE))["cost.total"]
    assert total.level == "info" and "on this machine" in total.message


def test_cost_estimates_scale_with_the_level():
    mols = advisor.molecule_features(inputs("PFOA"), cfg_for("water"))
    m = mols[0][1]
    svp = advisor._qm_seconds(m, "wb97x-d/def2-svp")
    assert 60 < svp < 300                                               # PFOA, 355 functions: about 2 min on 4 threads
    assert advisor._qm_seconds(m, "wb97x-d/def2-svpd") == pytest.approx(3 * svp)
    assert advisor._qm_seconds(m, "b3lyp-d3bj/def2-svp") == pytest.approx(0.72 * svp)
    base = advisor.molecule_seconds(m, cfg_for("water"), "tblite-gfn2")
    crest = advisor.molecule_seconds(m, cfg_for("water", conformer_search="crest"), "tblite-gfn2")
    assert crest["conformers"] > 10 * base["conformers"]


# ------------------------------------------------------------------ presets
def _valid(cfg):
    from quantabench.chem import crest, pipeline, qm, solvents

    assert cfg.method in engines.method_keys()
    assert cfg.conformer_search in ("rdkit", "crest") and cfg.crest_method in crest.METHOD_FLAGS
    if cfg.qm_method:
        qm.parse_level(cfg.qm_method)
    solvents.canonical(cfg.solvent)
    assert cfg.max_conformers >= 1 and cfg.refine_top >= 1 and cfg.energy_window_kcal > 0 and cfg.qm_top >= 1
    pipeline.use_batched(cfg, cfg.method, "cpu")


@pytest.mark.parametrize("names, solvent", [
    (["PFOA", "K_PFOS", "FBSA"], "water"), (["octane", "ethanol"], ""), (list(MOLS), "water"),
    (["ethyl_radical", "ethanol"], ""), (["octane"], "water"),
])
def test_presets_are_valid_build_configs(names, solvent):
    for available in (ALL, TBLITE):
        r = review(names, solvent, available=available)
        assert [p.key for p in r.presets] == ["fast", "recommended", "careful"] and r.recommended == "recommended"
        for p in r.presets:
            _valid(p.config)
            assert p.rationale and p.cost.hours >= 0 and p.name == advisor.PRESET_NAMES[p.key]
            assert p.config.solvent == r.settings["solvent"]                 # a preset keeps the phase
            for ident, m in advisor.molecule_features(inputs(*names), p.config):
                if not m.ok:
                    continue
                engine = advisor._Resolver(p.config, solvent, set(available))(m.z)[0]
                cov = guide.coverage_for_engine(engine)
                assert cov is None or cov.handles(m.z, m.charge, m.mult)[0], (p.key, ident, engine)


def test_recommended_preset_by_phase():
    water = review(["PFOA", "ethanol"], "water", available=ALL).preset("recommended")
    assert water.config.qm_method == "wb97x-d/def2-svpd" and water.config.qm_top == 8
    assert water.config.method in ("tblite-gfn2", "xtb-gfn2") and water.config.energy_window_kcal == 15
    gas = review(["ethanol", "octane"], "", available=ALL).preset("recommended")
    assert gas.config.method == "aimnet2" and not gas.config.qm_method
    radical = review(["ethanol", "ethyl_radical"], "", available=ALL).preset("recommended")
    assert radical.config.method == "gfn2-xtb"                               # never AIMNet2 for an open shell
    tin = review(["ethanol", "tributyltin"], "", available=ALL).preset("recommended")
    assert tin.config.method == "auto"
    hydro = review(["octane"], "water", available=ALL).preset("recommended")
    assert not hydro.config.qm_method
    careful = review(["PFOA"], "water", available=TBLITE).preset("careful")
    assert careful.needs == [] and careful.config.conformer_search == "rdkit"     # no CREST: more RDKit conformers
    assert careful.config.max_conformers == 200
    careful = review(["PFOA"], "water", available=ALL - {"crest"} | {"crest"}).preset("careful")
    assert careful.config.conformer_search == "crest" and careful.config.crest_method == "gfnff"
    no_psi4 = review(["PFOA"], "water", available={"tblite-gfn2"}).preset("recommended")
    assert no_psi4.needs == ["Psi4"]


# ------------------------------------------------------------------ the report and JSON
def test_text_header_and_json():
    r = review(["PFOA", "K_PFOS", "octane", "ethyl_radical"], "water", available=TBLITE)
    text = r.text()
    for part in ("Review of 4 molecules", "Batch", "Charge at pH 7", "PFAS", "Methods that can take these molecules",
                 "Where each step runs", "Findings (", "PROBLEM", "Presets", "Fast screen", "Recommended", "Careful",
                 "Recommended configuration: quantabench-cli build ... ", "Read: quantabench-cli methods"):
        assert part in text, part
    assert max(len(line) for line in text.splitlines()) <= 140
    head = r.header()
    assert head.startswith("Review: 4 molecules") and "--no-review" in head and "PROBLEM" in head
    d = json.loads(json.dumps(r.to_dict()))
    assert d["n"] == 4 and d["counts"]["problem"] >= 1 and d["presets"][1]["key"] == "recommended"
    assert {f["key"] for f in d["findings"]} >= {"rank.gfn2.pfas_ion", "input.fragments"}
    assert "config" not in d["presets"][0]
    assert advisor.cli_options({"method": "aimnet2", "energy_window_kcal": 15.0, "crest_quick": False,
                                "protonate": True, "solvent": ""}) == \
        "--method aimnet2 --window 15 --crest-full --protonate --solvent none"


def test_every_guide_key_of_every_finding_exists():
    r = review(list(MOLS), "water", available=ALL, qm_method="b3lyp/def2-svp")
    r2 = review(list(MOLS), "", available=ALL, qm_method="wb97x-d/6-31g*", refine_top=5)
    for f in r.findings + r2.findings:
        assert f.level in advisor.LEVELS and f.topic and f.message and f.key
        for k in f.guide:
            assert k in guide.ENTRIES, (f.key, k)
        assert len(f.ids) <= advisor.MAX_IDS and f.n >= len(f.ids)
        if f.level != "info" and f.key != "input.not-computable":
            assert f.why, f.key


# ------------------------------------------------------------------ the CLI
def test_cli_review_report_and_json(capsys):
    from quantabench import cli

    assert cli.main(["review", f"{PFOA} PFOA", "CCCCCCCC octane", "--phase", "water"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Review of 2 molecules (water phase") and "PFOA" in out and "Presets" in out
    assert "GFN2-xTB ranks the conformers of PFAS ions" in out or "PFAS ions" in out
    assert cli.main(["review", f"{PFOA} PFOA", "--phase", "gas", "--method", "tblite-gfn2", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["settings"]["phase"] == "gas" and d["settings"]["method"] == "tblite-gfn2"
    assert any(f["key"] == "solvent.gas-ions" for f in d["findings"])
    assert cli.main(["review"]) == 2


def test_build_prints_the_review_header_unless_no_review(capsys):
    from quantabench import cli
    from quantabench.chem import pipeline

    calls = []

    def fake_run(items, cfg, **_kw):
        calls.append(len(items))
        return [], None

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pipeline, "run", fake_run)
        cli.main(["build", f"{PFOA} PFOA", "--method", "mmff", "--solvent", "water"])
        out = capsys.readouterr().out
        assert out.startswith("Review: 1 molecules") and "--no-review skips this" in out
        cli.main(["build", f"{PFOA} PFOA", "--method", "mmff", "--no-review"])
        out = capsys.readouterr().out
        assert "Review:" not in out
    assert calls == [1, 1]


def test_grid_build_prints_the_review_header(capsys):
    from quantabench import cli
    from quantabench.dist import client

    class FakeGrid:
        id, missing = "g1", []

        def summary(self):
            return "done"

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(client, "grid_build", lambda *a, **k: FakeGrid())
        assert cli.main(["grid", "build", "CCO ethanol", "--method", "mmff", "--machines", "local"]) == 0
        out = capsys.readouterr().out
        assert out.startswith("Review: 1 molecules") and "grid g1: done" in out
        assert cli.main(["grid", "build", "CCO ethanol", "--method", "mmff", "--no-review"]) == 0
        assert "Review:" not in capsys.readouterr().out


def test_header_never_stops_a_build(capsys):
    from quantabench import cli_review

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(advisor, "review", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        lines = []
        cli_review.print_header(inputs("PFOA"), cfg_for(), out=lines.append)
    assert lines == ["Review skipped (RuntimeError: boom)\n"]


# ------------------------------------------------------------------ 14,000 molecules
def _synthetic(n):
    heads = ["OC(=O)", "OS(=O)(=O)", "NS(=O)(=O)", "OCC", "CCOC(=O)", "c1ccccc1C", "CC(C)N", "OCCOC", "C=CC", "CC(O)C"]
    tails = ["F", "Cl", "CC", "C(F)(F)S(=O)(=O)O"]
    out = []
    for i in range(n):
        head, tail = heads[i % len(heads)], tails[(i // len(heads)) % len(tails)]
        nf = 1 + (i // 40) % 10
        nh = (i // 400) % 35
        out.append(MoleculeInput(id=f"m{i}", smiles=head + "C" * nh + "C(F)(F)" * nf + tail))
    return out


def test_profile_of_14000_molecules_is_quick():
    items = _synthetic(14000)
    assert len({x.smiles for x in items}) == 14000
    advisor.clear_cache()
    t = time.perf_counter()
    r = advisor.review(items, cfg_for("water"), available=TBLITE)
    cold = time.perf_counter() - t
    t = time.perf_counter()
    r2 = advisor.review(items, cfg_for("water"), available=TBLITE)
    warm = time.perf_counter() - t
    not_pfas = [x.id for x in items if x.smiles.endswith("Cl") and x.smiles.count("(F)") == 2]  # one CF2Cl only
    assert r.n == 14000 and r.profile.group("pfas", "not_pfas").n == len(not_pfas) == 350
    assert r.profile.group("pfas", "pfas").n == 14000 - 350
    assert cold < 90 and warm < 5, (cold, warm)       # about 5 s cold on a 2-vCPU cloud VM, 0.3 s warm
    assert all(len(f.ids) <= advisor.MAX_IDS for f in r.findings)
    assert all(len(g.ids) <= advisor.MAX_IDS for gs in r.profile.sections.values() for g in gs)
    assert r2.counts() == r.counts()
    advisor.clear_cache()
