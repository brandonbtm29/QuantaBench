"""Import options (UI-4, Spartan's SDF Import Options): one test per option, the survey that decides whether File >
Open asks, the settings section, the command-line flags, and the build's use of what the import did."""

from __future__ import annotations

import json

import numpy as np
import pytest
from conftest import PFOA

from quantabench.chem import formats, inputs
from quantabench.chem import import_options as imp
from quantabench.chem.import_options import ImportOptions

TFA = "OC(=O)C(F)(F)F"
PFBA = "OC(=O)C(F)(F)C(F)(F)C(F)(F)F"
NONE = ImportOptions(convert_2d=False, grow_hydrogens=False, remove_water=False, remove_acids=False)


def canon(smi: str) -> str:
    from rdkit import Chem

    return Chem.MolToSmiles(Chem.MolFromSmiles(smi))


def write_sdf(path, records, three_d=False, hydrogens=False):
    """records: (title, SMILES, {property: value}); 2D (all z = 0, as CompTox exports) unless ``three_d``."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    w = Chem.SDWriter(str(path))
    for title, smi, props in records:
        m = Chem.MolFromSmiles(smi)
        if three_d:
            m = Chem.AddHs(m)
            AllChem.EmbedMolecule(m, randomSeed=11)
            if not hydrogens:
                m = Chem.RemoveHs(m)
        elif hydrogens:
            m = Chem.AddHs(m)
        m.SetProp("_Name", title)
        for k, v in props.items():
            m.SetProp(k, v)
        w.write(m)
    w.close()
    return path


def z_values(mol):
    return mol.GetConformer().GetPositions()[:, 2]


# ------------------------------------------------------------------------------------------------ fixup
def test_convert_2d_to_3d(tmp_path):
    p = write_sdf(tmp_path / "flat.sdf", [("PFOA", PFOA, {})])
    raw = inputs.read_raw(p)
    assert raw[0].meta[formats.COORDS] == "2d" and np.allclose(z_values(raw[0].mol), 0)
    seen = []
    item = inputs.read_file(p, options=ImportOptions(), log=seen.append)[0]
    assert item.meta[formats.COORDS] == "3d" and np.ptp(z_values(item.mol)) > 0.5
    assert item.mol.GetConformer().Is3D() and canon(item.smiles) == canon(PFOA)
    assert any(a.GetAtomicNum() == 1 for a in item.mol.GetAtoms())           # grown with the embedding
    assert "2D structure converted to 3D" in " ".join(item.meta[imp.NOTES]) and any("converted to 3D" in m for m in seen)
    flat = inputs.read_file(p, options=ImportOptions(convert_2d=False))[0]
    assert flat.meta[formats.COORDS] == "2d" and np.allclose(z_values(flat.mol), 0)
    bare = inputs.read_file(p, options=ImportOptions(grow_hydrogens=False))[0]
    assert np.ptp(z_values(bare.mol)) > 0.5 and not any(a.GetAtomicNum() == 1 for a in bare.mol.GetAtoms())


def test_grow_hydrogens_with_coordinates(tmp_path):
    p = write_sdf(tmp_path / "noh.sdf", [("TFA", TFA, {})], three_d=True)
    heavy = inputs.read_file(p, options=replace_opts(grow_hydrogens=False))[0]
    grown = inputs.read_file(p, options=replace_opts(grow_hydrogens=True))[0]
    assert heavy.mol.GetNumAtoms() == 7 and grown.mol.GetNumAtoms() == 8
    h = [a.GetIdx() for a in grown.mol.GetAtoms() if a.GetAtomicNum() == 1][0]
    o = grown.mol.GetAtomWithIdx(h).GetNeighbors()[0].GetIdx()
    d = np.linalg.norm(grown.mol.GetConformer().GetPositions()[h] - grown.mol.GetConformer().GetPositions()[o])
    assert 0.8 < d < 1.2                                                     # placed on its oxygen


def replace_opts(**kw) -> ImportOptions:
    from dataclasses import replace

    return replace(NONE, **kw)


def test_keep_coords_seeds_the_conformer_search(tmp_path):
    """The file's geometry is conformer 0 of the build, even though the build deprotonates the acid (the heavy atoms
    are matched by element and connection); max_conformers=1 embeds nothing new."""
    from rdkit.Chem import rdMolAlign

    from quantabench.chem.pipeline import BuildConfig, run_molecule

    p = write_sdf(tmp_path / "tfa3d.sdf", [("TFA", TFA, {})], three_d=True, hydrogens=True)
    item = inputs.read_file(p, options=replace_opts(keep_coords=True))[0]
    assert item.meta[imp.KEEP_COORDS] is True
    cfg = BuildConfig(method="mmff", max_conformers=1, refine_top=1, xtb_properties=False, write_files=False)
    res = run_molecule(item, cfg)
    assert res.status == "ok" and res.n_embedded == 0 and len(res.conformers) == 1
    assert any("file's coordinates are the first 1 conformer" in n for n in res.notes)
    from rdkit import Chem

    built = Chem.RemoveHs(res.mol)
    rms = rdMolAlign.GetBestRMS(Chem.RemoveHs(item.mol), built, prbId=-1, refId=res.conformers[0].conf_id)
    assert rms < 0.3
    more = run_molecule(item, BuildConfig(method="mmff", max_conformers=4, refine_top=4, xtb_properties=False,
                                          write_files=False))
    assert more.status == "ok" and more.n_embedded >= 1
    # without the option, or for a 2D structure, nothing is seeded
    assert imp.KEEP_COORDS not in inputs.read_file(p, options=NONE)[0].meta
    flat = write_sdf(tmp_path / "flat.sdf", [("TFA", TFA, {})])
    assert imp.KEEP_COORDS not in inputs.read_file(flat, options=replace_opts(keep_coords=True))[0].meta


def test_seed_coordinates_survive_the_job_table(tmp_path):
    from quantabench.chem import conformers, jobs

    p = write_sdf(tmp_path / "tfa3d.sdf", [("TFA", TFA, {})], three_d=True, hydrogens=True)
    item = inputs.read_file(p, options=replace_opts(keep_coords=True))[0]
    back = jobs.input_from_json(jobs.input_to_json(item))
    assert back.meta[imp.KEEP_COORDS] is True
    from quantabench.chem.standardize import prepare

    prepared, _notes = prepare(back.to_mol())
    seeds, why = conformers.seed_coordinates(back.to_mol(), prepared)
    assert not why and len(seeds) == 1 and seeds[0].shape == (prepared.GetNumAtoms(), 3)


# ------------------------------------------------------------------------------------------------ filters
SALTS = [("amine HCl", "CCN.Cl", {}), ("TFA hydrate", "OC(=O)C(F)(F)F.O", {}), ("ammonium chloride", "[NH3+]CC.[Cl-]", {}),
         ("sodium acetate", "CC(=O)[O-].[Na+]", {}), ("water", "O", {}), ("iron complex", "CCO.[Fe+2].[Cl-].[Cl-]", {})]


def test_filters_remove_water_and_hcl_and_say_so(tmp_path):
    p = write_sdf(tmp_path / "salts.sdf", SALTS)
    seen = []
    items = {i.id: i for i in inputs.read_file(p, options=ImportOptions(convert_2d=False), log=seen.append)}
    assert canon(items["amine_HCl"].smiles) == canon("CCN") and items["amine_HCl"].meta[imp.REMOVED] == ["Cl"]
    assert canon(items["TFA_hydrate"].smiles) == canon(TFA) and items["TFA_hydrate"].meta[imp.REMOVED] == ["O"]
    assert "removed on import: Cl (acid / small inorganic)" in items["amine_HCl"].meta[imp.NOTES]
    assert canon(items["ammonium_chloride"].smiles) == canon("[NH3+]CC.[Cl-]")       # counter-ions: off by default
    assert canon(items["water"].smiles) == "O"                                      # never the last fragment
    assert any("fragments removed by the import filters (water, acids)" in m for m in seen)
    off = {i.id: i for i in inputs.read_file(p, options=NONE)}
    assert canon(off["amine_HCl"].smiles) == canon("CCN.Cl") and imp.REMOVED not in off["amine_HCl"].meta


def test_counterion_and_largest_fragment_filters(tmp_path):
    p = write_sdf(tmp_path / "salts.sdf", SALTS)
    items = {i.id: i for i in inputs.read_file(p, options=replace_opts(strip_counterions=True))}
    assert canon(items["ammonium_chloride"].smiles) == canon("CC[NH3+]")
    assert canon(items["sodium_acetate"].smiles) == canon("CC(=O)[O-]")
    assert items["sodium_acetate"].meta[imp.FILE_CHARGE] == 0
    # the iron complex: Cl- go as counter-ions, the Fe2+ stays for the build to decide about
    assert canon(items["iron_complex"].smiles) == canon("CCO.[Fe+2]")
    largest = {i.id: i for i in inputs.read_file(p, options=replace_opts(keep_largest=True))}
    assert canon(largest["amine_HCl"].smiles) == canon("CCN")
    assert canon(largest["iron_complex"].smiles) == canon("CCO.[Fe+2].[Cl-].[Cl-]")
    assert any("holds a metal" in n for n in largest["iron_complex"].meta[imp.NOTES])


def test_filters_on_table_rows(tmp_path):
    p = tmp_path / "list.csv"
    p.write_text("id,smiles\nsalt,CCN.Cl\nplain,CCO\n")
    items = inputs.read_file(p, options=ImportOptions())
    assert [canon(i.smiles) for i in items] == [canon("CCN"), canon("CCO")]
    assert items[0].mol is None and items[0].meta[imp.REMOVED] == ["Cl"]


def test_the_build_reports_fragments_removed_on_import(tmp_path):
    from quantabench.chem.pipeline import BuildConfig, run_molecule

    p = write_sdf(tmp_path / "salt.sdf", [("amine HCl", "CCN.Cl", {})])
    item = inputs.read_file(p, options=ImportOptions())[0]
    res = run_molecule(item, BuildConfig(method="mmff", max_conformers=2, refine_top=1, xtb_properties=False,
                                         write_files=False))
    assert res.status == "ok" and res.removed_fragments == ["Cl"]
    assert any("removed on import" in n for n in res.notes) and res.summary_row()["removed_fragments"] == "Cl"


# ------------------------------------------------------------------------------------------------ fields
def test_sd_fields_become_columns(tmp_path):
    from quantabench.chem.pipeline import MoleculeResult

    p = write_sdf(tmp_path / "data.sdf", [("PFOA", PFOA, {"logKow": "4.81", "Hvap": "-12.5 kJ/mol", "CAS": "335-67-1",
                                                          "code": "007", "name": "perfluorooctanoic acid"})])
    item = inputs.read_file(p, options=replace_opts(post_fields=True))[0]
    assert item.meta[imp.FIELDS] == {"logKow": 4.81, "Hvap (kJ/mol)": -12.5, "CAS": "335-67-1", "code": "007",
                                     "name": "perfluorooctanoic acid"}
    row = MoleculeResult(input=item).summary_row()
    assert row["logKow"] == 4.81 and row["Hvap (kJ/mol)"] == -12.5 and row["file_name"] == "perfluorooctanoic acid"
    assert row["name"] == "PFOA"
    text = inputs.read_file(p, options=replace_opts(post_fields=True, parse_fields=False))[0]
    assert text.meta[imp.FIELDS]["logKow"] == "4.81" and text.meta[imp.FIELDS]["Hvap"] == "-12.5 kJ/mol"
    assert imp.FIELDS not in inputs.read_file(p, options=NONE)[0].meta


def test_id_and_name_fields(tmp_path):
    p = write_sdf(tmp_path / "data.sdf", [("Perfluorooctanoic acid", PFOA, {"CAS": "335-67-1", "short": "PFOA"}),
                                          ("Trifluoroacetic acid", TFA, {"short": "TFA"})])
    items = inputs.read_file(p, options=replace_opts(id_field="cas", name_field="short"))
    assert [i.id for i in items] == ["335-67-1", "Trifluoroacetic_acid"] and [i.name for i in items] == ["PFOA", "TFA"]
    assert inputs.ID_NOTE in items[1].meta                                    # read_sdf's note: no CAS property


def test_parse_value():
    assert imp.parse_value("12") == (12, "") and imp.parse_value(" -1.5e-3 ") == (-0.0015, "")
    assert imp.parse_value("25.3 °C") == (25.3, "°C") and imp.parse_value("4.2 mg/L") == (4.2, "mg/L")
    for text in ("335-67-1", "2026-11-07", "0123", "<0.5", "N/A", "1,234"):
        assert imp.parse_value(text) == (text, "")


# ------------------------------------------------------------------------------------------------ labeling
def test_standard_atom_order_does_not_depend_on_the_file(tmp_path):
    from rdkit import Chem

    a = write_sdf(tmp_path / "a.sdf", [("PFBA", PFBA, {})], three_d=True, hydrogens=True)
    b = write_sdf(tmp_path / "b.sdf", [("PFBA", "FC(F)(F)C(F)(F)C(F)(F)C(=O)O", {})], three_d=True, hydrogens=True)
    read = [inputs.read_file(x, options=NONE)[0].mol for x in (a, b)]
    std = [inputs.read_file(x, options=replace_opts(label_order="standard"))[0].mol for x in (a, b)]
    symbols = [[at.GetSymbol() for at in m.GetAtoms()] for m in (*read, *std)]
    assert symbols[0] != symbols[1] and symbols[2] == symbols[3]
    assert symbols[2][-1] == "H"                                              # hydrogens after the heavy atoms
    assert all(Chem.MolToSmiles(m) == Chem.MolToSmiles(read[0]) for m in std)
    pos = read[0].GetConformer().GetPositions()
    by_symbol = sorted(map(tuple, np.round(pos, 4)))
    assert sorted(map(tuple, np.round(std[0].GetConformer().GetPositions(), 4))) == by_symbol   # coordinates follow
    with pytest.raises(ValueError, match="label_order"):
        ImportOptions(label_order="alphabetical")


# ------------------------------------------------------------------------------------------------ same constitution
def _conformer_sdf(path):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    m = Chem.AddHs(Chem.MolFromSmiles(PFBA))
    ids = list(AllChem.EmbedMultipleConfs(m, numConfs=3, randomSeed=5))
    w = Chem.SDWriter(str(path))
    for cid in ids:
        m.SetProp("_Name", "PFBA")
        w.write(m, confId=cid)
    w.write(m, confId=ids[0])                       # an exact repeat
    e = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(e, randomSeed=1)
    e.SetProp("_Name", "ethanol")
    w.write(e)
    w.close()
    return path


def test_same_constitution_as_molecules_or_conformers(tmp_path):
    from quantabench.chem import jobs

    p = _conformer_sdf(tmp_path / "confs.sdf")
    seen = []
    mols = inputs.read_file(p, options=NONE, log=seen.append)
    assert [i.id for i in mols] == ["PFBA", "PFBA_2", "PFBA_3", "ethanol"]       # the exact repeat is dropped
    assert any("exact duplicate" in m for m in seen)
    confs = inputs.read_file(p, options=replace_opts(same_constitution="conformers"))
    assert [i.id for i in confs] == ["PFBA", "ethanol"] and confs[0].mol.GetNumConformers() == 4
    assert len(confs[0].meta[imp.CONFORMER_SOURCES]) == 4
    assert any("read as 4 conformer(s) of one molecule" in n for n in confs[0].meta[imp.NOTES])
    back = jobs.input_from_json(jobs.input_to_json(confs[0]))                 # lanes and resumed runs keep them
    assert back.mol.GetNumConformers() == 4 and jobs.EXTRA_CONFORMERS not in back.meta
    assert np.allclose(back.mol.GetConformer(2).GetPositions(), confs[0].mol.GetConformer(2).GetPositions(), atol=1e-4)
    # legacy read (no options): exact id + structure repeats are dropped, as before
    assert [i.id for i in inputs.read_file(p)] == ["PFBA", "ethanol"]


# ------------------------------------------------------------------------------------------------ survey, settings, CLI
def test_survey_decides_whether_to_ask(tmp_path):
    p = write_sdf(tmp_path / "flat.sdf", [("a", "CCN.Cl", {"logP": "1"}), ("b", "CCO", {}), ("c", "CCO", {})])
    s = imp.survey(inputs.read_raw(p))
    assert (s.n, s.n_2d, s.fields, s.n_multi_fragment) == (3, 3, ["logP"], 1)
    assert s.fragments["acid"] == 1 and s.n_same_constitution == 2 and s.has_choices()
    assert s.readers == ["RDKit (SD file)"]
    q = write_sdf(tmp_path / "plain.sdf", [("TFA", TFA, {})], three_d=True)
    assert not imp.survey(inputs.read_raw(q)).has_choices()


def test_options_from_text_settings_and_json_round_trip(tmp_path):
    from quantabench import settings as sm

    o = ImportOptions().with_text("keep_coords=true, label_order=standard; filter=water+hcl")
    assert o.keep_coords and o.label_order == "standard" and o.filters() == ["water", "acids"]
    o = o.with_text('{"post_fields": true, "same_constitution": "conformers", "filter": ["largest"]}')
    assert o.post_fields and o.same_constitution == "conformers" and o.filters() == ["largest"]
    assert ImportOptions().with_filters("none").filters() == []
    for bad, match in (("keep_coords", "key=value"), ("nope=1", "unknown option"), ("keep_coords=maybe", "true or false"),
                       ("filter=salt", "unknown filter")):
        with pytest.raises(ValueError, match=match):
            ImportOptions().with_text(bad)
    s = sm.Settings()
    o.to_settings(s)
    s.import_.ask = False
    sm.save(s)
    loaded = sm.load()
    assert loaded.import_.ask is False and ImportOptions.from_settings(loaded) == o
    data = json.loads((sm.settings_dir() / sm.SETTINGS_FILE).read_text())
    assert data["import_"]["same_constitution"] == "conformers"
    data["import_"]["label_order"] = "sideways"                              # a hand-edited bad value
    assert ImportOptions.from_settings(sm.Settings.from_dict(data)).label_order == "read"
    assert {f for f in ImportOptions.__dataclass_fields__} == set(sm.ImportSettings.__dataclass_fields__) - {"ask"}
    assert ImportOptions.from_settings(sm.Settings()) == ImportOptions()        # the same defaults


def test_cli_import_flags(tmp_path, capsys):
    from quantabench import cli

    p = write_sdf(tmp_path / "salts.sdf", SALTS[:2])
    args = cli.build_parser().parse_args(["build", "--file", str(p), "--keep-coords", "--no-grow-h",
                                          "--filter", "water", "--label-order", "standard",
                                          "--same-structures", "conformers", "--import-options", "post_fields=true"])
    o = cli.import_options(args)
    assert (o.keep_coords, o.grow_hydrogens, o.label_order, o.same_constitution, o.post_fields) == (
        True, False, "standard", "conformers", True)
    assert o.filters() == ["water"]
    items = cli._read_inputs(args)
    assert [canon(i.smiles) for i in items] == [canon("CCN.Cl"), canon(TFA)]        # HCl kept: only water filtered
    default = cli.import_options(cli.build_parser().parse_args(["build", "--file", str(p)]))
    assert default == ImportOptions()
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["build", "--label-order", "sideways"])
