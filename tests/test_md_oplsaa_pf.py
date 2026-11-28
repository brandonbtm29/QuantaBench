"""FF-1: OPLS-AA/PF (md.oplsaa_pf, the ``oplsaa-pf`` backend of md.forcefield). Typing of representative molecules,
total charges, the source of every parameter, the C6F14 model of the validated pilots unchanged, refusals, and a
LAMMPS smoke run (skipped without the ``lammps`` module)."""

import math

import numpy as np
import pytest
from rdkit import Chem

from quantabench.md import forcefield as F
from quantabench.md import liquid as L
from quantabench.md import oplsaa as O
from quantabench.md import oplsaa_pf as PF

C6F14 = "FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"


def typed(smiles: str) -> F.TypedMolecule:
    return F.type_oplsaa_pf(Chem.AddHs(Chem.MolFromSmiles(smiles)), smiles, smiles)


def _same_model(a: F.TypedMolecule, b: F.TypedMolecule) -> None:
    assert a.atom_types == b.atom_types
    assert np.array_equal(a.charges, b.charges)
    assert list(a.lj_types) == list(b.lj_types)
    for k in a.lj_types:
        assert (a.lj_types[k].epsilon, a.lj_types[k].sigma, a.lj_types[k].mass) == \
               (b.lj_types[k].epsilon, b.lj_types[k].sigma, b.lj_types[k].mass), k
    for attr in ("bond_types", "angle_types", "dihedral_types", "improper_types"):
        da, db = getattr(a, attr), getattr(b, attr)
        assert list(da) == list(db), attr
        for k in da:
            assert (da[k].style, da[k].coeffs) == (db[k].style, db[k].coeffs), (attr, k)
    for attr in ("bonds", "angles", "dihedrals", "impropers", "styles", "mixing", "special_lj", "special_coul"):
        assert getattr(a, attr) == getattr(b, attr), attr


# ---------------------------------------------------------------------------------- the validated C6F14 model
@pytest.mark.parametrize("smiles", [C6F14, "CCCCCC", "Fc1ccccc1", "FC1(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C1(F)F",
                                    "FC(F)(F)C(F)(C(F)(F)F)C(F)(F)F"])
def test_same_model_as_oplsaa_where_oplsaa_types(smiles):
    """The pilots' C6F14 (and every other molecule md.oplsaa types, but hexafluorobenzene: below) gets exactly the
    same types, charges, Lennard-Jones and bonded terms from both backends, and so the same ff.params numbers."""
    mol = F.prepare_molecule(smiles)
    a = F.type_molecule(smiles, "m", "oplsaa", mol=mol)
    b = F.type_molecule(smiles, "m", "oplsaa-pf", mol=mol)
    _same_model(a, b)
    assert b.forcefield == "oplsaa-pf" and b.tier == "published"


def test_c6f14_ff_params_numbers_unchanged(tmp_path):
    mol = F.prepare_molecule(C6F14)
    a = F.type_molecule(C6F14, "C6F14", "oplsaa", mol=mol)
    b = F.type_molecule(C6F14, "C6F14", "oplsaa-pf", mol=mol)
    L.write_ff_params(tmp_path / "a.params", a)
    L.write_ff_params(tmp_path / "b.params", b)
    body = [[ln for ln in (tmp_path / f).read_text(encoding="utf-8").splitlines() if not ln.startswith("#")]
            for f in ("a.params", "b.params")]
    assert body[0] == body[1]
    assert any("dihedral_coeff" in ln for ln in body[1])


def test_hexafluorobenzene_takes_its_own_types():
    """md.oplsaa types every aromatic C-F as fluorobenzene (opls_718/719); GROMACS' oplsaa.ff (and foyer, which
    cites Caleman et al. 2012) has hexafluorobenzene's own opls_720/721 (q 0.13 instead of 0.22). OPLS-AA/PF uses
    them, and 1,2-difluorobenzene's opls_727/728."""
    tm = typed("Fc1c(F)c(F)c(F)c(F)c1F")
    assert set(tm.atom_types) == {"opls_720", "opls_721"}
    assert set(np.round(tm.charges, 6)) == {0.13, -0.13}
    assert set(typed("Fc1ccccc1F").atom_types) == {"opls_727", "opls_728", "opls_145", "opls_146"}
    assert "opls_718" in typed("Fc1ccc(F)cc1").atom_types          # para: no C(F) neighbour, fluorobenzene's type


# ------------------------------------------------------------------------------------------------- typing
CASES = {
    # smiles: (expected types present, formal charge, tier)
    "CCCCF": ({"opls_957", "opls_956", "opls_958"}, 0, "published"),
    "CCCCC(C)F": ({"opls_959", "opls_956"}, 0, "published"),
    "CC(C)(C)F": ({"opls_960", "opls_956"}, 0, "published"),
    "Cc1ccc(F)cc1": ({"opls_148", "opls_718", "opls_719", "opls_145"}, 0, "published"),
    "FC(F)(F)c1ccccc1": ({"opls_724", "opls_725", "opls_726"}, 0, "published"),
    "Fc1ccc(Br)cc1": ({"opls_729", "opls_730", "opls_718"}, 0, "published"),
    "CCCCC(F)(F)C(F)(F)C(F)(F)C(F)(F)F": ({"opls_136", "opls_962", "opls_961"}, 0, "published"),
    "OCC(F)(F)F": ({"opls_160", "opls_161", "opls_162", "opls_163", "opls_164", "opls_165"}, 0, "published"),
    "OCC(F)(F)C(F)(F)C(F)(F)F": ({PF.SILVA_CF2, "opls_164", "opls_965", "opls_160"}, 0, "transferred"),
    "OCCC(F)(F)C(F)(F)C(F)(F)C(F)(F)F": ({"opls_157", "opls_154", "opls_155", "opls_962"}, 0, "composed"),
    "CCCO": ({"opls_157", "opls_154", "opls_155"}, 0, "published"),
    "[O-]S(=O)(=O)C(F)(F)F": ({"clp_CBT", "clp_F1", "clp_SBT", "clp_OTF"}, -1, "published"),
    "[O-]S(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F": ({"clp_C1F", "clp_F1", "clp_SBT", "opls_962", "opls_965"}, -1,
                                                   "published"),
    "[O-]C(=O)C(F)(F)F": ({"clp_CFA", "clp_FFA", "clp_CO2", "clp_O2F"}, -1, "published"),
    "[Na+]": ({"opls_407"}, 1, "published"),
}


@pytest.mark.parametrize("smiles", list(CASES))
def test_types_charges_and_tier(smiles):
    want, q, tier = CASES[smiles]
    tm = typed(smiles)
    assert want <= set(tm.atom_types), set(tm.atom_types)
    assert tm.net_charge == pytest.approx(q, abs=1e-9)        # no redistribution: the published charges add up
    assert tm.tier == tier, tm.notes
    tm.check()


def test_silva_charge_is_derived_by_the_published_rule():
    """Silva et al. 2016: TFE's parameters for -CF2CH2OH, the fluorinated carbon's charge set for neutrality."""
    tm = typed("OCC(F)(F)C(F)(F)C(F)(F)F")
    i = tm.atom_types.index(PF.SILVA_CF2)
    want = -(2 * -0.206 + 0.126 + 2 * 0.083 - 0.635 + 0.429)
    assert tm.charges[i] == pytest.approx(want, abs=1e-9) and want == pytest.approx(0.326)
    assert any("Silva" in n and "charge" in n for n in tm.notes)
    assert any(n.startswith("transfer:") and "CT_4" in n for n in tm.notes)      # the one labelled transfer
    src = [t.source for t in tm.dihedral_types.values() if "TRANSFER" in t.source]
    assert len(src) == 1 and "CT-CT-CT-OH" in src[0]


def test_semifluorinated_junction_uses_padua_torsions():
    tm = typed("CCCCC(F)(F)C(F)(F)C(F)(F)C(F)(F)F")
    padua = {k: t for k, t in tm.dihedral_types.items() if k.startswith("padua")}
    assert {"padua F-CF-CT-HC", "padua F-CF-CT-CT", "padua F-CF-CF-CT", "padua CF-CF-CT-CT"} <= set(padua) or \
        len(padua) >= 4
    assert all("Pádua 2002" in t.source and "il.ff line" in t.source for t in padua.values())
    assert "perfluoro CT-CT-CT-CT" in tm.dihedral_types              # the W&J torsion inside the perfluoro block
    assert "Pierce" in tm.recipe and "NOT applied" in tm.recipe


@pytest.mark.parametrize("smiles, why", [
    ("OCC(F)(F)C(F)F", "CHF2"),
    ("COC(F)(F)C(F)(F)C(F)(F)F", "O outside"),
    ("CCCCOC(=O)C(F)(F)C(F)(F)C(F)(F)F", "O outside"),
    ("FC(F)(Cl)C(F)(Cl)Cl", "not C, H, F, O, S or Br"),
    ("[O-]C(=O)C(F)(F)C(F)(F)C(F)(F)F", "carboxylate not on a CF3"),
    ("CCC(C)(F)F", "no perfluorinated carbon neighbour"),
    ("Fc1c(F)c(F)c(C(F)(F)F)c(F)c1F", r"ring C\(F\) next to"),
    ("FC(F)(F)C(F)(F)N(C(F)(F)C(F)(F)F)C(F)(F)C(F)(F)F", "not C, H, F, O, S or Br"),
    ("O=S(=O)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F", "S outside a sulfonate"),
    ("FC(F)=C(F)C(F)(F)F", "sp2/sp carbon"),
    ("Fc1cccc2ccccc12", "single benzene ring"),
])
def test_refuses_what_no_source_covers(smiles, why):
    with pytest.raises(F.ForceFieldError, match=why):
        typed(smiles)


def test_liquid_slab_refuses_ions_but_typing_allows_them():
    with pytest.raises(F.ForceFieldError, match="net charge"):
        F.type_molecule("[O-]S(=O)(=O)C(F)(F)F", ff="oplsaa-pf")
    assert typed("[O-]S(=O)(=O)C(F)(F)F").net_charge == pytest.approx(-1.0)


# -------------------------------------------------------------------------------------------- provenance
PROVENANCE = ["CCCCF", "Cc1ccc(F)cc1", "FC(F)(F)c1ccccc1", "Fc1ccc(Br)cc1", "Fc1c(F)c(F)c(F)c(F)c1F",
              "CCCCC(F)(F)C(F)(F)C(F)(F)C(F)(F)F", "OCC(F)(F)C(F)(F)C(F)(F)F", "OCCC(F)(F)C(F)(F)C(F)(F)F",
              "[O-]S(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F", "[O-]C(=O)C(F)(F)F", C6F14]


@pytest.mark.parametrize("smiles", PROVENANCE)
def test_every_term_names_its_source_line(smiles):
    tm = typed(smiles)
    for t in tm.lj_types.values():
        assert "line" in t.source and "[" in t.source, t.source
    for table in (tm.bond_types, tm.angle_types, tm.dihedral_types, tm.improper_types):
        for k, t in table.items():
            assert ("ffbonded.itp line" in t.source or "il.ff line" in t.source), (k, t.source)
    assert tm.references and all(r for r in tm.references)
    assert tm.recipe and tm.tier in ("published", "composed", "transferred")


def test_every_table_entry_has_a_source():
    for t in PF.TYPES.values():
        assert "line" in t.source and t.reference in PF.REFERENCES, t.name
        assert t.charge is not None or t.name == PF.SILVA_CF2
    for table in (PF.GMX_BONDS, PF.GMX_ANGLES, PF.GMX_DIHEDRALS, PF.CLP_BONDS, PF.CLP_ANGLES, PF.CLP_DIHEDRALS,
                  PF.CLP_IMPROPERS):
        for k, v in table.items():
            line = v[-2] if table in (PF.GMX_BONDS, PF.GMX_ANGLES, PF.GMX_DIHEDRALS) else v[-1]
            assert isinstance(line, int) and line > 0, k
    # the md.oplsaa entries are carried over with the same lines
    for name, t in O.ATOM_TYPES.items():
        assert PF.TYPES[name].source.endswith(f"line {t.line}") and PF.TYPES[name].charge == t.charge


def test_unit_conversions():
    """GROMACS ½kb(b - b0)² and CL&P's k/2 form both give LAMMPS K = k / 2 (kcal/mol, Å); OPLS V's /4.184."""
    t = PF.bond_parameters("clp_C1F", "clp_F1")
    assert t.coeffs == pytest.approx((3698.0 / 2 / 4.184, 1.323))
    t = PF.bond_parameters("opls_962", "opls_965")
    assert t.coeffs == pytest.approx(O.lammps_bond(("CT", "F")))
    t = PF.dihedral_parameters("opls_965", "opls_962", "opls_136", "opls_140")
    assert t.label == "padua F-CF-CT-HC" and t.coeffs == pytest.approx((0, 0, 1.2134 / 4.184, 0))
    # CL&P's perfluoroalkyl C-C-C-C is W&J's torsion: the same V's as GROMACS' #define
    v, _line = PF.CLP_DIHEDRALS[("CF", "CF", "CF", "CF")]
    assert np.allclose(np.array(v) / 4.184, O.rb_to_opls(O.PERFLUORO_CCCC_RB), atol=1e-5)


# -------------------------------------------------------------------------------------------- LAMMPS smoke
@pytest.mark.parametrize("smiles", ["OCC(F)(F)C(F)(F)C(F)(F)F", "Cc1ccc(F)cc1"])
def test_lammps_smoke_run(smiles, tmp_path):
    """A few hundred steps of the bulk and slab stages of an OPLS-AA/PF liquid: every coefficient is written in a
    form LAMMPS reads, and the energies stay finite (not a result: far too short)."""
    pytest.importorskip("lammps")
    from quantabench.md import surface_tension as S

    tm = F.type_molecule(smiles, "m", "oplsaa-pf")
    cfg = L.LiquidSlabConfig(lateral=25.5, slab_thickness=18.0, packing="lattice", dispersion="ewald",
                             thermo_every=100, dump_every=0)
    L.write_liquid_slab(tm, tmp_path, cfg, density_estimate=1.2)
    L.run_liquid_slab(tmp_path, variables=dict(warmup_ps=0.1, compress_ps=0.2, bulk_ns=0.0003, settle_ns=0.0001,
                                               production_ns=0.0003, prof_every=10, prof_repeat=10, prof_freq=100))
    r = S.analyze(tmp_path, skip_ns=0.0)
    assert math.isfinite(r["sigma"]["mean_mN_m"])
