"""UI-3: the Analyze table (molecules x properties), its plot, formula columns and coefficient fits.

The evaluator, the fit, the column store, the run-folder reader and the state are tested without Qt; the table model
at 14,000 rows, the selection sync, copy and export, the plot and persistence with a window (``needs_display``)."""

import builtins
import csv
import math
import time
from dataclasses import replace

import numpy as np
import pytest
from conftest import HAS_DISPLAY

from quantabench.analysis import formula as fm
from quantabench.analysis import sheet

needs_display = pytest.mark.skipif(not HAS_DISPLAY, reason="needs a display (run under xvfb-run on Linux)")


def _app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _wait(app, cond, timeout=30.0):
    from PySide6.QtCore import QEvent

    t = time.time()
    while not cond() and time.time() - t < timeout:
        app.processEvents()
        app.sendPostedEvents(None, QEvent.Type.DeferredDelete)    # Qt retires old tab bars with deleteLater
        time.sleep(0.02)
    return cond()


def _close(app, w):
    w.close()
    assert _wait(app, lambda: w.closed)


def _window(tmp_path, show=True, workspace="analyze"):
    from quantabench import settings as sm
    from quantabench.gui.main_window import MainWindow

    pytest.importorskip("pyqtgraph")
    w = MainWindow(sm.Settings(output_dir=str(tmp_path / "runs")))
    if show:
        w.resize(1500, 920)
        w.show()
        if workspace:
            w.set_workspace(workspace)
        _wait(_app(), lambda: False, 0.3)
    return w


def _cfg():
    from quantabench.chem import pipeline

    return pipeline.BuildConfig(method="mmff", max_conformers=2, refine_top=2, rmsd_threshold=0.1,
                                xtb_properties=False, protonate=False, write_files=False, threads=1)


MOLECULES = (("ethanol", "CCO"), ("butanol", "CCCCO"), ("octanoic", "OC(=O)CCCCCCC"), ("TFA", "OC(=O)C(F)(F)F"),
             ("PFBA", "OC(=O)C(F)(F)C(F)(F)C(F)(F)F"))


@pytest.fixture(scope="module")
def built():
    """A few molecules built with MMFF (about a second each): real descriptors for the table."""
    from quantabench.chem import pipeline
    from quantabench.chem.inputs import from_smiles

    out = [pipeline.run_molecule(from_smiles(smi, id=ident), _cfg()) for ident, smi in MOLECULES]
    assert all(r.status == "ok" for r in out)
    return out


def _fresh(results):
    """Copies with their own inputs: the window renames ids in place, the module fixture must stay as it is."""
    return [replace(r, input=replace(r.input)) for r in results]


# ============================================================================================ evaluator
def test_formula_evaluates_arithmetic_and_functions():
    x = np.array([1.0, 4.0, np.nan, -1.0])
    y = np.array([2.0, 0.5, 3.0, 2.0])
    cols = {"x": x, "y": y}
    ev = lambda text: fm.evaluate(text, cols)                       # noqa: E731
    np.testing.assert_allclose(ev("x + 2*y - 1")[:2], [4.0, 4.0])
    np.testing.assert_allclose(ev("-x**2")[:2], [-1.0, -16.0])     # ** binds tighter than unary minus, as in Python
    np.testing.assert_allclose(ev("(x + y) / 2")[:2], [1.5, 2.25])
    np.testing.assert_allclose(ev("sqrt(x)")[:2], [1.0, 2.0])
    np.testing.assert_allclose(ev("log(exp(y))"), y)
    np.testing.assert_allclose(ev("log10(100*x)")[:2], [2.0, math.log10(400)])
    np.testing.assert_allclose(ev("abs(x - 3)")[:2], [2.0, 1.0])
    np.testing.assert_allclose(ev("pow(y, 2)"), y ** 2)
    np.testing.assert_allclose(ev("min(x, y, 1.5)")[[0, 1, 3]], [1.0, 0.5, -1.0])
    np.testing.assert_allclose(ev("max(x, y)")[[0, 1, 3]], [2.0, 4.0, 2.0])
    assert np.isnan(ev("x + y")[2]) and np.isnan(ev("max(x, y)")[2])       # a missing value stays missing
    assert np.isnan(ev("sqrt(x)")[3]) and np.isnan(ev("log(x)")[3])        # outside the domain: NaN, no warning
    assert ev("x / 0")[0] == math.inf
    assert fm.evaluate("2**3 + 1e-3", cols, n=4).tolist() == [8.001] * 4   # a constant formula: every row
    assert fm.evaluate("2*3", {}, n=3).tolist() == [6.0, 6.0, 6.0]
    assert np.isinf(fm.evaluate("10**10**10", {}, n=1)[0])                 # floats: no endless integer power
    f = fm.parse("((a*alpha + b*mu**2 + c)/Vm)**4")
    assert f.names == ("a", "alpha", "b", "mu", "c", "Vm")
    assert f.params(["alpha", "mu", "Vm"]) == ["a", "b", "c"] and f.columns(["alpha", "mu", "Vm"]) == ["alpha", "mu", "Vm"]
    # parameters written back keep their precision and their sign under a power
    text = fm.parse("a**2 + b*x").substitute({"a": -1.5, "b": 1 / 3})
    assert fm.parse(text).evaluate({"x": np.array([3.0])})[0] == pytest.approx(2.25 + 1.0, rel=1e-15)


@pytest.mark.parametrize("text, message", [
    ("__import__('os').system('echo hi')", "attribute access"),
    ("__import__('os')", "not a formula function"),
    ("__import__", "names starting with __"),
    ("x.real", "attribute access"),
    ("().__class__.__bases__", "attribute access"),
    ("x[0]", "indexing"),
    ("open('f')", "not a formula function"),
    ("eval('1')", "not a formula function"),
    ("np.log(x)", "attribute access"),
    ("lambda: 1", "lambda"),
    ("[v for v in x]", "comprehensions"),
    ("sum(v for v in x)", "not a formula function"),
    ("f'{x}'", "f-strings"),
    ("x < 1", "comparisons"),
    ("x and y", "and/or"),
    ("x if y else 1", "if-else"),
    ("(x, y)", "tuples"),
    ("(z := 1)", "assignments"),
    ("'text'", "not a number"),
    ("True", "not a number"),
    ("None", "not a number"),
    ("1j", "not a number"),
    ("x % 2", "operator %"),
    ("x // 2", "operator //"),
    ("x @ y", "operator @"),
    ("~x", "operator ~"),
    ("not x", "operator not"),
    ("log", "is a function"),
    ("log(x, y)", "takes 1 argument"),
    ("min(x)", "2 or more"),
    ("pow(x)", "takes 2 argument"),
    ("log(x=1)", "keyword arguments"),
    ("max(*x)", "*arguments"),
    ("x +", "syntax error"),
    ("", "empty"),
    ("x = 1", "syntax error"),
    ("import os", "syntax error"),
    ("9" * 400, "too large"),
    ("(" * 300 + "x" + ")" * 300, ""),
    ("+".join(["x"] * 2000), ""),
])
def test_formula_rejects(text, message, monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("eval/exec must never run a formula")

    monkeypatch.setattr(builtins, "eval", boom)
    monkeypatch.setattr(builtins, "exec", boom)
    with pytest.raises(fm.FormulaError) as err:
        fm.evaluate(text, {"x": np.ones(2), "y": np.ones(2)})
    assert message in str(err.value)


def test_formula_names_must_be_columns():
    assert fm.check("x * 2", ["x"]).names == ("x",)
    with pytest.raises(fm.FormulaError, match="unknown name: os"):
        fm.check("x + os", ["x"])
    with pytest.raises(fm.FormulaError, match="unknown name"):
        fm.evaluate("x + y", {"x": np.ones(2)})
    with pytest.raises(fm.FormulaError, match="nothing to fit"):
        fm.check("x * 2", ["x"], allow_params=True)
    with pytest.raises(fm.FormulaError, match="text column"):
        fm.evaluate("x + 1", {"x": np.array(["a", "b"], dtype=object)})


def test_aliases_are_identifiers():
    names = ["polarizability_bosque_A3", "E (kJ/mol)", "σ (mN/m)", "Volume Å³", "2theta", "log", "class", "a-b",
             "a_b", "μ²/V", "", "___"]
    al = fm.aliases(names, reserved=["id"])
    assert al["polarizability_bosque_A3"] == "polarizability_bosque_A3"          # an identifier is its own alias
    assert al["E (kJ/mol)"] == "E_kJ_mol" and al["σ (mN/m)"] == "sigma_mN_m" and al["Volume Å³"] == "Volume_A3"
    assert al["2theta"] == "c_2theta" and al["log"] == "log_" and al["class"] == "class_"
    assert al["a-b"] == "a_b" and al["a_b"] == "a_b_2" and al["μ²/V"] == "mu2_V"
    assert len(set(al.values())) == len(al) and all(fm.is_name(a) for a in al.values())
    for a in al.values():                                                      # and the parser reads each one
        assert fm.parse(a).names == (a,)
    assert not fm.is_name("__x") and not fm.is_name("log") and not fm.is_name("lambda") and not fm.is_name("1a")


# ============================================================================================ fits
def _parachor_data(n=240, seed=7, noise=0.4):
    """Synthetic liquids: α (Å³), μ (D), V_m (cm³/mol) in realistic ranges, σ from the quantum parachor."""
    rng = np.random.default_rng(seed)
    vm = rng.uniform(60.0, 320.0, n)
    alpha = vm * rng.uniform(0.09, 0.13, n)            # α grows with the molar volume, as in real liquids
    mu = rng.uniform(0.0, 4.0, n)
    a, b, c = 17.5, 6.0, 25.0
    sigma = ((a * alpha + b * mu ** 2 + c) / vm) ** 4
    return alpha, mu, vm, sigma + rng.normal(0.0, noise, n), (a, b, c)


def test_fit_recovers_the_quantum_parachor_coefficients():
    alpha, mu, vm, sigma, (a, b, c) = _parachor_data()
    alpha[:5] = np.nan                                   # rows with a missing value are left out
    sigma[5:8] = np.nan
    cols = {"polarizability": alpha, "dipole": mu, "Vm": vm}
    res = fm.fit("((a*polarizability + b*dipole**2 + c)/Vm)**4", cols, sigma, target_name="sigma")
    assert res.success and res.n == len(sigma) - 8
    assert res.params["a"] == pytest.approx(a, rel=0.02)
    assert res.params["b"] == pytest.approx(b, rel=0.05)
    assert res.params["c"] == pytest.approx(c, abs=4.0)
    assert all(0 < res.stderr[p] < 2.5 for p in "abc")
    assert res.r2 > 0.98 and res.rmse == pytest.approx(0.4, rel=0.2) and res.mae < res.rmse
    assert list(res.params) == ["a", "b", "c"] and "R² = " in res.report()
    # the fitted formula is a formula column like any other and gives the fitted predictions
    pred = fm.evaluate(res.fitted, cols)
    assert fm.parse(res.fitted).names == ("polarizability", "dipole", "Vm")
    ok = np.isfinite(pred) & np.isfinite(sigma)
    mae, rmse, r2 = fm.stats(sigma[ok], pred[ok])
    assert (mae, rmse, r2) == pytest.approx((res.mae, res.rmse, res.r2), rel=1e-9)
    # start values matter for a power law far from 1: given ones close to the answer, the same result
    res2 = fm.fit("((a*polarizability + b*dipole**2 + c)/Vm)**4", cols, sigma, start={"a": 15, "b": 5, "c": 20})
    assert res2.params["a"] == pytest.approx(res.params["a"], rel=1e-4)


def test_fit_linear_and_errors():
    x = np.linspace(0, 10, 50)
    y = 3.0 * x - 2.0
    res = fm.fit("k*x + d", {"x": x}, y)
    assert res.params == pytest.approx({"k": 3.0, "d": -2.0}, abs=1e-8) and res.r2 == pytest.approx(1.0)
    assert fm.parse(res.fitted).evaluate({"x": x}) == pytest.approx(y)
    assert fm.linear_fit(x, y)[:2] == pytest.approx((3.0, -2.0))
    assert fm.linear_fit([1.0, 1.0], [2.0, 3.0]) is None and fm.linear_fit([1.0], [2.0]) is None
    with pytest.raises(fm.FitError, match="nothing to fit"):
        fm.fit("2*x", {"x": x}, y)
    with pytest.raises(fm.FitError, match="more than 2 needed"):
        fm.fit("k*x + d", {"x": x[:2]}, y[:2])
    with pytest.raises(fm.FitError, match="start values"):
        fm.fit("log(k - 2)*x", {"x": x}, y)                          # log(1 - 2) at the start: no value


# ============================================================================================ store, files, state
def test_column_store():
    st = sheet.ColumnStore()
    st.set_row("a", {"x": 1, "label": "foo", "missing": None, "method": "mmff", "nan": float("nan")})
    st.set_row("b", {"x": 2.5, "flag": True})
    assert st.names() == ["x", "label", "flag"] and st.is_numeric("x") and not st.is_numeric("label")
    assert "method" not in st.names()                                    # a row field: the row has it
    np.testing.assert_array_equal(st.column("x", st.slots(["b", "zz", "a"])), [2.5, np.nan, 1.0])
    assert list(st.column("label", st.slots(["a", "b"]))) == ["foo", None] and st.get("b", "flag") == "True"
    block = sheet.Block.from_rows(["b", "c"], [{"x": "3", "q": "1e-3", "r": "text"}, {"x": "", "q": "nan", "r": ""}])
    assert block.columns["x"].dtype == float and block.columns["r"].dtype == object
    v0 = st.version
    st.merge(block)
    assert st.version > v0
    np.testing.assert_array_equal(st.column("x", st.slots(["a", "b", "c"])), [1.0, 3.0, np.nan])
    st.set_row("a", {"x": "oops"})                                       # one text value: a text column
    assert not st.is_numeric("x") and list(st.column("x", st.slots(["a", "b"]))) == ["oops", 3.0]
    st.set_row("b", {}, replace=True)
    assert st.get("b", "q") is None
    for i in range(1000):
        st.set_row(f"m{i}", {"v": i})
    assert st.prune(["m1", "m2", "a"]) == 1000 and len(st) == 3
    np.testing.assert_array_equal(st.column("v", st.slots(["m1", "m2", "a"])), [1.0, 2.0, np.nan])
    assert "q" not in st.names()                                         # columns no molecule has any more go


def test_result_values_and_run_tables(tmp_path, built):
    from quantabench.chem import export, jobs

    res = built[2]
    vals = sheet.result_values(res)
    assert vals["mol_weight"] == pytest.approx(res.descriptors["mol_weight"])
    assert vals["inchikey"] == res.inchikey and "id" not in vals and "best_energy_eV" not in vals
    assert "formula" not in vals and "method" not in vals                # the row's own columns

    run = tmp_path / "run"
    with jobs.JobStore(run) as store:
        store.init_run(_cfg(), None)
        store.add_inputs([r.input for r in built])
        for r in built:
            store.record(replace(r, files=[]), "")
    jobs.rebuild_tables(run)
    block = sheet.read_run_tables(run)
    assert block.ids == [r.input.id for r in built]
    k = block.ids.index("octanoic")
    assert block.columns["mol_weight"][k] == pytest.approx(res.descriptors["mol_weight"])
    assert block.columns["dipole_debye"].dtype == float and block.columns["inchikey"][k] == res.inchikey
    # an interrupted batch has no descriptors.csv yet: the job table has the values
    (run / "descriptors.csv").unlink()
    (run / "summary.csv").unlink()
    block2 = sheet.read_run_tables(run)
    assert block2.ids == block.ids
    np.testing.assert_allclose(block2.columns["tpsa"], block.columns["tpsa"])
    assert export.KEY_COLUMNS[0] in block2.columns


def test_table_state_json():
    st = sheet.TableState(columns={"tpsa": False}, formulas=[sheet.FormulaSpec("f1", "tpsa*2", "note")],
                          plot={"x": "tpsa", "y": "fx:f1", "fit": True})
    back = sheet.TableState.from_json(st.to_json())
    assert back == st
    for junk in (None, "", "{", "[]", '{"version": 99, "formulas": [{"name": "f", "expression": "1"}]}',
                 '{"formulas": [{"name": 1}, "x", {"name": "ok", "expression": "2"}], "columns": [1]}'):
        s = sheet.TableState.from_json(junk)
        assert s.columns == {} and s.plot == {}
    assert sheet.TableState.from_json('{"formulas": [{"name": 1}, {"name": "ok", "expression": "2"}]}').formulas == \
        [sheet.FormulaSpec("ok", "2")]


def test_exports_round_trip(tmp_path):
    headers = ["ID", "E (kJ/mol)", "σ (mN/m)", "note"]
    rows = [["a", -1234.567890123456, 21.5, "=1+1"], ["b", float("nan"), None, "tab\there"], ["c", 3.0, 1e-12, ""]]
    path = sheet.write_csv(tmp_path / "t.csv", headers, rows)
    with open(path, newline="", encoding="utf-8-sig") as f:
        back = list(csv.reader(f))
    assert back[0] == headers and back[1][:3] == ["a", "-1234.56789012346", "21.5"] and back[2][1:3] == ["", ""]
    assert float(back[1][1]) == pytest.approx(rows[0][1], rel=1e-14) and back[3][1] == "3"
    assert sheet.tsv(rows[:2], headers).splitlines()[2] == "b\t\t\ttab here"
    if sheet.xlsx_available():
        from openpyxl import load_workbook

        ws = load_workbook(sheet.write_xlsx(tmp_path / "t.xlsx", headers, rows)).active
        got = [[c.value for c in r] for r in ws.iter_rows()]
        assert got[0] == headers and got[1][:3] == ["a", rows[0][1], 21.5]
        assert got[1][3] == "=1+1" and ws.cell(2, 4).data_type == "s"        # text, not an Excel formula
        assert got[2][1] is None and got[2][2] is None


# ============================================================================================ table model
def _big_model(n=14000):
    from quantabench.gui.molecule_list import MoleculeRow, MoleculeTableModel
    from quantabench.gui.table_panel import SheetModel

    molecules = MoleculeTableModel()
    rows = [MoleculeRow(id=f"DTXSID{i:07d}", name=f"PFAS {i}", smiles="OC(=O)C(F)(F)F" if i % 2 else "CCO",
                        status="ok" if i % 5 else "failed", energy_ev=None if i % 5 == 0 else -float(i % 113),
                        n_conformers=None if i % 5 == 0 else 1 + i % 7, method="mmff" if i % 5 else "",
                        formula="C2HF3O2" if i % 2 else "C2H6O") for i in range(n)]
    molecules.add_rows(rows)
    store = sheet.ColumnStore()
    ids = [r.id for r in rows]
    # numbers spanning decades (2, 10, 100 ...: text order would be wrong), some missing, and a text column
    vals = [{"x": float((i * 7919) % 1000) if i % 11 else None, "tag": f"t{i % 3}"} for i in range(n)]
    store.merge(sheet.Block.from_rows(ids, vals, parse=False))
    return molecules, store, SheetModel(molecules, store)


@needs_display
def test_table_model_at_14000_rows_is_quick_and_sorts_numbers_as_numbers():
    from PySide6.QtCore import Qt

    _app()
    t = time.perf_counter()
    molecules, store, model = _big_model()
    model.add_formula("double_x", "2*x + 1")
    built = time.perf_counter() - t
    assert model.rowCount() == 14000 and model.total_rows() == 14000
    assert built < 5.0, built                                    # well under a second on a desktop
    keys = [c.key for c in model.shown_columns()]
    assert "fx:double_x" in keys and "x" not in keys              # x is not a default column ...
    model.set_shown("x", True)                                    # ... until it is chosen
    cx = [c.key for c in model.shown_columns()].index("x")
    t = time.perf_counter()
    model.sort(cx, Qt.SortOrder.AscendingOrder)
    sort_s = time.perf_counter() - t
    assert sort_s < 1.0, sort_s
    x = np.array([store.get(i, "x") if store.get(i, "x") is not None else np.nan for i in model.view_ids()])
    fin = np.isfinite(x)
    assert fin[: fin.sum()].all() and not fin[fin.sum():].any()   # missing values last
    assert (np.diff(x[fin]) >= 0).all() and x[fin][-1] == 999.0   # numbers, not text: 999 after 100
    assert model.data(model.index(0, cx)) == "0" and model.data(model.index(model.rowCount() - 1, cx)) == ""
    model.sort(cx, Qt.SortOrder.DescendingOrder)
    assert model.data(model.index(0, cx)) == "999" and model.data(model.index(model.rowCount() - 1, cx)) == ""
    ce = [c.key for c in model.shown_columns()].index("best_energy_eV")
    model.sort(ce, Qt.SortOrder.AscendingOrder)
    energies = model.values("best_energy_eV")[model.view_positions()]
    assert energies[0] == -112.0 and np.isnan(energies[-1])           # lowest first, failed molecules last
    v = model.values("fx:double_x")
    np.testing.assert_allclose(v, 2 * model.values("x") + 1)
    model.set_shown("tag", True)
    cf = [c.key for c in model.shown_columns()].index("fx:double_x")
    ct = [c.key for c in model.shown_columns()].index("tag")
    model.sort(ct, Qt.SortOrder.AscendingOrder)
    assert model.data(model.index(0, ct)) == "t0" and model.data(model.index(13999, ct)) == "t2"
    t = time.perf_counter()
    model.set_filter("pfas 1234")
    assert time.perf_counter() - t < 1.0
    assert model.rowCount() == 11 and all("1234" in model.row_id(i) for i in range(11))   # 1234, 12340 ... 12349
    model.set_filter("")
    model.sort(-1)                                                # back to the Molecules dock's order
    assert model.view_ids()[:2] == ["DTXSID0000000", "DTXSID0000001"]
    assert model.data(model.index(0, cf)) == "" and model.data(model.index(1, cf)) == "1839"   # 2*919 + 1
    # a change of the Molecules dock (a molecule finished) reaches the table on the next refresh
    molecules.update("DTXSID0000000", status="ok", energy_ev=-1.0)
    model.refresh()
    assert model.data(model.index(0, [c.key for c in model.shown_columns()].index("status"))) == "built"
    molecules.remove_ids(["DTXSID0000001"])
    model.refresh()
    assert model.rowCount() == 13999 and "DTXSID0000001" not in store


@needs_display
def test_posted_fields_are_columns_and_fit_targets():
    """Data fields an imported table posted (a measured value, a source label) are columns; numbers can be a fit's
    target or part of a formula, by their alias."""
    from quantabench.gui.molecule_list import MoleculeRow, MoleculeTableModel
    from quantabench.gui.table_panel import SheetModel

    _app()
    molecules = MoleculeTableModel()
    x = np.linspace(1.0, 10.0, 12)
    molecules.add_rows([MoleculeRow(id=f"M{i}", fields={"σ exp (mN/m)": 2.0 * v + 1.0, "source": f"lab {i % 2}"})
                        for i, v in enumerate(x)] + [MoleculeRow(id="M_none", fields={"source": "lab 1"})])
    store = sheet.ColumnStore()
    store.merge(sheet.Block(ids=[f"M{i}" for i in range(12)], columns={"volume": x}))
    model = SheetModel(molecules, store)
    sig = model.column("field:σ exp (mN/m)")
    assert sig.alias == "sigma_exp_mN_m" and sig.numeric and model.is_shown(sig.key)
    assert not model.column("field:source").numeric
    assert np.isnan(model.values(sig.key)[-1])
    col = model.add_formula("resid", "sigma_exp_mN_m - 2*volume")
    np.testing.assert_allclose(model.values(col.key)[:12], 1.0)
    res = fm.fit("k*volume + d", model.formula_env(), model.values(sig.key))
    assert res.n == 12 and res.params == pytest.approx({"k": 2.0, "d": 1.0}, abs=1e-8)
    with pytest.raises(fm.FormulaError, match="text column"):
        model.add_formula("bad", "source * 2")


# ============================================================================================ window
@needs_display
def test_analyze_shows_the_table_and_the_plot(tmp_path, built):
    from quantabench.gui import workspaces

    app = _app()
    w = _window(tmp_path)
    try:
        assert workspaces.STATE_VERSION == 2
        ws = w.workspaces
        assert {"table", "plot"} <= ws.visible_docks()
        assert ws.dock("table") is w.table_dock and ws.dock("plot") is w.plot_dock
        assert not w.table_dock.visibleRegion().isEmpty() and not w.plot_dock.visibleRegion().isEmpty()   # in front
        assert set(w.tabifiedDockWidgets(w.table_dock)) >= {w.results_dock, w.spectra_dock}
        assert set(w.tabifiedDockWidgets(w.plot_dock)) >= {w.surface_dock, w.appearance_dock}
        assert w.viewer.width() > 400 and w.viewer.height() > 300                              # the 3D view stays
        view = next(a.menu() for a in w.menuBar().actions() if a.text().replace("&", "") == "View")
        texts = [a.text() for a in view.actions()]
        assert "Table" in texts and "Plot" in texts
        for key in ("build", "calculate", "library"):
            w.set_workspace(key)
            assert not {"table", "plot"} & ws.visible_docks()
        w.set_workspace("analyze")
        w.add_inputs([r.input for r in _fresh(built)], built=_fresh(built))
        assert _wait(app, lambda: w.table_panel.model.rowCount() == len(built), 5)
        assert "mol_weight" in [c.key for c in w.table_panel.model.shown_columns()]
        assert _wait(app, lambda: w.plot_panel.stats.text().endswith(f"n = {len(built)}"), 5), w.plot_panel.stats.text()
    finally:
        _close(app, w)


@needs_display
def test_a_failing_panel_does_not_stop_the_window(tmp_path, monkeypatch):
    from quantabench.gui import table_panel

    def broken(win):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(table_panel, "install", broken)
    app = _app()
    w = _window(tmp_path)
    try:
        log = w.log_view.toPlainText()
        assert "table_panel unavailable: RuntimeError: synthetic failure" in log
        assert "plot_panel unavailable" in log                    # the plot needs the table
        assert w.workspaces.dock("table") is None and w.workspaces.current == "analyze"
        assert "spectra" in w.workspaces.visible_docks()
    finally:
        _close(app, w)


@needs_display
def test_selection_follows_both_ways(tmp_path, built):
    from PySide6.QtCore import Qt

    app = _app()
    w = _window(tmp_path)
    try:
        res = _fresh(built)
        w.add_inputs([r.input for r in res], built=res)
        panel = w.table_panel
        assert _wait(app, lambda: panel.model.rowCount() == len(res), 5)
        w.mol_list.select_id("octanoic")                           # Molecules dock -> table
        assert _wait(app, lambda: panel.current_id() == "octanoic", 5)
        row = panel.model.view_row("butanol")                     # table -> Molecules dock
        panel.view.setCurrentIndex(panel.model.index(row, 2))
        assert w.mol_list.current_id() == "butanol"
        assert _wait(app, lambda: w.current_result() is not None and w.current_result().input.id == "butanol", 5)
        # sorted differently from the Molecules dock: still the same molecule
        panel.view.sortByColumn([c.key for c in panel.model.shown_columns()].index("mol_weight"),
                                Qt.SortOrder.DescendingOrder)
        assert panel.model.view_ids()[0] == "PFBA" and panel.current_id() == "butanol"
        panel.view.setCurrentIndex(panel.model.index(0, 1))
        assert w.mol_list.current_id() == "PFBA"
        # a molecule the Molecules dock's search hides: the search is cleared so it can be selected
        w.search.setText("ethanol")
        w._apply_search()
        assert w.mol_list.count() == 1
        panel.view.setCurrentIndex(panel.model.index(panel.model.view_row("TFA"), 1))
        assert w.mol_list.current_id() == "TFA" and w.mol_list.count() == len(res) and w.search.text() == ""
        # the table's filter hides rows; the Molecules dock's selection of a hidden row leaves the table alone
        panel.filter.setText("ol")
        panel.model.set_filter("ol")
        assert set(panel.model.view_ids()) == {"ethanol", "butanol"}
        w.mol_list.select_id("octanoic")
        assert panel.current_id() in ("", "ethanol", "butanol")
    finally:
        _close(app, w)


@needs_display
def test_copy_and_csv_export_round_trip(tmp_path, built):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    app = _app()
    w = _window(tmp_path)
    try:
        res = _fresh(built)
        w.add_inputs([r.input for r in res], built=res)
        panel, model = w.table_panel, w.table_panel.model
        assert _wait(app, lambda: model.rowCount() == len(res), 5)
        keys = [c.key for c in model.shown_columns()]
        ci, cw, ce = keys.index("id"), keys.index("mol_weight"), keys.index("best_energy_eV")
        sm = panel.view.selectionModel()
        panel.view.setCurrentIndex(model.index(1, ci))
        sm.select(model.index(1, ci), sm.SelectionFlag.ClearAndSelect)
        sm.select(model.index(2, cw), sm.SelectionFlag.Select)
        panel.view.setFocus()
        clip = QApplication.clipboard()
        QTest.keyClick(panel.view, Qt.Key.Key_C, Qt.KeyboardModifier.ControlModifier)
        # Windows: another program (clipboard history, a clipboard manager) can hold the clipboard for a moment; the
        # panel retries its write (set_clipboard_text) and the test waits for the text
        assert _wait(app, lambda: len(clip.text().splitlines()) == 2, 5), repr(clip.text())
        lines = clip.text().splitlines()
        assert lines[0].split("\t")[0] == model.row_id(1) and lines[0].split("\t")[-1] == ""
        weight = float(lines[1].split("\t")[-1])
        assert weight == pytest.approx(next(r for r in res if r.input.id == model.row_id(2)).descriptors["mol_weight"])
        QTest.keyClick(panel.view, Qt.Key.Key_C, Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier)
        assert _wait(app, lambda: clip.text().splitlines()[:1] == ["ID\tmol_weight"], 5), repr(clip.text())

        model.add_formula("vol_per_weight", "molar_volume_packing_cm3 / mol_weight")
        path = tmp_path / "out" / "table.csv"
        path.parent.mkdir()
        panel.export(path)                                           # on a worker thread
        assert _wait(app, lambda: panel.last_export == path, 10)
        with open(path, newline="", encoding="utf-8-sig") as f:
            back = list(csv.reader(f))
        titles, rows = model.export_rows()
        assert back[0] == titles and titles[ce] == "E (kJ/mol)" and titles[-1] == "vol_per_weight"
        assert len(back) == len(rows) + 1
        for got, want in zip(back[1:], rows):
            for g, v in zip(got, want):
                if isinstance(v, float):
                    assert (g == "" and math.isnan(v)) or float(g) == pytest.approx(v, rel=1e-13)
                else:
                    assert g == (v or "")
        e_kj = float(back[1][ce])                                    # energies in the display unit
        ev = next(r for r in res if r.input.id == back[1][ci]).conformers[0].energy_ev
        assert e_kj == pytest.approx(ev * 96.4853, rel=1e-4)
        # Settings > Display in eV: the header, the cells and the export follow
        w.s.display.energy_unit = "eV"
        w.apply_display()
        assert model.headerData(ce, Qt.Orientation.Horizontal) == "E (eV)"
        assert model.export_rows()[1][0][ce] == pytest.approx(next(r for r in res if r.input.id == model.row_id(0))
                                                              .conformers[0].energy_ev)
        panel.export(tmp_path / "out" / "ev.csv", background=False)
        with open(tmp_path / "out" / "ev.csv", newline="", encoding="utf-8-sig") as f:
            assert next(csv.reader(f))[ce] == "E (eV)"
    finally:
        _close(app, w)


@needs_display
def test_xlsx_export(tmp_path, built):
    openpyxl = pytest.importorskip("openpyxl")
    app = _app()
    w = _window(tmp_path)
    try:
        res = _fresh(built)
        w.add_inputs([r.input for r in res], built=res)
        panel, model = w.table_panel, w.table_panel.model
        assert _wait(app, lambda: model.rowCount() == len(res), 5)
        assert panel.a_xlsx.isEnabled()
        path = tmp_path / "table.xlsx"
        panel.export(path)
        assert _wait(app, lambda: panel.last_export == path, 20)
        ws = openpyxl.load_workbook(path).active
        got = [[c.value for c in r] for r in ws.iter_rows()]
        titles, rows = model.export_rows()
        assert got[0] == titles and len(got) == len(rows) + 1
        k = titles.index("mol_weight")
        assert [r[k] for r in got[1:]] == pytest.approx([r[k] for r in rows])
        assert isinstance(got[1][k], float) and got[1][titles.index("ID")] == rows[0][titles.index("ID")]
    finally:
        _close(app, w)


@needs_display
def test_xlsx_action_says_what_to_install(tmp_path, monkeypatch):
    monkeypatch.setattr(sheet, "xlsx_available", lambda: False)
    app = _app()
    w = _window(tmp_path, show=False)
    try:
        a = w.table_panel.a_xlsx
        assert not a.isEnabled() and "pip install openpyxl" in a.toolTip() and "openpyxl" in a.text()
        assert w.table_panel.a_csv.isEnabled()
    finally:
        _close(app, w)


@needs_display
def test_plot_point_click_selects_the_molecule(tmp_path, built):
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtTest import QTest

    app = _app()
    w = _window(tmp_path)
    try:
        res = _fresh(built)
        w.add_inputs([r.input for r in res], built=res)
        plot = w.plot_panel
        assert _wait(app, lambda: w.table_panel.model.rowCount() == len(res), 5)
        plot.set_axes("mol_weight", "tpsa", color="n_fluorine", fit=True)
        _wait(app, lambda: False, 0.4)
        assert plot.fit_info is not None and plot.fit_info[3] == len(res)
        slope, icpt, r2, _n = plot.fit_info
        x = np.array([r.descriptors["mol_weight"] for r in res])
        y = np.array([r.descriptors["tpsa"] for r in res])
        assert (slope, icpt) == pytest.approx(tuple(np.polyfit(x, y, 1)))
        assert f"R² = {r2:.4f}" in plot.stats.text() and "Colour: n_fluorine" in plot.legend.text()
        w.mol_list.select_id("ethanol")
        pos = plot.point_pos("octanoic")
        vb = plot.plot.getPlotItem().getViewBox()
        at = plot.plot.mapFromScene(vb.mapViewToScene(QPointF(*pos)))
        QTest.mouseClick(plot.plot.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, at)
        assert _wait(app, lambda: w.mol_list.current_id() == "octanoic", 5)
        assert _wait(app, lambda: w.table_panel.current_id() == "octanoic", 5)
        assert plot.marker.data["x"][0] == pytest.approx(pos[0])            # the selected molecule is ringed
        # the table's filter filters the plot
        w.table_panel.model.set_filter("anol")
        assert _wait(app, lambda: plot.stats.text().endswith("n = 2"), 5), plot.stats.text()
    finally:
        _close(app, w)


@needs_display
def test_formula_columns_and_choices_persist_across_restarts(tmp_path, built):
    from quantabench.gui.main_window import _window_settings
    from quantabench.gui.table_panel import STATE_KEY

    app = _app()
    w = _window(tmp_path)
    try:
        res = _fresh(built)
        w.add_inputs([r.input for r in res], built=res)
        panel, model = w.table_panel, w.table_panel.model
        assert _wait(app, lambda: model.rowCount() == len(res), 5)
        dlg = panel.edit_formula(modal=False)
        dlg.name.setText("alpha_per_v")
        dlg.expr.setText("polarizability_bosque_A3 / molar_volume_packing_cm3")
        assert dlg.buttons.button(dlg.buttons.StandardButton.Ok).isEnabled() and "OK: a value for 5" in dlg.status.text()
        dlg.accept()
        bad = panel.edit_formula(modal=False)
        bad.expr.setText("alpha_per_v.real")
        assert "attribute access" in bad.status.text()
        bad.expr.setText("nope * 2")
        assert "unknown name: nope" in bad.status.text()
        bad.reject()
        # Fit: the parachor-like form on these five molecules (a fit of three parameters to five points)
        fit = panel.open_fit(modal=False)
        fit.expr.setText("((a*polarizability_bosque_A3 + b*tpsa + c)/molar_volume_packing_cm3)**4")
        fit.target.setCurrentIndex(fit.target.findData("parachor_sugden"))
        assert [fit.params.item(r, 0).text() for r in range(fit.params.rowCount())] == ["a", "b", "c"]
        fit.set_start("a", 20.0)
        fit.run_fit(background=False)
        assert fit.result is not None and fit.result.n == len(res), fit.status.text()
        col = fit.add_column()
        assert col is not None and col.alias == "fit_parachor_sugden" and "R²" in col.note
        model.set_shown("tpsa", False)
        model.set_shown("cpk_area_A2_best", True)
        w.plot_panel.set_axes("mol_weight", "fx:alpha_per_v")
        want = model.values("fx:alpha_per_v").copy()
        fitted = model.values("fx:fit_parachor_sugden").copy()
        assert np.isfinite(want).all() and np.isfinite(fitted).all()
    finally:
        _close(app, w)
    assert "alpha_per_v" in str(_window_settings().value(STATE_KEY))
    w = _window(tmp_path)
    try:
        res = _fresh(built)
        w.add_inputs([r.input for r in res], built=res)
        model = w.table_panel.model
        assert _wait(app, lambda: model.rowCount() == len(res), 5)
        assert [f.name for f in model.formulas()] == ["alpha_per_v", "fit_parachor_sugden"]
        keys = [c.key for c in model.shown_columns()]
        assert "fx:alpha_per_v" in keys and "tpsa" not in keys and "cpk_area_A2_best" in keys
        np.testing.assert_allclose(model.values("fx:alpha_per_v"), want)
        np.testing.assert_allclose(model.values("fx:fit_parachor_sugden"), fitted, rtol=1e-12)
        assert w.plot_panel.y_box.currentData() == "fx:alpha_per_v"
        model.update_formula("alpha_per_v", "apv", "polarizability_bosque_A3 / molar_volume_packing_cm3 * 1000")
        model.add_formula("loop_b", "apv * 2")
        with pytest.raises(fm.FormulaError, match="circular"):
            model.update_formula("apv", "apv", "loop_b + 1")              # apv -> loop_b -> apv
        np.testing.assert_allclose(model.values("fx:apv"), want * 1000)
        np.testing.assert_allclose(model.values("fx:loop_b"), want * 2000)
        model.remove_formula("loop_b")
        model.remove_formula("fit_parachor_sugden")
        assert [f.name for f in model.formulas()] == ["apv"]
    finally:
        _close(app, w)


@needs_display
def test_run_folder_values_are_read_off_the_gui_thread(tmp_path, built, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    from quantabench.chem import export, jobs

    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: pytest.fail(str(a[2]))))
    run = tmp_path / "runs" / "synthetic"
    with jobs.JobStore(run) as store:
        store.init_run(_cfg(), None)
        store.add_inputs([r.input for r in built])
        for r in built:
            r2 = replace(r, files=[])
            r2.files = [str(p) for p in export.write_molecule(r2, run / r.input.id, export.DEFAULT_FORMATS)]
            store.record(r2, r.input.id)
    jobs.rebuild_tables(run)
    app = _app()
    w = _window(tmp_path)
    try:
        w.open_run_folder(folder=run)
        panel = w.table_panel
        assert _wait(app, lambda: w.mol_list.count() == len(built) and not panel.reading(), 10)
        model = panel.model
        assert _wait(app, lambda: "dipole_debye" in [c.key for c in model.columns()], 5)
        k = model.view_row("PFBA")
        col = [c.key for c in model.shown_columns()].index("mol_weight")
        assert float(model.data(model.index(k, col))) == pytest.approx(built[4].descriptors["mol_weight"], rel=1e-5)
        assert model.column("inchikey") is not None and not model.column("inchikey").numeric
    finally:
        _close(app, w)
