"""QSAR-4, property models on the table's columns (analysis.models): ridge regression with its leave-one-out
penalty, k-fold / leave-one-group-out / holdout scoring, the reduced target, the files the table joins.

The dialog (gui.model_dialog) is tested in tests/test_gui_table.py."""

from __future__ import annotations

import csv
import json
import math

import numpy as np
import pytest

from quantabench.analysis import models as mdl

FEATURES = ["f0", "f1", "f2", "f3"]


def _data(n=120, seed=0, missing=0.05):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + 0.1 * rng.normal(size=n)
    Xm = X.copy()
    Xm[rng.random(X.shape) < missing] = np.nan
    return Xm, y, rng


def test_ridge_matches_the_closed_form_and_its_loo_penalty():
    X, y, _ = _data(missing=0.0)
    m = mdl.RidgeModel().fit(X, y)
    Z = (X - X.mean(0)) / X.std(0)
    a = m.alpha
    want = np.linalg.solve(Z.T @ Z + a * np.eye(4), Z.T @ (y - y.mean()))
    np.testing.assert_allclose(m.coef, want, atol=1e-10)
    np.testing.assert_allclose(m.predict(X), Z @ want + y.mean(), atol=1e-9)
    # the chosen α has the smallest brute-force leave-one-out error on the grid
    def loo(alpha):
        errs = []
        for i in range(len(y)):
            keep = np.arange(len(y)) != i
            r = mdl.RidgeModel([alpha]).fit(X[keep], y[keep])
            errs.append(r.predict(X[i:i + 1])[0] - y[i])
        return float(np.mean(np.square(errs)))
    # standardization moves a little with each left-out row, so compare the ranking, not the numbers
    grid = mdl.RIDGE_ALPHAS[::6]
    best = grid[int(np.argmin([loo(g) for g in grid]))]
    near = mdl.RIDGE_ALPHAS[np.argmin(np.abs(np.log(mdl.RIDGE_ALPHAS / best)))]
    assert abs(math.log10(a) - math.log10(near)) <= 1.5
    assert [f for f, _c in m.coefficients(FEATURES)[:2]] == ["f0", "f1"]


def test_run_scores_kfold_groups_and_holdout():
    X, y, rng = _data()
    n = len(y)
    groups = np.array(["a", "b", "c", "d"], dtype=object)[rng.integers(0, 4, n)]
    hold = np.zeros(n)
    hold[:12] = 1
    y_missing = y.copy()
    y_missing[12:15] = np.nan                                  # rows without a target: predicted, never trained
    spec = mdl.ModelSpec("m", mdl.RIDGE, FEATURES, "y", folds=5, groups="g", holdout="h")
    res = mdl.run(spec, X, y_missing, groups=groups, holdout=hold)
    assert res.n_train == n - 15 and res.n_holdout == 12
    assert res.cv["n"] == n - 15 and res.cv["r2"] > 0.9 and res.cv["mae"] < 0.6
    assert res.lgo is not None and res.lgo["groups"] == 4 and res.lgo["n"] == n - 15
    assert res.holdout is not None and res.holdout["n"] == 12 and res.holdout["r2"] > 0.7
    assert np.isnan(res.oof[:15]).all() and np.isfinite(res.oof[15:]).all()
    assert np.isfinite(res.pred).all()                         # every row with a feature has a prediction
    assert res.coefficients[0][0] == "f0" and res.alpha is not None
    text = res.report()
    assert "5-fold CV" in text and "leave one g out (4 groups)" in text and "holdout (h)" in text
    # the same seed, the same folds
    again = mdl.run(spec, X, y_missing, groups=groups, holdout=hold)
    np.testing.assert_array_equal(again.oof, res.oof)
    # a holdout column of words
    words = np.array(["test" if h else "" for h in hold], dtype=object)
    assert mdl.run(spec, X, y_missing, groups=groups, holdout=words).n_holdout == 12


def test_reduced_target_models_the_ratio_and_multiplies_back():
    X, y, rng = _data(missing=0.0)
    scale = rng.uniform(1.0, 3.0, len(y))
    sigma = (y + 10.0) * scale                                  # σ* = σ/scale is linear in the features
    plain = mdl.run(mdl.ModelSpec("p", mdl.RIDGE, FEATURES, "sigma"), X, sigma)
    scale[0] = np.nan                                           # no scale: that row neither trains nor predicts
    red = mdl.run(mdl.ModelSpec("r", mdl.RIDGE, FEATURES, "sigma", reduced=True, scale="s"), X, sigma, scale=scale)
    assert red.n_train == len(y) - 1 and np.isnan(red.pred[0]) and np.isnan(red.oof[0])
    assert red.cv["mae"] < 0.5 * plain.cv["mae"]
    assert np.isfinite(red.pred[1:]).all()
    assert abs(np.nanmean(red.pred - sigma)) < 0.1


def test_run_refuses_what_it_cannot_fit():
    X, y, _ = _data(n=8)
    with pytest.raises(mdl.ModelError, match="at least one feature"):
        mdl.run(mdl.ModelSpec("m", mdl.RIDGE, [], "y"), np.empty((8, 0)), y)
    with pytest.raises(mdl.ModelError, match="also a feature"):
        mdl.run(mdl.ModelSpec("m", mdl.RIDGE, ["y", "f1", "f2", "f3"], "y"), X, y)
    with pytest.raises(mdl.ModelError, match="training row"):
        mdl.run(mdl.ModelSpec("m", mdl.RIDGE, FEATURES, "y", folds=10), X, y)
    with pytest.raises(mdl.ModelError, match="scale column"):
        mdl.run(mdl.ModelSpec("m", mdl.RIDGE, FEATURES, "y", reduced=True), X, y)
    with pytest.raises(mdl.ModelError, match="no model kind"):
        mdl.make("svm", 0)
    # a feature with no training values is imputed (and said so), not an error
    X2 = X.copy()
    X2[:, 3] = np.nan
    res = mdl.run(mdl.ModelSpec("m", mdl.RIDGE, FEATURES, "y", folds=4), X2, y)
    assert any("f3" in n for n in res.notes)


@pytest.mark.parametrize("kind", [mdl.ELASTIC_NET, mdl.RANDOM_FOREST, mdl.GRADIENT_BOOSTING])
def test_scikit_learn_models(kind):
    pytest.importorskip("sklearn")
    X, y, _ = _data(n=150)
    res = mdl.run(mdl.ModelSpec("m", kind, FEATURES, "y", folds=3), X, y)
    assert res.cv["n"] == 150 and res.cv["r2"] > 0.75
    if kind == mdl.ELASTIC_NET:
        assert res.coefficients[0][0] == "f0" and res.alpha is not None
    elif kind == mdl.RANDOM_FOREST:
        assert res.importances[0][0] == "f0" and sum(v for _f, v in res.importances) == pytest.approx(1.0)
    else:
        assert res.importances == [] and res.coefficients == []     # histogram boosting has no importances


def test_scikit_learn_missing_is_a_clear_message(monkeypatch):
    monkeypatch.setattr(mdl, "sklearn_available", lambda: False)
    with pytest.raises(mdl.ModelError, match="pip install scikit-learn"):
        mdl.make(mdl.RANDOM_FOREST, 0)
    assert isinstance(mdl.make(mdl.RIDGE, 0), mdl.RidgeModel)


def test_write_files_the_table_joins(tmp_path):
    from quantabench.analysis import sheet

    X, y, _ = _data(n=30)
    y[0] = np.nan
    res = mdl.run(mdl.ModelSpec("sig1", mdl.RIDGE, FEATURES, "y", folds=3), X, y)
    ids = [f"m{i}" for i in range(30)]
    csv_path, json_path = mdl.write(res, ids, tmp_path / "models")
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == ["id", "sig1_pred", "sig1_oof"] and len(rows) == 30
    assert rows[0]["id"] == "m0" and rows[0]["sig1_oof"] == "" and rows[0]["sig1_pred"] != ""
    doc = json.loads(json_path.read_text(encoding="utf-8"))
    assert doc["spec"]["features"] == FEATURES and doc["cv"]["n"] == 29 and doc["coefficients"][0][0] == "f0"
    got = sheet.read_join(sheet.JoinSpec(str(csv_path), "id"))
    assert got.block.names == ["sig1_pred", "sig1_oof"] and got.rows == 30
    np.testing.assert_allclose(got.block.columns["sig1_oof"][1:], res.oof[1:], rtol=1e-7)
