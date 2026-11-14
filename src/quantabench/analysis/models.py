"""Property models on the Analyze table's columns (the Model... dialog, QSAR-4): fit, cross-validate, predict.

Ridge regression is plain NumPy and always there. Elastic net, random forest and gradient boosting use scikit-learn
when it is installed (the ``qsar`` extra: ``pip install "quantabench[qsar]"``). Every model imputes missing
features with the training rows' medians and standardizes them, inside each training fold, so the cross-validation
never sees the rows it predicts.

    ModelSpec    what to fit: the kind, the feature columns, the target, the reduced target (predict σ* = target /
                 scale and multiply back: the scale is ``sigma_scale_kT_mN_m``, kT/v_m^⅔, for a surface tension),
                 the folds, a group column for leave-group-out, a holdout column, the seed
    run          -> ModelResult: k-fold out-of-fold predictions and their metrics, leave-one-group-out metrics, the
                 holdout's metrics (rows never used to fit or choose anything), the model fitted on every training
                 row with its predictions for every row of the table, the coefficients (standardized; linear models)
                 or the importances (trees)
    write        the predictions as a CSV (``id``, ``<name>_pred``, ``<name>_oof``) for the table to join, and the spec,
                 metrics and coefficients as JSON beside it

Rows: a row trains when it has a target (and a positive scale for the reduced target), is not held out, and has at
least one feature. The metrics are in the target's units (the reduced target's predictions are multiplied back).

Ridge's penalty is chosen on the training rows by leave-one-out (the closed form through the SVD, as RidgeCV's
efficient LOO), from 10⁻³ to 10³ in 25 steps; the elastic net's by 5-fold CV (``ElasticNetCV``).
"""

from __future__ import annotations

import json
import math
import time
import warnings
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

RIDGE = "ridge"
ELASTIC_NET = "elastic_net"
RANDOM_FOREST = "random_forest"
GRADIENT_BOOSTING = "gradient_boosting"
#: kind -> (label, needs scikit-learn)
KINDS: dict[str, tuple[str, bool]] = {
    RIDGE: ("Ridge regression (linear, penalty chosen by leave-one-out)", False),
    ELASTIC_NET: ("Elastic net (linear, sparse; penalty by 5-fold CV)", True),
    RANDOM_FOREST: ("Random forest (500 trees)", True),
    GRADIENT_BOOSTING: ("Gradient boosting (histogram, depth 3)", True),
}
RIDGE_ALPHAS = np.logspace(-3, 3, 25)
SKLEARN_MISSING = ("needs scikit-learn: pip install scikit-learn (or the qsar extra, pip install "
                   "\"quantabench[qsar]\"); ridge regression works without it")


def sklearn_available() -> bool:
    try:
        import sklearn  # noqa: F401
    except ImportError:
        return False
    return True


class ModelError(ValueError):
    """The spec cannot be fitted (too few rows, no features, scikit-learn missing): the message says why."""


@dataclass
class ModelSpec:
    name: str
    kind: str
    features: list[str]
    target: str
    reduced: bool = False                # model target/scale, predict × scale
    scale: str = ""                      # the scale column (sigma_scale_kT_mN_m)
    folds: int = 5
    seed: int = 20261113
    groups: str = ""                     # a column: leave-one-group-out
    holdout: str = ""                    # a column: rows where it is non-zero (or true) are held out


@dataclass
class ModelResult:
    spec: ModelSpec
    n_train: int
    n_holdout: int
    cv: dict                             # mae, rmse, r2, bias, n (out-of-fold, k-fold)
    lgo: dict | None                     # the same, leave-one-group-out (None: no group column, < 2 groups)
    holdout: dict | None                 # the final model on the held-out rows that have a target
    oof: np.ndarray                      # out-of-fold prediction per row (NaN: not a training row)
    pred: np.ndarray                     # the final model's prediction per row (NaN: no feature)
    coefficients: list[tuple[str, float]] = field(default_factory=list)   # standardized, by |size|
    importances: list[tuple[str, float]] = field(default_factory=list)
    alpha: float | None = None           # ridge / elastic-net penalty chosen on all training rows
    notes: list[str] = field(default_factory=list)
    seconds: float = 0.0

    def report(self) -> str:
        s = self.spec
        lines = [f"{s.name}: {KINDS[s.kind][0]}", f"target {s.target}" + (f", modelled as {s.target}/{s.scale} "
                                                                          "(reduced) and multiplied back"
                                                                          if s.reduced else ""),
                 f"{len(s.features)} feature(s); {self.n_train:,} training row(s)"
                 + (f", {self.n_holdout:,} held out" if self.n_holdout else ""),
                 _metric_line(f"{s.folds}-fold CV", self.cv)]
        if self.lgo is not None:
            lines.append(_metric_line(f"leave one {s.groups} out ({self.lgo['groups']} groups)", self.lgo))
        if self.holdout is not None:
            lines.append(_metric_line(f"holdout ({s.holdout})", self.holdout))
        if self.alpha is not None:
            lines.append(f"penalty α = {self.alpha:.3g}")
        if self.coefficients:
            lines.append("standardized coefficients (largest first):")
            lines += [f"  {f:<32s} {c:+.4g}" for f, c in self.coefficients[:20]]
        if self.importances:
            lines.append("importances (largest first):")
            lines += [f"  {f:<32s} {c:.4f}" for f, c in self.importances[:20]]
        lines += [f"note: {n}" for n in self.notes]
        lines.append(f"{self.seconds:.1f} s")
        return "\n".join(lines)


def _metric_line(what: str, m: dict) -> str:
    return (f"{what}: MAE {m['mae']:.4g}, RMSE {m['rmse']:.4g}, R² {m['r2']:.4f}, bias {m['bias']:+.3g}, "
            f"n {m['n']:,}")


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    m = np.isfinite(y) & np.isfinite(pred)
    y, pred = y[m], pred[m]
    if not len(y):
        return {"mae": math.nan, "rmse": math.nan, "r2": math.nan, "bias": math.nan, "n": 0}
    d = pred - y
    ss = float(np.sum((y - y.mean()) ** 2))
    return {"mae": float(np.mean(np.abs(d))), "rmse": float(np.sqrt(np.mean(d ** 2))),
            "r2": 1.0 - float(np.sum(d ** 2)) / ss if ss > 0 else math.nan, "bias": float(np.mean(d)),
            "n": int(len(y))}


# ============================================================================================ models
class _Prep:
    """Median imputation and standardization, fitted on training rows."""

    def fit(self, X: np.ndarray) -> _Prep:
        with warnings.catch_warnings():                  # an all-missing feature: its median is NaN, then 0
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(X, axis=0) if len(X) else np.zeros(X.shape[1])
        self.med = np.where(np.isfinite(med), med, 0.0)
        Xi = self._impute(X)
        self.mu = Xi.mean(axis=0)
        sd = Xi.std(axis=0)
        self.sd = np.where(sd > 0, sd, 1.0)
        return self

    def _impute(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, float, copy=True)
        bad = ~np.isfinite(X)
        if bad.any():
            X[bad] = np.broadcast_to(self.med, X.shape)[bad]
        return X

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (self._impute(X) - self.mu) / self.sd


class RidgeModel:
    """Ridge on standardized features, α by efficient leave-one-out over ``RIDGE_ALPHAS`` (NumPy only)."""

    def __init__(self, alphas=RIDGE_ALPHAS):
        self.alphas = np.asarray(alphas, float)

    def fit(self, X: np.ndarray, y: np.ndarray) -> RidgeModel:
        self.prep = _Prep().fit(X)
        Z = self.prep.transform(X)
        y = np.asarray(y, float)
        self.y0 = float(y.mean())
        yc = y - self.y0
        U, s, Vt = np.linalg.svd(Z, full_matrices=False)
        uty = U.T @ yc
        best = (math.inf, None, None)
        n = len(y)
        for a in self.alphas:
            f = s ** 2 / (s ** 2 + a)
            fitted = U @ (f * uty)
            h = (U ** 2) @ f + 1.0 / n                     # the intercept's leverage too
            loo = (yc - fitted) / np.clip(1.0 - h, 1e-9, None)
            err = float(np.mean(loo ** 2))
            if err < best[0]:
                best = (err, a, Vt.T @ (s / (s ** 2 + a) * uty))
        self.alpha, self.coef = float(best[1]), best[2]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.prep.transform(X) @ self.coef + self.y0

    def coefficients(self, names) -> list[tuple[str, float]]:
        return sorted(zip(names, map(float, self.coef)), key=lambda t: -abs(t[1]))


class SkModel:
    """A scikit-learn estimator behind median imputation and standardization."""

    def __init__(self, kind: str, seed: int):
        if not sklearn_available():
            raise ModelError(f"{KINDS[kind][0]} {SKLEARN_MISSING}")
        from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
        from sklearn.linear_model import ElasticNetCV

        self.kind = kind
        if kind == ELASTIC_NET:
            self.est = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], alphas=np.logspace(-5, 1, 25), cv=5, max_iter=20000,
                                    random_state=seed)
        elif kind == RANDOM_FOREST:
            self.est = RandomForestRegressor(n_estimators=500, min_samples_leaf=2, random_state=seed, n_jobs=-1)
        else:
            self.est = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=400, random_state=seed)

    def fit(self, X: np.ndarray, y: np.ndarray) -> SkModel:
        self.prep = _Prep().fit(X)
        self.est.fit(self.prep.transform(X), np.asarray(y, float))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.est.predict(self.prep.transform(X))

    @property
    def alpha(self) -> float | None:
        return float(self.est.alpha_) if self.kind == ELASTIC_NET else None

    def coefficients(self, names) -> list[tuple[str, float]]:
        if self.kind != ELASTIC_NET:
            return []
        return sorted(((n, float(c)) for n, c in zip(names, self.est.coef_) if c != 0), key=lambda t: -abs(t[1]))

    def importances(self, names) -> list[tuple[str, float]]:
        imp = getattr(self.est, "feature_importances_", None)
        if imp is None:
            return []
        return sorted(zip(names, map(float, imp)), key=lambda t: -t[1])


def make(kind: str, seed: int):
    if kind not in KINDS:
        raise ModelError(f"no model kind {kind!r} ({', '.join(KINDS)})")
    return RidgeModel() if kind == RIDGE else SkModel(kind, seed)


# ============================================================================================ run
def _flag(col: np.ndarray) -> np.ndarray:
    """A holdout column as booleans: non-zero numbers, or text that reads as true."""
    col = np.asarray(col)
    if col.dtype.kind in "fiub":
        x = col.astype(float)
        return np.isfinite(x) & (x != 0)
    return np.array([str(v).strip().lower() in ("1", "true", "yes", "y", "holdout", "test") for v in col])


def run(spec: ModelSpec, X: np.ndarray, y: np.ndarray, scale: np.ndarray | None = None,
        groups: np.ndarray | None = None, holdout: np.ndarray | None = None) -> ModelResult:
    """Fit and cross-validate ``spec`` (module docstring). ``X``: rows × ``spec.features`` (NaN = missing); ``y`` the
    target per row; ``scale`` the reduced target's scale per row; ``groups`` one label per row; ``holdout`` the
    holdout column's values per row."""
    t0 = time.perf_counter()
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n = len(y)
    if X.ndim != 2 or X.shape[0] != n or X.shape[1] != len(spec.features):
        raise ModelError("the feature table and the target do not match")
    if not spec.features:
        raise ModelError("choose at least one feature")
    if spec.target in spec.features:
        raise ModelError(f"the target {spec.target} is also a feature")
    notes = []
    if spec.reduced:
        if scale is None:
            raise ModelError("the reduced target needs its scale column")
        scale = np.asarray(scale, float)
        good_scale = np.isfinite(scale) & (scale > 0)
    else:
        good_scale = np.ones(n, bool)
    held = _flag(holdout) if holdout is not None else np.zeros(n, bool)
    has_feature = np.isfinite(X).any(axis=1)
    train = np.isfinite(y) & ~held & good_scale & has_feature
    idx = np.flatnonzero(train)
    if len(idx) < max(5, spec.folds):
        raise ModelError(f"{len(idx)} training row(s) (a target, not held out"
                         + (", a scale" if spec.reduced else "") + f"): {max(5, spec.folds)} are needed")
    empty = [f for f, ok in zip(spec.features, np.isfinite(X[idx]).any(axis=0)) if not ok]
    if empty:
        notes.append(f"no training values (imputed as 0): {', '.join(empty)}")
    yt = y / scale if spec.reduced else y

    def fit_predict(tr: np.ndarray, te: np.ndarray) -> np.ndarray:
        m = make(spec.kind, spec.seed).fit(X[tr], yt[tr])
        p = m.predict(X[te])
        return p * scale[te] if spec.reduced else p

    rng = np.random.default_rng(spec.seed)
    perm = idx[rng.permutation(len(idx))]
    k = min(spec.folds, len(idx))
    oof = np.full(n, np.nan)
    for f in range(k):
        te = perm[f::k]
        tr = np.setdiff1d(idx, te)
        oof[te] = fit_predict(tr, te)
    cv = metrics(y[idx], oof[idx])

    lgo = None
    if spec.groups and groups is not None:
        g = np.array([("" if v is None or (isinstance(v, float) and math.isnan(v)) else str(v)) for v in groups],
                     dtype=object)
        labels = sorted({g[i] for i in idx if g[i] != ""})
        if 2 <= len(labels) <= 500:
            pg = np.full(n, np.nan)
            for lab in labels:
                te = idx[g[idx] == lab]
                tr = idx[g[idx] != lab]
                if len(tr) >= 5:
                    pg[te] = fit_predict(tr, te)
            lgo = {**metrics(y[idx], pg[idx]), "groups": len(labels)}
        else:
            notes.append(f"leave-group-out skipped: {len(labels)} group(s) in {spec.groups}")

    final = make(spec.kind, spec.seed).fit(X[idx], yt[idx])
    pred = np.full(n, np.nan)
    ok = has_feature & (good_scale if spec.reduced else True)
    if ok.any():
        p = final.predict(X[ok])
        pred[ok] = p * scale[ok] if spec.reduced else p
    hold = None
    hidx = np.flatnonzero(held & np.isfinite(y))
    if len(hidx):
        hold = metrics(y[hidx], pred[hidx])
    res = ModelResult(spec=spec, n_train=len(idx), n_holdout=int(held.sum()), cv=cv, lgo=lgo, holdout=hold, oof=oof,
                      pred=pred, alpha=getattr(final, "alpha", None), notes=notes)
    if hasattr(final, "coefficients"):
        res.coefficients = final.coefficients(spec.features)
    if hasattr(final, "importances"):
        res.importances = final.importances(spec.features)
    res.seconds = time.perf_counter() - t0
    return res


# ============================================================================================ files
def write(res: ModelResult, ids: Sequence[str], folder: str | Path) -> tuple[Path, Path]:
    """``<folder>/<name>.csv`` (id, <name>_pred, <name>_oof) and ``<name>.json`` (spec, metrics, coefficients)."""
    import csv

    folder = Path(folder).expanduser()
    folder.mkdir(parents=True, exist_ok=True)
    name = res.spec.name
    csv_path, json_path = folder / f"{name}.csv", folder / f"{name}.json"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", f"{name}_pred", f"{name}_oof"])
        for ident, p, o in zip(ids, res.pred, res.oof):
            if math.isfinite(p) or math.isfinite(o):
                w.writerow([ident, "" if not math.isfinite(p) else f"{p:.8g}", "" if not math.isfinite(o) else f"{o:.8g}"])
    from .. import __version__

    doc = {"spec": asdict(res.spec), "n_train": res.n_train, "n_holdout": res.n_holdout, "cv": res.cv, "lgo": res.lgo,
           "holdout": res.holdout, "alpha": res.alpha, "coefficients": res.coefficients,
           "importances": res.importances, "notes": res.notes, "quantabench_version": __version__,
           "written": time.strftime("%Y-%m-%dT%H:%M:%S")}
    json_path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return csv_path, json_path
