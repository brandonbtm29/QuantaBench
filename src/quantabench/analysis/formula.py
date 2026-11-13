"""Formula columns and coefficient fits for the Analyze table (UI-3), with no Qt and no ``eval``.

A formula is an arithmetic expression over table columns, evaluated row by row on NumPy arrays:

    sigma_qp = ((a*polarizability_bosque_A3 + b*dipole_debye**2 + c)/molar_volume_packing_cm3)**4

**What a formula may contain:** numbers (``2``, ``0.5``, ``1e-3``), column names, ``+ - * / **``, unary minus
and plus, parentheses, and the functions ``log`` (natural), ``log10``, ``exp``, ``sqrt``, ``abs``, ``min(a, b,
...)`` and ``max(a, b, ...)`` (row by row, two or more arguments) and ``pow(a, b)``. Everything else is refused
with a message saying what: attribute access, indexing, calls of anything but those functions, keyword
arguments, comparisons, ``and``/``or``, ``if``-``else``, lambdas, comprehensions, f-strings, strings and other
constants, and names that are neither a column nor (in a fit) a parameter. ``__import__`` and every other name
starting with ``__`` are refused outright.

**How it is evaluated:** ``ast.parse(text, mode="eval")`` turns the text into a syntax tree (parsing runs no code);
``parse`` checks every node against the whitelist above; ``Formula.evaluate`` walks the tree with NumPy
(float64, ``errstate`` quiet). The text never reaches ``eval``, ``exec`` or ``compile``. Constants are floats
before any arithmetic, so ``10**10**10`` is ``inf``, not a Python integer that takes forever. A missing value (NaN)
gives NaN in that row, and so does a value outside a function's domain (``log`` of a negative number). The
length, the nesting depth and the number of nodes are capped.

**Column names (aliases):** a column is referenced by its alias, a plain ASCII identifier (``alias``). A name
that is one already is its own alias (``polarizability_bosque_A3``, ``best_energy_eV``). Otherwise Greek letters
become their names (``σ`` -> ``sigma``), accents and super/subscripts are dropped to their base character
(``Å`` -> ``A``, ``²`` -> ``2``), every other run of characters that cannot be in an identifier becomes ``_``,
a leading digit gets ``c_``, and a Python keyword or a function name gets a trailing ``_``. ``aliases`` adds
``_2``, ``_3`` to duplicates. The table's column chooser shows each column's alias.

**Fits:** ``fit`` adjusts the names of a formula that are not columns (its parameters) to a target column with
``scipy.optimize.least_squares`` on the rows where the target and every column used are finite, and reports the
values, their standard errors, n, MAE, RMSE and R². ``FitResult.fitted`` is the formula with the parameters
replaced by their values (at full precision), a formula column like any other.
"""

from __future__ import annotations

import ast
import keyword
import math
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

MAX_LENGTH = 4000          # characters
MAX_DEPTH = 60             # nesting of the syntax tree
MAX_NODES = 1000


def _reduce(fn):
    def call(*args):
        out = args[0]
        for a in args[1:]:
            out = fn(out, a)
        return out
    return call


#: name -> (NumPy function, fewest arguments, most arguments or None)
FUNCTIONS: dict[str, tuple] = {
    "log": (np.log, 1, 1),
    "log10": (np.log10, 1, 1),
    "exp": (np.exp, 1, 1),
    "sqrt": (np.sqrt, 1, 1),
    "abs": (np.abs, 1, 1),
    "min": (_reduce(np.minimum), 2, None),       # NaN in any argument: NaN (a missing value stays missing)
    "max": (_reduce(np.maximum), 2, None),
    "pow": (np.power, 2, 2),
}
_BINARY = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.divide, ast.Pow: np.power}
_UNARY = {ast.USub: np.negative, ast.UAdd: np.positive}
_OP_TEXT = {ast.Mod: "%", ast.FloorDiv: "//", ast.MatMult: "@", ast.LShift: "<<", ast.RShift: ">>",
            ast.BitAnd: "&", ast.BitOr: "|", ast.BitXor: "^", ast.Invert: "~", ast.Not: "not"}
_REFUSED = (
    ((ast.Attribute,), "attribute access (x.y) is not allowed"),
    ((ast.Subscript, ast.Slice), "indexing (x[i]) is not allowed"),
    ((ast.Lambda,), "lambda is not allowed"),
    ((ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp), "comprehensions are not allowed"),
    ((ast.JoinedStr, ast.FormattedValue), "f-strings are not allowed"),
    ((ast.Compare,), "comparisons are not allowed"),
    ((ast.BoolOp,), "and/or are not allowed"),
    ((ast.IfExp,), "if-else is not allowed"),
    ((ast.List, ast.Tuple, ast.Set, ast.Dict), "lists, tuples, sets and dicts are not allowed"),
    ((ast.NamedExpr,), "assignments (:=) are not allowed"),
    ((ast.Starred,), "*arguments are not allowed"),
    ((ast.Await, ast.Yield, ast.YieldFrom), "await and yield are not allowed"),
)


class FormulaError(ValueError):
    """A formula that cannot be parsed, is not allowed, or cannot be evaluated on these columns."""


# ============================================================================================ aliases
_NOT_IDENT = re.compile(r"[^0-9A-Za-z_]+")


def _ascii(name: str) -> str:
    out = []
    for ch in name:
        if ch.isascii():
            out.append(ch)
            continue
        uname = unicodedata.name(ch, "")
        if uname.startswith(("GREEK SMALL LETTER ", "GREEK CAPITAL LETTER ")):
            word = uname.split("LETTER ", 1)[1].split()[0].lower()
            out.append(word.capitalize() if "CAPITAL" in uname else word)
            continue
        base = unicodedata.normalize("NFKD", ch).encode("ascii", "ignore").decode()   # Å -> A, ² -> 2, é -> e
        out.append(base or " ")
    return "".join(out)


def alias(name: str) -> str:
    """The identifier a formula uses for a column called ``name`` (module docstring, "Column names")."""
    a = _NOT_IDENT.sub("_", _ascii(str(name)).strip()).strip("_")
    if not a:
        a = "column"
    if a[0].isdigit():
        a = "c_" + a
    if keyword.iskeyword(a) or a in FUNCTIONS:
        a += "_"
    return a


def aliases(names: Iterable[str], reserved: Iterable[str] = ()) -> dict[str, str]:
    """``{name: alias}`` for ``names`` in order, unique among themselves and ``reserved`` (``_2``, ``_3`` ...)."""
    taken = set(reserved)
    out: dict[str, str] = {}
    for name in names:
        if name in out:
            continue
        base = alias(name)
        a, k = base, 2
        while a in taken:
            a = f"{base}_{k}"
            k += 1
        taken.add(a)
        out[name] = a
    return out


def is_name(text: str) -> bool:
    """``text`` can name a formula column: an ASCII identifier, not a keyword, not a function, no leading ``__``."""
    return (bool(text) and text.isascii() and text.isidentifier() and not keyword.iskeyword(text)
            and text not in FUNCTIONS and not text.startswith("__"))


# ============================================================================================ parsing
def _refused(node: ast.AST) -> str:
    for types, what in _REFUSED:
        if isinstance(node, types):
            return f"{what} in a formula"
    return f"{type(node).__name__} is not allowed in a formula"


def _number(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        what = "text" if isinstance(value, (str, bytes)) else repr(value)
        raise FormulaError(f"{what} is not a number: a formula takes numbers and column names")
    try:
        return float(value)
    except OverflowError as exc:
        raise FormulaError(f"the number {str(value)[:20]}... is too large") from exc


class _Checker:
    def __init__(self):
        self.names: list[str] = []
        self.nodes = 0

    def visit(self, node: ast.AST, depth: int = 0) -> None:
        self.nodes += 1
        if depth > MAX_DEPTH:
            raise FormulaError("the formula is nested too deeply")
        if self.nodes > MAX_NODES:
            raise FormulaError("the formula is too long")
        if isinstance(node, ast.Expression):
            self.visit(node.body, depth + 1)
        elif isinstance(node, ast.Constant):
            _number(node.value)
        elif isinstance(node, ast.Name):
            name = node.id
            if name.startswith("__"):
                raise FormulaError(f"{name}: names starting with __ are not allowed")
            if name in FUNCTIONS:
                raise FormulaError(f"{name} is a function: write {name}(...)")
            if name not in self.names:
                self.names.append(name)
        elif isinstance(node, ast.BinOp):
            if type(node.op) not in _BINARY:
                op = _OP_TEXT.get(type(node.op), type(node.op).__name__)
                raise FormulaError(f"the operator {op} is not allowed (only + - * / **)")
            self.visit(node.left, depth + 1)
            self.visit(node.right, depth + 1)
        elif isinstance(node, ast.UnaryOp):
            if type(node.op) not in _UNARY:
                op = _OP_TEXT.get(type(node.op), type(node.op).__name__)
                raise FormulaError(f"the operator {op} is not allowed (only + - * / ** and a leading minus)")
            self.visit(node.operand, depth + 1)
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, (ast.Attribute, ast.Subscript, ast.Lambda)):
                raise FormulaError(_refused(func))
            if not isinstance(func, ast.Name):
                raise FormulaError("only these functions can be called: " + ", ".join(FUNCTIONS))
            if func.id not in FUNCTIONS:
                raise FormulaError(f"{func.id}() is not a formula function; the functions are "
                                   + ", ".join(FUNCTIONS))
            if node.keywords:
                raise FormulaError(f"{func.id}(): keyword arguments are not allowed")
            for a in node.args:
                if isinstance(a, ast.Starred):
                    raise FormulaError(_refused(a))
            _fn, lo, hi = FUNCTIONS[func.id]
            n = len(node.args)
            if n < lo or (hi is not None and n > hi):
                want = (f"{lo}" if lo == hi else f"{lo} or more" if hi is None else f"{lo} to {hi}")
                raise FormulaError(f"{func.id}() takes {want} argument{'s' if want != '1' else ''}, not {n}")
            for a in node.args:
                self.visit(a, depth + 1)
        else:
            raise FormulaError(_refused(node))


@dataclass(frozen=True)
class Formula:
    """A checked formula: its text, syntax tree and the names it uses (columns or parameters), in order."""

    text: str
    tree: ast.Expression = field(repr=False, compare=False)
    names: tuple[str, ...] = ()

    def columns(self, known: Iterable[str]) -> list[str]:
        known = set(known)
        return [n for n in self.names if n in known]

    def params(self, known: Iterable[str]) -> list[str]:
        """The names that are not columns: the parameters of a fit."""
        known = set(known)
        return [n for n in self.names if n not in known]

    def evaluate(self, values: Mapping[str, object], n: int | None = None) -> np.ndarray:
        """The formula on ``values`` (name -> array or number): a float64 array, one value per row. ``n``: the
        row count when no column is used (a constant formula); else the length of the arrays."""
        env: dict[str, object] = {}
        for name in self.names:
            try:
                v = values[name]
            except KeyError:
                raise FormulaError(f"unknown name {name!r}: not a column") from None
            env[name] = _as_numbers(name, v)
        with np.errstate(all="ignore"):
            out = _eval(self.tree.body, env)
        out = np.asarray(out, dtype=float)
        if out.ndim == 0:
            if n is None:
                n = next((len(v) for v in env.values() if np.ndim(v) == 1), 1)
            return np.full(n, float(out))
        return np.array(out, dtype=float)

    def substitute(self, values: Mapping[str, float]) -> str:
        """The formula's text with ``values`` (parameters) written in as numbers, at full precision."""
        class Sub(ast.NodeTransformer):
            def visit_Name(self, node):  # noqa: N802 - ast's name
                if node.id not in values:
                    return node
                v = float(values[node.id])
                if math.isfinite(v) and v < 0:
                    return ast.UnaryOp(op=ast.USub(), operand=ast.Constant(-v))
                return ast.Constant(v)

        import copy

        tree = Sub().visit(copy.deepcopy(self.tree))
        return ast.unparse(ast.fix_missing_locations(tree))


def _as_numbers(name: str, v) -> object:
    if isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool):
        return float(v)
    try:
        arr = np.asarray(v, dtype=float)
    except (TypeError, ValueError):
        raise FormulaError(f"{name} is a text column: a formula needs numbers") from None
    if arr.ndim > 1:
        raise FormulaError(f"{name}: one value per row expected")
    return arr


def _eval(node: ast.AST, env: Mapping[str, object]):
    if isinstance(node, ast.Constant):
        return np.float64(node.value)
    if isinstance(node, ast.Name):
        return env[node.id]
    if isinstance(node, ast.BinOp):
        return _BINARY[type(node.op)](_eval(node.left, env), _eval(node.right, env))
    if isinstance(node, ast.UnaryOp):
        return _UNARY[type(node.op)](_eval(node.operand, env))
    if isinstance(node, ast.Call):
        fn = FUNCTIONS[node.func.id][0]
        return fn(*(_eval(a, env) for a in node.args))
    raise FormulaError(_refused(node))           # parse() let nothing else through


def parse(text: str) -> Formula:
    """Check ``text`` against the whitelist (module docstring) and return the ``Formula``, or raise
    ``FormulaError`` saying what is wrong."""
    text = str(text or "").strip()
    if not text:
        raise FormulaError("the formula is empty")
    if len(text) > MAX_LENGTH:
        raise FormulaError(f"the formula is longer than {MAX_LENGTH} characters")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        where = f" at character {exc.offset}" if exc.offset else ""
        raise FormulaError(f"syntax error{where}: {exc.msg}") from None
    except (ValueError, RecursionError, MemoryError) as exc:
        raise FormulaError(f"the formula cannot be read ({type(exc).__name__})") from None
    checker = _Checker()
    checker.visit(tree)
    return Formula(text=text, tree=tree, names=tuple(checker.names))


def check(text: str, columns: Iterable[str], allow_params: bool = False) -> Formula:
    """``parse`` plus: every name is one of ``columns`` (or, with ``allow_params``, at least one is not)."""
    f = parse(text)
    unknown = f.params(columns)
    if unknown and not allow_params:
        listed = ", ".join(unknown)
        raise FormulaError(f"unknown name{'s' if len(unknown) > 1 else ''}: {listed} (not a column; the column "
                           "chooser lists every column's formula name; parameters to fit go in Fit...)")
    if allow_params and not unknown:
        raise FormulaError("nothing to fit: every name in the formula is a column (parameters are the other names)")
    return f


def evaluate(text: str | Formula, values: Mapping[str, object], n: int | None = None) -> np.ndarray:
    f = text if isinstance(text, Formula) else parse(text)
    return f.evaluate(values, n)


# ============================================================================================ fits
class FitError(ValueError):
    pass


@dataclass
class FitResult:
    formula: str
    target: str
    params: dict[str, float]
    stderr: dict[str, float]
    n: int
    mae: float
    rmse: float
    r2: float
    success: bool
    message: str
    fitted: str                          # the formula with the parameters written in

    def report(self) -> str:
        lines = [f"{self.target} = {self.formula}"]
        for k, v in self.params.items():
            se = self.stderr.get(k, math.nan)
            lines.append(f"  {k} = {v:.6g}" + (f" ± {se:.2g}" if math.isfinite(se) else ""))
        lines.append(f"n = {self.n}, MAE = {self.mae:.4g}, RMSE = {self.rmse:.4g}, R² = {self.r2:.4f}")
        if not self.success:
            lines.append(f"Not converged: {self.message}")
        return "\n".join(lines)


def stats(y: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    """(MAE, RMSE, R²) of ``pred`` against ``y`` (R² = 1 - SS_res/SS_tot; NaN when y does not vary)."""
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    if not len(y):
        return math.nan, math.nan, math.nan
    res = y - pred
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(res ** 2)) / ss_tot if ss_tot > 0 else math.nan
    return float(np.mean(np.abs(res))), float(np.sqrt(np.mean(res ** 2))), r2


def fit(text: str | Formula, columns: Mapping[str, object], target: Sequence[float] | np.ndarray,
        params: Sequence[str] | None = None, start: Mapping[str, float] | None = None,
        target_name: str = "target") -> FitResult:
    """Fit the parameters of a formula to ``target`` (module docstring, "Fits").

    ``columns``: alias -> array (a lazy mapping is fine: only the columns the formula uses are read). ``params``:
    the names to fit (default: the formula's names that are not in ``columns``). ``start``: start values (default
    1.0). Rows where the target or any column used is not finite are left out."""
    from scipy.optimize import least_squares

    f = text if isinstance(text, Formula) else parse(text)
    names = list(f.names)
    if params is None:
        params = [n for n in names if n not in columns]
    params = list(params)
    if not params:
        raise FitError("nothing to fit: every name in the formula is a column (parameters are the other names)")
    used = [n for n in names if n not in params]
    data: dict[str, np.ndarray] = {}
    for n in used:
        try:
            data[n] = np.asarray(_as_numbers(n, columns[n]), dtype=float)
        except KeyError:
            raise FitError(f"unknown name {n!r}") from None
    y = np.asarray(target, dtype=float)
    mask = np.isfinite(y)
    for n, v in data.items():
        if v.ndim:
            if v.shape != y.shape:
                raise FitError(f"{n} has {v.shape[0]} rows, the target {y.shape[0]}")
            mask &= np.isfinite(v)
    n_rows = int(mask.sum())
    if n_rows <= len(params):
        raise FitError(f"{n_rows} row(s) with every value present: more than {len(params)} needed to fit "
                       f"{', '.join(params)}")
    sub = {k: (v[mask] if v.ndim else v) for k, v in data.items()}
    y = y[mask]
    p0 = np.array([float((start or {}).get(p, 1.0)) for p in params])

    def model(p) -> np.ndarray:
        return f.evaluate({**sub, **dict(zip(params, p))}, n=n_rows)

    r0 = model(p0) - y
    bad = ~np.isfinite(r0)
    if bad.any():
        raise FitError(f"the formula has no value at the start values for {int(bad.sum())} of {n_rows} rows "
                       "(log or sqrt of a negative number, a division by zero?): change the start values")

    def residuals(p) -> np.ndarray:
        r = model(p) - y
        return np.nan_to_num(r, nan=1e12, posinf=1e12, neginf=-1e12)

    sol = least_squares(residuals, p0, x_scale="jac", method="trf")
    pred = model(sol.x)
    mae, rmse, r2 = stats(y, pred)
    se = np.full(len(params), math.nan)
    dof = n_rows - len(params)
    if dof > 0 and sol.jac is not None:
        jtj = sol.jac.T @ sol.jac
        try:
            cov = np.linalg.inv(jtj) * (2.0 * sol.cost / dof)
            se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        except np.linalg.LinAlgError:
            pass
    values = {p: float(v) for p, v in zip(params, sol.x)}
    return FitResult(formula=f.text, target=target_name, params=values,
                     stderr={p: float(s) for p, s in zip(params, se)}, n=n_rows, mae=mae, rmse=rmse, r2=r2,
                     success=bool(sol.success) and bool(np.all(np.isfinite(pred))), message=str(sol.message),
                     fitted=f.substitute(values))


def linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, int] | None:
    """Least-squares line through the finite (x, y) pairs: (slope, intercept, R², n), or None for fewer than two
    points or x without spread."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 2 or float(np.ptp(x)) == 0.0:
        return None
    slope, intercept = np.polyfit(x, y, 1)
    _mae, _rmse, r2 = stats(y, slope * x + intercept)
    return float(slope), float(intercept), r2, int(len(x))


def parity_stats(x: np.ndarray, y: np.ndarray) -> dict | None:
    """How far y is from x (the 1:1 line; y a prediction of x): MAE, RMSE, bias (mean of y − x), R² about the 1:1
    line (1 − Σ(y−x)² / Σ(x − x̄)²) and n, over the finite pairs; None for fewer than two."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 2:
        return None
    d = y - x
    ss = float(np.sum((x - x.mean()) ** 2))
    return {"mae": float(np.mean(np.abs(d))), "rmse": float(np.sqrt(np.mean(d ** 2))), "bias": float(np.mean(d)),
            "r2": 1.0 - float(np.sum(d ** 2)) / ss if ss > 0 else float("nan"), "n": int(len(x))}
