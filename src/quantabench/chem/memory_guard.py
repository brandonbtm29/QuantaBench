"""Memory guard for the dxtb steps of a molecule (QSAR-2).

dxtb 0.4 differentiates through its SCF in ``scf_mode="full"`` (its default): autograd keeps the matrices of
every SCF iteration, so the memory of a step grows as (SCF iterations) x nao² (nao = basis functions), and
the polarizability (second derivative with respect to the field) costs about 7x the energy + forces. Four
lane processes of the 325-molecule stress sample were OOM-killed at 2-3.9 GB this way. Before such a step,
``estimate_mb`` predicts its peak (resident memory above the process's current use) from the basis size,
the charge and the spin; the pipeline then takes a cheaper path that gives the same quantity, or skips the
step with a note, when ``current RSS + estimate`` would exceed the lane's budget (``resolve_budget_gb``).

Measured 2026-11-05 (dxtb 0.4.0, torch 2.10 CPU, float64, 1 thread; ru_maxrss above the ~0.58 GB of a
process that has imported torch + dxtb; nao = GFN1 basis functions; scratch/dxtb_memory.py):

    molecule                 atoms  nao  charge  GFN1 E+F  GFN1 charges  GFN1 polariz.  GFN2 E+F  tblite GFN1 polariz.
    PFBA anion (C4)             14   54     -1          -             -       182 MB           -             -
    PFOA anion (C8)             26  102     -1      45 MB             -       287 MB           -             -
    PFDoA (C12)                 38  150      0          -             -       333 MB           -             -
    PFDoA anion                 38  150     -1      80 MB             -       505 MB           -             -
    C16 PFCA anion              50  198     -1          -             -       915 MB           -             -
    C20 PFCA                    62  246      0     108 MB             -       644 MB           -             -
    C20 PFCA anion              62  246     -1     237 MB         27 MB      1508 MB           -         25 MB
    C12 diacid dianion          36  144     -2      62 MB             -       417 MB           -             -
    Ala5                        53  158      0      62 MB             -       370 MB      123 MB             -
    Ala10                      103  308      0     212 MB         37 MB      1480 MB      567 MB         35 MB
    Ala10, NH3+ N-terminus     105  312     +1     217 MB             -      1551 MB           -             -
    Ala10 + Lys, 2 x NH3+      117  344     +2     274 MB             -     > 1925 MB          -             -
    Ala15                      153  458      0     372 MB             -             -     1522 MB            -
    Ala20                      203  608      0     810 MB             -             -    > 1925 MB           -
    Ala25                      253  758      0     920 MB        122 MB             -           -        153 MB
    batched GFN1 E+F, B = 4: Ala5 235 MB, Ala10 831 MB, Ala15 1911 MB (B x one system)

The tblite GFN1-xTB finite-field polarizability is the same quantity as dxtb's autograd one (Ala10: 39.1012 vs
39.1009 Å³ isotropic, 8.5356 vs 8.5351 Å³ anisotropy) at a fraction of the memory, in about the same time
(7.3 vs 6.7 s for Ala10; 92 s for Ala25 with the PyPI wheel's reference BLAS). dxtb's ``scf_mode="implicit"``
is not a substitute: 4-5x less memory for energy + forces, but forces differed by 0.83 eV/Å on the C20 PFCA
anion and its polarizability came out 0.

The rule (``estimate_mb``): ``MB_PER_NAO2[step] x scf_factor x nao² x batch + FIXED_MB``. The coefficients
are the upper envelope of the neutral closed-shell measurements; ``scf_factor`` stands for the longer SCF of
anions (1.7-2.3x the neutral molecule, measured), cations (about 1x, measured; 1.5 kept for the dications
whose SCF took minutes in the stress sample) and open-shell systems or transition metals (the SCF may run to
dxtb's 100 iterations: 6x, from a closed-shell peptide with two electrons removed, which took 5.8x). The rule
is an estimate, not a bound: it is meant to catch the molecules that would take gigabytes, not to predict
each one to 10 %.

Frequencies stage (``frequencies_plan``, called by ``vibrations.analyze`` at the start of its Hessian step).
dxtb's analytic Hessian differentiates the autograd forces once more for each of the 3N coordinates, so its
peak grows as atoms x nao², far above any other step. Measured 2026-11-05 as above (1 thread, an RSS
watchdog killed a run at 2 GB resident, i.e. ~1.36 GB above the 0.64 GB baseline; scratch/hessian_memory.py):

    molecule             atoms  nao  charge  GFN1 analytic   GFN2 analytic   GFN1 finite diff.     GFN1 dipole deriv.
    CF4                      5   20      0     261 MB  2.6 s
    acetate                  7   22     -1     310 MB  2.9 s
    TFA anion (C2)           7   28     -1     353 MB  3.1 s   700 MB  8.3 s    21 MB  3.1 s (all)    27 MB  0.4 s
    C2F6                     8   32      0     413 MB  5.3 s   754 MB  7.0 s
    propane                 11   28      0     465 MB  4.2 s   934 MB 10.0 s
    glycine                 10   30      0     445 MB  4.0 s
    PFPrA anion (C3)        10   40     -1     610 MB  5.2 s
    C3F8                    11   44      0     647 MB  4.6 s
    alanine                 13   38      0     675 MB  5.0 s
    butane / pentane     14/17   36/44   0     653 / 924 MB
    PFBA (C4)               14   54      0    1042 MB  7.1 s
    PFBA anion              13   52     -1    1214 MB  7.1 s                    22 MB  ~7 s           49 MB  0.4 s
    Gly-Gly, Ala-Ala     17/23   52/68   0     > 1360 MB (killed)
    PFPeA anion (C5)        16   64     -1     > 1360 MB (killed)
    PFOA anion (C8)         25  100     -1     > 6.4 GB (killed at 7 GB, earlier run)    45 MB  ~21 s    146 MB  0.6 s
    acetate, GFN2            7   22     -1                     628 MB  6.2 s
    PFDoA anion (C12)       37  148     -1                                      74 MB  ~40 s
    Ala5                    53  158      0                                     135 MB  ~69 s         240 MB  0.9 s
    Ala10                  103  308      0                                     271 MB  ~445 s

(finite differences: 6 of the 3N rows timed, extrapolated; GFN2 finite differences on the PFOA anion: 57 MB,
~50 s; GFN2 dipole derivatives: PFOA anion 203 MB, Ala5 344 MB). Central differences of the analytic forces cost no more memory than one energy + forces evaluation
(the allocator's pool grows to ~2x over the 6N calls) and, on one CPU thread, about the same time as the
analytic Hessian where both run (TFA 3.1 vs 3.1 s, PFBA anion 7.4 vs 7.1 s). The rule:

    analytic   HESSIAN_MB_PER_N_NAO2[method] x scf_factor x atoms x nao² + HESSIAN_FIXED_MB
    finite     FINITE_FACTOR x MB_PER_NAO2[method] x scf_factor x nao² + FIXED_MB
    dipole     DIPOLE_MB_PER_NAO2[method] x scf_factor x nao² + FIXED_MB

(the neutral upper envelope; for GFN2 nao is still counted with the GFN1 basis, which over-counts H). Over
the budget the analytic Hessian becomes finite differences (also with ``hessian="analytic"``, since the
frequencies agree to ~1 cm-1), finite differences that do not fit skip the frequencies with a note, and
dipole derivatives that do not fit skip the IR intensities with a note. With a 5 GB lane (NOVA or the
MacBook with 6-7 lanes) the GFN1 analytic Hessian is kept up to about the C5 PFCA anion (16 atoms) and neutral
organics of ~20 atoms; larger molecules take finite differences, so the frequencies of the 14,000-molecule
runs stay at the memory of their refinement.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Peak resident memory per nao² (MB) of one step for a neutral closed-shell molecule (see the table above)
MB_PER_NAO2 = {
    "gfn1-xtb": 0.0025,             # energy + autograd forces (refinement)
    "gfn2-xtb": 0.0075,             # energy + autograd forces, with the multipole integrals
    "charges": 0.0003,              # GFN1 Mulliken charges and charge dipole, no gradient
    "polarizability": 0.016,        # GFN1 polarizability: autograd of the dipole w.r.t. the field
    "tblite-polarizability": 0.0003,  # GFN1 finite-field polarizability with tblite (7 SCFs, no graph)
}
FIXED_MB = 30.0
SCF_ANION, SCF_CATION, SCF_DIFFICULT = 2.5, 1.5, 6.0
#: Frequencies stage (module docstring): peak MB of the analytic Hessian per (atoms x nao²), neutral closed shell
HESSIAN_MB_PER_N_NAO2 = {"gfn1-xtb": 0.025, "gfn2-xtb": 0.08}
HESSIAN_FIXED_MB = 250.0
#: Finite differences: one energy + forces at a time, but the allocator's pool grows over the 6N calls
FINITE_FACTOR = 2.0
#: Analytic dipole derivatives (IR intensities): peak MB per nao², neutral closed shell
DIPOLE_MB_PER_NAO2 = {"gfn1-xtb": 0.009, "gfn2-xtb": 0.014}
#: The budget of one lane process is never set below this (torch + dxtb alone take ~0.6 GB)
FLOOR_GB = 1.5
TRANSITION_METALS = frozenset([*range(21, 31), *range(39, 49), *range(57, 81), *range(89, 113)])


class MemoryGuardError(RuntimeError):
    """A step would need more memory than the lane's budget allows (the message says how much)."""


def basis_size(numbers) -> int:
    """GFN1-xTB basis functions (H, He: 2 s functions; Li-Ne: s + p; heavier: s + p + d, an upper bound)."""
    return int(sum(2 if z <= 2 else 4 if z <= 10 else 9 for z in (int(x) for x in numbers)))


def scf_factor(numbers, charge: int = 0, multiplicity: int = 1) -> float:
    """How much longer than for a neutral closed-shell molecule the SCF (and so its autograd graph) is
    expected to be (module docstring)."""
    if int(multiplicity) > 1 or any(int(z) in TRANSITION_METALS for z in numbers):
        return SCF_DIFFICULT
    if charge < 0:
        return SCF_ANION
    if charge > 0:
        return SCF_CATION
    return 1.0


def estimate_mb(step: str, numbers, charge: int = 0, multiplicity: int = 1, batch: int = 1) -> float:
    """Estimated peak memory (MB) of ``step`` (a ``MB_PER_NAO2`` key) above what the process already uses."""
    if step not in MB_PER_NAO2:
        raise ValueError(f"no memory model for step {step!r}; known: {', '.join(MB_PER_NAO2)}")
    nao = basis_size(numbers)
    return MB_PER_NAO2[step] * scf_factor(numbers, charge, multiplicity) * nao * nao * max(1, int(batch)) + FIXED_MB


def estimate_frequencies_mb(step: str, method: str, numbers, charge: int = 0, multiplicity: int = 1) -> float:
    """Estimated peak memory (MB) of one part of the frequencies stage of a dxtb ``method``: ``step`` "analytic"
    (autograd Hessian), "finite" (central differences of the forces, one evaluation at a time) or "dipole"
    (autograd dipole derivatives, for the IR intensities). See the frequencies section of the module docstring."""
    if method not in HESSIAN_MB_PER_N_NAO2:
        raise ValueError(f"no frequencies memory model for method {method!r}; known: {', '.join(HESSIAN_MB_PER_N_NAO2)}")
    numbers = [int(z) for z in numbers]
    nao, f = basis_size(numbers), scf_factor(numbers, charge, multiplicity)
    if step == "analytic":
        return HESSIAN_MB_PER_N_NAO2[method] * f * len(numbers) * nao * nao + HESSIAN_FIXED_MB
    if step == "finite":
        return FINITE_FACTOR * MB_PER_NAO2[method] * f * nao * nao + FIXED_MB
    if step == "dipole":
        return DIPOLE_MB_PER_NAO2[method] * f * nao * nao + FIXED_MB
    raise ValueError(f"step must be 'analytic', 'finite' or 'dipole', not {step!r}")


def available_gb() -> float:
    import psutil

    return psutil.virtual_memory().available / 1024**3


def rss_mb() -> float:
    """Resident memory of this process (MB); 0 if it cannot be read."""
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1024**2
    except Exception:  # noqa: BLE001 - no reading: the estimate alone decides
        return 0.0


def resolve_budget_gb(setting: float, n_lanes: int = 1, available: float | None = None) -> float:
    """The memory budget of one lane process in GB: ``setting`` if > 0; for 0 (auto) the machine's available
    memory divided by the lanes, at least ``FLOOR_GB``; < 0 turns the guard off (returns 0)."""
    setting = float(setting or 0.0)
    if setting > 0:
        return setting
    if setting < 0:
        return 0.0
    try:
        avail = available_gb() if available is None else float(available)
    except Exception:  # noqa: BLE001 - psutil missing or failing: the floor
        avail = 0.0
    return max(FLOOR_GB, avail / max(1, int(n_lanes)))


def fits(step: str, numbers, charge: int, multiplicity: int, budget_gb: float, batch: int = 1,
         current_mb: float | None = None) -> tuple[bool, str]:
    """(True, "") when ``step`` fits the budget (or the guard is off: ``budget_gb`` <= 0), else (False, why)."""
    if not budget_gb or budget_gb <= 0:
        return True, ""
    return _check(estimate_mb(step, numbers, charge, multiplicity, batch), numbers, charge, multiplicity, budget_gb,
                  batch, current_mb)


def _check(est: float, numbers, charge: int, multiplicity: int, budget_gb: float, batch: int = 1,
           current_mb: float | None = None) -> tuple[bool, str]:
    numbers = list(numbers)
    now = rss_mb() if current_mb is None else float(current_mb)
    if now + est <= budget_gb * 1024:
        return True, ""
    return False, (f"estimated {est / 1024:.1f} GB for {len(numbers)} atoms ({basis_size(numbers)} basis "
                   f"functions, charge {int(charge):+d}"
                   + (f", multiplicity {int(multiplicity)}" if int(multiplicity) > 1 else "")
                   + (f", {batch} conformers at once" if batch > 1 else "")
                   + f") on top of {now / 1024:.1f} GB in use, budget {budget_gb:.1f} GB per lane")


def properties_plan(numbers, charge: int, multiplicity: int, budget_gb: float, *, polarizability: bool = True,
                    dxtb_polarizability: bool = True, tblite: bool = False,
                    current_mb: float | None = None) -> tuple[bool, str, str]:
    """What the GFN1-xTB properties step may do within the budget: (charges, polarizability path, note).

    The path is "dxtb" (autograd, the default), "tblite" (finite field, the same GFN1-xTB quantity, when
    dxtb's would exceed the budget and tblite is installed) or "" (skipped: not asked for, no dxtb field
    backend, or over the budget without tblite). ``charges`` False means even the charges do not fit: the
    whole step is skipped. ``note`` says what was skipped or replaced and why ("" when nothing was)."""
    ok, why = fits("charges", numbers, charge, multiplicity, budget_gb, current_mb=current_mb)
    if not ok:
        return False, "", f"xTB properties skipped by the memory guard: GFN1-xTB charges {why}"
    if not polarizability or not dxtb_polarizability:
        return True, "", ""
    ok, why = fits("polarizability", numbers, charge, multiplicity, budget_gb, current_mb=current_mb)
    if ok:
        return True, "dxtb", ""
    if tblite:
        ok_tb, why_tb = fits("tblite-polarizability", numbers, charge, multiplicity, budget_gb, current_mb=current_mb)
        if ok_tb:
            return True, "tblite", ("xTB polarizability from tblite GFN1-xTB (finite field) instead of dxtb (autograd): "
                                    f"memory guard, dxtb {why}")
        why = why_tb
    return True, "", (f"xTB polarizability skipped by the memory guard: {why}"
                      + ("" if tblite else " (installing tblite computes it with less memory)"))


def max_batch(method: str, numbers, charge: int, multiplicity: int, budget_gb: float,
              current_mb: float | None = None) -> int | None:
    """Most conformers of ``method`` (dxtb) one batched evaluation may hold within the budget; None for no
    limit (guard off), 0 when not even one fits."""
    if not budget_gb or budget_gb <= 0 or method not in MB_PER_NAO2:
        return None
    now = rss_mb() if current_mb is None else float(current_mb)
    one = estimate_mb(method, numbers, charge, multiplicity) - FIXED_MB
    room = budget_gb * 1024 - now - FIXED_MB
    return max(0, int(room // max(one, 1e-9)))


# ----------------------------------------------------------------------------------- frequencies stage
def process_budget_gb() -> float:
    """This process's budget (GB) for the steps that are not handed the run's config (the frequencies'
    Hessian): what the run recorded with ``resources.set_molecule_limits`` (a lane: its share, resolved by
    the parent), else auto for one process (available memory, at least ``FLOOR_GB``); 0 = guard off."""
    from .. import resources

    setting = resources.molecule_limits().memory_budget_gb
    return resolve_budget_gb(0.0 if setting is None else setting, 1)


@dataclass
class FrequencyPlan:
    """What the frequencies stage of a dxtb method may do within the budget (``frequencies_plan``)."""
    hessian: str                    # "analytic" | "finite" | "" (skipped: not even finite differences fit)
    ir: bool                        # analytic dxtb dipole derivatives (IR intensities) fit (or were not asked)
    notes: list[str] = field(default_factory=list)   # what was replaced or skipped, and why

    @property
    def skipped(self) -> bool:
        return not self.hessian


def _gpu_cap_gb(device: str) -> float | None:
    """Half the free memory of a CUDA/ROCm ``device`` (the dxtb Hessian then lives there, not in RAM)."""
    if not device.startswith("cuda"):
        return None
    try:
        import torch

        free, _total = torch.cuda.mem_get_info(torch.device(device))
        return 0.5 * free / 1024**3
    except Exception:  # noqa: BLE001 - no reading: the host budget decides
        return None


def frequencies_plan(method: str, numbers, charge: int = 0, multiplicity: int = 1, hessian: str = "auto",
                     ir: bool = True, budget_gb: float | None = None, analytic_cap_gb: float | None = None,
                     device: str = "cpu", current_mb: float | None = None) -> FrequencyPlan:
    """The frequencies stage of dxtb ``method`` within the memory budget (module docstring, frequencies).

    ``hessian`` "auto"/"analytic": the autograd Hessian when it fits, else central finite differences of the
    forces (the same frequencies to ~1 cm-1, at the memory of one energy + forces), else skipped; "finite":
    finite differences, or skipped. ``ir``: the autograd dipole derivatives, dropped with a note when they do
    not fit (frequencies and thermochemistry are unaffected).

    ``budget_gb``: the run's resolved budget (``pipeline.memory_budget``); None = ``process_budget_gb()``;
    <= 0 = guard off. ``analytic_cap_gb`` (``vibrations.analyze(max_analytic_memory_gb=)``): a cap on the
    analytic step's own estimate instead of the budget check; on a CUDA/ROCm ``device`` half its free memory
    is the cap (the Hessian is computed there)."""
    budget = process_budget_gb() if budget_gb is None else (float(budget_gb) if budget_gb > 0 else 0.0)
    plan = FrequencyPlan(hessian="finite" if hessian == "finite" else "analytic", ir=bool(ir))
    cap = analytic_cap_gb if analytic_cap_gb is not None else _gpu_cap_gb(device)
    if plan.hessian == "analytic" and (budget > 0 or cap is not None):
        est = estimate_frequencies_mb("analytic", method, numbers, charge, multiplicity)
        if cap is not None:
            ok, why = (est <= cap * 1024, f"estimated {est / 1024:.1f} GB, at most {cap:.1f} GB allowed")
        else:
            ok, why = _check(est, numbers, charge, multiplicity, budget, current_mb=current_mb)
        if not ok:
            plan.hessian = "finite"
            plan.notes.append(f"Hessian by finite differences instead of analytic (memory guard: analytic {why})")
    if budget <= 0:
        return plan
    if plan.hessian == "finite":
        est = estimate_frequencies_mb("finite", method, numbers, charge, multiplicity)
        ok, why = _check(est, numbers, charge, multiplicity, budget, current_mb=current_mb)
        if not ok:
            return FrequencyPlan("", False, [f"frequencies skipped by the memory guard: finite differences {why}"])
    if plan.ir:
        est = estimate_frequencies_mb("dipole", method, numbers, charge, multiplicity)
        ok, why = _check(est, numbers, charge, multiplicity, budget, current_mb=current_mb)
        if not ok:
            plan.ir = False
            plan.notes.append(f"IR intensities skipped by the memory guard: dipole derivatives {why} "
                              "(frequencies and thermochemistry are unaffected)")
    return plan
