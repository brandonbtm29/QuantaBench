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
"""

from __future__ import annotations

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
    est = estimate_mb(step, numbers, charge, multiplicity, batch)
    now = rss_mb() if current_mb is None else float(current_mb)
    if now + est <= budget_gb * 1024:
        return True, ""
    return False, (f"estimated {est / 1024:.1f} GB for {len(list(numbers))} atoms ({basis_size(numbers)} basis "
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
