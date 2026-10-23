import os

from quantabench import resources


def test_thread_resolution():
    total = resources.logical_cores()
    assert resources.resolve_threads(0) == max(1, total - 1)
    assert resources.resolve_threads(10**6) == total
    assert resources.resolve_threads(1) == 1


def test_apply_sets_env():
    n = resources.apply_thread_limits(1)
    assert n == 1 and os.environ["OMP_NUM_THREADS"] == "1"


def test_detect_and_cpu_device():
    rep = resources.detect()
    assert rep.logical_cores >= 1 and rep.summary()
    assert resources.select_device("off") == "cpu"
    assert resources.select_device("auto") in ("cpu", "mps") or resources.select_device("auto").startswith("cuda")


def test_import_checks_and_hints():
    checks = resources.import_checks()
    assert all(ok for _, mod, ok, _ in checks if mod.startswith(("rdkit", "ase"))), checks
    msg = "ImportError: DLL load failed while importing rdForceField: An Application Control policy has blocked this file."
    assert "Smart App Control" in resources.explain_import_error(msg)
    assert resources.explain_import_error("ModuleNotFoundError: x") == ""
