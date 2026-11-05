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


def test_hard_exit_only_after_windows_rocm_use(monkeypatch):
    import sys
    import types

    monkeypatch.delenv(resources.HARD_EXIT_ENV, raising=False)
    fake = types.SimpleNamespace(version=types.SimpleNamespace(hip="7.2"),
                                 cuda=types.SimpleNamespace(is_initialized=lambda: True))
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.setattr(sys, "platform", "linux")
    assert not resources.needs_hard_exit()                 # only Windows' ROCm runtime hangs at exit
    monkeypatch.setattr(sys, "platform", "win32")
    assert resources.needs_hard_exit()
    fake.cuda.is_initialized = lambda: False               # torch imported, GPU never touched
    assert not resources.needs_hard_exit()
    fake.version.hip, fake.cuda.is_initialized = None, (lambda: True)   # NVIDIA: exits normally
    assert not resources.needs_hard_exit()
    monkeypatch.setenv(resources.HARD_EXIT_ENV, "1")
    assert resources.needs_hard_exit()
    monkeypatch.setenv(resources.HARD_EXIT_ENV, "0")
    monkeypatch.setattr(sys, "platform", "linux")
    resources.hard_exit_if_needed(3)                       # returns: nothing to do


def test_warp_no_driver_warning_is_filtered(monkeypatch, capfd):
    """Warp prints "Could not find or load the NVIDIA CUDA driver" from C during init on every AMD/CPU lane;
    only that line is dropped, other native output still reaches stderr."""
    import sys
    import types

    calls = []
    fake = types.ModuleType("warp")
    fake.config = types.SimpleNamespace()
    fake.LOG_WARNING = 2

    def init():
        calls.append(1)
        os.write(2, (resources.WARP_NO_DRIVER_MESSAGE + ". GPU execution will not be available.\n").encode())
        os.write(2, b"Warp CUDA error: something real\n")

    fake.init = init
    monkeypatch.setitem(sys.modules, "warp", fake)
    monkeypatch.setattr(resources, "has_package", lambda name: True)
    monkeypatch.setattr(resources, "_WARP_INIT", [])
    assert resources.init_warp_quietly() is True
    assert resources.init_warp_quietly() is True and calls == [1]       # once per process
    err = capfd.readouterr().err
    assert "NVIDIA CUDA driver" not in err and "Warp CUDA error: something real" in err
    assert fake.config.log_level == 2
