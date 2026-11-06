"""The external-program contract (chem.external), solvents, and the engine plugin registry."""

from __future__ import annotations

import sys
import textwrap
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from ase.calculators.calculator import Calculator, all_changes

from quantabench import settings as settings_mod
from quantabench.chem import engines, external, solvents
from quantabench.chem.optimize import Cancelled


@pytest.fixture(autouse=True)
def _isolated_settings(tmp_path, monkeypatch):
    monkeypatch.setenv(settings_mod.ENV_SETTINGS_DIR, str(tmp_path / "settings"))
    external.clear_cache()
    yield
    external.clear_cache()


def _fake_program(tmp_path: Path, name: str, body: str) -> Path:
    """A tiny Python script that behaves like an external program (works on Windows and Linux)."""
    script = tmp_path / f"{name}_impl.py"
    script.write_text(textwrap.dedent(body), encoding="utf-8")
    if sys.platform == "win32":
        exe = tmp_path / f"{name}.bat"
        exe.write_text(f'@"{sys.executable}" "{script}" %*\n', encoding="utf-8")
    else:
        exe = tmp_path / name
        exe.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8")
        exe.chmod(0o755)
    return exe


# ----------------------------------------------------------------------------- discovery
def test_conda_envs_outside_the_usual_folders_are_found(tmp_path, monkeypatch):
    """Environments made by a conda that lives elsewhere (Homebrew's Miniforge cask on a Mac, which
    install_macos.sh uses when it finds it; any conda that registered them in ~/.conda/environments.txt)."""
    home, cask, other = tmp_path / "home", tmp_path / "Caskroom/miniforge/base", tmp_path / "opt/conda/envs/qb-psi4"
    for p in (home, cask / "envs/qb-tools/bin", other / "bin"):
        p.mkdir(parents=True)
    (home / ".conda").mkdir()
    (home / ".conda/environments.txt").write_text(f"{cask}\n\n{other}\n{tmp_path / 'deleted-env'}\n", encoding="utf-8")
    for var in ("CONDA_PREFIX", "MAMBA_ROOT_PREFIX", "CONDA_ROOT"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(external.Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(external, "HOMEBREW_MINIFORGE", (str(cask),))
    monkeypatch.setattr(external, "IS_WINDOWS", False)            # no %ProgramData% roots on a Windows runner
    monkeypatch.setattr(sys, "platform", "darwin")
    assert external.conda_env_prefixes() == [cask, cask / "envs/qb-tools", other]
    monkeypatch.setattr(sys, "platform", "linux")                 # no Caskroom there; the registry still counts
    assert external.conda_env_prefixes() == [cask, other]
    (home / ".conda/environments.txt").unlink()
    assert external.conda_env_prefixes() == []


def test_find_program_from_env_var_with_version(tmp_path, monkeypatch):
    exe = _fake_program(tmp_path, "xtb", 'print("      * xtb version 6.7.1 (edcfbbe) compiled by x")\n')
    monkeypatch.setenv("QUANTABENCH_XTB", str(exe))
    info = external.find_program("xtb")
    assert info is not None and info.source == "environment variable"
    assert info.version == "6.7.1"
    assert external.version_tuple(info.version) >= (6, 7, 1)


def test_settings_path_wins_over_env(tmp_path, monkeypatch):
    a = _fake_program(tmp_path, "mopac", 'print("MOPAC version 23.2.5")\n')
    sub = tmp_path / "other"
    sub.mkdir()
    b = _fake_program(sub, "mopac", 'print("MOPAC version 22.0.0")\n')
    monkeypatch.setenv("QUANTABENCH_MOPAC", str(b))
    s = settings_mod.Settings()
    s.engines.paths["mopac"] = str(a)
    settings_mod.save(s)
    info = external.find_program("mopac", refresh=True)
    assert info.source == "settings" and info.version == "23.2.5"


def test_missing_program_gives_install_hint(monkeypatch):
    monkeypatch.setattr(external, "_locate", lambda spec: None)
    assert external.find_program("crest", refresh=True) is None
    with pytest.raises(external.ExternalProgramError, match="conda install"):
        external.require_program("crest")


def test_wsl_paths_and_command():
    assert external.to_wsl_path(r"C:\Users\Brand\GitHub Projects\x") == "/mnt/c/Users/Brand/GitHub Projects/x"
    assert external.to_wsl_path("/home/u/x") == "/home/u/x"
    info = external.ProgramInfo("crest", "wsl:crest", "settings")
    assert info.via_wsl and info.command() == ["wsl.exe", "-e", "sh", "-c", external.WSL_WRAPPER, "crest"]


def test_find_python_with_current_interpreter():
    assert external.find_python_with("numpy") == sys.executable


def test_program_env_threads_stack_and_wslenv():
    env = external.program_env(3, {"XTBPATH": "/x"}, via_wsl=True)
    assert env["OMP_NUM_THREADS"] == env["MKL_NUM_THREADS"] == env["OPENBLAS_NUM_THREADS"] == "3"
    assert env["OMP_STACKSIZE"] == "4G" and env["OMP_MAX_ACTIVE_LEVELS"] == "1"
    assert "OMP_NUM_THREADS/u" in env["WSLENV"] and "XTBPATH/u" in env["WSLENV"]
    assert env["PSI_SCRATCH"]


# ----------------------------------------------------------------------------- running
def test_run_program_success_log_and_threads(tmp_path):
    run = external.run_program([sys.executable, "-c", "import os; print('threads', os.environ['OMP_NUM_THREADS'])"],
                               tmp_path / "job", threads=2, log_name="out.log")
    assert run.returncode == 0 and "threads 2" in run.text()
    assert (tmp_path / "job" / "out.log").is_file()


def test_run_program_failure_has_log_tail(tmp_path):
    with pytest.raises(external.ExternalProgramError, match="boom"):
        external.run_program([sys.executable, "-c", "print('boom'); raise SystemExit(3)"], tmp_path)
    run = external.run_program([sys.executable, "-c", "raise SystemExit(3)"], tmp_path, check=False)
    assert run.returncode == 3


def test_run_program_stdin(tmp_path):
    run = external.run_program([sys.executable, "-c", "import sys; print(sys.stdin.read().upper())"], tmp_path,
                               stdin_text="hello")
    assert "HELLO" in run.text()


def test_run_program_cancel_kills_process(tmp_path):
    flag = threading.Event()
    threading.Timer(0.5, flag.set).start()
    t = time.perf_counter()
    with pytest.raises(Cancelled):
        external.run_program([sys.executable, "-c", "import time; time.sleep(60)"], tmp_path,
                             should_stop=flag.is_set, poll_s=0.05)
    assert time.perf_counter() - t < 20


def test_run_program_timeout(tmp_path):
    with pytest.raises(external.ExternalProgramError, match="timed out"):
        external.run_program([sys.executable, "-c", "import time; time.sleep(60)"], tmp_path, timeout=0.5,
                             poll_s=0.05)


def test_work_dir_cleanup_and_keep_on_error(tmp_path, monkeypatch):
    s = settings_mod.Settings()
    s.engines.scratch_dir = str(tmp_path / "scratch")
    settings_mod.save(s)
    with external.work_dir("ok job") as d:
        (d / "a.txt").write_text("x")
    assert not d.exists()
    with pytest.raises(external.ExternalProgramError, match="job files kept in"), external.work_dir("bad") as d2:
        raise external.ExternalProgramError("failed")
    assert d2.exists() and d2.parent == tmp_path / "scratch"


# ----------------------------------------------------------------------------- solvents
def test_solvent_names_and_models():
    assert solvents.canonical("H2O") == "water" and solvents.canonical("gas") == ""
    assert solvents.for_engine("water", "alpb") == "water"
    assert solvents.for_engine("dichloromethane", "alpb") == "ch2cl2"
    assert solvents.for_engine("water", "cosmo") == pytest.approx(78.36)
    assert solvents.for_engine("octanol", "pcm") is None
    assert solvents.for_engine("", "alpb") is None
    with pytest.raises(ValueError):
        solvents.canonical("unobtainium")


# ----------------------------------------------------------------------------- plugin registry
class _HarmonicCalc(Calculator):
    """Bonds as springs to their starting length: a cheap stand-in engine with a solvent shift."""

    implemented_properties = ["energy", "free_energy", "forces", "dipole", "charges"]

    def __init__(self, solvent="", **kw):
        super().__init__(**kw)
        self.solvent = solvent

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        pos = self.atoms.get_positions()
        c = pos.mean(0)
        e = 0.05 * float(((pos - c) ** 2).sum()) - (1.0 if self.solvent else 0.0)
        self.results.update(energy=e, free_energy=e, forces=-0.1 * (pos - c),
                            charges=np.zeros(len(pos)), dipole=np.zeros(3))


@pytest.fixture
def fake_engine(monkeypatch):
    calls = {"factory": [], "props": [], "native": 0}

    def factory(numbers, charge, multiplicity, device, **options):
        calls["factory"].append(options)
        return _HarmonicCalc(solvent=options.get("solvent", ""))

    def props(numbers, positions, charge, multiplicity, **options):
        calls["props"].append(options)
        return {"fake_dipole_debye": 1.25, "fake_gap_ev": 5.0, "charges": [0.0] * len(numbers),
                "_notes": ["fake properties"]}

    def native(atoms, fmax, max_steps, should_stop, log, **options):
        from quantabench.chem.optimize import optimize

        calls["native"] += 1
        return optimize(atoms, fmax, max_steps, "lbfgs", should_stop)

    info = engines.EngineInfo("fake-tb", "Fake TB", "test engine", False, frozenset(range(1, 87)),
                              family="tight binding", solvation=("alpb",), external="", prop_prefix="fake")
    engines.load_plugins()
    monkeypatch.setitem(engines.ENGINES, "fake-tb", info)
    monkeypatch.setitem(engines._REGISTRY, "fake-tb",
                        engines._Plugin(info, lambda: (True, ""), factory, props, native))
    monkeypatch.setattr(engines, "AUTO_ORDER", ["fake-tb"])
    return calls


def test_registry_lists_and_resolves_plugin(fake_engine):
    assert "fake-tb" in engines.method_keys()
    assert engines.resolve_auto([6, 1, 1, 1, 1], "water") == "fake-tb"
    assert engines.supports_solvent("fake-tb", "water") and not engines.supports_solvent("mmff", "water")
    calc = engines.make_calculator("fake-tb", [6, 1], solvent="water", threads=2)
    assert isinstance(calc, _HarmonicCalc) and fake_engine["factory"][-1] == {"solvent": "water", "threads": 2}


def test_pipeline_uses_plugin_solvent_properties_and_native_optimizer(fake_engine):
    from quantabench.chem.inputs import from_smiles
    from quantabench.chem.pipeline import BuildConfig, run_molecule

    cfg = BuildConfig(method="fake-tb", solvent="water", optimizer="native", max_conformers=3, refine_top=2,
                      write_files=False, xtb_properties=False)
    res = run_molecule(from_smiles("CCO", id="ethanol"), cfg)
    assert res.status == "ok", res.error
    assert res.solvent == "water" and res.conformer_search == "rdkit"
    assert fake_engine["native"] >= 1
    assert all(o["solvent"] == "water" for o in fake_engine["factory"])
    assert res.descriptors["fake_dipole_debye"] == 1.25 and res.descriptors["solvent"] == "water"
    assert "fake properties" in res.notes
    assert res.conformers[0].charges is not None


def test_pipeline_notes_gas_phase_for_engine_without_solvent():
    from quantabench.chem.inputs import from_smiles
    from quantabench.chem.pipeline import BuildConfig, run_molecule

    cfg = BuildConfig(method="mmff", solvent="water", max_conformers=2, write_files=False, xtb_properties=False)
    res = run_molecule(from_smiles("CCO", id="ethanol"), cfg)
    assert res.status == "ok" and res.solvent == ""
    assert any("gas phase" in n for n in res.notes)


def test_crest_failure_falls_back_to_rdkit(monkeypatch):
    from quantabench.chem import crest
    from quantabench.chem.inputs import from_smiles
    from quantabench.chem.pipeline import BuildConfig, run_molecule

    def broken(*a, **k):
        raise external.ExternalProgramError("crest not found\nmore detail")

    monkeypatch.setattr(crest, "generate", broken)
    cfg = BuildConfig(method="mmff", conformer_search="crest", max_conformers=2, write_files=False,
                      xtb_properties=False)
    res = run_molecule(from_smiles("CCO", id="ethanol"), cfg)
    assert res.status == "ok" and res.conformer_search == "rdkit"
    assert any("CREST conformer search failed" in n and "more detail" not in n for n in res.notes)


def test_qm_stage_failure_is_a_note(monkeypatch):
    from quantabench.chem import qm
    from quantabench.chem.inputs import from_smiles
    from quantabench.chem.pipeline import BuildConfig, run_molecule

    def broken(*a, **k):
        raise external.ExternalProgramError("psi4 missing")

    monkeypatch.setattr(qm, "run_stage", broken)
    cfg = BuildConfig(method="mmff", qm_method="wb97x-d/def2-svp", max_conformers=2, write_files=False,
                      xtb_properties=False)
    res = run_molecule(from_smiles("CCO", id="ethanol"), cfg)
    assert res.status == "ok" and any("QM stage skipped" in n for n in res.notes)


# ----------------------------------------------------------------------------- review fixes (2026-11-01)
posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell wrapper")


@posix_only
def test_posix_children_get_the_raised_stack_limit(tmp_path):
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
    run = external.run_program(["sh", "-c", "ulimit -s"], tmp_path)
    want = "unlimited" if hard == resource.RLIM_INFINITY else str(hard // 1024)
    assert run.text().strip() == want


@posix_only
def test_missing_program_is_reported_not_run(tmp_path):
    with pytest.raises(external.ExternalProgramError, match="could not start"):
        external.run_program([str(tmp_path / "no_such_xtb"), "x.xyz"], tmp_path)


@posix_only
def test_keyboard_interrupt_kills_the_child(tmp_path):
    """The child has its own session, so Ctrl+C never reaches it: run_program must kill it."""
    import os
    import signal

    calls = {"n": 0}

    def interrupt():
        calls["n"] += 1
        if calls["n"] == 5:
            raise KeyboardInterrupt
        return False

    with pytest.raises(KeyboardInterrupt):
        external.run_program([sys.executable, "-c", "import os, time; open('pid', 'w').write(str(os.getpid())); "
                              "time.sleep(60)"], tmp_path, should_stop=interrupt, poll_s=0.1)
    pid = int((tmp_path / "pid").read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert signal.SIGTERM  # (silence unused-import linters on some platforms)


@posix_only
def test_wsl_wrapper_records_pid_and_cancel_kills_linux_side(tmp_path, monkeypatch):
    """A fake wsl.exe (``--cd DIR -e cmd...``) runs the real wrapper script on Linux."""
    import os

    bindir = tmp_path / "bin"
    bindir.mkdir()
    fake = bindir / "wsl.exe"
    fake.write_text('#!/bin/sh\n[ "$1" = "--cd" ] && { cd "$2"; shift 2; }\n[ "$1" = "-e" ] && shift\nexec "$@"\n')
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    job = tmp_path / "job"
    info = external.ProgramInfo("crest", "wsl:sleep", "settings")
    flag = threading.Event()
    threading.Timer(1.0, flag.set).start()
    with pytest.raises(Cancelled):
        external.run_program([*info.command(), "60"], job, should_stop=flag.is_set, poll_s=0.05)
    pid = int((job / ".qb_pid").read_text())
    time.sleep(0.2)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_wsl_paths_are_ignored_off_windows(tmp_path, monkeypatch):
    exe = _fake_program(tmp_path, "crest", 'print("Version 3.0.2, Tue")\n')
    monkeypatch.setattr(external, "IS_WINDOWS", False)
    s = settings_mod.Settings()
    s.engines.paths["crest"] = "wsl:/home/u/envs/bin/crest"     # synced from a Windows machine
    settings_mod.save(s)
    monkeypatch.setenv("QUANTABENCH_CREST", str(exe))
    info = external.find_program("crest", refresh=True)
    assert info.path == str(exe) and "is for Windows" in info.source


def test_missing_configured_path_is_noted(tmp_path, monkeypatch):
    exe = _fake_program(tmp_path, "xtb", 'print("* xtb version 6.7.1 (x)")\n')
    s = settings_mod.Settings()
    s.engines.paths["xtb"] = str(tmp_path / "moved" / "xtb")
    settings_mod.save(s)
    monkeypatch.setenv("QUANTABENCH_XTB", str(exe))
    info = external.find_program("xtb", refresh=True)
    assert info.path == str(exe) and "not found" in info.source


def test_versions_are_cached_on_disk(tmp_path, monkeypatch):
    exe = _fake_program(tmp_path, "xtb", 'print("* xtb version 6.7.1 (x)")\n')
    monkeypatch.setenv("QUANTABENCH_XTB", str(exe))
    assert external.find_program("xtb", refresh=True).version == "6.7.1"
    external.clear_cache()
    monkeypatch.setattr(external, "_probe_version", lambda *a, **k: pytest.fail("probed again"))
    assert external.find_program("xtb").version == "6.7.1"


def test_bad_settings_types_do_not_break_discovery(tmp_path, monkeypatch):
    import json

    d = tmp_path / "settings"
    d.mkdir(exist_ok=True)
    (d / "settings.json").write_text(json.dumps({"engines": {"paths": None, "keep_scratch": "yes"}}))
    s = settings_mod.load()
    assert s.engines.paths == {} and s.engines.keep_scratch is False
    monkeypatch.setattr(external, "_search", lambda spec: None)
    monkeypatch.delenv("QUANTABENCH_CREST", raising=False)
    assert external.find_program("crest", refresh=True) is None


def test_default_timeout_comes_from_settings(tmp_path):
    s = settings_mod.Settings()
    s.engines.timeout_s = 0.5
    settings_mod.save(s)
    with pytest.raises(external.ExternalProgramError, match="timed out"):
        external.run_program([sys.executable, "-c", "import time; time.sleep(30)"], tmp_path / "j", poll_s=0.05)


def test_ascii_scratch_names():
    assert external.ascii_name("α-pinene oxide") == "pinene_oxide"
    assert external.ascii_name("ααα") == "job"
