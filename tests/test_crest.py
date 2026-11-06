"""CREST conformer search (chem.crest): command lines, parsing, the topology filter and the WSL path with a
fake CREST, plus one real GFN-FF search on ethanol (marked ``external``)."""

from __future__ import annotations

import json
import os
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from quantabench import settings as settings_mod
from quantabench.chem import crest, external
from quantabench.chem.pipeline import BuildConfig

KCAL = crest.KCAL_PER_HARTREE


@pytest.fixture(autouse=True)
def _scratch(isolated_settings, tmp_path):
    s = settings_mod.Settings()
    s.engines.scratch_dir = str(tmp_path / "scratch")
    settings_mod.save(s)
    external.clear_cache()
    yield tmp_path / "scratch"
    external.clear_cache()


def _fake_program(folder: Path, name: str, body: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    script = folder / f"{name}_impl.py"
    script.write_text(textwrap.dedent(body), encoding="utf-8")
    if sys.platform == "win32":
        exe = folder / f"{name}.bat"
        exe.write_text(f'@"{sys.executable}" "{script}" %*\n', encoding="utf-8")
    else:
        exe = folder / name
        exe.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8")
        exe.chmod(0o755)
    return exe


# A fake CREST: three "conformers" of the input; the lowest-energy one has C0 and O2 swapped (bonds changed).
FAKE_CREST = r'''
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
if "--version" in args:
    print("       Version 3.0.2, fake")
    sys.exit(0)
with open(os.environ["FAKE_CREST_LOG"], "a") as f:
    f.write(json.dumps({"argv": args, "path": os.environ.get("PATH", ""),
                        "threads": os.environ.get("OMP_NUM_THREADS")}) + "\n")
lines = Path(args[0]).read_text().splitlines()
n = int(lines[0])
sym = [ln.split()[0] for ln in lines[2:2 + n]]
pos = [[float(v) for v in ln.split()[1:4]] for ln in lines[2:2 + n]]
mode = os.environ.get("FAKE_CREST_MODE", "")
def frame(s, p, e):
    return [f"  {len(s)}", f"        {e:.8f}"] + [f" {a:<2s} {x:18.10f} {y:18.10f} {z:18.10f}" for a, (x, y, z) in zip(s, p)]
shifted = [[x + 0.3, y, z] for x, y, z in pos]
swapped = list(pos)
swapped[0], swapped[2] = pos[2], pos[0]
frames = [frame(sym, pos, -11.30), frame(sym, shifted, -11.31), frame(sym, swapped, -11.32)]
if mode == "badorder":
    frames = [frame(sym[::-1], pos[::-1], -11.30)]
if mode == "allbad":
    frames = [frame(sym, swapped, -11.32)]
if mode == "macfail":   # conda-forge's osx-arm64 CREST 3.0.2 on 19+ atoms (docs/ENGINES.md)
    print("> Setting up backup calculator ... done.")
    print(" Initial Geometry Optimization")
    print("  Initial geometry optimization failed!")
    print("  Please check your input and, if present, crestopt.log.")
    sys.stderr.write("ERROR STOP\n")
    sys.exit(1)
if mode == "zeroopt":   # exits 0, but no structure was optimized (crest-lab/crest#412)
    print("> 0 of 12 structures successfully optimized (0.0% success)")
Path("crest_conformers.xyz").write_text("\n".join(l for fr in frames for l in fr) + "\n")
Path("crest_best.xyz").write_text("\n".join(frames[0]) + "\n")
print(" CREST terminated normally.")
'''


@pytest.fixture
def fake_crest(tmp_path, monkeypatch):
    exe = _fake_program(tmp_path / "crestbin", "crest", FAKE_CREST)
    xtb = _fake_program(tmp_path / "xtbbin", "xtb", 'print("      * xtb version 6.7.1 (fake)")\n')
    log = tmp_path / "crest_calls.jsonl"
    monkeypatch.setenv("QUANTABENCH_CREST", str(exe))
    monkeypatch.setenv("QUANTABENCH_XTB", str(xtb))
    monkeypatch.setenv("FAKE_CREST_LOG", str(log))
    return {"calls": lambda: [json.loads(x) for x in log.read_text().splitlines()], "xtb_dir": str(xtb.parent)}


def _ethanol():
    return Chem.AddHs(Chem.MolFromSmiles("CCO"))


# ----------------------------------------------------------------------------- command line
def test_command_line_flags():
    prog = external.ProgramInfo("crest", "/opt/crest/bin/crest", "PATH", "3.0.2")
    cfg = BuildConfig(crest_method="gfn2//gfnff", crest_quick=True, energy_window_kcal=4.5)
    cmd = crest.command(prog, cfg, threads=3, solvent="dichloromethane", charge=-1, multiplicity=2)
    assert cmd == ["/opt/crest/bin/crest", "input.xyz", "--gfn2//gfnff", "--legacy", "--quick", "--ewin", "4.5",
                   "--alpb", "ch2cl2", "--chrg", "-1", "--uhf", "1", "-T", "3"]
    cmd = crest.command(prog, BuildConfig(crest_method="gfnff", crest_quick=False), threads=1)
    assert cmd == ["/opt/crest/bin/crest", "input.xyz", "--gfnff", "--ewin", "6", "--chrg", "0", "--uhf", "0",
                   "-T", "1"]
    assert crest.command(prog, BuildConfig(crest_method="GFN1"))[2] == "--gfn1"
    with pytest.raises(ValueError):
        crest.command(prog, BuildConfig(crest_method="b3lyp"))


def test_available_without_crest(monkeypatch):
    monkeypatch.setattr(external, "_locate", lambda spec: None)
    ok, why = crest.available()
    assert not ok and "WSL" in why


def test_parse_real_crest_conformers():
    # crest_conformers.xyz of CREST 3.0.2, ethanol, --gfnff --quick (first two lines of each frame + 2 atoms)
    text = textwrap.dedent("""\
      3
             -1.22025980
     C         -1.2718551713        0.2488151456       -0.0151885219
     C          0.0090246320       -0.5584257801        0.0342336784
     O          1.1697400910        0.2385377840       -0.0588862654
      3
             -1.21907128
     C         -1.2812581867        0.2790372746       -0.0018078399
     C         -0.0093981518       -0.5417094025        0.0489977629
     O          1.1450792923        0.2698951219       -0.0453291406
    """)
    frames = crest.parse_conformers(text)
    assert [f[2] for f in frames] == [-1.22025980, -1.21907128]
    assert frames[1][0] == ["C", "C", "O"] and frames[0][1][2, 0] == pytest.approx(1.169740091)


def test_distance_bonds_ignore_counter_ions():
    mol = Chem.AddHs(Chem.MolFromSmiles("[K+].CC(=O)[O-]"))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    pos = mol.GetConformer().GetPositions()
    k = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "K")
    o = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "O" and a.GetFormalCharge() == -1)
    pos[k] = pos[o] + [2.6, 0, 0]          # K+ next to the carboxylate: "bonded" by distance
    radii = crest._radii(mol)
    bonded = {a.GetIdx() for a in mol.GetAtoms() if a.GetDegree() > 0}
    assert k not in bonded
    assert crest.distance_bonds(pos, radii, bonded) == crest.mol_bonds(mol, bonded)
    assert (min(k, o), max(k, o)) in crest.distance_bonds(pos, radii)


# ----------------------------------------------------------------------------- fake CREST
def test_generate_with_fake_crest(fake_crest, _scratch):
    messages = []
    cfg = BuildConfig(crest_method="gfnff", crest_quick=True, energy_window_kcal=6.0, max_conformers=10)
    cs = crest.generate(_ethanol(), cfg, threads=2, solvent="water", charge=0, multiplicity=1, log=messages.append)
    assert cs.forcefield == "CREST/gfnff" and cs.n_embedded == 3
    assert len(cs.conf_ids) == 2 and cs.energies_kcal == pytest.approx([-11.31 * KCAL, -11.30 * KCAL])
    assert cs.mol.GetNumConformers() == 2 and cs.mol.GetNumAtoms() == 9
    p_low = cs.mol.GetConformer(cs.conf_ids[0]).GetPositions()
    p_high = cs.mol.GetConformer(cs.conf_ids[1]).GetPositions()
    assert np.allclose(p_low - p_high, [0.3, 0, 0], atol=1e-8)      # the shifted copy is the lowest kept one
    assert [a.GetSymbol() for a in cs.mol.GetAtoms()][:3] == ["C", "C", "O"]
    assert any("dropped 1 of 3" in m for m in messages)
    (call,) = fake_crest["calls"]()
    assert call["argv"] == ["input.xyz", "--gfnff", "--quick", "--ewin", "6", "--alpb", "water", "--chrg", "0",
                            "--uhf", "0", "-T", "2"]
    assert call["path"].split(os.pathsep)[0] == fake_crest["xtb_dir"]     # the found xtb comes first on PATH
    assert call["threads"] == "2"
    assert not any(_scratch.iterdir())                                  # job folder removed


def test_generate_caps_conformers(fake_crest):
    cs = crest.generate(_ethanol(), BuildConfig(crest_method="gfn2", max_conformers=1), threads=1)
    assert len(cs.conf_ids) == 1 and cs.energies_kcal == pytest.approx([-11.31 * KCAL])


def test_generate_rejects_reordered_atoms(fake_crest, monkeypatch):
    monkeypatch.setenv("FAKE_CREST_MODE", "badorder")
    with pytest.raises(external.ExternalProgramError, match="different atom order"):
        crest.generate(_ethanol(), BuildConfig(crest_method="gfnff"), threads=1)


def test_generate_fails_when_every_conformer_changed_bonds(fake_crest, monkeypatch):
    monkeypatch.setenv("FAKE_CREST_MODE", "allbad")
    with pytest.raises(external.ExternalProgramError, match="changed the bonding"):
        crest.generate(_ethanol(), BuildConfig(crest_method="gfnff"), threads=1)


def test_wsl_crest_command_and_environment(monkeypatch, tmp_path):
    """Windows: CREST as ``wsl:crest``. The command goes through ProgramInfo.command() with relative names."""
    seen = {}
    real_find = external.find_program

    def find(name, **kw):
        if name == "crest":
            return external.ProgramInfo("crest", "wsl:crest", "settings", "3.0.2")
        return real_find(name, **kw)

    def run(cmd, cwd, **kw):
        seen["cmd"], seen["env"], seen["cwd"] = list(cmd), kw["env"], Path(cwd)
        lines = (Path(cwd) / "input.xyz").read_text().splitlines()
        (Path(cwd) / "crest_conformers.xyz").write_text("\n".join([lines[0], "  -11.5"] + lines[2:]) + "\n")
        (Path(cwd) / "crest.log").write_text("CREST terminated normally.\n")
        return external.ProgramRun(0, 0.1, Path(cwd) / "crest.log")

    monkeypatch.setattr(external, "find_program", find)
    monkeypatch.setattr(external, "run_program", run)
    cfg = BuildConfig(crest_method="gfn2//gfnff", crest_quick=False, energy_window_kcal=3.0)
    cs = crest.generate(_ethanol(), cfg, threads=4, solvent="water", charge=0)
    assert seen["cmd"] == ["wsl.exe", "-e", "sh", "-c", external.WSL_WRAPPER, "crest", "input.xyz", "--gfn2//gfnff", "--legacy", "--ewin", "3",
                           "--alpb", "water", "--chrg", "0", "--uhf", "0", "-T", "4"]
    assert not any(Path(a).is_absolute() or str(seen["cwd"]) in a for a in seen["cmd"][3:])   # relative names
    env = seen["env"]
    assert "OMP_NUM_THREADS/u" in env["WSLENV"] and "PATH/u" not in env["WSLENV"]
    assert env["PATH"] == os.environ["PATH"] and env["OMP_NUM_THREADS"] == "4"
    assert cs.forcefield == "CREST/gfn2//gfnff" and cs.energies_kcal == pytest.approx([-11.5 * KCAL])


def test_legacy_method_needs_xtb(fake_crest, monkeypatch):
    real_find = external.find_program
    monkeypatch.setattr(external, "find_program", lambda name, **kw: None if name == "xtb" else real_find(name, **kw))
    with pytest.raises(external.ExternalProgramError, match="legacy mode"):
        crest.generate(_ethanol(), BuildConfig(crest_method="gfn2//gfnff"), threads=1)


def test_pipeline_uses_crest_conformers(fake_crest):
    from quantabench.chem.inputs import from_smiles
    from quantabench.chem.pipeline import run_molecule

    cfg = BuildConfig(method="mmff", conformer_search="crest", crest_method="gfnff", max_conformers=5,
                      write_files=False, xtb_properties=False, threads=1)
    res = run_molecule(from_smiles("CCO", id="ethanol"), cfg)
    assert res.status == "ok", res.error
    assert res.conformer_search == "crest" and res.forcefield == "CREST/gfnff" and res.n_embedded == 3


def test_failure_reason_finds_the_crest_line():
    msg = ("/opt/crest input.xyz --gfn2 exited with code 1\n> Setting up backup calculator ... done.\n"
           " Initial Geometry Optimization\n  Initial geometry optimization failed!\n"
           "  Please check your input and, if present, crestopt.log.\nERROR STOP\n"
           "(job files kept in /tmp/QuantaBench/crest_ab12)")
    assert crest.failure_reason(msg) == ("Initial geometry optimization failed! "
                                         "(exit code 1; job files in /tmp/QuantaBench/crest_ab12)")
    rt = "x exited with code 2\nAt line 187 of file ancopt.f90\nFortran runtime error: Missing comma between descriptors"
    assert crest.failure_reason(rt) == "Fortran runtime error: Missing comma between descriptors (exit code 2)"
    assert crest.failure_reason("crest not found. Install it\nmore detail") == "crest not found. Install it"
    assert crest.failure_reason("") == "no message"


def test_optimizer_problem():
    assert "0 of 826" in crest.optimizer_problem("> 0 of 826 structures successfully optimized (0.0% success)")
    assert crest.optimizer_problem("> 204 of 204 structures successfully optimized (100.0% success)\n"
                                   "> 1 of 1 structures successfully optimized") == ""
    assert crest.optimizer_problem("> 0 of 1 structures successfully optimized") == ""   # one may fail
    assert crest.optimizer_problem("") == ""


def _crest_cfg(**kw):
    base = {"method": "mmff", "conformer_search": "crest", "crest_method": "gfnff", "max_conformers": 5,
            "write_files": False, "xtb_properties": False, "threads": 1}
    return BuildConfig(**{**base, **kw})


@pytest.mark.parametrize("mode", ["macfail", "zeroopt"])
def test_crest_failure_is_flagged(fake_crest, monkeypatch, mode):
    """CREST fails (or optimizes nothing): RDKit's conformers, flagged in the notes, the log and summary.csv."""
    from quantabench.chem.inputs import from_smiles
    from quantabench.chem.pipeline import run_molecule

    monkeypatch.setenv("FAKE_CREST_MODE", mode)
    lines = []
    res = run_molecule(from_smiles("CCO", id="ethanol"), _crest_cfg(), log=lines.append)
    assert res.status == "ok", res.error
    assert res.conformer_search == "rdkit" and res.forcefield == "MMFF94s"
    want = "Initial geometry optimization failed!" if mode == "macfail" else "0 of 12 structures"
    assert res.conformer_fallback.startswith("CREST failed: ExternalProgramError: ") and want in res.conformer_fallback
    assert res.summary_row()["conformer_fallback"] == res.conformer_fallback
    assert any(n.startswith("CREST conformer search failed") and want in n and "--crest-fallback fail" in n
               for n in res.notes)
    assert any(ln.startswith("  WARNING: CREST conformer search failed") for ln in lines)


def test_crest_failure_can_fail_the_molecule(fake_crest, monkeypatch):
    from quantabench.chem.inputs import from_smiles
    from quantabench.chem.pipeline import run_molecule

    monkeypatch.setenv("FAKE_CREST_MODE", "macfail")
    res = run_molecule(from_smiles("CCO", id="ethanol"), _crest_cfg(crest_fallback="fail"))
    assert res.status == "failed" and res.conformer_fallback == ""
    assert res.error.startswith("ConformerSearchError: CREST conformer search failed")
    assert "Initial geometry optimization failed!" in res.error
    res = run_molecule(from_smiles("CCO", id="ethanol"), _crest_cfg(crest_fallback="nope"))
    assert res.status == "failed" and "crest_fallback must be one of rdkit, fail" in res.error


def test_batch_counts_crest_fallbacks(fake_crest, monkeypatch, tmp_path):
    """The end of a batch and the run folder's status count the molecules CREST failed on."""
    from quantabench.chem import jobs
    from quantabench.chem.inputs import from_smiles

    monkeypatch.setenv("FAKE_CREST_MODE", "macfail")
    lines = []
    cfg = _crest_cfg(write_files=True, formats=("xyz",))
    results, run_dir = jobs.run_resumable([from_smiles("CCO", id="ethanol"), from_smiles("CCC", id="propane")],
                                          cfg, tmp_path / "run", log=lines.append)
    assert [r.status for r in results] == ["ok", "ok"]
    assert any(ln.startswith("WARNING: CREST failed for 2 of 2 molecule(s)") for ln in lines), lines
    assert any(ln.startswith("  ethanol: CREST failed: ExternalProgramError: Initial geometry") for ln in lines)
    st = jobs.open_run(run_dir)
    assert [i for i, _ in st.fallbacks] == ["ethanol", "propane"]
    assert "2 ok with RDKit conformers because CREST failed" in st.summary()
    summary = (run_dir / "summary.csv").read_text(encoding="utf-8")
    assert "conformer_fallback" in summary.splitlines()[0] and "Initial geometry optimization failed!" in summary
    monkeypatch.delenv("FAKE_CREST_MODE")                  # a working CREST: nothing to report
    results, run_dir = jobs.run_resumable([from_smiles("CCO", id="ethanol")], cfg, tmp_path / "run2",
                                          log=lines.append)
    assert jobs.open_run(run_dir).fallbacks == [] and "CREST failed" not in jobs.open_run(run_dir).summary()


# ----------------------------------------------------------------------------- the real program
@pytest.mark.external
def test_real_crest_gfnff_ethanol(_scratch):
    info = external.find_program("crest")
    if info is None:
        pytest.skip("CREST not installed")
    messages = []
    cs = crest.generate(_ethanol(), BuildConfig(crest_method="gfnff", crest_quick=True, energy_window_kcal=6.0),
                        threads=1, log=messages.append)
    # CREST 3.0.2: 2 conformers (anti and gauche OH), 0.74 kcal/mol apart, ~2 s on one core
    assert cs.forcefield == "CREST/gfnff" and 1 <= len(cs.conf_ids) <= 4 and cs.n_embedded >= len(cs.conf_ids)
    assert cs.energies_kcal == sorted(cs.energies_kcal)
    assert cs.energies_kcal[0] / KCAL == pytest.approx(-1.2203, abs=2e-3)       # GFN-FF energy in Eh
    assert [a.GetSymbol() for a in cs.mol.GetAtoms()] == [a.GetSymbol() for a in _ethanol().GetAtoms()]
    assert not any(_scratch.iterdir())
