import pytest
from conftest import PFOA

from quantabench import cli


def test_doctor(capsys):
    assert cli.main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "Engines" in out
    assert "Sensors" in out and "CPU utilization" in out and "CPU temperature" in out     # MON-1


def test_build_and_slab(tmp_path, capsys):
    assert cli.main(["build", f"{PFOA} PFOA", "--method", "mmff", "--max-confs", "3", "--out", str(tmp_path)]) == 0
    out = tmp_path / "system.data"
    assert cli.main(["slab", "--nx", "8", "--ny", "5", "--layers", "2", "--molecule", "CCO", "--method", "mmff",
                     "-o", str(out)]) == 0
    assert out.exists() and "atoms" in out.read_text()


def test_format_list_matches_export():
    from quantabench.chem.export import FORMATS

    assert cli.export_formats() == list(FORMATS)


def test_batch_build_is_resumable_and_status(tmp_path, capsys):
    run_dir = tmp_path / "run"
    args = ["build", "CCO eth", "OCC(F)(F)F tfe", "--method", "mmff", "--max-confs", "2", "--run-dir", str(run_dir)]
    assert cli.main(args) == 0
    assert (run_dir / "jobs.sqlite").exists() and (run_dir / "summary.csv").exists()
    capsys.readouterr()
    assert cli.main(["build", "--resume", str(run_dir)]) == 0
    assert "2 of 2 molecule(s) already finished" in capsys.readouterr().out
    assert cli.main(["status", str(run_dir)]) == 0
    assert "2/2 done" in capsys.readouterr().out
    # a different method in the same folder is refused
    assert cli.main(["build", "CCO eth", "--method", "gfn1-xtb", "--run-dir", str(run_dir)]) == 2


def test_freq_command_mmff(tmp_path, capsys):
    assert cli.main(["freq", "OC(=O)C(F)(F)F TFA", "--method", "mmff", "--max-confs", "2", "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "ZPE" in out and "15 modes" in out
    assert list(tmp_path.rglob("TFA_ir.csv")) and list(tmp_path.rglob("TFA_modes.csv"))


def test_freq_as_is(tmp_path, capsys):
    from ase.build import molecule
    from ase.io import write

    write(tmp_path / "water.xyz", molecule("H2O"))
    assert cli.main(["freq", "--as-is", "--file", str(tmp_path / "water.xyz"), "--method", "mmff",
                     "--out", str(tmp_path / "o")]) == 0
    assert "3 modes" in capsys.readouterr().out
    assert (tmp_path / "o" / "water_ir.csv").exists()


def test_slab_lammps_dir_and_md_commands(tmp_path, capsys):
    out = tmp_path / "sys.extxyz"
    smd = tmp_path / "smd"
    assert cli.main(["slab", "--nx", "6", "--ny", "4", "--layers", "2", "--molecule", "OC(=O)C(F)(F)F", "--method",
                     "mmff", "-o", str(out), "--lammps-dir", str(smd)]) == 0
    assert (smd / "in.smd").exists() and (smd / "system.data").exists()
    smd2 = tmp_path / "smd2"
    assert cli.main(["md", "smd", "--system", str(out), "-o", str(smd2), "--gpu"]) == 0
    assert "package gpu" in (smd2 / "in.smd").read_text()
    capsys.readouterr()
    assert cli.main(["md", "parse", str(smd2)]) == 0
    assert "no log.lammps" in capsys.readouterr().out


def test_no_freq_overrides_settings_saved_by_the_gui(tmp_path, monkeypatch):
    from quantabench import settings as sm

    s = sm.load()
    s.build.frequencies = True                    # ticked in the GUI, saved to settings.json
    sm.save(s)
    args = cli.build_parser().parse_args(["build", "CCO", "--method", "mmff"])
    assert cli._config(args).frequencies
    args = cli.build_parser().parse_args(["build", "CCO", "--method", "mmff", "--no-freq"])
    cfg = cli._config(args)
    assert not cfg.frequencies and "spectrum" not in cfg.formats
    assert not cli._config(cli.build_parser().parse_args(["build", "CCO", "--no-protonate"])).protonate
    assert cli._config(cli.build_parser().parse_args(["build", "CCO", "--protonate"])).protonate


def test_sheet_index_from_the_command_line(tmp_path, capsys):
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    wb.active.title = "Info"
    wb.active.append(["note"])
    main = wb.create_sheet("Main")
    main.append(["ID", "SMILES"])
    main.append(["eth", "CCO"])
    path = tmp_path / "two.xlsx"
    wb.save(path)
    args = cli.build_parser().parse_args(["build", "--file", str(path), "--sheet", "1"])
    assert [m.id for m in cli._read_inputs(args)] == ["eth"]


def test_freq_as_is_errors_are_reported(tmp_path, capsys):
    assert cli.main(["freq", "--as-is", "--file", str(tmp_path / "missing.xyz"), "--method", "mmff",
                     "--out", str(tmp_path)]) == 1
    assert "missing: FAILED" in capsys.readouterr().err


def test_merge_and_md_parse_fail_on_missing_folders(tmp_path, capsys):
    assert cli.main(["merge", str(tmp_path / "out"), str(tmp_path / "nope1"), str(tmp_path / "nope2")]) == 1
    assert "no such folder" in capsys.readouterr().err
    assert cli.main(["md", "parse", str(tmp_path / "nope")]) == 2


def test_console_streams_never_fail_on_unencodable_characters():
    """A cp1252 console (Windows, redirected) printing σ or β: replaced, not UnicodeEncodeError."""
    import io

    from quantabench import cli

    raw = io.BytesIO()
    s = io.TextIOWrapper(raw, encoding="cp1252", errors="strict")
    cli.safe_console_streams([s])
    print("σ_CMC β-Tocopherol", file=s)
    s.flush()
    assert raw.getvalue().decode("cp1252").startswith("?_CMC ?-Tocopherol")
    u = io.TextIOWrapper(io.BytesIO(), encoding="utf-8", errors="strict")
    cli.safe_console_streams([u])
    assert u.errors == "strict"


def test_settings_engine_path_is_per_machine(capsys, monkeypatch):
    from quantabench import settings as sm

    monkeypatch.setattr(sm, "_this_machine", lambda: "Brandons-MacBook-Pro")
    assert cli.main(["settings", "--engine-path", "crest=~/qb/crest-macos-arm64/bin/crest"]) == 0
    assert cli.main(["settings", "--engine-path", "crest=wsl:/home/b/envs/qb-tools/bin/crest",
                     "--machine", "BTM-NOVA"]) == 0
    assert "crest on btm-nova: wsl:" in capsys.readouterr().out
    s = sm.load()
    assert s.engines.paths == {}
    assert s.engine_paths_for()["crest"] == ("~/qb/crest-macos-arm64/bin/crest", "this machine")
    assert s.engine_paths_for("btm-nova")["crest"][0] == "wsl:/home/b/envs/qb-tools/bin/crest"
    assert cli.main(["settings", "--engine-path", "crest="]) == 0          # automatic again on this machine
    assert "crest" not in sm.load().engine_paths_for()
    assert cli.main(["settings", "--engine-path", "nonsense=/x"]) == 2
    # --machine with no name is this machine, before or after --engine-path
    capsys.readouterr()
    assert cli.main(["settings", "--engine-path", "crest=/opt/crest/bin/crest", "--machine"]) == 0
    assert "crest on brandons-macbook-pro: /opt/crest/bin/crest" in capsys.readouterr().out
    assert cli.main(["settings", "--machine", "--engine-path", "xtb=/opt/xtb/bin/xtb"]) == 0
    s = sm.load()
    assert s.engine_paths_for()["crest"] == ("/opt/crest/bin/crest", "this machine")
    assert s.engine_paths_for()["xtb"] == ("/opt/xtb/bin/xtb", "this machine")
    assert s.engine_paths_for("btm-nova")["crest"][0] == "wsl:/home/b/envs/qb-tools/bin/crest"     # untouched
    with pytest.raises(SystemExit):
        cli.main(["settings", "--help"])
    assert "--machine [NAME]" in " ".join(capsys.readouterr().out.split())
