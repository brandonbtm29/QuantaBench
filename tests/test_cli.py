from conftest import PFOA

from quantabench import cli


def test_doctor(capsys):
    assert cli.main(["doctor"]) == 0
    assert "Engines" in capsys.readouterr().out


def test_build_and_slab(tmp_path, capsys):
    assert cli.main(["build", f"{PFOA} PFOA", "--method", "mmff", "--max-confs", "3", "--out", str(tmp_path)]) == 0
    out = tmp_path / "system.data"
    assert cli.main(["slab", "--nx", "8", "--ny", "5", "--layers", "2", "--molecule", "CCO", "--method", "mmff",
                     "-o", str(out)]) == 0
    assert out.exists() and "atoms" in out.read_text()
