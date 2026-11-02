"""Experimental spectrum readers (chem.spectra_io): CSV/TSV tables and JCAMP-DX.

The JCAMP snippets are written by hand. One data set (12 ordinates) is encoded in every form the JCAMP-DX
4.24 specification defines (McDonald & Wilks, Appl. Spectrosc. 1988, 42, 151): AFFN, PAC, SQZ, DIF and
DIFDUP, with the DIF Y check value (the last ordinate of a line repeated at the start of the next line, and
a final check line). The decoded expectation follows from the specification's character tables:

    SQZ  @ A-I / a-i = leading digit 0, +1..+9 / -1..-9
    DIF  % J-R / j-r = difference 0, +1..+9 / -1..-9
    DUP  S-Z, s      = the previous value or difference occurs 1..9 times in total
"""

import math

import numpy as np
import pytest

from quantabench.chem import spectra_io
from quantabench.chem.spectra_io import decode_xydata, read_spectrum

Y_RAW = [100, 102, 105, 105, 105, 105, 103, 99, 90, 90, 72, 45]
X = np.arange(1000.0, 1012.0)
HEADER = """##TITLE= hand-written test spectrum
##JCAMP-DX= 4.24
##DATA TYPE= INFRARED SPECTRUM
##XUNITS= 1/CM
##YUNITS= ABSORBANCE
##XFACTOR= 1.0
##YFACTOR= 0.01
##FIRSTX= 1000
##LASTX= 1011
##DELTAX= 1
##FIRSTY= 1.00
##NPOINTS= 12
##XYDATA= (X++(Y..Y))
"""

FORMS = {
    "AFFN": """1000 100 102 105 105
1004 105 105 103 99
1008 90 90 72 45
""",
    "PAC": """1000+100+102+105+105
1004+105+105+103+99
1008+90+90+72+45
""",
    "SQZ": """1000A00A02A05A05
1004A05A05A03I9
1008I0I0G2D5
""",
    # absolute 100, then differences +2 +3 0 | check 105, 0 0 -2 -4 | check 99, -9 0 -18 -27 | check 45
    "DIF": """1000A00KL%
1003A05%%km
1007I9r%j8k7
1011D5
""",
    "DIFDUP": """1000A00KL%
1003A05%Tkm
1007I9r%j8k7
1011D5
""",
}


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="latin-1")
    return p


@pytest.mark.parametrize("form", list(FORMS))
def test_jcamp_xydata_every_compression(tmp_path, form):
    p = _write(tmp_path, f"{form}.jdx", HEADER + FORMS[form] + "##END=\n")
    s = read_spectrum(p)
    assert s.source_format == "jcamp" and not s.peaks
    assert s.title == "hand-written test spectrum"
    assert s.x_units == "cm-1" and s.y_units == "ABSORBANCE"
    assert np.allclose(s.x, X)
    assert np.allclose(s.y, np.array(Y_RAW) * 0.01)
    assert s.warnings == []                      # FIRSTY, NPOINTS and every Y check agree


def test_spec_style_short_examples():
    # 2 3 3 2 1 0 -1 -2 -3 at X = 1..9 in each form
    expect = [2, 3, 3, 2, 1, 0, -1, -2, -3]
    for line in ("1 2 3 3 2 1 0 -1 -2 -3", "1+2+3+3+2+1+0-1-2-3", "1BCCBA@abc", "1BJ%jjjjjj", "1BJ%jX"):
        data, warnings = decode_xydata([line])
        assert [v for d in data for v in d.values] == expect, line
        assert warnings == []
    data, _ = decode_xydata(["1A05U@T"])         # DUP of absolute values: 105 three times, 0 twice
    assert data[0].values == [105, 105, 105, 0, 0]
    data, _ = decode_xydata(["1A00S0"])           # multi-digit DUP count: S0 = 10 occurrences
    assert data[0].values == [100] * 10


def test_dif_y_check_mismatch_is_reported():
    data, warnings = decode_xydata(["1000A00KL%", "1003A07%%"])   # check 107 != 105
    assert [v for d in data for v in d.values] == [100, 102, 105, 105, 107, 107]
    assert len(warnings) == 1 and "Y check" in warnings[0]
    assert data[1].checked


def test_affn_with_exponents_and_missing_values():
    data, _ = decode_xydata(["4000.0 1.5E-3 2.0e-3 -1.25E+00"])
    assert np.allclose(data[0].values, [1.5e-3, 2.0e-3, -1.25])
    data, _ = decode_xydata(["10 A00?A02"])       # JCAMP 5 '?' marks a missing ordinate (forces ASDF)
    assert data[0].values[0] == 100 and math.isnan(data[0].values[1]) and data[0].values[2] == 102


def test_abscissas_from_line_x_when_npoints_disagrees(tmp_path):
    """Without a usable FIRSTX/LASTX/NPOINTS, abscissas come from each line's X and DELTAX; a line that
    starts with a DIF check value has its X on that check point."""
    head = HEADER.replace("##NPOINTS= 12\n", "##NPOINTS= 13\n")
    s = read_spectrum(_write(tmp_path, "dif.jdx", head + FORMS["DIF"] + "##END=\n"))
    assert np.allclose(s.x, X) and np.allclose(s.y, np.array(Y_RAW) * 0.01)
    assert any("NPOINTS" in w for w in s.warnings)


def test_labels_compare_without_case_spaces_or_underscores(tmp_path):
    text = ("##title= odd labels\n##X_UNITS= 1/CM\n##Y Units= ABSORBANCE\n##x-factor=1\n##YFACTOR = 1\n"
            "##First X= 1000\n##LAST_X= 1003\n##N POINTS= 4\n##XY DATA= (X++(Y..Y)) $$ a comment\n"
            "1000 1 2 3 4 $$ trailing comment\n$$ a comment line\n##END=\n")
    s = read_spectrum(_write(tmp_path, "labels.dx", text))
    assert np.allclose(s.x, [1000, 1001, 1002, 1003]) and np.allclose(s.y, [1, 2, 3, 4])


@pytest.mark.parametrize("form,body", [
    ("(XY..XY)", "1595.0, 0.80; 3657.0, 0.10\n3756.0, 0.55\n"),
    ("(XYXY)", "1595.0 0.80 3657.0 0.10 3756.0 0.55\n"),
    ("(XYW..XYW)", "1595.0, 0.80, 12; 3657.0, 0.10, 9\n3756.0, 0.55, 10\n"),
])
def test_jcamp_peak_tables(tmp_path, form, body):
    text = (f"##TITLE= peaks\n##JCAMP-DX= 5.01\n##DATA TYPE= INFRARED PEAK TABLE\n##XUNITS= 1/CM\n"
            f"##YUNITS= ABSORBANCE\n##NPOINTS= 3\n##PEAK TABLE= {form}\n{body}##END=\n")
    s = read_spectrum(_write(tmp_path, "peaks.jdx", text))
    assert s.peaks
    assert np.allclose(s.x, [1595.0, 3657.0, 3756.0]) and np.allclose(s.y, [0.80, 0.10, 0.55])


def test_jcamp_xypoints_are_a_trace(tmp_path):
    text = "##TITLE= pts\n##XUNITS= 1/CM\n##YUNITS= ABSORBANCE\n##XYPOINTS= (XY..XY)\n3000,0.1 2000,0.2 1000,0.3\n##END=\n"
    s = read_spectrum(_write(tmp_path, "pts.jdx", text))
    assert not s.peaks and np.allclose(s.x, [1000, 2000, 3000]) and np.allclose(s.y, [0.3, 0.2, 0.1])


def test_transmittance_to_absorbance_and_micrometers(tmp_path):
    text = ("##TITLE= T in micrometers\n##XUNITS= MICROMETERS\n##YUNITS= TRANSMITTANCE\n##XFACTOR= 0.01\n"
            "##YFACTOR= 0.001\n##FIRSTX= 2.5\n##LASTX= 10.0\n##NPOINTS= 4\n##XYDATA= (X++(Y..Y))\n"
            "250 1000 100 0 500\n##END=\n")
    s = read_spectrum(_write(tmp_path, "t.jdx", text))
    # x: 2.5, 5, 7.5, 10 µm -> 4000, 2000, 1333.3, 1000 cm-1, sorted ascending
    assert np.allclose(s.x, [1000.0, 1e4 / 7.5, 2000.0, 4000.0])
    t = np.array([0.5, 0.0, 0.1, 1.0])          # in ascending-x order
    assert np.allclose(s.y, -np.log10(np.clip(t, spectra_io.T_FLOOR, None)))
    assert np.isfinite(s.y).all() and s.y_units == "ABSORBANCE"
    assert any("clipped" in w for w in s.warnings)


def test_percent_transmittance(tmp_path):
    text = ("##TITLE= %T\n##XUNITS= 1/CM\n##YUNITS= %T\n##FIRSTX= 4000\n##LASTX= 3997\n##NPOINTS= 4\n"
            "##XYDATA= (X++(Y..Y))\n4000 100 50 10 1\n##END=\n")
    s = read_spectrum(_write(tmp_path, "pt.jdx", text))
    assert np.allclose(s.x, [3997, 3998, 3999, 4000])
    assert np.allclose(s.y, [2.0, 1.0, np.log10(2), 0.0])


def test_multiblock_takes_first_spectrum(tmp_path):
    text = ("##TITLE= link block\n##JCAMP-DX= 5.01\n##DATA TYPE= LINK\n##BLOCKS= 2\n"
            "##TITLE= first\n##BLOCK_ID= 1\n##XUNITS= 1/CM\n##YUNITS= ABSORBANCE\n##FIRSTX= 1\n##LASTX= 3\n"
            "##NPOINTS= 3\n##XYDATA= (X++(Y..Y))\n1 1 2 3\n##END=\n"
            "##TITLE= second\n##BLOCK_ID= 2\n##XUNITS= 1/CM\n##YUNITS= ABSORBANCE\n##FIRSTX= 1\n##LASTX= 3\n"
            "##NPOINTS= 3\n##XYDATA= (X++(Y..Y))\n1 7 8 9\n##END=\n##END=\n")
    s = read_spectrum(_write(tmp_path, "multi.jdx", text))
    assert s.title == "first" and np.allclose(s.y, [1, 2, 3])


def test_jcamp_errors(tmp_path):
    with pytest.raises(ValueError, match="no ##XYDATA"):
        read_spectrum(_write(tmp_path, "empty.jdx", "##TITLE= nothing\n##END=\n"))
    with pytest.raises(ValueError, match="NTUPLES"):
        read_spectrum(_write(tmp_path, "nt.jdx", "##TITLE= nt\n##NTUPLES= INFRARED\n##END NTUPLES=\n##END=\n"))


# ------------------------------------------------------------------------------------- tables
@pytest.mark.parametrize("name,text", [
    ("comma.csv", "wavenumber_cm-1,absorbance\n1000,0.1\n2000,0.5\n3000,0.2\n"),
    ("tab.tsv", "x\ty\n1000\t0.1\n2000\t0.5\n3000\t0.2\n"),
    ("space.txt", "# exported by some instrument\n3000 0.2\n2000   0.5\n1000 0.1\n"),
    ("semicolon.csv", "Wellenzahl;Absorption\n1000,0;0,1\n2000,0;0,5\n3000,0;0,2\n"),
    ("tab_decimal_comma.txt", "1000,0\t0,1\n2000,0\t0,5\n3000,0\t0,2\n"),
    ("three_cols.csv", "x,y,z\n1000,0.1,9\n2000,0.5,9\n3000,0.2,9\n"),
])
def test_tables(tmp_path, name, text):
    s = read_spectrum(_write(tmp_path, name, text))
    assert s.source_format == "csv"
    assert np.allclose(s.x, [1000, 2000, 3000]) and np.allclose(s.y, [0.1, 0.5, 0.2])


def test_table_units_from_header(tmp_path):
    s = read_spectrum(_write(tmp_path, "t.csv", "Wavelength (um),%T\n2.5,50\n5.0,10\n"))
    assert np.allclose(s.x, [2000, 4000]) and np.allclose(s.y, [1.0, np.log10(2)])
    assert s.y_units == "ABSORBANCE"


def test_table_round_trip_of_panel_export(tmp_path):
    p = spectra_io.write_table(tmp_path / "out.csv", {"wavenumber_cm-1": np.array([400.0, 401.0]),
                                                      "intensity_normalized": np.array([0.5, 1.0])})
    assert p.read_text().splitlines()[0] == "wavenumber_cm-1,intensity_normalized"
    s = read_spectrum(p)
    assert np.allclose(s.x, [400, 401]) and np.allclose(s.y, [0.5, 1.0])


def test_table_errors(tmp_path):
    with pytest.raises(ValueError):
        read_spectrum(_write(tmp_path, "words.csv", "a,b\nc,d\n"))
    with pytest.raises(ValueError):
        read_spectrum(_write(tmp_path, "blank.csv", "\n\n"))
    with pytest.raises(FileNotFoundError):
        read_spectrum(tmp_path / "missing.csv")


def test_dif_abscissas_without_firstx_lastx_deltax(tmp_path):
    """Line 2 starts with the Y-check value, so its abscissa is that of line 1's last point."""
    from quantabench.chem import spectra_io

    p = tmp_path / "dif.jdx"
    p.write_text("##TITLE=t\n##JCAMP-DX=5.01\n##XUNITS=1/CM\n##YUNITS=ABSORBANCE\n##XYDATA=(X++(Y..Y))\n"
                 "0 10JK\n2 13LMN\n##END=\n")
    sp = spectra_io.read_spectrum(p)
    assert list(sp.x) == [0, 1, 2, 3, 4, 5] and list(sp.y) == [10, 11, 13, 16, 20, 25]


def test_bare_t_header_is_transmittance():
    from quantabench.chem import spectra_io

    assert spectra_io._units_from_header(["Wavelength (um)", "T"]) == ("MICROMETERS", "TRANSMITTANCE")
