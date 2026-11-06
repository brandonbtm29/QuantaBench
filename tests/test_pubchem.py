"""PubChem client, PUG View parsing, structure validation and `pubchem enrich`, offline.

The responses in tests/data/pubchem were captured from PubChem on 2026-11-05 (index.json: URL, HTTP status and
X-Throttling-Control header of each). FakePubChem serves them by URL, the way PubChem would, and builds the
batched answers (properties, synonyms, registry numbers for any list of CIDs) from them. No test uses the network.
"""

from __future__ import annotations

import csv
import json
import re
import urllib.parse
from pathlib import Path

import pytest
from conftest import K_PFOS, PFOA

from quantabench import __version__, cli
from quantabench.chem import pubchem as pc
from quantabench.chem import pubchem_enrich as pe

DATA = Path(__file__).parent / "data" / "pubchem"
INDEX = json.loads((DATA / "index.json").read_text(encoding="utf-8"))
GREEN = "Request Count status: Green (0%), Request Time status: Green (0%), Service status: Green (2%)"
NOT_FOUND = json.dumps({"Fault": {"Code": "PUGREST.NotFound", "Message": "No CID found"}}).encode()
AMBIGUOUS_CAS = "1234-56-6"                # valid check digit; served with two CIDs below


def fixture(name: str) -> dict:
    return json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))


def _key(url: str) -> tuple:
    parts = urllib.parse.urlsplit(url)
    return parts.path, tuple(sorted(urllib.parse.parse_qsl(parts.query)))


class FakePubChem:
    """A transport (``PubChemClient(transport=...)``) that answers like PubChem from the captured responses."""

    def __init__(self, throttle: str | None = None):
        self.calls: list[tuple[str, str, bytes | None, dict]] = []
        self.throttle = throttle
        self.fail_next: list[int] = []        # statuses to return before answering (503, 504, ...)
        self.routes = {_key(m["url"]): (m["status"], (DATA / f"{n}.json").read_bytes(), m["throttling"])
                       for n, m in INDEX.items()}
        self.routes[_key(f"{pc.REST}/compound/name/{AMBIGUOUS_CAS}/cids/JSON")] = (
            200, json.dumps({"IdentifierList": {"CID": [702, 8058]}}).encode(), GREEN)
        self.props = {p["CID"]: p for p in fixture("properties_batch")["PropertyTable"]["Properties"]}
        info = fixture("xref_rn_pfoa")["InformationList"]["Information"][0]
        self.rns = {info["CID"]: info["RN"]}
        info = fixture("synonyms_pfoa")["InformationList"]["Information"][0]
        self.syns = {info["CID"]: info["Synonym"]}
        self.views = {(9554, "Experimental Properties"): "view_experimental_pfoa",
                      (9554, "Surface Tension"): "view_surface_tension_pfoa",
                      (8058, "Surface Tension"): "view_surface_tension_hexane",
                      (702, "Surface Tension"): "view_surface_tension_ethanol"}

    def _answer(self, url: str) -> tuple[int, bytes, str]:
        hit = self.routes.get(_key(url))
        if hit:
            return hit
        path, query = _key(url)
        m = re.search(r"/compound/cid/([\d,]+)/(property/([^/]+)|synonyms|xrefs/RN)/JSON$", path)
        if m:
            cids = [int(c) for c in m.group(1).split(",")]
            if m.group(3):
                names = m.group(3).split(",")
                rows = [{"CID": c, **{k: v for k, v in self.props[c].items() if k in names}}
                        for c in cids if c in self.props]
                return (200, json.dumps({"PropertyTable": {"Properties": rows}}).encode(), GREEN) if rows \
                    else (404, NOT_FOUND, GREEN)
            table, key = (self.syns, "Synonym") if m.group(2) == "synonyms" else (self.rns, "RN")
            info = [{"CID": c, key: table[c]} for c in cids if c in table]
            return (200, json.dumps({"InformationList": {"Information": info}}).encode(), GREEN) if info \
                else (404, NOT_FOUND, GREEN)
        m = re.search(r"/pug_view/data/compound/(\d+)/JSON$", path)
        if m:
            name = self.views.get((int(m.group(1)), dict(query).get("heading", "")))
            if name:
                return 200, (DATA / f"{name}.json").read_bytes(), GREEN
            return 404, json.dumps({"Fault": {"Code": "PUGVIEW.NotFound", "Message": "No data"}}).encode(), GREEN
        return 404, NOT_FOUND, GREEN

    def __call__(self, method, url, data, headers, timeout):
        self.calls.append((method, url, data, headers))
        if self.fail_next:
            return pc.HttpResponse(self.fail_next.pop(0), b"", {"X-Throttling-Control": GREEN})
        if data:                                  # a POST: the form fields become the query
            url = url + "?" + data.decode()
        status, body, throttle = self._answer(url)
        return pc.HttpResponse(status, body, {"X-Throttling-Control": self.throttle or throttle})


class FakeClock:
    def __init__(self):
        self.t = 1000.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.slept.append(round(s, 6))
        self.t += s


def make_client(tmp_path, fake=None, clock=None, **kw) -> pc.PubChemClient:
    clock = clock or FakeClock()
    return pc.PubChemClient(cache=tmp_path / "cache", transport=fake or FakePubChem(), clock=clock,
                            sleep=clock.sleep, **kw)


# ------------------------------------------------------------------------------------------------ CAS numbers
def test_cas_check_digit():
    for cas in ("335-67-1", "1763-23-1", "7732-18-5", "50-00-0", "64-17-5", "110-54-3", AMBIGUOUS_CAS):
        assert pc.is_cas(cas), cas
    assert pc.cas_check_digit("33567") == 1
    for bad in ("335-67-2", "1763-23-0", "206-397-9", "335671", "35-67-1x", "", None):
        assert not pc.is_cas(bad), bad            # 206-397-9 is PFOA's EC number
    assert pc.normalize_cas("CAS-0335-67-1") == "335-67-1"
    assert pc.normalize_cas("CAS RN 1763-23-1 (PFOS)") == "1763-23-1"
    assert pc.normalize_cas("335-67-2") == ""
    rns = fixture("xref_rn_pfoa")["InformationList"]["Information"][0]["RN"]
    assert "206-397-9" in rns and "206-397-9" not in pc.cas_numbers(rns)
    assert "335-67-1" in pc.cas_numbers(rns)
    syns = fixture("synonyms_pfoa")["InformationList"]["Information"][0]["Synonym"]
    assert pc.cas_numbers(syns) == ["335-67-1"]   # "CAS-335-67-1" folds in; EINECS 206-397-9 does not count


# ------------------------------------------------------------------------------------------------ identifiers
def test_identifier_resolution(tmp_path):
    fake = FakePubChem()
    client = make_client(tmp_path, fake)
    assert client.cids_by_name("335-67-1") == [9554]
    assert client.cids_by_name("1763-23-1") == [74483]
    assert client.cids_by_inchikey("SNGREZUHAYWORS-UHFFFAOYSA-N") == [9554]
    assert client.cids_by_smiles("OC(=O)C(F)(F)C(F)(F)F") == [62356]      # PFPrA
    assert client.cids_by_name("notarealchemicalnamexyzzy") == []          # the 404 fault
    ua = fake.calls[0][3]["User-Agent"]
    assert ua == f"QuantaBench/{__version__} (+https://github.com/brandonbtm29/QuantaBench)" and "@" not in ua
    assert all(c[0] == "GET" for c in fake.calls)
    client.cids_by_name("a/b name")                                         # a slash goes in a POST body
    assert fake.calls[-1][0] == "POST" and fake.calls[-1][2] == b"name=a%2Fb+name"


def test_not_found_and_other_faults(tmp_path):
    fake = FakePubChem()
    client = make_client(tmp_path, fake)
    assert client.get_json(f"{pc.REST}/compound/name/notarealchemicalnamexyzzy/cids/JSON") is None
    assert fixture("not_found_name")["Fault"]["Code"] == "PUGREST.NotFound"
    bad = f"{pc.REST}/compound/smiles/cids/JSON?smiles=C1CC"
    fake.routes[_key(bad)] = (400, json.dumps({"Fault": {"Code": "PUGREST.BadRequest", "Message": "Unable to "
                                                         "standardize the given structure"}}).encode(), GREEN)
    with pytest.raises(pc.PubChemFault, match="BadRequest"):
        client.cids_by_smiles("C1CC")
    with pytest.raises(pc.PubChemError, match="no compound"):
        pc.lookup("notarealchemicalnamexyzzy", client=client)


def test_inputs_lookup_name_uses_the_client(tmp_path, monkeypatch):
    from quantabench.chem import inputs

    fake = FakePubChem()
    fake.routes[_key(f"{pc.REST}/compound/name/PFOA/property/SMILES,ConnectivitySMILES,Title/JSON")] = (
        200, json.dumps({"PropertyTable": {"Properties": [{"CID": 9554, "SMILES": PFOA, "Title": "Perfluorooctanoic "
                                                                                                "acid"}]}}).encode(),
        GREEN)
    monkeypatch.setattr(pc, "_DEFAULT", make_client(tmp_path, fake))
    m = inputs.lookup_name("PFOA")
    assert (m.smiles, m.name, m.meta["pubchem_cid"], m.source) == (PFOA, "Perfluorooctanoic acid", 9554, "pubchem")
    with pytest.raises(ValueError, match="PubChem lookup failed"):
        inputs.lookup_name("notarealchemicalnamexyzzy")


# ------------------------------------------------------------------------------------------------ batches
def test_batch_properties_new_and_old_smiles_names(tmp_path):
    captured = fixture("properties_batch")["PropertyTable"]["Properties"]
    # PubChem answered the request for CanonicalSMILES and IsomericSMILES with the new names only
    assert all("SMILES" in p and "ConnectivitySMILES" in p for p in captured)
    assert not any("CanonicalSMILES" in p or "IsomericSMILES" in p for p in captured)
    fake = FakePubChem()
    client = make_client(tmp_path, fake)
    props = client.properties([9554, 74483, 8058, 702])
    assert len(fake.calls) == 1 and set(props) == {9554, 74483, 8058, 702}
    assert "property/Title,MolecularFormula" in fake.calls[0][1] and "CanonicalSMILES" not in fake.calls[0][1]
    pfoa = props[9554]
    assert pfoa["InChIKey"] == "SNGREZUHAYWORS-UHFFFAOYSA-N" and pfoa["MolecularWeight"] == pytest.approx(414.07)
    assert pfoa["ExactMass"] == pytest.approx(413.9737017) and pfoa["XLogP"] == 4.9 and pfoa["HBondAcceptorCount"] == 17
    assert props[702]["SMILES"] == "CCO" and props[8058]["ConnectivitySMILES"] == "CCCCCC"
    # old names (older cached answers) are read as the new ones
    old = pc._normalize_props({"CID": 1, "CanonicalSMILES": "CC(N)C(=O)O", "IsomericSMILES": "C[C@@H](C(=O)O)N"})
    assert old == {"CID": 1, "SMILES": "C[C@@H](C(=O)O)N", "ConnectivitySMILES": "CC(N)C(=O)O"}
    # 250 CIDs: three property requests; synonyms and registry numbers go 50 at a time
    fake.calls.clear()
    client.properties(range(1, 251))
    assert len(fake.calls) == 3
    syn = client.synonyms([9554, 74483])
    rn = client.registry_numbers([9554])
    assert syn[9554][0] == "PERFLUOROOCTANOIC ACID" and "335-67-1" in rn[9554] and 74483 not in syn


# ------------------------------------------------------------------------------------------------ PUG View
def _st(name: str) -> pc.ExperimentalRecord:
    recs = pc.parse_view(fixture(name))
    assert len(recs) == 1 and recs[0].heading == "Surface Tension"
    return recs[0]


def test_surface_tension_records():
    hexane, ethanol, pfoa = (_st(f"view_surface_tension_{n}") for n in ("hexane", "ethanol", "pfoa"))
    assert (hexane.cid, hexane.value_std, hexane.unit, hexane.temperature_c) == (8058, 17.89, "mN/m", 25.0)
    assert (ethanol.cid, ethanol.value_std, ethanol.temperature_c) == (702, 21.97, 25.0)
    for r in (hexane, ethanol):
        assert (r.condition_class, r.condition_basis, r.needs_review) == ("pure_liquid", "reference", False)
        assert "CRC Handbook" in r.reference and r.source_name == "Hazardous Substances Data Bank (HSDB)"
    assert (pfoa.cid, pfoa.value, pfoa.unit, pfoa.value_std, pfoa.unit_std) == (9554, 15.2, "dyn/cm", 15.2, "mN/m")
    assert pfoa.temperature_c is None and pfoa.raw == "15.2 dynes/cm"
    assert (pfoa.source_name, pfoa.source_id, pfoa.doi) == ("Hazardous Substances Data Bank (HSDB)", "7137",
                                                             "10.1021/es991359u")
    assert pfoa.url == "https://pubchem.ncbi.nlm.nih.gov/source/hsdb/7137" and "Moody CA, Field JA" in pfoa.reference
    assert pfoa.description == "PEER REVIEWED"
    # alone (no melting point): unknown, because the reference is about surfactants in fire-fighting foams
    assert pfoa.condition_class == "unknown" and pfoa.needs_review and "CMC" in pfoa.condition_note


def test_experimental_properties_tree():
    recs = pc.parse_view(fixture("view_experimental_pfoa"))
    by = {}
    for r in recs:
        by.setdefault(r.heading, []).append(r)
    assert {"Surface Tension", "Density", "Boiling Point", "Melting Point", "Vapor Pressure", "Solubility", "LogP",
            "Dissociation Constants", "Physical Description"} <= set(by)
    assert "Collision Cross Section" not in by and "Kovats Retention Index" not in by
    st = by["Surface Tension"][0]
    assert st.value_std == 15.2 and st.condition_class == "unknown" and st.needs_review
    assert "solid at 25 °C (mp 53.65 °C)" in st.condition_note and "gamma_CMC" in st.condition_note
    assert [r.condition_class for r in by["Density"]] == ["solid"] * 3        # PFOA's density is the solid's
    assert by["Density"][0].value_std == 1.792 and by["Density"][0].temperature_c == 20.0
    mp = by["Melting Point"]
    assert mp[0].value_std == 54.3 and (mp[1].value_low, mp[1].value_high, mp[1].value) == (52.0, 54.0, 53.0)
    bp = by["Boiling Point"]
    assert [r.value_std for r in bp] == [192.0, 189.0, 189.0] and bp[2].pressure == "760 mmHg"
    vp = {r.raw: r for r in by["Vapor Pressure"]}
    assert vp["3.16X10-2 mm Hg at 25 °C"].value_std == pytest.approx(4.213, rel=1e-3)
    assert vp["0.15 [mm Hg] @20 °C"].temperature_c == 20.0
    sol = [r for r in by["Solubility"] if r.value_std is not None]
    assert [(r.value_std, r.temperature_c) for r in sol] == [(3300.0, 25.0), (4340.0, 24.1)]
    assert by["LogP"][0].value_std == 6.3
    pka = by["Dissociation Constants"]
    assert [r.value_std for r in pka[:2]] == [1.3, 2.8] and (pka[2].value_low, pka[2].value_high) == (-0.5, 4.2)
    assert {r.source_name for r in recs} >= {"Hazardous Substances Data Bank (HSDB)",
                                             "ILO-WHO International Chemical Safety Cards (ICSCs)"}
    assert len(pc.parse_view(fixture("view_experimental_pfoa"), headings="all")) > len(recs)


@pytest.mark.parametrize("text, heading, value, unit, std, temp", [
    ("15.2 dynes/cm", "Surface Tension", 15.2, "dyn/cm", 15.2, None),
    ("72.8 mN/m at 20 °C", "Surface Tension", 72.8, "mN/m", 72.8, 20.0),
    ("0.0218 N/m", "Surface Tension", 0.0218, "N/m", 21.8, None),
    ("27.5 dyn/cm at 68 °F", "Surface Tension", 27.5, "dyn/cm", 27.5, 20.0),
    ("20.1 mJ/m² at 298.15 K", "Surface Tension", 20.1, "mJ/m2", 20.1, 25.0),
    ("Surface tension of a 0.1% aqueous solution: 30 dyn/cm at 25 deg C", "Surface Tension", 30.0, "dyn/cm", 30.0,
     25.0),
    ("0.6606 g/cu cm at 20 °C", "Density", 0.6606, "g/cm3", 0.6606, 20.0),
    ("1.792 @ 20°C", "Density", 1.792, "", 1.792, 20.0),
    ("789 kg/m3 at 25 °C", "Density", 789.0, "kg/m3", 0.789, 25.0),
    ("0.300 cP at 25 °C", "Viscosity", 0.3, "cP", 0.3, 25.0),
    ("1.2 mPa·s at 20 °C", "Viscosity", 1.2, "mPa.s", 1.2, 20.0),
    ("-24.0 - -14.0 °C", "Melting Point", -19.0, "°C", -19.0, None),
    ("156 °F", "Boiling Point", 156.0, "°F", pytest.approx(68.89, abs=0.01), None),
    ("3.16X10-2 mm Hg at 25 °C", "Vapor Pressure", 0.0316, "mmHg", pytest.approx(4.213, rel=1e-3), 25.0),
    ("17 kPa at 20 °C", "Vapor Pressure", 17.0, "kPa", 17000.0, 20.0),
    ("log Kow = 4.81", "LogP", 4.81, "", 4.81, None),
])
def test_value_parsing(text, heading, value, unit, std, temp):
    (pv,) = pc.parse_values(text, heading)
    assert (pv.value, pv.unit, pv.value_std, pv.temperature_c) == (value, unit, std, temp)


def test_value_parsing_several_values_and_units_carried():
    vals = pc.parse_values("22.3 dyn/cm at 20 °C; 21.1 at 30 °C", "Surface Tension")
    assert [(v.value_std, v.temperature_c) for v in vals] == [(22.3, 20.0), (21.1, 30.0)]
    assert pc.parse_values("White solid", "Physical Description") == []
    assert pc.parse_values("In water, 3300 mg/L at 25 °C", "Surface Tension") == []   # no surface tension in it


def _rec(raw, reference="", heading="Surface Tension", cid=1):
    (pv,) = pc.parse_values(raw, heading)
    return pc.ExperimentalRecord(cid=cid, heading=heading, raw=raw, value=pv.value, value_std=pv.value_std,
                                 unit=pv.unit, temperature_c=pv.temperature_c, reference=reference)


def test_condition_classification():
    mp_liquid = _rec("-95.3 °C", heading="Melting Point")
    recs = pc.classify_conditions([
        _rec("30.2 dyn/cm at 25 °C (0.1% aqueous solution)"),
        _rec("Surface tension of 1 mM solution: 55 mN/m"),
        _rec("27.6 dyn/cm at 20 °C (pure liquid, in contact with air)"),
        _rec("28.9 mN/m at 20 °C", reference="Jasper JJ; J Phys Chem Ref Data 1: 841 (1972)"),
        _rec("18.4 mN/m at 20 °C", reference="Moody CA; Fluorinated surfactants in AFFF"),
        _rec("25.0 dyn/cm at 20 °C"),
    ])
    aq1, aq2, pure, jasper, surf, bare = recs
    assert (aq1.condition_class, aq1.concentration, aq1.condition_basis) == ("aqueous_solution", "0.1%", "text")
    assert (aq2.condition_class, aq2.concentration) == ("aqueous_solution", "1 mM")
    assert (pure.condition_class, pure.condition_basis) == ("pure_liquid", "text")
    assert (jasper.condition_class, jasper.condition_basis) == ("pure_liquid", "reference")
    assert (surf.condition_class, surf.needs_review) == ("unknown", True) and "CMC" in surf.condition_note
    assert (bare.condition_class, bare.needs_review) == ("unknown", True)      # nothing to tell from
    # the same bare value for a compound that is liquid at 20 °C: pure liquid, inferred from the phase
    bare2 = _rec("25.0 dyn/cm at 20 °C")
    pc.classify_conditions([bare2, mp_liquid])
    assert (bare2.condition_class, bare2.condition_basis, bare2.needs_review) == ("pure_liquid", "phase", False)
    # ... and for one that is solid at 20 °C without a surfactant reference: unknown, likely aqueous
    solid = _rec("25.0 dyn/cm at 20 °C")
    pc.classify_conditions([solid, _rec("120 °C", heading="Melting Point")])
    assert solid.condition_class == "unknown" and "likely an aqueous-solution value" in solid.condition_note
    # a melt measured above its melting point is not "solid"
    melt = _rec("Surface tension: 19.5 mN/m at 130 °C")
    pc.classify_conditions([melt, _rec("120 °C", heading="Melting Point")])
    assert melt.condition_class == "pure_liquid"


def test_parachor():
    # hexane: M 86.18, sigma 17.89 mN/m, rho 0.6548 g/cm3 (25 °C) -> P ~ 270 (Sugden group sum ~ 271)
    assert pc.parachor(86.18, 17.89, 0.6548) == pytest.approx(86.18 * 17.89 ** 0.25 / 0.6548)
    assert 265 < pc.parachor(86.18, 17.89, 0.6548) < 275
    assert pc.parachor(None, 17.89, 0.65) is None and pc.parachor(86.18, 0, 0.65) is None


# ------------------------------------------------------------------------------------------------ throttling
def test_throttling_header_and_backoff():
    red_service = INDEX["name_cas_pfoa_cids"]["throttling"]
    st = pc.parse_throttling(red_service)
    assert st == {"request count": ("green", 0), "request time": ("green", 0), "service": ("red", 87)}
    assert pc.throttle_factor(st) == (2.0, 0.0)
    assert pc.throttle_factor(pc.parse_throttling(INDEX["inchikey_pfoa_cids"]["throttling"]))[0] == 1.5  # yellow
    own_red = "Request Count status: Red (80%), Request Time status: Yellow (60%), Service status: Green (10%)"
    assert pc.throttle_factor(pc.parse_throttling(own_red)) == (4.0, 0.0)
    black = "Request Count status: Black (100%), Request Time status: Green (0%), Service status: Green (2%)"
    assert pc.throttle_factor(pc.parse_throttling(black)) == (4.0, pc.BLACK_PAUSE_S)

    clock = FakeClock()
    lim = pc.RateLimiter(rate=4.0, clock=clock, sleep=clock.sleep)
    for _ in range(3):
        lim.wait()
    assert clock.slept == [0.25, 0.25]
    lim.update(red_service)
    lim.wait()
    assert clock.slept[-1] == 0.5
    lim.update(own_red)
    lim.wait()
    assert clock.slept[-1] == 1.0
    lim.update(GREEN)
    lim.wait()
    assert clock.slept[-1] == 0.25
    lim.update(black)
    lim.wait()
    assert clock.slept[-1] == pytest.approx(pc.BLACK_PAUSE_S)
    assert pc.RateLimiter(rate=50).rate == pc.MAX_RATE                  # never above the policy


def test_per_minute_window():
    clock = FakeClock()
    lim = pc.RateLimiter(rate=5.0, per_minute=10, clock=clock, sleep=clock.sleep)
    start = clock.t
    for _ in range(11):
        lim.wait()
    assert clock.t - start == pytest.approx(60.0)                       # the 11th waits for the 1st to age out


def test_client_follows_the_header_and_retries_busy(tmp_path):
    clock = FakeClock()
    fake = FakePubChem()
    client = make_client(tmp_path, fake, clock=clock)
    client.cids_by_name("335-67-1")             # answered with Service status: Red (87%)
    assert client.limiter.factor == 2.0
    client.cids_by_name("1763-23-1")
    assert clock.slept[-1] == pytest.approx(0.5)
    fake.fail_next = [503, 504]
    assert client.cids_by_inchikey("SNGREZUHAYWORS-UHFFFAOYSA-N") == [9554]
    assert len(fake.calls) == 5 and client.requests == 5
    assert clock.slept[-2:] == [pytest.approx(2.0), pytest.approx(4.0)]  # backoff, then the answer
    fake.fail_next = [503] * 3
    client.max_retries = 2
    with pytest.raises(pc.PubChemUnavailable, match="HTTP 503"):
        client.cids_by_smiles("CCO")
    assert client.cache.get(f"GET {pc.REST}/compound/smiles/cids/JSON?smiles=CCO") is None  # errors not cached


def test_network_errors_become_unavailable(tmp_path):
    def down(*_a):
        raise OSError("Name or service not known")

    client = make_client(tmp_path, down, max_retries=1)
    with pytest.raises(pc.PubChemUnavailable, match="Name or service"):
        client.cids_by_name("335-67-1")


# ------------------------------------------------------------------------------------------------ cache
def test_cache_and_offline(tmp_path):
    fake = FakePubChem()
    client = make_client(tmp_path, fake)
    assert client.cids_by_name("335-67-1") == [9554]
    assert client.cids_by_name("notarealchemicalnamexyzzy") == []
    assert client.cids_by_name("335-67-1") == [9554] and client.requests == 2 and client.cache_hits == 1
    assert len(client.cache) == 2
    client.cache.close()

    def no_network(*_a):
        raise AssertionError("offline mode sent a request")

    off = pc.PubChemClient(cache=tmp_path / "cache", offline=True, transport=no_network)
    assert off.cids_by_name("335-67-1") == [9554]
    assert off.cids_by_name("notarealchemicalnamexyzzy") == []           # the 404 is cached too
    with pytest.raises(pc.NotCached):
        off.cids_by_name("1763-23-1")
    assert off.requests == 0 and off.cache_hits == 2
    with pytest.raises(ValueError, match="needs a cache"):
        pc.PubChemClient(cache=False, offline=True)
    assert pc.default_cache_path().parent.parent == tmp_path / "cache"   # conftest's QUANTABENCH_CACHE_DIR


# ------------------------------------------------------------------------------------------------ validation
def test_compare_inchikeys():
    pfoa = "SNGREZUHAYWORS-UHFFFAOYSA-N"
    assert pe.compare_inchikeys(pfoa, pfoa) == "match"
    assert pe.compare_inchikeys(pfoa, "SNGREZUHAYWORS-ZETCQYMHSA-N") == "stereo_differs"
    assert pe.compare_inchikeys(pfoa, "SNGREZUHAYWORS-UHFFFAOYSA-M") == "charge_or_protonation_differs"
    assert pe.compare_inchikeys(pfoa, "YFSUTJLHUFNCNZ-UHFFFAOYSA-N") == "different_structure"
    assert pe.compare_structures([], pfoa)[0] == "no_structure"
    out, note = pe.compare_structures([("as drawn", "SNGREZUHAYWORS-UHFFFAOYSA-M"), ("computed species", pfoa)], pfoa)
    assert out == "match" and "computed species" in note



def test_component_of_a_mixture_record():
    """A CAS RN whose PubChem record has several components (a mixture, a salt with an organic counter-ion):
    QuantaBench's structure equal to one of them is component_of_mixture, and keeping a smaller one is said."""
    from rdkit import Chem

    def ik(smi):
        return Chem.MolToInchiKey(Chem.MolFromSmiles(smi))

    salt = "CCCNCCC.C(=O)(C(C(F)(F)F)(OC(C(C(F)(F)F)(F)F)(F)F)F)O"           # 165951-17-7 in PubChem
    rec = ik(salt)
    out, note = pe.compare_structures([("as drawn", ik("CCCNCCC"))], rec, salt)
    assert out == "component_of_mixture" and "smaller component" in note and "CCCNCCC" in note
    acid = "OC(=O)C(F)(OC(F)(F)C(F)(F)C(F)(F)F)C(F)(F)F"
    out, note = pe.compare_structures([("as drawn", ik(acid))], rec, salt)
    assert out == "component_of_mixture" and "smaller" not in note
    mix = "COC(C(C(F)(F)F)(F)F)(F)F.C(=C/Cl)\\Cl"                          # 218462-19-2: ether + DCE
    out, _ = pe.compare_structures([("as drawn", ik("COC(F)(F)C(F)(F)C(F)(F)F"))], ik(mix), mix)
    assert out == "component_of_mixture"
    assert pe.compare_structures([("as drawn", ik("CCO"))], rec, salt)[0] == "different_structure"
    assert pe.compare_structures([("as drawn", ik("CCO"))], ik("CCCO"), "CCCO")[0] == "different_structure"
    assert pe.OUTCOMES.index("component_of_mixture") < pe.OUTCOMES.index("different_structure")

def test_structure_validation_outcomes(tmp_path):
    pfoa_anion = PFOA.replace("OC(=O)", "[O-]C(=O)")
    queries = [
        pe.Query("pfoa", cas="335-67-1", smiles=PFOA),
        pe.Query("pfoa-anion", cas="335-67-1", smiles=pfoa_anion),
        pe.Query("k-pfos", cas="1763-23-1", smiles=K_PFOS),
        pe.Query("wrong", cas="335-67-1", smiles="CCCCCC"),
        pe.Query("missing", cas="7732-18-5", smiles="O"),
        pe.Query("ambiguous", cas=AMBIGUOUS_CAS, smiles="CCO"),
        pe.Query("no-cas", smiles=PFOA, inchikey="SNGREZUHAYWORS-UHFFFAOYSA-N"),
    ]
    res = pe.enrich(queries, make_client(tmp_path), validate=True)
    got = {r["id"]: (r["pubchem_match"], r["pubchem_status"], str(r["pubchem_cid"])) for r in res.rows}
    assert got["pfoa"] == ("match", "ok", "9554")
    assert got["pfoa-anion"][0] == "charge_or_protonation_differs"
    assert got["k-pfos"][:2] == ("charge_or_protonation_differs", "ok")       # salt drawn, PubChem's CAS = acid
    assert got["wrong"][0] == "different_structure"
    assert got["missing"][:2] == ("cas_not_in_pubchem", "not_found")
    assert got["ambiguous"] == ("ambiguous_cas", "ambiguous", "702")          # the CID with QB's structure is used
    assert got["no-cas"] == ("no_cas", "ok", "9554")                          # found by InChIKey
    rows = {r["id"]: r for r in res.rows}
    assert "QuantaBench's structure is CID 702" in rows["ambiguous"]["pubchem_match_note"]
    assert rows["pfoa"]["pubchem_cas"] == "335-67-1" and rows["pfoa"]["pubchem_cas_in_xrefs"] == "yes"
    assert rows["pfoa"]["pubchem_inchikey"] == "SNGREZUHAYWORS-UHFFFAOYSA-N"
    assert res.counts["match"]["charge_or_protonation_differs"] == 2
    assert "structure check: match 1" in res.summary()


# ------------------------------------------------------------------------------------------------ CLI
def _write_csv(path: Path, rows: list[dict]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_cli_enrich_csv(tmp_path, monkeypatch, capsys):
    fake = FakePubChem()
    monkeypatch.setattr(pc, "urllib_transport", fake)
    monkeypatch.setattr(pc, "_sleep", lambda s: None)
    src = _write_csv(tmp_path / "list.csv", [
        {"Name": "PFOA", "CAS": "335-67-1", "SMILES": PFOA},
        {"Name": "PFOS potassium salt as PFOS", "CAS": "1763-23-1", "SMILES": K_PFOS},
        {"Name": "notarealchemicalnamexyzzy", "CAS": "", "SMILES": ""},
    ])
    out, rec = tmp_path / "enriched.csv", tmp_path / "experimental.csv"
    args = ["pubchem", "enrich", str(src), "--experimental", "--validate", "--out", str(out), "--records", str(rec),
            "--cache", str(tmp_path / "pc"), "--rate", "5"]
    assert cli.main(args) == 0
    text = capsys.readouterr().out
    assert "3 molecule(s), 2 with a CAS RN" in text and "structure check: match 1" in text
    rows = _read_csv(out)
    assert [r["pubchem_status"] for r in rows] == ["ok", "ok", "not_found"]
    pfoa, pfos, nothing = rows
    assert list(pfoa)[:3] == ["Name", "CAS", "SMILES"] and pfoa["id"] == "335-67-1"
    assert (pfoa["pubchem_cid"], pfoa["pubchem_cas"], pfoa["pubchem_match"]) == ("9554", "335-67-1", "match")
    assert pfoa["pubchem_st_unclassified_mN_m"] == "15.2" and pfoa["pubchem_st_review"] == "yes"
    assert pfoa["pubchem_st_pure_mN_m"] == "" and pfoa["pubchem_density_class"] == "solid"
    assert pfoa["pubchem_mp_C"] == "53.65" and pfoa["pubchem_bp_C"] == "189.0"
    assert pfos["pubchem_match"] == "charge_or_protonation_differs" and pfos["pubchem_exp_records"] == "0"
    assert nothing["pubchem_match"] == "no_cas" and nothing["pubchem_query_type"] == "name"
    records = _read_csv(rec)
    st = [r for r in records if r["heading"] == "Surface Tension"]
    assert len(st) == 1 and st[0]["id"] == "335-67-1" and st[0]["condition_class"] == "unknown"
    assert st[0]["needs_review"] == "yes" and st[0]["source_id"] == "7137"
    n_requests = len(fake.calls)
    assert n_requests == 8   # 2 CAS + name; RN, properties, synonyms; 2 PUG View records (PFOS has none)

    # a rerun from the cache only: same table, nothing sent
    out2 = tmp_path / "offline.csv"
    assert cli.main([*args[:-6], "--out", str(out2), "--records", str(tmp_path / "r2.csv"), "--cache",
                     str(tmp_path / "pc"), "--offline"]) == 0
    assert "0 request(s) to PubChem" in capsys.readouterr().out and len(fake.calls) == n_requests
    assert _read_csv(out2) == rows


def test_cli_lookup(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pc, "urllib_transport", FakePubChem())
    monkeypatch.setattr(pc, "_sleep", lambda s: None)
    assert cli.main(["pubchem", "lookup", "335-67-1", "--experimental", "--cache", str(tmp_path / "pc")]) == 0
    out = capsys.readouterr().out
    assert "pubchem_cid" in out and "9554" in out and "SNGREZUHAYWORS-UHFFFAOYSA-N" in out
    assert "Surface Tension" in out and "[unknown]" in out
    assert cli.main(["pubchem", "lookup", "notarealchemicalnamexyzzy", "--cache", str(tmp_path / "pc")]) == 1


def test_cli_enrich_run_folder(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pc, "urllib_transport", FakePubChem())
    monkeypatch.setattr(pc, "_sleep", lambda s: None)
    run = tmp_path / "run"
    run.mkdir()
    # a QuantaBench summary.csv: the computed species is PFOA's anion (pH 7), the input as drawn the acid
    _write_csv(run / "summary.csv", [
        {"id": "335-67-1", "name": "PFOA", "status": "ok", "input_smiles": PFOA, "smiles": "x",
         "inchikey": "SNGREZUHAYWORS-UHFFFAOYSA-M", "canonical_smiles": "x",
         "input_inchikey": "SNGREZUHAYWORS-UHFFFAOYSA-N"},
        {"id": "DTXSID8031865", "name": "", "status": "ok", "input_smiles": "", "smiles": "", "inchikey": "",
         "canonical_smiles": "", "input_inchikey": ""},
    ])
    out = tmp_path / "enriched.csv"
    assert cli.main(["pubchem", "enrich", str(run), "--validate", "--out", str(out), "--cache",
                     str(tmp_path / "pc")]) == 0
    rows = _read_csv(out)
    assert rows[0]["pubchem_match"] == "match" and rows[0]["qb_inchikey"] == "SNGREZUHAYWORS-UHFFFAOYSA-N"
    assert rows[1]["pubchem_query"] == "name:DTXSID8031865" and rows[1]["pubchem_match"] == "no_cas"
    assert cli.main(["pubchem", "enrich", str(tmp_path / "nope.csv"), "--out", str(out)]) == 2
