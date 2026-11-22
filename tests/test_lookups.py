"""Settings > Lookups (SETTINGS-1, Spartan's Preferences > URLs and Databases): URL templates with %inchikey, %inchi,
%cas, %smiles and %name, and PubChem's cache folder and maximum age. Not science settings."""

import sqlite3
import urllib.parse

import pytest

from quantabench import lookups
from quantabench import settings as settings_mod
from quantabench.chem import jobs
from quantabench.chem import pubchem as pc

PFOA = "OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"
PFOA_KEY = "SNGREZUHAYWORS-UHFFFAOYSA-N"


def test_values_and_expansion():
    v = lookups.values_for(smiles=PFOA, name="PFOA", ident="pfoa")
    assert v["inchikey"] == PFOA_KEY and v["inchi"].startswith("InChI=1S/C8HF15O2/") and v["cas"] == ""
    assert (v["smiles"], v["name"]) == (PFOA, "PFOA")
    url, missing = lookups.expand("https://x.org/?k=%inchikey&i=%inchi&s=%smiles&n=%name&c=%cas", v)
    q = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))
    assert q == {"k": PFOA_KEY, "i": v["inchi"], "s": PFOA, "n": "PFOA"} and missing == ["%cas"]
    assert "%2F" in url and "=InChI%3D1S" in url                      # every value percent-encoded
    assert lookups.placeholders("https://x/%inchikey/%inchi/%inchikey") == ["%inchikey", "%inchi"]
    assert lookups.values_for(ident="x")["inchikey"] == ""               # no structure, no keys


def test_cas_from_fields_id_or_name():
    assert lookups.find_cas({"CASRN": "335-67-1", "other": "1763-23-1"}) == "335-67-1"
    assert lookups.find_cas({"CAS_RN": "CAS-0335-67-1"}) == "335-67-1"
    assert lookups.find_cas({"note": "1763-23-1"}) == "1763-23-1"          # a field that is a CAS RN as a whole
    assert lookups.find_cas({"note": "see 1763-23-1"}, "pfos") == ""       # not a number inside text
    assert lookups.find_cas(None, "335-67-1", "PFOA") == "335-67-1"        # the id
    assert lookups.find_cas(None, "pfoa", "335-67-2") == ""                # a wrong check digit is not a CAS RN


def test_templates_and_defaults():
    s = settings_mod.Settings()
    found = lookups.templates(s.lookups.urls)
    assert [t.name for t in found] == ["PubChem", "NIST WebBook", "nmrshiftdb2 (13C spectrum, JCAMP-DX)", "EPA CompTox"]
    used = {p for t in found for p in lookups.placeholders(t.url)}
    assert used == {"%inchikey", "%cas", "%inchi"}
    assert lookups.check("ftp://x") and lookups.check("https://") and not lookups.check("https://a.org/%name")
    bad = [{"name": "", "url": "https://a"}, {"name": "x", "url": "file:///etc"}, "junk", {"name": "ok", "url": "http://b"}]
    assert lookups.templates(bad) == [lookups.Lookup("ok", "http://b")]
    v = lookups.values_for(smiles=PFOA, ident="335-67-1")
    urls = {t.name: lookups.expand(t.url, v)[0] for t in found}
    assert urls["PubChem"] == f"https://pubchem.ncbi.nlm.nih.gov/#query={PFOA_KEY}"
    assert urls["NIST WebBook"] == "https://webbook.nist.gov/cgi/cbook.cgi?ID=335-67-1&Units=SI"
    assert urls["EPA CompTox"].endswith(f"results?search={PFOA_KEY}")


def test_settings_round_trip_and_not_science(isolated_settings):
    s = settings_mod.Settings()
    s.lookups.urls = [{"name": "Wikipedia", "url": "https://en.wikipedia.org/wiki/%name"}]
    s.lookups.pubchem_cache_dir, s.lookups.pubchem_max_age_days = "~/qb-pubchem", 30.0
    settings_mod.save(s)
    again = settings_mod.load()
    assert again.lookups.urls == [{"name": "Wikipedia", "url": "https://en.wikipedia.org/wiki/%name"}]
    assert (again.lookups.pubchem_cache_dir, again.lookups.pubchem_max_age_days) == ("~/qb-pubchem", 30.0)
    assert settings_mod.Settings.from_dict({"lookups": {"urls": None, "pubchem_max_age_days": "x"}}).lookups.urls == \
        [dict(u) for u in settings_mod.DEFAULT_LOOKUP_URLS]                 # a bad value: the defaults
    assert not {"urls", "pubchem_cache_dir", "pubchem_max_age_days"} & set(jobs.SCIENCE_FIELDS)


def test_pubchem_cache_folder_from_the_settings(isolated_settings, monkeypatch, tmp_path):
    assert pc.cache_settings() == ("", 0.0)
    monkeypatch.delenv("QUANTABENCH_CACHE_DIR")
    s = settings_mod.Settings()
    s.lookups.pubchem_cache_dir = str(tmp_path / "pc")
    s.lookups.pubchem_max_age_days = 7.0
    settings_mod.save(s)
    assert pc.cache_settings() == (str(tmp_path / "pc"), 7.0)
    assert pc.default_cache_path() == tmp_path / "pc" / "pubchem_cache.sqlite"
    client = pc.PubChemClient(transport=lambda *a: pc.HttpResponse(200, b"{}", {}))
    assert client.cache.path == tmp_path / "pc" / "pubchem_cache.sqlite" and client.cache.max_age_s == 7 * 86400
    client.cache.close()
    monkeypatch.setenv("QUANTABENCH_CACHE_DIR", str(tmp_path / "env"))      # the environment still wins (tests)
    assert pc.default_cache_path() == tmp_path / "env" / "pubchem" / "pubchem_cache.sqlite"


def test_pubchem_cache_maximum_age(tmp_path):
    calls = []

    def transport(method, url, data, headers, timeout):
        calls.append(url)
        return pc.HttpResponse(200, f'{{"n": {len(calls)}}}'.encode(), {})

    url = f"{pc.REST}/compound/name/x/cids/JSON"
    client = pc.PubChemClient(cache=tmp_path / "c", transport=transport, max_age_days=1)
    assert client.get_json(url) == {"n": 1} and client.get_json(url) == {"n": 1} and len(calls) == 1
    with sqlite3.connect(client.cache.path) as db:                            # stored 3 days ago
        db.execute("UPDATE responses SET fetched_utc = '2000-01-01T00:00:00+00:00'")
    assert client.get_json(url) == {"n": 2} and len(calls) == 2              # too old: asked again and replaced
    with sqlite3.connect(client.cache.path) as db:
        db.execute("UPDATE responses SET fetched_utc = '2000-01-01T00:00:00+00:00'")
    off = pc.PubChemClient(cache=client.cache, offline=True)
    assert off.get_json(url) == {"n": 2}                                     # offline: an old answer is still used
    forever = pc.PubChemClient(cache=tmp_path / "c", transport=transport, max_age_days=0)
    assert forever.get_json(url) == {"n": 2} and len(calls) == 2
    with pytest.raises(pc.NotCached):
        off.get_json(f"{pc.REST}/compound/name/y/cids/JSON")
