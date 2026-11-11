"""The conformer rank step (RANK-2, chem.rank): the levels and the auto rules with their fallbacks, g-xTB single points
(a stand-in program), the GFN1-xTB CPCM term (tblite), the single-point cache, the pipeline's order, window and
Boltzmann weights with fake rank energies and which conformer the descriptors come from, the record and the exports,
resume detecting a rank change, the CLI and worker options, and the GUI control."""

import json
import sqlite3
import stat
import sys
import textwrap

import numpy as np
import pytest
from conftest import HAS_DISPLAY

from quantabench import settings as settings_mod
from quantabench.chem import conformers as confs
from quantabench.chem import eng_gxtb, eng_tblite, engines, export, external, jobs, pipeline, rank
from quantabench.chem.engines import HARTREE_EV
from quantabench.chem.inputs import from_smiles

ACETATE = [6, 6, 8, 8, 1, 1, 1]
_real_available = eng_gxtb.available
HEXANOL = "CCCCCCO"
needs_tblite = pytest.mark.skipif(not eng_tblite.available()[0], reason="needs tblite")
posix_only = pytest.mark.skipif(sys.platform == "win32", reason="the stand-in g-xTB is a POSIX script")


@pytest.fixture
def have(monkeypatch):
    """Pretend which rank engines are installed: ``have(gxtb=..., tblite=..., aimnet=...)`` (all True first)."""
    state = {"gxtb": True, "tblite": True, "aimnet": True}
    real_available = engines.available

    monkeypatch.setattr(eng_gxtb, "available", lambda: (state["gxtb"], "" if state["gxtb"] else "g-xTB not found"))
    monkeypatch.setattr(eng_tblite, "available", lambda: (state["tblite"], "" if state["tblite"] else "no tblite"))
    monkeypatch.setattr(engines, "available", lambda key, quick=False: (
        (state["aimnet"], "" if state["aimnet"] else "no aimnet") if key == "aimnet2" else real_available(key, quick)))

    def set_(**kw):
        state.update(kw)
    return set_


# ------------------------------------------------------------------ choices and defaults
def test_rank_methods_agree_everywhere():
    from quantabench import cli
    from quantabench.dist import worker

    assert cli.rank_choices() == list(rank.RANK_METHODS) == list(worker.CHOICES["rank_method"])
    assert [k for k, _label in rank.CHOICES] == list(rank.RANK_METHODS)
    s = settings_mod.Settings()
    assert pipeline.BuildConfig().rank_method == s.build.rank_method == "auto"
    assert pipeline.BuildConfig().rank_window_kj == s.build.rank_window_kj == rank.DEFAULT_WINDOW_KJ == 30.0
    cfg = pipeline.BuildConfig.from_settings(s)
    assert (cfg.rank_method, cfg.rank_window_kj) == ("auto", 30.0)
    assert {"rank_method", "rank_window_kj"} <= set(jobs.SCIENCE_FIELDS)
    with pytest.raises(ValueError):
        rank.resolve("dft", ACETATE)


# ------------------------------------------------------------------ auto and the fallbacks
def test_auto_by_phase_and_coverage(have):
    p = rank.resolve("auto", ACETATE, -1, 1, "water")
    assert (p.level, p.notes) == ("gxtb+cpcm", [])
    assert rank.resolve("auto", [6, 1, 1, 1, 1], 0, 1, "").level == "aimnet2"
    assert rank.resolve("auto", [6, 1, 1, 1], 0, 2, "").level == "gxtb"                 # AIMNet2: closed shells only
    assert rank.resolve("auto", [6, 50, 1, 1, 1], 0, 1, "").level == "gxtb"             # Sn: not AIMNet2's
    p = rank.resolve("auto", [92, 8, 8], 2, 1, "water")                                 # U: g-xTB yes, GFN1 no
    assert p.level == "refine" and any("GFN1-xTB" in n for n in p.notes)
    assert rank.resolve("auto", [92, 8, 8], 2, 1, "").level == "gxtb"
    have(gxtb=False)                                     # no g-xTB (or a machine without a build): AIMNet2 + CPCM
    p = rank.resolve("auto", ACETATE, -1, 1, "water")
    assert p.level == "aimnet2+cpcm" and p.notes == ["rank gxtb+cpcm: g-xTB not found"]
    assert rank.summary(p) == "rank auto -> aimnet2+cpcm: g-xTB not found"
    assert rank.resolve("auto", [6, 1, 1, 1], 0, 2, "").level == "refine"
    have(aimnet=False)                                   # then the refinement, noted
    p = rank.resolve("auto", ACETATE, -1, 1, "water")
    assert p.level == "refine" and len(p.notes) == 2 and "no aimnet" in rank.summary(p)
    have(gxtb=True, tblite=False)                        # the CPCM term needs tblite
    assert rank.resolve("auto", ACETATE, -1, 1, "water").level == "refine"
    assert rank.resolve("auto", ACETATE, -1, 1, "").level == "gxtb"


def test_explicit_levels_and_their_fallbacks(have, monkeypatch):
    assert rank.resolve("refine", ACETATE, -1, 1, "water").level == "refine"
    p = rank.resolve("gxtb+cpcm", ACETATE, -1, 1, "")
    assert p.level == "gxtb" and "gas phase has no solvation term" in p.notes[0]
    assert rank.resolve("aimnet2+cpcm", ACETATE, -1, 1, "").level == "aimnet2"
    assert rank.resolve("aimnet2", ACETATE, -1, 1, "water").level == "aimnet2"          # allowed: gas-phase energies
    have(gxtb=False)
    p = rank.resolve("gxtb", ACETATE, -1, 1, "")
    assert p.level == "refine" and p.notes == ["rank gxtb: g-xTB not found"]           # explicit: to the refinement
    p = rank.resolve("qm", ACETATE, -1, 1, "water")
    assert p.level == "refine" and "no QM level" in p.notes[0]
    from quantabench.chem import qm

    monkeypatch.setattr(qm, "available", lambda: (True, ""))
    assert rank.resolve("qm", ACETATE, -1, 1, "water", qm_method="wb97x-d/def2-svpd").level == "qm"
    assert rank.label("gxtb+cpcm", "water") == "g-xTB + CPCM(GFN1, water)"
    assert rank.label("refine", "", "GFN2-xTB (tblite)") == "GFN2-xTB (tblite) (refinement)"


# ------------------------------------------------------------------ g-xTB, through a stand-in program
FAKE_GXTB = textwrap.dedent("""\
    #!{python}
    # a stand-in for the g-xTB build of xtb: prints xtb's summary line; E depends on the geometry
    import json, os, sys
    args = sys.argv[1:]
    if os.environ.get("FAKE_GXTB_BROKEN"):
        print("dyld[1]: Library not loaded: @rpath/libgcc_s.1.1.dylib", file=sys.stderr); sys.exit(134)
    with open(os.environ["FAKE_GXTB_LOG"], "a") as f:
        f.write(json.dumps({{"args": args, "omp": os.environ.get("OMP_NUM_THREADS"),
                            "stack": os.environ.get("OMP_STACKSIZE"), "path0": os.environ["PATH"].split(os.pathsep)[0]}})
                + "\\n")
    if "--version" in args:
        print("   * xtb version 6.7.1 (fake) compiled by 'test'")
        sys.exit(0)
    if os.environ.get("FAKE_GXTB_FAIL"):
        print("SCF not converged"); sys.exit(1)
    rows = [ln.split() for ln in open(args[0]).read().splitlines()[2:] if ln.strip()]
    xyz = [[float(v) for v in r[1:4]] for r in rows]
    e = -10.0 - 0.001 * sum(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5
                            for i, a in enumerate(xyz) for b in xyz[i + 1:])
    print(f"         :: total energy {{e:22.12f}} Eh    ::")
    """)


@pytest.fixture
def fake_gxtb(tmp_path, monkeypatch):
    prog = tmp_path / "bin" / "gxtb"
    prog.parent.mkdir()
    prog.write_text(FAKE_GXTB.format(python=sys.executable), encoding="utf-8")
    prog.chmod(prog.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "gxtb_calls.jsonl"
    monkeypatch.setenv("QUANTABENCH_GXTB", str(prog))
    monkeypatch.setenv("FAKE_GXTB_LOG", str(log))
    external.clear_cache()
    eng_gxtb.clear_cache()
    yield prog, log
    external.clear_cache()
    eng_gxtb.clear_cache()


def _calls(log):
    return [json.loads(ln) for ln in log.read_text().splitlines()] if log.exists() else []


@posix_only
def test_gxtb_single_point_command_and_build_info(fake_gxtb, monkeypatch):
    prog, log = fake_gxtb
    assert eng_gxtb.available() == (True, "")
    info = eng_gxtb.build_info()
    assert not info.known and info.sha256 == eng_gxtb.file_sha256(prog) and "unknown build" in info.label()
    assert info.xtb_version == "6.7.1 (fake)"
    pos = np.array([[0, 0, 0], [1.2, 0, 0], [0, 1.1, 0]], float)
    folder = prog.parent.parent / "job"
    e = eng_gxtb.single_point(folder, [6, 8, 1], pos, charge=-1, multiplicity=1, threads=3)
    assert e == pytest.approx(-10.0 - 0.001 * (1.2 + 1.1 + np.hypot(1.2, 1.1)), abs=1e-9)
    call = _calls(log)[-1]
    assert call["args"] == ["calc.xyz", "--gxtb", "--chrg", "-1", "--norestart"]       # no --uhf for a singlet
    assert call["omp"] == "3" and call["stack"] == "1G" and call["path0"] == str(prog.parent)
    eng_gxtb.single_point(folder, [6, 1, 1, 1], pos[[0, 1, 2, 2]] + np.array([[0, 0, 0]] * 3 + [[0, 0, 1]]),
                          multiplicity=2)
    assert _calls(log)[-1]["args"][-2:] == ["--uhf", "1"]
    monkeypatch.setenv("FAKE_GXTB_FAIL", "1")
    with pytest.raises(external.ExternalProgramError, match="g-xTB single point failed"):
        eng_gxtb.single_point(folder, [6, 8, 1], pos)
    # a pinned build is recognised by its SHA-256
    eng_gxtb.clear_cache()
    monkeypatch.setitem(eng_gxtb.KNOWN_BUILDS, info.sha256, ("2.0.1", "linux-x86_64"))
    assert eng_gxtb.build_info().label().startswith("g-xTB 2.0.1 (sha256 ")


@posix_only
def test_a_gxtb_that_does_not_start_is_not_available(fake_gxtb, monkeypatch, have):
    monkeypatch.setattr(eng_gxtb, "available", _real_available)         # the real check (``have`` faked it)
    monkeypatch.setenv("FAKE_GXTB_BROKEN", "1")
    external.clear_cache()
    eng_gxtb.clear_cache()
    ok, why = eng_gxtb.available()
    assert not ok and "does not start" in why and "libgcc_s" in why
    p = rank.resolve("auto", ACETATE, -1, 1, "water")                 # auto falls back, noted
    assert p.level == "aimnet2+cpcm" and "does not start" in p.notes[0]


def test_stack_size_cap():
    assert eng_gxtb._stacksize("4G") == "1G" and eng_gxtb._stacksize("512M") == "512M"
    assert eng_gxtb._stacksize("1G") == "1G" and eng_gxtb._stacksize("junk") == "1G"
    assert eng_gxtb.KNOWN_BUILDS["c2af5f3db91794cc40b5e38712f28f0d24acb9af040abb0bfb627313e926cff6"][1] == \
        "windows-x86_64"
    assert "gxtb" in external.PROGRAMS and external.PROGRAMS["gxtb"].executables == ("gxtb",)   # never plain xtb


@needs_tblite
def test_cpcm_term_and_the_cache(tmp_path):
    water = [8, 1, 1]
    pos = np.array([[0, 0, 0.1173], [0, 0.7572, -0.4692], [0, -0.7572, -0.4692]])
    cache = rank.SinglePointCache(tmp_path / "c.sqlite")
    e_gas, e_cpcm = rank.cpcm_term_eh(water, pos, 0, 1, "water", cache)
    dg_kj = (e_cpcm - e_gas) * HARTREE_EV * rank.KJ_PER_EV
    assert -60 < dg_kj < -5                              # the solvation free energy of water, roughly
    assert cache.misses == 2 and cache.hits == 0
    assert rank.cpcm_term_eh(water, pos, 0, 1, "water", cache) == (e_gas, e_cpcm)
    assert cache.hits == 2
    assert rank.cache_key("a", "x", water, pos, 0, 1) == rank.cache_key("a", "x", water, pos + 4e-7, 0, 1)
    assert rank.cache_key("a", "x", water, pos, 0, 1) != rank.cache_key("a", "x", water, pos, -1, 1)
    cache.close()
    broken = rank.SinglePointCache(tmp_path / "missing" / "\0bad")     # never fails a molecule
    assert broken.get("k") is None
    broken.put("k", 1.0)


def _pool(smiles=HEXANOL, n=6):
    """An MMFF94s conformer set and its 'refined' conformers (the force-field energies)."""
    from quantabench.chem.standardize import prepare

    mol, _notes = prepare(from_smiles(smiles, id="m").to_mol(), None, True)
    cs = confs.generate(mol, max_conformers=n * 3, window_kcal=50, rmsd_threshold=0.3, seed=7)
    refined = [pipeline.ConformerResult(rank=0, conf_id=cid, energy_ev=e / confs.KCAL_PER_EV, ff_energy_kcal=e)
               for cid, e in zip(cs.conf_ids[:n], cs.energies_kcal[:n])]
    return cs, refined


@posix_only
@needs_tblite
def test_compute_gxtb_plus_cpcm_uses_the_cache(fake_gxtb, tmp_path):
    prog, log = fake_gxtb
    cs, pool = _pool(n=3)
    cache = rank.SinglePointCache(tmp_path / "c.sqlite")
    r = rank.compute("gxtb+cpcm", cs.mol, pool, solvent="water", threads=1, cache=cache)
    assert len(r.energies_ev) == 3 and all(v is not None for v in r.energies_ev.values())
    t = r.terms[pool[0].conf_id]
    assert r.energies_ev[pool[0].conf_id] == pytest.approx((t["gxtb_eh"] + t["gfn1_cpcm_eh"] - t["gfn1_gas_eh"])
                                                           * HARTREE_EV)
    assert t["dgsolv_kj"] < 0 and "unknown build" in rank.engine_label(r) and "CPCM (ddX, water)" in rank.engine_label(r)
    n_runs = sum("--gxtb" in c["args"] for c in _calls(log))
    assert n_runs == 3
    again = rank.compute("gxtb+cpcm", cs.mol, pool, solvent="water", threads=1, cache=cache)
    assert again.energies_ev == r.energies_ev and sum("--gxtb" in c["args"] for c in _calls(log)) == n_runs
    assert again.cache_hits >= 9
    cache.close()


# ------------------------------------------------------------------ the pipeline step, with fake rank energies
def _fake_rank(monkeypatch, energies_of, level="gxtb+cpcm", fail=False, seen=None):
    """rank.resolve -> ``level``; rank.compute -> ``energies_of(pool)`` (conf_id -> eV or None)."""
    monkeypatch.setattr(rank, "resolve", lambda req, *a, **k: rank.Plan(req, "refine" if req == "refine" else level))

    def compute(lv, mol, pool, **kw):
        if seen is not None:
            seen.append([c.conf_id for c in pool])
        if fail:
            raise RuntimeError("g-xTB exploded")
        return rank.Ranked(lv, energies_of(pool), engines={"gxtb": {}}, notes=[])
    monkeypatch.setattr(rank, "compute", compute)


def cfg(tmp_path, **kw):
    base = dict(method="mmff", max_conformers=18, refine_top=8, xtb_properties=False, output_dir=str(tmp_path),
                rmsd_threshold=0.3, write_files=False, protonate=False, seed=7)
    base.update(kw)
    return pipeline.BuildConfig(**base)


def test_rank_energies_order_window_and_weights(tmp_path, monkeypatch):
    """Reversed energies: the refinement's highest becomes the pick; one conformer 40 kJ/mol up is dropped; the window
    at the refinement level is not applied before the rank step (energy_window_kcal tiny)."""
    seen = []

    def reversed_energies(pool):
        out = {c.conf_id: -c.energy_ev for c in pool}
        out[pool[0].conf_id] = max(out.values()) + 40.0 / rank.KJ_PER_EV          # the refinement's lowest: +40
        return out
    _fake_rank(monkeypatch, reversed_energies, seen=seen)
    res = pipeline.run_molecule(from_smiles(HEXANOL, id="hexanol"), cfg(tmp_path, energy_window_kcal=0.01,
                                                                      solvent="water"))
    assert res.status == "ok", res.error
    pool = seen[0]
    assert len(pool) >= 4                                  # RMSD pruning only: no 0.01 kcal/mol window first
    by_refine = sorted((c for c in res.conformers), key=lambda c: c.energy_ev)
    ranks = [c.rank_energy_ev for c in res.conformers]
    assert ranks == sorted(ranks) and len(res.conformers) == len(pool) - 1
    assert pool[0] not in [c.conf_id for c in res.conformers]                    # 40 kJ/mol > the 30 kJ/mol window
    e0 = res.conformers[0].rank_energy_ev
    assert [c.rel_kcal for c in res.conformers] == pytest.approx([(e - e0) * confs.KCAL_PER_EV for e in ranks])
    w = confs.boltzmann_weights([(e - e0) * confs.KCAL_PER_EV for e in ranks], 298.15)
    assert [c.weight for c in res.conformers] == pytest.approx(list(w))
    assert res.conformers[0].conf_id == by_refine[-1].conf_id                    # the refinement's highest kept
    assert res.conformers[0].refine_rel_kcal == pytest.approx(max(c.refine_rel_kcal for c in res.conformers))
    r = res.rank
    assert r["level"] == "gxtb+cpcm" and r["method"] == "auto" and r["pick_changed"] is True
    assert r["n_pool"] == len(pool) and r["n_kept"] == len(pool) - 1 and r["window_kj"] == 30.0
    assert len(r["conformers"]) == len(pool) and sum(x["kept"] for x in r["conformers"]) == r["n_kept"]
    row = res.summary_row()
    assert row["rank_level"] == "gxtb+cpcm" and row["rank_pick_changed"] is True
    assert row["rank_energy_eV"] == res.conformers[0].rank_energy_ev and row["rank_n_pool"] == len(pool)
    assert "rank: g-xTB + CPCM(GFN1, water)" in res.phase_notes()
    assert res.descriptors["rank_method"] == "gxtb+cpcm" and "rank" in res.timings
    stage = next(s for s in res.telemetry["stages"] if s["stage"] == "rank")
    assert stage["outcome"] == "ok" and stage["n_in"] == len(pool) and stage["n_out"] == len(pool) - 1
    # a wider window keeps it; refine keeps the old behaviour (no rank energies, the refinement's window)
    wide = pipeline.run_molecule(from_smiles(HEXANOL, id="h"), cfg(tmp_path, rank_window_kj=50.0, solvent="water"))
    assert len(wide.conformers) == len(pool)
    old = pipeline.run_molecule(from_smiles(HEXANOL, id="h"), cfg(tmp_path, rank_method="refine",
                                                                  energy_window_kcal=0.01))
    assert len(old.conformers) == 1 and old.conformers[0].rank_energy_ev is None
    assert old.rank["level"] == "refine" and old.summary_row()["rank_pick_changed"] == ""


def test_descriptors_follow_the_rank_pick(tmp_path, monkeypatch):
    """xTB properties, frequencies, the 3D descriptors and the QM stage's conformers all take the rank level's order."""
    _fake_rank(monkeypatch, lambda pool: {c.conf_id: -c.energy_ev for c in pool})
    got = {}
    monkeypatch.setattr(pipeline, "_xtb_properties",
                        lambda result, mol, best, cfg, solvated, log: got.setdefault("xtb", best.conf_id) and True)
    monkeypatch.setattr(pipeline, "_frequencies",
                        lambda result, cs, best, cfg, stop, log: got.setdefault("freq", best.conf_id) and True)
    monkeypatch.setattr(pipeline, "_qm_stage", lambda result, cs, cfg, threads, stop, log: got.setdefault(
        "qm", [c.conf_id for c in result.conformers][:cfg.qm_top]))
    from quantabench.chem import properties

    monkeypatch.setattr(properties, "tblite_available", lambda: True)
    monkeypatch.setattr(properties, "available", lambda: True)
    res = pipeline.run_molecule(from_smiles(HEXANOL, id="h"), cfg(tmp_path, solvent="water", xtb_properties=True,
                                                                  frequencies=True, qm_method="hf/sto-3g", qm_top=2))
    assert res.status == "ok", res.error
    best = res.conformers[0].conf_id
    refine_best = min(res.conformers, key=lambda c: c.energy_ev).conf_id
    assert best != refine_best and got["xtb"] == got["freq"] == best
    assert got["qm"] == [c.conf_id for c in res.conformers[:2]]
    from quantabench.chem import descriptors as desc

    ens = desc.ensemble_descriptors(res.mol, [c.conf_id for c in res.conformers], [c.weight for c in res.conformers],
                                    pipeline.surface_options(cfg(tmp_path)))
    assert res.descriptors["volume_A3_best"] == ens["volume_A3_best"]
    assert res.descriptors["volume_A3_boltz"] == pytest.approx(ens["volume_A3_boltz"])


def test_rank_failures_fall_back_to_the_refinement(tmp_path, monkeypatch):
    _fake_rank(monkeypatch, None, fail=True)
    res = pipeline.run_molecule(from_smiles(HEXANOL, id="h"), cfg(tmp_path, solvent="water"))
    assert res.status == "ok" and res.rank["level"] == "refine"
    assert any("g-xTB exploded" in n and "ranked at the refinement level" in n for n in res.notes)
    assert next(s for s in res.telemetry["stages"] if s["stage"] == "rank")["outcome"] == "failed"
    assert res.phase_steps["rank"].startswith("refinement")
    # one conformer without an energy is left out, the others ranked
    _fake_rank(monkeypatch, lambda pool: {c.conf_id: (None if i == 0 else -c.energy_ev) for i, c in enumerate(pool)})
    res = pipeline.run_molecule(from_smiles(HEXANOL, id="h"), cfg(tmp_path, solvent="water"))
    assert res.rank["level"] == "gxtb+cpcm" and res.rank["n_kept"] == res.rank["n_pool"] - 1
    _fake_rank(monkeypatch, lambda pool: {c.conf_id: None for c in pool})
    res = pipeline.run_molecule(from_smiles(HEXANOL, id="h"), cfg(tmp_path, solvent="water"))
    assert res.rank["level"] == "refine" and any("no conformer got an energy" in n for n in res.notes)


def test_auto_without_rank_engines_ranks_at_the_refinement(tmp_path, have):
    have(gxtb=False, aimnet=False)
    res = pipeline.run_molecule(from_smiles(HEXANOL, id="h"), cfg(tmp_path))
    assert res.status == "ok" and res.rank["level"] == "refine" and res.rank["method"] == "auto"
    assert any(n.startswith("rank auto -> refine:") for n in res.notes)
    assert res.summary_row()["rank_level"] == "refine"


def test_qm_rank_runs_the_qm_stage_once_on_every_conformer(tmp_path, monkeypatch):
    from quantabench.chem import qm

    monkeypatch.setattr(rank, "resolve", lambda req, *a, **k: rank.Plan(req, "qm"))
    calls = []

    def fake_stage(result, cs, cfg, *, threads=1, should_stop=None, log=None):
        calls.append((cfg.qm_top, [c.conf_id for c in result.conformers]))
        confs_ = [{"conf_id": c.conf_id, "energy_ev": -c.energy_ev} for c in result.conformers]   # reversed order
        result.qm = {"level": cfg.qm_method, "program": "psi4", "conformers": confs_, "solvent": "water"}
    monkeypatch.setattr(qm, "run_stage", fake_stage)
    res = pipeline.run_molecule(from_smiles(HEXANOL, id="h"), cfg(tmp_path, solvent="water", qm_method="hf/sto-3g",
                                                                  qm_top=1, rank_method="qm"))
    assert res.status == "ok", res.error
    assert len(calls) == 1 and calls[0][0] == res.rank["n_pool"] == len(calls[0][1])   # once, on every conformer
    assert res.rank["level"] == "qm" and res.rank["qm_done"] and res.phase_steps["qm"].endswith("(the rank step)")
    ranks = [c.rank_energy_ev for c in res.conformers]
    assert ranks == sorted(ranks) and res.conformers[0].conf_id == max(res.conformers, key=lambda c: c.energy_ev).conf_id
    assert "qm" not in res.timings and "rank" in res.timings


def test_light_results_drop_the_rank_pool():
    res = pipeline.MoleculeResult(input=from_smiles("CCO", id="e"), rank={"level": "gxtb", "conformers": [{}] * 5})
    assert jobs.light_result(res).rank == {"level": "gxtb"} and res.rank["conformers"]


def test_aimnet2_rank_of_an_aimnet2_refinement_reuses_its_energies(monkeypatch):
    cs, pool = _pool(n=3)
    monkeypatch.setattr(engines, "make_calculator", lambda *a, **k: pytest.fail("no single points needed"))
    r = rank.compute("aimnet2", cs.mol, pool, refine_method="aimnet2")
    assert r.reused_refinement and r.energies_ev == {c.conf_id: c.energy_ev for c in pool}


def test_rank_is_recorded_in_the_files(tmp_path, monkeypatch):
    _fake_rank(monkeypatch, lambda pool: {c.conf_id: -c.energy_ev for c in pool})
    res = pipeline.run_molecule(from_smiles(HEXANOL, id="h"), cfg(tmp_path, solvent="water", write_files=True))
    files = export.write_molecule(res, tmp_path / "h", ("sdf", "xyz-all"))
    from rdkit import Chem

    sdf = [m for m in Chem.SDMolSupplier(str(tmp_path / "h" / "h_conformers.sdf"), removeHs=False, sanitize=False)]
    assert sdf[0].GetProp("QB_RankLevel") == "gxtb+cpcm"
    assert float(sdf[0].GetProp("QB_RankEnergy_eV")) == pytest.approx(res.conformers[0].rank_energy_ev, abs=1e-6)
    back = export.read_molecule(tmp_path / "h", "h")
    assert back.rank["level"] == "gxtb+cpcm" and back.rank["pick_changed"] == res.rank["pick_changed"]
    assert [c.rank_energy_ev for c in back.conformers] == [c.rank_energy_ev for c in res.conformers]
    assert any(str(p).endswith("h_conformers.xyz") for p in files)
    assert "rank_energy_eV" in (tmp_path / "h" / "h_conformers.xyz").read_text()


# ------------------------------------------------------------------ resume, CLI, worker
def _fast_cfg(tmp_path, **kw):
    base = dict(method="mmff", max_conformers=2, refine_top=1, xtb_properties=False, threads=1, gpu="off",
                output_dir=str(tmp_path / "runs"), formats=("xyz",))
    base.update(kw)
    return pipeline.BuildConfig(**base)


def test_resume_detects_a_rank_change_and_old_runs_resume_as_refine(tmp_path, have):
    have(gxtb=False, aimnet=False)
    run_dir = tmp_path / "run"
    mols = [from_smiles("CCO", id="ethanol"), from_smiles("CCCO", id="propanol")]
    jobs.run_resumable(mols[:1], _fast_cfg(tmp_path), run_dir)
    with pytest.raises(jobs.ConfigMismatchError) as err:
        jobs.run_resumable(mols, _fast_cfg(tmp_path, rank_method="gxtb+cpcm"), run_dir)
    assert err.value.differences == {"rank_method": ("auto", "gxtb+cpcm")}
    with pytest.raises(jobs.ConfigMismatchError) as err:
        jobs.run_resumable(mols, _fast_cfg(tmp_path, rank_window_kj=20.0), run_dir)
    assert set(err.value.differences) == {"rank_window_kj"}
    # a run folder from before RANK-2 (no rank fields): it ranked at the refinement level
    with sqlite3.connect(run_dir / jobs.DB_NAME) as con:
        config = json.loads(con.execute("SELECT config FROM run").fetchone()[0])
        for k in ("rank_method", "rank_window_kj"):
            config.pop(k)
        con.execute("UPDATE run SET config = ?", (json.dumps(config),))
    assert jobs.stored_science(config)["rank_method"] == "refine"
    assert jobs.stored_config(config, _fast_cfg(tmp_path)).rank_method == "refine"
    with pytest.raises(jobs.ConfigMismatchError) as err:
        jobs.run_resumable(mols, _fast_cfg(tmp_path), run_dir)                      # today's default auto
    assert err.value.differences == {"rank_method": ("refine", "auto")}
    # refine resumes it, and the rank window does not matter without a rank step
    jobs.run_resumable(mols, _fast_cfg(tmp_path, rank_method="refine", rank_window_kj=12.0), run_dir)
    assert jobs.open_run(run_dir).counts["ok"] == 2
    # build --resume takes the run's own settings: refine
    from quantabench import cli

    assert cli.main(["build", "--resume", str(run_dir), "--no-review"]) == 0


def test_cli_rank_options(capsys):
    from quantabench import cli

    args = cli.build_parser().parse_args(["build", "CCO", "--rank", "gxtb", "--rank-window", "20"])
    c = cli.config_from_args(args)
    assert (c.rank_method, c.rank_window_kj) == ("gxtb", 20.0)
    c = cli.config_from_args(cli.build_parser().parse_args(["build", "CCO"]))
    assert (c.rank_method, c.rank_window_kj) == ("auto", 30.0)
    g = cli.build_parser().parse_args(["grid", "build", "CCO", "--rank", "refine"])
    assert cli.config_from_args(g).rank_method == "refine"
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["build", "CCO", "--rank", "dft"])
    capsys.readouterr()
    from quantabench.chem import advisor

    assert advisor.cli_options({"rank_method": "auto", "rank_window_kj": 30.0}) == "--rank auto --rank-window 30"


def test_worker_accepts_the_rank_fields():
    from quantabench.dist import worker

    ok = worker.validate_config({"rank_method": "gxtb+cpcm", "rank_window_kj": 20.0})
    assert ok == {"rank_method": "gxtb+cpcm", "rank_window_kj": 20.0}
    for bad in ({"rank_method": "dft"}, {"rank_window_kj": -1.0}):
        with pytest.raises(worker.RequestError):
            worker.validate_config(bad)


def test_run_metadata_records_the_gxtb_build(monkeypatch):
    from quantabench.chem import telemetry

    info = eng_gxtb.BuildInfo("2.0.1", "linux-x86_64", "ab" * 32, "6.7.1 (x)", "/opt/gxtb/bin/xtb")
    monkeypatch.setattr(eng_gxtb, "build_info", lambda program=None: info)
    d = telemetry.engines_info(pipeline.BuildConfig(rank_method="auto"))
    assert d["programs"]["gxtb"]["sha256"] == "ab" * 32 and d["programs"]["gxtb"]["release"] == "2.0.1"
    assert "gxtb" not in telemetry.engines_info(pipeline.BuildConfig(rank_method="refine"))["programs"]
    assert {"rank_method", "rank_window_kj"} <= set(telemetry.COST_FIELDS)


# ------------------------------------------------------------------ the GUI control
@pytest.mark.skipif(not HAS_DISPLAY, reason="needs a display (run under xvfb-run on Linux)")
def test_gui_rank_page(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from quantabench.gui.main_window import STAGES, MainWindow

    app = QApplication.instance() or QApplication([])
    w = MainWindow(settings_mod.Settings(output_dir=str(tmp_path)))
    keys = [k for k, *_ in STAGES]
    assert keys.index("refine") < keys.index("rank") < keys.index("properties") and "rank" in w.stage_items
    assert w.w_rank_method.currentData() == "auto" and w.w_rank_window.stored() == 30.0
    assert w.w_rank_window.suffix() == " kJ/mol" and "conformer" in w.rank_info.text().lower()
    w._set_combo(w.w_rank_method, "gxtb+cpcm")
    w.w_rank_window.set_stored(20.0)
    c = w.current_config()
    assert (c.rank_method, c.rank_window_kj) == ("gxtb+cpcm", 20.0)
    assert (w.s.build.rank_method, w.s.build.rank_window_kj) == ("gxtb+cpcm", 20.0)
    w.s.display.relative_energy_unit = "kcal/mol"
    w.apply_display()
    assert w.w_rank_window.suffix() == " kcal/mol" and w.current_config().rank_window_kj == 20.0
    heads = [w.conf_table.horizontalHeaderItem(i).text() for i in range(w.conf_table.columnCount())]
    assert heads[-2].startswith("E rank") and heads[-1].startswith("ΔE refine")
    from quantabench.gui import review_panel

    assert review_panel.apply_changes(w, {"rank_method": "refine", "rank_window_kj": 25.0}) == []
    assert w.current_config().rank_method == "refine" and w.current_config().rank_window_kj == 25.0
    w.close()
    app.processEvents()
