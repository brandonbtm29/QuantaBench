"""Writers for single molecules, conformer ensembles and batch summaries."""

from __future__ import annotations

import contextlib
import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from ase import Atoms
from ase.io import write as ase_write
from rdkit import Chem

if TYPE_CHECKING:
    from .pipeline import MoleculeResult

FORMATS = {
    "xyz": "Best conformer, XYZ",
    "xyz-all": "All conformers, multi-frame XYZ",
    "extxyz": "Best conformer, extended XYZ with charges",
    "sdf": "All conformers, SDF with energies and weights",
    "pdb": "Best conformer, PDB",
    "mol": "Best conformer, MDL Molfile",
    "lammps": "Best conformer, LAMMPS data file (atom_style charge, real units, padded box)",
    "spectrum": "IR spectrum and normal-mode table, CSV (when frequencies were computed)",
}
DEFAULT_FORMATS = ("xyz", "sdf", "extxyz")


def mol_to_atoms(mol: Chem.Mol, conf_id: int = -1) -> Atoms:
    conf = mol.GetConformer(conf_id)
    return Atoms([a.GetSymbol() for a in mol.GetAtoms()], positions=np.array(conf.GetPositions()))


def _write_text_format(path: Path, images, **kwargs) -> None:
    """ase.io.write into a file opened as UTF-8. Given a path, ASE opens it in the locale's encoding (cp1252 on
    Windows), and a name such as "β-Tocopherol" in an XYZ comment then fails the export."""
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        ase_write(fh, images, **kwargs)


def write_lammps_data(atoms: Atoms, path: Path, charges=None, padding: float = 10.0) -> None:
    a = atoms.copy()
    a.center(vacuum=padding)
    a.pbc = True
    if charges is not None:
        a.set_initial_charges(np.asarray(charges, dtype=float))
    _write_text_format(path, a, format="lammps-data", atom_style="charge", units="real", masses=True,
                       specorder=None)


def write_molecule(result: MoleculeResult, out_dir: Path, formats=DEFAULT_FORMATS) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = result.input.id
    written: list[Path] = []
    mol = result.mol
    if mol is None or not result.conformers:
        return written
    best = result.conformers[0]
    best_atoms = mol_to_atoms(mol, best.conf_id)
    comment = (f"{result.input.name or stem} | {result.method} | E={best.energy_ev:.6f} eV | "
               f"charge={result.charge} | smiles={result.smiles}")
    for fmt in formats:
        if fmt == "xyz":
            p = out_dir / f"{stem}.xyz"
            _write_text_format(p, best_atoms, format="xyz", comment=comment)
        elif fmt == "xyz-all":
            p = out_dir / f"{stem}_conformers.xyz"
            frames = []
            for c in result.conformers:
                at = mol_to_atoms(mol, c.conf_id)
                at.info.update(rank=c.rank, energy_eV=c.energy_ev, rel_kcal=c.rel_kcal, weight=c.weight)
                if c.rank_energy_ev is not None:     # the rank step's energy (chem.rank); rel_kcal follows it
                    at.info.update(rank_energy_eV=c.rank_energy_ev, refine_rel_kcal=c.refine_rel_kcal)
                frames.append(at)
            _write_text_format(p, frames, format="extxyz")
        elif fmt == "extxyz":
            p = out_dir / f"{stem}.extxyz"
            at = best_atoms.copy()
            if best.charges is not None:
                at.set_initial_charges(np.asarray(best.charges))
            at.info.update(energy_eV=best.energy_ev, method=result.method, total_charge=result.charge,
                           smiles=result.smiles)
            _write_text_format(p, at, format="extxyz")
        elif fmt == "sdf":
            p = out_dir / f"{stem}_conformers.sdf"
            w = Chem.SDWriter(str(p))
            for c in result.conformers:
                mol.SetProp("_Name", f"{stem} conf {c.rank}")
                ranked = c.rank_energy_ev is not None     # QB_RelEnergy_kcal and the weight follow the rank level
                for key, val in (("QB_Rank", c.rank), ("QB_Method", result.method), ("QB_Energy_eV", c.energy_ev),
                                 ("QB_RelEnergy_kcal", c.rel_kcal), ("QB_BoltzmannWeight", c.weight),
                                 ("QB_Converged", int(c.converged)),
                                 *((("QB_RankLevel", (result.rank or {}).get("level", "")),
                                    ("QB_RankEnergy_eV", c.rank_energy_ev),
                                    ("QB_RefineRelEnergy_kcal", c.refine_rel_kcal)) if ranked else ())):
                    mol.SetProp(key, f"{val:.6f}" if isinstance(val, float) else str(val))
                if not ranked:
                    for key in ("QB_RankLevel", "QB_RankEnergy_eV", "QB_RefineRelEnergy_kcal"):
                        if mol.HasProp(key):
                            mol.ClearProp(key)
                w.write(mol, confId=c.conf_id)
            w.close()
        elif fmt == "pdb":
            p = out_dir / f"{stem}.pdb"
            Chem.MolToPDBFile(mol, str(p), confId=best.conf_id)
        elif fmt == "mol":
            p = out_dir / f"{stem}.mol"
            Chem.MolToMolFile(mol, str(p), confId=best.conf_id)
        elif fmt == "lammps":
            p = out_dir / f"{stem}.lammps.data"
            write_lammps_data(best_atoms, p, best.charges)
        elif fmt == "spectrum":
            if not result.vibrations:
                continue
            written += write_vibrations(result.vibrations, out_dir, stem)
            continue
        else:
            raise ValueError(f"Unknown export format {fmt!r} (known: {', '.join(FORMATS)})")
        written.append(p)
    if result.qm:
        written += write_qm(result.qm, out_dir, stem, spectrum="spectrum" in formats)
    summary = out_dir / f"{stem}_summary.json"
    summary.write_text(json.dumps(result.to_dict(), indent=2, default=str) + "\n", encoding="utf-8")
    written.append(summary)
    return written


def read_molecule(folder: str | Path, stem: str, inp=None) -> MoleculeResult:
    """Read back what ``write_molecule`` wrote into ``folder``: the result's fields from ``<stem>_summary.json``
    and the molecule with every kept conformer from ``<stem>_conformers.sdf`` (without the SDF, only the best
    conformer, from ``<stem>.mol`` or ``<stem>.xyz``). ``inp`` (``MoleculeInput``) replaces the input record of
    the summary, which has no structure-file molecule. Files that moved with the folder (another machine, a
    synced copy) are pointed at their copies here."""
    from dataclasses import fields

    from .inputs import MoleculeInput
    from .pipeline import ConformerResult, MoleculeResult

    folder = Path(folder)
    d = json.loads((folder / f"{stem}_summary.json").read_text(encoding="utf-8"))
    if inp is None:
        known = {f.name for f in fields(MoleculeInput)} - {"mol"}
        inp = MoleculeInput(**{k: v for k, v in (d.get("input") or {}).items() if k in known})
    known = {f.name for f in fields(MoleculeResult)} - {"input", "mol", "conformers"}
    res = MoleculeResult(input=inp, **{k: v for k, v in d.items() if k in known})
    conf_known = {f.name for f in fields(ConformerResult)}
    conformers = [ConformerResult(**{k: v for k, v in c.items() if k in conf_known}) for c in d.get("conformers") or []]
    res.mol, res.conformers = _read_conformers(folder, stem, conformers, res.charge)
    # the summary is written before the file list is known: the folder's own listing is the list
    res.files = [_moved(p, folder) for p in res.files] or sorted(
        str(p) for p in folder.iterdir() if p.is_file() and p.name.startswith(stem))
    if res.qm and isinstance(res.qm.get("files"), dict):
        res.qm["files"] = {k: _moved(p, folder) for k, p in res.qm["files"].items()}
    return res


def _moved(path: str, folder: Path) -> str:
    """``path`` if it exists, else the file of that name in ``folder`` (a run folder that moved), else ``path``."""
    from pathlib import PureWindowsPath

    if not path or Path(path).exists():
        return path
    here = folder / (PureWindowsPath(path).name if "\\" in path else Path(path).name)
    return str(here) if here.exists() else path


def _read_conformers(folder: Path, stem: str, conformers: list, charge: int) -> tuple[Chem.Mol | None, list]:
    """(molecule with one RDKit conformer per kept ``ConformerResult``, the conformers that have coordinates)."""
    sdf = folder / f"{stem}_conformers.sdf"
    if sdf.is_file() and conformers:
        by_rank = {}
        for m in Chem.SDMolSupplier(str(sdf), removeHs=False, sanitize=False):
            if m is not None and m.HasProp("QB_Rank"):
                by_rank.setdefault(int(m.GetProp("QB_Rank")), m)
        kept = [c for c in conformers if c.rank in by_rank]
        if kept:
            mol = Chem.Mol(by_rank[kept[0].rank])
            mol.RemoveAllConformers()
            for c in kept:
                conf = Chem.Conformer(by_rank[c.rank].GetConformer())
                conf.SetId(c.conf_id)
                mol.AddConformer(conf, assignId=False)
            return _sanitized(mol), kept
    if not conformers:
        return None, []
    best = conformers[0]
    mol = None
    if (folder / f"{stem}.mol").is_file():
        mol = Chem.MolFromMolFile(str(folder / f"{stem}.mol"), removeHs=False, sanitize=False)
    elif (folder / f"{stem}.xyz").is_file():
        from rdkit.Chem import rdDetermineBonds

        mol = Chem.MolFromXYZFile(str(folder / f"{stem}.xyz"))
        if mol is not None:
            try:
                rdDetermineBonds.DetermineBonds(mol, charge=charge)
            except ValueError:
                rdDetermineBonds.DetermineConnectivity(mol)
    if mol is None or mol.GetNumConformers() == 0:
        return None, []
    mol.GetConformer().SetId(best.conf_id)
    return _sanitized(mol), [best]


def _sanitized(mol: Chem.Mol) -> Chem.Mol:
    try:
        Chem.SanitizeMol(mol)
    except Exception:  # noqa: BLE001 - keep what was written; ring info is enough for the views
        mol.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(mol)
    return mol


def write_vibrations(vib: dict, out_dir: Path, stem: str, fwhm: float = 20.0) -> list[Path]:
    """``<stem>_ir.csv`` (Lorentzian-broadened spectrum, unscaled frequencies) and ``<stem>_modes.csv``
    from a ``VibrationalResult.to_dict()``."""
    from types import SimpleNamespace

    from . import vibrations

    spec = vibrations.ir_spectrum(vib["frequencies_cm1"], vib.get("ir_intensities_km_mol"), fwhm=fwhm)
    paths = [vibrations.write_spectrum_csv(out_dir / f"{stem}_ir.csv", spec)]
    fields = ("method", "frequencies_cm1", "ir_intensities_km_mol", "reduced_masses_amu", "force_constants_mdyn_A")
    paths.append(vibrations.write_modes_csv(out_dir / f"{stem}_modes.csv",
                                            SimpleNamespace(**{k: vib.get(k) for k in fields}),
                                            scale=vib.get("scale_factor", 1.0)))
    if vib.get("positions_A") and vib.get("symbols"):
        # the tightly re-optimized geometry the frequencies belong to (the other files hold the ranked one)
        e = (vib.get("thermo") or {}).get("potential_energy_ev")
        lines = [str(len(vib["symbols"])),
                 f"{stem}: frequency geometry ({vib.get('method', '')})" + (f" E = {e:.6f} eV" if e is not None else "")]
        lines += [f"{s} {x:.6f} {y:.6f} {z:.6f}" for s, (x, y, z) in zip(vib["symbols"], vib["positions_A"])]
        p = out_dir / f"{stem}_freq.xyz"
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        paths.append(p)
    return paths


def write_qm(qm: dict, out_dir: Path, stem: str, spectrum: bool = False) -> list[Path]:
    """Copy the QM stage's files (cubes, logs; ``qm["files"]`` label -> path) next to the other results as
    ``<stem>_qm_<label><suffix>`` and point ``qm["files"]`` at the copies. Plus the QM IR spectrum."""
    import shutil

    written: list[Path] = []
    files = qm.get("files") or {}
    for label, src in list(files.items()):
        src = Path(src)
        if not src.is_file():
            continue
        dst = out_dir / f"{stem}_qm_{label}{src.suffix}"
        if src.resolve() != dst.resolve():
            from . import external

            if external.scratch_root().resolve() in src.resolve().parents:   # QM scratch: move, don't pile up
                shutil.move(str(src), dst)
                with contextlib.suppress(OSError):
                    src.parent.rmdir()          # the emptied qm_<id>_* folder
            else:
                shutil.copy2(src, dst)
        files[label] = str(dst)
        written.append(dst)
    if spectrum and qm.get("vibrations"):
        written += write_vibrations(qm["vibrations"], out_dir, f"{stem}_qm")
    prof = (qm.get("surface") or {}).get("esp_profile")
    if prof and prof.get("area_A2"):            # the ESP distribution on the density surface (chem.surfaces)
        from . import surfaces

        path = out_dir / f"{stem}_esp_profile.csv"
        path.write_text(surfaces.profile_csv(prof), encoding="utf-8")
        written.append(path)
    return written


#: summary.csv columns repeated at the front of descriptors.csv, so that table joins by structure on its own
KEY_COLUMNS = ("inchikey", "canonical_smiles", "input_inchikey")


def descriptor_row(result) -> dict:
    """One descriptors.csv row: the id, the structure keys (``KEY_COLUMNS``, when the summary row has them;
    results from before they existed have none), then the descriptors."""
    row = result.summary_row()
    keys = {k: row[k] for k in KEY_COLUMNS if k in row}
    return {"id": result.input.id, **keys, **result.descriptors}


def write_batch_tables(results: list[MoleculeResult], run_dir: Path) -> list[Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = [r.summary_row() for r in results]
    desc_rows = [descriptor_row(r) for r in results if r.descriptors]
    out = []
    for name, data in (("summary.csv", rows), ("descriptors.csv", desc_rows)):
        if not data:
            continue
        keys: list[str] = []
        for row in data:
            keys += [k for k in row if k not in keys]
        path = run_dir / name
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(data)
        out.append(path)
    return out
