"""Implicit solvents: one canonical name per solvent, mapped onto each engine's own keyword.

Engines take the canonical name (``BuildConfig.solvent``; "" = gas phase) and translate it with
``for_engine``. An engine that cannot model a solvent returns ``None`` and the pipeline notes that the
calculation ran in the gas phase.

Models: xtb/CREST ALPB (and GBSA), tblite ALPB/GBSA/CPCM, MOPAC COSMO (dielectric constant),
Psi4 PCM (solvent name for PCMSolver).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Solvent:
    name: str              # canonical, lowercase
    label: str
    epsilon: float         # static dielectric constant at 25 °C
    alpb: str | None       # xtb/tblite ALPB/GBSA keyword (None = not parametrized)
    pcm: str | None        # PCMSolver name used by Psi4 (None = not in its table)


SOLVENTS: dict[str, Solvent] = {s.name: s for s in (
    Solvent("water", "Water", 78.36, "water", "water"),
    Solvent("methanol", "Methanol", 32.61, "methanol", "methanol"),
    Solvent("ethanol", "Ethanol", 24.85, "ethanol", "ethanol"),
    Solvent("octanol", "1-Octanol", 9.86, "octanol", None),
    Solvent("acetonitrile", "Acetonitrile", 35.69, "acetonitrile", "acetonitrile"),
    Solvent("acetone", "Acetone", 20.49, "acetone", "acetone"),
    Solvent("dmso", "DMSO", 46.83, "dmso", "dimethylsulfoxide"),
    Solvent("dmf", "DMF", 37.22, "dmf", "n,n-dimethylformamide"),
    Solvent("thf", "THF", 7.43, "thf", "tetrahydrofuran"),
    Solvent("dichloromethane", "Dichloromethane", 8.93, "ch2cl2", "methylenechloride"),
    Solvent("chloroform", "Chloroform", 4.71, "chcl3", "chloroform"),
    Solvent("toluene", "Toluene", 2.37, "toluene", "toluene"),
    Solvent("benzene", "Benzene", 2.27, "benzene", "benzene"),
    Solvent("hexane", "n-Hexane", 1.88, "hexane", "n-hexane"),
    Solvent("diethylether", "Diethyl ether", 4.24, "ether", "diethylether"),
    Solvent("ethylacetate", "Ethyl acetate", 5.99, "ethylacetate", None),
)}

ALIASES = {"h2o": "water", "meoh": "methanol", "etoh": "ethanol", "1-octanol": "octanol", "mecn": "acetonitrile",
           "ch3cn": "acetonitrile", "ch2cl2": "dichloromethane", "dcm": "dichloromethane", "chcl3": "chloroform",
           "ether": "diethylether", "n-hexane": "hexane", "etoac": "ethylacetate"}


def canonical(name: str | None) -> str:
    """'' for gas phase; raises ValueError for an unknown solvent."""
    key = (name or "").strip().lower()
    if key in ("", "none", "gas", "vacuum"):
        return ""
    key = ALIASES.get(key, key)
    if key not in SOLVENTS:
        raise ValueError(f"unknown solvent {name!r}; choose one of {', '.join(SOLVENTS)}")
    return key


def get(name: str | None) -> Solvent | None:
    key = canonical(name)
    return SOLVENTS[key] if key else None


def for_engine(name: str | None, model: str) -> str | float | None:
    """Engine keyword for ``name``: model 'alpb'/'gbsa' -> xtb name, 'cosmo'/'cpcm' -> epsilon,
    'pcm' -> PCMSolver name. ``None`` when the solvent has no parameters for that model."""
    s = get(name)
    if s is None:
        return None
    model = model.lower()
    if model in ("alpb", "gbsa"):
        return s.alpb
    if model in ("cosmo", "cpcm", "eps"):
        return s.epsilon
    if model == "pcm":
        return s.pcm
    raise ValueError(f"unknown solvation model {model!r}")
