"""Deterministic weighted splitting of molecule ids between targets (machines or lanes).

``weighted_index(mol_id, weights)`` maps the SHA-256 of the id to [0, 1) and picks the target whose
cumulative-weight bin contains it. With equal weights over n targets, the shares match ``jobs.chunk_index``
statistically (not id by id: chunk_index uses a modulo). Every machine computes the same answer, and adding
molecules to a list never moves the existing ones.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence


def unit_interval(mol_id: str) -> float:
    digest = hashlib.sha256(mol_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2.0**64


def weighted_index(mol_id: str, weights: Sequence[float]) -> int:
    w = [max(0.0, float(x)) for x in weights]
    total = sum(w)
    if not w or total <= 0:
        raise ValueError("weights must contain at least one positive value")
    u = unit_interval(mol_id) * total
    acc = 0.0
    for i, x in enumerate(w):
        acc += x
        if u < acc:
            return i
    return max(i for i, x in enumerate(w) if x > 0)


def split(ids: Sequence[str], weights: dict[str, float]) -> dict[str, list[str]]:
    """{target: [ids]} for every target in ``weights`` (empty lists included), in input order."""
    names = list(weights)
    out: dict[str, list[str]] = {n: [] for n in names}
    for mol_id in ids:
        out[names[weighted_index(mol_id, [weights[n] for n in names])]].append(mol_id)
    return out
