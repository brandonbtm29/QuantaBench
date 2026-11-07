"""The 2D sketcher's hand-off (UI-6): MOL blocks to and from the Sketch dock, checked with RDKit.

The Sketch dock (``gui.sketcher_dock``, Schrödinger's Sketcher) exports the drawing as an MDL MOL V3000 block;
``input_from_molblock`` reads it with RDKit, reports structure problems (valence, aromatic rings that cannot be
kekulized, query atoms, R-groups) in plain words before anything is added, and returns a
``chem.inputs.MoleculeInput`` that goes the same way as a SMILES typed into the Molecules box (the pipeline embeds
it in 3D as usual). ``molblock_for_input`` goes the other way for "Edit selected": the molecule's input as a MOL
V3000 block with 2D coordinates (stereo kept as wedges).

``FRAGMENTS``: the PFAS pieces of the dock's Insert fragment menu, as SMILES. Each is a capped piece whose
attachment atom carries a hydrogen, so drawing a bond to it in the sketcher joins it without a valence error.
(The 3D model kit keeps its own list for now, ``chem.fragments``; the two will be unified.)
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field

from rdkit import Chem, rdBase
from rdkit.Chem import rdDepictor, rdMolDescriptors

from .inputs import MoleculeInput, safe_id

SOURCE = "sketch"

#: (menu label, SMILES) of the Insert fragment menu; None = the (CF2)n chain, which asks for n (``cf2_chain``)
FRAGMENTS: tuple[tuple[str, str | None], ...] = (
    ("CF3 (trifluoromethyl)", "C(F)(F)F"),
    ("(CF2)n chain...", None),
    ("Carboxylic acid (COOH)", "C(=O)O"),
    ("Carboxylate (COO-)", "C(=O)[O-]"),
    ("Sulfonic acid (SO3H)", "S(=O)(=O)O"),
    ("Sulfonate (SO3-)", "S(=O)(=O)[O-]"),
    ("Sulfonamide (SO2NH2)", "S(=O)(=O)N"),
    ("Fluorotelomer linker (CH2CH2)", "CC"),
    ("Ether oxygen (O)", "O"),
    ("Phosphate (OPO3H2)", "OP(=O)(O)O"),
)
MAX_CHAIN = 20


class SketchError(ValueError):
    """The drawing cannot be handed to the pipeline; the message says why, in words a user can act on."""


@dataclass
class SketchResult:
    input: MoleculeInput
    notes: list[str] = field(default_factory=list)   # things worth a log line (several pieces, radicals)
    formula: str = ""


def cf2_chain(n: int) -> str:
    """SMILES of a perfluoroalkyl chain piece of ``n`` CF2 units (both ends carry an H to bond to)."""
    if not 1 <= int(n) <= MAX_CHAIN:
        raise ValueError(f"a (CF2)n chain needs 1 <= n <= {MAX_CHAIN} (got {n})")
    return "C(F)(F)" * int(n)


def _atom_label(atom: Chem.Atom) -> str:
    q = atom.GetFormalCharge()
    return atom.GetSymbol() + ("" if q == 0 else ("+" if q > 0 else "-") + (str(abs(q)) if abs(q) > 1 else ""))


def _problem_text(mol: Chem.Mol, problem) -> str:
    kind = problem.GetType()
    if kind == "AtomValenceException":
        atom = mol.GetAtomWithIdx(problem.GetAtomIdx())
        bonds = sum(b.GetBondTypeAsDouble() for b in atom.GetBonds()) + atom.GetNumExplicitHs()
        label = _atom_label(atom)
        return f"atom {atom.GetIdx() + 1} ({label}) has {bonds:g} bonds, more than {label} allows"
    if kind in ("KekulizeException", "AtomKekulizeException"):
        idx = list(problem.GetAtomIndices()) if kind == "KekulizeException" else [problem.GetAtomIdx()]
        atoms = ", ".join(f"{i + 1} ({mol.GetAtomWithIdx(i).GetSymbol()})" for i in idx)
        return (f"atoms {atoms}: this aromatic ring cannot be written with alternating single and double bonds "
                "(an aromatic N may need its H, or a bond order is wrong)")
    return problem.Message()


def check_molblock(block: str) -> Chem.Mol:
    """The drawing as an RDKit molecule (sanitized, hydrogens implicit), or ``SketchError`` with every problem.
    Atom numbers in the messages count from 1 in drawing order."""
    if not block or not block.strip():
        raise SketchError("The sketcher returned no structure.")
    with rdBase.BlockLogs():
        raw = Chem.MolFromMolBlock(block, sanitize=False, removeHs=False)
        if raw is None:
            raise SketchError("RDKit could not read the drawing (MOL block). Try Clean Up in the sketcher, or redraw "
                              "the part that was changed last.")
        if raw.GetNumAtoms() == 0:
            raise SketchError("The sketch is empty: draw a molecule first.")
        queries = [a for a in raw.GetAtoms() if a.GetAtomicNum() == 0 or a.HasQuery()]
        if queries:
            atoms = ", ".join(f"{a.GetIdx() + 1} ({a.GetSymbol()})" for a in queries[:8])
            raise SketchError(f"The drawing has query atoms, R-groups or attachment points ({atoms}). QuantaBench "
                              "builds real molecules only: replace them with elements.")
        if any(b.HasQuery() for b in raw.GetBonds()):
            raise SketchError("The drawing has query bonds (any, single or double, ...). Use single, double, triple "
                              "or aromatic bonds.")
        problems = Chem.DetectChemistryProblems(raw)
        if problems:
            lines = [_problem_text(raw, p) for p in problems]
            raise SketchError("The drawing has a structure problem:\n- " + "\n- ".join(lines)
                              + "\n(Atoms are numbered from 1 in the order they were drawn.)")
        mol = Chem.MolFromMolBlock(block)          # sanitized, explicit H removed (stereo-defining ones kept)
    if mol is None:
        raise SketchError("RDKit could not sanitize the drawing.")
    return mol


def input_from_molblock(block: str, id: str | None = None, name: str = "") -> SketchResult:
    """A ``MoleculeInput`` for the drawing in ``block`` (MOL V2000/V3000), like a SMILES typed into the Molecules
    box: isomeric canonical SMILES (charges, isotopes, stereo and radicals kept), source "sketch". The id is
    ``id`` (Replace selected keeps the molecule's id) or ``sketch_<formula>``; the list makes it unique."""
    mol = check_molblock(block)
    with rdBase.BlockLogs():
        smiles = Chem.MolToSmiles(mol)
        if Chem.MolFromSmiles(smiles) is None:          # never expected: RDKit's own SMILES of a sanitized mol
            raise SketchError(f"RDKit wrote a SMILES it cannot read back ({smiles!r}); please report this drawing.")
    formula = rdMolDescriptors.CalcMolFormula(mol)
    notes = []
    pieces = len(Chem.GetMolFrags(mol))
    if pieces > 1:
        notes.append(f"the drawing has {pieces} separate pieces ({smiles}); the pipeline keeps the largest one "
                     "unless 'Keep largest fragment' is off")
    radicals = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
    if radicals:
        notes.append(f"{radicals} unpaired electron(s) drawn: the spin multiplicity follows from them")
    inp = MoleculeInput(id=safe_id(id or f"sketch_{formula}", fallback="sketch"), smiles=smiles, name=name,
                        source=SOURCE)
    return SketchResult(inp, notes, formula)


def molblock_for_mol(mol: Chem.Mol) -> str:
    """MOL V3000 block of ``mol`` with fresh 2D coordinates and implicit hydrogens (stereo kept: chiral centers as
    wedges, double bonds from the coordinates). 3D structures (SD files) take their stereo from the 3D
    coordinates first."""
    mol = Chem.Mol(mol)
    with rdBase.BlockLogs():
        if mol.GetNumConformers() and mol.GetConformer().Is3D():
            Chem.AssignStereochemistryFrom3D(mol)
        with contextlib.suppress(Exception):      # an H pattern RDKit cannot remove: keep them, the sketcher shows them
            mol = Chem.RemoveHs(mol)
        mol.RemoveAllConformers()
        rdDepictor.Compute2DCoords(mol)
        return Chem.MolToV3KMolBlock(mol)


def molblock_for_input(inp: MoleculeInput) -> str:
    """The input of a listed molecule (its SMILES, or the structure read from its file) for "Edit selected"."""
    try:
        mol = inp.to_mol()
    except ValueError as exc:
        raise SketchError(str(exc)) from exc
    return molblock_for_mol(mol)
