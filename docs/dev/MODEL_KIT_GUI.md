# Model kit build mode (UI-7, GUI part)

First written 2026-11-07 after the core (`chem/edit.py`, `chem/fragments.py`) as the plan for this work. Rewritten
2026-11-22 to describe what is built (branch `feat/ui-7-build-mode`: `gui/model_kit.py`, Build mode in
`gui/viewer3d.py`, `gui/keymap.py`; tests `tests/test_gui_model_kit.py`, `tests/test_edit.py`).

The kit is QuantaBench's own design. Other model builders were a reference for *what* a 3D builder needs to do (place
atoms on free valences, replace atoms, groups and rings, undo, minimize). The layout, the names, the order of choices
and the drawings are our own, and none of their tab names, lists or icons are copied.

## The 3D view's part

**The view stays a view** (`gui/viewer3d.py`, `MoleculeViewer3D`). The controller (`gui/model_kit.py`) owns the
model and calls `show_rdkit(model.mol, reset_camera=False)` after each change. The view keeps the selection when the
atom count does not change.

- **Mode.** `MODES` has `"build"` next to rotate, select and move: the action `view.mode_build` in the View toolbar.
  A drag still turns the camera; only clicks build. Ctrl+click selects, as in Rotate mode.
- **Clicks.** `_click` emits `buildClicked(kind, index, point, shift)`:
  - `kind` is `"stub"`, `"atom"` or `"space"`;
  - `index` is the stub row or the atom index;
  - `point` is the stub tip, the atom centre or the point on the focal plane under the cursor;
  - `shift` says whether Shift was held.

  When a stub tip and an atom are both under the cursor, the one nearer to the viewer wins. A stub that points away
  from the viewer projects onto its own atom, so "stubs first" would steal clicks meant to replace that atom.
- **Stubs.** In Build mode, hydrogens are not drawn as atoms. Every free valence the controller passes
  (`set_stubs(atoms, directions, orders)`) is drawn as a cylinder 0.6 Å long and 0.08 Å thick, starting at the atom's
  drawn surface. One `vtkGlyph3DMapper` actor, `"build_stubs"`, is built like the bonds.
  - Colour shows the bond order a free valence takes: teal single, magenta double, orange triple. With the atom
    labels, this is how a double free valence shows.
  - `pick_stub` ray-casts against the stub tips (radius 0.3 Å), and the rows keep the order of `all_free_valences()`.
- **First atom.** After the first atom of a new model, the view is reset to span 10 Å around it
  (`reset_camera(min_extent=...)`).

## Choosing what to build

The dock asks in this order, top to bottom:

1. **Element.**
   - A row of tiles for the common elements (`edit.COMMON_ELEMENTS`: H C N O F P S Cl Br I B Si).
   - Below it, the collapsible section **Inorganic and other elements**: a compact periodic table of periods 1-6
     (no lanthanides after La). Tiles are coloured by kind: nonmetal, halogen, metalloid, s-block metal, d-block
     metal and p-block metal. Noble gases are greyed out.
   - A line under the tiles names the element and says whether it is a coordination centre and whether Minimize
     will use UFF.
2. **Geometry.** These are the shapes the element can take at the chosen charge (`edit.geometry_options`). Each one
   is a button with a small schematic drawn by QuantaBench: a centre, a dot per bond site, a dashed ring per lone
   pair, and the faint edges of the polyhedron. Buttons are named by shape, with the hybridization in the tooltip.
   - **Main-group elements** follow VSEPR: tetrahedral, pyramidal, bent, trigonal planar, linear or terminal.
     - Each option is a hybridization (sp3, sp2, sp) that leaves at least one bond and no unpaired electron.
     - S, Se, P and As also offer their higher valences, as geometries of their own: "tetrahedral (6)" is S(VI).
     - Named types keep their short names (`F`, `S(VI)`, `O-`).
   - **Metals** (`edit.METALS`: the s- and d-block, Al, Ga, In, Tl, Sn, Pb, Bi) are coordination centres. Their
     options are terminal, linear, trigonal, tetrahedral, square planar, trigonal bipyramidal and octahedral, each
     offered only if RDKit accepts that many bonds at that charge (neutral Al takes 3, Al⁻ takes 4).
3. **Charge** (−3 to +3) and **Bond to it** (Auto, Single, Double, Triple). The geometries follow the charge (N⁺ is
   tetrahedral). The bond order applies when attaching:
   - a carbonyl O is O, terminal, Double;
   - a nitrile N is N, terminal, Triple;
   - Auto gives a double bond between two atoms that each still have a π bond to make.
4. **Or a fragment**, from four menus over `chem.fragments`:
   - Functional groups;
   - Ring systems;
   - Small substituents;
   - PFAS pieces, with an *n* box for `(CF2)n` and `CnF2n+1`.

   A **SMILES or InChI** box takes any fragment: `[*:1]` marks the attachment, otherwise the first atom bonds.

`choose(...)` also takes a type name (`"C sp2"`, `"=O"`, `"Fe octahedral"`, `"Zn2+ tetrahedral"`;
`edit.atom_type`) and sets the pickers to match. Tests and scripts use it.

## What a click does

| Click | Current choice | Call |
| --- | --- | --- |
| stub | atom type | `model.attach_atom(valence, type)` |
| stub | fragment | `model.attach_fragment(valence, fragment)` |
| atom | atom type | `model.replace_atom(atom, type)` |
| empty space, empty model (or Shift+click) | atom type or fragment | `model.add_atom(type, point)` / `add_fragment(fragment, point)` |

The edit tools act on the selection (Ctrl+click in Build mode, or a click in Rotate mode):
- Delete: `delete_atoms`, or `delete_fragment` for the whole piece.
- Bond order: `set_bond_order`, cycling 1 → 2 → 3.
- Charge + and −: `set_charge`.
- Make bond and Break bond, on two atoms.
- The value box: sets the measured distance, angle or dihedral (`set_distance`, `set_angle`, `set_dihedral`).
- Add H: `fill_hydrogens`.
- Minimize: `minimize(frozen=...)`, with *Freeze selected*.

A Move-mode drag during an edit is one undoable step (`set_positions`). An `EditError` goes to the dock's footer and
the status bar, and the model does not change. *Relax after each edit* sets `model.auto_relax`, so the local relax
runs inside each command and Undo takes it back too.

## Metals and the force field

- **Sites.** A metal centre has no lone pairs, and each site is a single bond. Open sites stay empty: no hydrogens are
  put on a metal, and `problems()` does not report its open sites. Sites follow the existing bonds: the ideal set of
  the geometry (`edit.IDEAL_SITES`) is turned to fit them best (Kabsch, each bond matched to its own site).
- **Why UFF.** MMFF94 has no parameters for bonded metals, so `Minimize` and the relax use UFF for them. The dock's
  *Force field* line says which force field applies and why (`model.forcefield()`).
- **Typing the metal.** RDKit's UFF types a metal by its hybridization, so the coordination geometry sets it
  (octahedral SP3D2, square planar SP2D, and so on). When UFF has no type for that geometry (square-planar Cu,
  five-coordinate Fe), the metal's other types are tried in turn. `MinimizeResult.note`, and the dock, say which
  one was used.
- **Trans ligands.** RDKit's UFF has a special angle term only for octahedral centres. For a square-planar centre,
  the general term has a maximum at 180°, so mixed ligands folded it toward a tetrahedron. Under UFF, trans ligands
  of square-planar, trigonal-bipyramidal and octahedral metals are held at 179-180°.
- **Bonds to metals are single.** For dative ligands, use formal charges: cisplatin is Pt²⁻ with two Cl and two N⁺
  (NH₃⁺), as in its usual SMILES.

## Undo

`UndoAdaptor` is the `model.history`:
1. `push(command)` calls `command.apply(model)` first. An `EditError` there means nothing is pushed.
2. It then pushes a `QUndoCommand` whose first `redo()` does nothing, because `QUndoStack.push` calls `redo()` at
   once and the command has already run. Later `redo()` calls `command.apply(model)`, which replays the stored
   result, so the coordinates come back exactly. `undo()` calls `command.undo(model)`.

Edit > Undo / Redo are the actions `edit.undo` and `edit.redo`.

## An edit and the molecule list

- **Starting an edit.** Entering Build mode opens an edit of the molecule selected in the list (its shown conformer
  when it is built), or of an empty model when nothing is selected. **New** and **Edit selected** in the dock do the
  same on demand.
- **Who owns the views.** While an edit is open it owns the 3D and 2D views, as the Surface Builder does:
  `MainWindow.kit_owns_view()` guards `_on_molecule_selected`, `_show_conformer`, `_preview_input` and `clear_all`.
- **Ending an edit.**
  - **Add to list** hands `model.to_input()` to `win.add_inputs`, with the 3D coordinates and source "model kit".
  - **Replace <id>** updates the molecule the edit started from.
  - **Close** drops the edit, asking first when there are changes.

## Shortcuts

The kit's actions are named, with their default keys in one place (`gui.keymap.SPECS`), for the rebinding page
(UI-9):

| Action | Default keys |
| --- | --- |
| `view.mode_rotate`, `view.mode_select`, `view.mode_move`, `view.mode_build` | Ctrl+1 to Ctrl+4, in toolbar order |
| `edit.undo` | the platform's Undo key |
| `edit.redo` | the platform's Redo key, plus Ctrl+Y |
| `build.model_kit` | none |
| `kit.minimize` | none (Avogadro 2's keymap: Ctrl+Alt+O) |

None of these defaults copies another program's. Settings > Controls rebinds them (docs/CONTROLS.md).

## Tests

- **GUI tests** (`tests/test_gui_model_kit.py`) drive the kit through its handlers (`click_stub`, `click_atom`,
  `click_space`, `choose`, `choose_element`, `choose_geometry`, the tools).
  - One test goes through the view's VTK-free mouse methods to a projected stub tip, in perspective and orthographic.
  - The handlers do not need a drawing 3D view. Only the checks of drawn stubs skip where VTK cannot draw, such as
    Windows with Qt's offscreen platform.
- **Edit model tests** (`tests/test_edit.py`) cover:
  - the geometry options;
  - metal centres built and minimized with UFF (octahedral, tetrahedral, linear, square planar, trigonal);
  - a mixed-ligand square-planar Pt staying planar;
  - the UFF fallback type.
- **Screenshots** in `docs/dev/screenshots/UI-7/` come from `scripts/model_kit_smoke.py`.

## Left (BACKLOG UI-7)

- The Clipboard button.
- Peptide and nucleotide builders.
- Sharing the fragment list with the UI-6 sketcher.
- Dragging a dihedral in the view (today the value is typed).
- A hover highlight on stubs.
- A Delete key in the 3D view.
- Hand checks on Windows and macOS.
