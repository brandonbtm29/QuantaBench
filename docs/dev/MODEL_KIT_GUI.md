# Model kit build mode (UI-7, GUI part): design note

Written 2026-11-07, after the core (`chem/edit.py`, `chem/fragments.py`). Not implemented yet. This is the
starting point for the task that adds the build mode to the 3D view.

## What exists

**The 3D view** (`gui/viewer3d.py`, `MoleculeViewer3D`):
- `MODES = ("rotate", "select", "move")`, set with `set_mode`. The main window's `MODE_ACTIONS`
  (Ctrl+1 to Ctrl+3) and `set_view_mode` drive it.
- The mouse logic is VTK-free and testable without a display: `_mouse_press`, `_mouse_move`,
  `_mouse_release` and `_click` get display coordinates from the VTK observers in `_install_interaction`.
  A press returns True when the view consumes it (the camera then does not move).
- A press and release closer than `CLICK_TOLERANCE_PX` is a click. In rotate mode a click goes to
  `_click`, which selects through `SelectionModel`.
- Picking: `pick_atom(x, y)` casts `display_ray(x, y)` and calls `ray_sphere_pick` on the drawn spheres
  (`_radii[_vis]`; hidden hydrogens are not pickable).
- Drawing: `show_rdkit` / `show_atoms` rebuild the scene (`_rebuild`); `set_positions` only moves atoms
  (`_refresh_geometry`). Atoms and bonds are `vtkGlyph3DMapper` actors (`_glyph_actor`, per-point
  `rgb`); bonds are unit cylinders scaled and oriented per point (`_bond_arrays`). Actors added with
  `_add(name, actor)` belong to the scene and go on rebuild.
- Moves emit `coordinatesChanged(positions)`. Two to four selected atoms show a distance, angle or
  dihedral (`selection.measure`).

**The edit model** (`chem/edit.py`): `EditableMolecule` with `free_valences`, `all_free_valences`,
`valence_of(h)`, the edit methods (each runs a `Command` through `model.history`), `problems()`,
`to_input()`. `model.history` only needs `push(command)`, so a QUndoStack adaptor can replace it.

## Build mode in the view

- Add `"build"` to `MODES` and a fourth mode action (Ctrl+4, "Build"). Drag still rotates the camera, as
  in Spartan; only clicks build.
- In `_click`, when `mode == "build"`, do not select. Emit a new signal instead:
  `buildClicked(kind, index, point)`, where `kind` is `"stub"`, `"atom"` or `"space"`, `index` is the stub
  or atom index, and `point` is the world point on the focal plane (`display_to_world` at the focal point's
  depth), for placing a first atom in empty space.
- Pick stubs before atoms: a stub tip is small and usually in front of its atom.
- The view stays a view. A controller (below) owns the model and calls `show_rdkit(model.mol,
  reset_camera=False)` after each change. Keep the selection when the atom count does not change.

## Free valences as stubs

- In build mode, hide hydrogens as atoms and draw every `model.all_free_valences()` as a stub: a short
  cylinder from the atom's sphere surface, about 0.6 Å long and 0.08 Å thick. Use one more
  `vtkGlyph3DMapper` actor, named `"build_stubs"`, built like the bonds (starts, `dir`, `scale` arrays).
- A hydrogen is a stub (attaching replaces it; the new atom takes the hydrogen's index). An open direction
  (`fill_h=False`) is a stub too, with `h = None`.
- Colour by `max_order`: single valences one colour, double valences (`max_order == 2`) another, or two
  thin parallel stubs as in Spartan.
- Picking a stub: `ray_sphere_pick(origin, direction, tips, radii)` with the stub tips as centres and a
  radius of about 0.25 Å. The row is the index into the list from `all_free_valences()`, which the
  controller keeps in the same order as the drawn stubs.
- Later: a hover highlight needs a mouse-move observer outside a gesture (not there today).

## What a click does

| Click | Palette selection | Call |
| --- | --- | --- |
| stub | atom type | `model.attach_atom(valence, kind)` |
| stub | group / ring / PFAS piece | `model.attach_fragment(valence, fragment)` |
| atom | atom type | `model.replace_atom(atom, kind)` |
| empty space, empty model | atom type or fragment | `model.add_atom(kind, point)` / `add_fragment(fragment, point)` |

Edit tools act on the selection (selected in rotate mode or with Ctrl+click in build mode):
Delete (`delete_atoms`), bond order (`set_bond_order`, cycling 1, 2, 3), charge + and −
(`set_charge`), make and break bond (two atoms), set the measured value (a value box next to the
measurement: `set_distance`, `set_angle`, `set_dihedral`), Minimize (`minimize(frozen=...)`, with a
"Freeze selected" check box), Add H (`fill_hydrogens`). An `EditError` goes to the status bar; the model
does not change.

Set `model.auto_relax = True` in the GUI: the local relax then runs inside each command, so undo covers it.

## Undo

A small adaptor with `push(command)`:
1. call `command.apply(model)` first, and show the `EditError` if it fails (nothing is pushed);
2. push a `QUndoCommand` whose `redo()` skips its first call (the command already ran) and later calls
   `command.apply(model)`, and whose `undo()` calls `command.undo(model)`. `QUndoStack.push` calls
   `redo()` at once, so applying inside the first `redo()` would leave a failed command on the stack.
3. return `command.result`.

The text is `command.text`. Edit > Undo / Redo come from `stack.createUndoAction` /
`createRedoAction` (Ctrl+Z, Ctrl+Y).

## Palette dock

A "Model Kit" dock in the Build workspace (UI-1), following Spartan's tabs:
- **Organic:** a grid of tool buttons for `edit.ORGANIC` in its order (C sp3 / sp2 / sp; N sp3 / sp2 / sp;
  P; O sp3 / sp2; S; Si; H; F; Cl; Br; I; =O; =S; ≡N). Each icon is the atom with its stubs in their
  geometry, drawn once with QPainter. Below: **Groups**, **Rings** and **More** menus
  (`fragments.by_category("group")`, `"ring"`, and the extra palette types N+, O−, S(IV), S(VI), P(V), B),
  a **PFAS** menu (`"pfas"`, with a spin box for `(CF2)n` and `CnF2n+1`), and the **Clipboard**,
  **SMILES** and **InChI** buttons. SMILES text becomes a fragment with `fragments.from_smiles`
  (its first atom attaches when there is no `[*:1]`); InChI goes through RDKit's `MolFromInchi` first;
  Clipboard uses the UI-4 clipboard parser.
- **Substituent:** `fragments.by_category("substituent")`.
- **Inorganic, Peptide, Nucleotide:** later. Inorganic needs 5- and 6-domain geometries, which
  `domain_directions` does not build yet.
- Bottom row: the edit tools above, the measured value box, Minimize, and **Done**. Done fills open
  valences and hands `model.to_input(name=...)` to the molecule list (new entry, or replacing the
  molecule being edited: `EditableMolecule.from_input(item)` starts an edit).
- A line under the palette shows `model.problems()` (unfilled valences).

## Tests for that task

Drive the build mode through the VTK-free mouse methods (`_mouse_press` / `_mouse_release` with a
build-mode click), as `test_viewer3d.py` does for select and move. Stub picking can be tested with
`ray_sphere_pick` on known tips. Check that a palette choice plus a stub click attaches, that Ctrl+Z
undoes through the QUndoStack, that an `EditError` leaves the stack unchanged, and take a screenshot.
