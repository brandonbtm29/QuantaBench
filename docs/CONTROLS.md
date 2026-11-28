# Controls: keyboard shortcuts and keymaps

QuantaBench's keyboard shortcuts are listed and changed in **Settings > Controls** (also **Tools > Keyboard
controls...**). This page gives the default keymap, how it was chosen, the other keymaps (presets) and where each
comes from, and how a keymap is kept, exported and imported. The code is `src/quantabench/gui/keymap.py` (the
registry of named actions and the presets) and `src/quantabench/gui/controls_page.py` (the page). UI-9 in
`docs/dev/BACKLOG.md`.

## Using Settings > Controls

- **Keymap:** *QuantaBench (default)*, or a preset modelled on another program's documented keys: *Avogadro 2*,
  *VMD*, *Spartan*. The line under it says what the preset changes and links its source.
- **The table** lists every action of the window with its category and keys. The search box filters by action,
  category or key ("zoom", "ctrl+shift"). A key you changed is **bold** (its tooltip gives the keymap's own key); a
  key that two actions share where both would be active is **red**, with the other action named.
- **Rebind:** select a row, click the key field next to *Clear* and press the new key. When another action already
  has that key in the same place, QuantaBench asks:
  - **Swap:** the other action gets this action's old keys;
  - **Clear the other:** the other action loses the key;
  - **Cancel:** nothing changes.
- **Clear** leaves the selected action without a key; **Reset this action** gives it the keymap's keys again;
  **Reset to QuantaBench defaults** chooses the QuantaBench keymap and drops all your changes.
- **Export... / Import...** write and read a keymap file (JSON, below).
- Nothing takes effect until **OK**; *Cancel* drops the page's changes, like every other Settings tab.

Your changes are kept as **overrides on top of the chosen keymap**: switching from *QuantaBench* to *VMD* keeps a key
you set yourself, and *Reset this action* returns to the new keymap's key.

### Per computer, not shared

The keymap is kept **on each computer**, in the window settings next to the window layout (the registry on Windows,
`~/Library/Preferences` on macOS, `~/.config` on Linux), not in the synced `settings.json`:

- **The modifier keys differ.** Qt writes Command as "Ctrl" on macOS and the Control key as "Meta", so a key that
  is right on one system can be wrong on the other: Alt+1 switches workspaces on Windows, but Option+1 types "¡" on
  a Mac; Meta+1 is the Control key on a Mac but the Windows key (reserved for Windows) on a PC.
- **The keyboards differ.** A MacBook has no Insert, Home, End or Page keys without fn, and its F-keys need fn.
- **It is a display setting**, like the window layout and the dock arrangement, which are per computer too. It never
  changes a calculation and is never sent to a worker.

To use the same keymap on two computers, **Export** it on one and **Import** it on the other. A file made on a Mac
and read on Windows (or the reverse) says so, as a reminder to check the modifier keys.

### Where keys work

Each action has a scope:

| Scope | Where the key works |
|---|---|
| window | anywhere in the main window (menus, toolbars, every dock) |
| views | in the 2D and 3D structure views |
| 3D view, 2D view | in that view only |
| Molecules dock, table | in that dock only |
| Methods guide | in the Methods guide window |

Two actions may share a key only where they can never both be active: the 2D and the 3D view both zoom with + and -;
Analyze's table and the structure views both select all with Ctrl+A. A key shared within one scope, or between a
window action and a view action, is a conflict: Qt would run neither. The page flags it and asks before making one.

**Plain keys** (letters, digits, arrows, Space, Esc) are only used in the views by default, so they never take
keys from a text field, a list's type-ahead or a table. A key typed in a text field always goes to the field.

## How the default keymap was chosen

QuantaBench's keys were chosen from platform conventions and QuantaBench's own logic, not from another molecular
program's assignments. In order:

1. **The platform's standard keys** wherever the platform has one: Open, Save, New, Close, Undo, Redo, Paste, Copy,
   Select all, Find, Help. They are Qt's `QKeySequence.StandardKey` (marked \* in the table), so each system gets its
   own: Ctrl+O on Windows and Linux, ⌘O on macOS; Redo is Ctrl+Y on Windows and ⇧⌘Z on macOS; Help is F1 on Windows
   and Linux and ⌘? on macOS. Where Qt has no standard for Windows (Save as, Quit), the common key (Ctrl+Shift+S,
   Ctrl+Q) is used, which Qt makes ⇧⌘S and ⌘Q on macOS. Settings is Ctrl+, (⌘, on macOS, Apple's standard).
2. **What the platform guidelines say not to do:**
   - no Ctrl+Alt combinations: Windows reads them as AltGr on many keyboards, which types characters
     ([Microsoft: Keyboard, Win32 guidelines](https://learn.microsoft.com/en-us/windows/win32/uxguide/inter-keyboard)).
     *Open project* moved from Ctrl+Alt+O to Ctrl+Shift+O for this reason;
   - no Control-key shortcuts on macOS, which the system uses (Mission Control's Control+digit switches desktops),
     and Option+digit types characters
     ([Apple Human Interface Guidelines: Keyboards](https://developer.apple.com/design/human-interface-guidelines/keyboards)).
     The workspaces moved from Control+1-5 to ⌥⌘1-5 on macOS;
   - a standard key is never given another meaning, and Shift adds to or reverses the plain key's command
     (Shift+X looks from the other side; Ctrl+Shift+S saves as);
   - macOS's ⌘. (Cancel an operation) also cancels a run there.
3. **QuantaBench's own logic:**
   - *Ctrl + a digit* chooses the 3D view's mouse mode (Ctrl+1 Rotate, Ctrl+2 Select, Ctrl+3 Move, Ctrl+4 Build with
     the Model Kit): Microsoft recommends Ctrl+number for frequent commands, and the toolbar shows them in order.
   - *Alt + a digit* (⌥⌘ + a digit on macOS) chooses the workspace, in the activity bar's order.
   - *F5 runs* the build (as Run in an IDE), *F6 reviews* it first (the next key), *Shift+Esc cancels* (Esc alone
     clears the selection in the views, and a window-wide Esc would stop a batch by accident).
   - *In the structure views*, arrows move through what the window shows: Left/Right the previous and next conformer
     (as the playback bar's slider, which is horizontal), Up/Down the previous and next molecule (as the Molecules
     list, which is vertical), Home/End the first and last conformer, Ctrl+Home/End the first and last molecule, Space
     plays and pauses. R resets the view, X/Y/Z look down an axis (Shift: from the other side), P switches the
     projection, + and - zoom, Ctrl+arrows turn the 3D view 15°, Esc and Ctrl+A clear and select the atoms.
   - Letters are the command's initial where they can be: Ctrl+J New **j**ob, Ctrl+D Show 2**D**, Ctrl+E **E**xport,
     Ctrl+Shift+R open **r**un folder, F **f**its the 2D view, P **p**rojection.
4. **No key for rare commands** (Open output folder, Hardware report, Screenshot, the Copy as formats): the menus
   reach them, and you can give them one.

### How it differs from Spartan's

Spartan's Tutorial and User's Guide names few keys in its text: Page Up/Page Down to step through the open molecules,
Home/End for the first and last, Insert or Delete held while clicking to insert or delete in a model, Ctrl+V to
paste, Enter in the spreadsheet, and the mouse modifiers. QuantaBench's defaults share only the platforms' standard
keys with it (Ctrl+V and the like, which every program uses). For stepping through molecules QuantaBench uses the
arrow keys (Up/Down, Ctrl+Home/End) in the structure views instead of Spartan's Page Up/Page Down/Home/End, and its
mouse modes, workspaces and view keys have no counterpart in Spartan's documentation. Spartan's menu accelerators,
which its manual shows only in pictures of its menus, were not looked at or used. If you want Spartan's documented
keys, choose the *Spartan* keymap.

### What changed from QuantaBench's keys before UI-9

- Open project: Ctrl+Alt+O → **Ctrl+Shift+O**; Open run folder: Ctrl+Shift+O → **Ctrl+Shift+R**.
- Close project: none → **Ctrl+W** (the platform's Close).
- Methods guide: F1 → the platform's Help key (F1 on Windows and Linux, ⌘? on macOS).
- Workspaces on macOS: Control+1-5 → **⌥⌘1-5** (Windows and Linux keep Alt+1-5).
- Cancel: Shift+Esc, and ⌘. on macOS too.
- Reset view (R) works in the structure views instead of window-wide (lists keep their type-ahead); Esc clears the
  selection in the 2D view as well as the 3D view.
- New: Find a molecule (Ctrl+F: the Molecules list's search box), Select all atoms (Ctrl+A in the views), the
  playback and molecule keys above, zoom (+/-) and turn (Ctrl+arrows) in the 3D view, Ctrl+Left/Right to rotate the
  2D view (as well as [ and ], which some keyboard layouts can only type with AltGr).
- Analyze's table: Copy (Ctrl+C), Copy with column titles (Ctrl+Shift+C) and Select all are actions; the table's
  right-click menu shows their real keys.

## The default keymap

\* the platform's standard key (`QKeySequence.StandardKey`): it follows the system. "Ctrl" is ⌘ on macOS. The
Model Kit's actions (UI-7) and Paste, Copy molecules and Copy as (UI-4) are in the registry like every other.

| Category | Action | Windows / Linux | macOS | Where | Name |
|---|---|---|---|---|---|
| File | New project | Ctrl+N\* | ⌘N | window | `file.new_project` |
| File | Open... | Ctrl+O\* | ⌘O | window | `file.open` |
| File | Open project... | Ctrl+Shift+O | ⇧⌘O | window | `file.open_project` |
| File | Open run folder... | Ctrl+Shift+R | ⇧⌘R | window | `file.open_run_folder` |
| File | Import calculation output... | - | - | window | `file.import_output` |
| File | Save project | Ctrl+S\* | ⌘S | window | `file.save_project` |
| File | Save project as... | Ctrl+Shift+S | ⇧⌘S | window | `file.save_project_as` |
| File | Close project | Ctrl+W | ⌘W | window | `file.close_project` |
| File | Export... | Ctrl+E | ⌘E | window | `file.export` |
| File | Open output folder | - | - | window | `file.open_output_folder` |
| File | Quit | Ctrl+Q | ⌘Q | window | `file.quit` |
| Edit | Undo | Ctrl+Z\* | ⌘Z | window | `edit.undo` |
| Edit | Redo | Ctrl+Y\*, Ctrl+Shift+Z | ⇧⌘Z, ⌘Y | window | `edit.redo` |
| Edit | Paste | Ctrl+V\* | ⌘V | window | `edit.paste` |
| Edit | Copy molecules | Ctrl+Shift+C | ⇧⌘C | Molecules dock | `edit.copy_molecules` |
| Edit | Copy as SMILES | - | - | window | `edit.copy_as.smiles` |
| Edit | Copy as Canonical SMILES | - | - | window | `edit.copy_as.canonical` |
| Edit | Copy as InChI | - | - | window | `edit.copy_as.inchi` |
| Edit | Copy as InChIKey | - | - | window | `edit.copy_as.inchikey` |
| Edit | Copy as MOL / SDF | - | - | window | `edit.copy_as.mol` |
| Edit | Copy as XYZ | - | - | window | `edit.copy_as.xyz` |
| Edit | Copy as CDXML | - | - | window | `edit.copy_as.cdxml` |
| Edit | Find a molecule | Ctrl+F\* | ⌘F | window | `edit.find_molecule` |
| Selection | Select all atoms | Ctrl+A\* | ⌘A | views | `selection.select_all` |
| Selection | Clear the selection | Esc (Ctrl+Shift+A on Linux desktops\*) | Esc | views | `selection.clear` |
| Build | New job... | Ctrl+J | ⌘J | window | `build.new_job` |
| Build | Review molecules... | F6 | F6 | window | `build.review` |
| Build | Run | F5 | F5 | window | `build.run` |
| Build | Cancel | Shift+Esc | ⇧Esc, ⌘. | window | `build.cancel` |
| Build | Resume run folder | - | - | window | `build.resume` |
| Build | Sketch a molecule... | - | - | window | `build.sketch` |
| Build | Surface builder... | - | - | window | `build.surface_builder` |
| Build | Model kit (Build mode) | - | - | window | `build.model_kit` |
| Build | Minimize the Model Kit's structure | - | - | window | `kit.minimize` |
| Build | Clean up the 2D layout | - | - | window | `build.clean_2d` |
| Build | Clean up the 3D structure | - | - | window | `build.clean_3d` |
| Mouse mode | Rotate mode | Ctrl+1 | ⌘1 | window | `view.mode_rotate` |
| Mouse mode | Select mode | Ctrl+2 | ⌘2 | window | `view.mode_select` |
| Mouse mode | Move mode | Ctrl+3 | ⌘3 | window | `view.mode_move` |
| Mouse mode | Build mode (Model Kit) | Ctrl+4 | ⌘4 | window | `view.mode_build` |
| View | Reset view | R | R | views | `view.reset` |
| View | Show 2D | Ctrl+D | ⌘D | window | `view.show_2d` |
| View | Hydrogens in 2D | - | - | window | `view.hydrogens_2d` |
| View | Screenshot... | - | - | window | `view.screenshot` |
| View | Open in OVITO viewport | - | - | window | `view.open_ovito` |
| View | Reset layout | - | - | window | `view.reset_layout` |
| View | Reset all workspace layouts | - | - | window | `view.reset_all_layouts` |
| View | Dock all panels | Ctrl+Shift+D | ⇧⌘D | window | `view.dock_all` |
| Workspace | Build workspace | Alt+1 | ⌥⌘1 | window | `workspace.build` |
| Workspace | Calculate workspace | Alt+2 | ⌥⌘2 | window | `workspace.calculate` |
| Workspace | Simulate workspace | Alt+3 | ⌥⌘3 | window | `workspace.simulate` |
| Workspace | Analyze workspace | Alt+4 | ⌥⌘4 | window | `workspace.analyze` |
| Workspace | Library workspace | Alt+5 | ⌥⌘5 | window | `workspace.library` |
| Playback | Previous conformer (or molecule, in the bar's Molecules mode) | Left | Left | views | `playback.previous` |
| Playback | Next conformer (or molecule) | Right | Right | views | `playback.next` |
| Playback | First conformer | Home | Home | views | `playback.first` |
| Playback | Last conformer | End | End | views | `playback.last` |
| Playback | Play / pause | Space | Space | views | `playback.play` |
| Playback | Previous molecule in the list | Up | Up | views | `molecules.previous` |
| Playback | Next molecule in the list | Down | Down | views | `molecules.next` |
| Playback | First molecule in the list | Ctrl+Home | ⌘Home | views | `molecules.first` |
| Playback | Last molecule in the list | Ctrl+End | ⌘End | views | `molecules.last` |
| 3D view | Look down the x axis | X | X | 3D view | `view3d.along_x` |
| 3D view | Look down the y axis | Y | Y | 3D view | `view3d.along_y` |
| 3D view | Look down the z axis | Z | Z | 3D view | `view3d.along_z` |
| 3D view | Look down the x axis from the negative side | Shift+X | ⇧X | 3D view | `view3d.along_minus_x` |
| 3D view | Look down the y axis from the negative side | Shift+Y | ⇧Y | 3D view | `view3d.along_minus_y` |
| 3D view | Look down the z axis from the negative side | Shift+Z | ⇧Z | 3D view | `view3d.along_minus_z` |
| 3D view | Perspective / orthographic projection | P | P | 3D view | `view3d.projection` |
| 3D view | Zoom in | +, = | +, = | 3D view | `view3d.zoom_in` |
| 3D view | Zoom out | -, _ | -, _ | 3D view | `view3d.zoom_out` |
| 3D view | Turn left | Ctrl+Left | ⌘Left | 3D view | `view3d.rotate_left` |
| 3D view | Turn right | Ctrl+Right | ⌘Right | 3D view | `view3d.rotate_right` |
| 3D view | Turn up | Ctrl+Up | ⌘Up | 3D view | `view3d.rotate_up` |
| 3D view | Turn down | Ctrl+Down | ⌘Down | 3D view | `view3d.rotate_down` |
| 3D view | Roll counter-clockwise | - | - | 3D view | `view3d.roll_ccw` |
| 3D view | Roll clockwise | - | - | 3D view | `view3d.roll_cw` |
| 2D view | Zoom in | +, = | +, = | 2D view | `view2d.zoom_in` |
| 2D view | Zoom out | -, _ | -, _ | 2D view | `view2d.zoom_out` |
| 2D view | Rotate counter-clockwise | Ctrl+Left, [ | ⌘Left, [ | 2D view | `view2d.rotate_ccw` |
| 2D view | Rotate clockwise | Ctrl+Right, ] | ⌘Right, ] | 2D view | `view2d.rotate_cw` |
| 2D view | Fit the structure | F | F | 2D view | `view2d.fit` |
| Table | Copy cells | Ctrl+C\* | ⌘C | table | `table.copy` |
| Table | Copy cells with column titles | Ctrl+Shift+C | ⇧⌘C | table | `table.copy_with_titles` |
| Table | Select all cells | Ctrl+A\* | ⌘A | table | `table.select_all` |
| Tools | Settings... | Ctrl+, | ⌘, | window | `tools.settings` |
| Tools | Keyboard controls... | - | - | window | `tools.controls` |
| Tools | Hardware report | - | - | window | `tools.hardware_report` |
| Help | Methods guide... | F1\* | ⌘? | window | `help.methods_guide` |
| Help | Find in the Methods guide | Ctrl+F\* | ⌘F | Methods guide | `help.guide_find` |
| Help | Check for updates... | - | - | window | `help.check_updates` |
| Help | Create app shortcuts... | - | - | window | `help.create_shortcuts` |
| Help | About QuantaBench | - | - | window | `help.about` |

## Presets

Each preset was built only from the program's public documentation, linked below, and copies only **key
assignments** (which key does what), mapped onto QuantaBench's actions; no text, icons or layouts. Where the program
has no counterpart for a QuantaBench action, the QuantaBench key stays; where a program's key would take a key
QuantaBench needs in the same place, the table says which gave way. Every preset is checked by the tests for keys two
actions would share.

### Avogadro 2

Source: [Avogadro 2 documentation, Keyboard Shortcuts](https://avogadro.cc/docs/tools/keyboard-shortcuts.html) and
[File menu](https://avogadro.cc/docs/menus/file-menu.html) (Avogadro 2.1).

| Avogadro 2 | Key | QuantaBench action |
|---|---|---|
| Navigation tool | Ctrl+1 | Rotate mode (as the default) |
| Draw tool | Ctrl+2 | Build mode, Model Kit (UI-7) |
| Selection tool | Ctrl+4 | Select mode |
| Manipulation tool | Ctrl+5 | Move mode |
| Select all / Deselect all atoms | Ctrl+A / Ctrl+Shift+A | Select all atoms / Clear the selection (Esc is not Avogadro's, so it is off) |
| Optimize geometry | Ctrl+Alt+O | Minimize the Model Kit's structure (UI-7) |
| New, Open, Close, Save, Save As, Quit | Ctrl+N, Ctrl+O, Ctrl+W, Ctrl+S, Ctrl+Shift+S, Ctrl+Q | New project, Open..., Close project, Save project, Save project as, Quit (the defaults already) |
| Arrow keys, WASD or HJKL: rotate | Left/A/H, Right/D/L, Up/W/K, Down/S/J | Turn left, right, up, down (3D view) |
| Shift+Left/Right: rotate about z | Shift+Left, Shift+Right | Roll counter-clockwise, clockwise |
| Shift+Up/Down: zoom | Shift+Up, Shift+Down | Zoom in, out (3D view) |

No counterpart, so not mapped: the Template, Bond-centric manipulation, Auto-optimization, Measure and Animation tools
(Ctrl+3, Ctrl+6-9), Ctrl+arrows to pan, Alt for larger steps, the Draw tool's element and bond-order keys, and the
Build menu's Adjust hydrogens (Ctrl+Alt+H) and Bond atoms (Ctrl+B): the Model Kit's tools are buttons, not actions
with keys, so far.
Gave way: QuantaBench's arrow keys for conformers and molecules (Avogadro's arrows turn the view), and Ctrl+arrows
(QuantaBench's turn keys, now the plain arrows).

### VMD

Source: [VMD User's Guide, Hot Keys](https://tcbg.illinois.edu/Research/vmd/current/ug/node34.html) (the OpenGL
window's hot keys, tables 5.1-5.4).

| VMD | Key | QuantaBench action |
|---|---|---|
| Rotate mode | R | Rotate mode |
| Move molecule (pick mode 8) | 8 | Move mode |
| Reset view (`display resetview`) | Ctrl+R | Reset view |
| Rotate about x by ±2° | J, K | Turn down, up |
| Rotate about y by ±2° | L, H | Turn right, left |
| Rotate about z by ±2° | G, Shift+G | Roll counter-clockwise, clockwise |
| Next frame | +, F, Shift+F | Next conformer (or molecule) |
| Previous frame | -, B, Shift+B | Previous conformer (or molecule) |
| Play forward / pause | . and >, / and ? | Play / pause (one toggle) |

No counterpart, so not mapped: translate and scale modes (T, S), the pick modes 0-7 and 9 (query, centre, atom,
bond, angle, dihedral labels and moves: QuantaBench measures whatever 2-4 atoms are selected), apply-force modes,
spin and rock (x, y, z: QuantaBench's X, Y, Z still look down the axes), playing backwards (, and <), and the Alt menu
keys and Alt+Q (Alt+letter opens QuantaBench's menus). Left out on purpose: Ctrl+A and Ctrl+Z (scale up and down) and
Ctrl+N, Ctrl+P, Ctrl+F, Ctrl+B (alternatives of J, K, L, H), because Ctrl+A, Ctrl+Z, Ctrl+N and Ctrl+F are the
platforms' Select all, Undo, New and Find. Gave way: R for Reset view (now Ctrl+R), the zoom keys + and - in both views
and F (fit) in the 2D view (VMD's frame keys).

### Spartan

Source: [Spartan'26 Tutorial and User's Guide](https://downloads-s3.wavefun.com/support/Spartan26Manual.pdf)
(Wavefunction; Chapter 1, "Keyboard Operations", and the text of the later chapters). Brandon asked for this keymap
as an option; it is not the default.

| Spartan | Key | QuantaBench action |
|---|---|---|
| Move up / down through the open molecules | Page Up / Page Down | Previous / next molecule in the list |
| Top / bottom of the open molecules | Home / End | First / last molecule in the list |
| Paste | Ctrl+V | Paste (as the default) |

No counterpart, so not mapped: Insert or Delete held while clicking (insert a fragment, delete what is clicked: mouse
actions; the Model Kit's Delete is a button), 3 (toggle stereo 3D), Enter in the spreadsheet (Qt's own), and the
mouse-button modifiers (QuantaBench's mouse is not rebindable). Spartan's menus have accelerators too, but the manual
shows them only in pictures of the menus, not in its text, so they are not here. Gave way: Home and End for the first
and last conformer.

### Considered and not built

PyMOL's documented default keys
([Edit Keys](https://wiki.pymol.org/index.php/Edit_Keys)) edit atoms in the viewer (Ctrl+C replaces an atom by carbon,
Ctrl+D deletes, Ctrl+A redoes...): with no counterpart in QuantaBench before the Model Kit, and several taking the
platforms' standard keys, a PyMOL keymap would change almost nothing. GaussView 6's reference manual documents mouse
modifiers rather than keys, and the Maestro keyboard pages found were copies on a third-party host, not on
Schrödinger's own site. Any of them can be added
as a preset later (`PRESETS` in `gui/keymap.py`).

## The keymap file (Export / Import)

```json
{
  "quantabench_keymap": 1,
  "preset": "vmd",
  "overrides": {"build.cancel": [], "file.export": ["Ctrl+Shift+E"]},
  "platform": "win32",
  "note": "..."
}
```

`preset` is one of `quantabench`, `avogadro`, `vmd`, `spartan`; `overrides` maps an action's name (the table above)
to its keys in Qt's portable text ("Ctrl" is ⌘ on macOS, "Meta" is the Control key there), `[]` for no key. An import
leaves out actions it does not know and keys it cannot read, and says which. `platform` is where the file was made.

## For developers

- A new command with a key: add an `ActionSpec` to `keymap.SPECS` (name, title, category, default keys, scope) and
  make the action with `win.keymap.action(...)` or bind an existing one with `keymap.bind(win, action, name, tip)`.
  View-local actions (`views`, `view3d`, `view2d`, `table`) are made on their widget with `keymap.local_action` and
  bound by the window; a widget that eats its keys calls `keymap.dispatch_key` (or installs a `KeyDispatcher`).
- Never call `setShortcut` or create a `QShortcut` with a key: `tests/test_gui_controls.py` fails on any outside
  `keymap.py`, on an action with a key that is not in the registry, and on a preset that gives two actions one key.
- A tooltip passed to `bind` gets the current key appended, and `Keymap.follow_tooltip` does the same for a button
  that does what an action does.
