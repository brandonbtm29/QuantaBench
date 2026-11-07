"""Import Options (UI-4), modeled on Spartan's SDF Import Options: File > Open shows the dialog when a file has
choices (2D structures, data fields, several fragments, several structures of one constitution). The same form is
the Import tab of Settings, where the stored choices and "ask" live (``Settings.import_``).

    ImportOptionsWidget   the form: Fixup, Additional data, Filters, Labeling, Several structures of one constitution
    ImportOptionsDialog   the form, what the file holds (``chem.import_options.Survey``) and "Don't ask again"
    choose(win, s, items) the options for ``items``: asks when the file has choices and the settings say to ask
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from ..chem.import_options import ImportOptions, Survey, survey

#: File > Open hands these to File > Import calculation output (cclib outputs, Spartan documents)
CALCULATION_OUTPUTS = (".spartan", ".log", ".out")
AUTOMATIC = "(automatic)"


def is_calculation_output(path: str | Path) -> bool:
    return Path(path).suffix.lower() in CALCULATION_OUTPUTS


def survey_text(s: Survey, names: list[str] | None = None) -> str:
    """One paragraph: what the file holds that the options act on."""
    what = f"{s.n} structure(s)" + (f" in {', '.join(names[:3])}{', ...' if len(names) > 3 else ''}" if names else "")
    bits = []
    if s.n_2d:
        bits.append(f"{s.n_2d} drawn in 2D")
    if s.fields:
        bits.append(f"{len(s.fields)} data field(s) ({', '.join(s.fields[:4])}{', ...' if len(s.fields) > 4 else ''})")
    if s.n_multi_fragment:
        kinds = {"water": "water", "acid": "HCl and other acids", "counterion": "counter-ions",
                 "other": "mixtures of organic fragments"}
        frag = ", ".join(f"{label} {s.fragments[k]}" for k, label in kinds.items() if s.fragments.get(k))
        bits.append(f"{s.n_multi_fragment} with several fragments" + (f" ({frag})" if frag else ""))
    if s.n_same_constitution:
        bits.append(f"{s.n_same_constitution} sharing a constitution with another")
    reader = f" Read with {', '.join(s.readers)}." if s.readers else ""
    return what + (": " + "; ".join(bits) if bits else "") + "." + reader


class ImportOptionsWidget(QWidget):
    """The options form. ``fields``: the data fields offered for the id and name (the file's, in the dialog)."""

    def __init__(self, options: ImportOptions, fields: list[str] | None = None, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        fix = QGroupBox("Fixup")
        fl = QVBoxLayout(fix)
        self.convert_2d = QCheckBox("Convert 2D structures to 3D")
        self.convert_2d.setToolTip("A MOL/SDF record with the 2D flag or every z = 0, a ChemDraw or 2D Marvin drawing\n"
                                   "gets 3D coordinates (ETKDGv3 + MMFF94s). The build makes its own conformers anyway.")
        self.grow_h = QCheckBox("Grow hydrogens")
        self.grow_h.setToolTip("Add the implicit hydrogens as atoms, with coordinates")
        self.keep_coords = QCheckBox("Keep the file's 3D coordinates as the first conformer")
        self.keep_coords.setToolTip("The build starts its conformer search from the structure in the file (MMFF94s-\n"
                                    "optimized like the others) besides new ETKDG conformers; with Max conformers 1 it\n"
                                    "uses only the file's structure.")
        for w in (self.convert_2d, self.grow_h, self.keep_coords):
            fl.addWidget(w)
        lay.addWidget(fix)

        data = QGroupBox("Additional data")
        df = QFormLayout(data)
        self.post_fields = QCheckBox("Post data fields to the molecule table")
        self.post_fields.setToolTip("SD properties and table columns become columns of the molecule list and of\n"
                                    "summary.csv")
        self.parse_fields = QCheckBox("Post-process data fields (read numbers; a unit goes into the column name)")
        df.addRow(self.post_fields)
        df.addRow(self.parse_fields)
        self.id_field, self.name_field = QComboBox(), QComboBox()
        for combo, current, tip in ((self.id_field, options.id_field, "SD property or column for the molecule ids\n"
                                     f"{AUTOMATIC}: DTXSID, ID, then the title or file name"),
                                    (self.name_field, options.name_field, "SD property or column for the names\n"
                                     f"{AUTOMATIC}: the title")):
            combo.setEditable(True)
            combo.addItem(AUTOMATIC)
            for f in fields or []:
                combo.addItem(f)
            combo.setCurrentText(current or AUTOMATIC)
            combo.setToolTip(tip)
        df.addRow("ID field", self.id_field)
        df.addRow("Name field", self.name_field)
        lay.addWidget(data)

        filt = QGroupBox("Filters (fragments removed on import)")
        ff = QVBoxLayout(filt)
        self.water = QCheckBox("Water")
        self.acids = QCheckBox("HCl, HBr, HI and other small inorganic molecules")
        self.counterions = QCheckBox("Counter-ions (Na+, K+, Cl-, SO4 2-, NH4+, ...)")
        self.largest = QCheckBox("Keep only the largest fragment")
        self.largest.setToolTip("The fragment the build would keep. Fragments with metals other than Na+, K+, ... are\n"
                                "left for the build, which does not compute a metal complex drawn as separate ions.")
        for w in (self.water, self.acids, self.counterions, self.largest):
            ff.addWidget(w)
        note = QLabel("The build's <i>Keep largest fragment</i> (Build defaults) still applies to what is kept.")
        note.setWordWrap(True)
        ff.addWidget(note)
        lay.addWidget(filt)

        lab = QGroupBox("Labeling")
        ll = QVBoxLayout(lab)
        self.read_order = QRadioButton("Label atoms in read order")
        self.standard_order = QRadioButton("Label atoms in standard order (RDKit canonical ranks)")
        self._labels = QButtonGroup(self)
        for w in (self.read_order, self.standard_order):
            self._labels.addButton(w)
            ll.addWidget(w)
        lay.addWidget(lab)

        same = QGroupBox("Several structures of one constitution")
        sl = QVBoxLayout(same)
        self.as_molecules = QRadioButton("Separate molecules")
        self.as_conformers = QRadioButton("Conformers of one molecule")
        self._same = QButtonGroup(self)
        for w in (self.as_molecules, self.as_conformers):
            self._same.addButton(w)
            sl.addWidget(w)
        lay.addWidget(same)
        self.set_options(options)

    def set_options(self, o: ImportOptions) -> None:
        self.convert_2d.setChecked(o.convert_2d)
        self.grow_h.setChecked(o.grow_hydrogens)
        self.keep_coords.setChecked(o.keep_coords)
        self.post_fields.setChecked(o.post_fields)
        self.parse_fields.setChecked(o.parse_fields)
        self.id_field.setCurrentText(o.id_field or AUTOMATIC)
        self.name_field.setCurrentText(o.name_field or AUTOMATIC)
        self.water.setChecked(o.remove_water)
        self.acids.setChecked(o.remove_acids)
        self.counterions.setChecked(o.strip_counterions)
        self.largest.setChecked(o.keep_largest)
        (self.standard_order if o.label_order == "standard" else self.read_order).setChecked(True)
        (self.as_conformers if o.same_constitution == "conformers" else self.as_molecules).setChecked(True)

    def options(self) -> ImportOptions:
        def field(combo: QComboBox) -> str:
            text = combo.currentText().strip()
            return "" if text == AUTOMATIC else text

        return ImportOptions(
            convert_2d=self.convert_2d.isChecked(), grow_hydrogens=self.grow_h.isChecked(),
            keep_coords=self.keep_coords.isChecked(), post_fields=self.post_fields.isChecked(),
            parse_fields=self.parse_fields.isChecked(), id_field=field(self.id_field), name_field=field(self.name_field),
            remove_water=self.water.isChecked(), remove_acids=self.acids.isChecked(),
            strip_counterions=self.counterions.isChecked(), keep_largest=self.largest.isChecked(),
            label_order="standard" if self.standard_order.isChecked() else "read",
            same_constitution="conformers" if self.as_conformers.isChecked() else "molecules")


class ImportOptionsDialog(QDialog):
    def __init__(self, s: Survey, options: ImportOptions, names: list[str] | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import Options")
        lay = QVBoxLayout(self)
        info = QLabel(survey_text(s, names))
        info.setWordWrap(True)
        lay.addWidget(info)
        self.form = ImportOptionsWidget(options, s.fields, self)
        lay.addWidget(self.form)
        self.dont_ask = QCheckBox("Don't ask again (Settings > Import keeps these choices and can ask again)")
        lay.addWidget(self.dont_ask)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

    def options(self) -> ImportOptions:
        return self.form.options()


def choose(parent, settings, items, names: list[str] | None = None) -> ImportOptions | None:
    """The import options for ``items`` (read with ``inputs.read_raw``): the stored ones (``settings.import_``), or
    the dialog's when the file has choices and ``settings.import_.ask``. The choices made are stored for next time
    (and "Don't ask again" stops the dialog). None when the user cancels."""
    from .. import settings as settings_mod

    opts = ImportOptions.from_settings(settings)
    s = survey(items)
    if not settings.import_.ask or not s.has_choices():
        return opts
    dlg = ImportOptionsDialog(s, opts, names, parent)
    try:
        if not dlg.exec():                  # rejected (0): Cancel or Esc
            return None
        opts = dlg.options()
        opts.to_settings(settings)
        if dlg.dont_ask.isChecked():
            settings.import_.ask = False
    finally:
        dlg.deleteLater()
    try:
        settings_mod.save(settings)
    except OSError:
        pass                     # a read-only settings folder: the choices still apply to this file
    return opts
