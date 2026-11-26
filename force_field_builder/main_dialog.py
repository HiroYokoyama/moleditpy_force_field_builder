"""The Force Field Builder dialog."""

from __future__ import annotations

import logging
import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from . import builder, charges, display, external, topology, water, writers
from .mdff import dumps

logger = logging.getLogger(__name__)

LABEL_ACTOR = "force_field_builder_labels"

_ATOM_COLS = [
    "#",
    "Name",
    "Element",
    "Type",
    "Charge (e)",
    "Mass",
    "Sigma (nm)",
    "Epsilon (kJ/mol)",
]
_BOND_COLS = ["i", "j", "Atoms", "r0 (nm)", "k (kJ/mol/nm²)", "Parameter"]
_ANGLE_COLS = ["i", "j", "k", "Atoms", "θ0 (deg)", "k (kJ/mol/rad²)", "Parameter"]
_TORSION_COLS = [
    "i",
    "j",
    "k",
    "l",
    "Kind",
    "n",
    "Phase (deg)",
    "k (kJ/mol)",
    "Parameter",
]
_TYPE_COLS = ["Type", "Kind", "Copies", "Atoms", "Net charge", "Charges", "Notes"]


def _item(text, editable=False, number=None):
    item = QTableWidgetItem(str(text))
    flags = item.flags()
    if editable:
        item.setFlags(flags | Qt.ItemFlag.ItemIsEditable)
    else:
        item.setFlags(flags & ~Qt.ItemFlag.ItemIsEditable)
    if number is not None:
        item.setData(Qt.ItemDataRole.UserRole, number)
    return item


class ForceFieldBuilderDialog(QDialog):
    def __init__(
        self,
        parent=None,
        context=None,
        persistent_settings=None,
        get_molecule=None,
        mark_modified=None,
        restored=None,
        version="0.1.0",
        set_molecule=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Force Field Builder")
        self.resize(1100, 720)
        self.context = context
        self.persistent_settings = (
            persistent_settings if persistent_settings is not None else {}
        )
        self.get_molecule = get_molecule
        self.set_molecule = set_molecule
        self.mark_modified = mark_modified
        self.version = version
        self.result = None
        self.fingerprint = None
        self.imported_charges = None
        self.imported_label = ""
        self._filling = False
        self._updating = False
        self._display_applied = False

        layout = QVBoxLayout(self)

        options = QGroupBox("Parameters")
        form = QFormLayout(options)
        self.ff_combo = QComboBox()
        available_ff = builder.available_force_fields()
        for key, label in builder.FORCE_FIELDS.items():
            self.ff_combo.addItem(label, key)
            if key not in available_ff:
                self._disable_last(self.ff_combo, self._missing_tool_text(key))
        form.addRow("Force field:", self.ff_combo)

        charge_row = QHBoxLayout()
        self.charge_combo = QComboBox()
        available_charges = charges.available_methods()
        for key, label in charges.METHOD_LABELS.items():
            self.charge_combo.addItem(label, key)
            if key not in available_charges:
                self._disable_last(self.charge_combo, self._missing_tool_text(key))
        charge_row.addWidget(self.charge_combo, 1)
        self.import_button = QPushButton("Import Charges...")
        self.import_button.setToolTip(
            "ORCA output (RESP / CHELPG / Hirshfeld / Mulliken), PySCF Calculator properties.json, "
            "or a text file with one charge per atom (last number on each line)"
        )
        charge_row.addWidget(self.import_button)
        form.addRow("Charges:", charge_row)
        self.import_label = QLabel("")
        self.import_label.setWordWrap(True)
        form.addRow("", self.import_label)

        self.water_combo = QComboBox()
        for key, model in water.WATER_MODELS.items():
            self.water_combo.addItem(f"{model['label']}", key)
        self.water_combo.setToolTip(
            "Rigid three-site water. OPC (four-site) needs a virtual site that the "
            "moleditpy-mdff format cannot describe, so OPC3 is offered."
        )
        form.addRow("Water model:", self.water_combo)
        self.constrain_check = QCheckBox(
            "Mark bonds to hydrogen as constraints (Sage is fitted that way)"
        )
        form.addRow("", self.constrain_check)
        self.cross_check = QCheckBox("Cross-check with openff-toolkit")
        if not external.has_openff_toolkit():
            self.cross_check.setEnabled(False)
            self.cross_check.setToolTip("openff-toolkit is not installed")
        form.addRow("", self.cross_check)

        build_row = QHBoxLayout()
        self.build_button = QPushButton("Assign Parameters")
        self.build_button.setDefault(True)
        build_row.addWidget(self.build_button)
        self.status_label = QLabel(
            "Press Assign Parameters to parameterise the current molecule."
        )
        self.status_label.setWordWrap(True)
        build_row.addWidget(self.status_label, 1)
        form.addRow(build_row)
        layout.addWidget(options)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(
            QLabel("Molecule types (identical molecules are parameterised once):")
        )
        self.types_table = self._table(_TYPE_COLS)
        self.types_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        left_layout.addWidget(self.types_table)
        splitter.addWidget(left)

        self.tabs = QTabWidget()
        self.atoms_table = self._table(_ATOM_COLS)
        self.bonds_table = self._table(_BOND_COLS)
        self.angles_table = self._table(_ANGLE_COLS)
        self.torsions_table = self._table(_TORSION_COLS)
        self.warnings_text = QPlainTextEdit()
        self.warnings_text.setReadOnly(True)
        self.tabs.addTab(self.atoms_table, "Atoms")
        self.tabs.addTab(self.bonds_table, "Bonds")
        self.tabs.addTab(self.angles_table, "Angles")
        self.tabs.addTab(self.torsions_table, "Torsions")
        self.tabs.addTab(self.warnings_text, "Warnings")
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        layout.addWidget(splitter, 1)

        bottom = QHBoxLayout()
        bottom.addWidget(QLabel("3D view:"))
        self.display_combo = QComboBox()
        for key, label in display.DISPLAY_MODES.items():
            self.display_combo.addItem(label, key)
        bottom.addWidget(self.display_combo)
        bottom.addStretch(1)
        self.save_mdff_button = QPushButton("Save MDFF JSON...")
        self.save_mdff_button.setToolTip(
            "The hand-off file the GROMACS / OpenMM / LAMMPS generators read"
        )
        self.save_gmx_button = QPushButton("Save GROMACS...")
        self.save_omm_button = QPushButton("Save OpenMM XML...")
        self.save_lmp_button = QPushButton("Save LAMMPS Data...")
        close_button = QPushButton("Close")
        for button in (
            self.save_mdff_button,
            self.save_gmx_button,
            self.save_omm_button,
            self.save_lmp_button,
        ):
            bottom.addWidget(button)
            button.setEnabled(False)
        bottom.addWidget(close_button)
        layout.addLayout(bottom)

        self.build_button.clicked.connect(self.run_build)
        self.import_button.clicked.connect(self.import_charges)
        self.types_table.itemSelectionChanged.connect(self.show_selected_type)
        for table in (
            self.atoms_table,
            self.bonds_table,
            self.angles_table,
            self.torsions_table,
        ):
            table.itemChanged.connect(self._cell_edited)
        self.display_combo.currentIndexChanged.connect(self._display_changed)
        self.save_mdff_button.clicked.connect(self.save_mdff)
        self.save_gmx_button.clicked.connect(self.save_gromacs)
        self.save_omm_button.clicked.connect(self.save_openmm)
        self.save_lmp_button.clicked.connect(self.save_lammps)
        close_button.clicked.connect(self.close)
        for combo in (self.ff_combo, self.charge_combo, self.water_combo):
            combo.currentIndexChanged.connect(self._settings_changed)
        for check in (self.constrain_check, self.cross_check):
            check.toggled.connect(self._settings_changed)

        self.apply_settings(self.persistent_settings)
        if restored and restored.get("mdff"):
            self.restore_project_state(restored)

    # -- helpers -----------------------------------------------------------------

    @staticmethod
    def _missing_tool_text(key):
        return {
            "gaff2": "Needs acpype (AmberTools) on PATH",
            "gaff": "Needs acpype (AmberTools) on PATH",
            "omff-gaff": "Needs openmmforcefields and openff-toolkit",
            "am1bcc": "Needs AmberTools antechamber and sqm on PATH",
            "nagl": "Needs openff-toolkit and openff-nagl",
            "xtb": "Needs the xtb executable on PATH",
        }.get(key, "Not available")

    @staticmethod
    def _disable_last(combo, tooltip):
        model = combo.model()
        item = model.item(combo.count() - 1) if hasattr(model, "item") else None
        if item is not None:
            item.setEnabled(False)
        combo.setItemData(combo.count() - 1, tooltip, Qt.ItemDataRole.ToolTipRole)

    @staticmethod
    def _table(columns):
        table = QTableWidget(0, len(columns))
        table.setHorizontalHeaderLabels(columns)
        table.verticalHeader().setVisible(False)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        table.horizontalHeader().setStretchLastSection(True)
        table.setFont(QFont("Courier New", 9))
        return table

    def _set_combo(self, combo, key):
        index = combo.findData(key)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _molecule(self):
        try:
            return self.get_molecule() if self.get_molecule is not None else None
        except Exception:  # noqa: BLE001 - pragma: no cover - host API guard
            return None

    def _notify(self):
        if self.mark_modified is not None:
            try:
                self.mark_modified()
            except Exception:
                logger.debug("mark_project_modified failed", exc_info=True)

    # -- settings ---------------------------------------------------------------

    def apply_settings(self, settings):
        settings = settings or {}
        self._updating = True
        try:
            self._set_combo(self.ff_combo, settings.get("force_field", "sage"))
            self._set_combo(self.charge_combo, settings.get("charge_method", "auto"))
            self._set_combo(self.water_combo, settings.get("water_model", "TIP3P"))
            self.constrain_check.setChecked(bool(settings.get("constrain_h", True)))
            self.cross_check.setChecked(
                bool(settings.get("cross_check", True)) and self.cross_check.isEnabled()
            )
            self._set_combo(self.display_combo, settings.get("display", "none"))
        finally:
            self._updating = False

    def read_settings(self):
        return {
            "force_field": self.ff_combo.currentData(),
            "charge_method": self.charge_combo.currentData(),
            "water_model": self.water_combo.currentData(),
            "constrain_h": self.constrain_check.isChecked(),
            "cross_check": self.cross_check.isChecked(),
            "display": self.display_combo.currentData(),
        }

    def _settings_changed(self, *_args):
        if self._updating:
            return
        self.persistent_settings.update(self.read_settings())
        self._notify()

    # -- building ----------------------------------------------------------------

    def run_build(self):
        mol = self._molecule()
        if mol is None or mol.GetNumAtoms() == 0:
            QMessageBox.warning(
                self, "Force Field Builder", "There is no molecule in the editor."
            )
            return None
        mol = self._offer_reorder(mol)
        options = self.read_settings()
        imported = None
        if options["charge_method"] == "imported":
            imported = self.imported_charges
            if imported is None:
                QMessageBox.warning(
                    self, "Force Field Builder", "Import a charge file first."
                )
                return None
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            result = builder.build(
                mol, options, imported_charges=imported, version=self.version
            )
        except builder.BuildError as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Force Field Builder", str(exc))
            return None
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            logger.exception("Force Field Builder: build failed")
            QMessageBox.critical(
                self, "Force Field Builder", f"Parameter assignment failed:\n{exc}"
            )
            return None
        QApplication.restoreOverrideCursor()
        result.edited = False
        self.result = result
        self.fingerprint = display.molecule_fingerprint(mol)
        self.persistent_settings.update(options)
        self._show_result()
        self._notify()
        return result

    def _offer_reorder(self, mol):
        """Offer to make each molecule's atoms contiguous (needed for one type per molecule)."""
        from rdkit import Chem

        n_frags = len(Chem.GetMolFrags(mol, sanitizeFrags=False))
        if self.set_molecule is None or n_frags < 2:
            return mol
        if len(topology.fragment_blocks(mol)) == n_frags:
            return mol
        answer = QMessageBox.question(
            self,
            "Force Field Builder",
            f"The {n_frags} molecules in the editor have interleaved atom numbers "
            "(the 3D conversion puts all hydrogens last), so they cannot be listed "
            "molecule by molecule for MD.\n\nReorder the atoms molecule by molecule? "
            "Coordinates do not change and the step can be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return mol
        reordered = topology.reorder_by_molecule(mol)
        try:
            self.set_molecule(reordered)
        except Exception:  # pragma: no cover - host API guard
            logger.warning(
                "Force Field Builder: could not reorder the atoms", exc_info=True
            )
            return mol
        current = self._molecule()
        if current is not None and current.GetNumAtoms() == reordered.GetNumAtoms():
            return current
        return reordered

    def _show_result(self):
        result = self.result
        n_atoms = len(result.atom_types)
        n_warn = sum(len(t.warnings) for t in result.types)
        own = all(t.charges_from_force_field for t in result.types)
        text = (
            f"{n_atoms} atoms in {sum(t.count for t in result.types)} molecules, "
            f"{len(result.types)} molecule types. "
            + (
                "Charges follow the force field's own scheme."
                if own
                else "Some charges are NOT the force field's own scheme (see Charges column)."
            )
        )
        if n_warn:
            text += f" {n_warn} notes in Warnings."
        self.status_label.setText(text)
        self._fill_types()
        for button in (
            self.save_mdff_button,
            self.save_gmx_button,
            self.save_omm_button,
            self.save_lmp_button,
        ):
            button.setEnabled(True)
        if self.types_table.rowCount():
            self.types_table.selectRow(0)
        self._fill_warnings()
        self.apply_display()

    def _fill_types(self):
        self._filling = True
        try:
            types = self.result.types
            self.types_table.setRowCount(len(types))
            mtypes = {mt["name"]: mt for mt in self.result.mdff["molecule_types"]}
            for row, info in enumerate(types):
                mt = mtypes[info.name]
                net = sum(a["charge"] for a in mt["atoms"])
                charge_text = info.charge_method + (
                    "" if info.charges_from_force_field else "  [not FF scheme]"
                )
                notes = f"{len(info.warnings)} note(s)" if info.warnings else ""
                values = [
                    info.name,
                    info.kind,
                    info.count,
                    len(mt["atoms"]),
                    f"{net:+.4f}",
                    charge_text,
                    notes,
                ]
                for col, value in enumerate(values):
                    self.types_table.setItem(row, col, _item(value))
        finally:
            self._filling = False

    def _fill_warnings(self):
        lines = list(self.result.warnings)
        for info in self.result.types:
            for warning in info.warnings:
                lines.append(f"{info.name}: {warning}")
        if getattr(self.result, "edited", False):
            lines.append("Parameters were edited by hand after assignment.")
        self.warnings_text.setPlainText("\n".join(lines) if lines else "No warnings.")
        self.tabs.setTabText(4, f"Warnings ({len(lines)})" if lines else "Warnings")

    def selected_type(self):
        if self.result is None:
            return None
        rows = (
            self.types_table.selectionModel().selectedRows()
            if self.types_table.selectionModel()
            else []
        )
        row = rows[0].row() if rows else 0
        if row >= len(self.result.types):
            return None
        name = self.result.types[row].name
        return next(
            mt for mt in self.result.mdff["molecule_types"] if mt["name"] == name
        )

    def show_selected_type(self):
        mt = self.selected_type()
        if mt is None:
            return
        self._filling = True
        try:
            atoms = mt["atoms"]

            def label(*idx):
                return "-".join(atoms[i]["name"] for i in idx)

            self.atoms_table.setRowCount(len(atoms))
            for row, atom in enumerate(atoms):
                values = [
                    (row + 1, False),
                    (atom["name"], False),
                    (atom["element"], False),
                    (atom["type"], False),
                    (f"{atom['charge']:.6f}", True),
                    (f"{atom['mass']:.4f}", False),
                    (f"{atom['sigma']:.6f}", True),
                    (f"{atom['epsilon']:.6f}", True),
                ]
                for col, (value, editable) in enumerate(values):
                    self.atoms_table.setItem(row, col, _item(value, editable, row))
            bonds = mt.get("bonds", [])
            self.bonds_table.setRowCount(len(bonds))
            for row, b in enumerate(bonds):
                values = [
                    (b["i"] + 1, False),
                    (b["j"] + 1, False),
                    (label(b["i"], b["j"]), False),
                    (f"{b['r0']:.6f}", True),
                    (f"{b['k']:.2f}", True),
                    (b.get("id") or "", False),
                ]
                for col, (value, editable) in enumerate(values):
                    self.bonds_table.setItem(row, col, _item(value, editable, row))
            angles = mt.get("angles", [])
            self.angles_table.setRowCount(len(angles))
            for row, a in enumerate(angles):
                values = [
                    (a["i"] + 1, False),
                    (a["j"] + 1, False),
                    (a["k_atom"] + 1, False),
                    (label(a["i"], a["j"], a["k_atom"]), False),
                    (f"{a['theta0']:.4f}", True),
                    (f"{a['k']:.3f}", True),
                    (a.get("id") or "", False),
                ]
                for col, (value, editable) in enumerate(values):
                    self.angles_table.setItem(row, col, _item(value, editable, row))
            torsions = mt.get("torsions", [])
            self.torsions_table.setRowCount(len(torsions))
            for row, t in enumerate(torsions):
                values = [
                    (t["i"] + 1, False),
                    (t["j"] + 1, False),
                    (t["k_atom"] + 1, False),
                    (t["l"] + 1, False),
                    ("improper" if t.get("improper") else "proper", False),
                    (t["periodicity"], True),
                    (f"{t['phase']:.2f}", True),
                    (f"{t['k']:.6f}", True),
                    (t.get("id") or "", False),
                ]
                for col, (value, editable) in enumerate(values):
                    self.torsions_table.setItem(row, col, _item(value, editable, row))
        finally:
            self._filling = False

    # -- manual edits ----------------------------------------------------------------

    def _cell_edited(self, item):
        if self._filling or self.result is None:
            return
        mt = self.selected_type()
        table = item.tableWidget()
        row = item.data(Qt.ItemDataRole.UserRole)
        col = item.column()
        try:
            value = float(item.text())
        except ValueError:
            self.show_selected_type()
            return
        if table is self.atoms_table:
            atom = mt["atoms"][row]
            if col == 4:
                atom["charge"] = value
            elif col in (6, 7):
                key = "sigma" if col == 6 else "epsilon"
                # An atom type has one LJ pair: change every atom of that type.
                for other in self.result.mdff["molecule_types"]:
                    for a in other["atoms"]:
                        if a["type"] == atom["type"]:
                            a[key] = value
        elif table is self.bonds_table:
            mt["bonds"][row]["r0" if col == 3 else "k"] = value
        elif table is self.angles_table:
            mt["angles"][row]["theta0" if col == 4 else "k"] = value
        elif table is self.torsions_table:
            key = {5: "periodicity", 6: "phase", 7: "k"}[col]
            mt["torsions"][row][key] = int(value) if key == "periodicity" else value
        self.result.edited = True
        self.result.atom_types, self.result.atom_charges = display.per_atom_data(
            self.result.mdff
        )
        self._fill_types()
        self._fill_warnings()
        if (
            table is self.atoms_table
            and col == 4
            and self.display_combo.currentData() != "none"
        ):
            self.apply_display()
        self._notify()

    # -- charges import ----------------------------------------------------------------

    def import_charges(self, path=None):
        if not path:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Import Charges",
                "",
                "Charge files (*.out *.log *.json *.chg *.txt *.csv);;All files (*)",
            )
        if not path:
            return False
        mol = self._molecule()
        try:
            values, elements, description = charges.parse_charge_file(path)
        except (OSError, ValueError, charges.ChargeError) as exc:
            QMessageBox.warning(
                self, "Force Field Builder", f"Could not read charges:\n{exc}"
            )
            return False
        if mol is not None and len(values) != mol.GetNumAtoms():
            QMessageBox.warning(
                self,
                "Force Field Builder",
                f"{description} has {len(values)} charges but the molecule has {mol.GetNumAtoms()} atoms.",
            )
            return False
        if mol is not None and elements:
            mine = [a.GetSymbol().upper() for a in mol.GetAtoms()]
            theirs = [str(e).upper() for e in elements]
            if mine != theirs:
                QMessageBox.warning(
                    self,
                    "Force Field Builder",
                    "The element order in the charge file differs from the molecule.",
                )
                return False
        self.imported_charges = values
        self.imported_label = description
        self.import_label.setText(
            f"Imported: {description} ({len(values)} atoms, total {sum(values):+.4f})"
        )
        self._set_combo(self.charge_combo, "imported")
        return True

    # -- 3D display -------------------------------------------------------------------

    def _display_changed(self, *_args):
        self._settings_changed()
        self.apply_display()

    def apply_display(self):
        mode = self.display_combo.currentData()
        if self.context is None:
            return
        self.clear_display()
        if mode == "none" or self.result is None:
            return
        mol = self._molecule()
        if mol is None or display.molecule_fingerprint(mol) != self.fingerprint:
            return
        try:
            controller = self.context.get_3d_controller()
            if mode in ("charge_colours", "charge_labels"):
                controller.set_atom_colors(
                    display.charge_colours(self.result.atom_charges)
                )
                self._display_applied = True
            if mode in ("charge_labels", "type_labels"):
                labels = (
                    [f"{q:+.2f}" for q in self.result.atom_charges]
                    if mode == "charge_labels"
                    else list(self.result.atom_types)
                )
                positions = display.molecule_positions(mol)
                if positions and len(positions) <= display.MAX_LABELS:
                    plotter = self.context.plotter
                    if plotter is not None:
                        plotter.add_point_labels(
                            positions,
                            labels,
                            name=LABEL_ACTOR,
                            font_size=11,
                            point_size=0,
                            shape_opacity=0.35,
                            always_visible=True,
                            text_color="black",
                        )
                        self._display_applied = True
                elif positions:
                    self.status_label.setText(
                        f"Labels are shown for up to {display.MAX_LABELS} atoms; use colours for this system."
                    )
        except Exception as exc:  # noqa: BLE001 - pragma: no cover - host 3D API guard
            logger.warning("Force Field Builder: 3D display failed: %s", exc)

    def clear_display(self):
        if self.context is None or not self._display_applied:
            return
        try:
            plotter = self.context.plotter
            if plotter is not None:
                plotter.remove_actor(LABEL_ACTOR)
            mol = self._molecule()
            if mol is not None:
                self.context.get_3d_controller().set_atom_colors(
                    {i: None for i in range(mol.GetNumAtoms())}
                )
        except Exception as exc:  # noqa: BLE001 - pragma: no cover - host 3D API guard
            logger.warning(
                "Force Field Builder: could not clear the 3D display: %s", exc
            )
        self._display_applied = False

    # -- output ------------------------------------------------------------------------

    def _check_current(self):
        if self.result is None:
            return False
        mol = self._molecule()
        if mol is not None and display.molecule_fingerprint(mol) != self.fingerprint:
            answer = QMessageBox.question(
                self,
                "Force Field Builder",
                "The molecule changed since the parameters were assigned. Save the old parameters anyway?",
            )
            return answer == QMessageBox.StandardButton.Yes
        return True

    def _write(self, path, text):
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)

    def save_mdff(self, path=None):
        if not self._check_current():
            return None
        if not path:
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Save MDFF JSON",
                "system.mdff.json",
                "MoleditPy force field (*.mdff.json);;JSON (*.json)",
            )
        if not path:
            return None
        try:
            self._write(path, dumps(self.result.mdff))
        except (OSError, ValueError) as exc:
            QMessageBox.critical(
                self, "Force Field Builder", f"Could not write the file:\n{exc}"
            )
            return None
        self.status_label.setText(f"Wrote {path}")
        return path

    def save_gromacs(self, path=None):
        if not self._check_current():
            return None
        if not path:
            path, _ = QFileDialog.getSaveFileName(
                self, "Save GROMACS Topology", "topol.top", "GROMACS topology (*.top)"
            )
        if not path:
            return None
        folder = os.path.dirname(os.path.abspath(path))
        base = os.path.splitext(os.path.basename(path))[0]
        try:
            files = writers.gromacs_files(self.result.mdff, base)
            for name, text in files.items():
                self._write(os.path.join(folder, name), text)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(
                self, "Force Field Builder", f"Could not write the files:\n{exc}"
            )
            return None
        self.status_label.setText(f"Wrote {len(files)} files to {folder}")
        return [os.path.join(folder, name) for name in files]

    def save_openmm(self, path=None):
        if not self._check_current():
            return None
        if not path:
            path, _ = QFileDialog.getSaveFileName(
                self, "Save OpenMM ForceField XML", "system_ff.xml", "XML (*.xml)"
            )
        if not path:
            return None
        try:
            self._write(path, writers.openmm_xml(self.result.mdff))
        except (OSError, ValueError) as exc:
            QMessageBox.critical(
                self, "Force Field Builder", f"Could not write the file:\n{exc}"
            )
            return None
        self.status_label.setText(f"Wrote {path}")
        return path

    def save_lammps(self, path=None):
        if not self._check_current():
            return None
        mol = self._molecule()
        positions = display.molecule_positions(mol)
        if len(positions) != len(self.result.atom_types):
            QMessageBox.warning(
                self,
                "Force Field Builder",
                "The LAMMPS data file needs the current 3D coordinates of every atom.",
            )
            return None
        cell = display.molecule_cell(mol)
        if not path:
            path, _ = QFileDialog.getSaveFileName(
                self, "Save LAMMPS Data", "system.data", "LAMMPS data (*.data *.lmp)"
            )
        if not path:
            return None
        padding = float(self.persistent_settings.get("lammps_padding", 10.0))
        try:
            self._write(
                path, writers.lammps_data(self.result.mdff, positions, cell, padding)
            )
        except (OSError, ValueError) as exc:
            QMessageBox.critical(
                self, "Force Field Builder", f"Could not write the file:\n{exc}"
            )
            return None
        box = (
            "the moleditpy_cell box"
            if cell
            else f"a non-periodic bounding box with {padding:g} A padding"
        )
        self.status_label.setText(f"Wrote {path} using {box}")
        return path

    # -- project state -------------------------------------------------------------------

    def project_state(self):
        if self.result is None:
            return None
        return display.result_to_state(self.result, self.fingerprint)

    def restore_project_state(self, state):
        try:
            self.result = display.state_to_result(state)
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning(
                "Force Field Builder: saved parameters are unreadable: %s", exc
            )
            return
        self.fingerprint = state.get("fingerprint")
        self._show_result()
        self.status_label.setText(
            self.status_label.text() + " (restored from the project)"
        )

    def mark_stale(self):
        if self.result is not None:
            self.status_label.setText(
                "A new document was opened: press Assign Parameters for its molecule."
            )

    def closeEvent(self, event):
        self.clear_display()
        super().closeEvent(event)
