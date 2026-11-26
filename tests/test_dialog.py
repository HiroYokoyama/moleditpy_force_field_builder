"""The dialog, driven headless with real PySide6 (skipped without it)."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers import HAS_QT, HAS_RDKIT, combine, mol3d, shifted, water_box

pytestmark = pytest.mark.skipif(
    not (HAS_QT and HAS_RDKIT), reason="PySide6 and RDKit needed"
)


def _read(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read()


class FakeController:
    def __init__(self):
        self.calls = []

    def set_atom_colors(self, colors):
        self.calls.append(dict(colors))


class FakePlotter:
    def __init__(self):
        self.labels = None
        self.removed = []

    def add_point_labels(self, points, labels, **kwargs):
        self.labels = (list(points), list(labels), kwargs)

    def remove_actor(self, name):
        self.removed.append(name)


class FakeContext:
    def __init__(self):
        self.controller = FakeController()
        self.plotter = FakePlotter()

    def get_3d_controller(self):
        return self.controller


@pytest.fixture
def mol():
    return combine(mol3d("CCO"), shifted(water_box(2), 6))


@pytest.fixture
def dialog(qapp, mol, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    from force_field_builder.main_dialog import ForceFieldBuilderDialog

    messages = []
    for name in ("warning", "critical", "information"):
        monkeypatch.setattr(
            QMessageBox,
            name,
            staticmethod(lambda *a, _n=name: messages.append((_n, a[2]))),
        )
    state = {"mol": mol}
    ctx = FakeContext()
    settings = {}
    dlg = ForceFieldBuilderDialog(
        context=ctx, persistent_settings=settings, get_molecule=lambda: state["mol"]
    )
    dlg.messages = messages
    dlg.state = state
    dlg.ctx = ctx
    yield dlg
    dlg.close()


def test_build_fills_tables(dialog):
    dialog._set_combo(dialog.charge_combo, "mmff94")
    result = dialog.run_build()
    assert result is not None
    assert dialog.types_table.rowCount() == 2
    assert dialog.types_table.item(0, 0).text() == "MOL"
    assert dialog.types_table.item(1, 2).text() == "2"
    assert dialog.atoms_table.rowCount() == 9
    assert "[not FF scheme]" in dialog.types_table.item(0, 5).text()
    assert dialog.save_mdff_button.isEnabled()
    assert dialog.persistent_settings["charge_method"] == "mmff94"


def test_no_molecule_warns(dialog):
    dialog.state["mol"] = None
    assert dialog.run_build() is None
    assert dialog.messages and dialog.messages[0][0] == "warning"


def test_editing_a_charge_updates_the_document(dialog):
    dialog.run_build()
    item = dialog.atoms_table.item(0, 4)
    item.setText("0.5")
    assert dialog.result.mdff["molecule_types"][0]["atoms"][0]["charge"] == 0.5
    assert dialog.result.atom_charges[0] == 0.5
    assert dialog.result.edited
    assert "edited by hand" in dialog.warnings_text.toPlainText()
    item.setText("not a number")  # ignored, table restored
    assert dialog.result.mdff["molecule_types"][0]["atoms"][0]["charge"] == 0.5


def test_editing_sigma_changes_the_whole_atom_type(dialog):
    dialog.run_build()
    mt = dialog.result.mdff["molecule_types"][0]
    h_rows = [
        n for n, a in enumerate(mt["atoms"]) if a["type"] == mt["atoms"][3]["type"]
    ]
    dialog.atoms_table.item(3, 6).setText("0.2")
    assert all(mt["atoms"][n]["sigma"] == 0.2 for n in h_rows)


def test_bond_and_torsion_edits(dialog):
    dialog.run_build()
    dialog.tabs.setCurrentIndex(1)
    dialog.bonds_table.item(0, 3).setText("0.2")
    assert dialog.result.mdff["molecule_types"][0]["bonds"][0]["r0"] == 0.2
    dialog.torsions_table.item(0, 5).setText("4")
    assert dialog.result.mdff["molecule_types"][0]["torsions"][0]["periodicity"] == 4


def test_saving_all_formats(dialog, tmp_path):
    dialog.run_build()
    path = dialog.save_mdff(str(tmp_path / "sys.mdff.json"))
    data = json.loads(_read(path))
    assert data["format"] == "moleditpy-mdff"
    files = dialog.save_gromacs(str(tmp_path / "topol.top"))
    assert {os.path.basename(f) for f in files} == {
        "topol.top",
        "topol_MOL.itp",
        "topol_SOL.itp",
    }
    assert dialog.save_openmm(str(tmp_path / "ff.xml"))
    lmp = dialog.save_lammps(str(tmp_path / "s.data"))
    assert "15 atoms" in _read(lmp)
    assert "bounding box" in dialog.status_label.text()


def test_lammps_uses_the_molecule_cell(dialog, tmp_path):
    cell = {
        "lattice": [[20, 0, 0], [0, 20, 0], [0, 0, 20]],
        "origin": [-5, -5, -5],
        "pbc": [True] * 3,
    }
    dialog.state["mol"].SetProp("moleditpy_cell", json.dumps(cell))
    dialog.run_build()
    text = _read(dialog.save_lammps(str(tmp_path / "c.data")))
    assert "0.0 20 xlo xhi" in text
    assert "moleditpy_cell" in dialog.status_label.text()


def test_changed_molecule_asks_before_saving(dialog, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    dialog.run_build()
    dialog.state["mol"] = mol3d("CCCC")
    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a: QMessageBox.StandardButton.No)
    )
    assert dialog.save_mdff(str(tmp_path / "x.json")) is None


def test_3d_display_modes(dialog):
    dialog.run_build()
    dialog._set_combo(dialog.display_combo, "charge_labels")
    colours = dialog.ctx.controller.calls[-1]
    assert len(colours) == 15 and all(c.startswith("#") for c in colours.values())
    points, labels, kwargs = dialog.ctx.plotter.labels
    assert (
        len(points) == len(labels) == 15
        and kwargs["name"] == "force_field_builder_labels"
    )
    dialog._set_combo(dialog.display_combo, "type_labels")
    assert dialog.ctx.plotter.labels[1][9] == "OW_TIP3P"
    dialog._set_combo(dialog.display_combo, "none")
    assert "force_field_builder_labels" in dialog.ctx.plotter.removed
    assert all(v is None for v in dialog.ctx.controller.calls[-1].values())


def test_project_state_round_trip(qapp, dialog):
    from force_field_builder.main_dialog import ForceFieldBuilderDialog

    dialog.run_build()
    dialog.atoms_table.item(0, 4).setText("0.25")
    state = json.loads(json.dumps(dialog.project_state()))
    other = ForceFieldBuilderDialog(
        get_molecule=lambda: dialog.state["mol"], restored=state
    )
    assert other.result.mdff["molecule_types"][0]["atoms"][0]["charge"] == 0.25
    assert other.types_table.rowCount() == 2
    assert "restored" in other.status_label.text()
    other.close()


def test_import_charges(dialog, tmp_path):
    mol = dialog.state["mol"]
    values = [0.0] * mol.GetNumAtoms()
    values[0], values[1] = 0.3, -0.3
    path = tmp_path / "q.txt"
    path.write_text("\n".join(str(v) for v in values), encoding="utf-8")
    assert dialog.import_charges(str(path))
    assert dialog.charge_combo.currentData() == "imported"
    result = dialog.run_build()
    assert result.mdff["molecule_types"][0]["atoms"][0]["charge"] == pytest.approx(0.3)
    bad = tmp_path / "bad.txt"
    bad.write_text("0.1\n0.2\n", encoding="utf-8")
    assert not dialog.import_charges(str(bad))


def test_settings_round_trip(dialog):
    dialog.apply_settings(
        {"water_model": "OPC3", "constrain_h": False, "force_field": "sage"}
    )
    settings = dialog.read_settings()
    assert settings["water_model"] == "OPC3" and settings["constrain_h"] is False


def _interleaved():
    """Ethanol + water + Na+ with all hydrogens last, as after MoleditPy's 3D conversion."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles("CCO.O.[Na+]"))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    return mol


def test_interleaved_molecules_are_reordered_on_request(dialog, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    dialog.state["mol"] = _interleaved()
    applied = []

    def set_molecule(mol):
        applied.append(mol)
        dialog.state["mol"] = mol

    dialog.set_molecule = set_molecule
    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a: QMessageBox.StandardButton.Yes)
    )
    result = dialog.run_build()
    assert len(applied) == 1
    assert [t.name for t in result.types] == ["MOL", "SOL", "NA"]
    assert result.type_info("SOL").charge_method == "TIP3P water model"


def test_interleaved_molecules_kept_when_declined(dialog, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    dialog.state["mol"] = _interleaved()
    dialog.set_molecule = lambda mol: pytest.fail("must not change the molecule")
    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a: QMessageBox.StandardButton.No)
    )
    result = dialog.run_build()
    assert len(result.types) == 1
    assert any("interleaved" in w for w in result.warnings)


def test_import_charges_of_one_molecule(dialog, tmp_path):
    ethanol = mol3d("CCO")
    rows = [
        f"{a.GetSymbol()} {0.1 if a.GetSymbol() == 'C' else -0.1}"
        for a in ethanol.GetAtoms()
    ]
    path = tmp_path / "ethanol.chg"
    path.write_text("\n".join(rows), encoding="utf-8")
    assert dialog.import_charges(str(path))
    assert "applied to 1 matching copy" in dialog.import_label.text()
    result = dialog.run_build()
    assert result is not None
    assert result.types[0].charge_method.startswith("Imported")
    assert result.types[1].kind == "water"


def test_import_asks_which_charge_set(dialog, tmp_path, monkeypatch):
    from force_field_builder import charges, main_dialog

    n = dialog.state["mol"].GetNumAtoms()
    sets = [
        ([0.0] * n, None, "job.log: Gaussian ESP-fitted charges"),
        ([0.1] + [0.0] * (n - 1), None, "job.log: Gaussian Mulliken charges"),
    ]
    monkeypatch.setattr(charges, "charge_sets", lambda _path, _problems=None: sets)
    asked = []

    def get_item(_parent, _title, _text, labels, current, _editable):
        asked.append((labels, current))
        return labels[1], True

    monkeypatch.setattr(main_dialog.QInputDialog, "getItem", get_item)
    assert dialog.import_charges(str(tmp_path / "job.log"))
    assert asked == [(["Gaussian ESP-fitted charges", "Gaussian Mulliken charges"], 0)]
    assert dialog.imported_label == "job.log: Gaussian Mulliken charges"
    monkeypatch.setattr(main_dialog.QInputDialog, "getItem", lambda *a: ("", False))
    assert not dialog.import_charges(str(tmp_path / "job.log"))  # cancelled
