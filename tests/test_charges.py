"""Charge methods and charge-file import."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers import HAS_RDKIT, mol3d

from force_field_builder import charges

# Real ORCA 6 layout (table header, rows "idx El : q", closing rule).
ORCA_OUT = """
---------------------
MULLIKEN ATOMIC CHARGES
-----------------------
   0 O :   -0.400000
   1 H :    0.200000
   2 H :    0.200000
Sum of atomic charges:    0.0000000

CHELPG CHARGES GENERATION
CHELPG Charges
--------------------------------
  0   O   :      -0.709308
  1   H   :       0.354383
  2   H   :       0.354925
--------------------------------
Total charge:     0.000000
--------------------------------

CHELPG charges calculated...
"""


def test_normalise_spreads_the_residual():
    values, residual = charges.normalise([0.1, 0.1, 0.1], 0)
    assert residual == pytest.approx(-0.3)
    assert sum(values) == pytest.approx(0.0)
    assert values[0] == pytest.approx(0.0)
    assert charges.normalise([], 1) == ([], 0.0)


def test_orca_prefers_esp_charges(tmp_path):
    path = tmp_path / "water.out"
    path.write_text(ORCA_OUT, encoding="utf-8")
    values, elements, description = charges.parse_charge_file(str(path))
    assert values == [-0.709308, 0.354383, 0.354925]
    assert elements == ["O", "H", "H"]
    assert "CHELPG" in description


def test_orca_mulliken_only(tmp_path):
    path = tmp_path / "m.out"
    path.write_text(ORCA_OUT.split("CHELPG CHARGES GENERATION")[0], encoding="utf-8")
    values, _elements, description = charges.parse_charge_file(str(path))
    assert values == [-0.4, 0.2, 0.2]
    assert "Mulliken" in description


def test_pyscf_properties_json(tmp_path):
    path = tmp_path / "properties.json"
    path.write_text(
        json.dumps(
            {"mulliken_charges": [-0.5, 0.25, 0.25], "atom_symbols": ["O", "H", "H"]}
        )
    )
    values, elements, description = charges.parse_charge_file(str(path))
    assert values == [-0.5, 0.25, 0.25] and elements == ["O", "H", "H"]
    assert "mulliken_charges" in description


def test_plain_and_multiwfn_style_lists(tmp_path):
    plain = tmp_path / "q.txt"
    plain.write_text("# charges\n-0.8\n0.4\n0.4\n", encoding="utf-8")
    assert charges.parse_charge_file(str(plain))[0] == [-0.8, 0.4, 0.4]
    chg = tmp_path / "w.chg"
    chg.write_text(
        "O 0.0 0.0 0.0 -0.8\nH 0.9 0.0 0.0 0.4\nH -0.3 0.9 0.0 0.4\n", encoding="utf-8"
    )
    values, elements, _ = charges.parse_charge_file(str(chg))
    assert values == [-0.8, 0.4, 0.4] and elements == ["O", "H", "H"]
    empty = tmp_path / "e.txt"
    empty.write_text("nothing here\n", encoding="utf-8")
    with pytest.raises(charges.ChargeError):
        charges.parse_charge_file(str(empty))


def test_available_methods_always_has_builtins():
    methods = charges.available_methods()
    assert {"auto", "gasteiger", "mmff94", "imported"} <= set(methods)


@pytest.mark.skipif(not HAS_RDKIT, reason="RDKit not installed")
def test_builtin_methods_give_one_charge_per_atom():
    mol = mol3d("CC(=O)O")
    for method in ("gasteiger", "mmff94"):
        values = charges.compute(method, mol)
        assert len(values) == mol.GetNumAtoms()
        assert sum(values) == pytest.approx(0.0, abs=0.05)
    with pytest.raises(charges.ChargeError):
        charges.compute("gasteiger", mol3d("[Na+]"))
    with pytest.raises(charges.ChargeError):
        charges.compute("nope", mol)
    with pytest.raises(charges.ChargeError):
        charges.compute("imported", mol, imported=[0.0])


@pytest.mark.skipif(not HAS_RDKIT, reason="RDKit not installed")
def test_mol2_round_trip(tmp_path):
    mol = mol3d("c1ccccc1O")
    values = [0.01 * i for i in range(mol.GetNumAtoms())]
    path = tmp_path / "x.mol2"
    charges.write_mol2(mol, str(path), charges=values)
    assert charges.read_mol2_charges(str(path)) == pytest.approx(values)
    text = path.read_text(encoding="utf-8")
    assert " ar" in text  # aromatic bonds keep their mol2 type


def test_average_equivalent_keeps_total_and_groups_equivalent_atoms():
    mol = mol3d("CC(=O)[O-]")  # acetate: both oxygens and the three H are equivalent
    values = [-0.3, 0.7, -0.7, -0.9, 0.05, 0.1, 0.05]
    out, changed = charges.average_equivalent(mol, values)
    assert sum(out) == pytest.approx(sum(values))
    assert out[2] == pytest.approx(out[3]) == pytest.approx(-0.8)
    assert out[4] == pytest.approx(out[5]) == pytest.approx(out[6])
    assert changed == 5
    assert charges.average_equivalent(mol, out)[1] == 0
