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


# Layouts below follow each program's own output (the Gaussian ESP block as printed
# by Gaussian 16 for Pop=MK; headers cross-checked with cclib's
# parsers and, for xtb, its property printer). Water: O, H, H.
GAUSSIAN_LOG = """
 Entering Gaussian System, Link 0=g16
 Mulliken charges:
               1
     1  O   -0.758364
     2  H    0.379182
     3  H    0.379182
 Sum of Mulliken charges =  -0.00000
 Mulliken charges with hydrogens summed into heavy atoms:
               1
     1  O    0.000000
 Sum of Mulliken charges with hydrogens summed into heavy atoms =   0.00000
 Hirshfeld charges, spin densities, dipoles, and CM5 charges using IRadAn=      4:
                Q-H        S-H        Dx         Dy         Dz        Q-CM5
     1  O   -0.344015   0.000000   0.000000   0.000000  -0.218331  -0.668180
     2  H    0.172008   0.000000   0.000000   0.123455   0.081733   0.334090
     3  H    0.172007   0.000000   0.000000  -0.123455   0.081733   0.334090
       Tot   0.000000   0.000000   0.000000   0.000000  -0.054865   0.000000
 Charges from ESP fit, RMS=   0.00133 RRMS=   0.10427:
 ESP charges:
               1
     1  O   -0.834020
     2  H    0.417010
     3  H    0.417010
 Sum of ESP charges =   0.00000
 ESP charges with hydrogens summed into heavy atoms:
               1
     1  O    0.000000
 Charge=   0.00000 Dipole=    0.0000    0.0000   -2.4034 Tot=    2.4034
 -----------------------------------------------------------------

 Summary of Natural Population Analysis:

                                       Natural Population
                Natural  -----------------------------------------------
    Atom  No    Charge         Core      Valence    Rydberg      Total
 -----------------------------------------------------------------------
      O    1   -0.91432      1.99991     6.90537    0.00904     8.91432
      H    2    0.45716      0.00000     0.54064    0.00220     0.54284
      H    3    0.45716      0.00000     0.54064    0.00220     0.54284
 =======================================================================
   * Total *    0.00000      1.99991     7.98665    0.01344    10.00000
"""

QCHEM_OUT = """
          Ground-State Mulliken Net Atomic Charges

     Atom                 Charge (a.u.)
  ----------------------------------------
      1 O                    -0.876191
      2 H                     0.438096
      3 H                     0.438095
  ----------------------------------------
  Sum of atomic charges =     0.000000

          Ground-State ChElPG Net Atomic Charges

     Atom                 Charge (a.u.)
  ----------------------------------------
      1 O                    -0.795000
      2 H                     0.397500
      3 H                     0.397500
  ----------------------------------------
"""

PSI4_OUT = """
  Mulliken Charges: (a.u.)
   Center  Symbol    Alpha    Beta     Spin     Total
       1     O     4.35460  4.35460  0.00000 -0.70920
       2     H     0.32270  0.32270  0.00000  0.35460
       3     H     0.32270  0.32270  0.00000  0.35460

   Total alpha =  5.00000, Total beta =  5.00000, Total charge =  0.00000
"""

XTB_OUT = """
     #   Z          covCN         q      C6AA      α(0)
     1   8 O        1.613    -0.565    25.523     8.112
     2   1 H        0.807     0.283     2.453     2.149
     3   1 H        0.807     0.282     2.453     2.149

 Mol. C6AA /au·bohr⁶  :         45.142312
"""


def _sets(tmp_path, text, name="job.out"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return charges.charge_sets(str(path))


def test_gaussian_tables_best_first(tmp_path):
    sets = _sets(tmp_path, GAUSSIAN_LOG, "water.log")
    labels = [d.split(": ", 1)[1] for _v, _e, d in sets]
    assert labels == [
        "Gaussian ESP-fitted charges",
        "Gaussian CM5 charges",
        "Gaussian Hirshfeld charges",
        "Gaussian NPA charges",
        "Gaussian Mulliken charges",
    ]
    by_label = {d.split(": ", 1)[1]: (v, e) for v, e, d in sets}
    assert by_label["Gaussian ESP-fitted charges"] == (
        [-0.83402, 0.41701, 0.41701],
        ["O", "H", "H"],
    )
    assert by_label["Gaussian CM5 charges"][0] == [-0.66818, 0.33409, 0.33409]
    assert by_label["Gaussian Hirshfeld charges"][0][0] == -0.344015
    assert by_label["Gaussian NPA charges"] == (
        [-0.91432, 0.45716, 0.45716],
        ["O", "H", "H"],
    )
    # Not the "summed into heavy atoms" table that follows.
    assert by_label["Gaussian Mulliken charges"][0] == [-0.758364, 0.379182, 0.379182]
    assert charges.parse_charge_file(str(tmp_path / "water.log"))[0][0] == -0.83402


def test_qchem_psi4_and_xtb_tables(tmp_path):
    sets = _sets(tmp_path, QCHEM_OUT)
    assert [d.split(": ", 1)[1] for _v, _e, d in sets] == [
        "Q-Chem CHELPG charges",
        "Q-Chem Mulliken charges",
    ]
    assert sets[0][0] == [-0.795, 0.3975, 0.3975] and sets[0][1] == ["O", "H", "H"]

    ((values, elements, description),) = _sets(tmp_path, PSI4_OUT)
    assert values == [-0.7092, 0.3546, 0.3546] and elements == ["O", "H", "H"]
    assert "Psi4 Mulliken" in description

    ((values, elements, description),) = _sets(tmp_path, XTB_OUT)
    assert values == [-0.565, 0.283, 0.282] and elements == ["O", "H", "H"]
    assert "xtb" in description


def test_xtb_json_and_orca_open_shell(tmp_path):
    ((values, _e, description),) = _sets(
        tmp_path, json.dumps({"partial charges": [-0.565, 0.283, 0.282]}), "xtbout.json"
    )
    assert values == [-0.565, 0.283, 0.282] and "partial charges" in description
    orca = ORCA_OUT.split("CHELPG CHARGES GENERATION")[0].replace(
        "MULLIKEN ATOMIC CHARGES\n", "MULLIKEN ATOMIC CHARGES AND SPIN POPULATIONS\n"
    )
    ((values, _e, _d),) = _sets(tmp_path, orca)
    assert values == [-0.4, 0.2, 0.2]


def test_last_table_of_a_kind_wins(tmp_path):
    first = PSI4_OUT.replace("-0.70920", "-0.60000")
    ((values, _e, _d),) = _sets(tmp_path, first + PSI4_OUT)
    assert values[0] == -0.7092


GAUSSIAN_ESP_LOG = (
    " Entering Gaussian System, Link 0=g16\n"
    " Charge =  0 Multiplicity = 1\n"
    "      Atomic Center    1 is at   0.000000  0.000000  0.117300\n"
    "      ESP Fit Center    4 is at   0.000000  0.000000  1.900000\n"
    + GAUSSIAN_LOG.split("Entering Gaussian System, Link 0=g16\n", 1)[1]
)


def _fake_antechamber(monkeypatch, written):
    import subprocess

    monkeypatch.setattr(charges.shutil, "which", lambda name: f"/amber/{name}")

    def run(cmd, cwd, **_kw):
        with open(os.path.join(cwd, "esp.log"), encoding="utf-8") as log:
            written.append((cmd, log.read()))
        with open(os.path.join(cwd, "resp.mol2"), "w") as out:
            out.write(
                "@<TRIPOS>MOLECULE\nWAT\n@<TRIPOS>ATOM\n"
                "1 O 0 0 0 oh 1 WAT -0.780000\n"
                "2 H1 0 0 0 ho 1 WAT 0.390000\n"
                "3 H2 0 0 0 ho 1 WAT 0.390000\n"
                "@<TRIPOS>BOND\n"
            )
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(charges.subprocess, "run", run)


def test_resp_from_gaussian_esp_job(tmp_path, monkeypatch):
    written = []
    _fake_antechamber(monkeypatch, written)
    sets = _sets(tmp_path, GAUSSIAN_ESP_LOG, "water esp.log")
    values, elements, description = sets[0]
    assert values == [-0.78, 0.39, 0.39] and elements == ["O", "H", "H"]
    assert description == "water esp.log: RESP charges (AmberTools)"
    assert sets[1][2].endswith("Gaussian ESP-fitted charges")
    cmd, text = written[0]
    assert cmd[cmd.index("-fi") + 1] == "gout" and cmd[cmd.index("-c") + 1] == "resp"
    assert cmd[cmd.index("-nc") + 1] == "0" and "ESP Fit Center" in text


def test_no_resp_without_esp_points_or_amber(tmp_path, monkeypatch):
    written = []
    _fake_antechamber(monkeypatch, written)
    sets = _sets(tmp_path, GAUSSIAN_LOG)  # no ESP points printed
    assert "RESP" not in sets[0][2] and not written
    monkeypatch.setattr(charges.shutil, "which", lambda name: None)
    sets = _sets(tmp_path, GAUSSIAN_ESP_LOG)
    assert "RESP" not in sets[0][2]
    with pytest.raises(charges.ChargeError, match="not on PATH"):
        charges.resp_from_gaussian(str(tmp_path / "job.out"))


def test_failed_resp_fit_keeps_the_other_sets(tmp_path, monkeypatch):
    import subprocess

    monkeypatch.setattr(charges.shutil, "which", lambda name: f"/amber/{name}")
    monkeypatch.setattr(
        charges.subprocess,
        "run",
        lambda cmd, **_kw: subprocess.CompletedProcess(cmd, 1, "boom", ""),
    )
    sets = _sets(tmp_path, GAUSSIAN_ESP_LOG)
    assert sets[0][2].endswith("Gaussian ESP-fitted charges")
    assert charges.gaussian_net_charge(" Charge = -1 Multiplicity = 2\n") == -1
