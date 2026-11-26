"""Regenerate the example files in this folder (run from the repository root).

    python examples/make_examples.py

Coordinates come from RDKit with fixed seeds, charges from MMFF94 so the
output does not depend on optional tools.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tests"))

from helpers import combine, mol3d, shifted, water_box

from force_field_builder import builder, mdff, writers
from force_field_builder.display import molecule_positions

OPTIONS = {"charge_method": "mmff94"}


def write(path, text):
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main():
    systems = {
        "water_tip3p": water_box(3),
        "nacl_in_water": combine(
            water_box(4), shifted(mol3d("[Na+]"), 14), shifted(mol3d("[Cl-]"), 18)
        ),
        "ethanol_in_water": combine(mol3d("CCO"), shifted(water_box(3), 6)),
    }
    for name, mol in systems.items():
        result = builder.build(mol, OPTIONS)
        write(os.path.join(HERE, f"{name}.mdff.json"), mdff.dumps(result.mdff))
    # Every output format for the ethanol example.
    mol = systems["ethanol_in_water"]
    data = builder.build(mol, OPTIONS).mdff
    out = os.path.join(HERE, "ethanol_in_water")
    os.makedirs(out, exist_ok=True)
    for fname, text in writers.gromacs_files(data, "topol").items():
        write(os.path.join(out, fname), text)
    write(os.path.join(out, "ethanol_in_water_ff.xml"), writers.openmm_xml(data))
    write(
        os.path.join(out, "ethanol_in_water.data"),
        writers.lammps_data(data, molecule_positions(mol)),
    )


if __name__ == "__main__":
    main()
