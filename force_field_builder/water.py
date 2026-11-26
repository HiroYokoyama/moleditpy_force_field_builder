"""Rigid three-site water models from published tables (mdff units).

Each model is the standard set used by GROMACS and AMBER. All of them are rigid:
the molecule type carries ``settles`` and also flexible bond/angle terms with the
reference geometry, so engines that ignore constraints still see a sensible
water.

OPC is a four-site model with a massless virtual site. ``moleditpy-mdff``
version 1 has no field for virtual sites, so the three-site OPC3 is offered
instead.
"""

from __future__ import annotations

import math

# Flexible terms written next to the settles (GROMACS flexible SPC values).
_FLEX_BOND_K = 345000.0  # kJ/mol/nm^2
_FLEX_ANGLE_K = 383.0  # kJ/mol/rad^2

MASS_O = 15.9994
MASS_H = 1.008

WATER_MODELS = {
    "TIP3P": {
        "label": "TIP3P",
        "q_o": -0.834,
        "q_h": 0.417,
        "sigma_o": 0.315061,  # nm
        "epsilon_o": 0.636386,  # kJ/mol
        "doh": 0.09572,  # nm
        "angle": 104.52,  # degrees
        "reference": "Jorgensen et al., J. Chem. Phys. 79, 926 (1983)",
    },
    "SPC/E": {
        "label": "SPC/E",
        "q_o": -0.8476,
        "q_h": 0.4238,
        "sigma_o": 0.316557,
        "epsilon_o": 0.650194,
        "doh": 0.1,
        "angle": 109.47,
        "reference": "Berendsen et al., J. Phys. Chem. 91, 6269 (1987)",
    },
    "OPC3": {
        "label": "OPC3",
        "q_o": -0.89517,
        "q_h": 0.447585,
        "sigma_o": 0.317427,
        "epsilon_o": 0.683690,
        "doh": 0.097888,
        "angle": 109.47,
        "reference": "Izadi and Onufriev, J. Chem. Phys. 145, 074501 (2016)",
    },
}


def hh_distance(model: dict) -> float:
    """H-H distance (nm) implied by the O-H length and the H-O-H angle."""
    return 2.0 * model["doh"] * math.sin(math.radians(model["angle"]) / 2.0)


def water_molecule_type(model_name: str, name: str = "SOL", o_index: int = 0) -> dict:
    """An mdff molecule type for one water molecule.

    *o_index* is the position of the oxygen among the three atoms (0, 1 or 2);
    the two other positions are the hydrogens, in host order.
    """
    model = WATER_MODELS[model_name]
    order = ["H", "H", "H"]
    order[o_index] = "O"
    atoms = []
    h_count = 0
    h_indices = []
    for idx, element in enumerate(order):
        if element == "O":
            atoms.append(
                {
                    "name": "OW",
                    "element": "O",
                    "type": f"OW_{model['label'].replace('/', '')}",
                    "charge": model["q_o"],
                    "mass": MASS_O,
                    "sigma": model["sigma_o"],
                    "epsilon": model["epsilon_o"],
                }
            )
        else:
            h_count += 1
            h_indices.append(idx)
            atoms.append(
                {
                    "name": f"HW{h_count}",
                    "element": "H",
                    "type": f"HW_{model['label'].replace('/', '')}",
                    "charge": model["q_h"],
                    "mass": MASS_H,
                    "sigma": 0.0,
                    "epsilon": 0.0,
                }
            )
    h1, h2 = h_indices
    return {
        "name": name,
        "atoms": atoms,
        "bonds": [
            {
                "i": min(o_index, h1),
                "j": max(o_index, h1),
                "r0": model["doh"],
                "k": _FLEX_BOND_K,
            },
            {
                "i": min(o_index, h2),
                "j": max(o_index, h2),
                "r0": model["doh"],
                "k": _FLEX_BOND_K,
            },
        ],
        "angles": [
            {
                "i": h1,
                "j": o_index,
                "k_atom": h2,
                "theta0": model["angle"],
                "k": _FLEX_ANGLE_K,
            }
        ],
        "torsions": [],
        "exclusions_nrexcl": 2,
        "settles": {
            "i": o_index,
            "doh": model["doh"],
            "dhh": round(hh_distance(model), 6),
        },
        "charge_method": f"{model['label']} water model",
        "charges_from_force_field": True,
        "force_field": f"{model['label']} water ({model['reference']})",
    }
