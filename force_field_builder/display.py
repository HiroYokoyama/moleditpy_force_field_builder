"""Qt-free helpers for the dialog: colours, labels, project state, cell lookup."""

from __future__ import annotations

import json

from .builder import BuildResult, TypeInfo
from .mdff import expand_system

DISPLAY_MODES = {
    "none": "Off",
    "charge_colours": "Colour atoms by charge",
    "charge_labels": "Colour by charge + charge labels",
    "type_labels": "Atom-type labels",
}

MAX_LABELS = 3000


def charge_colour(q, limit=1.0):
    """Blue (negative) - white - red (positive) hex colour for charge *q*."""
    limit = limit if limit > 0 else 1.0
    t = max(-1.0, min(1.0, q / limit))
    if t >= 0:
        r, g, b = 255, round(255 * (1 - t)), round(255 * (1 - t))
    else:
        r, g, b = round(255 * (1 + t)), round(255 * (1 + t)), 255
    return f"#{r:02x}{g:02x}{b:02x}"


def charge_colours(charges):
    if not charges:
        return {}
    limit = max(abs(q) for q in charges) or 1.0
    limit = min(limit, 1.0)
    return {i: charge_colour(q, limit) for i, q in enumerate(charges)}


def per_atom_data(mdff):
    """(types, charges) for every host atom from an mdff description."""
    types, charges = [], []
    for mt, _copy, _first in expand_system(mdff):
        for atom in mt["atoms"]:
            types.append(atom["type"])
            charges.append(atom["charge"])
    return types, charges


def molecule_fingerprint(mol):
    """Atom count plus element sequence: tells whether a build still fits the molecule."""
    if mol is None:
        return None
    return [mol.GetNumAtoms(), "".join(a.GetSymbol() for a in mol.GetAtoms())]


def result_to_state(result, fingerprint):
    return {
        "mdff": result.mdff,
        "fingerprint": fingerprint,
        "types": [
            {
                "name": t.name,
                "kind": t.kind,
                "count": t.count,
                "smiles": t.smiles,
                "warnings": list(t.warnings),
                "charge_method": t.charge_method,
                "charges_from_force_field": t.charges_from_force_field,
            }
            for t in result.types
        ],
        "warnings": list(result.warnings),
        "edited": bool(getattr(result, "edited", False)),
    }


def state_to_result(state):
    result = BuildResult()
    result.mdff = state["mdff"]
    for t in state.get("types", []):
        info = TypeInfo(t["name"], t.get("kind", "molecule"))
        info.count = t.get("count", 0)
        info.smiles = t.get("smiles", "")
        info.warnings = list(t.get("warnings", []))
        info.charge_method = t.get("charge_method", "")
        info.charges_from_force_field = bool(t.get("charges_from_force_field", False))
        result.types.append(info)
    result.warnings = list(state.get("warnings", []))
    result.atom_types, result.atom_charges = per_atom_data(result.mdff)
    result.edited = bool(state.get("edited", False))
    return result


def molecule_cell(mol):
    """The ``moleditpy_cell`` dict of *mol*, or None."""
    if mol is None:
        return None
    try:
        if mol.HasProp("moleditpy_cell"):
            cell = json.loads(mol.GetProp("moleditpy_cell"))
            if isinstance(cell, dict) and len(cell.get("lattice", [])) == 3:
                return cell
    except (ValueError, TypeError, AttributeError):
        return None
    return None


def molecule_positions(mol):
    if mol is None or not mol.GetNumConformers():
        return []
    conf = mol.GetConformer()
    out = []
    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        out.append([p.x, p.y, p.z])
    return out
