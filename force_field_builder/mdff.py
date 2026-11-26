"""The ``moleditpy-mdff`` hand-off format: validation, JSON I/O and helpers.

Units: nm, kJ/mol, degrees, elementary charge, amu. Bonds are harmonic,
E = k/2 (r - r0)^2; angles are harmonic in the angle, E = k/2 (theta -
theta0)^2 with k in kJ/mol/rad^2; torsions are periodic, E = k (1 + cos(n phi
- phase)), with one entry per Fourier term. Improper torsions use the same
form; their four atoms are listed in the order that defines the dihedral angle
(the central atom first for SMIRNOFF impropers, each of the three trefoil
permutations listed separately with a third of the force constant). Atom
indices are 0-based within the molecule type.
"""

from __future__ import annotations

import json

FORMAT_NAME = "moleditpy-mdff"
FORMAT_VERSION = 1

_ATOM_KEYS = ("name", "element", "type", "charge", "mass", "sigma", "epsilon")


def validate(data):
    """Return a list of problems (empty when *data* is a valid mdff document)."""
    errors = []
    if not isinstance(data, dict):
        return ["top level is not an object"]
    if data.get("format") != FORMAT_NAME:
        errors.append(f"format is {data.get('format')!r}, expected {FORMAT_NAME!r}")
    if data.get("version") != FORMAT_VERSION:
        errors.append(f"version is {data.get('version')!r}, expected {FORMAT_VERSION}")
    for key in ("combination_rule", "fudge_lj", "fudge_qq"):
        if not isinstance(data.get(key), (int, float)):
            errors.append(f"{key} is missing or not a number")
    types = data.get("molecule_types")
    if not isinstance(types, list) or not types:
        errors.append("molecule_types is missing or empty")
        types = []
    names = {}
    for mt in types:
        name = mt.get("name")
        if not name:
            errors.append("a molecule type has no name")
            continue
        if name in names:
            errors.append(f"duplicate molecule type {name}")
        atoms = mt.get("atoms") or []
        names[name] = len(atoms)
        if not atoms:
            errors.append(f"{name}: no atoms")
        for n, atom in enumerate(atoms):
            missing = [k for k in _ATOM_KEYS if k not in atom]
            if missing:
                errors.append(f"{name} atom {n}: missing {', '.join(missing)}")
        n_atoms = len(atoms)

        def check(label, items, keys, name=name, n_atoms=n_atoms):
            for item in items or []:
                for key in keys:
                    idx = item.get(key)
                    if not isinstance(idx, int) or not 0 <= idx < n_atoms:
                        errors.append(
                            f"{name} {label}: index {key}={idx!r} out of range"
                        )
                        break

        check("bond", mt.get("bonds"), ("i", "j"))
        check("angle", mt.get("angles"), ("i", "j", "k_atom"))
        check("torsion", mt.get("torsions"), ("i", "j", "k_atom", "l"))
        check("constraint", mt.get("constraints"), ("i", "j"))
        for bond in mt.get("bonds") or []:
            if bond.get("r0") is None or bond.get("k") is None:
                errors.append(
                    f"{name} bond {bond.get('i')}-{bond.get('j')}: missing r0/k"
                )
        for tor in mt.get("torsions") or []:
            if not isinstance(tor.get("periodicity"), int):
                errors.append(f"{name} torsion: periodicity must be an integer")
                break
    offset = 0
    for entry in data.get("system") or []:
        name = entry.get("type")
        if name not in names:
            errors.append(f"system refers to unknown type {name!r}")
            continue
        if entry.get("atom_offset") != offset:
            errors.append(
                f"system entry {name}: atom_offset {entry.get('atom_offset')} but {offset} expected"
            )
        offset = entry.get("atom_offset", offset) + names[name] * int(
            entry.get("count", 0)
        )
    if not data.get("system"):
        errors.append("system is missing or empty")
    return errors


def total_atoms(data):
    sizes = {mt["name"]: len(mt["atoms"]) for mt in data["molecule_types"]}
    return sum(sizes[e["type"]] * e["count"] for e in data["system"])


def dumps(data):
    """JSON text with a stable key order (no NaN)."""
    return json.dumps(data, indent=1, allow_nan=False) + "\n"


def loads(text):
    data = json.loads(text)
    errors = validate(data)
    if errors:
        raise ValueError("Not a valid moleditpy-mdff file:\n" + "\n".join(errors[:20]))
    return data


def expand_system(data):
    """Yield (molecule_type, copy_index, first_host_atom) in host order."""
    types = {mt["name"]: mt for mt in data["molecule_types"]}
    for entry in data["system"]:
        mt = types[entry["type"]]
        n = len(mt["atoms"])
        for copy in range(entry["count"]):
            yield mt, copy, entry["atom_offset"] + copy * n


def neighbour_distances(n_atoms, bonds, max_distance=3):
    """{(i, j): graph distance} for i < j and distance <= max_distance."""
    adj = [[] for _ in range(n_atoms)]
    for b in bonds:
        adj[b["i"]].append(b["j"])
        adj[b["j"]].append(b["i"])
    out = {}
    for start in range(n_atoms):
        frontier = [start]
        seen = {start: 0}
        for depth in range(1, max_distance + 1):
            nxt = []
            for a in frontier:
                for b in adj[a]:
                    if b not in seen:
                        seen[b] = depth
                        nxt.append(b)
            frontier = nxt
        for other, dist in seen.items():
            if other > start and dist > 0:
                out[(start, other)] = dist
    return out


def pairs_14(mt):
    """1-4 pairs (graph distance exactly 3) of a molecule type, sorted."""
    dist = neighbour_distances(len(mt["atoms"]), mt.get("bonds") or [])
    return sorted(k for k, d in dist.items() if d == 3)
