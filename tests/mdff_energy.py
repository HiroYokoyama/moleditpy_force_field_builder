"""A plain-numpy reference evaluator for mdff descriptions (tests only).

Non-periodic, no cutoff. 1-2 and 1-3 pairs are excluded, 1-4 pairs are scaled
by fudge_lj / fudge_qq, Lorentz-Berthelot mixing. Returns kJ/mol per term.
"""

import math

import numpy as np

COULOMB = 138.935458  # kJ mol^-1 nm e^-2


def _dihedral(p0, p1, p2, p3):
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1n = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    x = np.dot(v, w)
    y = np.dot(np.cross(b1n, v), w)
    return math.atan2(y, x)


def energy(data, coords_nm, nrexcl_from_type=False):
    coords = np.asarray(coords_nm, dtype=float)
    out = {"bond": 0.0, "angle": 0.0, "torsion": 0.0, "lj": 0.0, "coulomb": 0.0}
    types = {mt["name"]: mt for mt in data["molecule_types"]}
    charges, sigmas, epsilons = [], [], []
    scale = {}
    for entry in data["system"]:
        mt = types[entry["type"]]
        n = len(mt["atoms"])
        for copy in range(entry["count"]):
            first = entry["atom_offset"] + copy * n
            for atom in mt["atoms"]:
                charges.append(atom["charge"])
                sigmas.append(atom["sigma"])
                epsilons.append(atom["epsilon"])
            for b in mt.get("bonds", []):
                r = np.linalg.norm(coords[first + b["i"]] - coords[first + b["j"]])
                out["bond"] += 0.5 * b["k"] * (r - b["r0"]) ** 2
            for a in mt.get("angles", []):
                u = coords[first + a["i"]] - coords[first + a["j"]]
                v = coords[first + a["k_atom"]] - coords[first + a["j"]]
                cos = np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)
                theta = math.acos(max(-1.0, min(1.0, cos)))
                out["angle"] += 0.5 * a["k"] * (theta - math.radians(a["theta0"])) ** 2
            for t in mt.get("torsions", []):
                phi = _dihedral(
                    *(coords[first + t[key]] for key in ("i", "j", "k_atom", "l"))
                )
                out["torsion"] += t["k"] * (
                    1 + math.cos(t["periodicity"] * phi - math.radians(t["phase"]))
                )
            # exclusions / 1-4 from the bond graph
            adj = {i: set() for i in range(n)}
            for b in mt.get("bonds", []):
                adj[b["i"]].add(b["j"])
                adj[b["j"]].add(b["i"])
            for i in range(n):
                dist = {i: 0}
                frontier = [i]
                for depth in (1, 2, 3):
                    frontier = [m for f in frontier for m in adj[f] if m not in dist]
                    for m in frontier:
                        dist.setdefault(m, depth)
                for j, d in dist.items():
                    if j > i and d > 0:
                        scale[(first + i, first + j)] = (
                            (0.0, 0.0)
                            if d < 3
                            else (data["fudge_lj"], data["fudge_qq"])
                        )
            if mt.get("settles"):
                for i in range(n):
                    for j in range(i + 1, n):
                        scale[(first + i, first + j)] = (0.0, 0.0)
    natoms = len(charges)
    for i in range(natoms):
        for j in range(i + 1, natoms):
            s_lj, s_qq = scale.get((i, j), (1.0, 1.0))
            if s_lj == 0.0 and s_qq == 0.0:
                continue
            r = np.linalg.norm(coords[i] - coords[j])
            sig = 0.5 * (sigmas[i] + sigmas[j])
            eps = math.sqrt(epsilons[i] * epsilons[j])
            out["lj"] += s_lj * 4 * eps * ((sig / r) ** 12 - (sig / r) ** 6)
            out["coulomb"] += s_qq * COULOMB * charges[i] * charges[j] / r
    return out
