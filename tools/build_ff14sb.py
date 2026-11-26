"""Convert AmberTools' ff14SB protein force field into force_field_builder/data/amber_ff14sb.json.

Reads the files leaprc.protein.ff14SB loads (in AmberTools these are in dat/leap/parm and
dat/leap/lib; AmberTools states that they "have been placed in the public domain by
their authors"):

    python tools/build_ff14sb.py <dir with parm10.dat frcmod.ff14SB amino12.lib aminont12.lib aminoct12.lib>

The JSON uses mdff units: nm, kJ/mol, degrees, amu, e. Bonds and angles are
E = k/2 (x - x0)^2; torsions are E = k (1 + cos(n phi - phase)), one entry per term.
frcmod.ff14SB is loaded after parm10.dat, so its entries replace parm10's for the same key.
"""

from __future__ import annotations

import json
import os
import re
import sys

KCAL = 4.184
LIBS = ("amino12.lib", "aminont12.lib", "aminoct12.lib")
OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "force_field_builder",
    "data",
    "amber_ff14sb.json",
)


def _read(path):
    with open(path, encoding="latin-1") as fh:
        return fh.read()


def _types(field, n):
    parts = [p.strip() for p in field.split("-")]
    if len(parts) != n:
        raise ValueError(field)
    return parts


def _key(types):
    """Canonical key: the type tuple or its reverse, whichever sorts first."""
    types = list(types)
    rev = types[::-1]
    return "-".join(min(types, rev))


class Params:
    def __init__(self):
        self.mass = {}
        self.bonds = {}
        self.angles = {}
        self.propers = {}
        self.impropers = {}
        self.rmin_half = {}
        self.eps = {}
        self.equivalent = {}

    def add_bond(self, line):
        a, b = _types(line[:5], 2)
        k, r0 = (float(x) for x in line[5:].split()[:2])
        self.bonds[_key((a, b))] = [round(r0 / 10, 6), round(2 * k * KCAL * 100, 3)]

    def add_angle(self, line):
        a, b, c = _types(line[:8], 3)
        k, theta = (float(x) for x in line[8:].split()[:2])
        self.angles[_key((a, b, c))] = [theta, round(2 * k * KCAL, 4)]

    def read_proper(self, lines, i, pending):
        """One DIHE line; terms of one key continue while the periodicity is negative."""
        line = lines[i]
        types = _types(line[:11], 4)
        idivf, pk, phase, pn = (float(x) for x in line[11:].split()[:4])
        key = _key(types)
        if key not in pending:
            pending[key] = []
        pending[key].append([abs(round(pn)), phase, round(pk / idivf * KCAL, 6)])
        if pn > 0:
            self.propers[key] = pending.pop(key)

    def add_improper(self, line):
        types = _types(line[:11], 4)
        pk, phase, pn = (float(x) for x in line[11:].split()[:3])
        # Improper keys keep their order: the third type is the central atom.
        self.impropers["-".join(types)] = [abs(round(pn)), phase, round(pk * KCAL, 6)]


def read_parm(path, params):
    lines = _read(path).splitlines()
    i = 1
    while lines[i].strip():  # MASS
        t = lines[i][:2].strip()
        params.mass[t] = float(lines[i][2:].split()[0])
        i += 1
    i += 2  # blank, hydrophilic list
    while lines[i].strip():
        params.add_bond(lines[i])
        i += 1
    i += 1
    while lines[i].strip():
        params.add_angle(lines[i])
        i += 1
    i += 1
    pending = {}
    while lines[i].strip():
        params.read_proper(lines, i, pending)
        i += 1
    i += 1
    while lines[i].strip():
        params.add_improper(lines[i])
        i += 1
    i += 1
    while lines[i].strip():  # 10-12 H-bond terms (none used)
        i += 1
    i += 1
    while lines[i].strip():  # LJ equivalences
        parts = lines[i].split()
        for other in parts[1:]:
            params.equivalent[other] = parts[0]
        i += 1
    i += 1
    assert lines[i].startswith("MOD4") and "RE" in lines[i], lines[i]
    i += 1
    while lines[i].strip():
        t, r, e = lines[i].split()[:3]
        params.rmin_half[t] = float(r)
        params.eps[t] = float(e)
        i += 1


def read_frcmod(path, params):
    section = None
    pending = {}
    lines = _read(path).splitlines()
    for i, line in enumerate(lines[1:], start=1):
        head = line.strip()
        if head in (
            "MASS",
            "BOND",
            "ANGL",
            "ANGLE",
            "DIHE",
            "IMPR",
            "IMPROPER",
            "NONB",
            "NONBON",
        ):
            section = head[:4]
            continue
        if not head:
            continue
        if section == "MASS":
            params.mass[line[:2].strip()] = float(line[2:].split()[0])
        elif section == "BOND":
            params.add_bond(line)
        elif section == "ANGL":
            params.add_angle(line)
        elif section == "DIHE":
            params.read_proper(lines, i, pending)
        elif section == "IMPR":
            params.add_improper(line)
        elif section == "NONB":
            t, r, e = line.split()[:3]
            params.rmin_half[t] = float(r)
            params.eps[t] = float(e)
    assert not pending, pending


def _entries(text, name, table):
    """Rows of one '!entry.<name>.unit.<table>' block."""
    m = re.search(rf"^!entry\.{re.escape(name)}\.unit\.{table} .*$", text, re.MULTILINE)
    rows = []
    for line in text[m.end() + 1 :].splitlines():
        if line.startswith("!"):
            break
        rows.append(line)
    return rows


def read_lib(path, residues):
    text = _read(path)
    m = re.search(r"^!!index array str\n((?: \".*\"\n)+)", text, re.MULTILINE)
    names = re.findall(r'"([^"]+)"', m.group(1))
    for name in names:
        atoms = []
        for row in _entries(text, name, "atoms"):
            parts = row.split()
            atoms.append(
                [
                    parts[0].strip('"'),
                    parts[1].strip('"'),
                    int(parts[6]),
                    float(parts[7]),
                ]
            )
        bonds = []
        for row in _entries(text, name, "connectivity"):
            a, b, _flags = (int(x) for x in row.split())
            bonds.append([a - 1, b - 1])
        head, tail = (int(x) for x in _entries(text, name, "connect")[:2])
        positions = [
            [round(float(x), 4) for x in row.split()]
            for row in _entries(text, name, "positions")
        ]
        residues[name] = {
            "atoms": atoms,
            "bonds": bonds,
            "head": head - 1 if head else None,
            "tail": tail - 1 if tail else None,
            "positions": positions,
        }


def main(src):
    params = Params()
    read_parm(os.path.join(src, "parm10.dat"), params)
    read_frcmod(os.path.join(src, "frcmod.ff14SB"), params)
    residues = {}
    for lib in LIBS:
        read_lib(os.path.join(src, lib), residues)

    used = {a[1] for r in residues.values() for a in r["atoms"]}
    lj = {}
    for t in sorted(used):
        base = t if t in params.rmin_half else params.equivalent.get(t)
        if base is None:
            raise SystemExit(f"no LJ parameters for type {t}")
        r, e = params.rmin_half[base], params.eps[base]
        sigma = 2 * r / 2 ** (1 / 6) / 10
        lj[t] = [round(sigma, 7), round(e * KCAL, 7)]
    missing = sorted(used - set(params.mass))
    if missing:
        raise SystemExit(f"no masses for {missing}")
    data = {
        "name": "Amber ff14SB",
        "source": "AmberTools leaprc.protein.ff14SB: parm10.dat, frcmod.ff14SB, "
        + ", ".join(LIBS),
        "citation": "Maier et al., J. Chem. Theory Comput. 11, 3696 (2015)",
        "units": "nm, kJ/mol, degrees, amu, e; E = k/2 (x - x0)^2, E = k (1 + cos(n phi - phase))",
        "fudge_lj": 0.5,
        "fudge_qq": 0.8333333333,
        "masses": {t: params.mass[t] for t in sorted(used)},
        "lj": lj,
        "bonds": dict(sorted(params.bonds.items())),
        "angles": dict(sorted(params.angles.items())),
        "propers": dict(sorted(params.propers.items())),
        "impropers": dict(sorted(params.impropers.items())),
        "residues": dict(sorted(residues.items())),
    }
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(data, fh, separators=(",", ":"), sort_keys=False)
        fh.write("\n")
    print(
        f"{OUT}: {len(residues)} residues, {len(used)} atom types, {len(params.bonds)} bonds, "
        f"{len(params.angles)} angles, {len(params.propers)} torsions, "
        f"{len(params.impropers)} impropers, {os.path.getsize(OUT)} bytes"
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
