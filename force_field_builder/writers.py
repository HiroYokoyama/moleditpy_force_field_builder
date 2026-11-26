"""Write an mdff description as GROMACS, OpenMM ForceField XML or a LAMMPS data file."""

from __future__ import annotations

import json
import math
from xml.sax.saxutils import quoteattr

from .mdff import expand_system, pairs_14

KJ_PER_KCAL = 4.184


def _safe(name):
    return "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in name) or "MOL"


def _f(value, digits=8):
    text = f"{value:.{digits}g}"
    return "0.0" if text in ("-0", "0") else text


# ---------------------------------------------------------------------------
# GROMACS
# ---------------------------------------------------------------------------


def _atomtypes(data):
    """{type name: (atomic number, mass, sigma, epsilon)} over all molecule types."""
    from .elements_table import atomic_number

    out = {}
    for mt in data["molecule_types"]:
        for atom in mt["atoms"]:
            out.setdefault(
                atom["type"],
                (
                    atomic_number(atom["element"]),
                    atom["mass"],
                    atom["sigma"],
                    atom["epsilon"],
                ),
            )
    return out


def gromacs_itp(mt):
    """The ``[ moleculetype ]`` block of one molecule type."""
    name = _safe(mt["name"])
    res = name[:5]
    lines = [
        f"; {mt['name']}: {mt.get('force_field', '')}",
        f"; charges: {mt.get('charge_method', '')}",
        "",
    ]
    lines += [
        "[ moleculetype ]",
        "; name  nrexcl",
        f"{name}  {mt.get('exclusions_nrexcl', 3)}",
        "",
    ]
    lines += [
        "[ atoms ]",
        ";   nr  type            resnr res   atom   cgnr   charge        mass",
    ]
    for n, atom in enumerate(mt["atoms"], start=1):
        lines.append(
            f"{n:6d}  {atom['type']:<15s} 1  {res:<5s} {atom['name']:<6s} {n:5d}  {atom['charge']:12.8f}  {atom['mass']:10.5f}"
        )
    settles = mt.get("settles")
    if settles:
        lines += _gromacs_water_terms(mt)
        return "\n".join(lines) + "\n"

    if mt.get("bonds"):
        lines += ["", "[ bonds ]", ";  ai    aj funct  r0 (nm)       k (kJ/mol/nm^2)"]
        for b in mt["bonds"]:
            lines.append(
                f"{b['i'] + 1:5d} {b['j'] + 1:5d}   1   {_f(b['r0'])}  {_f(b['k'])}"
            )
    pairs = pairs_14(mt)
    if pairs:
        lines += ["", "[ pairs ]", ";  ai    aj funct"]
        lines += [f"{i + 1:5d} {j + 1:5d}   1" for i, j in pairs]
    if mt.get("angles"):
        lines += [
            "",
            "[ angles ]",
            ";  ai    aj    ak funct  theta0 (deg)  k (kJ/mol/rad^2)",
        ]
        for a in mt["angles"]:
            lines.append(
                f"{a['i'] + 1:5d} {a['j'] + 1:5d} {a['k_atom'] + 1:5d}   1   {_f(a['theta0'])}  {_f(a['k'])}"
            )
    propers = [t for t in mt.get("torsions", []) if not t.get("improper")]
    impropers = [t for t in mt.get("torsions", []) if t.get("improper")]
    if propers:
        lines += [
            "",
            "[ dihedrals ]",
            "; propers: periodic, funct 9 (several terms per quadruple allowed)",
            ";  ai    aj    ak    al funct  phase (deg)  k (kJ/mol)  n",
        ]
        for t in propers:
            lines.append(
                f"{t['i'] + 1:5d} {t['j'] + 1:5d} {t['k_atom'] + 1:5d} {t['l'] + 1:5d}   9   "
                f"{_f(t['phase'])}  {_f(t['k'])}  {t['periodicity']}"
            )
    if impropers:
        lines += [
            "",
            "[ dihedrals ]",
            "; impropers: periodic, funct 4",
            ";  ai    aj    ak    al funct  phase (deg)  k (kJ/mol)  n",
        ]
        for t in impropers:
            lines.append(
                f"{t['i'] + 1:5d} {t['j'] + 1:5d} {t['k_atom'] + 1:5d} {t['l'] + 1:5d}   4   "
                f"{_f(t['phase'])}  {_f(t['k'])}  {t['periodicity']}"
            )
    if mt.get("constraints"):
        lines += [
            "",
            "; Bonds to hydrogen are meant to be constrained: use",
            "; constraints = h-bonds in the .mdp (the bonds above stay flexible otherwise).",
        ]
    return "\n".join(lines) + "\n"


def _gromacs_water_terms(mt):
    s = mt["settles"]
    o = s["i"]
    hs = [n for n in range(3) if n != o]
    lines = [""]
    if o == 0:
        lines += [
            "#ifndef FLEXIBLE",
            "",
            "[ settles ]",
            "; OW funct doh dhh",
            f"1  1  {_f(s['doh'])}  {_f(s['dhh'])}",
            "",
            "#else",
        ]
    else:
        lines += [
            "#ifndef FLEXIBLE",
            "",
            "[ constraints ]",
            "; oxygen is not the first atom, so SETTLE cannot be used",
            f"{o + 1} {hs[0] + 1} 1 {_f(s['doh'])}",
            f"{o + 1} {hs[1] + 1} 1 {_f(s['doh'])}",
            f"{hs[0] + 1} {hs[1] + 1} 1 {_f(s['dhh'])}",
            "",
            "#else",
        ]
    lines += ["", "[ bonds ]"]
    for b in mt["bonds"]:
        lines.append(
            f"{b['i'] + 1:5d} {b['j'] + 1:5d}   1   {_f(b['r0'])}  {_f(b['k'])}"
        )
    lines += ["", "[ angles ]"]
    for a in mt["angles"]:
        lines.append(
            f"{a['i'] + 1:5d} {a['j'] + 1:5d} {a['k_atom'] + 1:5d}   1   {_f(a['theta0'])}  {_f(a['k'])}"
        )
    lines += ["", "#endif", "", "[ exclusions ]", "1 2 3", "2 1 3", "3 1 2"]
    return lines


def gromacs_files(data, base="topol"):
    """{filename: text}: ``<base>.top`` plus one ``<base>_<TYPE>.itp`` per molecule type."""
    files = {}
    rule = int(data.get("combination_rule", 2))
    top = [
        f"; {data.get('source', '')}",
        "; written from a moleditpy-mdff description",
        "",
        "[ defaults ]",
        "; nbfunc  comb-rule  gen-pairs  fudgeLJ  fudgeQQ",
        f"  1        {rule}          yes        {_f(data['fudge_lj'], 6)}      {_f(data['fudge_qq'], 6)}",
        "",
        "[ atomtypes ]",
        "; name             at.num   mass       charge  ptype  sigma (nm)     epsilon (kJ/mol)",
    ]
    for name, (z, mass, sigma, eps) in _atomtypes(data).items():
        top.append(
            f"  {name:<17s} {z:4d}  {mass:10.5f}  0.0000  A      {_f(sigma, 10):<14s} {_f(eps, 10)}"
        )
    top.append("")
    for mt in data["molecule_types"]:
        fname = f"{base}_{_safe(mt['name'])}.itp"
        files[fname] = gromacs_itp(mt)
        top.append(f'#include "{fname}"')
    top += [
        "",
        "[ system ]",
        data.get("source", "MoleditPy system"),
        "",
        "[ molecules ]",
        "; name      count",
    ]
    for entry in data["system"]:
        top.append(f"{_safe(entry['type']):<10s} {entry['count']}")
    files[f"{base}.top"] = "\n".join(top) + "\n"
    return files


# ---------------------------------------------------------------------------
# OpenMM ForceField XML
# ---------------------------------------------------------------------------


def _trefoil_groups(impropers):
    """Split impropers into SMIRNOFF trefoils (one entry each) and the rest."""
    remaining = list(impropers)
    trefoils = []
    used = set()
    index = {}
    for n, t in enumerate(remaining):
        index.setdefault(
            (
                t["i"],
                t["j"],
                t["k_atom"],
                t["l"],
                t["periodicity"],
                round(t["phase"], 6),
                round(t["k"], 9),
            ),
            [],
        ).append(n)
    for n, t in enumerate(remaining):
        if n in used:
            continue
        c, a, b, d = t["i"], t["j"], t["k_atom"], t["l"]
        key = (t["periodicity"], round(t["phase"], 6), round(t["k"], 9))
        partners = []
        for perm in ((c, b, d, a), (c, d, a, b)):
            cand = [m for m in index.get(perm + key, []) if m not in used and m != n]
            if not cand:
                break
            partners.append(cand[0])
        if len(partners) == 2:
            used.update([n] + partners)
            trefoils.append(t)
    rest = [t for n, t in enumerate(remaining) if n not in used]
    return trefoils, rest


def _torsion_attrs(terms):
    parts = []
    for n, t in enumerate(terms, start=1):
        parts.append(
            f'periodicity{n}="{t["periodicity"]}" phase{n}="{_f(math.radians(t["phase"]), 12)}" k{n}="{_f(t["k"], 12)}"'
        )
    return " ".join(parts)


def openmm_xml(data):
    """An OpenMM ForceField XML with one atom type per atom of each molecule type.

    Unique types make every bonded term exact. SMIRNOFF trefoil impropers go
    in a ``PeriodicTorsionForce`` with ``ordering="smirnoff"`` (OpenMM adds
    the three permutations itself); other impropers use ``ordering="charmm"``,
    which keeps the listed atom order when no wildcard is involved.
    """
    out = [
        "<ForceField>",
        " <Info>",
        f"  <Source>{_xml_text(data.get('source', ''))}</Source>",
        "  <Reference>moleditpy-mdff version 1</Reference>",
        " </Info>",
        " <AtomTypes>",
    ]
    tname = {}
    for mt in data["molecule_types"]:
        prefix = _safe(mt["name"])
        for n, atom in enumerate(mt["atoms"]):
            name = f"{prefix}-{n + 1}"
            tname[(mt["name"], n)] = name
            element = f' element="{atom["element"]}"' if atom.get("element") else ""
            out.append(
                f'  <Type name="{name}" class="{name}"{element} mass="{_f(atom["mass"], 10)}"/>'
            )
    out += [" </AtomTypes>", " <Residues>"]
    for mt in data["molecule_types"]:
        out.append(f"  <Residue name={quoteattr(mt['name'])}>")
        for n, atom in enumerate(mt["atoms"]):
            out.append(
                f'   <Atom name={quoteattr(atom["name"])} type="{tname[(mt["name"], n)]}" charge="{_f(atom["charge"], 12)}"/>'
            )
        for b in mt.get("bonds", []):
            out.append(
                f"   <Bond atomName1={quoteattr(mt['atoms'][b['i']]['name'])} atomName2={quoteattr(mt['atoms'][b['j']]['name'])}/>"
            )
        out.append("  </Residue>")
    out += [" </Residues>", " <HarmonicBondForce>"]
    for mt in data["molecule_types"]:
        for b in mt.get("bonds", []):
            out.append(
                f'  <Bond class1="{tname[(mt["name"], b["i"])]}" class2="{tname[(mt["name"], b["j"])]}" '
                f'length="{_f(b["r0"], 12)}" k="{_f(b["k"], 12)}"/>'
            )
    out += [" </HarmonicBondForce>", " <HarmonicAngleForce>"]
    for mt in data["molecule_types"]:
        for a in mt.get("angles", []):
            out.append(
                f'  <Angle class1="{tname[(mt["name"], a["i"])]}" class2="{tname[(mt["name"], a["j"])]}" '
                f'class3="{tname[(mt["name"], a["k_atom"])]}" angle="{_f(math.radians(a["theta0"]), 12)}" k="{_f(a["k"], 12)}"/>'
            )
    out += [" </HarmonicAngleForce>", ' <PeriodicTorsionForce ordering="smirnoff">']
    leftovers = []
    for mt in data["molecule_types"]:
        grouped = {}
        for t in mt.get("torsions", []):
            if not t.get("improper"):
                grouped.setdefault((t["i"], t["j"], t["k_atom"], t["l"]), []).append(t)
        for (i, j, k, l), terms in grouped.items():
            classes = " ".join(
                f'class{n}="{tname[(mt["name"], a)]}"'
                for n, a in enumerate((i, j, k, l), start=1)
            )
            out.append(f"  <Proper {classes} {_torsion_attrs(terms)}/>")
        trefoils, rest = _trefoil_groups(
            [t for t in mt.get("torsions", []) if t.get("improper")]
        )
        for t in trefoils:
            classes = " ".join(
                f'class{n}="{tname[(mt["name"], a)]}"'
                for n, a in enumerate((t["i"], t["j"], t["k_atom"], t["l"]), start=1)
            )
            out.append(f"  <Improper {classes} {_torsion_attrs([t])}/>")
        leftovers += [(mt, t) for t in rest]
    out.append(" </PeriodicTorsionForce>")
    if leftovers:
        out.append(' <PeriodicTorsionForce ordering="charmm">')
        for mt, t in leftovers:
            classes = " ".join(
                f'class{n}="{tname[(mt["name"], a)]}"'
                for n, a in enumerate((t["i"], t["j"], t["k_atom"], t["l"]), start=1)
            )
            out.append(f"  <Improper {classes} {_torsion_attrs([t])}/>")
        out.append(" </PeriodicTorsionForce>")
    out.append(
        f' <NonbondedForce coulomb14scale="{_f(data["fudge_qq"], 10)}" lj14scale="{_f(data["fudge_lj"], 10)}">'
    )
    out.append('  <UseAttributeFromResidue name="charge"/>')
    for mt in data["molecule_types"]:
        for n, atom in enumerate(mt["atoms"]):
            out.append(
                f'  <Atom class="{tname[(mt["name"], n)]}" sigma="{_f(atom["sigma"] or 1.0, 12)}" epsilon="{_f(atom["epsilon"], 12)}"/>'
            )
    out += [" </NonbondedForce>", "</ForceField>"]
    return "\n".join(out) + "\n"


def _xml_text(text):
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ---------------------------------------------------------------------------
# LAMMPS data file (units real, atom_style full)
# ---------------------------------------------------------------------------


def lammps_box(positions, cell=None, padding=10.0):
    """(header lines, transformed positions) for a LAMMPS box.

    *cell* is a ``moleditpy_cell`` dict (Angstrom, rows are cell vectors).
    Without one, an orthogonal box around the atoms with *padding* on each
    side is used.
    """
    if cell:
        a, b, c = (list(map(float, v)) for v in cell["lattice"])
        origin = [float(x) for x in cell.get("origin", [0.0, 0.0, 0.0])]
        la = math.sqrt(sum(x * x for x in a))
        ax = [x / la for x in a]
        xy = sum(p * q for p, q in zip(b, ax))
        ly = math.sqrt(max(sum(x * x for x in b) - xy * xy, 0.0))
        xz = sum(p * q for p, q in zip(c, ax))
        yz = (sum(p * q for p, q in zip(b, c)) - xy * xz) / ly
        lz = math.sqrt(max(sum(x * x for x in c) - xz * xz - yz * yz, 0.0))
        new = [[la, 0.0, 0.0], [xy, ly, 0.0], [xz, yz, lz]]
        inv = _inverse3([a, b, c])
        out = []
        for p in positions:
            rel = [p[n] - origin[n] for n in range(3)]
            frac = [sum(rel[r] * inv[r][col] for r in range(3)) for col in range(3)]
            out.append(
                [sum(frac[r] * new[r][col] for r in range(3)) for col in range(3)]
            )
        header = [
            f"0.0 {_f(la, 12)} xlo xhi",
            f"0.0 {_f(ly, 12)} ylo yhi",
            f"0.0 {_f(lz, 12)} zlo zhi",
        ]
        if max(abs(xy), abs(xz), abs(yz)) > 1e-8:
            header.append(f"{_f(xy, 12)} {_f(xz, 12)} {_f(yz, 12)} xy xz yz")
        return header, out
    if positions:
        lo = [min(p[n] for p in positions) - padding for n in range(3)]
        hi = [max(p[n] for p in positions) + padding for n in range(3)]
    else:
        lo, hi = [0.0] * 3, [padding] * 3
    header = [
        f"{_f(lo[n], 12)} {_f(hi[n], 12)} {axis}lo {axis}hi"
        for n, axis in enumerate("xyz")
    ]
    return header, [list(p) for p in positions]


def _inverse3(m):
    (a, b, c), (d, e, f), (g, h, i) = m
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if abs(det) < 1e-12:
        raise ValueError("The cell vectors are linearly dependent")
    return [
        [(e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det],
        [(f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det],
        [(d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det],
    ]


def lammps_data(data, positions, cell=None, padding=10.0):
    """LAMMPS data file in ``real`` units (kcal/mol, Angstrom) with ``atom_style full``.

    *positions* are host coordinates in Angstrom (one per host atom). All
    torsions, impropers included, are written as ``dihedral_style fourier``
    terms, which compute the same i-j-k-l dihedral as the mdff entry.
    """
    atom_type_ids, bond_type_ids, angle_type_ids, dihedral_type_ids = {}, {}, {}, {}
    atom_type_info = {}
    atoms, bonds, angles, dihedrals = [], [], [], []
    for mol_id, (mt, _copy, first) in enumerate(expand_system(data), start=1):
        for n, atom in enumerate(mt["atoms"]):
            if atom["type"] not in atom_type_ids:
                atom_type_ids[atom["type"]] = len(atom_type_ids) + 1
                atom_type_info[atom["type"]] = atom
            atoms.append(
                (first + n + 1, mol_id, atom_type_ids[atom["type"]], atom["charge"])
            )
        for b in mt.get("bonds", []):
            key = (
                round(b["k"] / KJ_PER_KCAL / 100.0 / 2.0, 9),
                round(b["r0"] * 10.0, 9),
            )
            bond_type_ids.setdefault(key, len(bond_type_ids) + 1)
            bonds.append((bond_type_ids[key], first + b["i"] + 1, first + b["j"] + 1))
        for a in mt.get("angles", []):
            key = (round(a["k"] / KJ_PER_KCAL / 2.0, 9), round(a["theta0"], 9))
            angle_type_ids.setdefault(key, len(angle_type_ids) + 1)
            angles.append(
                (
                    angle_type_ids[key],
                    first + a["i"] + 1,
                    first + a["j"] + 1,
                    first + a["k_atom"] + 1,
                )
            )
        grouped = {}
        for t in mt.get("torsions", []):
            grouped.setdefault(
                (t["i"], t["j"], t["k_atom"], t["l"], bool(t.get("improper"))), []
            ).append(t)
        for (i, j, k, l, _imp), terms in grouped.items():
            key = tuple(
                (round(t["k"] / KJ_PER_KCAL, 8), t["periodicity"], round(t["phase"], 6))
                for t in terms
            )
            dihedral_type_ids.setdefault(key, len(dihedral_type_ids) + 1)
            dihedrals.append(
                (
                    dihedral_type_ids[key],
                    first + i + 1,
                    first + j + 1,
                    first + k + 1,
                    first + l + 1,
                )
            )

    header_box, coords = lammps_box(positions, cell, padding)
    if len(coords) != len(atoms):
        raise ValueError(f"{len(coords)} positions for {len(atoms)} atoms")
    lines = [
        (
            f"LAMMPS data file from {data.get('source', 'moleditpy-mdff')} (units real, atom_style full; "
            f"pair lj/cut/coul/long mix arithmetic; special_bonds lj 0 0 {_f(data['fudge_lj'], 6)} "
            f"coul 0 0 {_f(data['fudge_qq'], 6)}; bond/angle harmonic; dihedral fourier)"
        ),
        "",
        f"{len(atoms)} atoms",
        f"{len(bonds)} bonds",
        f"{len(angles)} angles",
        f"{len(dihedrals)} dihedrals",
        "0 impropers",
        "",
        f"{len(atom_type_ids)} atom types",
        f"{len(bond_type_ids)} bond types",
        f"{len(angle_type_ids)} angle types",
        f"{len(dihedral_type_ids)} dihedral types",
        "",
    ]
    lines += header_box + ["", "Masses", ""]
    for name, tid in atom_type_ids.items():
        lines.append(f"{tid} {_f(atom_type_info[name]['mass'], 10)} # {name}")
    lines += ["", "Pair Coeffs # lj/cut/coul/long", ""]
    for name, tid in atom_type_ids.items():
        atom = atom_type_info[name]
        lines.append(
            f"{tid} {_f(atom['epsilon'] / KJ_PER_KCAL, 10)} {_f(atom['sigma'] * 10.0, 10)} # {name}"
        )
    if bond_type_ids:
        lines += ["", "Bond Coeffs # harmonic", ""]
        lines += [
            f"{tid} {_f(k, 10)} {_f(r0, 10)}" for (k, r0), tid in bond_type_ids.items()
        ]
    if angle_type_ids:
        lines += ["", "Angle Coeffs # harmonic", ""]
        lines += [
            f"{tid} {_f(k, 10)} {_f(t0, 10)}" for (k, t0), tid in angle_type_ids.items()
        ]
    if dihedral_type_ids:
        lines += ["", "Dihedral Coeffs # fourier", ""]
        for key, tid in dihedral_type_ids.items():
            terms = " ".join(f"{_f(k, 10)} {n} {_f(phase, 10)}" for k, n, phase in key)
            lines.append(f"{tid} {len(key)} {terms}")
    lines += ["", "Atoms # full", ""]
    for (aid, mid, tid, q), (x, y, z) in zip(atoms, coords):
        lines.append(f"{aid} {mid} {tid} {_f(q, 10)} {x:.6f} {y:.6f} {z:.6f}")
    for title, items in (
        ("Bonds", bonds),
        ("Angles", angles),
        ("Dihedrals", dihedrals),
    ):
        if items:
            lines += ["", title, ""]
            lines += [
                f"{n} " + " ".join(str(v) for v in item)
                for n, item in enumerate(items, start=1)
            ]
    return "\n".join(lines) + "\n"


def mdff_json(data):
    return json.dumps(data, indent=1, allow_nan=False) + "\n"
