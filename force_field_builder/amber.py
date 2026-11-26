"""Amber ff14SB for proteins: residue templates, charges and bonded parameters.

The parameters come from AmberTools' ff14SB files (parm10.dat, frcmod.ff14SB and the
amino12 libraries, in the public domain), converted by tools/build_ff14sb.py into
data/amber_ff14sb.json.

Each residue (from the PDB residue information of the host molecule) is matched to an
ff14SB template by its bond graph, so hydrogen names do not matter and the protonation
state picks the template: HIS becomes HID, HIE or HIP, ASP/ASH, GLU/GLH, LYS/LYN and
CYS/CYX/CYM likewise, and chain ends take the charged N- and C-terminal templates.
Bonded terms are assigned the way tleap does: torsions by their exact atom types first
and the X-B-C-X wildcard otherwise; impropers on every atom with three neighbours, the
most specific parameter, with tleap's ordering of the outer atoms.
"""

from __future__ import annotations

import itertools
import json
import os

DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "amber_ff14sb.json"
)
FORCE_FIELD_LABEL = "Amber ff14SB (proteins)"

#: PDB residue name -> ff14SB templates that residue may be (chosen by the bond graph)
FAMILIES = {
    "ALA": ("ALA",),
    "ARG": ("ARG",),
    "ASN": ("ASN",),
    "ASP": ("ASP", "ASH"),
    "ASH": ("ASH", "ASP"),
    "CYS": ("CYS", "CYX", "CYM"),
    "CYX": ("CYX", "CYS", "CYM"),
    "CYM": ("CYM", "CYS", "CYX"),
    "GLN": ("GLN",),
    "GLU": ("GLU", "GLH"),
    "GLH": ("GLH", "GLU"),
    "GLY": ("GLY",),
    "HIS": ("HIE", "HID", "HIP"),
    "HIE": ("HIE", "HID", "HIP"),
    "HID": ("HID", "HIE", "HIP"),
    "HIP": ("HIP", "HIE", "HID"),
    "HSD": ("HID", "HIE", "HIP"),
    "HSE": ("HIE", "HID", "HIP"),
    "HSP": ("HIP", "HIE", "HID"),
    "HYP": ("HYP",),
    "ILE": ("ILE",),
    "LEU": ("LEU",),
    "LYS": ("LYS", "LYN"),
    "LYN": ("LYN", "LYS"),
    "MET": ("MET",),
    "PHE": ("PHE",),
    "PRO": ("PRO",),
    "SER": ("SER",),
    "THR": ("THR",),
    "TRP": ("TRP",),
    "TYR": ("TYR",),
    "VAL": ("VAL",),
    "ACE": ("ACE",),
    "NME": ("NME",),
    "NHE": ("NHE",),
}


class AmberError(RuntimeError):
    """The protein cannot be parameterised with ff14SB."""


_cache = {}


def load():
    if "ff" not in _cache:
        with open(DATA_PATH, encoding="utf-8") as fh:
            _cache["ff"] = json.load(fh)
    return _cache["ff"]


def _key(types):
    types = list(types)
    return "-".join(min(types, types[::-1]))


# --------------------------------------------------------------------------- residues


def residues_of(mol):
    """[(label, resname, [atom indices])] in atom order, from PDB residue information."""
    groups = []
    index = {}
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None:
            return []
        key = (
            info.GetChainId(),
            info.GetResidueNumber(),
            info.GetInsertionCode(),
            info.GetResidueName().strip().upper(),
        )
        if key not in index:
            index[key] = len(groups)
            label = f"{key[3]}{key[1]}{key[2].strip()}" + (
                f":{key[0]}" if key[0].strip() else ""
            )
            groups.append((label, key[3], []))
        groups[index[key]][2].append(atom.GetIdx())
    return groups


def is_protein(mol):
    """True when every atom belongs to a residue ff14SB has templates for."""
    groups = residues_of(mol)
    return bool(groups) and all(resname in FAMILIES for _, resname, _ in groups)


def protein_atoms(mol):
    """Indices of atoms in residues ff14SB has templates for."""
    out = set()
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is not None and info.GetResidueName().strip().upper() in FAMILIES:
            out.add(atom.GetIdx())
    return out


def _template_mol(template):
    from rdkit import Chem

    rw = Chem.RWMol()
    for _name, _type, z, _q in template["atoms"]:
        rw.AddAtom(Chem.Atom(z))
    for i, j in template["bonds"]:
        rw.AddBond(i, j, Chem.BondType.SINGLE)
    return rw.GetMol()


def _residue_graph(mol, atoms):
    from rdkit import Chem

    pos = {a: n for n, a in enumerate(atoms)}
    rw = Chem.RWMol()
    for a in atoms:
        rw.AddAtom(Chem.Atom(mol.GetAtomWithIdx(a).GetAtomicNum()))
    for a in atoms:
        for bond in mol.GetAtomWithIdx(a).GetBonds():
            b = bond.GetOtherAtomIdx(a)
            if b in pos and pos[a] < pos[b]:
                rw.AddBond(pos[a], pos[b], Chem.BondType.SINGLE)
    return rw.GetMol()


def _candidates(resname, first, last, ff):
    """Template names to try, the chain-end forms first at the ends of a chain."""
    names = []
    for base in FAMILIES[resname]:
        if first and f"N{base}" in ff["residues"]:
            names.append(f"N{base}")
        if last and f"C{base}" in ff["residues"]:
            names.append(f"C{base}")
        names.append(base)
    return names


def match_residue(mol, atoms, resname, ff, outside):
    """(template name, {host atom: template atom index}) or raise AmberError.

    *outside* maps host atoms of this residue to their bonded atoms in other residues.
    """
    from rdkit import Chem

    graph = _residue_graph(mol, atoms)
    host_names = [
        (mol.GetAtomWithIdx(a).GetPDBResidueInfo().GetName() or "").strip().upper()
        for a in atoms
    ]
    externals = {n for n, a in enumerate(atoms) if outside.get(a)}
    first = not any(mol.GetAtomWithIdx(atoms[n]).GetAtomicNum() == 7 for n in externals)
    last = not any(mol.GetAtomWithIdx(atoms[n]).GetAtomicNum() == 6 for n in externals)
    tried = []
    for name in _candidates(resname, first, last, ff):
        template = ff["residues"][name]
        tried.append(name)
        if len(template["atoms"]) != len(atoms):
            continue
        tmol = _template_mol(template)
        params = Chem.SubstructMatchParameters()
        params.maxMatches = 5000
        params.uniquify = False
        matches = graph.GetSubstructMatches(tmol, params)
        if not matches:
            continue
        allowed = {template["head"], template["tail"]} - {None}
        if name.endswith("CYX"):
            allowed |= {k for k, a in enumerate(template["atoms"]) if a[0] == "SG"}
        best = None
        for match in matches:  # match[template atom] = residue position
            inverse = {pos: t for t, pos in enumerate(match)}
            if any(inverse[n] not in allowed for n in externals):
                continue
            score = sum(
                template["atoms"][inverse[n]][0] == host_names[n]
                for n in range(len(atoms))
            )
            if best is None or score > best[0]:
                best = (score, inverse)
                if score == len(atoms):
                    break
        if best is not None:
            mapping = {atoms[n]: t for n, t in best[1].items()}
            if name.endswith("CYX"):
                sg = next(
                    a for a, t in mapping.items() if template["atoms"][t][0] == "SG"
                )
                if not any(
                    mol.GetAtomWithIdx(b).GetAtomicNum() == 16
                    for b in outside.get(sg, ())
                ):
                    raise AmberError(
                        "the cysteine sulfur has neither a hydrogen nor a disulfide bond; "
                        "add the S-S bond or the hydrogen (MD System Builder > Protein does both)"
                    )
            return name, mapping
    raise AmberError(
        f"residue {resname} with {len(atoms)} atoms matches no ff14SB template "
        f"(tried {', '.join(tried)}); check its hydrogens and missing atoms"
    )


# --------------------------------------------------------------------------- terms


def _improper_order(center, outer, types, res_of, tidx, elements, wildcard):
    """tleap's order of the three outer atoms (as in OpenMM's 'amber' ordering)."""
    a2, a3, a4 = outer

    def later(x, y):
        return res_of[x] > res_of[y] or (res_of[x] == res_of[y] and tidx[x] > tidx[y])

    if not wildcard:
        if types[a2] == types[a4] and later(a2, a4):
            a2, a4 = a4, a2
        if types[a3] == types[a4] and later(a3, a4):
            a3, a4 = a4, a3
        if types[a2] == types[a3] and later(a2, a3):
            a2, a3 = a3, a2
    else:
        if elements[a2] == elements[a4] and later(a2, a4):
            a2, a4 = a4, a2
        if elements[a3] == elements[a4] and later(a3, a4):
            a3, a4 = a4, a3
        if later(a2, a3):
            a2, a3 = a3, a2
    return a2, a3, center, a4


def _improper_defs(ff):
    if "impropers" not in _cache:
        defs = []
        for key, term in ff["impropers"].items():
            t = key.split("-")
            defs.append((sum(x != "X" for x in t), t, term))
        defs.sort(key=lambda d: -d[0])
        _cache["impropers"] = defs
    return _cache["impropers"]


def molecule_type(mol, name, atom_names):
    """An mdff molecule type for a protein block (one or more chains)."""
    ff = load()
    groups = residues_of(mol)
    if not groups:
        raise AmberError("the molecule has no PDB residue information")
    res_of = {}
    for r, (_label, _resname, atoms) in enumerate(groups):
        for a in atoms:
            res_of[a] = r

    n = mol.GetNumAtoms()
    neighbours = [
        sorted(b.GetOtherAtomIdx(a.GetIdx()) for b in a.GetBonds())
        for a in mol.GetAtoms()
    ]
    types = [None] * n
    charges = [0.0] * n
    tidx = [0] * n
    residue_templates = []
    for r, (label, resname, atoms) in enumerate(groups):
        outside = {a: [b for b in neighbours[a] if res_of[b] != r] for a in atoms}
        try:
            tname, mapping = match_residue(mol, atoms, resname, ff, outside)
        except AmberError as exc:
            raise AmberError(f"{label}: {exc}") from exc
        template = ff["residues"][tname]
        residue_templates.append(tname)
        for a, t in mapping.items():
            _tname, atype, z, q = template["atoms"][t]
            if mol.GetAtomWithIdx(a).GetAtomicNum() != z:
                raise AmberError(f"{label}: element mismatch")
            types[a], charges[a], tidx[a] = atype, q, t

    elements = [a.GetSymbol() for a in mol.GetAtoms()]
    atoms_out = []
    for i in range(n):
        sigma, epsilon = ff["lj"][types[i]]
        atoms_out.append(
            {
                "name": atom_names[i],
                "element": elements[i],
                "type": types[i],
                "charge": charges[i],
                "mass": ff["masses"][types[i]],
                "sigma": sigma,
                "epsilon": epsilon,
            }
        )

    missing = []
    bonds = []
    for i in range(n):
        for j in neighbours[i]:
            if i < j:
                p = ff["bonds"].get(_key((types[i], types[j])))
                if p is None:
                    missing.append(f"bond {types[i]}-{types[j]}")
                    continue
                bonds.append({"i": i, "j": j, "r0": p[0], "k": p[1], "id": "ff14SB"})
    angles = []
    for j in range(n):
        for i, k in itertools.combinations(neighbours[j], 2):
            p = ff["angles"].get(_key((types[i], types[j], types[k])))
            if p is None:
                missing.append(f"angle {types[i]}-{types[j]}-{types[k]}")
                continue
            angles.append(
                {"i": i, "j": j, "k_atom": k, "theta0": p[0], "k": p[1], "id": "ff14SB"}
            )
    torsions = []
    for j in range(n):
        for k in neighbours[j]:
            if k <= j:
                continue
            for i in neighbours[j]:
                if i == k:
                    continue
                for l in neighbours[k]:
                    if l in (j, i):
                        continue
                    t = (types[i], types[j], types[k], types[l])
                    terms = ff["propers"].get(_key(t)) or ff["propers"].get(
                        _key(("X", t[1], t[2], "X"))
                    )
                    if terms is None:
                        missing.append("torsion " + "-".join(t))
                        continue
                    for per, phase, kk in terms:
                        if kk == 0.0:
                            continue
                        torsions.append(
                            {
                                "i": i,
                                "j": j,
                                "k_atom": k,
                                "l": l,
                                "periodicity": per,
                                "phase": phase,
                                "k": kk,
                                "improper": False,
                                "id": "ff14SB",
                            }
                        )
    defs = _improper_defs(ff)
    for center in range(n):
        if len(neighbours[center]) != 3:
            continue
        found = None
        for _specific, t, term in defs:
            if t[2] != types[center]:
                continue
            for outer in itertools.permutations(neighbours[center]):
                if all(t[p] in ("X", types[a]) for p, a in zip((0, 1, 3), outer)):
                    found = (t, term, outer)
                    break
            if found:
                break
        if not found:
            continue
        t, (per, phase, kk), outer = found
        a, b, c, d = _improper_order(
            center, outer, types, res_of, tidx, elements, "X" in t
        )
        torsions.append(
            {
                "i": a,
                "j": b,
                "k_atom": c,
                "l": d,
                "periodicity": per,
                "phase": phase,
                "k": kk,
                "improper": True,
                "id": "ff14SB",
            }
        )
    if missing:
        unique = sorted(set(missing))
        raise AmberError(
            f"no ff14SB parameters for {len(unique)} term(s), e.g. "
            + "; ".join(unique[:5])
        )
    return {
        "name": name,
        "atoms": atoms_out,
        "bonds": bonds,
        "angles": angles,
        "torsions": torsions,
        "exclusions_nrexcl": 3,
        "force_field": FORCE_FIELD_LABEL,
    }, residue_templates
