"""Split the host molecule into molecule types and system entries.

The ``moleditpy-mdff`` ``system`` list describes the host molecule in atom
order as runs of identical molecule types: ``{"type", "count",
"atom_offset"}``. Each copy of a type has to occupy a contiguous block of host
atoms. A covalent fragment whose atoms interleave with another fragment is
merged with it into one block (one molecule type may hold several fragments).

Blocks with the same local topology (element, formal charge and bond list in
the same local order) are one type, parameterised once. A block with the same
chemistry but another atom order reuses the parameters of the first one
through a graph match.
"""

from __future__ import annotations


def fragment_blocks(mol):
    """Contiguous (start, end) atom ranges, each holding whole fragments."""
    from rdkit import Chem

    n_atoms = mol.GetNumAtoms()
    if n_atoms == 0:
        return []
    spans = sorted((min(f), max(f)) for f in Chem.GetMolFrags(mol, sanitizeFrags=False))
    blocks = []
    start, end = spans[0]
    for s, e in spans[1:]:
        if s <= end:
            end = max(end, e)
        else:
            blocks.append((start, end + 1))
            start, end = s, e
    blocks.append((start, end + 1))
    return blocks


def block_signature(mol, start, end):
    """A hashable description of the block's local topology."""
    atoms = tuple(
        (mol.GetAtomWithIdx(i).GetAtomicNum(), mol.GetAtomWithIdx(i).GetFormalCharge())
        for i in range(start, end)
    )
    bonds = []
    for i in range(start, end):
        for bond in mol.GetAtomWithIdx(i).GetBonds():
            j = bond.GetOtherAtomIdx(i)
            if i < j:
                bonds.append((i - start, j - start, bond.GetBondTypeAsDouble()))
    return atoms, tuple(sorted(bonds))


def extract_block(mol, start, end):
    """A new RDKit molecule holding host atoms start..end-1 in the same order."""
    from rdkit import Chem
    from rdkit.Geometry import Point3D

    rw = Chem.RWMol()
    for i in range(start, end):
        src = mol.GetAtomWithIdx(i)
        atom = Chem.Atom(src.GetAtomicNum())
        atom.SetFormalCharge(src.GetFormalCharge())
        atom.SetNumRadicalElectrons(src.GetNumRadicalElectrons())
        atom.SetIsotope(src.GetIsotope())
        atom.SetNoImplicit(src.GetNoImplicit())
        atom.SetNumExplicitHs(src.GetNumExplicitHs())
        info = src.GetPDBResidueInfo()
        if info is not None:
            atom.SetMonomerInfo(info)
        rw.AddAtom(atom)
    for i in range(start, end):
        for bond in mol.GetAtomWithIdx(i).GetBonds():
            j = bond.GetOtherAtomIdx(i)
            if i < j < end:
                rw.AddBond(i - start, j - start, bond.GetBondType())
    out = rw.GetMol()
    if mol.GetNumConformers():
        conf_src = mol.GetConformer()
        conf = Chem.Conformer(end - start)
        for i in range(start, end):
            p = conf_src.GetAtomPosition(i)
            conf.SetAtomPosition(i - start, Point3D(p.x, p.y, p.z))
        conf.Set3D(conf_src.Is3D())
        out.AddConformer(conf, assignId=True)
    try:
        Chem.SanitizeMol(out)
    except Exception:  # noqa: BLE001 - Leave unsanitisable blocks to the caller: assignment reports the error.
        out.UpdatePropertyCache(strict=False)
    return out


def implicit_hydrogen_count(mol, skip=()):
    """Total implicit hydrogens of the host molecule (they need to be atoms)."""
    total = 0
    for atom in mol.GetAtoms():
        if atom.GetIdx() in skip:
            continue
        try:
            total += atom.GetNumImplicitHs()
        except Exception:  # noqa: BLE001
            atom.UpdatePropertyCache(strict=False)
            total += atom.GetNumImplicitHs()
    return total


def classify_block(sig):
    """'water', 'ion' or 'molecule' from a block signature."""
    atoms, bonds = sig
    if len(atoms) == 3 and len(bonds) == 2 and all(c == 0 for _, c in atoms):
        elements = sorted(z for z, _ in atoms)
        if elements == [1, 1, 8]:
            o_idx = next(i for i, (z, _) in enumerate(atoms) if z == 8)
            if all(o_idx in (i, j) and order == 1.0 for i, j, order in bonds):
                return "water"
    if len(atoms) == 1 and atoms[0][1] != 0:
        return "ion"
    return "molecule"


def residue_name(mol, start):
    info = mol.GetAtomWithIdx(start).GetPDBResidueInfo()
    if info is None:
        return ""
    return (info.GetResidueName() or "").strip()


def atom_names(mol, start, end):
    """PDB names when every atom has one, unique within its residue; else element+count."""
    names = []
    keys = []
    for i in range(start, end):
        info = mol.GetAtomWithIdx(i).GetPDBResidueInfo()
        name = (info.GetName() or "").strip() if info is not None else ""
        names.append(name)
        if info is not None:
            keys.append(
                (
                    info.GetChainId(),
                    info.GetResidueNumber(),
                    info.GetInsertionCode(),
                    info.GetResidueName(),
                    name,
                )
            )
    if all(names) and len(set(keys)) == len(keys):
        return names
    counts = {}
    out = []
    for i in range(start, end):
        symbol = mol.GetAtomWithIdx(i).GetSymbol()
        counts[symbol] = counts.get(symbol, 0) + 1
        out.append(f"{symbol}{counts[symbol]}")
    return out


def graph_permutation(template, block):
    """Map template atom index -> block atom index for the same molecule, or None."""
    if template.GetNumAtoms() != block.GetNumAtoms():
        return None
    try:
        match = block.GetSubstructMatch(template, useChirality=False)
    except Exception:  # noqa: BLE001
        return None
    if len(match) != template.GetNumAtoms():
        return None
    return list(match)


def remap_molecule_type(mtype, perm, new_name):
    """Re-index an mdff molecule type: template atom t becomes block atom perm[t]."""
    n = len(mtype["atoms"])
    atoms = [None] * n
    for t, b in enumerate(perm):
        atoms[b] = dict(mtype["atoms"][t])
    out = dict(mtype)
    out["name"] = new_name
    out["atoms"] = atoms

    def m(i):
        return perm[i]

    out["bonds"] = [dict(b, i=m(b["i"]), j=m(b["j"])) for b in mtype.get("bonds", [])]
    out["angles"] = [
        dict(a, i=m(a["i"]), j=m(a["j"]), k_atom=m(a["k_atom"]))
        for a in mtype.get("angles", [])
    ]
    out["torsions"] = [
        dict(t, i=m(t["i"]), j=m(t["j"]), k_atom=m(t["k_atom"]), l=m(t["l"]))
        for t in mtype.get("torsions", [])
    ]
    if mtype.get("constraints"):
        out["constraints"] = [
            dict(c, i=m(c["i"]), j=m(c["j"])) for c in mtype["constraints"]
        ]
    if mtype.get("settles"):
        out["settles"] = dict(mtype["settles"], i=m(mtype["settles"]["i"]))
    return out


def system_entries(block_types):
    """Collapse a per-block list of type names into runs of the mdff ``system`` list.

    *block_types* is [(type_name, start, end), ...] in host order.
    """
    entries = []
    for name, start, end in block_types:
        if entries and entries[-1]["type"] == name and entries[-1]["_next"] == start:
            entries[-1]["count"] += 1
            entries[-1]["_next"] = end
        else:
            entries.append(
                {"type": name, "count": 1, "atom_offset": start, "_next": end}
            )
    for entry in entries:
        del entry["_next"]
    return entries


def molecule_order(mol):
    """Atom order that makes every covalent molecule contiguous, or None if it already is.

    Molecules keep the order of their first atom; atoms keep their relative
    order inside each molecule. MoleditPy's 2D->3D conversion appends all
    hydrogens after all heavy atoms, which interleaves separate molecules.
    """
    from rdkit import Chem

    frags = sorted(Chem.GetMolFrags(mol, sanitizeFrags=False), key=min)
    order = [i for frag in frags for i in sorted(frag)]
    if order == list(range(mol.GetNumAtoms())):
        return None
    return order


def reorder_by_molecule(mol):
    """A copy of *mol* with each molecule's atoms contiguous (props and conformers kept)."""
    from rdkit import Chem

    order = molecule_order(mol)
    if order is None:
        return Chem.Mol(mol)
    out = Chem.RenumberAtoms(mol, order)
    for name in mol.GetPropNames():
        if not out.HasProp(name):
            out.SetProp(name, mol.GetProp(name))
    return out
