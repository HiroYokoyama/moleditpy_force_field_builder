"""Shared test helpers: 3D molecules and fake host objects."""

import importlib.util


def has(module):
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


HAS_RDKIT = has("rdkit")
HAS_NUMPY = has("numpy")
HAS_QT = has("PySide6")


def mol3d(smiles, seed=7):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if mol.GetNumAtoms() > 1:
        assert AllChem.EmbedMolecule(mol, randomSeed=seed) == 0
    else:
        conf = Chem.Conformer(1)
        mol.AddConformer(conf, assignId=True)
    return mol


def combine(*mols):
    from rdkit import Chem

    out = mols[0]
    for m in mols[1:]:
        out = Chem.CombineMols(out, m)
    return out


def shifted(mol, dx):
    """A copy of *mol* translated along x (keeps combined systems from overlapping)."""
    from rdkit import Chem
    from rdkit.Geometry import Point3D

    out = Chem.Mol(mol)
    conf = out.GetConformer()
    for i in range(out.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, Point3D(p.x + dx, p.y, p.z))
    return out


def water_box(n, spacing=3.1):
    """n waters on a line (O first), separated by *spacing* Angstrom."""
    return combine(*(shifted(mol3d("O", seed=n + 1), spacing * k) for k in range(n)))
