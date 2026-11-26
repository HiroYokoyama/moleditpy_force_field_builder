"""Amber ff14SB for proteins: template matching, parameters, energies against sander."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers import HAS_NUMPY, HAS_RDKIT, combine, shifted, water_box

pytestmark = pytest.mark.skipif(
    not (HAS_RDKIT and HAS_NUMPY), reason="RDKit and numpy needed"
)

from force_field_builder import amber, builder, mdff

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
PEPTIDE = os.path.join(DATA, "peptide_amber.pdb")
KCAL = 4.184

#: sander (AmberTools 25, tleap + leaprc.protein.ff14SB, ntb=0, igb=0, cut=999) on
#: peptide_amber.pdb, kcal/mol. LJ and Coulomb include the scaled 1-4 terms.
SANDER = {
    "bond": 35.6233,
    "angle": 45.5336,
    "torsion": 88.7096,
    "lj": -29.5029 + 26.1856,
    "coulomb": -611.7558 + 372.3476,
}


def _peptide():
    from rdkit import Chem

    return Chem.MolFromPDBFile(PEPTIDE, removeHs=False, sanitize=False)


def _names(mol):
    return [a.GetPDBResidueInfo().GetName().strip() for a in mol.GetAtoms()]


def test_data_file():
    ff = amber.load()
    assert ff["fudge_lj"] == 0.5 and ff["fudge_qq"] == pytest.approx(1 / 1.2)
    assert {"ALA", "NALA", "CALA", "HIE", "CYX", "ACE", "NME"} <= set(ff["residues"])
    for name, res in ff["residues"].items():
        total = sum(a[3] for a in res["atoms"])
        assert total == pytest.approx(round(total), abs=1e-3), name
    # backbone phi of ff14SB (C-N-CX-C), kcal/mol x 4.184
    phi = {term[0]: term[2] for term in ff["propers"]["C-CX-N-C"]}
    assert phi[3] == pytest.approx(0.42 * KCAL) and phi[2] == pytest.approx(0.27 * KCAL)
    assert ff["impropers"]["X-X-C-O"] == [2, 180.0, pytest.approx(10.5 * KCAL)]


def test_residues_are_matched_by_graph():
    mol = _peptide()
    assert amber.is_protein(mol)
    mtype, templates = amber.molecule_type(mol, "PROTEIN", _names(mol))
    assert templates == ["NALA", "CYX", "ASP", "GLU", "HIE", "LYS", "ARG", "CCYX"]
    assert sum(a["charge"] for a in mtype["atoms"]) == pytest.approx(0.0, abs=1e-6)
    assert mtype["force_field"] == amber.FORCE_FIELD_LABEL
    ss = [
        b
        for b in mtype["bonds"]
        if {mtype["atoms"][b["i"]]["type"], mtype["atoms"][b["j"]]["type"]} == {"S"}
    ]
    assert len(ss) == 1


def test_hydrogen_names_do_not_matter():
    mol = _peptide()
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "H":
            atom.GetPDBResidueInfo().SetName(f" H{atom.GetIdx() % 7} ")
    _mtype, templates = amber.molecule_type(mol, "P", _names(mol))
    assert templates == ["NALA", "CYX", "ASP", "GLU", "HIE", "LYS", "ARG", "CCYX"]


def test_protonation_picks_the_template():
    from rdkit import Chem

    mol = Chem.RWMol(_peptide())
    # move HE2 of His5 to ND1: HIE becomes HID
    names = _names(mol)
    he2 = next(
        a.GetIdx()
        for a in mol.GetAtoms()
        if names[a.GetIdx()] == "HE2" and a.GetPDBResidueInfo().GetResidueNumber() == 5
    )
    ring = {
        names[a.GetIdx()]: a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetPDBResidueInfo().GetResidueNumber() == 5
    }
    mol.RemoveBond(ring["NE2"], he2)
    mol.AddBond(ring["ND1"], he2, Chem.BondType.SINGLE)
    _mtype, templates = amber.molecule_type(mol, "P", _names(mol))
    assert templates[4] == "HID"


def test_cysteine_without_partner_or_hydrogen_is_an_error():
    from rdkit import Chem

    mol = Chem.RWMol(_peptide())
    sg = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "S"]
    mol.RemoveBond(sg[0], sg[1])
    with pytest.raises(amber.AmberError, match="CYX2.*disulfide"):
        amber.molecule_type(mol, "P", _names(mol))


def test_unknown_residue_is_not_a_protein():
    mol = _peptide()
    mol.GetAtomWithIdx(0).GetPDBResidueInfo().SetResidueName("XYZ")
    assert not amber.is_protein(mol)


def test_energy_matches_sander():
    from mdff_energy import energy

    mol = _peptide()
    result = builder.build(mol, {"constrain_h": False})
    assert [t.kind for t in result.types] == ["protein"]
    assert not mdff.validate(result.mdff)
    assert "ff14SB" in result.mdff["source"]
    xyz = mol.GetConformer().GetPositions() / 10.0
    ours = {k: v / KCAL for k, v in energy(result.mdff, xyz).items()}
    for term in ("bond", "angle", "torsion", "lj"):
        assert ours[term] == pytest.approx(SANDER[term], abs=2e-3), term
    # Amber's Coulomb constant is 0.003 % smaller than the one mdff uses
    assert ours["coulomb"] == pytest.approx(SANDER["coulomb"], rel=1e-4)


def test_protein_with_water_and_ions_in_one_build():
    from helpers import mol3d

    system = combine(
        _peptide(),
        shifted(water_box(2), 30),
        shifted(mol3d("[Na+]"), 40),
    )
    result = builder.build(system)
    kinds = {t.name: t.kind for t in result.types}
    assert kinds["PROTEIN"] == "protein"
    assert set(kinds.values()) == {"protein", "water", "ion"}
    protein = next(t for t in result.types if t.kind == "protein")
    assert protein.charges_from_force_field
    assert protein.charge_method == "ff14SB residue charges"
    mt = next(m for m in result.mdff["molecule_types"] if m["name"] == "PROTEIN")
    assert mt["constraints"]  # bonds to hydrogen constrained like any molecule
    assert not mdff.validate(result.mdff)


def test_imported_charges_are_not_used_for_proteins():
    mol = _peptide()
    charges = [0.0] * mol.GetNumAtoms()
    result = builder.build(mol, {"charge_method": "imported"}, imported_charges=charges)
    info = result.types[0]
    assert info.charge_method == "ff14SB residue charges"
    assert any("Imported charges are not used" in w for w in info.warnings)
