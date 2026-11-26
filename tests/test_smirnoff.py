"""SMIRNOFF reader and RDKit assignment."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers import HAS_RDKIT, mol3d

from force_field_builder import smirnoff

needs_rdkit = pytest.mark.skipif(not HAS_RDKIT, reason="RDKit not installed")

TINY = """<?xml version="1.0" encoding="utf-8"?>
<SMIRNOFF version="0.3" aromaticity_model="OEAroModel_MDL">
 <Bonds version="0.4" potential="harmonic">
  <Bond smirks="[*:1]~[*:2]" id="b-any" length="1.0 * angstrom" k="100.0 * kilocalorie_per_mole ** 1 * angstrom ** -2"/>
  <Bond smirks="[#6:1]-[#8:2]" id="b-co" length="1.4 * angstrom ** 1" k="300 * kilocalorie_per_mole ** 1 * angstrom ** -2"/>
 </Bonds>
 <Angles version="0.3" potential="harmonic">
  <Angle smirks="[*:1]~[*:2]~[*:3]" id="a-any" angle="109.5 * degree ** 1" k="50 * kilocalorie_per_mole ** 1 * radian ** -2"/>
 </Angles>
 <ProperTorsions version="0.4" potential="k*(1+cos(periodicity*theta-phase))" default_idivf="auto">
  <Proper smirks="[*:1]~[*:2]~[*:3]~[*:4]" id="t-any" periodicity1="3" phase1="0.0 * degree" k1="0.3 * kilocalorie_per_mole" idivf1="2.0"/>
 </ProperTorsions>
 <ImproperTorsions version="0.3" potential="k*(1+cos(periodicity*theta-phase))" default_idivf="auto">
  <Improper smirks="[*:1]~[#6X3:2](~[*:3])~[*:4]" id="i1" periodicity1="2" phase1="180.0 * degree" k1="3.0 * kilocalorie_per_mole"/>
 </ImproperTorsions>
 <vdW version="0.4" potential="Lennard-Jones-12-6" combining_rules="Lorentz-Berthelot" scale14="0.5">
  <Atom smirks="[*:1]" id="v-any" epsilon="0.1 * kilocalorie_per_mole" rmin_half="1.0 * angstrom"/>
  <Atom smirks="[#8:1]" id="v-o" epsilon="0.2 * mole ** -1 * kilocalorie ** 1" sigma="3.0 * angstrom"/>
 </vdW>
 <Electrostatics version="0.4" scale14="0.8333333333"/>
</SMIRNOFF>
"""


def test_parse_quantity_units():
    assert smirnoff.parse_quantity("1.5 * angstrom ** 1") == pytest.approx(0.15)
    assert smirnoff.parse_quantity(
        "0.3 * mole ** -1 * kilocalorie ** 1"
    ) == pytest.approx(1.2552)
    assert smirnoff.parse_quantity(
        "2 * kilocalorie_per_mole ** 1 * angstrom ** -2"
    ) == pytest.approx(836.8)
    assert smirnoff.parse_quantity("180.0 * degree ** 1") == 180.0
    assert smirnoff.parse_quantity("-1.0 * elementary_charge ** 1") == -1.0
    assert smirnoff.parse_quantity("1e-3 * nanometer") == pytest.approx(1e-3)
    with pytest.raises(smirnoff.SmirnoffError):
        smirnoff.parse_quantity("1 * furlong")
    with pytest.raises(smirnoff.SmirnoffError):
        smirnoff.parse_quantity("abc")


def test_sage_sections_are_all_read():
    ff = smirnoff.SmirnoffForceField()
    assert ff.unsupported_sections == []
    assert (
        len(ff.bonds),
        len(ff.angles),
        len(ff.propers),
        len(ff.impropers),
        len(ff.vdw),
    ) == (93, 55, 259, 7, 38)
    assert ff.scale14_lj == 0.5
    assert ff.scale14_qq == pytest.approx(0.8333333333)
    assert "AM1-BCC" in ff.charge_model
    # Impropers without idivf: "auto" means a third of k on each trefoil permutation.
    i1 = next(p for p in ff.impropers if p.id == "i1")
    assert i1.values["terms"][0]["k"] == pytest.approx(10.14166779508 * 4.184 / 3)


def test_rejects_other_aromaticity_and_bad_xml():
    with pytest.raises(smirnoff.SmirnoffError):
        smirnoff.SmirnoffForceField(
            text='<SMIRNOFF aromaticity_model="OEAroModel_Other"/>'
        )
    with pytest.raises(smirnoff.SmirnoffError):
        smirnoff.SmirnoffForceField(text="<nope")
    with pytest.raises(smirnoff.SmirnoffError):
        smirnoff.SmirnoffForceField(text="<Other/>")


@needs_rdkit
def test_last_match_wins_and_unit_conversion():
    ff = smirnoff.SmirnoffForceField(text=TINY)
    res = smirnoff.assign_parameters(ff, mol3d("CO"), use_fallback=False)
    co = next(b for b in res.bonds if {b["i"], b["j"]} == {0, 1})
    assert co["id"] == "b-co"
    assert co["r0"] == pytest.approx(0.14)
    assert co["k"] == pytest.approx(300 * 418.4)
    assert {b["id"] for b in res.bonds if b is not co} == {"b-any"}
    assert res.vdw[1]["id"] == "v-o" and res.vdw[1]["sigma"] == pytest.approx(0.3)
    # rmin_half 1 A -> sigma = 2 * 0.1 / 2^(1/6) nm
    assert res.vdw[0]["sigma"] == pytest.approx(0.2 / 2 ** (1 / 6))
    assert res.vdw[0]["epsilon"] == pytest.approx(0.4184)
    # explicit idivf divides k
    assert all(t["k"] == pytest.approx(0.3 * 4.184 / 2) for t in res.torsions)
    assert res.missing == []


@needs_rdkit
def test_improper_trefoil_matches_toolkit_convention():
    ff = smirnoff.SmirnoffForceField(text=TINY)
    res = smirnoff.assign_parameters(ff, mol3d("C=O"), use_fallback=False)
    imps = [t for t in res.torsions if t["improper"]]
    assert len(imps) == 3
    center = imps[0]["i"]
    others = sorted({imps[0]["j"], imps[0]["k_atom"], imps[0]["l"]})
    a, b, c = others
    assert [(t["j"], t["k_atom"], t["l"]) for t in imps] == [
        (a, b, c),
        (b, c, a),
        (c, a, b),
    ]
    assert all(t["i"] == center for t in imps)
    assert all(t["k"] == pytest.approx(3.0 * 4.184 / 3) for t in imps)


@needs_rdkit
def test_sage_ethanol_assignment():
    ff = smirnoff.SmirnoffForceField()
    mol = mol3d("CCO")
    res = smirnoff.assign_parameters(ff, mol)
    assert [res.vdw[i]["id"] for i in range(3)] == ["n16", "n16", "n19"]
    ids = {(b["i"], b["j"]): b["id"] for b in res.bonds}
    assert ids[(0, 1)] == "b1" and ids[(1, 2)] == "b14"
    assert len(res.angles) == 13
    assert res.library_charges is None
    assert res.missing == [] and res.fallbacks == []
    assert len(res.constraints) == 6  # every bond to hydrogen


@needs_rdkit
def test_sage_library_charges_cover_water_and_ions():
    ff = smirnoff.SmirnoffForceField()
    assert smirnoff.assign_parameters(ff, mol3d("O")).library_charges == [
        -0.834,
        0.417,
        0.417,
    ]
    assert smirnoff.assign_parameters(ff, mol3d("[Na+]")).library_charges == [1.0]
    assert smirnoff.assign_parameters(ff, mol3d("[Cl-]")).library_charges == [-1.0]


@needs_rdkit
def test_aromatic_molecules_get_aromatic_parameters():
    ff = smirnoff.SmirnoffForceField()
    res = smirnoff.assign_parameters(ff, mol3d("c1ccccc1"))
    ring = [b for b in res.bonds if b["i"] < 6 and b["j"] < 6]
    assert {b["id"] for b in ring} == {"b5"}
    assert len([t for t in res.torsions if t["improper"]]) == 6 * 3


@needs_rdkit
def test_implicit_hydrogens_are_rejected():
    from rdkit import Chem

    with pytest.raises(smirnoff.SmirnoffError, match="implicit hydrogens"):
        smirnoff.assign_parameters(
            smirnoff.SmirnoffForceField(), Chem.MolFromSmiles("CCO")
        )


@needs_rdkit
def test_uff_fallback_for_unparameterised_atoms():
    ff = smirnoff.SmirnoffForceField()
    res = smirnoff.assign_parameters(ff, mol3d("[Fe]"))
    assert res.vdw[0]["id"] == "UFF"
    assert res.vdw[0]["sigma"] > 0
    assert res.fallbacks
    nofb = smirnoff.assign_parameters(ff, mol3d("[Fe]"), use_fallback=False)
    assert any("vdW" in m for m in nofb.missing)


@needs_rdkit
def test_topology_terms_counts():
    work = smirnoff.prepare_for_matching(mol3d("C1CC1"))
    bonds, angles, propers = smirnoff.topology_terms(work)
    assert len(bonds) == 9
    # checked against a brute-force count of i-j-k triples around each centre
    brute = set()
    for a in work.GetAtoms():
        nb = [n.GetIdx() for n in a.GetNeighbors()]
        for x in range(len(nb)):
            for y in range(x + 1, len(nb)):
                brute.add(tuple(sorted((nb[x], nb[y]))) + (a.GetIdx(),))
    assert len(angles) == len(brute)
    for i, j, k, l in propers:
        assert len({i, j, k, l}) == 4
        assert (
            work.GetBondBetweenAtoms(i, j)
            and work.GetBondBetweenAtoms(j, k)
            and work.GetBondBetweenAtoms(k, l)
        )
