"""Optional sources: GROMACS itp import, OpenMM System import, toolkit cross-check."""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers import HAS_RDKIT, has, mol3d

from force_field_builder import external, smirnoff

# acpype-style output (atomtypes with bond_type column, propers funct 9, impropers funct 4)
ACPYPE_ITP = """
[ atomtypes ]
;name   bond_type     mass     charge   ptype   sigma         epsilon       Amb
 c3       c3          0.00000  0.00000   A     3.39771e-01   4.51035e-01 ; 1.91  0.1078
 hc       hc          0.00000  0.00000   A     2.60018e-01   8.70272e-02 ; 1.46  0.0208
 oh       oh          0.00000  0.00000   A     3.24287e-01   3.89112e-01 ; 1.82  0.0930

[ moleculetype ]
;name            nrexcl
 MOL              3

[ atoms ]
;   nr  type  resi  res  atom  cgnr     charge      mass       ; qtot   bond_type
     1   c3     1   MOL    C1    1    0.116700     12.01000 ; qtot 0.117
     2   oh     1   MOL    O1    2   -0.599000     16.00000 ; qtot -0.482
     3   hc     1   MOL    H1    3    0.028100      1.00800 ; qtot -0.454

[ bonds ]
;   ai     aj funct   r             k
     1      2   1    1.4233e-01    2.6870e+05 ;     C1 - O1
     1      3   1    1.0969e-01    2.7665e+05 ;     C1 - H1

[ angles ]
;   ai     aj     ak    funct   theta         cth
     2      1      3      1    1.0988e+02    4.2585e+02 ;     O1 - C1     - H1

[ dihedrals ] ; propers
;    i      j      k      l   func   phase     kd      pn
     3      1      2      4      9     0.00   0.69873   3 ;     H1-    C1-    O1-    HO
[ dihedrals ] ; impropers
     1      2      3      4      4   180.00   4.60240   2
     1      2      3      4      3     0.1 0.2 0.3 0.4 0.5 0.6
"""


def test_parse_acpype_itp():
    itp = ACPYPE_ITP.replace(
        "     3      1      2      4      9", "     3      1      2      3      9"
    )
    mt, warnings = external.parse_gromacs_itp(
        itp.replace("1      2      3      4      4", "1      2      3      2      4")
    )
    assert mt["name"] == "MOL" and mt["exclusions_nrexcl"] == 3
    assert [a["type"] for a in mt["atoms"]] == ["c3", "oh", "hc"]
    assert mt["atoms"][0]["sigma"] == pytest.approx(0.339771)
    assert mt["atoms"][1]["mass"] == 16.0
    assert mt["bonds"][0] == {"i": 0, "j": 1, "r0": 0.14233, "k": 268700.0}
    assert mt["angles"][0]["theta0"] == pytest.approx(109.88)
    proper, improper = mt["torsions"]
    assert proper["improper"] is False and proper["periodicity"] == 3
    assert improper["improper"] is True and improper["phase"] == 180.0
    assert any("funct 3" in w for w in warnings)


def test_missing_atomtype_is_reported():
    mt, warnings = external.parse_gromacs_itp("[ atoms ]\n 1 zz 1 MOL X1 1 0.0 1.0\n")
    assert mt["atoms"][0]["sigma"] == 0.0
    assert any("zz" in w for w in warnings)


class _Q:
    """Stands in for an OpenMM Quantity in MD units."""

    def __init__(self, value):
        self._value = value


class _Force:
    def __init__(self, name, **items):
        self.__class__ = type(name, (_Force,), {})
        self.items = items

    def getNumBonds(self):
        return len(self.items["bonds"])

    def getBondParameters(self, n):
        return self.items["bonds"][n]

    def getNumAngles(self):
        return len(self.items["angles"])

    def getAngleParameters(self, n):
        return self.items["angles"][n]

    def getNumTorsions(self):
        return len(self.items["torsions"])

    def getTorsionParameters(self, n):
        return self.items["torsions"][n]

    def getNumParticles(self):
        return len(self.items["particles"])

    def getParticleParameters(self, n):
        return self.items["particles"][n]


class _System:
    def __init__(self, forces, masses):
        self.forces = forces
        self.masses = masses

    def getForces(self):
        return self.forces

    def getParticleMass(self, i):
        return _Q(self.masses[i])


@pytest.mark.skipif(not HAS_RDKIT, reason="RDKit not installed")
def test_openmm_system_import():
    mol = mol3d("C=O")  # C0 O1 H2 H3
    forces = [
        _Force("HarmonicBondForce", bonds=[(0, 1, _Q(0.123), _Q(5e5))]),
        _Force(
            "HarmonicAngleForce", angles=[(2, 0, 1, _Q(math.radians(120.0)), _Q(400.0))]
        ),
        _Force(
            "PeriodicTorsionForce", torsions=[(0, 2, 1, 3, 2, _Q(math.pi), _Q(4.0))]
        ),
        _Force(
            "NonbondedForce",
            particles=[
                (_Q(0.5), _Q(0.3), _Q(0.4)),
                (_Q(-0.5), _Q(0.29), _Q(0.8)),
                (_Q(0.0), _Q(0.25), _Q(0.06)),
                (_Q(0.0), _Q(0.25), _Q(0.06)),
            ],
        ),
        _Force("CustomExternalForce"),
    ]
    mt, warnings = external.mdff_type_from_openmm_system(
        _System(forces, [12.0, 16.0, 1.0, 1.0]), mol, "FOR"
    )
    assert mt["bonds"] == [{"i": 0, "j": 1, "r0": 0.123, "k": 5e5}]
    assert mt["angles"][0]["theta0"] == pytest.approx(120.0)
    assert mt["torsions"][0]["improper"] is True and mt["torsions"][0][
        "phase"
    ] == pytest.approx(180.0)
    assert mt["atoms"][0]["charge"] == 0.5 and mt["atoms"][1]["epsilon"] == 0.8
    assert mt["atoms"][2]["type"] == mt["atoms"][3]["type"] != mt["atoms"][0]["type"]
    assert warnings == ["CustomExternalForce is not converted"]


def test_compare_labels():
    ours = {"Bonds": {(0, 1): "b1"}, "vdW": {(0,): "n1"}}
    theirs = {"Bonds": {(0, 1): "b2"}, "vdW": {(0,): "n1"}}
    diffs = external.compare_labels(ours, theirs)
    assert diffs == ["Bonds (0, 1): builder b1 vs toolkit b2"]
    assert external.compare_labels(ours, ours) == []


def test_missing_tools_raise_readable_errors(monkeypatch):
    monkeypatch.setattr(external.shutil, "which", lambda _name: None)
    with pytest.raises(external.ExternalError, match="acpype"):
        external.run_acpype(None)


# -- the cross-check the brief asks for, run only where openff-toolkit is installed --------

CROSS_CHECK_SMILES = [
    "CCO",
    "c1ccccc1O",
    "CC(=O)N",
    "CC(=O)[O-]",
    "C[NH3+]",
    "c1ccncc1",
    "c1ccc2ccccc2c1",
    "C#N",
    "CS(=O)(=O)N",
    "CP(=O)(O)O",
    "FC(F)(F)c1ccccc1Br",
    "c1cc[nH]c1",
    "c1ccsc1",
    "O=C1CCCCC1",
    "NC(=N)N",
    "O=[N+]([O-])c1ccccc1",
    "CN(C)C=O",
    "C1CC1",
    "OC(=O)C(N)Cc1ccc(O)cc1",
    "CC(=O)Oc1ccccc1C(=O)O",
    "OB(O)c1ccccc1",
    "[Na+]",
    "[Cl-]",
    "O",
    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
]


@pytest.mark.skipif(
    not (HAS_RDKIT and has("openff.toolkit")), reason="openff-toolkit not installed"
)
@pytest.mark.parametrize("smiles", CROSS_CHECK_SMILES)
def test_assignment_matches_openff_toolkit(smiles):
    from rdkit import Chem

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    ff = smirnoff.SmirnoffForceField()
    ours = external.ours_as_labels(
        smirnoff.assign_parameters(ff, mol, use_fallback=False)
    )
    theirs = external.toolkit_labels(mol, smirnoff.SAGE_PATH)
    assert external.compare_labels(ours, theirs) == []
