"""GROMACS, OpenMM XML and LAMMPS writers checked against the mdff energy."""

import math
import os
import sys
import xml.etree.ElementTree as ET

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers import (
    HAS_NUMPY,
    HAS_RDKIT,
    combine,
    has,
    mol3d,
    shifted,
    water_box,
)

pytestmark = pytest.mark.skipif(
    not (HAS_RDKIT and HAS_NUMPY), reason="RDKit and numpy needed"
)

from force_field_builder import builder, external, writers
from force_field_builder.display import molecule_positions


@pytest.fixture(scope="module")
def system():
    from rdkit.Chem import AllChem

    solute = mol3d("CC(=O)Oc1ccccc1C(=O)O")
    AllChem.MMFFOptimizeMolecule(solute)
    mol = combine(
        solute,
        shifted(water_box(2), 9),
        shifted(mol3d("[Na+]"), 18),
        shifted(mol3d("[Cl-]"), 22),
    )
    res = builder.build(mol, {"charge_method": "mmff94", "constrain_h": False})
    return mol, res.mdff


def test_gromacs_files_structure(system):
    _mol, data = system
    files = writers.gromacs_files(data, "sys")
    assert set(files) == {
        "sys.top",
        "sys_MOL.itp",
        "sys_SOL.itp",
        "sys_NA.itp",
        "sys_CL.itp",
    }
    top = files["sys.top"]
    assert "[ defaults ]" in top and "yes        0.5" in top
    assert (
        top.index("[ atomtypes ]")
        < top.index('#include "sys_MOL.itp"')
        < top.index("[ molecules ]")
    )
    molecules = top.split("[ molecules ]")[1].split("\n")[2:]
    assert [line.split() for line in molecules if line.strip()] == [
        ["MOL", "1"],
        ["SOL", "2"],
        ["NA", "1"],
        ["CL", "1"],
    ]
    sol = files["sys_SOL.itp"]
    assert (
        "[ settles ]" in sol and "[ exclusions ]" in sol and "#ifndef FLEXIBLE" in sol
    )


def test_gromacs_itp_round_trip(system):
    _mol, data = system
    files = writers.gromacs_files(data, "sys")
    parsed, warnings = external.parse_gromacs_itp(
        files["sys_MOL.itp"], files["sys.top"]
    )
    assert warnings == []
    mol_type = data["molecule_types"][0]
    assert len(parsed["atoms"]) == len(mol_type["atoms"])
    for a, b in zip(parsed["atoms"], mol_type["atoms"]):
        assert a["charge"] == pytest.approx(b["charge"], abs=1e-8)
        assert a["sigma"] == pytest.approx(b["sigma"], rel=1e-8)
        assert a["epsilon"] == pytest.approx(b["epsilon"], rel=1e-8)
    assert len(parsed["torsions"]) == len(mol_type["torsions"])
    for t, u in zip(parsed["torsions"], mol_type["torsions"]):
        assert (
            t["i"],
            t["j"],
            t["k_atom"],
            t["l"],
            t["periodicity"],
            t["improper"],
        ) == (
            u["i"],
            u["j"],
            u["k_atom"],
            u["l"],
            u["periodicity"],
            u["improper"],
        )
        assert t["k"] == pytest.approx(u["k"], rel=1e-7)


def test_gromacs_pairs_are_1_4_only(system):
    _mol, data = system
    itp = writers.gromacs_files(data, "s")["s_MOL.itp"]
    pairs = itp.split("[ pairs ]")[1].split("[ angles ]")[0]
    rows = [
        tuple(int(x) for x in line.split()[:2])
        for line in pairs.splitlines()[2:]
        if line.strip()
    ]
    from force_field_builder.mdff import neighbour_distances

    mt = data["molecule_types"][0]
    dist = neighbour_distances(len(mt["atoms"]), mt["bonds"])
    assert rows and all(dist[(i - 1, j - 1)] == 3 for i, j in rows)


def test_gromacs_water_without_oxygen_first_uses_constraints():
    from rdkit import Chem

    rw = Chem.RWMol()
    for z in (1, 8, 1):
        rw.AddAtom(Chem.Atom(z))
    rw.AddBond(0, 1, Chem.BondType.SINGLE)
    rw.AddBond(1, 2, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    data = builder.build(mol).mdff
    itp = writers.gromacs_itp(data["molecule_types"][0])
    assert "[ settles ]" not in itp and "[ constraints ]" in itp


def test_openmm_xml_is_well_formed(system):
    _mol, data = system
    root = ET.fromstring(writers.openmm_xml(data))
    n_atoms = sum(len(mt["atoms"]) for mt in data["molecule_types"])
    assert len(root.find("AtomTypes")) == n_atoms
    assert len(root.find("Residues")) == len(data["molecule_types"])
    torsion_forces = root.findall("PeriodicTorsionForce")
    assert torsion_forces[0].get("ordering") == "smirnoff"
    impropers = torsion_forces[0].findall("Improper")
    n_imp = sum(
        1 for mt in data["molecule_types"] for t in mt["torsions"] if t["improper"]
    )
    assert len(impropers) * 3 == n_imp  # one entry per SMIRNOFF trefoil
    nb = root.find("NonbondedForce")
    assert float(nb.get("lj14scale")) == 0.5


def test_trefoil_grouping_leaves_single_impropers():
    imps = [
        {
            "i": 0,
            "j": 1,
            "k_atom": 2,
            "l": 3,
            "periodicity": 2,
            "phase": 180.0,
            "k": 1.0,
        },
        {
            "i": 0,
            "j": 2,
            "k_atom": 3,
            "l": 1,
            "periodicity": 2,
            "phase": 180.0,
            "k": 1.0,
        },
        {
            "i": 0,
            "j": 3,
            "k_atom": 1,
            "l": 2,
            "periodicity": 2,
            "phase": 180.0,
            "k": 1.0,
        },
        {
            "i": 5,
            "j": 6,
            "k_atom": 7,
            "l": 8,
            "periodicity": 2,
            "phase": 180.0,
            "k": 4.0,
        },
    ]
    trefoils, rest = writers._trefoil_groups(imps)
    assert len(trefoils) == 1 and rest == [imps[3]]


@pytest.mark.skipif(not has("openmm"), reason="OpenMM not installed")
def test_openmm_energy_matches_reference_evaluator(system, tmp_path):
    import mdff_energy
    import numpy as np
    import openmm
    from openmm import app, unit

    mol, data = system
    data = dict(
        data, molecule_types=[data["molecule_types"][0]], system=[data["system"][0]]
    )
    n = len(data["molecule_types"][0]["atoms"])
    path = tmp_path / "ff.xml"
    path.write_text(writers.openmm_xml(data), encoding="utf-8")
    ff = app.ForceField(str(path))
    top = app.Topology()
    chain = top.addChain()
    res = top.addResidue("MOL", chain)
    atoms = []
    for a in data["molecule_types"][0]["atoms"]:
        atoms.append(top.addAtom(a["name"], app.Element.getBySymbol(a["element"]), res))
    for b in data["molecule_types"][0]["bonds"]:
        top.addBond(atoms[b["i"]], atoms[b["j"]])
    sys_ = ff.createSystem(top, nonbondedMethod=app.NoCutoff, constraints=None)
    pos = np.array(molecule_positions(mol)[:n]) / 10.0
    ctx = openmm.Context(
        sys_,
        openmm.VerletIntegrator(1.0),
        openmm.Platform.getPlatformByName("Reference"),
    )
    ctx.setPositions(pos * unit.nanometer)
    e_omm = (
        ctx.getState(getEnergy=True)
        .getPotentialEnergy()
        .value_in_unit(unit.kilojoule_per_mole)
    )
    e_ref = sum(mdff_energy.energy(data, pos).values())
    assert e_omm == pytest.approx(e_ref, abs=1e-3)


# -- LAMMPS ---------------------------------------------------------------------------------


def _parse_lammps(text):
    sections = {}
    current = None
    header = []
    for line in text.splitlines()[1:]:
        stripped = line.split("#")[0].strip()
        if not stripped:
            continue
        words = stripped.split()
        if words[0] in (
            "Masses",
            "Pair",
            "Bond",
            "Angle",
            "Dihedral",
            "Atoms",
            "Bonds",
            "Angles",
            "Dihedrals",
        ):
            current = (
                " ".join(words[:2])
                if words[0] in ("Pair", "Bond", "Angle", "Dihedral")
                and len(words) > 1
                and words[1] == "Coeffs"
                else words[0]
            )
            sections[current] = []
            continue
        if current is None:
            header.append(words)
        else:
            sections[current].append([float(x) for x in words])
    return header, sections


def _lammps_bonded_energy(text):
    """Bond/angle/dihedral energy (kcal/mol) with LAMMPS harmonic/fourier formulas."""
    import numpy as np

    _header, s = _parse_lammps(text)
    pos = {int(r[0]): np.array(r[4:7]) for r in s["Atoms"]}
    bc = {int(r[0]): r[1:] for r in s.get("Bond Coeffs", [])}
    ac = {int(r[0]): r[1:] for r in s.get("Angle Coeffs", [])}
    dc = {int(r[0]): r[1:] for r in s.get("Dihedral Coeffs", [])}
    e_b = e_a = e_d = 0.0
    for r in s.get("Bonds", []):
        k, r0 = bc[int(r[1])]
        d = np.linalg.norm(pos[int(r[2])] - pos[int(r[3])])
        e_b += k * (d - r0) ** 2
    for r in s.get("Angles", []):
        k, t0 = ac[int(r[1])]
        u = pos[int(r[2])] - pos[int(r[3])]
        v = pos[int(r[4])] - pos[int(r[3])]
        theta = math.acos(np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v))
        e_a += k * (theta - math.radians(t0)) ** 2
    import mdff_energy

    for r in s.get("Dihedrals", []):
        coeffs = dc[int(r[1])]
        m = int(coeffs[0])
        phi = mdff_energy._dihedral(*(pos[int(x)] for x in r[2:6]))
        for n in range(m):
            k, mult, d = coeffs[1 + 3 * n : 4 + 3 * n]
            e_d += k * (1 + math.cos(mult * phi - math.radians(d)))
    return e_b, e_a, e_d


def test_lammps_data_energy_matches_mdff(system):
    import mdff_energy
    import numpy as np

    mol, data = system
    positions = molecule_positions(mol)
    text = writers.lammps_data(data, positions)
    e_b, e_a, e_d = _lammps_bonded_energy(text)
    ref = mdff_energy.energy(data, np.array(positions) / 10.0)
    assert e_b * 4.184 == pytest.approx(ref["bond"], rel=1e-5, abs=1e-5)
    assert e_a * 4.184 == pytest.approx(ref["angle"], rel=1e-5, abs=1e-5)
    assert e_d * 4.184 == pytest.approx(ref["torsion"], rel=1e-5, abs=1e-5)
    header, s = _parse_lammps(text)
    assert len(s["Atoms"]) == len(positions)
    assert sum(r[3] for r in s["Atoms"]) == pytest.approx(0.0, abs=1e-6)
    # molecule ids: aspirin, 2 waters, Na, Cl
    assert sorted({int(r[1]) for r in s["Atoms"]}) == [1, 2, 3, 4, 5]
    assert any(w[-2:] == ["xlo", "xhi"] for w in header)


def test_lammps_triclinic_cell_keeps_fractional_coordinates():
    import numpy as np

    lattice = [[10.0, 0.0, 0.0], [3.0, 9.0, 0.0], [1.0, 2.0, 8.0]]
    rot = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )  # 90 deg about z
    lat_rot = (np.array(lattice) @ rot.T).tolist()
    cell = {"lattice": lat_rot, "origin": [1.0, 2.0, 3.0], "pbc": [True, True, True]}
    frac = np.array([[0.1, 0.2, 0.3], [0.5, 0.5, 0.5]])
    pos = (frac @ np.array(lat_rot) + np.array(cell["origin"])).tolist()
    header, out = writers.lammps_box(pos, cell)
    assert header[3].endswith("xy xz yz")
    xy, xz, yz = (float(x) for x in header[3].split()[:3])
    lx, ly, lz = (float(h.split()[1]) for h in header[:3])
    new = np.array([[lx, 0, 0], [xy, ly, 0], [xz, yz, lz]])
    assert np.allclose(np.array(out), frac @ new)
    # lengths and angles are preserved
    for a, b in zip(new, np.array(lattice)):
        assert np.linalg.norm(a) == pytest.approx(np.linalg.norm(b))
    assert np.dot(new[0], new[1]) == pytest.approx(np.dot(lattice[0], lattice[1]))


def test_lammps_bounding_box_padding():
    header, out = writers.lammps_box(
        [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], None, padding=5.0
    )
    assert header[0] == "-5 6 xlo xhi"
    assert len(header) == 3
    assert out[1] == [1.0, 2.0, 3.0]


def test_lammps_rejects_mismatched_positions(system):
    _mol, data = system
    with pytest.raises(ValueError):
        writers.lammps_data(data, [[0.0, 0.0, 0.0]])
