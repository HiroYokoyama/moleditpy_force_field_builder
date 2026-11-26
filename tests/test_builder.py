"""Molecule types, system entries, charges and the mdff document."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers import HAS_RDKIT, combine, mol3d, shifted, water_box

pytestmark = pytest.mark.skipif(not HAS_RDKIT, reason="RDKit not installed")

from force_field_builder import builder, mdff, topology, water


def _mixed():
    return combine(
        mol3d("CCO"),
        shifted(water_box(3), 6),
        shifted(mol3d("[Na+]"), 20),
        shifted(mol3d("[Cl-]"), 24),
        shifted(mol3d("CCO", seed=3), 30),
    )


def test_mixed_system_types_and_counts():
    res = builder.build(_mixed(), {"charge_method": "gasteiger"})
    assert mdff.validate(res.mdff) == []
    assert [(e["type"], e["count"], e["atom_offset"]) for e in res.mdff["system"]] == [
        ("MOL", 1, 0),
        ("SOL", 3, 9),
        ("NA", 1, 18),
        ("CL", 1, 19),
        ("MOL", 1, 20),
    ]
    assert [t.name for t in res.types] == ["MOL", "SOL", "NA", "CL"]
    assert res.type_info("MOL").count == 2
    assert res.type_info("SOL").count == 3
    assert mdff.total_atoms(res.mdff) == 29
    assert len(res.atom_types) == len(res.atom_charges) == 29
    assert res.atom_types[9] == "OW_TIP3P"


def test_identical_molecules_parameterised_once(monkeypatch):
    calls = []
    real = builder._sage_type

    def spy(*args, **kwargs):
        calls.append(args[1])
        return real(*args, **kwargs)

    monkeypatch.setattr(builder, "_sage_type", spy)
    mols = [shifted(mol3d("CCO", seed=s), 6 * s) for s in range(5)]
    res = builder.build(combine(*mols), {"charge_method": "mmff94"})
    assert calls == ["MOL"]
    assert res.mdff["system"] == [{"type": "MOL", "count": 5, "atom_offset": 0}]


def test_same_molecule_other_atom_order_is_remapped():
    a = mol3d("CCO")
    b = shifted(mol3d("OCC"), 8)
    res = builder.build(combine(a, b), {"charge_method": "mmff94"})
    t1, t2 = res.mdff["molecule_types"]
    assert t2["name"] == "MOL2"
    assert "reused" in res.type_info("MOL2").warnings[0]
    # Oxygen of MOL2 is atom 0; it must carry MOL's oxygen parameters.
    assert t2["atoms"][0]["element"] == "O"
    assert t2["atoms"][0]["type"] == t1["atoms"][2]["type"]
    assert t2["atoms"][0]["charge"] == pytest.approx(t1["atoms"][2]["charge"])
    bonded = {(b["i"], b["j"]) for b in t2["bonds"]}
    for b in t2["bonds"]:
        assert b["i"] != b["j"]
    # every MOL2 bond exists in the host graph of the second molecule
    host_b = {
        tuple(sorted((x.GetBeginAtomIdx(), x.GetEndAtomIdx())))
        for x in mol3d("OCC").GetBonds()
    }
    assert {tuple(sorted(p)) for p in bonded} == host_b


def test_interleaved_fragments_merge_into_one_block():
    from rdkit import Chem

    # Atom order: C(frag A) O(frag B) C(frag A) -> one block of three atoms.
    rw = Chem.RWMol()
    for z in (6, 8, 6):
        rw.AddAtom(Chem.Atom(z))
    rw.AddBond(0, 2, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    assert topology.fragment_blocks(mol) == [(0, 3)]
    mol = Chem.AddHs(mol)
    blocks = topology.fragment_blocks(mol)
    covered = sum(e - s for s, e in blocks)
    assert covered == mol.GetNumAtoms()
    for s, e in blocks:
        frags = Chem.GetMolFrags(mol)
        for f in frags:
            inside = [s <= i < e for i in f]
            assert all(inside) or not any(inside)


def test_charges_sum_to_formal_charge():
    res = builder.build(
        combine(mol3d("CC(=O)[O-]"), shifted(mol3d("C[NH3+]"), 8)),
        {"charge_method": "gasteiger"},
    )
    nets = [
        round(sum(a["charge"] for a in mt["atoms"]), 8)
        for mt in res.mdff["molecule_types"]
    ]
    assert nets == [-1.0, 1.0]


def test_charge_scheme_is_flagged():
    res = builder.build(mol3d("CCO"), {"charge_method": "mmff94"})
    info = res.types[0]
    assert info.charges_from_force_field is False
    assert any("not the force field" in w for w in info.warnings)
    assert res.mdff["molecule_types"][0]["charges_from_force_field"] is False
    assert res.warnings


def test_imported_charges_are_used():
    mol = mol3d("CO")
    values = [0.1 * (i - 2) for i in range(mol.GetNumAtoms())]
    res = builder.build(
        mol,
        {"charge_method": "imported", "average_equivalent": False},
        imported_charges=values,
    )
    got = [a["charge"] for a in res.mdff["molecule_types"][0]["atoms"]]
    shift = -sum(values) / len(values)
    assert got == pytest.approx([v + shift for v in values])
    with pytest.raises(builder.BuildError):
        builder.build(mol, {"charge_method": "imported"}, imported_charges=[0.0])


def test_water_models_and_settles():
    for name, model in water.WATER_MODELS.items():
        res = builder.build(water_box(2), {"water_model": name})
        mt = res.mdff["molecule_types"][0]
        assert mt["settles"]["i"] == 0
        assert mt["settles"]["dhh"] == pytest.approx(water.hh_distance(model), abs=1e-6)
        assert sum(a["charge"] for a in mt["atoms"]) == pytest.approx(0.0, abs=1e-9)
        assert res.mdff["system"] == [{"type": "SOL", "count": 2, "atom_offset": 0}]
    tip3p = water.WATER_MODELS["TIP3P"]
    assert water.hh_distance(tip3p) == pytest.approx(0.15139, abs=1e-5)


def test_water_with_hydrogen_first():
    from rdkit import Chem

    rw = Chem.RWMol()
    for z in (1, 8, 1):
        rw.AddAtom(Chem.Atom(z))
    rw.AddBond(0, 1, Chem.BondType.SINGLE)
    rw.AddBond(1, 2, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    mt = builder.build(mol).mdff["molecule_types"][0]
    assert [a["element"] for a in mt["atoms"]] == ["H", "O", "H"]
    assert mt["settles"]["i"] == 1


def test_h_constraints_toggle():
    on = builder.build(
        mol3d("CCO"), {"charge_method": "mmff94", "constrain_h": True}
    ).mdff
    off = builder.build(
        mol3d("CCO"), {"charge_method": "mmff94", "constrain_h": False}
    ).mdff
    assert len(on["molecule_types"][0]["constraints"]) == 6
    assert "constraints" not in off["molecule_types"][0]
    assert len(on["molecule_types"][0]["bonds"]) == 8  # bonds stay complete


def test_build_errors():
    from rdkit import Chem

    with pytest.raises(builder.BuildError):
        builder.build(None)
    with pytest.raises(builder.BuildError, match="implicit hydrogens"):
        builder.build(Chem.MolFromSmiles("CCO"))
    with pytest.raises(builder.BuildError):
        builder.build(mol3d("C"), {"water_model": "TIP5P"})


def test_residue_names_become_type_names():
    from rdkit import Chem

    mol = mol3d("CCO")
    for atom in mol.GetAtoms():
        info = Chem.AtomPDBResidueInfo()
        info.SetResidueName("ETH")
        info.SetName(f" {atom.GetSymbol()}{atom.GetIdx()}")
        atom.SetMonomerInfo(info)
    res = builder.build(mol, {"charge_method": "mmff94"})
    mt = res.mdff["molecule_types"][0]
    assert mt["name"] == "ETH"
    assert mt["atoms"][0]["name"] == "C0"


def test_document_is_json_serialisable_and_round_trips():
    res = builder.build(_mixed(), {"charge_method": "gasteiger"})
    text = mdff.dumps(res.mdff)
    again = mdff.loads(text)
    assert again == json.loads(text)
    assert again["format"] == "moleditpy-mdff" and again["version"] == 1


def test_validate_reports_problems():
    res = builder.build(mol3d("CO"), {"charge_method": "gasteiger"})
    bad = json.loads(mdff.dumps(res.mdff))
    bad["molecule_types"][0]["bonds"][0]["i"] = 99
    bad["system"][0]["atom_offset"] = 3
    bad["format"] = "x"
    errors = mdff.validate(bad)
    assert any("out of range" in e for e in errors)
    assert any("atom_offset" in e for e in errors)
    assert any("format" in e for e in errors)
    with pytest.raises(ValueError):
        mdff.loads(json.dumps(bad))


def test_shipped_examples_are_valid():
    folder = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples"
    )
    names = [n for n in os.listdir(folder) if n.endswith(".mdff.json")]
    assert len(names) >= 3
    for name in names:
        with open(os.path.join(folder, name), encoding="utf-8") as handle:
            assert mdff.validate(json.load(handle)) == [], name


def test_molecule_order_and_reorder():
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles("CCO.O.[Na+]"))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    mol.SetProp("moleditpy_cell", '{"lattice": [[10,0,0],[0,10,0],[0,0,10]]}')
    for atom in mol.GetAtoms():
        atom.SetIntProp("_original_atom_id", 100 + atom.GetIdx())
    assert topology.fragment_blocks(mol) == [(0, mol.GetNumAtoms())]
    order = topology.molecule_order(mol)
    out = topology.reorder_by_molecule(mol)
    assert [a.GetIntProp("_original_atom_id") for a in out.GetAtoms()] == [
        100 + i for i in order
    ]
    assert out.HasProp("moleditpy_cell")
    assert topology.fragment_blocks(out) == [(0, 9), (9, 12), (12, 13)]
    assert topology.molecule_order(out) is None
    pos_old = mol.GetConformer().GetPositions()
    pos_new = out.GetConformer().GetPositions()
    assert (pos_new == pos_old[order]).all()


def test_imported_charges_are_averaged_over_equivalent_atoms():
    mol = mol3d("CO")  # C O H H H H: the three methyl hydrogens are equivalent
    values = [0.1, -0.6, 0.0, 0.05, 0.1, 0.35]
    res = builder.build(mol, {"charge_method": "imported"}, imported_charges=values)
    got = [a["charge"] for a in res.mdff["molecule_types"][0]["atoms"]]
    assert got[2] == pytest.approx(got[3]) == pytest.approx(got[4])
    assert got[2] == pytest.approx(0.05)
    assert got[5] == pytest.approx(0.35)
    assert sum(got) == pytest.approx(0.0)
    info = res.types[0]
    assert "averaged" in info.charge_method
    assert any("2 symmetry-equivalent" in w for w in info.warnings)


def test_charges_of_one_molecule_apply_to_every_copy():
    system = _mixed()  # ethanol, 3 waters, Na+, Cl-, ethanol in another conformer
    ethanol = mol3d("CCO")
    values = [0.1 * (i % 3) - 0.1 for i in range(ethanol.GetNumAtoms())]
    elements = [a.GetSymbol() for a in ethanol.GetAtoms()]
    assert builder.imported_matches(system, values, elements) == ("molecule", 2)
    res = builder.build(
        system,
        {"charge_method": "imported", "average_equivalent": False},
        imported_charges=values,
        imported_elements=elements,
    )
    assert mdff.validate(res.mdff) == []
    mol_type = res.mdff["molecule_types"][0]
    shift = -sum(values) / len(values)
    assert [a["charge"] for a in mol_type["atoms"]] == pytest.approx(
        [v + shift for v in values]
    )
    assert res.type_info("MOL").count == 2
    assert res.type_info("MOL").charge_method.startswith("Imported")
    # Ions keep their library charges, water its model.
    na = res.mdff["molecule_types"][[t.name for t in res.types].index("NA")]
    assert na["atoms"][0]["charge"] == pytest.approx(1.0)
    n = ethanol.GetNumAtoms()
    assert res.atom_charges[20 : 20 + n] == pytest.approx(res.atom_charges[:n])


def test_other_molecules_fall_back_to_auto():
    system = combine(mol3d("CCO"), shifted(mol3d("CC(=O)O"), 8))
    acid = mol3d("CC(=O)O")
    values = [0.0] * acid.GetNumAtoms()
    values[0] = 0.2
    values[1] = -0.2
    res = builder.build(
        system,
        {"charge_method": "imported"},
        imported_charges=values,
        imported_elements=[a.GetSymbol() for a in acid.GetAtoms()],
    )
    first, second = res.types
    assert first.charge_method != second.charge_method
    assert any("another molecule" in w for w in first.warnings)
    assert "Imported" in second.charge_method


def test_charges_that_fit_nothing_are_rejected():
    system = _mixed()
    assert builder.imported_matches(system, [0.0] * 5) == (None, 0)
    assert builder.imported_matches(system, [0.0] * 3) == (None, 0)  # water only
    with pytest.raises(builder.BuildError, match="fit neither"):
        builder.build(system, {"charge_method": "imported"}, imported_charges=[0.0] * 5)
    ethanol = mol3d("CCO")
    wrong_order = [a.GetSymbol() for a in ethanol.GetAtoms()][::-1]
    assert builder.imported_matches(
        system, [0.0] * ethanol.GetNumAtoms(), wrong_order
    ) == (None, 0)
