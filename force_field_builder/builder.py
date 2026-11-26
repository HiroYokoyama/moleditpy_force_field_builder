"""Turn the host molecule into a ``moleditpy-mdff`` description.

``build(mol, options)`` splits the molecule into molecule types (identical
molecules parameterised once), assigns parameters from the chosen source,
assigns charges, and returns a ``BuildResult`` with the mdff dictionary plus
per-atom display data and warnings.
"""

from __future__ import annotations

from . import charges as charge_mod
from . import external, smirnoff, topology, water

FORMAT_NAME = "moleditpy-mdff"
FORMAT_VERSION = 1

FORCE_FIELDS = {
    "sage": "OpenFF Sage 2.3.0 (built-in SMIRNOFF engine)",
    "gaff2": "GAFF2 via acpype/antechamber",
    "gaff": "GAFF via acpype/antechamber",
    "omff-gaff": "GAFF 2.11 via openmmforcefields",
}


def available_force_fields():
    keys = ["sage"]
    if external.has_acpype():
        keys += ["gaff2", "gaff"]
    if external.has_openmmforcefields() and external.has_openff_toolkit():
        keys.append("omff-gaff")
    return keys


def default_options():
    return {
        "force_field": "sage",
        "charge_method": "auto",
        "water_model": "TIP3P",
        "constrain_h": True,
        "cross_check": True,
        "average_equivalent": True,
    }


class BuildError(RuntimeError):
    """The molecule cannot be parameterised as requested."""


class TypeInfo:
    """Display data for one molecule type."""

    def __init__(self, name, kind):
        self.name = name
        self.kind = kind
        self.count = 0
        self.smiles = ""
        self.warnings = []
        self.charge_method = ""
        self.charges_from_force_field = False


class BuildResult:
    def __init__(self):
        self.mdff = None
        self.types = []  # TypeInfo in mdff order
        self.atom_types = []  # per host atom
        self.atom_charges = []  # per host atom
        self.warnings = []  # global

    def type_info(self, name):
        return next((t for t in self.types if t.name == name), None)


_sage_cache = {}


def sage_force_field():
    if "sage" not in _sage_cache:
        _sage_cache["sage"] = smirnoff.SmirnoffForceField(smirnoff.SAGE_PATH)
    return _sage_cache["sage"]


def _unique_name(base, used):
    base = "".join(ch for ch in base.upper() if ch.isalnum() or ch == "_") or "MOL"
    if base not in used:
        used.add(base)
        return base
    n = 2
    while f"{base}{n}" in used:
        n += 1
    used.add(f"{base}{n}")
    return f"{base}{n}"


def _charges_for(block_mol, method, ff_scheme_available, imported, average=False):
    """(charges, label, from_force_field, warnings) for a non-library molecule."""
    warnings = []
    if method == "auto":
        attempts = []
        if charge_mod.has_module("openff.nagl") and external.has_openff_toolkit():
            attempts.append("nagl")
        if charge_mod.has_executable("antechamber") and charge_mod.has_executable(
            "sqm"
        ):
            attempts.append("am1bcc")
        attempts += ["mmff94", "gasteiger"]
        for key in attempts:
            try:
                values = charge_mod.compute(key, block_mol)
            except Exception as exc:  # noqa: BLE001 - each fallback is tried in turn
                warnings.append(f"{charge_mod.METHOD_LABELS[key]} failed: {exc}")
                continue
            from_ff = ff_scheme_available and key in ("nagl", "am1bcc")
            if not from_ff:
                warnings.append(
                    f"Charges are {charge_mod.METHOD_LABELS[key]}, not the force field's own "
                    "AM1-BCC scheme (install AmberTools or openff-nagl for that)."
                )
            return values, charge_mod.METHOD_LABELS[key], from_ff, warnings
        raise BuildError("No charge method worked for this molecule")
    values = charge_mod.compute(method, block_mol, imported=imported)
    label = charge_mod.METHOD_LABELS[method]
    from_ff = ff_scheme_available and method in ("nagl", "am1bcc")
    if not from_ff:
        warnings.append(f"Charges are {label}, not the force field's own scheme.")
    if average and method in charge_mod.AVERAGED_METHODS:
        values, changed = charge_mod.average_equivalent(block_mol, values)
        label += ", equivalent atoms averaged"
        if changed:
            warnings.append(
                f"Charges of {changed} symmetry-equivalent atoms were averaged."
            )
    return values, label, from_ff, warnings


def _register_atom_type(registry, name, atom):
    """Keep atom-type names unique per (mass, sigma, epsilon); returns the final name."""
    key = (round(atom["mass"], 4), round(atom["sigma"], 7), round(atom["epsilon"], 7))
    candidate = name
    n = 2
    while candidate in registry and registry[candidate] != key:
        candidate = f"{name}_{n}"
        n += 1
    registry[candidate] = key
    return candidate


def _sage_type(block_mol, name, atom_names, options, imported, info):
    ff = sage_force_field()
    assignment = smirnoff.assign_parameters(ff, block_mol)
    info.warnings.extend(assignment.missing)
    if assignment.fallbacks:
        info.warnings.append(
            "Filled from UFF (check these): " + "; ".join(assignment.fallbacks)
        )

    if (
        assignment.library_charges is not None
        and options["charge_method"] != "imported"
    ):
        values = assignment.library_charges
        info.charge_method = "Sage library charges"
        info.charges_from_force_field = True
    else:
        values, label, from_ff, warns = _charges_for(
            block_mol,
            options["charge_method"],
            True,
            imported,
            options.get("average_equivalent", True),
        )
        info.charge_method = label
        info.charges_from_force_field = from_ff
        info.warnings.extend(warns)

    atoms = []
    for rd_atom in block_mol.GetAtoms():
        idx = rd_atom.GetIdx()
        vdw = assignment.vdw[idx]
        symbol = rd_atom.GetSymbol()
        atoms.append(
            {
                "name": atom_names[idx],
                "element": symbol,
                "type": f"{symbol}_{vdw['id'] or 'X'}",
                "charge": float(values[idx]),
                "mass": float(rd_atom.GetMass()),
                "sigma": float(vdw["sigma"]),
                "epsilon": float(vdw["epsilon"]),
            }
        )
    mtype = {
        "name": name,
        "atoms": atoms,
        "bonds": [
            {k: b[k] for k in ("i", "j", "r0", "k", "id")} for b in assignment.bonds
        ],
        "angles": [
            {k: a[k] for k in ("i", "j", "k_atom", "theta0", "k", "id")}
            for a in assignment.angles
        ],
        "torsions": assignment.torsions,
        "exclusions_nrexcl": 3,
        "force_field": FORCE_FIELDS["sage"],
    }
    if options.get("cross_check") and external.has_openff_toolkit():
        try:
            theirs = external.toolkit_labels(block_mol, smirnoff.SAGE_PATH)
            diffs = external.compare_labels(external.ours_as_labels(assignment), theirs)
            if diffs:
                info.warnings.append(
                    f"openff-toolkit cross-check: {len(diffs)} differences, e.g. "
                    + "; ".join(diffs[:3])
                )
            else:
                info.warnings.append(
                    "openff-toolkit cross-check: identical parameter assignment."
                )
        except Exception as exc:  # noqa: BLE001
            info.warnings.append(f"openff-toolkit cross-check failed: {exc}")
    return mtype


def _external_type(block_mol, name, atom_names, options, imported, info):
    ff_key = options["force_field"]
    method = options["charge_method"]
    if ff_key in ("gaff", "gaff2"):
        user = None
        if method != "auto":
            values, label, _from_ff, warns = _charges_for(
                block_mol,
                method,
                False,
                imported,
                options.get("average_equivalent", True),
            )
            user = values
            info.charge_method = label
            info.warnings.extend(warns)
        else:
            info.charge_method = "AM1-BCC (acpype/sqm)"
            info.charges_from_force_field = True
        mtype, warns = external.run_acpype(
            block_mol, atom_type=ff_key, user_charges=user
        )
        info.warnings.extend(warns)
    elif ff_key == "omff-gaff":
        mtype, warns = external.run_openmmforcefields_gaff(block_mol, name)
        info.warnings.extend(warns)
        info.charge_method = "AM1-BCC (openmmforcefields)"
        info.charges_from_force_field = True
        if method != "auto":
            values, label, _from_ff, warns = _charges_for(
                block_mol,
                method,
                False,
                imported,
                options.get("average_equivalent", True),
            )
            for atom, q in zip(mtype["atoms"], values):
                atom["charge"] = float(q)
            info.charge_method = label
            info.charges_from_force_field = False
            info.warnings.extend(warns)
    else:
        raise BuildError(f"Unknown force field {ff_key!r}")
    mtype["name"] = name
    for atom, atom_name in zip(mtype["atoms"], atom_names):
        atom["name"] = atom_name
    mtype["force_field"] = FORCE_FIELDS[ff_key]
    return mtype


def _ion_or_molecule_type(block_mol, name, atom_names, options, imported, info):
    if options["force_field"] == "sage":
        return _sage_type(block_mol, name, atom_names, options, imported, info)
    if block_mol.GetNumAtoms() == 1:
        # Monatomic ions always come from the Sage (Joung-Cheatham) table.
        return _sage_type(
            block_mol,
            name,
            atom_names,
            dict(options, cross_check=False),
            imported,
            info,
        )
    return _external_type(block_mol, name, atom_names, options, imported, info)


def _imported_fits(block_mol, charges, elements):
    if block_mol is None or block_mol.GetNumAtoms() != len(charges):
        return False
    if not elements:
        return True
    mine = [a.GetSymbol().upper() for a in block_mol.GetAtoms()]
    return mine == [str(e).upper() for e in elements]


def imported_matches(mol, charges, elements=None):
    """How imported charges apply to *mol*.

    ("system", n_atoms) when there is one charge per atom; ("molecule", n_copies)
    when they fit single molecules (same atom count and element order, water
    excluded); (None, 0) otherwise.
    """
    from rdkit import Chem

    if mol is None or not charges:
        return None, 0
    if len(charges) == mol.GetNumAtoms():
        return "system", mol.GetNumAtoms()
    copies = 0
    for start, end in topology.fragment_blocks(mol):
        if end - start != len(charges):
            continue
        if (
            topology.classify_block(topology.block_signature(mol, start, end))
            == "water"
        ):
            continue
        block = topology.extract_block(Chem.Mol(mol), start, end)
        if _imported_fits(block, charges, elements):
            copies += 1
    return ("molecule", copies) if copies else (None, 0)


def _h_constraints(mtype):
    hydrogens = {n for n, a in enumerate(mtype["atoms"]) if a["element"] == "H"}
    return [
        {"i": b["i"], "j": b["j"], "r0": b["r0"]}
        for b in mtype["bonds"]
        if (b["i"] in hydrogens or b["j"] in hydrogens) and b.get("r0")
    ]


def build(
    mol, options=None, imported_charges=None, version="0.1.0", imported_elements=None
):
    """Parameterise the host molecule. Raises BuildError for unusable input.

    Imported charges cover either every atom of the system or one molecule. One
    molecule's charges go to every molecule with the same atom count and element
    order; the other molecules use the automatic charge method.
    """
    from rdkit import Chem

    opts = default_options()
    opts.update(options or {})
    if mol is None or mol.GetNumAtoms() == 0:
        raise BuildError("There is no molecule to parameterise.")
    per_molecule = False
    if opts["charge_method"] == "imported":
        if not imported_charges:
            raise BuildError("Import a charge file first.")
        per_molecule = len(imported_charges) != mol.GetNumAtoms()
    matched = 0
    if opts["water_model"] not in water.WATER_MODELS:
        raise BuildError(f"Unknown water model {opts['water_model']!r}")

    host = Chem.Mol(mol)
    host.UpdatePropertyCache(strict=False)
    if topology.implicit_hydrogen_count(host):
        raise BuildError(
            "The molecule has implicit hydrogens. Convert it to 3D first so that "
            "every hydrogen is an explicit atom."
        )

    result = BuildResult()
    used_names = set()
    atom_type_registry = {}
    by_signature = {}  # signature -> type name
    by_smiles = {}  # canonical SMILES -> (type name, block mol)
    types = {}  # name -> mdff type
    block_types = []

    merged = 0
    for start, end in topology.fragment_blocks(host):
        if (
            len(
                Chem.GetMolFrags(
                    topology.extract_block(host, start, end), sanitizeFrags=False
                )
            )
            > 1
        ):
            merged += 1
        sig = topology.block_signature(host, start, end)
        if sig in by_signature:
            name = by_signature[sig]
            types[name]["_count"] += 1
            block_types.append((name, start, end))
            continue

        kind = topology.classify_block(sig)
        block_mol = None
        if kind == "water":
            o_index = next(i for i, (z, _) in enumerate(sig[0]) if z == 8)
            name = _unique_name("SOL", used_names)
            mtype = water.water_molecule_type(opts["water_model"], name, o_index)
            info = TypeInfo(name, kind)
            info.smiles = "O"
            info.charge_method = mtype["charge_method"]
            info.charges_from_force_field = True
            if opts["force_field"] == "sage" and opts["water_model"] != "TIP3P":
                info.warnings.append(
                    "Sage was fitted with TIP3P water; another water model is your choice."
                )
        else:
            block_mol = topology.extract_block(host, start, end)
            smiles = (
                Chem.MolToSmiles(Chem.RemoveHs(block_mol, sanitize=False))
                if block_mol.GetNumAtoms()
                else ""
            )
            reuse = by_smiles.get(smiles)
            perm = topology.graph_permutation(reuse[1], block_mol) if reuse else None
            if reuse and perm is not None:
                base = types[reuse[0]]
                name = _unique_name(reuse[0], used_names)
                mtype = topology.remap_molecule_type(
                    {k: v for k, v in base.items() if k != "_count"}, perm, name
                )
                old = result.type_info(reuse[0])
                info = TypeInfo(name, kind)
                info.smiles = smiles
                info.charge_method = old.charge_method
                info.charges_from_force_field = old.charges_from_force_field
                info.warnings = [
                    f"Same molecule as {reuse[0]} in another atom order; parameters reused."
                ]
            else:
                if kind == "ion":
                    base_name = block_mol.GetAtomWithIdx(0).GetSymbol()
                else:
                    base_name = topology.residue_name(host, start) or "MOL"
                    if base_name in ("HOH", "WAT", "SOL") or len(base_name) > 8:
                        base_name = "MOL"
                name = _unique_name(base_name, used_names)
                info = TypeInfo(name, kind)
                info.smiles = smiles
                block_opts = opts
                imported_slice = None
                if opts["charge_method"] == "imported":
                    if not per_molecule:
                        imported_slice = imported_charges[start:end]
                    elif _imported_fits(block_mol, imported_charges, imported_elements):
                        imported_slice = list(imported_charges)
                        matched += 1
                    else:
                        block_opts = dict(opts, charge_method="auto")
                        if kind != "ion":
                            info.warnings.append(
                                "The imported charges are for another molecule; "
                                "this one uses the automatic charge method."
                            )
                atom_names = topology.atom_names(host, start, end)
                try:
                    mtype = _ion_or_molecule_type(
                        block_mol, name, atom_names, block_opts, imported_slice, info
                    )
                except (
                    smirnoff.SmirnoffError,
                    charge_mod.ChargeError,
                    external.ExternalError,
                ) as exc:
                    raise BuildError(
                        f"{name} (atoms {start + 1}-{end}): {exc}"
                    ) from exc
                by_smiles.setdefault(smiles, (name, block_mol))

        if kind != "water":
            target = charge_mod.formal_charge(block_mol)
            values, residual = charge_mod.normalise(
                [a["charge"] for a in mtype["atoms"]], target
            )
            for atom, q in zip(mtype["atoms"], values):
                atom["charge"] = q
            if abs(residual) > 0.01:
                info.warnings.append(
                    f"Charges summed to {target - residual:+.4f}; shifted evenly to the formal charge {target:+d}."
                )
        if opts["constrain_h"] and kind != "water":
            cons = _h_constraints(mtype)
            if cons:
                mtype["constraints"] = cons
        mtype["charge_method"] = info.charge_method
        mtype["charges_from_force_field"] = info.charges_from_force_field
        if info.smiles:
            mtype["smiles"] = info.smiles
        for atom in mtype["atoms"]:
            atom["type"] = _register_atom_type(atom_type_registry, atom["type"], atom)
        mtype["_count"] = 1
        types[name] = mtype
        by_signature[sig] = name
        result.types.append(info)
        block_types.append((name, start, end))

    if per_molecule and not matched:
        raise BuildError(
            f"The {len(imported_charges)} imported charges fit neither the whole system "
            f"({host.GetNumAtoms()} atoms) nor any molecule in it (same atom count and "
            "element order)."
        )

    # Per-atom display data and counts.
    result.atom_types = [""] * host.GetNumAtoms()
    result.atom_charges = [0.0] * host.GetNumAtoms()
    for name, start, end in block_types:
        mtype = types[name]
        for n, atom in enumerate(mtype["atoms"]):
            result.atom_types[start + n] = atom["type"]
            result.atom_charges[start + n] = atom["charge"]
    for info in result.types:
        info.count = types[info.name].pop("_count")

    ff = sage_force_field()
    molecule_types = [types[info.name] for info in result.types]
    result.mdff = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "source": f"MoleditPy Force Field Builder {version}: {FORCE_FIELDS[opts['force_field']]}",
        "combination_rule": 2,
        "fudge_lj": ff.scale14_lj,
        "fudge_qq": round(ff.scale14_qq, 6),
        "molecule_types": molecule_types,
        "system": topology.system_entries(block_types),
    }
    if merged:
        result.warnings.append(
            f"{merged} molecule type(s) hold several molecules because their atoms are "
            "interleaved in the editor's atom order; reorder the atoms molecule by "
            "molecule to get one type per molecule."
        )
    if not all(info.charges_from_force_field for info in result.types):
        result.warnings.append(
            "Some molecules do not use the force field's own charge scheme (see the list)."
        )
    return result
