"""SMIRNOFF (.offxml) reader and RDKit-based parameter assignment.

Implements the parts of the SMIRNOFF specification that OpenFF Sage uses:
harmonic bonds and angles, periodic proper and improper torsions, Lennard-Jones
vdW (``sigma`` or ``rmin_half``), library charges and constraints.  Every
handler applies its parameters in file order and the *last* matching
parameter wins for each bond / angle / torsion / atom, as the specification
requires.  SMIRKS are matched with RDKit on a copy of the molecule perceived
with the MDL aromaticity model, which is what the OpenFF toolkit's RDKit
backend does for ``aromaticity_model="OEAroModel_MDL"``.

All returned values are in the GROMACS unit system used by the
``moleditpy-mdff`` hand-off format: nm, kJ/mol, degrees, elementary charge.
"""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SAGE_FILENAME = "openff-2.3.0.offxml"
SAGE_PATH = os.path.join(DATA_DIR, SAGE_FILENAME)

# Factors that convert each unit to nm / kJ / mol / degree / e.  ``mole`` is 1
# because every energy is per mole; ``radian`` is 1 because force constants
# stay per radian^2 (GROMACS convention), while angles themselves are given
# in degrees in every Sage file.
_UNIT_FACTORS = {
    "angstrom": 0.1,
    "angstroms": 0.1,
    "nanometer": 1.0,
    "nanometers": 1.0,
    "kilocalorie": 4.184,
    "kilocalories": 4.184,
    "kilocalorie_per_mole": 4.184,
    "kilojoule": 1.0,
    "kilojoules": 1.0,
    "kilojoule_per_mole": 1.0,
    "mole": 1.0,
    "radian": 1.0,
    "radians": 1.0,
    "degree": 1.0,
    "degrees": 1.0,
    "elementary_charge": 1.0,
    "dimensionless": 1.0,
}

_QTY_RE = re.compile(r"^\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*(.*)$")
_UNIT_RE = re.compile(r"\*\s*([A-Za-z_]+)(?:\s*\*\*\s*(-?\d+))?")

# 2^(1/6): sigma = 2 * rmin_half / 2^(1/6)
_TWO_SIXTH = 2.0 ** (1.0 / 6.0)


class SmirnoffError(ValueError):
    """Raised for a malformed or unsupported SMIRNOFF file."""


def parse_quantity(text: str) -> float:
    """Return the value of a SMIRNOFF quantity string in mdff units.

    ``"1.5 * angstrom ** 1"`` -> 0.15, ``"0.3 * mole ** -1 * kilocalorie ** 1"``
    -> 1.2552, ``"457.9 * kilocalorie_per_mole ** 1 * angstrom ** -2"`` ->
    457.9 * 418.4.
    """
    if text is None:
        raise SmirnoffError("missing quantity")
    match = _QTY_RE.match(str(text))
    if not match:
        raise SmirnoffError(f"cannot parse quantity {text!r}")
    value = float(match.group(1))
    rest = match.group(2)
    consumed = 0
    for unit_match in _UNIT_RE.finditer(rest):
        unit = unit_match.group(1)
        power = int(unit_match.group(2) or 1)
        if unit not in _UNIT_FACTORS:
            raise SmirnoffError(f"unsupported unit {unit!r} in {text!r}")
        value *= _UNIT_FACTORS[unit] ** power
        consumed += 1
    if rest.strip() and consumed == 0:
        raise SmirnoffError(f"cannot parse units in {text!r}")
    return value


class Parameter:
    """One SMIRKS-keyed parameter with its values already converted."""

    __slots__ = ("_map", "_query", "id", "smirks", "values")

    def __init__(self, smirks: str, pid: str, values: dict):
        self.smirks = smirks
        self.id = pid
        self.values = values
        self._query = None
        self._map = None

    def query(self):
        """The compiled RDKit query and the atom indices of the tagged atoms in tag order."""
        if self._query is None:
            from rdkit import Chem

            qmol = Chem.MolFromSmarts(self.smirks)
            if qmol is None:
                raise SmirnoffError(
                    f"RDKit cannot parse SMIRKS {self.smirks!r} ({self.id})"
                )
            tagged = {}
            for atom in qmol.GetAtoms():
                num = atom.GetAtomMapNum()
                if num:
                    tagged[num] = atom.GetIdx()
            self._map = [tagged[k] for k in sorted(tagged)]
            self._query = qmol
        return self._query, self._map

    def __repr__(self):  # pragma: no cover - debugging aid
        return f"Parameter({self.id!r}, {self.smirks!r})"


def _attr_float(elem, name, default=None):
    if name not in elem.attrib:
        return default
    return parse_quantity(elem.attrib[name])


def _torsion_terms(elem, default_idivf):
    """Collect periodicityN/phaseN/kN/idivfN into a list of term dicts."""
    terms = []
    n = 1
    while f"periodicity{n}" in elem.attrib:
        periodicity = int(float(elem.attrib[f"periodicity{n}"]))
        phase = parse_quantity(elem.attrib[f"phase{n}"])
        if f"k{n}" not in elem.attrib:
            raise SmirnoffError(
                f"torsion {elem.attrib.get('id')} uses k{n}_bondorder (fractional bond "
                "orders), which this builder does not support"
            )
        k = parse_quantity(elem.attrib[f"k{n}"])
        idivf = float(elem.attrib.get(f"idivf{n}", default_idivf))
        terms.append({"periodicity": periodicity, "phase": phase, "k": k / idivf})
        n += 1
    return terms


class SmirnoffForceField:
    """The subset of a SMIRNOFF force field that the builder applies."""

    def __init__(self, path: str | None = None, text: str | None = None):
        if text is None:
            path = path or SAGE_PATH
            with open(path, encoding="utf-8") as handle:
                text = handle.read()
        self.path = path
        self.name = os.path.basename(path) if path else "custom.offxml"
        try:
            root = ET.fromstring(text)
        except ET.ParseError as exc:
            raise SmirnoffError(f"not valid XML: {exc}") from exc
        if root.tag != "SMIRNOFF":
            raise SmirnoffError("root element is not <SMIRNOFF>")
        self.aromaticity_model = root.attrib.get("aromaticity_model", "OEAroModel_MDL")
        if self.aromaticity_model != "OEAroModel_MDL":
            raise SmirnoffError(
                f"unsupported aromaticity model {self.aromaticity_model}"
            )

        self.bonds: list[Parameter] = []
        self.angles: list[Parameter] = []
        self.propers: list[Parameter] = []
        self.impropers: list[Parameter] = []
        self.vdw: list[Parameter] = []
        self.library_charges: list[Parameter] = []
        self.constraints: list[Parameter] = []
        self.scale14_lj = 0.5
        self.scale14_qq = 0.8333333333
        self.combining_rules = "Lorentz-Berthelot"
        self.charge_model = "AM1-BCC"
        self.unsupported_sections: list[str] = []

        for section in root:
            tag = section.tag
            if tag in ("Author", "Date"):
                continue
            if tag == "Bonds":
                self._read_bonds(section)
            elif tag == "Angles":
                self._read_angles(section)
            elif tag == "ProperTorsions":
                self._read_torsions(section, self.propers, "Proper", "1.0")
            elif tag == "ImproperTorsions":
                self._read_torsions(section, self.impropers, "Improper", "3.0")
            elif tag == "vdW":
                self._read_vdw(section)
            elif tag == "Electrostatics":
                self.scale14_qq = float(section.attrib.get("scale14", self.scale14_qq))
            elif tag == "LibraryCharges":
                self._read_library_charges(section)
            elif tag == "Constraints":
                self._read_constraints(section)
            elif tag == "NAGLCharges":
                model = section.attrib.get("model_file", "NAGL")
                self.charge_model = f"AM1-BCC (NAGL graph model {model})"
            elif tag == "ToolkitAM1BCC":
                self.charge_model = "AM1-BCC"
            else:
                self.unsupported_sections.append(tag)

    # -- readers ------------------------------------------------------------

    def _read_bonds(self, section):
        potential = section.attrib.get("potential", "harmonic")
        if potential != "harmonic":
            raise SmirnoffError(f"unsupported bond potential {potential}")
        for elem in section.findall("Bond"):
            if "length" not in elem.attrib or "k" not in elem.attrib:
                raise SmirnoffError(
                    f"bond {elem.attrib.get('id')} has no length/k "
                    "(fractional bond orders are not supported)"
                )
            self.bonds.append(
                Parameter(
                    elem.attrib["smirks"],
                    elem.attrib.get("id", ""),
                    {"r0": _attr_float(elem, "length"), "k": _attr_float(elem, "k")},
                )
            )

    def _read_angles(self, section):
        potential = section.attrib.get("potential", "harmonic")
        if potential != "harmonic":
            raise SmirnoffError(f"unsupported angle potential {potential}")
        for elem in section.findall("Angle"):
            self.angles.append(
                Parameter(
                    elem.attrib["smirks"],
                    elem.attrib.get("id", ""),
                    {"theta0": _attr_float(elem, "angle"), "k": _attr_float(elem, "k")},
                )
            )

    def _read_torsions(self, section, target, child, auto_idivf):
        potential = section.attrib.get(
            "potential", "k*(1+cos(periodicity*theta-phase))"
        )
        if potential.replace(" ", "") != "k*(1+cos(periodicity*theta-phase))":
            raise SmirnoffError(f"unsupported torsion potential {potential}")
        default_idivf = section.attrib.get("default_idivf", "auto")
        if default_idivf == "auto":
            # The OpenFF toolkit treats "auto" as 1 for propers and 3 for
            # impropers (the k is shared over three trefoil permutations).
            default_idivf = auto_idivf
        for elem in section.findall(child):
            target.append(
                Parameter(
                    elem.attrib["smirks"],
                    elem.attrib.get("id", ""),
                    {"terms": _torsion_terms(elem, default_idivf)},
                )
            )

    def _read_vdw(self, section):
        potential = section.attrib.get("potential", "Lennard-Jones-12-6")
        if potential != "Lennard-Jones-12-6":
            raise SmirnoffError(f"unsupported vdW potential {potential}")
        self.scale14_lj = float(section.attrib.get("scale14", self.scale14_lj))
        self.combining_rules = section.attrib.get(
            "combining_rules", self.combining_rules
        )
        for elem in section.findall("Atom"):
            epsilon = _attr_float(elem, "epsilon")
            if "sigma" in elem.attrib:
                sigma = _attr_float(elem, "sigma")
            elif "rmin_half" in elem.attrib:
                sigma = 2.0 * _attr_float(elem, "rmin_half") / _TWO_SIXTH
            else:
                raise SmirnoffError(
                    f"vdW {elem.attrib.get('id')} has neither sigma nor rmin_half"
                )
            self.vdw.append(
                Parameter(
                    elem.attrib["smirks"],
                    elem.attrib.get("id", ""),
                    {"sigma": sigma, "epsilon": epsilon},
                )
            )

    def _read_library_charges(self, section):
        for elem in section.findall("LibraryCharge"):
            charges = []
            n = 1
            while f"charge{n}" in elem.attrib:
                charges.append(parse_quantity(elem.attrib[f"charge{n}"]))
                n += 1
            self.library_charges.append(
                Parameter(
                    elem.attrib["smirks"],
                    elem.attrib.get("id", ""),
                    {"charges": charges},
                )
            )

    def _read_constraints(self, section):
        for elem in section.findall("Constraint"):
            distance = _attr_float(elem, "distance")
            self.constraints.append(
                Parameter(
                    elem.attrib["smirks"],
                    elem.attrib.get("id", ""),
                    {"distance": distance},
                )
            )


# ---------------------------------------------------------------------------
# Molecule perception and matching
# ---------------------------------------------------------------------------


def prepare_for_matching(mol):
    """Return a copy of *mol* perceived with the MDL aromaticity model.

    Raises ``SmirnoffError`` when hydrogens are implicit: SMIRNOFF parameters
    are assigned to explicit atoms, so every hydrogen must be an atom.
    """
    from rdkit import Chem

    work = Chem.Mol(mol)
    work.RemoveAllConformers()
    for atom in work.GetAtoms():
        atom.SetAtomMapNum(0)
    try:
        Chem.SanitizeMol(work)
    except Exception as exc:
        raise SmirnoffError(f"RDKit cannot sanitise the molecule: {exc}") from exc
    for atom in work.GetAtoms():
        if atom.GetNumImplicitHs() or atom.GetNumExplicitHs():
            raise SmirnoffError(
                "The molecule has implicit hydrogens. Convert it to 3D (or add "
                "hydrogens) so that every hydrogen is an atom before assigning "
                "force-field parameters."
            )
    Chem.Kekulize(work, clearAromaticFlags=True)
    Chem.SetAromaticity(work, Chem.AromaticityModel.AROMATICITY_MDL)
    return work


def find_matches(work, param):
    """All matches of *param* in *work* as tuples of tagged-atom indices."""
    qmol, tag_map = param.query()
    full = work.GetSubstructMatches(
        qmol, uniquify=False, useChirality=False, maxMatches=2**31 - 1
    )
    return [tuple(match[i] for i in tag_map) for match in full]


def _canonical_bond(match):
    i, j = match
    return (i, j) if i < j else (j, i)


def _canonical_angle(match):
    i, j, k = match
    return (i, j, k) if i < k else (k, j, i)


def _canonical_proper(match):
    rev = tuple(reversed(match))
    return min(match, rev)


def _improper_key(match):
    i, center, k, l = match
    return (center,) + tuple(sorted((i, k, l)))


def _assign(work, params, key_fn):
    """Apply *params* in order, the last match per key winning.

    Returns {key: (param, match)}.
    """
    assigned = {}
    for param in params:
        for match in find_matches(work, param):
            assigned[key_fn(match)] = (param, match)
    return assigned


def topology_terms(work):
    """Bonds, angles and proper torsions of the molecular graph (canonical keys)."""
    bonds = sorted(
        _canonical_bond((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
        for b in work.GetBonds()
    )
    neighbours = {
        a.GetIdx(): sorted(n.GetIdx() for n in a.GetNeighbors())
        for a in work.GetAtoms()
    }
    angles = set()
    for j, nbrs in neighbours.items():
        for a in range(len(nbrs)):
            for b in range(a + 1, len(nbrs)):
                angles.add(_canonical_angle((nbrs[a], j, nbrs[b])))
    propers = set()
    for j, k in bonds:
        for i in neighbours[j]:
            if i == k:
                continue
            for l in neighbours[k]:
                if l in (j, i):
                    continue
                propers.add(_canonical_proper((i, j, k, l)))
    return bonds, sorted(angles), sorted(propers)


class Assignment:
    """Parameters assigned to one molecule (indices local to that molecule)."""

    def __init__(self):
        self.vdw = {}  # atom -> {"id", "sigma", "epsilon"}
        self.bonds = []  # {"i", "j", "r0", "k", "id"}
        self.angles = []  # {"i", "j", "k_atom", "theta0", "k", "id"}
        self.torsions = []  # mdff torsion dicts (+ "id")
        self.constraints = []  # {"i", "j", "r0"}
        self.library_charges = (
            None  # list of charges when the library covers every atom
        )
        self.missing = []  # human-readable warnings
        self.fallbacks = []  # terms filled by the UFF fallback


def assign_parameters(
    ff: SmirnoffForceField, mol, use_fallback: bool = True
) -> Assignment:
    """Assign *ff* to *mol* (explicit hydrogens required)."""
    work = prepare_for_matching(mol)
    result = Assignment()
    n_atoms = work.GetNumAtoms()

    # vdW
    vdw = _assign(work, ff.vdw, lambda m: m[0])
    for atom_idx in range(n_atoms):
        if atom_idx in vdw:
            param, _ = vdw[atom_idx]
            result.vdw[atom_idx] = {
                "id": param.id,
                "sigma": param.values["sigma"],
                "epsilon": param.values["epsilon"],
            }

    bonds_top, angles_top, propers_top = topology_terms(work)

    bond_params = _assign(work, ff.bonds, _canonical_bond)
    bond_lengths = {}
    for key in bonds_top:
        if key in bond_params:
            param, _ = bond_params[key]
            entry = {
                "i": key[0],
                "j": key[1],
                "r0": param.values["r0"],
                "k": param.values["k"],
                "id": param.id,
            }
            result.bonds.append(entry)
            bond_lengths[key] = param.values["r0"]
        else:
            result.bonds.append(
                {"i": key[0], "j": key[1], "r0": None, "k": None, "id": None}
            )

    angle_params = _assign(work, ff.angles, _canonical_angle)
    for key in angles_top:
        if key in angle_params:
            param, _ = angle_params[key]
            result.angles.append(
                {
                    "i": key[0],
                    "j": key[1],
                    "k_atom": key[2],
                    "theta0": param.values["theta0"],
                    "k": param.values["k"],
                    "id": param.id,
                }
            )
        else:
            result.angles.append(
                {
                    "i": key[0],
                    "j": key[1],
                    "k_atom": key[2],
                    "theta0": None,
                    "k": None,
                    "id": None,
                }
            )

    proper_params = _assign(work, ff.propers, _canonical_proper)
    for key in propers_top:
        if key in proper_params:
            param, _ = proper_params[key]
            for term in param.values["terms"]:
                result.torsions.append(
                    {
                        "i": key[0],
                        "j": key[1],
                        "k_atom": key[2],
                        "l": key[3],
                        "periodicity": term["periodicity"],
                        "phase": term["phase"],
                        "k": term["k"],
                        "improper": False,
                        "id": param.id,
                    }
                )
        else:
            result.missing.append(
                f"No proper torsion parameter for atoms {key}; it is left without a torsion term."
            )

    improper_params = _assign(work, ff.impropers, _improper_key)
    for key in sorted(improper_params):
        param, match = improper_params[key]
        # The OpenFF toolkit stores impropers under (min, centre, mid, max) and
        # builds the trefoil from that sorted order; do the same so the atom
        # order (and handedness) of every term matches the toolkit's.
        center = key[0]
        others = key[1:]
        for p0, p1, p2 in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            for term in param.values["terms"]:
                result.torsions.append(
                    {
                        "i": center,
                        "j": others[p0],
                        "k_atom": others[p1],
                        "l": others[p2],
                        "periodicity": term["periodicity"],
                        "phase": term["phase"],
                        "k": term["k"],
                        "improper": True,
                        "id": param.id,
                    }
                )

    # Constraints: a distance-less constraint takes the bond's equilibrium length.
    cons = _assign(work, ff.constraints, _canonical_bond)
    for key in sorted(cons):
        param, _ = cons[key]
        distance = param.values.get("distance")
        if distance is None:
            distance = bond_lengths.get(key)
        if distance is not None and key in bond_lengths:
            result.constraints.append({"i": key[0], "j": key[1], "r0": distance})

    # Library charges cover the molecule only if every atom gets one.
    lib = {}
    for param in ff.library_charges:
        for match in find_matches(work, param):
            for atom_idx, charge in zip(match, param.values["charges"]):
                lib[atom_idx] = charge
    if n_atoms and len(lib) == n_atoms:
        result.library_charges = [lib[i] for i in range(n_atoms)]

    if use_fallback:
        _uff_fallback(mol, result)
    else:
        _report_missing(result, n_atoms)
    return result


def _report_missing(result, n_atoms):
    for atom_idx in range(n_atoms):
        if atom_idx not in result.vdw:
            result.missing.append(f"No vdW parameter for atom {atom_idx}.")
    for bond in result.bonds:
        if bond["r0"] is None:
            result.missing.append(
                f"No bond parameter for atoms ({bond['i']}, {bond['j']})."
            )
    for angle in result.angles:
        if angle["theta0"] is None:
            result.missing.append(
                f"No angle parameter for atoms ({angle['i']}, {angle['j']}, {angle['k_atom']})."
            )


def _uff_fallback(mol, result):
    """Fill missing vdW / bond / angle terms from RDKit's UFF, or report them."""
    from rdkit import Chem

    try:
        from rdkit.Chem import ChemicalForceFields as cff

        uff_mol = Chem.Mol(mol)
        Chem.SanitizeMol(uff_mol)
        has_uff = True
    except Exception:  # noqa: BLE001 - pragma: no cover - RDKit build without UFF helpers
        has_uff = False
        uff_mol = None

    n_atoms = mol.GetNumAtoms()
    for atom_idx in range(n_atoms):
        if atom_idx in result.vdw:
            continue
        params = None
        if has_uff:
            try:
                params = cff.GetUFFVdWParams(uff_mol, atom_idx, atom_idx)
            except Exception:  # noqa: BLE001
                params = None
        if params:
            x_ij, d_ij = params  # UFF minimum distance (A) and well depth (kcal/mol)
            result.vdw[atom_idx] = {
                "id": "UFF",
                "sigma": 0.1 * x_ij / _TWO_SIXTH,
                "epsilon": 4.184 * d_ij,
            }
            result.fallbacks.append(f"vdW of atom {atom_idx} from UFF")
        else:
            result.vdw[atom_idx] = {"id": None, "sigma": 0.0, "epsilon": 0.0}
            result.missing.append(f"No vdW parameter for atom {atom_idx}; set to zero.")

    for bond in result.bonds:
        if bond["r0"] is not None:
            continue
        params = None
        if has_uff:
            try:
                params = cff.GetUFFBondStretchParams(uff_mol, bond["i"], bond["j"])
            except Exception:  # noqa: BLE001
                params = None
        if params:
            kb, r0 = params  # kcal/mol/A^2 with E = k/2 (r - r0)^2, A
            bond.update({"r0": 0.1 * r0, "k": kb * 418.4, "id": "UFF"})
            result.fallbacks.append(f"bond ({bond['i']}, {bond['j']}) from UFF")
        else:
            bond.update({"r0": 0.0, "k": 0.0, "id": None})
            result.missing.append(
                f"No bond parameter for atoms ({bond['i']}, {bond['j']}); set to zero."
            )

    for angle in result.angles:
        if angle["theta0"] is not None:
            continue
        params = None
        if has_uff:
            try:
                params = cff.GetUFFAngleBendParams(
                    uff_mol, angle["i"], angle["j"], angle["k_atom"]
                )
            except Exception:  # noqa: BLE001
                params = None
        if params:
            ka, theta0 = params  # kcal/mol/rad^2, degrees
            angle.update({"theta0": float(theta0), "k": ka * 4.184, "id": "UFF"})
            result.fallbacks.append(
                f"angle ({angle['i']}, {angle['j']}, {angle['k_atom']}) from UFF (harmonic approximation)"
            )
        else:
            angle.update({"theta0": 0.0, "k": 0.0, "id": None})
            result.missing.append(
                f"No angle parameter for atoms ({angle['i']}, {angle['j']}, {angle['k_atom']}); set to zero."
            )


def sigma_from_rmin_half(rmin_half_nm: float) -> float:
    """Convenience for tests: Lennard-Jones sigma from Rmin/2."""
    return 2.0 * rmin_half_nm / _TWO_SIXTH
