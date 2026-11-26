"""Partial-charge methods.

Built in (RDKit only): Gasteiger and MMFF94. Optional, when present: the
OpenFF NAGL graph model (``openff-nagl``), AM1-BCC through AmberTools
``antechamber``/``sqm``, GFN2-xTB through the ``xtb`` executable, and charges
imported from a QM output (ORCA CHELPG/RESP/Hirshfeld/Mulliken tables, PySCF
Calculator ``properties.json``) or a plain list.

Every method returns one charge per atom of the molecule it is given, already
shifted so that the total equals the molecule's net formal charge.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile

logger = logging.getLogger(__name__)

METHOD_LABELS = {
    "auto": "Force field's own scheme (fallback: MMFF94)",
    "gasteiger": "Gasteiger",
    "mmff94": "MMFF94",
    "am1bcc": "AM1-BCC (AmberTools sqm)",
    "nagl": "AM1-BCC (OpenFF NAGL)",
    "xtb": "GFN2-xTB (xtb)",
    "imported": "Imported from file",
}


class ChargeError(RuntimeError):
    """A charge method is unavailable or failed."""


def formal_charge(mol) -> int:
    return sum(a.GetFormalCharge() for a in mol.GetAtoms())


def normalise(charges, target):
    """Spread the residual evenly so the total equals *target*; returns (charges, residual)."""
    charges = [float(q) for q in charges]
    if not charges:
        return charges, 0.0
    residual = target - sum(charges)
    if abs(residual) > 1e-9:
        shift = residual / len(charges)
        charges = [q + shift for q in charges]
    return charges, residual


def average_equivalent(mol, values):
    """Average charges over symmetry-equivalent atoms.

    Atoms are equivalent when they are equivalent in the bond graph with explicit
    hydrogens, ignoring bond orders and formal charges, so resonance forms count as
    one (the three hydrogens of a methyl group, the two oxygens of a carboxylate or
    of a nitro group). Returns (charges, number of atoms whose charge changed). The
    total charge is unchanged.
    """
    from rdkit import Chem

    work = Chem.RWMol(mol)
    for atom in work.GetAtoms():
        atom.SetFormalCharge(0)
        atom.SetIsAromatic(False)
        atom.SetIsotope(0)
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(0)
    for bond in work.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
    work.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(work)
    ranks = list(Chem.CanonicalRankAtoms(work, breakTies=False))
    groups = {}
    for index, rank in enumerate(ranks):
        groups.setdefault(rank, []).append(index)
    out = [float(q) for q in values]
    changed = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        mean = sum(out[i] for i in members) / len(members)
        changed += sum(1 for i in members if abs(out[i] - mean) > 1e-6)
        for i in members:
            out[i] = mean
    return out, changed


#: Methods whose charges can differ between equivalent atoms.
AVERAGED_METHODS = ("imported", "xtb")


# -- availability ---------------------------------------------------------------


def has_executable(name):
    return shutil.which(name) is not None


def has_module(name):
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def available_methods():
    """Method keys usable right now."""
    methods = ["auto", "gasteiger", "mmff94"]
    if has_executable("antechamber") and has_executable("sqm"):
        methods.append("am1bcc")
    if has_module("openff.nagl"):
        methods.append("nagl")
    if has_executable("xtb"):
        methods.append("xtb")
    methods.append("imported")
    return methods


# -- built-in methods --------------------------------------------------------------


def gasteiger(mol):
    from rdkit import Chem
    from rdkit.Chem import rdPartialCharges

    work = Chem.Mol(mol)
    rdPartialCharges.ComputeGasteigerCharges(work, throwOnParamFailure=True)
    charges = [float(a.GetDoubleProp("_GasteigerCharge")) for a in work.GetAtoms()]
    if any(math.isnan(q) for q in charges):
        raise ChargeError("Gasteiger charges are undefined for this molecule")
    return charges


def mmff94(mol):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    work = Chem.Mol(mol)
    props = AllChem.MMFFGetMoleculeProperties(work)
    if props is None:
        raise ChargeError("MMFF94 has no parameters for this molecule")
    return [float(props.GetMMFFPartialCharge(i)) for i in range(work.GetNumAtoms())]


# -- optional methods -------------------------------------------------------------------


def nagl(mol):
    """OpenFF NAGL AM1-BCC graph model (needs openff-toolkit and openff-nagl)."""
    try:
        from openff.toolkit import Molecule
    except ImportError as exc:
        raise ChargeError("openff-toolkit is not installed") from exc
    try:
        offmol = Molecule.from_rdkit(
            mol, allow_undefined_stereo=True, hydrogens_are_explicit=True
        )
        offmol.assign_partial_charges("openff-gnn-am1bcc-1.0.0.pt")
        return [float(q) for q in offmol.partial_charges.m_as("elementary_charge")]
    except Exception as exc:
        raise ChargeError(f"NAGL failed: {exc}") from exc


def write_mol2(mol, path, charges=None, resname="MOL"):
    """A minimal Tripos MOL2 file (element names as atom types) in atom order."""
    conf = mol.GetConformer() if mol.GetNumConformers() else None
    counts = {}
    lines = [
        "@<TRIPOS>MOLECULE",
        resname,
        f"{mol.GetNumAtoms()} {mol.GetNumBonds()} 1 0 0",
        "SMALL",
        "USER_CHARGES" if charges is not None else "NO_CHARGES",
        "",
        "@<TRIPOS>ATOM",
    ]
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        symbol = atom.GetSymbol()
        counts[symbol] = counts.get(symbol, 0) + 1
        x, y, z = (0.0, 0.0, 0.0)
        if conf is not None:
            p = conf.GetAtomPosition(idx)
            x, y, z = p.x, p.y, p.z
        q = charges[idx] if charges is not None else 0.0
        lines.append(
            f"{idx + 1:7d} {symbol + str(counts[symbol]):<8s} {x:10.4f} {y:10.4f} {z:10.4f} "
            f"{symbol:<5s} 1 {resname:<4s} {q:10.6f}"
        )
    lines.append("@<TRIPOS>BOND")
    for n, bond in enumerate(mol.GetBonds(), start=1):
        if bond.GetIsAromatic():
            order = "ar"
        else:
            order = {1.0: "1", 2.0: "2", 3.0: "3"}.get(bond.GetBondTypeAsDouble(), "1")
        lines.append(
            f"{n:6d} {bond.GetBeginAtomIdx() + 1:5d} {bond.GetEndAtomIdx() + 1:5d} {order}"
        )
    lines.append("@<TRIPOS>SUBSTRUCTURE")
    lines.append(f"     1 {resname:<4s}  1 TEMP 0 **** **** 0 ROOT")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")


def read_mol2_charges(path):
    charges = []
    in_atoms = False
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("@<TRIPOS>"):
                in_atoms = line.strip() == "@<TRIPOS>ATOM"
                continue
            if in_atoms and line.strip():
                charges.append(float(line.split()[8]))
    return charges


def am1bcc(mol, timeout=600):
    """AM1-BCC through AmberTools antechamber (runs sqm)."""
    exe = shutil.which("antechamber")
    if exe is None or shutil.which("sqm") is None:
        raise ChargeError("AmberTools antechamber/sqm are not on PATH")
    with tempfile.TemporaryDirectory(prefix="ffb_am1bcc_") as tmp:
        write_mol2(mol, os.path.join(tmp, "in.mol2"))
        cmd = [
            exe,
            "-i",
            "in.mol2",
            "-fi",
            "mol2",
            "-o",
            "out.mol2",
            "-fo",
            "mol2",
            "-c",
            "bcc",
            "-nc",
            str(formal_charge(mol)),
            "-at",
            "gaff2",
            "-dr",
            "no",
            "-pf",
            "y",
        ]
        proc = subprocess.run(
            cmd, cwd=tmp, capture_output=True, text=True, timeout=timeout, check=False
        )
        out = os.path.join(tmp, "out.mol2")
        if proc.returncode != 0 or not os.path.exists(out):
            raise ChargeError(
                f"antechamber failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
            )
        charges = read_mol2_charges(out)
    if len(charges) != mol.GetNumAtoms():
        raise ChargeError("antechamber returned a different number of atoms")
    return charges


# -- RESP from a Gaussian ESP job (AmberTools) ----------------------------------------

# antechamber -c resp runs these three in turn.
RESP_PROGRAMS = ("antechamber", "espgen", "respgen", "resp")
_GAUSSIAN_CHARGE = re.compile(
    r"^\s*Charge\s*=\s*(-?\d+)\s+Multiplicity\s*=", re.MULTILINE
)


def has_resp():
    return all(has_executable(name) for name in RESP_PROGRAMS)


def has_esp_points(text):
    """True for a Gaussian output that lists the fitted ESP points (Pop=MK IOp(6/33=2))."""
    return "Entering Gaussian System" in text and "ESP Fit Center" in text


def gaussian_net_charge(text):
    found = _GAUSSIAN_CHARGE.findall(text)
    if not found:
        raise ChargeError("The Gaussian output gives no charge and multiplicity")
    return int(found[0])


def resp_from_gaussian(path, text=None, timeout=600):
    """Two-stage RESP charges fitted by AmberTools to the ESP points in a Gaussian output."""
    if not has_resp():
        raise ChargeError("AmberTools antechamber/espgen/respgen/resp are not on PATH")
    if text is None:
        with open(path, encoding="utf-8", errors="replace") as handle:
            text = handle.read()
    if not has_esp_points(text):
        raise ChargeError(
            "No ESP points in this Gaussian output: run it with Pop=MK IOp(6/33=2)"
        )
    with tempfile.TemporaryDirectory(prefix="ffb_resp_") as tmp:
        # A plain name in the work folder: antechamber splits paths on spaces.
        with open(os.path.join(tmp, "esp.log"), "w", encoding="utf-8") as handle:
            handle.write(text)
        cmd = [
            shutil.which("antechamber"),
            "-i",
            "esp.log",
            "-fi",
            "gout",
            "-o",
            "resp.mol2",
            "-fo",
            "mol2",
            "-c",
            "resp",
            "-nc",
            str(gaussian_net_charge(text)),
            "-at",
            "gaff2",
            "-dr",
            "no",
            "-pf",
            "y",
        ]
        proc = subprocess.run(
            cmd, cwd=tmp, capture_output=True, text=True, timeout=timeout, check=False
        )
        out = os.path.join(tmp, "resp.mol2")
        if proc.returncode != 0 or not os.path.exists(out):
            raise ChargeError(
                f"antechamber RESP failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
            )
        charges = read_mol2_charges(out)
    if not charges:
        raise ChargeError("antechamber wrote no RESP charges")
    return charges


def xtb_charges(mol, timeout=600):
    """GFN2-xTB charges (the ``charges`` file xtb writes after a single point)."""
    exe = shutil.which("xtb")
    if exe is None:
        raise ChargeError("xtb is not on PATH")
    if not mol.GetNumConformers():
        raise ChargeError("xtb needs 3D coordinates")
    conf = mol.GetConformer()
    with tempfile.TemporaryDirectory(prefix="ffb_xtb_") as tmp:
        lines = [str(mol.GetNumAtoms()), "moleditpy force field builder"]
        for atom in mol.GetAtoms():
            p = conf.GetAtomPosition(atom.GetIdx())
            lines.append(f"{atom.GetSymbol()} {p.x:.6f} {p.y:.6f} {p.z:.6f}")
        with open(
            os.path.join(tmp, "in.xyz"), "w", encoding="utf-8", newline="\n"
        ) as handle:
            handle.write("\n".join(lines) + "\n")
        n_unpaired = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
        cmd = [
            exe,
            "in.xyz",
            "--sp",
            "--chrg",
            str(formal_charge(mol)),
            "--uhf",
            str(n_unpaired),
        ]
        proc = subprocess.run(
            cmd, cwd=tmp, capture_output=True, text=True, timeout=timeout, check=False
        )
        path = os.path.join(tmp, "charges")
        if proc.returncode != 0 or not os.path.exists(path):
            raise ChargeError(
                f"xtb failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
            )
        with open(path, encoding="utf-8") as handle:
            charges = [float(x) for x in handle.read().split()]
    if len(charges) != mol.GetNumAtoms():
        raise ChargeError("xtb returned a different number of atoms")
    return charges


# -- imported charges ------------------------------------------------------------

_NUMBERS = r"((?:[-+]?\d+\.\d+(?:[eE][-+]?\d+)?\s*)+)"
# "idx El [:] numbers": ORCA, Gaussian, Q-Chem and Psi4 tables.
_ROW_INDEX_ELEMENT = re.compile(r"^\s*\d+\s+([A-Za-z]{1,3})\s*:?\s+" + _NUMBERS + "$")
# "idx Z El covCN q C6AA alpha": the xtb property table.
_ROW_XTB = re.compile(r"^\s*\d+\s+\d+\s+([A-Za-z]{1,3})\s+" + _NUMBERS + "$")
# "El idx charge core valence ...": the NBO natural population summary.
_ROW_NPA = re.compile(r"^\s*([A-Za-z]{1,3})\s+\d+\s+" + _NUMBERS + "$")

FIRST, LAST, SECOND = 0, -1, 1  # which number on a row is the charge


def _section(program, scheme, header, row=_ROW_INDEX_ELEMENT, pick=FIRST):
    return {
        "program": program,
        "scheme": scheme,
        "header": re.compile(header, re.IGNORECASE),
        "row": row,
        "pick": pick,
    }


# Every table the importer knows, in order of preference for MD: ESP-fitted charges
# first, then the density-partitioning schemes, then orbital-based populations.
# Headers are taken from the programs' output (and cross-checked with cclib).
CHARGE_SECTIONS = (
    _section("ORCA", "RESP", r"^\s*RESP CHARGES\s*$"),
    _section("Q-Chem", "RESP", r"^\s*Merz-Kollman RESP Net Atomic Charges\s*$"),
    _section("ORCA", "CHELPG", r"^\s*CHELPG CHARGES\s*$"),
    _section("Q-Chem", "CHELPG", r"^\s*Ground-State ChElPG Net Atomic Charges\s*$"),
    _section(
        "Q-Chem", "Merz-Kollman ESP", r"^\s*Merz-Kollman ESP Net Atomic Charges\s*$"
    ),
    # Pop=CHelpG / MK / HLY: "Charges from ESP fit, RMS= ..." then "ESP charges:".
    _section(
        "Gaussian",
        "ESP-fitted",
        r"^\s*(?:ESP charges\b.*:|Charges from ESP fit\b.*)\s*$",
    ),
    _section(
        "Psi4", "MBIS", r"^\s*MBIS Charges(?:: \(a\.u\.\)| \[a\.u\.\]:)", pick=LAST
    ),
    _section(
        "Gaussian",
        "CM5",
        r"^\s*Hirshfeld charges, spin densities, dipoles, and CM5 charges",
        pick=LAST,
    ),
    _section("Q-Chem", "CM5", r"^\s*Charge Model 5\b"),
    _section("ORCA", "Hirshfeld", r"^\s*HIRSHFELD ANALYSIS\s*$"),
    _section(
        "Gaussian",
        "Hirshfeld",
        r"^\s*Hirshfeld charges, spin densities, dipoles, and CM5 charges",
    ),
    _section("Q-Chem", "Hirshfeld", r"^\s*Hirshfeld Atomic Charges\s*$"),
    _section(
        "Gaussian",
        "NPA",
        r"^\s*Summary of Natural Population Analysis:\s*$",
        row=_ROW_NPA,
    ),
    _section("xtb", "GFN", r"^\s*#\s+Z\s+covCN\s+q\b", row=_ROW_XTB, pick=SECOND),
    _section(
        "ORCA",
        "Loewdin",
        r"^\s*LOEWDIN ATOMIC CHARGES(?: AND SPIN (?:POPULATIONS|DENSITIES))?\s*$",
    ),
    _section(
        "Psi4", "Lowdin", r"^\s*Lowdin Charges(?:: \(a\.u\.\)| \[a\.u\.\]:)", pick=LAST
    ),
    _section(
        "Gaussian",
        "Lowdin",
        r"^\s*Lowdin (?:atomic )?charges(?: and spin densities)?:\s*$",
    ),
    _section("Gaussian", "APT", r"^\s*APT (?:atomic )?charges:\s*$"),
    _section(
        "ORCA",
        "Mulliken",
        r"^\s*MULLIKEN ATOMIC CHARGES(?: AND SPIN (?:POPULATIONS|DENSITIES))?\s*$",
    ),
    _section("Q-Chem", "Mulliken", r"^\s*Ground-State Mulliken Net Atomic Charges\s*$"),
    _section(
        "Psi4",
        "Mulliken",
        r"^\s*Mulliken Charges(?:: \(a\.u\.\)| \[a\.u\.\]:)",
        pick=LAST,
    ),
    _section(
        "Gaussian",
        "Mulliken",
        r"^\s*Mulliken (?:atomic )?charges(?: and spin densities)?:\s*$",
    ),
)

# Lines allowed between a header and the first row (column titles, rules, notes).
_MAX_PREAMBLE = 12


def _element(symbol):
    return symbol[:1].upper() + symbol[1:].lower()


def _read_section(lines, start, section):
    """[(element, charge), ...] from the rows following the header at *start*."""
    rows = []
    for offset, line in enumerate(lines[start + 1 :]):
        m = section["row"].match(line)
        if m:
            numbers = [float(x) for x in m.group(2).split()]
            try:
                rows.append((_element(m.group(1)), numbers[section["pick"]]))
            except IndexError:
                break
        elif rows or offset >= _MAX_PREAMBLE:
            break  # the table ended (blank line, rule, "Sum ..." or "Total ...")
    return rows


def find_charge_sets(text):
    """Every charge table in a QM output, best first: [(program, scheme, rows), ...].

    The last table of each kind is used (an optimisation prints one per step).
    """
    lines = text.splitlines()
    found = []
    for section in CHARGE_SECTIONS:
        starts = [n for n, line in enumerate(lines) if section["header"].match(line)]
        for start in reversed(starts):
            rows = _read_section(lines, start, section)
            if rows:
                found.append((section["program"], section["scheme"], rows))
                break
    return found


def parse_charge_file(path):
    """Read charges from a file. Returns (charges, elements_or_None, description).

    For a QM output with several charge tables, the preferred one (see
    CHARGE_SECTIONS); charge_sets() lists them all.
    """
    return charge_sets(path)[0]


def charge_sets(path):
    """Every charge set in a file, best first: [(charges, elements, description), ...]."""
    with open(path, encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    name = os.path.basename(path)

    if name.lower().endswith(".json"):
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ChargeError(f"{name} has no charge list")
        sets = [
            ([float(q) for q in data[key]], data.get("atom_symbols"), f"{name}: {key}")
            for key in (
                "resp_charges",
                "esp_charges",
                "chelpg_charges",
                "charges",
                "partial charges",  # xtb --json (xtbout.json)
                "mulliken_charges",
            )
            if isinstance(data.get(key), list)
        ]
        if not sets:
            raise ChargeError(f"{name} has no charge list")
        return sets

    tables = find_charge_sets(text)
    resp = _resp_set(path, name, text, tables)
    if tables:
        return resp + [
            (
                [q for _, q in rows],
                [e for e, _ in rows],
                f"{name}: {program} {scheme} charges",
            )
            for program, scheme, rows in tables
        ]

    # Plain text: one charge per line, or "[index] [element] [x y z] charge" rows
    # (the last number on each line is the charge, as in Multiwfn .chg files).
    charges, elements = [], []
    for line in text.splitlines():
        parts = line.replace(",", " ").split()
        if not parts or parts[0].startswith("#"):
            continue
        try:
            charges.append(float(parts[-1]))
        except ValueError:
            continue
        symbol = next((p for p in parts[:-1] if p.isalpha()), None)
        elements.append(symbol)
    if not charges:
        raise ChargeError(f"No charges found in {name}")
    if not all(elements):
        elements = None
    return [(charges, elements, f"{name}: plain charge list")]


def _resp_set(path, name, text, tables):
    """[RESP charge set] for a Gaussian ESP job when AmberTools is installed, else []."""
    if not (has_esp_points(text) and has_resp()):
        return []
    try:
        values = resp_from_gaussian(path, text)
    except (ChargeError, OSError, subprocess.SubprocessError) as exc:
        logger.warning("Force Field Builder: RESP fit skipped: %s", exc)
        return []
    # Same atoms, same order as the ESP-fitted table antechamber read.
    elements = next(
        ([e for e, _ in rows] for _p, _s, rows in tables if len(rows) == len(values)),
        None,
    )
    return [(values, elements, f"{name}: RESP charges (AmberTools)")]


def compute(method, mol, imported=None):
    """Charges for *mol* by *method* (not normalised). *imported* is a list for 'imported'."""
    try:
        return _compute(method, mol, imported)
    except ChargeError:
        raise
    except Exception as exc:
        raise ChargeError(f"{METHOD_LABELS.get(method, method)} failed: {exc}") from exc


def _compute(method, mol, imported=None):
    if method == "gasteiger":
        return gasteiger(mol)
    if method == "mmff94":
        return mmff94(mol)
    if method == "am1bcc":
        return am1bcc(mol)
    if method == "nagl":
        return nagl(mol)
    if method == "xtb":
        return xtb_charges(mol)
    if method == "imported":
        if imported is None or len(imported) != mol.GetNumAtoms():
            raise ChargeError("Imported charges do not match the molecule's atom count")
        return list(imported)
    raise ChargeError(f"Unknown charge method {method!r}")
