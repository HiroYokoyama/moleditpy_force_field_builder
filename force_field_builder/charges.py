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
import math
import os
import re
import shutil
import subprocess
import tempfile

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

# ORCA tables, in order of preference for MD (ESP-fitted first).
_ORCA_TABLES = (
    ("RESP", re.compile(r"^\s*RESP CHARGES\s*$", re.IGNORECASE)),
    ("CHELPG", re.compile(r"^\s*CHELPG CHARGES\s*$", re.IGNORECASE)),
    ("Hirshfeld", re.compile(r"^\s*HIRSHFELD ANALYSIS\s*$", re.IGNORECASE)),
    ("Mulliken", re.compile(r"^\s*MULLIKEN ATOMIC CHARGES\s*$", re.IGNORECASE)),
)
_ORCA_ROW = re.compile(r"^\s*(\d+)\s*([A-Za-z]{1,3})\s*:?\s*([-+]?\d+\.\d+)")


def _parse_orca(text):
    """{scheme: [(element, charge), ...]} from the last table of each kind."""
    lines = text.splitlines()
    found = {}
    for scheme, header in _ORCA_TABLES:
        start = None
        for n, line in enumerate(lines):
            if header.match(line):
                start = n
        if start is None:
            continue
        rows = []
        for line in lines[start + 1 :]:
            m = _ORCA_ROW.match(line)
            if m:
                rows.append((m.group(2), float(m.group(3))))
            elif rows and (
                not line.strip()
                or line.strip().startswith("-")
                or "Sum" in line
                or "Total" in line
            ):
                break
        if rows:
            found[scheme] = rows
    return found


def parse_charge_file(path):
    """Read charges from a file. Returns (charges, elements_or_None, description)."""
    with open(path, encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    name = os.path.basename(path)

    if name.lower().endswith(".json"):
        data = json.loads(text)
        for key in (
            "resp_charges",
            "esp_charges",
            "chelpg_charges",
            "charges",
            "mulliken_charges",
        ):
            if isinstance(data.get(key), list):
                return (
                    [float(q) for q in data[key]],
                    data.get("atom_symbols"),
                    f"{name}: {key}",
                )
        raise ChargeError(f"{name} has no charge list")

    orca = _parse_orca(text)
    if orca:
        for scheme, _ in _ORCA_TABLES:
            if scheme in orca:
                rows = orca[scheme]
                return (
                    [q for _, q in rows],
                    [e for e, _ in rows],
                    f"{name}: ORCA {scheme} charges",
                )

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
    return charges, elements, f"{name}: plain charge list"


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
