"""Optional parameter sources: acpype (GAFF/GAFF2), openmmforcefields, openff-toolkit.

None of these packages is required. Each entry point raises ``ExternalError``
with a readable message when its tool is missing or fails, and the pure
converters (GROMACS ``.itp`` text and OpenMM ``System`` objects to an mdff
molecule type) are usable and tested on their own.
"""

from __future__ import annotations

import glob
import math
import os
import shutil
import subprocess
import tempfile

from . import charges as charge_mod


class ExternalError(RuntimeError):
    """An optional tool is unavailable or failed."""


def has_acpype():
    return shutil.which("acpype") is not None


def has_openmmforcefields():
    return charge_mod.has_module("openmmforcefields") and charge_mod.has_module(
        "openmm"
    )


def has_openff_toolkit():
    return charge_mod.has_module("openff.toolkit")


# ---------------------------------------------------------------------------
# GROMACS .itp -> mdff
# ---------------------------------------------------------------------------


def _strip(line):
    return line.split(";", 1)[0].strip()


def parse_gromacs_itp(text, atomtypes_text=None):
    """Convert one GROMACS molecule (``[ atoms ]`` ... ``[ dihedrals ]``) to an mdff type.

    ``[ atomtypes ]`` may be in *text* or in *atomtypes_text*. Supported terms:
    bonds funct 1, angles funct 1, dihedrals funct 1/9 (proper) and 4
    (improper periodic). Other functional forms are reported in ``warnings``.
    Returns (molecule_type, warnings).
    """
    atomtypes = {}
    warnings = []
    mtype = {
        "name": "MOL",
        "atoms": [],
        "bonds": [],
        "angles": [],
        "torsions": [],
        "exclusions_nrexcl": 3,
    }

    def sections(src):
        current = None
        for raw in src.splitlines():
            line = _strip(raw)
            if not line or line.startswith("#"):
                continue
            if line.startswith("[") and line.endswith("]"):
                current = line[1:-1].strip().lower()
                continue
            yield current, line.split()

    for source in (atomtypes_text or "", text):
        for section, parts in sections(source):
            if section != "atomtypes":
                continue
            ptype_pos = next(
                (n for n, p in enumerate(parts) if p in ("A", "S", "V", "D")), None
            )
            if ptype_pos is None or ptype_pos < 3 or len(parts) < ptype_pos + 3:
                warnings.append(f"Unreadable atomtype line: {' '.join(parts)}")
                continue
            atomtypes[parts[0]] = {
                "mass": float(parts[ptype_pos - 2]),
                "sigma": float(parts[ptype_pos + 1]),
                "epsilon": float(parts[ptype_pos + 2]),
            }

    for section, parts in sections(text):
        if section == "moleculetype":
            mtype["name"] = parts[0]
            if len(parts) > 1:
                mtype["exclusions_nrexcl"] = int(parts[1])
        elif section == "atoms":
            atype = parts[1]
            info = atomtypes.get(atype)
            if info is None:
                warnings.append(f"Atom type {atype} has no [ atomtypes ] entry")
                info = {"mass": 0.0, "sigma": 0.0, "epsilon": 0.0}
            mass = float(parts[7]) if len(parts) > 7 else info["mass"]
            mtype["atoms"].append(
                {
                    "name": parts[4],
                    "element": "",
                    "type": atype,
                    "charge": float(parts[6]),
                    "mass": mass,
                    "sigma": info["sigma"],
                    "epsilon": info["epsilon"],
                }
            )
        elif section == "bonds":
            i, j, funct = int(parts[0]) - 1, int(parts[1]) - 1, int(parts[2])
            if funct != 1 or len(parts) < 5:
                warnings.append(f"Bond {i}-{j} funct {funct} not converted")
                continue
            mtype["bonds"].append(
                {"i": i, "j": j, "r0": float(parts[3]), "k": float(parts[4])}
            )
        elif section == "angles":
            i, j, k, funct = (
                int(parts[0]) - 1,
                int(parts[1]) - 1,
                int(parts[2]) - 1,
                int(parts[3]),
            )
            if funct != 1 or len(parts) < 6:
                warnings.append(f"Angle {i}-{j}-{k} funct {funct} not converted")
                continue
            mtype["angles"].append(
                {
                    "i": i,
                    "j": j,
                    "k_atom": k,
                    "theta0": float(parts[4]),
                    "k": float(parts[5]),
                }
            )
        elif section == "dihedrals":
            idx = [int(p) - 1 for p in parts[:4]]
            funct = int(parts[4])
            if funct in (1, 4, 9) and len(parts) >= 8:
                mtype["torsions"].append(
                    {
                        "i": idx[0],
                        "j": idx[1],
                        "k_atom": idx[2],
                        "l": idx[3],
                        "periodicity": int(float(parts[7])),
                        "phase": float(parts[5]),
                        "k": float(parts[6]),
                        "improper": funct == 4,
                    }
                )
            else:
                warnings.append(
                    f"Dihedral {idx} funct {funct} not converted (only periodic forms are)"
                )
    return mtype, warnings


def run_acpype(mol, atom_type="gaff2", user_charges=None, timeout=1800):
    """Parameterise *mol* with acpype/antechamber. Returns (molecule_type, warnings)."""
    exe = shutil.which("acpype")
    if exe is None:
        raise ExternalError("acpype is not on PATH")
    with tempfile.TemporaryDirectory(prefix="ffb_acpype_") as tmp:
        charge_mod.write_mol2(mol, os.path.join(tmp, "MOL.mol2"), charges=user_charges)
        cmd = [
            exe,
            "-i",
            "MOL.mol2",
            "-b",
            "MOL",
            "-a",
            atom_type,
            "-n",
            str(charge_mod.formal_charge(mol)),
            "-o",
            "gmx",
            "-c",
            "user" if user_charges is not None else "bcc",
        ]
        proc = subprocess.run(
            cmd, cwd=tmp, capture_output=True, text=True, timeout=timeout, check=False
        )
        itps = glob.glob(os.path.join(tmp, "**", "MOL_GMX.itp"), recursive=True)
        if proc.returncode != 0 or not itps:
            raise ExternalError(
                f"acpype failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
            )
        with open(itps[0], encoding="utf-8") as handle:
            text = handle.read()
    mtype, warnings = parse_gromacs_itp(text)
    if len(mtype["atoms"]) != mol.GetNumAtoms():
        raise ExternalError("acpype returned a different number of atoms")
    for atom, rd_atom in zip(mtype["atoms"], mol.GetAtoms()):
        atom["element"] = rd_atom.GetSymbol()
    return mtype, warnings


# ---------------------------------------------------------------------------
# OpenMM System -> mdff
# ---------------------------------------------------------------------------


def _md(value):
    """A plain float from an OpenMM Quantity in MD units (nm, kJ/mol, rad, e)."""
    inner = getattr(value, "_value", value)
    return float(inner)


def mdff_type_from_openmm_system(system, mol, name="MOL"):
    """Read bonds, angles, periodic torsions and LJ/charges from an OpenMM System.

    *mol* provides elements and the bond graph (to tell propers from impropers).
    """
    bonded = set()
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bonded.add((a, b))
        bonded.add((b, a))
    n_atoms = mol.GetNumAtoms()
    atoms = [
        {
            "name": f"{a.GetSymbol()}{a.GetIdx() + 1}",
            "element": a.GetSymbol(),
            "type": "",
            "charge": 0.0,
            "mass": _md(system.getParticleMass(a.GetIdx())),
            "sigma": 0.0,
            "epsilon": 0.0,
        }
        for a in mol.GetAtoms()
    ]
    mtype = {
        "name": name,
        "atoms": atoms,
        "bonds": [],
        "angles": [],
        "torsions": [],
        "exclusions_nrexcl": 3,
    }
    warnings = []
    for force in system.getForces():
        kind = type(force).__name__
        if kind == "HarmonicBondForce":
            for n in range(force.getNumBonds()):
                i, j, r0, k = force.getBondParameters(n)
                mtype["bonds"].append({"i": i, "j": j, "r0": _md(r0), "k": _md(k)})
        elif kind == "HarmonicAngleForce":
            for n in range(force.getNumAngles()):
                i, j, k_atom, theta0, k = force.getAngleParameters(n)
                mtype["angles"].append(
                    {
                        "i": i,
                        "j": j,
                        "k_atom": k_atom,
                        "theta0": math.degrees(_md(theta0)),
                        "k": _md(k),
                    }
                )
        elif kind == "PeriodicTorsionForce":
            for n in range(force.getNumTorsions()):
                i, j, k_atom, l, per, phase, k = force.getTorsionParameters(n)
                proper = (
                    (i, j) in bonded and (j, k_atom) in bonded and (k_atom, l) in bonded
                )
                mtype["torsions"].append(
                    {
                        "i": i,
                        "j": j,
                        "k_atom": k_atom,
                        "l": l,
                        "periodicity": int(per),
                        "phase": math.degrees(_md(phase)),
                        "k": _md(k),
                        "improper": not proper,
                    }
                )
        elif kind == "NonbondedForce":
            for n in range(min(n_atoms, force.getNumParticles())):
                q, sigma, eps = force.getParticleParameters(n)
                atoms[n].update(
                    {"charge": _md(q), "sigma": _md(sigma), "epsilon": _md(eps)}
                )
        elif kind in ("CMMotionRemover",):
            continue
        else:
            warnings.append(f"{kind} is not converted")
    # One LJ type per distinct (element, sigma, epsilon).
    seen = {}
    for atom in atoms:
        key = (atom["element"], round(atom["sigma"], 6), round(atom["epsilon"], 6))
        if key not in seen:
            seen[key] = f"{name}_{atom['element']}{len(seen) + 1}"
        atom["type"] = seen[key]
    return mtype, warnings


def run_openmmforcefields_gaff(mol, name="MOL", forcefield="gaff-2.11"):
    """GAFF through openmmforcefields' GAFFTemplateGenerator (needs openff-toolkit)."""
    try:
        from openff.toolkit import Molecule
        from openmm import app
        from openmmforcefields.generators import GAFFTemplateGenerator
    except ImportError as exc:
        raise ExternalError(f"openmmforcefields path unavailable: {exc}") from exc
    try:
        offmol = Molecule.from_rdkit(
            mol, allow_undefined_stereo=True, hydrogens_are_explicit=True
        )
        generator = GAFFTemplateGenerator(molecules=[offmol], forcefield=forcefield)
        ff = app.ForceField()
        ff.registerTemplateGenerator(generator.generator)
        topology = offmol.to_topology().to_openmm()
        system = ff.createSystem(
            topology, nonbondedMethod=app.NoCutoff, constraints=None, rigidWater=False
        )
    except Exception as exc:
        raise ExternalError(f"openmmforcefields failed: {exc}") from exc
    return mdff_type_from_openmm_system(system, mol, name)


# ---------------------------------------------------------------------------
# openff-toolkit cross-check
# ---------------------------------------------------------------------------


def toolkit_labels(mol, offxml_path):
    """Parameter ids per term from openff-toolkit's ``label_molecules``.

    Returns {"Bonds": {(i, j): id}, "Angles": {...}, "ProperTorsions": {...},
    "vdW": {(i,): id}} with canonical keys as in ``smirnoff``.
    """
    from openff.toolkit import ForceField, Molecule

    ff = ForceField(offxml_path, load_plugins=False)
    offmol = Molecule.from_rdkit(
        mol, allow_undefined_stereo=True, hydrogens_are_explicit=True
    )
    labels = ff.label_molecules(offmol.to_topology())[0]
    out = {}
    for handler in ("Bonds", "Angles", "ProperTorsions", "vdW", "ImproperTorsions"):
        mapping = {}
        for key, param in labels.get(handler, {}).items():
            key = tuple(key)
            if handler in ("Bonds", "Angles", "ProperTorsions"):
                rev = tuple(reversed(key))
                key = min(key, rev)
            elif handler == "ImproperTorsions":
                key = (key[1],) + tuple(sorted((key[0], key[2], key[3])))
            mapping[key] = param.id
        out[handler] = mapping
    return out


def ours_as_labels(assignment):
    """The same mapping built from a ``smirnoff.Assignment``."""
    out = {
        "Bonds": {},
        "Angles": {},
        "ProperTorsions": {},
        "vdW": {},
        "ImproperTorsions": {},
    }
    for atom, info in assignment.vdw.items():
        out["vdW"][(atom,)] = info["id"]
    for bond in assignment.bonds:
        out["Bonds"][(bond["i"], bond["j"])] = bond["id"]
    for angle in assignment.angles:
        out["Angles"][(angle["i"], angle["j"], angle["k_atom"])] = angle["id"]
    for tor in assignment.torsions:
        if tor["improper"]:
            key = (tor["i"],) + tuple(sorted((tor["j"], tor["k_atom"], tor["l"])))
            out["ImproperTorsions"][key] = tor["id"]
        else:
            out["ProperTorsions"][(tor["i"], tor["j"], tor["k_atom"], tor["l"])] = tor[
                "id"
            ]
    return out


def compare_labels(ours, theirs):
    """List of human-readable differences (empty when they agree)."""
    diffs = []
    for handler in ("vdW", "Bonds", "Angles", "ProperTorsions", "ImproperTorsions"):
        a, b = ours.get(handler, {}), theirs.get(handler, {})
        for key in sorted(set(a) | set(b)):
            if a.get(key) != b.get(key):
                diffs.append(
                    f"{handler} {key}: builder {a.get(key)} vs toolkit {b.get(key)}"
                )
    return diffs
