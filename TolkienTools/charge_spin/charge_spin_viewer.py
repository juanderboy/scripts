#!/usr/bin/env python3
"""HTML molecular viewers for automatic spin-localization results.

The viewer highlights selected atoms on an XYZ or ORCA geometry and writes an
interactive 3D representation for visual inspection.
"""

import glob
import math
import os
import webbrowser
from pathlib import Path


def parse_first_xyz_frame(xyz_path):
    """
    Parse the first frame of a simple XYZ file.
    """
    with open(xyz_path, "r") as f:
        natoms_line = f.readline()
        if not natoms_line:
            raise ValueError("empty XYZ file")
        natoms = int(natoms_line.strip())
        comment = f.readline().rstrip("\n")
        atoms = []
        for atom_idx in range(1, natoms + 1):
            line = f.readline()
            if not line:
                raise ValueError("incomplete first XYZ frame")
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"invalid XYZ atom line: {line.strip()}")
            atoms.append(
                {
                    "index": atom_idx,
                    "model_index": atom_idx - 1,
                    "element": normalize_element_symbol(parts[0]),
                    "x": float(parts[1]),
                    "y": float(parts[2]),
                    "z": float(parts[3]),
                }
            )
    return comment, atoms


def parse_orca_cartesian_coordinates(fname):
    """
    Parse the CARTESIAN COORDINATES (ANGSTROEM) block from an ORCA output.

    ORCA population tables are indexed from zero, so the returned atom index is
    zero-based to match the charge/spin analysis selections.
    """
    atoms = []
    in_block = False
    saw_separator = False

    with open(fname, "r") as f:
        for line in f:
            stripped = line.strip()
            if stripped == "CARTESIAN COORDINATES (ANGSTROEM)":
                in_block = True
                saw_separator = False
                atoms = []
                continue
            if not in_block:
                continue
            if stripped.startswith("---"):
                saw_separator = True
                continue
            if not saw_separator:
                continue
            if not stripped:
                break

            parts = stripped.split()
            if len(parts) < 4:
                break
            try:
                x, y, z = map(float, parts[1:4])
            except ValueError:
                break
            model_index = len(atoms)
            atoms.append(
                {
                    "index": model_index,
                    "model_index": model_index,
                    "element": normalize_element_symbol(parts[0]),
                    "x": x,
                    "y": y,
                    "z": z,
                }
            )

    if not atoms:
        raise ValueError(f"no CARTESIAN COORDINATES (ANGSTROEM) block found in {fname}")
    return f"{os.path.basename(fname)} | ORCA Cartesian coordinates | indices 0-based", atoms


def find_default_xyz_for_spin_viewer():
    """
    Return a likely XYZ file for the spin-localization viewer.
    """
    candidates = (
        "qm_completo.xyz",
        "qm.xyz",
        "QM.xyz",
        "molecule.xyz",
        "mol.xyz",
    )
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    xyz_files = sorted(glob.glob("*.xyz"))
    return xyz_files[0] if xyz_files else None


def open_html_viewer(path):
    """
    Try to open a generated HTML viewer in the default browser.
    """
    try:
        opened = webbrowser.open(Path(path).resolve().as_uri(), new=2)
    except Exception as exc:
        print(f"[WARN] Could not open viewer automatically: {exc}")
        return
    if not opened:
        print(f"[WARN] Could not open viewer automatically. Open '{path}' manually.")


def write_spin_localization_viewer(
    xyz_path,
    output_path,
    highlighted_atom_ids,
    atom_types,
    avg_fraction_by_atom,
):
    """
    Write a py3Dmol-based HTML viewer highlighting localized-spin atoms.
    """
    comment, atoms = parse_first_xyz_frame(xyz_path)
    title = f"{os.path.basename(xyz_path)} | XYZ first frame | indices 1-based"
    write_spin_localization_viewer_from_atoms(
        title,
        atoms,
        output_path,
        highlighted_atom_ids,
        atom_types,
        avg_fraction_by_atom,
    )


def write_orca_spin_localization_viewer(
    orca_output_path,
    output_path,
    highlighted_atom_ids,
    atom_types,
    avg_fraction_by_atom,
):
    """
    Write a spin-localization viewer from an ORCA output geometry block.
    """
    title, atoms = parse_orca_cartesian_coordinates(orca_output_path)
    write_spin_localization_viewer_from_atoms(
        title,
        atoms,
        output_path,
        highlighted_atom_ids,
        atom_types,
        avg_fraction_by_atom,
    )


def find_orca_geometry_file_for_viewer(orca_files):
    """
    Return one ORCA output containing a Cartesian coordinate block.

    The viewer only needs a representative snapshot, so prefer the first file
    in the sorted ORCA list and only scan further if that one lacks geometry.
    """
    for fname in orca_files:
        try:
            parse_orca_cartesian_coordinates(fname)
        except Exception:
            continue
        return fname
    return None


def write_spin_localization_viewer_from_atoms(
    title,
    atoms,
    output_path,
    highlighted_atom_ids,
    atom_types,
    avg_fraction_by_atom,
):
    """
    Write a py3Dmol-based HTML viewer from parsed atoms.
    """
    import py3Dmol

    highlighted = set(highlighted_atom_ids)

    view = py3Dmol.view(width=980, height=720)
    view.addModel(atoms_to_mol_block(title, atoms), "mol")
    view.setStyle({"stick": {"radius": 0.16}, "sphere": {"scale": 0.28}})

    for atom in atoms:
        aid = atom["index"]
        if aid not in highlighted:
            continue
        frac = 100.0 * float(avg_fraction_by_atom.get(aid, 0.0))
        element = atom_types.get(aid, atom["element"])
        view.addLabel(
            f"{aid} {element} {frac:.1f}%",
            {
                "position": {"x": atom["x"], "y": atom["y"], "z": atom["z"]},
                "fontColor": "black",
                "backgroundColor": "white",
                "backgroundOpacity": 0.9,
                "fontSize": 15,
                "inFront": True,
            },
        )

    view.zoomTo()
    body = view._make_html()
    html = (
        "<!doctype html>\n"
        "<html>\n"
        "<head><meta charset=\"utf-8\"><title>Spin localization viewer</title></head>\n"
        "<body>\n"
        f"<h3 style=\"font-family: sans-serif; margin: 8px 0;\">Spin localization viewer | {title}</h3>\n"
        f"{body}\n"
        "</body>\n"
        "</html>\n"
    )
    with open(output_path, "w") as out:
        out.write(html)
    print(f"[OK] Spin-localization viewer saved to '{output_path}'.")


def atoms_to_mol_block(title, atoms):
    """
    Build a V2000 MOL block with explicit bonds for py3Dmol.
    """
    bonds = infer_bonds_from_distances(atoms)
    lines = [
        str(title)[:80],
        "TolkienTools",
        "",
        f"{len(atoms):3d}{len(bonds):3d}  0  0  0  0            999 V2000",
    ]
    for atom in atoms:
        lines.append(
            f"{atom['x']:10.4f}{atom['y']:10.4f}{atom['z']:10.4f} "
            f"{atom['element'][:3]:<3s} 0  0  0  0  0  0  0  0  0  0  0  0"
        )
    for i, j in bonds:
        lines.append(f"{i + 1:3d}{j + 1:3d}  1  0  0  0  0")
    lines.append("M  END")
    return "\n".join(lines) + "\n"


def infer_bonds_from_distances(atoms):
    """
    Infer conservative single bonds from Cartesian distances.
    """
    covalent_radii = {
        "H": 0.31,
        "C": 0.76,
        "N": 0.71,
        "O": 0.66,
        "S": 1.05,
        "P": 1.07,
        "Fe": 1.24,
        "Ru": 1.46,
        "Cu": 1.32,
        "Zn": 1.22,
    }
    max_coordination = {
        "H": 1,
        "C": 4,
        "N": 4,
        "O": 2,
        "S": 6,
        "P": 5,
        "Fe": 6,
        "Ru": 6,
        "Cu": 5,
        "Zn": 5,
    }
    candidates = []
    for i, atom_i in enumerate(atoms):
        elem_i = normalize_element_symbol(atom_i["element"])
        for j in range(i + 1, len(atoms)):
            atom_j = atoms[j]
            elem_j = normalize_element_symbol(atom_j["element"])
            dist = distance_between_atoms(atom_i, atom_j)
            if is_plausible_bond(elem_i, elem_j, dist, covalent_radii):
                candidates.append((dist, i, j))

    bonds = []
    degree = [0] * len(atoms)
    for _dist, i, j in sorted(candidates):
        elem_i = normalize_element_symbol(atoms[i]["element"])
        elem_j = normalize_element_symbol(atoms[j]["element"])
        if degree[i] >= max_coordination.get(elem_i, 4):
            continue
        if degree[j] >= max_coordination.get(elem_j, 4):
            continue
        bonds.append((i, j))
        degree[i] += 1
        degree[j] += 1
    return bonds


def normalize_element_symbol(element):
    """
    Normalize element capitalization for simple bonding/color rules.
    """
    element = str(element).strip()
    if not element:
        return "X"
    atomic_number_to_symbol = {
        1: "H",
        2: "He",
        3: "Li",
        4: "Be",
        5: "B",
        6: "C",
        7: "N",
        8: "O",
        9: "F",
        10: "Ne",
        11: "Na",
        12: "Mg",
        13: "Al",
        14: "Si",
        15: "P",
        16: "S",
        17: "Cl",
        18: "Ar",
        19: "K",
        20: "Ca",
        21: "Sc",
        22: "Ti",
        23: "V",
        24: "Cr",
        25: "Mn",
        26: "Fe",
        27: "Co",
        28: "Ni",
        29: "Cu",
        30: "Zn",
        44: "Ru",
    }
    try:
        atomic_number = int(element)
    except ValueError:
        atomic_number = None
    if atomic_number is not None:
        return atomic_number_to_symbol.get(atomic_number, element)
    return element[0].upper() + element[1:].lower()


def distance_between_atoms(atom_i, atom_j):
    dx = atom_i["x"] - atom_j["x"]
    dy = atom_i["y"] - atom_j["y"]
    dz = atom_i["z"] - atom_j["z"]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def is_plausible_bond(elem_i, elem_j, dist, covalent_radii):
    if dist < 0.35:
        return False
    if elem_i == "H" and elem_j == "H":
        return dist <= 0.85

    metal_elements = {"Fe", "Ru", "Cu", "Zn"}
    if elem_i in metal_elements or elem_j in metal_elements:
        ligand = elem_j if elem_i in metal_elements else elem_i
        if ligand == "H":
            return False
        return dist <= 2.65

    ri = covalent_radii.get(elem_i, 0.77)
    rj = covalent_radii.get(elem_j, 0.77)
    return dist <= 1.20 * (ri + rj) + 0.20
