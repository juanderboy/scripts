#!/usr/bin/env python3
"""File readers, mergers and output writers for charge/spin populations.

It handles fragmented LIO population files, parsed frame tables and combined
time-series exports for individual atoms or grouped entities.
"""

import glob
import os
import re

import numpy as np

from charge_spin_common import sanitize_output_token


def get_sorted_files(prefix):
    """
    Find files matching prefix_*.dat and sort them numerically.
    Example: prefix='mq' -> mq_0.dat, mq_1.dat, ...
    """
    pattern = f"{prefix}_*.dat"
    files = []
    for fname in glob.glob(pattern):
        m = re.match(rf"{prefix}_(\d+)\.dat$", fname)
        if m:
            idx = int(m.group(1))
            files.append((idx, fname))

    files.sort(key=lambda x: x[0])
    return [f for _, f in files]


def merge_files(files, outname, skip_initial_population_block=False):
    """
    Concatenate a list of files into outname.
    """
    with open(outname, "w") as out:
        for fname in files:
            population_blocks = 0
            with open(fname, "r") as f:
                for line in f:
                    if skip_initial_population_block:
                        stripped = line.strip()
                        if stripped.startswith("#") and "Population Analysis" in stripped:
                            population_blocks += 1
                        if population_blocks < 2:
                            continue
                    out.write(line)
    print(f"[OK] Files {files[0]} ... {files[-1]} merged into '{outname}'.")


def parse_frame_data(fullfile, dt_ps, atom_ids, kind, header_start, spin_sign=1.0):
    """
    Parsea un archivo full y retorna datos por frame.

    Returns
    -------
    times : np.ndarray
    values : np.ndarray
        shape: (n_frames, n_atoms)
    frame_totals : np.ndarray
        Valor total informado al cierre de cada frame.
    """
    atom_ids = list(atom_ids)
    data = []
    frame_totals = []
    current_vals = {aid: None for aid in atom_ids}
    frame_index = -1
    inside_frame = False

    with open(fullfile, "r") as f:
        for line in f:
            stripped = line.strip()

            if stripped.startswith(header_start):
                frame_index += 1
                inside_frame = True
                current_vals = {aid: None for aid in atom_ids}
                continue

            if not inside_frame:
                continue

            if "Total Charge" in stripped:
                time_ps = frame_index * dt_ps
                row = [time_ps]
                for aid in atom_ids:
                    v = current_vals.get(aid, None)
                    if v is None:
                        v = float("nan")
                    row.append(v)
                data.append(row)
                try:
                    total_val = float(stripped.split("=")[-1])
                except ValueError:
                    total_val = float("nan")
                frame_totals.append(total_val)
                inside_frame = False
                continue

            if not stripped or stripped.startswith("#"):
                continue

            parts = stripped.split()
            if len(parts) >= 3 and parts[0].isdigit():
                atom_idx = int(parts[0])
                try:
                    value = float(parts[2])
                except ValueError:
                    continue
                if kind == "spin":
                    value = spin_sign * value
                if atom_idx in current_vals:
                    current_vals[atom_idx] = value

    if not data:
        return np.array([]), np.empty((0, len(atom_ids))), np.array([])

    data = np.array(data, dtype=float)
    return data[:, 0], data[:, 1:], np.asarray(frame_totals, dtype=float)


def get_atom_list_from_full(fullfile, header_start, lio=False):
    """
    Lee el primer bloque de un archivo full y retorna lista de (atom_id, atom_type).
    """
    atoms = []
    inside_frame = False
    z_to_symbol = {
        1: "H", 2: "He",
        3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne",
        11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 18: "Ar",
        19: "K", 20: "Ca", 21: "Sc", 22: "Ti", 23: "V", 24: "Cr", 25: "Mn", 26: "Fe", 27: "Co", 28: "Ni", 29: "Cu", 30: "Zn",
        31: "Ga", 32: "Ge", 33: "As", 34: "Se", 35: "Br", 36: "Kr",
        37: "Rb", 38: "Sr", 39: "Y", 40: "Zr", 41: "Nb", 42: "Mo", 43: "Tc", 44: "Ru", 45: "Rh", 46: "Pd", 47: "Ag", 48: "Cd",
        49: "In", 50: "Sn", 51: "Sb", 52: "Te", 53: "I", 54: "Xe",
        55: "Cs", 56: "Ba", 57: "La", 58: "Ce", 59: "Pr", 60: "Nd", 61: "Pm", 62: "Sm", 63: "Eu", 64: "Gd", 65: "Tb", 66: "Dy", 67: "Ho", 68: "Er", 69: "Tm", 70: "Yb", 71: "Lu",
        72: "Hf", 73: "Ta", 74: "W", 75: "Re", 76: "Os", 77: "Ir", 78: "Pt", 79: "Au", 80: "Hg",
        81: "Tl", 82: "Pb", 83: "Bi", 84: "Po", 85: "At", 86: "Rn",
        87: "Fr", 88: "Ra", 89: "Ac", 90: "Th", 91: "Pa", 92: "U", 93: "Np", 94: "Pu", 95: "Am", 96: "Cm", 97: "Bk", 98: "Cf", 99: "Es", 100: "Fm", 101: "Md", 102: "No", 103: "Lr",
        104: "Rf", 105: "Db", 106: "Sg", 107: "Bh", 108: "Hs", 109: "Mt", 110: "Ds", 111: "Rg", 112: "Cn",
        113: "Nh", 114: "Fl", 115: "Mc", 116: "Lv", 117: "Ts", 118: "Og"
    }
    with open(fullfile, "r") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith(header_start):
                inside_frame = True
                continue
            if not inside_frame:
                continue
            if "Total Charge" in stripped:
                break
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) >= 3 and parts[0].isdigit():
                atom_idx = int(parts[0])
                atom_type = parts[1]
                if lio and atom_type.isdigit():
                    z = int(atom_type)
                    atom_type = z_to_symbol.get(z, atom_type)
                atoms.append((atom_idx, atom_type))
    return atoms


def load_atom_timeseries_file(fname):
    """
    Read an atom_<id>_..._timeseries.dat file and return charge and spin arrays.
    If the spin column does not exist, return an empty spin array.

    Returns
    -------
    charges : np.ndarray
    spins : np.ndarray
    spin_label : str | None
        Name of the third file column, for example 'spin' or
        'spin_fraction'. If there is no third column, return None.
    """
    charges = []
    spins = []
    spin_label = None

    with open(fname, "r") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("#"):
                header_parts = stripped[1:].split()
                if len(header_parts) >= 3:
                    spin_label = header_parts[2]
                continue
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) >= 2:
                charges.append(float(parts[1]))
            if len(parts) >= 3:
                spins.append(float(parts[2]))

    return np.asarray(charges, dtype=float), np.asarray(spins, dtype=float), spin_label


def write_combined_entity_timeseries(
    entity_ids,
    times,
    per_entity_charge,
    per_entity_spin,
    spin_column_label,
    suffix,
    actor_config=None
):
    """
    Write per-entity combined time-series files, including an optional grouped actor.
    """
    actor_id = actor_config["id"] if actor_config is not None else None
    for entity_id in entity_ids:
        q_vals = np.asarray(per_entity_charge.get(entity_id, []), dtype=float)

        if times.size != q_vals.size:
            n = min(times.size, q_vals.size)
            t_use = times[:n]
            q_use = q_vals[:n]
        else:
            t_use = times
            q_use = q_vals

        if entity_id == actor_id:
            atom_out = f"actor_{sanitize_output_token(actor_config['label'])}_{suffix}_timeseries.dat"
            entity_label = actor_config["label"]
        else:
            atom_out = f"atom_{entity_id}_{suffix}_timeseries.dat"
            entity_label = f"atom {entity_id}"

        with open(atom_out, "w") as out:
            s_array = np.asarray(per_entity_spin.get(entity_id, []), dtype=float)
            if s_array.size > 0:
                if s_array.size != t_use.size:
                    n = min(t_use.size, s_array.size)
                    t_use = t_use[:n]
                    q_use = q_use[:n]
                    s_array = s_array[:n]
                out.write(f"# time_ps  charge  {spin_column_label}\n")
                for t, q, s in zip(t_use, q_use, s_array):
                    out.write(f"{t: .7f} {q: .7f} {s: .7f}\n")
            else:
                out.write("# time_ps  charge\n")
                for t, q in zip(t_use, q_use):
                    out.write(f"{t: .7f} {q: .7f}\n")

        print(f"[OK] Combined time series for {entity_label} written to '{atom_out}'.")
