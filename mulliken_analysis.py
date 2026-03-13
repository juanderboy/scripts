#!/usr/bin/env python3
import glob
import os
import re
import sys
import math

import numpy as np
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


# ==========================
# General utilities
# ==========================

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


def prompt_numbered_choice(title, options, default_idx=None):
    """
    Show a numbered menu and return the value associated with the chosen option.
    options: list of tuples (label, value)
    default_idx: 0-based index of the default option, or None
    """
    print(title)
    for idx, (label, _value) in enumerate(options, start=1):
        default_tag = " [default]" if default_idx is not None and idx - 1 == default_idx else ""
        print(f"  {idx}. {label}{default_tag}")

    prompt = "Select an option by number"
    if default_idx is not None:
        prompt += f" (Enter = {default_idx + 1})"
    prompt += ": "

    choice = input(prompt).strip()
    if choice == "":
        if default_idx is None:
            print("Error: you must select an option.")
            sys.exit(1)
        return options[default_idx][1]

    if not choice.isdigit():
        print("Error: you must enter the number of an option.")
        sys.exit(1)

    selected = int(choice)
    if selected < 1 or selected > len(options):
        print("Error: option out of range.")
        sys.exit(1)

    return options[selected - 1][1]

def get_sorted_orca_files(prefix=None):
    """
    Find ORCA files named <prefix>_N.out or <prefix>_N.dat
    and sort them by frame number N.

    If prefix is None or "", autodetect any prefix.
    """
    normalized_prefix = prefix.strip() if prefix else None
    patterns = [f"{normalized_prefix}_*.out", f"{normalized_prefix}_*.dat"] if normalized_prefix else ["*_*.out", "*_*.dat"]
    files = {}
    detected_prefixes = set()

    if normalized_prefix:
        regex = re.compile(rf"^{re.escape(normalized_prefix)}_(\d+)\.(out|dat)$")
    else:
        regex = re.compile(r"^(.+?)_(\d+)\.(out|dat)$")

    for pattern in patterns:
        for fname in glob.glob(pattern):
            m = regex.match(fname)
            if not m:
                continue

            if normalized_prefix:
                idx = int(m.group(1))
                ext = m.group(2)
            else:
                detected_prefix = m.group(1)
                idx = int(m.group(2))
                ext = m.group(3)
                detected_prefixes.add(detected_prefix)

            # Prefer .out if duplicates are present
            if idx not in files or ext == "out":
                files[idx] = fname

    if not normalized_prefix and len(detected_prefixes) > 1:
        print("Error: multiple ORCA prefixes were detected in the current directory:")
        for detected_prefix in sorted(detected_prefixes):
            print(f"  {detected_prefix}")
        print("Please enter the desired prefix explicitly to avoid mixing files.")
        sys.exit(1)

    return [files[k] for k in sorted(files.keys())]


def merge_files(files, outname):
    """
    Concatenate a list of files into outname.
    """
    with open(outname, "w") as out:
        for fname in files:
            with open(fname, "r") as f:
                for line in f:
                    out.write(line)
    print(f"[OK] Files {files[0]} ... {files[-1]} merged into '{outname}'.")


def print_welcome_banner():
    """
    Print the program welcome banner.
    """
    print("")
    print("=" * 78)
    print("                  CHARGE AND SPIN ENSEMBLE ANALYZER")
    print("=" * 78)
    print("        Statistical analysis of charges and spin populations")
    print("        Time series, KDE modes, histograms, and comparisons")
    print("")
    print("                 Vibecoded by Tolkien, 2026")
    print("=" * 78)
    print("")


# ==========================
# KDE and mode analysis
# ==========================

def analyze_modes_kde(vals, n_grid=200, prominence_factor=0.05):
    """
    Compute a 1D KDE and search for maxima (modes) in the density.

    Parameters
    ----------
    vals : array-like
        Data (charges or spins) for one atom.
    n_grid : int
        Number of grid points used to evaluate the KDE.
    prominence_factor : float
        Fraction of the maximum height used as the minimum prominence
        for peak detection.

    Returns
    -------
    xs : np.ndarray
        Grid values where the density was evaluated.
    dens : np.ndarray
        Valores de la KDE en xs.
    peak_positions : np.ndarray
        Posiciones (en el eje x) de los picos detectados.
    peak_heights : np.ndarray
        Altura de la densidad en cada pico.
    """
    vals = np.asarray(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    vmin, vmax = vals.min(), vals.max()
    if vmin == vmax:
        xs = np.array([vmin])
        dens = np.array([1.0])
        return xs, dens, np.array([vmin]), np.array([1.0])

    kde = gaussian_kde(vals)
    xs = np.linspace(vmin, vmax, n_grid)
    dens = kde(xs)

    prom = prominence_factor * dens.max()
    peaks, props = find_peaks(dens, prominence=prom)

    if peaks.size == 0:
        idx_max = np.argmax(dens)
        return xs, dens, np.array([xs[idx_max]]), np.array([dens[idx_max]])

    peak_positions = xs[peaks]
    peak_heights = dens[peaks]
    return xs, dens, peak_positions, peak_heights


def get_histogram_edges(vals, bins_spec=50, value_range=None, min_bins=8, max_bins=80):
    """
    Determine histogram bin edges from a fixed or automatic specification.

    bins_spec puede ser:
      - int: fixed number of bins
      - 'fd': regla de Freedman-Diaconis
      - 'sturges': regla de Sturges
      - 'auto': equivalente a numpy ('max' entre sturges y fd)
    """
    vals = np.asarray(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return np.array([])

    if value_range is None:
        vmin = float(vals.min())
        vmax = float(vals.max())
    else:
        vmin, vmax = map(float, value_range)

    if vmin == vmax:
        return np.array([vmin, vmax])

    if isinstance(bins_spec, int):
        nbins = max(1, bins_spec)
        return np.linspace(vmin, vmax, nbins + 1)

    edges = np.histogram_bin_edges(vals, bins=bins_spec, range=(vmin, vmax))
    nbins = max(1, len(edges) - 1)
    nbins = min(max(nbins, min_bins), max_bins)
    return np.linspace(vmin, vmax, nbins + 1)


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


def normalize_selected_spin_values(values, zero_tol=1e-12):
    """
    Normalize spins frame by frame using the sum over the selected atoms.

    Para cada snapshot:
      spin_frac_i = spin_i / sum_j(spin_j)

    If any selected atom is missing or the sum is ~0, that frame is marked as NaN.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values.copy()

    normalized = values.copy()
    missing_mask = np.isnan(values).any(axis=1)
    selected_totals = np.nansum(values, axis=1)
    invalid_total_mask = np.abs(selected_totals) <= zero_tol
    invalid_mask = missing_mask | invalid_total_mask

    valid_mask = ~invalid_mask
    if np.any(valid_mask):
        normalized[valid_mask, :] = values[valid_mask, :] / selected_totals[valid_mask, np.newaxis]
    if np.any(invalid_mask):
        normalized[invalid_mask, :] = np.nan

    return normalized


# ==========================
# Reading merged files and statistics
# ==========================

def build_timeseries_and_stats(
    fullfile,
    dt_ps,
    atom_ids,
    kind,
    header_start,
    ts_outname,
    avg_outname,
    hist_prefix,
    modes_outname,
    nbins_hist=50,
    spin_sign=1.0,
    keep_mask=None,
    normalize_spin_fraction=False
):
    """
    Analyze a merged full file (mq_full.dat or ms_full.dat).

    kind: 'charge' or 'spin'
    header_start:
        - "# Mulliken Population Analysis"  (charges)
        - "# Mulliken Spin Population Analysis" (spin)

    Returns
    -------
    times : np.ndarray
        Time vector (ps) for each frame.
    per_atom_values : dict
        {atom_id: np.ndarray of valid values (without NaN)}.
    hist_data_for_plot : dict
        {atom_id: dict with histogram and KDE information for plotting}.
    """

    atom_ids = list(atom_ids)
    times, values, _frame_totals = parse_frame_data(
        fullfile,
        dt_ps,
        atom_ids,
        kind,
        header_start,
        spin_sign=spin_sign
    )

    if kind == "spin" and normalize_spin_fraction:
        values = normalize_selected_spin_values(values)

    if keep_mask is not None:
        keep_mask = np.asarray(keep_mask, dtype=bool)
        if keep_mask.size != times.size:
            print(
                f"[WARN] An incompatible snapshot mask was ignored for '{fullfile}': "
                f"mask={keep_mask.size}, frames={times.size}."
            )
        else:
            times = times[keep_mask]
            values = values[keep_mask, :]

    data = np.column_stack((times, values)) if times.size > 0 else np.empty((0, len(atom_ids) + 1))

    # Time series
    with open(ts_outname, "w") as out:
        if kind == "spin" and normalize_spin_fraction:
            header = ["time_ps"] + [f"spin_fraction_atom_{aid}" for aid in atom_ids]
        else:
            header = ["time_ps"] + [f"{kind}_atom_{aid}" for aid in atom_ids]
        out.write("# " + " ".join(header) + "\n")
        for row in data:
            formatted = []
            for val in row:
                if isinstance(val, float):
                    formatted.append(f"{val: .7f}")
                else:
                    formatted.append(str(val))
            out.write(" ".join(formatted) + "\n")

    print(f"[OK] {kind.capitalize()} time series written to '{ts_outname}'.")

    if data.size == 0:
        times = np.array([])
        values = np.empty((0, len(atom_ids)))
    else:
        data = np.array(data, dtype=float)
        times = data[:, 0]
        values = data[:, 1:]  # shape: (n_frames, n_atoms)

    per_atom_values = {aid: [] for aid in atom_ids}
    for i, aid in enumerate(atom_ids):
        col = values[:, i]
        mask = ~np.isnan(col)
        per_atom_values[aid] = col[mask]

    # Averages
    with open(avg_outname, "w") as out:
        if kind == "spin" and normalize_spin_fraction:
            out.write("# Spin averages as fractions of the total spin of the selected atoms\n")
        else:
            out.write(f"# Average {kind} values\n")
        out.write("# atom_id  avg_value\n")
        for i, aid in enumerate(atom_ids):
            vals = per_atom_values[aid]
            if vals.size > 0:
                avg = vals.mean()
                out.write(f"{aid:4d}  {avg: .7f}\n")
            else:
                out.write(f"{aid:4d}  nan\n")

    print(f"[OK] Average {kind} values written to '{avg_outname}'.")

    # Histograms + KDE + mode analysis
    modes_summary = []
    hist_data_for_plot = {}

    for i, aid in enumerate(atom_ids):
        vals = per_atom_values[aid]
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            print(f"[WARN] No valid {kind} data found for atom {aid}.")
            continue

        vmin, vmax = vals.min(), vals.max()

        if vmin == vmax:
            # Degenerate case
            xs = np.array([vmin])
            dens = np.array([1.0])
            peaks = np.array([vmin])
            peak_heights = np.array([1.0])

            hist_outname = f"{hist_prefix}_atom_{aid}.dat"
            with open(hist_outname, "w") as out:
                out.write("# Histogram: bin_center  count\n")
                out.write(f"{vmin: .7f} {vals.size}\n")
                out.write("\n# KDE: x  density\n")
                out.write(f"{vmin: .7f} {dens[0]: .7e}\n")
            print(f"[OK] Trivial {kind} histogram for atom {aid} written to '{hist_outname}'.")

            modes_summary.append((aid, peaks, peak_heights))
            hist_data_for_plot[aid] = {
                "bin_centers": np.array([vmin]),
                "counts_norm": np.array([1.0]),
                "xs": xs,
                "dens": dens,
                "peaks": peaks,
                "axis_label": "Spin fraction" if kind == "spin" and normalize_spin_fraction else None
            }
            continue

        bin_edges = get_histogram_edges(vals, bins_spec=nbins_hist, value_range=(vmin, vmax))
        counts, bin_edges = np.histogram(vals, bins=bin_edges)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        counts_norm = counts / counts.sum()

        # KDE for this value set
        xs, dens, peak_positions, peak_heights = analyze_modes_kde(vals)

        hist_outname = f"{hist_prefix}_atom_{aid}.dat"
        with open(hist_outname, "w") as out:
            out.write("# Histogram: bin_center  count\n")
            for c, center in zip(counts, bin_centers):
                out.write(f"{center: .7f} {c:d}\n")
            # Also save the KDE curve to simplify post-processing
            out.write("\n# KDE: x  density\n")
            if xs.size > 0 and dens.size > 0:
                for x_val, d_val in zip(xs, dens):
                    out.write(f"{x_val: .7f} {d_val: .7e}\n")

        print(f"[OK] {kind.capitalize()} histogram for atom {aid} written to '{hist_outname}'.")

        modes_summary.append((aid, peak_positions, peak_heights))

        if len(peak_positions) == 1:
            print(f"    Atom {aid}: unimodal {kind} distribution. Peak ≈ {peak_positions[0]: .4f}")
        else:
            print(f"    Atom {aid}: {len(peak_positions)} modes detected in {kind}.")
            for k, mu in enumerate(peak_positions):
                print(f"        Mode {k+1}: {kind} ≈ {mu: .4f}")

        hist_data_for_plot[aid] = {
            "bin_centers": bin_centers,
            "counts_norm": counts_norm,
            "xs": xs,
            "dens": dens,
            "peaks": peak_positions,
            "axis_label": "Spin fraction" if kind == "spin" and normalize_spin_fraction else None
        }

    # Mode summary
    with open(modes_outname, "w") as out:
        out.write(f"# atom_id  n_modes  peak_positions_{kind}(approx)\n")
        for aid, peaks, heights in modes_summary:
            if len(peaks) == 0:
                out.write(f"{aid:4d}  0\n")
            else:
                peaks_str = " ".join(f"{mu: .6f}" for mu in peaks)
                out.write(f"{aid:4d}  {len(peaks):2d}  {peaks_str}\n")

    print(f"[OK] {kind.capitalize()} mode summary written to '{modes_outname}'.")

    return times, per_atom_values, hist_data_for_plot


def apply_keep_mask_or_warn(keep_mask, n_frames, analysis_label):
    """
    Validate a snapshot mask against a given analysis.

    If the frame count does not match, return None and emit a warning.
    """
    if keep_mask is None:
        return None

    keep_mask = np.asarray(keep_mask, dtype=bool)
    if keep_mask.size != n_frames:
        print(
            f"[WARN] Snapshot filtering was not applied to {analysis_label}: "
            f"the mask has {keep_mask.size} frames and the analysis has {n_frames}."
        )
        return None
    return keep_mask


# ==========================
# Utility: atom list
# ==========================

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


# ==========================
# ORCA: parseo de Mulliken
# ==========================

def extract_orca_population_block(fname, header_line):
    """
    Extract a charge/spin table (Mulliken or Loewdin) from an ORCA output.
    Return a list of tuples: (atom_idx, element, charge, spin)
    """
    data = []
    in_section = False
    with open(fname, "r") as f:
        for line in f:
            if header_line in line:
                in_section = True
                continue
            if not in_section:
                continue

            stripped = line.strip()
            if not stripped:
                break
            if stripped.startswith("Sum of atomic charges"):
                break
            if stripped.startswith("MULLIKEN REDUCED"):
                break
            if stripped.startswith("---"):
                continue

            m = re.match(r"^\s*(\d+)\s+([A-Za-z]+)\s*:\s*([-\d\.Ee+]+)\s+([-\d\.Ee+]+)", line)
            if m:
                atom_idx = int(m.group(1))
                element = m.group(2)
                charge = float(m.group(3))
                spin = float(m.group(4))
                data.append((atom_idx, element, charge, spin))

    return data


def extract_orca_charge_only_block(fname, header_line):
    """
    Extract a charge table without a spin column from an ORCA output.
    Return a list of tuples: (atom_idx, element, charge)
    """
    data = []
    in_section = False
    with open(fname, "r") as f:
        for line in f:
            if header_line in line:
                in_section = True
                continue
            if not in_section:
                continue

            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("Total charge"):
                break
            if stripped.startswith("CHELPG charges calculated"):
                break
            if stripped.startswith("---"):
                continue

            m = re.match(r"^\s*(\d+)\s+([A-Za-z]+)\s*:\s*([-\d\.Ee+]+)\s*$", line)
            if m:
                atom_idx = int(m.group(1))
                element = m.group(2)
                charge = float(m.group(3))
                data.append((atom_idx, element, charge))

    return data


def build_orca_full_files(
    orca_files,
    out_charge,
    out_spin,
    charge_label,
    charge_header_line,
    spin_label=None,
    spin_header_line=None
):
    """
    Build merged charge and spin files from multiple ORCA <prefix>_N.out/.dat files
    using the format expected by build_timeseries_and_stats.
    """
    if not orca_files:
        print("Error: no ORCA files matching '<prefix>_N.out' or '<prefix>_N.dat' were found.")
        sys.exit(1)

    with open(out_charge, "w") as q_out, open(out_spin, "w") as s_out:
        for fname in orca_files:
            if spin_header_line is None:
                block = extract_orca_population_block(fname, charge_header_line)
                if not block:
                    print(f"[WARN] {charge_label} table not found in '{fname}'.")
                    continue
                rows = block
                spin_header_label = spin_label or charge_label
            else:
                charge_block = extract_orca_charge_only_block(fname, charge_header_line)
                spin_block = extract_orca_population_block(fname, spin_header_line)
                if not charge_block:
                    print(f"[WARN] {charge_label} table not found in '{fname}'.")
                    continue
                if not spin_block:
                    spin_source = spin_label or "spin"
                    print(f"[WARN] {spin_source} spin table not found in '{fname}'.")
                    continue

                spin_by_atom = {
                    atom_idx: (element, spin)
                    for atom_idx, element, _charge, spin in spin_block
                }
                rows = []
                missing_spin = False
                for atom_idx, element, charge in charge_block:
                    spin_info = spin_by_atom.get(atom_idx)
                    if spin_info is None:
                        print(
                            f"[WARN] Missing spin for atom {atom_idx} while combining "
                            f"{charge_label} + {spin_label or 'spin'} in '{fname}'."
                        )
                        missing_spin = True
                        break
                    spin_element, spin = spin_info
                    rows.append((atom_idx, spin_element or element, charge, spin))
                if missing_spin:
                    continue
                spin_header_label = spin_label or charge_label

            q_out.write(f"# {charge_label} Population Analysis\n")
            q_out.write("# Atom   Type   Population\n")
            s_out.write(f"# {spin_header_label} Spin Population Analysis\n")
            s_out.write("# Atom   Type   Population\n")

            sum_q = 0.0
            sum_s = 0.0
            for atom_idx, element, charge, spin in rows:
                q_out.write(f"{atom_idx:4d} {element:>3s} {charge: .7f}\n")
                s_out.write(f"{atom_idx:4d} {element:>3s} {spin: .7f}\n")
                sum_q += charge
                sum_s += spin

            # Closing line compatible with the parser
            q_out.write(f"  Total Charge = {sum_q: .7f}\n\n")
            s_out.write(f"  Total Charge = {sum_s: .7f}\n\n")

    spin_print_label = spin_label or charge_label
    print(f"[OK] ORCA charge file ({charge_label}) merged into '{out_charge}'.")
    print(f"[OK] ORCA spin file ({spin_print_label}) merged into '{out_spin}'.")


# ==========================
# Combined q/s figure
# ==========================

def make_combined_hist_figure(
    atom_ids,
    hist_charge,
    hist_spin,
    atom_labels=None,
    charge_axis_label="Mulliken charge",
    fig_outname="qs_histograms.png"
):
    """
    Generate a combined figure with charge and spin histograms.
    One row is created per atom:
      - column 1: charge (if available)
      - column 2: spin (if available)
    Peaks are marked and labeled with their value.
    """
    has_charge = bool(hist_charge)
    has_spin = bool(hist_spin)
    if not has_charge and not has_spin:
        print("[NOTICE] Combined figure was not generated (no charge or spin data available).")
        return

    n_atoms = len(atom_ids)
    ncols = 1 + int(has_spin)  # 1 (q) or 2 (q+s)
    nrows = n_atoms

    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                             figsize=(4*ncols, 3*nrows))
    axes = axes  # 2D matrix [row, col]

    for i, aid in enumerate(atom_ids):
        # CHARGE (column 0)
        ax_q = axes[i, 0]
        if aid in hist_charge:
            info = hist_charge[aid]
            bin_centers = info["bin_centers"]
            counts_norm = info["counts_norm"]
            xs = info["xs"]
            dens = info["dens"]
            peaks = info["peaks"]

            if len(bin_centers) > 1:
                width = bin_centers[1] - bin_centers[0]
            else:
                width = 0.1

            ax_q.bar(bin_centers, counts_norm, width=width, alpha=0.5, label="Hist (q)")
            if dens.size > 0:
                ax_q.plot(xs, dens / dens.max() * counts_norm.max(), label="KDE (q)")
            for mu in peaks:
                ax_q.axvline(mu, linestyle="--")
                ymax = counts_norm.max()
                ax_q.text(mu, ymax*1.02, f"{mu:.2f}",
                          rotation=90, ha="center", va="bottom", fontsize=7)
            label = atom_labels.get(aid, str(aid)) if atom_labels is not None else str(aid)
            ax_q.set_ylabel(f"Atom {label}")
            ax_q.set_xlabel(charge_axis_label)
            ax_q.legend(fontsize=8)
        else:
            ax_q.text(0.5, 0.5, "No q data", transform=ax_q.transAxes,
                      ha="center", va="center")
            ax_q.set_axis_off()

        # SPIN (column 1, if present)
        if has_spin:
            ax_s = axes[i, 1]
            if aid in hist_spin:
                info = hist_spin[aid]
                bin_centers = info["bin_centers"]
                counts_norm = info["counts_norm"]
                xs = info["xs"]
                dens = info["dens"]
                peaks = info["peaks"]

                if len(bin_centers) > 1:
                    width = bin_centers[1] - bin_centers[0]
                else:
                    width = 0.1

                ax_s.bar(bin_centers, counts_norm, width=width, alpha=0.5, label="Hist (s)")
                if dens.size > 0:
                    ax_s.plot(xs, dens / dens.max() * counts_norm.max(), label="KDE (s)")
                for mu in peaks:
                    ax_s.axvline(mu, linestyle="--")
                    ymax = counts_norm.max()
                    ax_s.text(mu, ymax*1.02, f"{mu:.2f}",
                              rotation=90, ha="center", va="bottom", fontsize=7)
                axis_label = info.get("axis_label") or "Mulliken spin"
                ax_s.set_xlabel(axis_label)
                ax_s.legend(fontsize=8)
            else:
                ax_s.text(0.5, 0.5, "No s data", transform=ax_s.transAxes,
                          ha="center", va="center")
                ax_s.set_axis_off()

    plt.tight_layout()
    fig.savefig(fig_outname, dpi=300)
    plt.close(fig)
    print(f"[OK] Combined charge/spin figure saved to '{fig_outname}'.")


def make_timeseries_figure(
    times,
    per_atom_charge,
    per_atom_spin,
    atom_ids,
    atom_labels=None,
    fig_outname="qs_timeseries.png",
    spin_ylabel="Spin"
):
    """
    Generate a figure with charge and spin as a function of time for the selected atoms.
    """
    has_charge = any(np.asarray(per_atom_charge.get(aid, []), dtype=float).size > 0 for aid in atom_ids)
    has_spin = any(np.asarray(per_atom_spin.get(aid, []), dtype=float).size > 0 for aid in atom_ids)
    if not has_charge and not has_spin:
        print("[NOTICE] Time-series figure was not generated (no charge or spin data available).")
        return

    ncols = 1 + int(has_spin)
    fig, axes = plt.subplots(1, ncols, squeeze=False, figsize=(5 * ncols, 4))
    cmap = plt.get_cmap("tab10")

    ax_q = axes[0, 0]
    if has_charge:
        for idx, aid in enumerate(atom_ids):
            q_vals = np.asarray(per_atom_charge.get(aid, []), dtype=float)
            if q_vals.size == 0:
                continue
            n = min(times.size, q_vals.size)
            label = atom_labels.get(aid, str(aid)) if atom_labels is not None else str(aid)
            ax_q.plot(times[:n], q_vals[:n], linewidth=1.6, color=cmap(idx % 10), label=f"Atom {label}")
        ax_q.set_xlabel("Time (ps)")
        ax_q.set_ylabel("Charge")
        ax_q.legend(fontsize=8)
    else:
        ax_q.text(0.5, 0.5, "No q data", transform=ax_q.transAxes, ha="center", va="center")
        ax_q.set_axis_off()

    if has_spin:
        ax_s = axes[0, 1]
        for idx, aid in enumerate(atom_ids):
            s_vals = np.asarray(per_atom_spin.get(aid, []), dtype=float)
            if s_vals.size == 0:
                continue
            n = min(times.size, s_vals.size)
            label = atom_labels.get(aid, str(aid)) if atom_labels is not None else str(aid)
            ax_s.plot(times[:n], s_vals[:n], linewidth=1.6, color=cmap(idx % 10), label=f"Atom {label}")
        ax_s.set_xlabel("Time (ps)")
        ax_s.set_ylabel(spin_ylabel)
        ax_s.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(fig_outname, dpi=300)
    plt.close(fig)
    print(f"[OK] Charge/spin time-series figure saved to '{fig_outname}'.")


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


def parse_atom_id_list(atom_ids_str):
    """
    Convert a space-separated string of integers into a list of atom IDs.
    """
    try:
        atom_ids = [int(x) for x in atom_ids_str.split()]
    except ValueError:
        print("Error: atom numbers must be integers separated by spaces.")
        sys.exit(1)
    return atom_ids


def get_spin_representation_config(choice):
    """
    Return the configuration associated with the selected spin representation.
    """
    if choice == "fraction":
        return {
            "normalize": True,
            "column_label": "spin_fraction",
            "axis_label": "Spin fraction",
            "global_axis_label": "Spin fraction"
        }
    return {
        "normalize": False,
        "column_label": "spin",
        "axis_label": "Spin",
        "global_axis_label": None
    }


def get_population_analysis_config(choice):
    """
    Return which population analyses must be processed.
    """
    if choice == "mulliken":
        return {"mulliken": True, "loewdin": False, "chelpg_loewdin": False, "chelpg_mulliken": False}
    if choice == "loewdin":
        return {"mulliken": False, "loewdin": True, "chelpg_loewdin": False, "chelpg_mulliken": False}
    if choice == "chelpg_loewdin":
        return {"mulliken": False, "loewdin": False, "chelpg_loewdin": True, "chelpg_mulliken": False}
    if choice == "chelpg_mulliken":
        return {"mulliken": False, "loewdin": False, "chelpg_loewdin": False, "chelpg_mulliken": True}
    return {"mulliken": True, "loewdin": True, "chelpg_loewdin": True, "chelpg_mulliken": True}


def get_histogram_binning_config(choice):
    """
    Return the bin specification for histograms.
    """
    if choice == "fixed_custom":
        return "fixed_custom"
    if choice == "sturges":
        return "sturges"
    if choice == "auto":
        return "auto"
    return "fd"


def get_analysis_display_label(analysis_kind):
    """
    Return a human-readable label for the population analysis.
    """
    labels = {
        "mulliken": "Mulliken",
        "loewdin": "Loewdin",
        "chelpg": "CHELPG",
        "chelpg_loewdin": "CHELPG",
        "chelpg_mulliken": "CHELPG",
    }
    return labels.get(analysis_kind, analysis_kind.capitalize())


def resolve_histogram_bins_spec(choice):
    """
    Resolve the final bin specification, including a user-selected fixed-bin variant.
    """
    bins_spec = get_histogram_binning_config(choice)
    if bins_spec != "fixed_custom":
        return bins_spec

    bins_str = input("Enter the fixed number of bins to use (Enter = 50): ").strip()
    if bins_str == "":
        return 50
    if not bins_str.isdigit():
        print("Error: the number of bins must be a positive integer.")
        sys.exit(1)

    bins_value = int(bins_str)
    if bins_value <= 0:
        print("Error: the number of bins must be greater than zero.")
        sys.exit(1)
    return bins_value


def analyze_spin_consistency(times, spin_values, spin_totals, tolerance, report_outname):
    """
    Compare the sum of spins over the selected atoms against the total spin per frame.
    """
    if times.size == 0:
        return np.array([], dtype=bool)

    missing_selected = np.isnan(spin_values).any(axis=1)
    selected_sum = np.nansum(spin_values, axis=1)
    diff = selected_sum - spin_totals
    keep_mask = (np.abs(diff) <= tolerance) & (~missing_selected)

    with open(report_outname, "w") as out:
        out.write("# snapshot time_ps selected_spin_sum total_spin diff missing_selected_atom use_in_analysis\n")
        for iframe, (t, sel, total, delta, missing, keep) in enumerate(
            zip(times, selected_sum, spin_totals, diff, missing_selected, keep_mask)
        ):
            use_label = "Yes" if keep else "No"
            out.write(f"{iframe:d} {t:.7f} {sel:.7f} {total:.7f} {delta:.7f} {int(missing)} {use_label}\n")

    n_bad = int((~keep_mask).sum())
    if n_bad > 0:
        print(
            f"[WARN] In {n_bad} snapshots, the sum of spins over the selected atoms differs from the total spin "
            f"by more than {tolerance:.4f}. Some atoms may be missing from the analysis or artifacts may be present."
        )
    else:
        print(f"[OK] The sum of spins over the selected atoms reproduces the total spin within {tolerance:.4f}.")

    print(f"[OK] Spin consistency report saved to '{report_outname}'.")
    return keep_mask


def suggest_missing_spin_atoms(
    spin_full,
    dt_ps,
    selected_atom_ids,
    bad_mask,
    header_start,
    spin_sign,
    report_outname="spin_missing_atom_suggestions.dat",
    top_n=12
):
    """
    Suggest unselected atoms that help reconcile the spin sum.
    """
    if bad_mask.size == 0 or not bad_mask.any():
        return []

    atom_info = get_atom_list_from_full(spin_full, header_start)
    all_atom_ids = [aid for aid, _atype in atom_info]
    atom_types = {aid: atype for aid, atype in atom_info}
    omitted_atom_ids = [aid for aid in all_atom_ids if aid not in set(selected_atom_ids)]
    if not omitted_atom_ids:
        return []

    times_all, all_spin_values, spin_totals = parse_frame_data(
        spin_full,
        dt_ps,
        all_atom_ids,
        kind="spin",
        header_start=header_start,
        spin_sign=spin_sign
    )
    if times_all.size == 0 or times_all.size != bad_mask.size:
        return []

    bad_values = all_spin_values[bad_mask, :]
    bad_totals = spin_totals[bad_mask]
    selected_idx = [all_atom_ids.index(aid) for aid in selected_atom_ids if aid in all_atom_ids]
    omitted_idx = [all_atom_ids.index(aid) for aid in omitted_atom_ids]

    selected_sum = np.nansum(bad_values[:, selected_idx], axis=1)
    missing_target = bad_totals - selected_sum
    mean_missing_target = float(np.nanmean(missing_target))

    suggestions = []
    for idx in omitted_idx:
        aid = all_atom_ids[idx]
        vals = bad_values[:, idx]
        valid = ~np.isnan(vals)
        if not valid.any():
            continue
        vals_valid = vals[valid]
        target_valid = missing_target[valid]
        if vals_valid.size == 0:
            continue
        mean_spin = float(np.mean(vals_valid))
        mean_abs_spin = float(np.mean(np.abs(vals_valid)))
        alignment = float(np.mean(vals_valid * target_valid))
        sign_match = 1.0 if mean_missing_target == 0.0 else np.sign(mean_spin) == np.sign(mean_missing_target)
        suggestions.append({
            "atom_id": aid,
            "atom_type": atom_types.get(aid, "?"),
            "mean_spin": mean_spin,
            "mean_abs_spin": mean_abs_spin,
            "alignment": alignment,
            "sign_match": bool(sign_match),
        })

    suggestions.sort(
        key=lambda item: (
            int(item["sign_match"]),
            item["alignment"],
            item["mean_abs_spin"]
        ),
        reverse=True
    )

    chosen = []
    cumulative = 0.0
    if mean_missing_target != 0.0:
        for item in suggestions:
            if np.sign(item["mean_spin"]) != np.sign(mean_missing_target):
                continue
            chosen.append(item)
            cumulative += item["mean_spin"]
            if abs(cumulative) >= abs(mean_missing_target):
                break

    with open(report_outname, "w") as out:
        out.write("# atom_id atom_type mean_spin_bad_snapshots mean_abs_spin_bad_snapshots alignment_with_missing_target sign_match\n")
        for item in suggestions[:top_n]:
            sign_match_label = "Yes" if item["sign_match"] else "No"
            out.write(
                f"{item['atom_id']:d} {item['atom_type']} {item['mean_spin']:.7f} "
                f"{item['mean_abs_spin']:.7f} {item['alignment']:.7f} {sign_match_label}\n"
            )

    if suggestions:
        print("[INFO] Candidate atoms to complete the missing spin:")
        for item in suggestions[:min(5, len(suggestions))]:
            print(
                f"  atom {item['atom_id']} ({item['atom_type']}): "
                f"mean spin={item['mean_spin']:.4f}, mean |spin|={item['mean_abs_spin']:.4f}"
            )
        if chosen:
            chosen_str = ", ".join(f"{item['atom_id']}({item['atom_type']})" for item in chosen)
            print(f"[INFO] Automatic suggestion of atoms to add: {chosen_str}")
    print(f"[OK] Missing-atom suggestions saved to '{report_outname}'.")

    return suggestions


def discover_global_analysis_dirs(base_dir):
    """
    Search immediate subdirectories for previous analysis results.
    """
    subdirs = []
    for entry in sorted(os.listdir(base_dir)):
        fullpath = os.path.join(base_dir, entry)
        if not os.path.isdir(fullpath):
            continue
        if any(
            os.path.exists(os.path.join(fullpath, marker))
            for marker in (
                "qs_histograms.png",
                "mulliken_histograms.png",
                "loewdin_histograms.png",
                "chelpg_histograms.png",
                "chelpg_loewdin_histograms.png",
                "chelpg_mulliken_histograms.png",
            )
        ):
            subdirs.append(fullpath)
    return subdirs


def collect_global_hist_data(base_dir, atom_map, analysis_kind):
    """
    Collect previously analyzed time series from subdirectories.

    analysis_kind:
      - "mulliken" -> atom_<id>_qs_timeseries.dat
      - "loewdin"  -> atom_<id>_loewdin_timeseries.dat
    """
    suffix_map = {
        "mulliken": ["qs"],
        "loewdin": ["loewdin"],
        "chelpg": ["chelpg", "chelpg_loewdin"],
        "chelpg_loewdin": ["chelpg_loewdin", "chelpg"],
        "chelpg_mulliken": ["chelpg_mulliken"],
    }
    suffixes = suffix_map[analysis_kind]
    systems_data = {}
    missing = []
    spin_labels_found = {}

    for system_dir in discover_global_analysis_dirs(base_dir):
        system_name = os.path.basename(system_dir)
        per_atom = {}
        found_any = False

        for aid in atom_map.get(system_name, []):
            fname = None
            for suffix in suffixes:
                candidate = os.path.join(system_dir, f"atom_{aid}_{suffix}_timeseries.dat")
                if os.path.exists(candidate):
                    fname = candidate
                    break
            if fname is None:
                missing.append((system_name, aid, os.path.join(system_dir, f"atom_{aid}_{suffixes[0]}_timeseries.dat")))
                continue

            charges, spins, spin_label = load_atom_timeseries_file(fname)
            if charges.size == 0 and spins.size == 0:
                missing.append((system_name, aid, fname))
                continue

            per_atom[aid] = {"charge": charges, "spin": spins}
            if spins.size > 0 and spin_label is not None:
                spin_labels_found.setdefault(system_name, set()).add(spin_label)
            found_any = True

        if found_any:
            systems_data[system_name] = per_atom

    return systems_data, missing, spin_labels_found


def get_central_range(data_chunks, percentile=95.0):
    """
    Compute a robust range containing the requested central percentage.
    """
    if not data_chunks:
        return None

    vals = np.concatenate(data_chunks)
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return None

    tail = 0.5 * (100.0 - percentile)
    xmin, xmax = np.percentile(vals, [tail, 100.0 - tail])
    if xmin == xmax:
        pad = max(abs(xmin) * 0.05, 1e-3)
        xmin -= pad
        xmax += pad
    else:
        pad = 0.05 * (xmax - xmin)
        xmin -= pad
        xmax += pad
    return xmin, xmax


def export_global_plot_data(output_dir, atom_ids, systems_data, analysis_kind, percentile=95.0):
    """
    Export the values actually used to build the global histograms.
    """
    os.makedirs(output_dir, exist_ok=True)
    values_out = os.path.join(output_dir, f"{analysis_kind}_global_plot_values.dat")
    ranges_out = os.path.join(output_dir, f"{analysis_kind}_global_plot_ranges.dat")

    with open(values_out, "w") as f_values, open(ranges_out, "w") as f_ranges:
        f_values.write("# system atom property value\n")
        f_ranges.write("# atom property x_min x_max n_total n_used percentile_central\n")

        for aid in atom_ids:
            charge_pool = []
            spin_pool = []
            for system_name in systems_data:
                atom_data = systems_data[system_name].get(aid)
                if not atom_data:
                    continue
                if atom_data["charge"].size > 0:
                    charge_pool.append(atom_data["charge"])
                if atom_data["spin"].size > 0:
                    spin_pool.append(atom_data["spin"])

            charge_xlim = get_central_range(charge_pool, percentile=percentile)
            spin_xlim = get_central_range(spin_pool, percentile=percentile)

            for prop, xlim in (("charge", charge_xlim), ("spin", spin_xlim)):
                n_total = 0
                n_used = 0
                for system_name in systems_data:
                    atom_data = systems_data[system_name].get(aid)
                    if not atom_data:
                        continue
                    vals = atom_data[prop]
                    if vals.size == 0:
                        continue
                    vals = vals[~np.isnan(vals)]
                    n_total += vals.size
                    vals_plot = vals
                    if xlim is not None:
                        vals_plot = vals[(vals >= xlim[0]) & (vals <= xlim[1])]
                    n_used += vals_plot.size
                    for val in vals_plot:
                        f_values.write(f"{system_name} {aid:d} {prop} {val:.7f}\n")

                if xlim is not None:
                    f_ranges.write(
                        f"{aid:d} {prop} {xlim[0]:.7f} {xlim[1]:.7f} {n_total:d} {n_used:d} {percentile:.1f}\n"
                    )
                else:
                    f_ranges.write(
                        f"{aid:d} {prop} nan nan {n_total:d} {n_used:d} {percentile:.1f}\n"
                    )

    print(f"[OK] Global histogram values saved to '{values_out}'.")
    print(f"[OK] Global plotting ranges saved to '{ranges_out}'.")


def make_global_overlay_hist_figure(
    atom_ids,
    systems_data,
    atom_labels=None,
    analysis_label="Loewdin",
    spin_axis_label=None,
    fig_outname="global_histograms.png",
    bins_spec=50,
    percentile=95.0
):
    """
    Generate overlaid charge and spin histograms for multiple systems.
    Each row corresponds to one atom and each color to a different system.
    """
    if not systems_data:
        print("[NOTICE] Global figure was not generated: no data were found for comparison.")
        return

    system_names = list(systems_data.keys())
    cmap = plt.get_cmap("tab10")

    fig, axes = plt.subplots(len(atom_ids), 2, squeeze=False, figsize=(10, 3 * len(atom_ids)))

    for row, aid in enumerate(atom_ids):
        label = atom_labels.get(aid, str(aid)) if atom_labels is not None else str(aid)
        ax_q = axes[row, 0]
        ax_s = axes[row, 1]
        has_q = False
        has_s = False

        charge_pool = []
        spin_pool = []
        for system_name in system_names:
            atom_data = systems_data[system_name].get(aid)
            if not atom_data:
                continue
            if atom_data["charge"].size > 0:
                charge_pool.append(atom_data["charge"])
            if atom_data["spin"].size > 0:
                spin_pool.append(atom_data["spin"])

        charge_xlim = get_central_range(charge_pool, percentile=percentile)
        spin_xlim = get_central_range(spin_pool, percentile=percentile)
        charge_vals_all = np.concatenate(charge_pool) if charge_pool else np.array([])
        spin_vals_all = np.concatenate(spin_pool) if spin_pool else np.array([])
        charge_edges = get_histogram_edges(charge_vals_all, bins_spec=bins_spec, value_range=charge_xlim) if charge_pool else np.array([])
        spin_edges = get_histogram_edges(spin_vals_all, bins_spec=bins_spec, value_range=spin_xlim) if spin_pool else np.array([])

        for idx, system_name in enumerate(system_names):
            atom_data = systems_data[system_name].get(aid)
            if not atom_data:
                continue

            color = cmap(idx % 10)
            charges = atom_data["charge"]
            spins = atom_data["spin"]

            if charges.size > 0:
                charges_plot = charges
                if charge_xlim is not None:
                    charges_plot = charges[(charges >= charge_xlim[0]) & (charges <= charge_xlim[1])]
                if charges_plot.size > 0:
                    ax_q.hist(
                        charges_plot,
                        bins=charge_edges if charge_edges.size > 1 else bins_spec,
                        density=True,
                        alpha=0.30,
                        color=color,
                        edgecolor=color,
                        linewidth=0.6,
                        label=system_name
                    )
                    if charges_plot.size > 1 and np.unique(charges_plot).size > 1:
                        xs = np.linspace(charge_xlim[0], charge_xlim[1], 200) if charge_xlim is not None else np.linspace(charges_plot.min(), charges_plot.max(), 200)
                        dens = gaussian_kde(charges_plot)(xs)
                        ax_q.plot(xs, dens, color=color, linewidth=1.8)
                    has_q = True

            if spins.size > 0:
                spins_plot = spins
                if spin_xlim is not None:
                    spins_plot = spins[(spins >= spin_xlim[0]) & (spins <= spin_xlim[1])]
                if spins_plot.size > 0:
                    ax_s.hist(
                        spins_plot,
                        bins=spin_edges if spin_edges.size > 1 else bins_spec,
                        density=True,
                        alpha=0.30,
                        color=color,
                        edgecolor=color,
                        linewidth=0.6,
                        label=system_name
                    )
                    if spins_plot.size > 1 and np.unique(spins_plot).size > 1:
                        xs = np.linspace(spin_xlim[0], spin_xlim[1], 200) if spin_xlim is not None else np.linspace(spins_plot.min(), spins_plot.max(), 200)
                        dens = gaussian_kde(spins_plot)(xs)
                        ax_s.plot(xs, dens, color=color, linewidth=1.8)
                    has_s = True

        if has_q:
            if charge_xlim is not None:
                ax_q.set_xlim(charge_xlim)
            ax_q.set_ylabel(f"Atom {label}")
            ax_q.set_xlabel(f"{analysis_label} charge")
            ax_q.legend(fontsize=8)
        else:
            ax_q.text(0.5, 0.5, "No q data", transform=ax_q.transAxes, ha="center", va="center")
            ax_q.set_axis_off()

        if has_s:
            if spin_xlim is not None:
                ax_s.set_xlim(spin_xlim)
            spin_xlabel = spin_axis_label if spin_axis_label is not None else f"{analysis_label} spin"
            ax_s.set_xlabel(spin_xlabel)
            ax_s.legend(fontsize=8)
        else:
            ax_s.text(0.5, 0.5, "No s data", transform=ax_s.transAxes, ha="center", va="center")
            ax_s.set_axis_off()

    plt.tight_layout()
    fig.savefig(fig_outname, dpi=300)
    plt.close(fig)
    print(f"[OK] Global comparison figure saved to '{fig_outname}'.")


# ==========================
# MAIN
# ==========================

def main():
    print_welcome_banner()
    mode = prompt_numbered_choice(
        "Execution mode:",
        [("Single-system analysis", "i"), ("Global analysis across subdirectories", "g")],
        default_idx=0
    )

    if mode in ("g",):
        global_binning_choice = prompt_numbered_choice(
            "Binning for global histograms:",
            [("Automatic (Freedman-Diaconis)", "fd"),
             ("Automatic (numpy auto)", "auto"),
             ("Automatic (Sturges)", "sturges"),
             ("Fixed: choose bin count", "fixed_custom")],
            default_idx=0
        )
        global_bins_spec = resolve_histogram_bins_spec(global_binning_choice)

        analysis_kind = prompt_numbered_choice(
            "Population analysis to compare in global mode:",
            [("Loewdin", "loewdin"),
             ("Mulliken", "mulliken"),
             ("CHELPG (charge) + Loewdin (spin)", "chelpg_loewdin"),
             ("CHELPG (charge) + Mulliken (spin)", "chelpg_mulliken")],
            default_idx=0
        )

        base_dir = os.getcwd()
        system_dirs = discover_global_analysis_dirs(base_dir)
        if not system_dirs:
            print("Error: no compatible previous analyses were found in subdirectories of the current directory.")
            sys.exit(1)

        system_names = [os.path.basename(system_dir) for system_dir in system_dirs]
        print("Systems detected for global analysis:")
        for system_name in system_names:
            print("  ", system_name)

        atom_selection_mode = prompt_numbered_choice(
            "Atom selection for global mode:",
            [("Same atoms for all systems", "mismos"),
             ("Specific atoms for each system", "especificos")],
            default_idx=0
        )

        atom_map = {}
        if atom_selection_mode == "mismos":
            atom_ids_str = input(
                "Enter atom numbers to compare, separated by spaces (for example: 88 89 90): "
            ).strip()
            atom_ids = parse_atom_id_list(atom_ids_str)
            if not atom_ids:
                print("No atoms were provided for comparison.")
                sys.exit(1)
            for system_name in system_names:
                atom_map[system_name] = list(atom_ids)
        else:
            for system_name in system_names:
                atom_ids_str = input(
                    f"Enter atoms to compare for {system_name}, separated by spaces (Enter = skip system): "
                ).strip()
                if atom_ids_str == "":
                    atom_map[system_name] = []
                    continue
                atom_ids = parse_atom_id_list(atom_ids_str)
                atom_map[system_name] = atom_ids

        atom_ids = sorted({aid for aids in atom_map.values() for aid in aids})
        if not atom_ids:
            print("No atoms were provided for comparison.")
            sys.exit(1)

        atom_labels = {aid: str(aid) for aid in atom_ids}
        use_labels = prompt_numbered_choice(
            "Do you want to assign custom labels to the atoms?",
            [("No", False), ("Yes", True)],
            default_idx=0
        )
        if use_labels:
            print("Enter a label for each atom; leave blank to use the atom number.")
            for aid in atom_ids:
                label = input(f"Label for atom {aid} (leave blank to use '{aid}'): ").strip()
                if label:
                    atom_labels[aid] = label

        systems_data, missing, spin_labels_found = collect_global_hist_data(base_dir, atom_map, analysis_kind)
        if not systems_data:
            print("Error: no compatible previous analyses were found in subdirectories of the current directory.")
            sys.exit(1)

        all_spin_labels = sorted({label for labels in spin_labels_found.values() for label in labels})
        if analysis_kind in ("chelpg", "chelpg_loewdin"):
            spin_axis_label = "Loewdin spin"
        elif analysis_kind == "chelpg_mulliken":
            spin_axis_label = "Mulliken spin"
        else:
            spin_axis_label = None
        if all_spin_labels:
            if len(all_spin_labels) > 1:
                print("Error: global analyses with incompatible spin representations were detected:")
                for system_name in sorted(spin_labels_found):
                    labels = ", ".join(sorted(spin_labels_found[system_name]))
                    print(f"  {system_name}: {labels}")
                print("Reprocess the directories so that all of them use the same spin representation.")
                sys.exit(1)
            if all_spin_labels[0] == "spin_fraction":
                if analysis_kind in ("chelpg", "chelpg_loewdin"):
                    spin_axis_label = "Loewdin spin fraction"
                elif analysis_kind == "chelpg_mulliken":
                    spin_axis_label = "Mulliken spin fraction"
                else:
                    spin_axis_label = f"{get_analysis_display_label(analysis_kind)} spin fraction"

        print("Systems included in the plot:")
        for system_name in systems_data:
            atoms_for_system = " ".join(str(aid) for aid in sorted(systems_data[system_name].keys()))
            print(f"  {system_name}: {atoms_for_system}")

        if missing:
            print("[WARN] Some time-series files are missing and were omitted from the plot:")
            for system_name, aid, fname in missing[:20]:
                print(f"  {system_name}: atom {aid} -> {os.path.basename(fname)}")
            if len(missing) > 20:
                print(f"  ... and {len(missing) - 20} more cases.")

        global_dir = os.path.join(base_dir, "global")
        os.makedirs(global_dir, exist_ok=True)

        export_global_plot_data(
            global_dir,
            atom_ids,
            systems_data,
            analysis_kind,
            percentile=95.0
        )

        fig_outname = os.path.join(global_dir, f"global_{analysis_kind}_histograms.png")
        make_global_overlay_hist_figure(
            atom_ids,
            systems_data,
            atom_labels=atom_labels,
            analysis_label=get_analysis_display_label(analysis_kind),
            spin_axis_label=spin_axis_label,
            fig_outname=fig_outname,
            bins_spec=global_bins_spec,
            percentile=95.0
        )
        return

    # Select program
    prog = prompt_numbered_choice(
        "Program to use:",
        [("LIO", "lio"), ("ORCA", "orca")],
        default_idx=1
    )

    have_spin = False
    spin_sign = 1.0
    orca_mode = (prog == "orca")
    population_config = {"mulliken": True, "loewdin": False, "chelpg_loewdin": False, "chelpg_mulliken": False}
    primary_analysis_kind = None
    active_charge_header = "# Mulliken Population Analysis"
    active_spin_header = "# Mulliken Spin Population Analysis"
    active_charge_axis_label = "Charge"
    active_spin_axis_label = "Spin"

    if prog == "lio":
        # Charge files
        mq_files = get_sorted_files("mq")
        if not mq_files:
            print("Error: no 'mq_*.dat' charge files were found.")
            sys.exit(1)

        print("Charge files (mq_*.dat) to be merged:")
        for f in mq_files:
            print("  ", f)
        merge_files(mq_files, "mq_full.dat")

        # Spin files (optional)
        ms_files = get_sorted_files("ms")
        have_spin = bool(ms_files)
        if have_spin:
            print("Spin files (ms_*.dat) to be merged:")
            for f in ms_files:
                print("  ", f)
            merge_files(ms_files, "ms_full.dat")
        else:
            print("[INFO] No 'ms_*.dat' files were found. Only charges will be analyzed.")

        charge_full = "mq_full.dat"
        spin_full = "ms_full.dat"
        spin_sign = -1.0
    else:
        # ORCA: each <prefix>_N.out/.dat file is a frame
        population_choice = prompt_numbered_choice(
            "Population analysis to process for ORCA:",
            [("Mulliken", "mulliken"),
             ("Loewdin", "loewdin"),
             ("CHELPG (charges) + Loewdin (spins)", "chelpg_loewdin"),
             ("CHELPG (charges) + Mulliken (spins)", "chelpg_mulliken"),
             ("All", "all")],
            default_idx=4
        )
        population_config = get_population_analysis_config(population_choice)

        orca_prefix = input(
            "Enter the ORCA file prefix before _N (for example, TD or SP; Enter = autodetect): "
        ).strip()

        orca_files = get_sorted_orca_files(orca_prefix)
        if not orca_files:
            if orca_prefix:
                print(f"Error: no files matching '{orca_prefix}_*.out' or '{orca_prefix}_*.dat' were found.")
            else:
                print("Error: no ORCA files matching '<prefix>_N.out' or '<prefix>_N.dat' were found.")
            sys.exit(1)

        if orca_prefix:
            print(f"ORCA files ({orca_prefix}_*.out/.dat) to be analyzed:")
        else:
            print("Autodetected ORCA files (*_N.out/.dat) to be analyzed:")
        for f in orca_files:
            print("  ", f)

        if population_config["mulliken"]:
            build_orca_full_files(
                orca_files,
                "mq_orca_todo.dat",
                "ms_orca_todo.dat",
                charge_label="Mulliken",
                charge_header_line="MULLIKEN ATOMIC CHARGES AND SPIN POPULATIONS"
            )
        if population_config["loewdin"]:
            build_orca_full_files(
                orca_files,
                "lq_orca_todo.dat",
                "ls_orca_todo.dat",
                charge_label="Loewdin",
                charge_header_line="LOEWDIN ATOMIC CHARGES AND SPIN POPULATIONS"
            )
        if population_config["chelpg_loewdin"]:
            build_orca_full_files(
                orca_files,
                "cq_loewdin_orca_todo.dat",
                "cs_loewdin_orca_todo.dat",
                charge_label="CHELPG",
                charge_header_line="CHELPG Charges",
                spin_label="Loewdin",
                spin_header_line="LOEWDIN ATOMIC CHARGES AND SPIN POPULATIONS"
            )
        if population_config["chelpg_mulliken"]:
            build_orca_full_files(
                orca_files,
                "cq_mulliken_orca_todo.dat",
                "cs_mulliken_orca_todo.dat",
                charge_label="CHELPG",
                charge_header_line="CHELPG Charges",
                spin_label="Mulliken",
                spin_header_line="MULLIKEN ATOMIC CHARGES AND SPIN POPULATIONS"
            )
        have_spin = True
        if population_config["mulliken"]:
            primary_analysis_kind = "mulliken"
            charge_full = "mq_orca_todo.dat"
            spin_full = "ms_orca_todo.dat"
            active_charge_header = "# Mulliken Population Analysis"
            active_spin_header = "# Mulliken Spin Population Analysis"
            active_charge_axis_label = "Mulliken charge"
        elif population_config["loewdin"]:
            primary_analysis_kind = "loewdin"
            charge_full = "lq_orca_todo.dat"
            spin_full = "ls_orca_todo.dat"
            active_charge_header = "# Loewdin Population Analysis"
            active_spin_header = "# Loewdin Spin Population Analysis"
            active_charge_axis_label = "Loewdin charge"
        elif population_config["chelpg_loewdin"]:
            primary_analysis_kind = "chelpg_loewdin"
            charge_full = "cq_loewdin_orca_todo.dat"
            spin_full = "cs_loewdin_orca_todo.dat"
            active_charge_header = "# CHELPG Population Analysis"
            active_spin_header = "# Loewdin Spin Population Analysis"
            active_charge_axis_label = "CHELPG charge"
        else:
            primary_analysis_kind = "chelpg_mulliken"
            charge_full = "cq_mulliken_orca_todo.dat"
            spin_full = "cs_mulliken_orca_todo.dat"
            active_charge_header = "# CHELPG Population Analysis"
            active_spin_header = "# Mulliken Spin Population Analysis"
            active_charge_axis_label = "CHELPG charge"
        spin_sign = 1.0

    # Show available atoms
    atoms_list = get_atom_list_from_full(charge_full, active_charge_header, lio=(prog == "lio"))
    if atoms_list:
        print("\nAvailable atoms (id, type):")
        for aid, atype in atoms_list:
            print(f"  {aid:4d}  {atype}")
        print("")

    atoms_str = input("Enter atom numbers to track, separated by spaces (for example: 45 46 47): ").strip()
    try:
        atom_ids = [int(x) for x in atoms_str.split()]
    except ValueError:
        print("Error: atom numbers must be integers separated by spaces.")
        sys.exit(1)

    if not atom_ids:
        print("No atoms were provided for tracking.")
        sys.exit(1)

    # Custom labels for atoms (optional)
    atom_labels = {aid: str(aid) for aid in atom_ids}
    use_labels = prompt_numbered_choice(
        "Do you want to assign custom labels to the atoms?",
        [("No", False), ("Yes", True)],
        default_idx=0
    )
    if use_labels:
        print("Enter a label for each atom; leave blank to use the default atom number.")
        for aid in atom_ids:
            label = input(f"Label for atom {aid} (for example, 'Fe'; leave blank to use '{aid}'): ").strip()
            if label:
                atom_labels[aid] = label

    make_time_plots = prompt_numbered_choice(
        "Do you want to generate charge/spin time-series plots?",
        [("No", False), ("Yes", True)],
        default_idx=0
    )
    if make_time_plots:
        dt_str = input("Enter the time step in picoseconds (for example, 0.001): ").strip()
        try:
            dt_ps = float(dt_str)
        except ValueError:
            print("Error: the time step must be a number (float).")
            sys.exit(1)
    else:
        # Keep downstream analysis working without prompting for dt when no time plots are requested.
        dt_ps = 1.0

    hist_binning_choice = prompt_numbered_choice(
        "Binning for single-system histograms:",
        [("Automatic (Freedman-Diaconis)", "fd"),
         ("Automatic (numpy auto)", "auto"),
         ("Automatic (Sturges)", "sturges"),
         ("Fixed: choose bin count", "fixed_custom")],
        default_idx=0
    )
    hist_bins_spec = resolve_histogram_bins_spec(hist_binning_choice)

    spin_mode = prompt_numbered_choice(
        "Spin representation for the analysis:",
        [("Raw spin from the output", "raw"),
         ("Spin as a fraction of the total spin of the selected atoms", "fraction")],
        default_idx=1
    )
    spin_config = get_spin_representation_config(spin_mode)
    if orca_mode and primary_analysis_kind in ("loewdin", "chelpg_loewdin"):
        active_spin_axis_label = "Loewdin spin fraction" if spin_config["normalize"] else "Loewdin spin"
    elif orca_mode and primary_analysis_kind == "chelpg_mulliken":
        active_spin_axis_label = "Mulliken spin fraction" if spin_config["normalize"] else "Mulliken spin"
    else:
        active_spin_axis_label = spin_config["axis_label"]

    spin_keep_mask = None
    if have_spin:
        run_spin_check = prompt_numbered_choice(
            "Spin consistency check per snapshot:",
            [("Skip the check", False),
             ("Compare the sum over selected atoms with the total system spin", True)],
            default_idx=1
        )
        if run_spin_check:
            tol_str = input(
                "Enter the allowed tolerance for |selected_spin_sum - total_spin| (Enter = 0.20): "
            ).strip()
            if tol_str == "":
                spin_tolerance = 0.20
            else:
                try:
                    spin_tolerance = float(tol_str)
                except ValueError:
                    print("Error: the tolerance must be a number.")
                    sys.exit(1)

            spin_times_check, spin_values_check, spin_totals_check = parse_frame_data(
                spin_full,
                dt_ps,
                atom_ids,
                kind="spin",
                header_start=active_spin_header,
                spin_sign=spin_sign
            )
            spin_keep_mask = analyze_spin_consistency(
                spin_times_check,
                spin_values_check,
                spin_totals_check,
                tolerance=spin_tolerance,
                report_outname="spin_consistency_report.dat"
            )

            if (~spin_keep_mask).any():
                suggest_missing_spin_atoms(
                    spin_full,
                    dt_ps,
                    atom_ids,
                    bad_mask=(~spin_keep_mask),
                    header_start=active_spin_header,
                    spin_sign=spin_sign,
                    report_outname="spin_missing_atom_suggestions.dat"
                )
                filter_bad = prompt_numbered_choice(
                    "Snapshots with poor spin consistency were detected:",
                    [("Keep all snapshots", False),
                     ("Remove snapshots outside the tolerance from the analysis", True)],
                    default_idx=0
                )
                if not filter_bad:
                    spin_keep_mask = None
                elif spin_keep_mask.sum() == 0:
                    print("Error: all snapshots were excluded by the tolerance filter.")
                    sys.exit(1)
                else:
                    n_kept = int(spin_keep_mask.sum())
                    n_removed = int((~spin_keep_mask).sum())
                    print(
                        f"[INFO] Spin-consistency filtering applied: "
                        f"{n_removed} snapshots removed, {n_kept} snapshots kept."
                    )
                   

    if orca_mode and primary_analysis_kind == "mulliken":
        q_ts_out = "mulliken_charge_timeseries.dat"
        q_avg_out = "mulliken_charge_averages.dat"
        q_hist_prefix = "mulliken_charge_hist"
        q_modes_out = "mulliken_charge_modes.dat"
        s_ts_out = "mulliken_spin_timeseries.dat"
        s_avg_out = "mulliken_spin_averages.dat"
        s_hist_prefix = "mulliken_spin_hist"
        s_modes_out = "mulliken_spin_modes.dat"
        mulliken_fig_out = "mulliken_histograms.png"
    elif orca_mode and primary_analysis_kind == "loewdin":
        q_ts_out = "loewdin_charge_timeseries.dat"
        q_avg_out = "loewdin_charge_averages.dat"
        q_hist_prefix = "loewdin_charge_hist"
        q_modes_out = "loewdin_charge_modes.dat"
        s_ts_out = "loewdin_spin_timeseries.dat"
        s_avg_out = "loewdin_spin_averages.dat"
        s_hist_prefix = "loewdin_spin_hist"
        s_modes_out = "loewdin_spin_modes.dat"
        mulliken_fig_out = "loewdin_histograms.png"
    elif orca_mode and primary_analysis_kind == "chelpg_loewdin":
        q_ts_out = "chelpg_loewdin_charge_timeseries.dat"
        q_avg_out = "chelpg_loewdin_charge_averages.dat"
        q_hist_prefix = "chelpg_loewdin_charge_hist"
        q_modes_out = "chelpg_loewdin_charge_modes.dat"
        s_ts_out = "loewdin_spin_timeseries_for_chelpg_loewdin.dat"
        s_avg_out = "loewdin_spin_averages_for_chelpg_loewdin.dat"
        s_hist_prefix = "loewdin_spin_hist_for_chelpg_loewdin"
        s_modes_out = "loewdin_spin_modes_for_chelpg_loewdin.dat"
        mulliken_fig_out = "chelpg_loewdin_histograms.png"
    elif orca_mode and primary_analysis_kind == "chelpg_mulliken":
        q_ts_out = "chelpg_mulliken_charge_timeseries.dat"
        q_avg_out = "chelpg_mulliken_charge_averages.dat"
        q_hist_prefix = "chelpg_mulliken_charge_hist"
        q_modes_out = "chelpg_mulliken_charge_modes.dat"
        s_ts_out = "mulliken_spin_timeseries_for_chelpg_mulliken.dat"
        s_avg_out = "mulliken_spin_averages_for_chelpg_mulliken.dat"
        s_hist_prefix = "mulliken_spin_hist_for_chelpg_mulliken"
        s_modes_out = "mulliken_spin_modes_for_chelpg_mulliken.dat"
        mulliken_fig_out = "chelpg_mulliken_histograms.png"
    else:
        q_ts_out = "mq_charge_timeseries.dat"
        q_avg_out = "mq_charge_averages.dat"
        q_hist_prefix = "mq_charge_hist"
        q_modes_out = "mq_charge_modes.dat"
        s_ts_out = "ms_spin_timeseries.dat"
        s_avg_out = "ms_spin_averages.dat"
        s_hist_prefix = "ms_spin_hist"
        s_modes_out = "ms_spin_modes.dat"
        mulliken_fig_out = "qs_histograms.png"

    spin_column_label = spin_config["column_label"]
    spin_axis_label = spin_config["axis_label"]
    primary_spin_ylabel = active_spin_axis_label if orca_mode else spin_axis_label

    if not orca_mode or primary_analysis_kind is not None:
        mulliken_charge_mask = apply_keep_mask_or_warn(
            spin_keep_mask,
            parse_frame_data(charge_full, dt_ps, atom_ids, "charge", active_charge_header)[0].size,
            active_charge_axis_label if orca_mode else "charge"
        )
        # --- Charge analysis ---
        times, per_atom_q, hist_q = build_timeseries_and_stats(
            charge_full,
            dt_ps,
            atom_ids,
            kind="charge",
            header_start=active_charge_header,
            ts_outname=q_ts_out,
            avg_outname=q_avg_out,
            hist_prefix=q_hist_prefix,
            modes_outname=q_modes_out,
            nbins_hist=hist_bins_spec,
            keep_mask=mulliken_charge_mask
        )

        # --- Spin analysis, if available ---
        hist_s = {}
        per_atom_s = {aid: np.array([]) for aid in atom_ids}
        if have_spin:
            mulliken_spin_mask = apply_keep_mask_or_warn(
                spin_keep_mask,
                parse_frame_data(spin_full, dt_ps, atom_ids, "spin", active_spin_header, spin_sign=spin_sign)[0].size,
                "Mulliken spin" if orca_mode else "spin"
            )
            _times_spin, per_atom_s, hist_s = build_timeseries_and_stats(
                spin_full,
                dt_ps,
                atom_ids,
                kind="spin",
                header_start=active_spin_header,
                ts_outname=s_ts_out,
                avg_outname=s_avg_out,
                hist_prefix=s_hist_prefix,
                modes_outname=s_modes_out,
                nbins_hist=hist_bins_spec,
                spin_sign=spin_sign,
                keep_mask=mulliken_spin_mask,
                normalize_spin_fraction=spin_config["normalize"]
            )
        else:
            print("[INFO] No spin analysis was performed because no ms_*.dat files were found.")

        for aid in atom_ids:
            q_vals = np.asarray(per_atom_q.get(aid, []), dtype=float)
            if times.size != q_vals.size:
                n = min(times.size, q_vals.size)
                t_use = times[:n]
                q_use = q_vals[:n]
            else:
                t_use = times
                q_use = q_vals

            if orca_mode and primary_analysis_kind == "loewdin":
                atom_out = f"atom_{aid}_loewdin_timeseries.dat"
            elif orca_mode and primary_analysis_kind == "chelpg_loewdin":
                atom_out = f"atom_{aid}_chelpg_loewdin_timeseries.dat"
            elif orca_mode and primary_analysis_kind == "chelpg_mulliken":
                atom_out = f"atom_{aid}_chelpg_mulliken_timeseries.dat"
            else:
                atom_out = f"atom_{aid}_qs_timeseries.dat"
            with open(atom_out, "w") as out:
                s_array = np.asarray(per_atom_s.get(aid, []), dtype=float)
                if have_spin and s_array.size > 0:
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

            print(f"[OK] Combined time series t, q, s for atom {aid} written to '{atom_out}'.")

        make_combined_hist_figure(
            atom_ids,
            hist_charge=hist_q,
            hist_spin=hist_s if have_spin else {},
            atom_labels=atom_labels,
            charge_axis_label=active_charge_axis_label,
            fig_outname=mulliken_fig_out
        )

        if make_time_plots:
            if orca_mode and primary_analysis_kind == "mulliken":
                mulliken_ts_fig_out = "mulliken_timeseries.png"
            elif orca_mode and primary_analysis_kind == "loewdin":
                mulliken_ts_fig_out = "loewdin_timeseries.png"
            elif orca_mode and primary_analysis_kind == "chelpg_loewdin":
                mulliken_ts_fig_out = "chelpg_loewdin_timeseries.png"
            elif orca_mode and primary_analysis_kind == "chelpg_mulliken":
                mulliken_ts_fig_out = "chelpg_mulliken_timeseries.png"
            else:
                mulliken_ts_fig_out = "qs_timeseries.png"
            make_timeseries_figure(
                times,
                per_atom_q,
                per_atom_s if have_spin else {},
                atom_ids,
                atom_labels=atom_labels,
                fig_outname=mulliken_ts_fig_out,
                spin_ylabel=primary_spin_ylabel
            )

        if orca_mode and primary_analysis_kind == "mulliken" and mulliken_fig_out != "qs_histograms.png":
            make_combined_hist_figure(
                atom_ids,
                hist_charge=hist_q,
                hist_spin=hist_s if have_spin else {},
                atom_labels=atom_labels,
                charge_axis_label=active_charge_axis_label,
                fig_outname="qs_histograms.png"
            )
            if make_time_plots and mulliken_ts_fig_out != "qs_timeseries.png":
                make_timeseries_figure(
                    times,
                    per_atom_q,
                    per_atom_s if have_spin else {},
                    atom_ids,
                    atom_labels=atom_labels,
                    fig_outname="qs_timeseries.png",
                    spin_ylabel=primary_spin_ylabel
                )

    # --- ORCA: additional Loewdin analysis ---
    if orca_mode and population_config["loewdin"] and primary_analysis_kind != "loewdin":
        loewdin_charge_mask = apply_keep_mask_or_warn(
            spin_keep_mask,
            parse_frame_data("lq_orca_todo.dat", dt_ps, atom_ids, "charge", "# Loewdin Population Analysis")[0].size,
            "Loewdin charge"
        )
        times_l, per_atom_q_l, hist_q_l = build_timeseries_and_stats(
            "lq_orca_todo.dat",
            dt_ps,
            atom_ids,
            kind="charge",
            header_start="# Loewdin Population Analysis",
            ts_outname="loewdin_charge_timeseries.dat",
            avg_outname="loewdin_charge_averages.dat",
            hist_prefix="loewdin_charge_hist",
            modes_outname="loewdin_charge_modes.dat",
            nbins_hist=hist_bins_spec,
            keep_mask=loewdin_charge_mask
        )

        hist_s_l = {}
        per_atom_s_l = {aid: np.array([]) for aid in atom_ids}
        loewdin_spin_mask = apply_keep_mask_or_warn(
            spin_keep_mask,
            parse_frame_data("ls_orca_todo.dat", dt_ps, atom_ids, "spin", "# Loewdin Spin Population Analysis", spin_sign=spin_sign)[0].size,
            "Loewdin spin"
        )
        _times_spin_l, per_atom_s_l, hist_s_l = build_timeseries_and_stats(
            "ls_orca_todo.dat",
            dt_ps,
            atom_ids,
            kind="spin",
            header_start="# Loewdin Spin Population Analysis",
            ts_outname="loewdin_spin_timeseries.dat",
            avg_outname="loewdin_spin_averages.dat",
            hist_prefix="loewdin_spin_hist",
            modes_outname="loewdin_spin_modes.dat",
            nbins_hist=hist_bins_spec,
            spin_sign=spin_sign,
            keep_mask=loewdin_spin_mask,
            normalize_spin_fraction=spin_config["normalize"]
        )

        for aid in atom_ids:
            q_vals = np.asarray(per_atom_q_l.get(aid, []), dtype=float)

            if times_l.size != q_vals.size:
                n = min(times_l.size, q_vals.size)
                t_use = times_l[:n]
                q_use = q_vals[:n]
            else:
                t_use = times_l
                q_use = q_vals

            atom_out = f"atom_{aid}_loewdin_timeseries.dat"
            with open(atom_out, "w") as out:
                s_array = np.asarray(per_atom_s_l.get(aid, []), dtype=float)
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

            print(f"[OK] Loewdin time series for atom {aid} written to '{atom_out}'.")

        make_combined_hist_figure(
            atom_ids,
            hist_charge=hist_q_l,
            hist_spin=hist_s_l,
            atom_labels=atom_labels,
            charge_axis_label="Loewdin charge",
            fig_outname="loewdin_histograms.png"
        )

        if make_time_plots:
            make_timeseries_figure(
                times_l,
                per_atom_q_l,
                per_atom_s_l,
                atom_ids,
                atom_labels=atom_labels,
                fig_outname="loewdin_timeseries.png",
                spin_ylabel="Loewdin spin fraction" if spin_config["normalize"] else "Loewdin spin"
            )

    # --- ORCA: additional CHELPG + Loewdin analysis ---
    if orca_mode and population_config["chelpg_loewdin"] and primary_analysis_kind != "chelpg_loewdin":
        chelpg_charge_mask = apply_keep_mask_or_warn(
            spin_keep_mask,
            parse_frame_data("cq_loewdin_orca_todo.dat", dt_ps, atom_ids, "charge", "# CHELPG Population Analysis")[0].size,
            "CHELPG charge"
        )
        times_c, per_atom_q_c, hist_q_c = build_timeseries_and_stats(
            "cq_loewdin_orca_todo.dat",
            dt_ps,
            atom_ids,
            kind="charge",
            header_start="# CHELPG Population Analysis",
            ts_outname="chelpg_loewdin_charge_timeseries.dat",
            avg_outname="chelpg_loewdin_charge_averages.dat",
            hist_prefix="chelpg_loewdin_charge_hist",
            modes_outname="chelpg_loewdin_charge_modes.dat",
            nbins_hist=hist_bins_spec,
            keep_mask=chelpg_charge_mask
        )

        hist_s_c = {}
        per_atom_s_c = {aid: np.array([]) for aid in atom_ids}
        chelpg_spin_mask = apply_keep_mask_or_warn(
            spin_keep_mask,
            parse_frame_data("cs_loewdin_orca_todo.dat", dt_ps, atom_ids, "spin", "# Loewdin Spin Population Analysis", spin_sign=spin_sign)[0].size,
            "Loewdin spin"
        )
        _times_spin_c, per_atom_s_c, hist_s_c = build_timeseries_and_stats(
            "cs_loewdin_orca_todo.dat",
            dt_ps,
            atom_ids,
            kind="spin",
            header_start="# Loewdin Spin Population Analysis",
            ts_outname="loewdin_spin_timeseries_for_chelpg_loewdin.dat",
            avg_outname="loewdin_spin_averages_for_chelpg_loewdin.dat",
            hist_prefix="loewdin_spin_hist_for_chelpg_loewdin",
            modes_outname="loewdin_spin_modes_for_chelpg_loewdin.dat",
            nbins_hist=hist_bins_spec,
            spin_sign=spin_sign,
            keep_mask=chelpg_spin_mask,
            normalize_spin_fraction=spin_config["normalize"]
        )

        for aid in atom_ids:
            q_vals = np.asarray(per_atom_q_c.get(aid, []), dtype=float)

            if times_c.size != q_vals.size:
                n = min(times_c.size, q_vals.size)
                t_use = times_c[:n]
                q_use = q_vals[:n]
            else:
                t_use = times_c
                q_use = q_vals

            atom_out = f"atom_{aid}_chelpg_loewdin_timeseries.dat"
            with open(atom_out, "w") as out:
                s_array = np.asarray(per_atom_s_c.get(aid, []), dtype=float)
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

            print(f"[OK] CHELPG+Loewdin time series for atom {aid} written to '{atom_out}'.")

        make_combined_hist_figure(
            atom_ids,
            hist_charge=hist_q_c,
            hist_spin=hist_s_c,
            atom_labels=atom_labels,
            charge_axis_label="CHELPG charge",
            fig_outname="chelpg_loewdin_histograms.png"
        )

        if make_time_plots:
            make_timeseries_figure(
                times_c,
                per_atom_q_c,
                per_atom_s_c,
                atom_ids,
                atom_labels=atom_labels,
                fig_outname="chelpg_loewdin_timeseries.png",
                spin_ylabel="Loewdin spin fraction" if spin_config["normalize"] else "Loewdin spin"
            )

    # --- ORCA: additional CHELPG + Mulliken analysis ---
    if orca_mode and population_config["chelpg_mulliken"] and primary_analysis_kind != "chelpg_mulliken":
        chelpg_charge_mask = apply_keep_mask_or_warn(
            spin_keep_mask,
            parse_frame_data("cq_mulliken_orca_todo.dat", dt_ps, atom_ids, "charge", "# CHELPG Population Analysis")[0].size,
            "CHELPG charge"
        )
        times_c, per_atom_q_c, hist_q_c = build_timeseries_and_stats(
            "cq_mulliken_orca_todo.dat",
            dt_ps,
            atom_ids,
            kind="charge",
            header_start="# CHELPG Population Analysis",
            ts_outname="chelpg_mulliken_charge_timeseries.dat",
            avg_outname="chelpg_mulliken_charge_averages.dat",
            hist_prefix="chelpg_mulliken_charge_hist",
            modes_outname="chelpg_mulliken_charge_modes.dat",
            nbins_hist=hist_bins_spec,
            keep_mask=chelpg_charge_mask
        )

        hist_s_c = {}
        per_atom_s_c = {aid: np.array([]) for aid in atom_ids}
        chelpg_spin_mask = apply_keep_mask_or_warn(
            spin_keep_mask,
            parse_frame_data("cs_mulliken_orca_todo.dat", dt_ps, atom_ids, "spin", "# Mulliken Spin Population Analysis", spin_sign=spin_sign)[0].size,
            "Mulliken spin"
        )
        _times_spin_c, per_atom_s_c, hist_s_c = build_timeseries_and_stats(
            "cs_mulliken_orca_todo.dat",
            dt_ps,
            atom_ids,
            kind="spin",
            header_start="# Mulliken Spin Population Analysis",
            ts_outname="mulliken_spin_timeseries_for_chelpg_mulliken.dat",
            avg_outname="mulliken_spin_averages_for_chelpg_mulliken.dat",
            hist_prefix="mulliken_spin_hist_for_chelpg_mulliken",
            modes_outname="mulliken_spin_modes_for_chelpg_mulliken.dat",
            nbins_hist=hist_bins_spec,
            spin_sign=spin_sign,
            keep_mask=chelpg_spin_mask,
            normalize_spin_fraction=spin_config["normalize"]
        )

        for aid in atom_ids:
            q_vals = np.asarray(per_atom_q_c.get(aid, []), dtype=float)

            if times_c.size != q_vals.size:
                n = min(times_c.size, q_vals.size)
                t_use = times_c[:n]
                q_use = q_vals[:n]
            else:
                t_use = times_c
                q_use = q_vals

            atom_out = f"atom_{aid}_chelpg_mulliken_timeseries.dat"
            with open(atom_out, "w") as out:
                s_array = np.asarray(per_atom_s_c.get(aid, []), dtype=float)
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

            print(f"[OK] CHELPG+Mulliken time series for atom {aid} written to '{atom_out}'.")

        make_combined_hist_figure(
            atom_ids,
            hist_charge=hist_q_c,
            hist_spin=hist_s_c,
            atom_labels=atom_labels,
            charge_axis_label="CHELPG charge",
            fig_outname="chelpg_mulliken_histograms.png"
        )

        if make_time_plots:
            make_timeseries_figure(
                times_c,
                per_atom_q_c,
                per_atom_s_c,
                atom_ids,
                atom_labels=atom_labels,
                fig_outname="chelpg_mulliken_timeseries.png",
                spin_ylabel="Mulliken spin fraction" if spin_config["normalize"] else "Mulliken spin"
            )


if __name__ == "__main__":
    main()
