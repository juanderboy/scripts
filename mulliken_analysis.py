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
# Utilidades generales
# ==========================

def get_sorted_files(prefix):
    """
    Busca archivos prefix_*.dat y los ordena por el número.
    Ej: prefix='mq' -> mq_0.dat, mq_1.dat, ...
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
    Muestra un menú numerado y retorna el valor asociado a la opción elegida.
    options: lista de tuplas (etiqueta, valor)
    default_idx: índice 0-based de la opción por defecto, o None
    """
    print(title)
    for idx, (label, _value) in enumerate(options, start=1):
        default_tag = " [default]" if default_idx is not None and idx - 1 == default_idx else ""
        print(f"  {idx}. {label}{default_tag}")

    prompt = "Seleccione una opción por número"
    if default_idx is not None:
        prompt += f" (Enter = {default_idx + 1})"
    prompt += ": "

    choice = input(prompt).strip()
    if choice == "":
        if default_idx is None:
            print("Error: debe seleccionar una opción.")
            sys.exit(1)
        return options[default_idx][1]

    if not choice.isdigit():
        print("Error: debe ingresar el número de una opción.")
        sys.exit(1)

    selected = int(choice)
    if selected < 1 or selected > len(options):
        print("Error: opción fuera de rango.")
        sys.exit(1)

    return options[selected - 1][1]

def get_sorted_orca_files(prefix=None):
    """
    Busca archivos ORCA con formato <prefijo>_N.out o <prefijo>_N.dat
    y los ordena por el número N.

    Si prefix es None o "", autodetecta cualquier prefijo.
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

            # Preferir .out si hay duplicados
            if idx not in files or ext == "out":
                files[idx] = fname

    if not normalized_prefix and len(detected_prefixes) > 1:
        print("Error: se detectaron múltiples prefijos ORCA en la carpeta:")
        for detected_prefix in sorted(detected_prefixes):
            print(f"  {detected_prefix}")
        print("Ingrese el prefijo deseado explícitamente para evitar mezclar archivos.")
        sys.exit(1)

    return [files[k] for k in sorted(files.keys())]


def merge_files(files, outname):
    """
    Concatena una lista de archivos en outname.
    """
    with open(outname, "w") as out:
        for fname in files:
            with open(fname, "r") as f:
                for line in f:
                    out.write(line)
    print(f"[OK] Archivos {files[0]} ... {files[-1]} combinados en '{outname}'.")


# ==========================
# Análisis KDE y modos
# ==========================

def analyze_modes_kde(vals, n_grid=200, prominence_factor=0.05):
    """
    Calcula una KDE 1D y busca máximos (modos) en la densidad.

    Parameters
    ----------
    vals : array-like
        Datos (cargas o spines) para un átomo.
    n_grid : int
        Número de puntos del grid para evaluar la KDE.
    prominence_factor : float
        Fracción de la altura máxima usada como prominencia mínima
        para detectar picos.

    Returns
    -------
    xs : np.ndarray
        Grid de valores donde se evaluó la densidad.
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
    Determina bordes de histograma a partir de una especificación fija o automática.

    bins_spec puede ser:
      - int: número fijo de bins
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
    Normaliza los spines por frame usando la suma de los átomos seleccionados.

    Para cada snapshot:
      spin_frac_i = spin_i / sum_j(spin_j)

    Si falta algún átomo seleccionado o la suma es ~0, ese frame se marca como NaN.
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
# Lectura de archivos full y estadística
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
    Analiza un archivo full (mq_full.dat o ms_full.dat).

    kind: 'charge' o 'spin'
    header_start:
        - "# Mulliken Population Analysis"  (cargas)
        - "# Mulliken Spin Population Analysis" (spin)

    Returns
    -------
    times : np.ndarray
        Vector de tiempos (ps) para cada frame.
    per_atom_values : dict
        {atom_id: np.ndarray de valores válidos (sin NaN)}.
    hist_data_for_plot : dict
        {atom_id: dict con info de histograma y KDE para ploteo}.
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
                f"[WARN] Se ignoró una máscara de snapshots incompatible en '{fullfile}': "
                f"máscara={keep_mask.size}, frames={times.size}."
            )
        else:
            times = times[keep_mask]
            values = values[keep_mask, :]

    data = np.column_stack((times, values)) if times.size > 0 else np.empty((0, len(atom_ids) + 1))

    # Serie temporal
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

    print(f"[OK] Serie temporal de {kind} escrita en '{ts_outname}'.")

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

    # Promedios
    with open(avg_outname, "w") as out:
        if kind == "spin" and normalize_spin_fraction:
            out.write("# Promedios de spin como fraccion del spin total de los atomos seleccionados\n")
        else:
            out.write(f"# Promedios de {kind}\n")
        out.write("# atom_id  avg_value\n")
        for i, aid in enumerate(atom_ids):
            vals = per_atom_values[aid]
            if vals.size > 0:
                avg = vals.mean()
                out.write(f"{aid:4d}  {avg: .7f}\n")
            else:
                out.write(f"{aid:4d}  nan\n")

    print(f"[OK] Promedios de {kind} escritos en '{avg_outname}'.")

    # Histogramas + KDE + análisis de modos
    modes_summary = []
    hist_data_for_plot = {}

    for i, aid in enumerate(atom_ids):
        vals = per_atom_values[aid]
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            print(f"[WARN] No hay datos válidos de {kind} para el átomo {aid}.")
            continue

        vmin, vmax = vals.min(), vals.max()

        if vmin == vmax:
            # Caso degenerado
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
            print(f"[OK] Histograma (trivial) de {kind} para átomo {aid} en '{hist_outname}'.")

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

        # KDE para este conjunto de valores
        xs, dens, peak_positions, peak_heights = analyze_modes_kde(vals)

        hist_outname = f"{hist_prefix}_atom_{aid}.dat"
        with open(hist_outname, "w") as out:
            out.write("# Histogram: bin_center  count\n")
            for c, center in zip(counts, bin_centers):
                out.write(f"{center: .7f} {c:d}\n")
            # Agregamos también la curva KDE para facilitar el post-procesado
            out.write("\n# KDE: x  density\n")
            if xs.size > 0 and dens.size > 0:
                for x_val, d_val in zip(xs, dens):
                    out.write(f"{x_val: .7f} {d_val: .7e}\n")

        print(f"[OK] Histograma de {kind} para átomo {aid} escrito en '{hist_outname}'.")

        modes_summary.append((aid, peak_positions, peak_heights))

        if len(peak_positions) == 1:
            print(f"    Átomo {aid}: distribución {kind} unimodal. Pico ≈ {peak_positions[0]: .4f}")
        else:
            print(f"    Átomo {aid}: se detectaron {len(peak_positions)} modos en {kind}.")
            for k, mu in enumerate(peak_positions):
                print(f"        Modo {k+1}: {kind} ≈ {mu: .4f}")

        hist_data_for_plot[aid] = {
            "bin_centers": bin_centers,
            "counts_norm": counts_norm,
            "xs": xs,
            "dens": dens,
            "peaks": peak_positions,
            "axis_label": "Spin fraction" if kind == "spin" and normalize_spin_fraction else None
        }

    # Resumen de modos
    with open(modes_outname, "w") as out:
        out.write(f"# atom_id  n_modes  peak_positions_{kind}(aprox)\n")
        for aid, peaks, heights in modes_summary:
            if len(peaks) == 0:
                out.write(f"{aid:4d}  0\n")
            else:
                peaks_str = " ".join(f"{mu: .6f}" for mu in peaks)
                out.write(f"{aid:4d}  {len(peaks):2d}  {peaks_str}\n")

    print(f"[OK] Resumen de modos de {kind} escrito en '{modes_outname}'.")

    return times, per_atom_values, hist_data_for_plot


def apply_keep_mask_or_warn(keep_mask, n_frames, analysis_label):
    """
    Valida una máscara de snapshots contra un análisis dado.

    Si la cantidad de frames no coincide, retorna None y emite un aviso.
    """
    if keep_mask is None:
        return None

    keep_mask = np.asarray(keep_mask, dtype=bool)
    if keep_mask.size != n_frames:
        print(
            f"[WARN] No se aplicó el filtrado de snapshots a {analysis_label}: "
            f"la máscara tiene {keep_mask.size} frames y el análisis tiene {n_frames}."
        )
        return None
    return keep_mask


# ==========================
# Utilidad: lista de átomos
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
    Extrae la tabla de cargas/spines (Mulliken o Loewdin) desde un output de ORCA.
    Retorna lista de tuplas: (atom_idx, element, charge, spin)
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


def build_orca_full_files(orca_files, out_charge, out_spin, label, header_line):
    """
    A partir de múltiples archivos ORCA <prefijo>_N.out/.dat arma:
      - archivo de cargas
      - archivo de spines
    con el formato que espera build_timeseries_and_stats.
    """
    if not orca_files:
        print("Error: no se encontraron archivos ORCA con formato '<prefijo>_N.out' o '<prefijo>_N.dat'.")
        sys.exit(1)

    with open(out_charge, "w") as q_out, open(out_spin, "w") as s_out:
        for fname in orca_files:
            block = extract_orca_population_block(fname, header_line)
            if not block:
                print(f"[WARN] No se encontró tabla {label} en '{fname}'.")
                continue

            q_out.write(f"# {label} Population Analysis\n")
            q_out.write("# Atom   Type   Population\n")
            s_out.write(f"# {label} Spin Population Analysis\n")
            s_out.write("# Atom   Type   Population\n")

            sum_q = 0.0
            sum_s = 0.0
            for atom_idx, element, charge, spin in block:
                q_out.write(f"{atom_idx:4d} {element:>3s} {charge: .7f}\n")
                s_out.write(f"{atom_idx:4d} {element:>3s} {spin: .7f}\n")
                sum_q += charge
                sum_s += spin

            # Línea de cierre compatible con el parser
            q_out.write(f"  Total Charge = {sum_q: .7f}\n\n")
            s_out.write(f"  Total Charge = {sum_s: .7f}\n\n")

    print(f"[OK] Archivo ORCA de cargas ({label}) combinado en '{out_charge}'.")
    print(f"[OK] Archivo ORCA de spines ({label}) combinado en '{out_spin}'.")


# ==========================
# Figura combinada q/s
# ==========================

def make_combined_hist_figure(
    atom_ids,
    hist_charge,
    hist_spin,
    atom_labels=None,
    fig_outname="qs_histograms.png"
):
    """
    Genera una figura conjunta con histogramas de cargas y spins.
    Por cada átomo hay una fila:
      - columna 1: carga (si existe)
      - columna 2: spin (si existe)
    Los picos se marcan y se etiquetan con el valor.
    """
    has_charge = bool(hist_charge)
    has_spin = bool(hist_spin)
    if not has_charge and not has_spin:
        print("[AVISO] No se generó figura combinada (no hay ni cargas ni spins).")
        return

    n_atoms = len(atom_ids)
    ncols = 1 + int(has_spin)  # 1 (q) o 2 (q+s)
    nrows = n_atoms

    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                             figsize=(4*ncols, 3*nrows))
    axes = axes  # matriz 2D [row, col]

    for i, aid in enumerate(atom_ids):
        # CARGA (columna 0)
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
            ax_q.set_xlabel("Mulliken charge")
            ax_q.legend(fontsize=8)
        else:
            ax_q.text(0.5, 0.5, "No q data", transform=ax_q.transAxes,
                      ha="center", va="center")
            ax_q.set_axis_off()

        # SPIN (columna 1, si hay)
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
    print(f"[OK] Figura combinada cargas/spines guardada en '{fig_outname}'.")


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
    Genera una figura con carga y spin en función del tiempo para los átomos elegidos.
    """
    has_charge = any(np.asarray(per_atom_charge.get(aid, []), dtype=float).size > 0 for aid in atom_ids)
    has_spin = any(np.asarray(per_atom_spin.get(aid, []), dtype=float).size > 0 for aid in atom_ids)
    if not has_charge and not has_spin:
        print("[AVISO] No se generó figura temporal (no hay datos de carga ni spin).")
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
    print(f"[OK] Figura temporal cargas/spines guardada en '{fig_outname}'.")


def load_atom_timeseries_file(fname):
    """
    Lee un archivo atom_<id>_..._timeseries.dat y retorna arrays de carga y spin.
    Si no existe columna de spin, devuelve un array vacío para spin.

    Returns
    -------
    charges : np.ndarray
    spins : np.ndarray
    spin_label : str | None
        Nombre de la tercera columna del archivo, por ejemplo 'spin' o
        'spin_fraction'. Si no hay tercera columna, retorna None.
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
    Convierte una cadena con enteros separados por espacios en una lista de átomos.
    """
    try:
        atom_ids = [int(x) for x in atom_ids_str.split()]
    except ValueError:
        print("Error: los números de átomo deben ser enteros separados por espacios.")
        sys.exit(1)
    return atom_ids


def get_spin_representation_config(choice):
    """
    Retorna configuración asociada a la representación de spin elegida.
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
    Retorna qué análisis de población deben procesarse.
    """
    if choice == "mulliken":
        return {"mulliken": True, "loewdin": False}
    if choice == "loewdin":
        return {"mulliken": False, "loewdin": True}
    return {"mulliken": True, "loewdin": True}


def get_histogram_binning_config(choice):
    """
    Retorna la especificación de bins para histogramas.
    """
    if choice == "fixed_custom":
        return "fixed_custom"
    if choice == "sturges":
        return "sturges"
    if choice == "auto":
        return "auto"
    return "fd"


def resolve_histogram_bins_spec(choice):
    """
    Resuelve la especificación final de bins, incluyendo la variante fija elegida por el usuario.
    """
    bins_spec = get_histogram_binning_config(choice)
    if bins_spec != "fixed_custom":
        return bins_spec

    bins_str = input("Ingrese la cantidad fija de bins a usar (Enter = 50): ").strip()
    if bins_str == "":
        return 50
    if not bins_str.isdigit():
        print("Error: la cantidad de bins debe ser un entero positivo.")
        sys.exit(1)

    bins_value = int(bins_str)
    if bins_value <= 0:
        print("Error: la cantidad de bins debe ser mayor que cero.")
        sys.exit(1)
    return bins_value


def analyze_spin_consistency(times, spin_values, spin_totals, tolerance, report_outname):
    """
    Compara la suma de spines de los átomos seleccionados contra el spin total por frame.
    """
    if times.size == 0:
        return np.array([], dtype=bool)

    missing_selected = np.isnan(spin_values).any(axis=1)
    selected_sum = np.nansum(spin_values, axis=1)
    diff = selected_sum - spin_totals
    keep_mask = (np.abs(diff) <= tolerance) & (~missing_selected)

    with open(report_outname, "w") as out:
        out.write("# snapshot time_ps selected_spin_sum total_spin diff missing_selected_atom usar_en_analisis\n")
        for iframe, (t, sel, total, delta, missing, keep) in enumerate(
            zip(times, selected_sum, spin_totals, diff, missing_selected, keep_mask)
        ):
            use_label = "Si" if keep else "No"
            out.write(f"{iframe:d} {t:.7f} {sel:.7f} {total:.7f} {delta:.7f} {int(missing)} {use_label}\n")

    n_bad = int((~keep_mask).sum())
    if n_bad > 0:
        print(
            f"[WARN] En {n_bad} snapshots la suma de spin de los átomos elegidos difiere del spin total "
            f"más de {tolerance:.4f}. Puede faltar analizar algún átomo o haber artifacts."
        )
    else:
        print(f"[OK] La suma de spin de los átomos elegidos reproduce el spin total dentro de {tolerance:.4f}.")

    print(f"[OK] Reporte de consistencia de spin guardado en '{report_outname}'.")
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
    Sugiere átomos no seleccionados que ayudan a reconciliar la suma de spin.
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
            sign_match_label = "Si" if item["sign_match"] else "No"
            out.write(
                f"{item['atom_id']:d} {item['atom_type']} {item['mean_spin']:.7f} "
                f"{item['mean_abs_spin']:.7f} {item['alignment']:.7f} {sign_match_label}\n"
            )

    if suggestions:
        print("[INFO] Átomos candidatos para completar el spin faltante:")
        for item in suggestions[:min(5, len(suggestions))]:
            print(
                f"  átomo {item['atom_id']} ({item['atom_type']}): "
                f"spin medio={item['mean_spin']:.4f}, |spin| medio={item['mean_abs_spin']:.4f}"
            )
        if chosen:
            chosen_str = ", ".join(f"{item['atom_id']}({item['atom_type']})" for item in chosen)
            print(f"[INFO] Sugerencia automática de átomos a agregar: {chosen_str}")
    print(f"[OK] Sugerencias de átomos faltantes guardadas en '{report_outname}'.")

    return suggestions


def discover_global_analysis_dirs(base_dir):
    """
    Busca subdirectorios inmediatos con resultados de análisis previos.
    """
    subdirs = []
    for entry in sorted(os.listdir(base_dir)):
        fullpath = os.path.join(base_dir, entry)
        if not os.path.isdir(fullpath):
            continue
        if any(
            os.path.exists(os.path.join(fullpath, marker))
            for marker in ("qs_histograms.png", "mulliken_histograms.png", "loewdin_histograms.png")
        ):
            subdirs.append(fullpath)
    return subdirs


def collect_global_hist_data(base_dir, atom_map, analysis_kind):
    """
    Recolecta series temporales ya analizadas desde subcarpetas.

    analysis_kind:
      - "mulliken" -> atom_<id>_qs_timeseries.dat
      - "loewdin"  -> atom_<id>_loewdin_timeseries.dat
    """
    suffix = "qs" if analysis_kind == "mulliken" else "loewdin"
    systems_data = {}
    missing = []
    spin_labels_found = {}

    for system_dir in discover_global_analysis_dirs(base_dir):
        system_name = os.path.basename(system_dir)
        per_atom = {}
        found_any = False

        for aid in atom_map.get(system_name, []):
            fname = os.path.join(system_dir, f"atom_{aid}_{suffix}_timeseries.dat")
            if not os.path.exists(fname):
                missing.append((system_name, aid, fname))
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
    Calcula un rango robusto que contiene el porcentaje central indicado.
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
    Exporta los valores realmente usados para construir los histogramas globales.
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

    print(f"[OK] Datos globales usados en los histogramas guardados en '{values_out}'.")
    print(f"[OK] Rangos globales de ploteo guardados en '{ranges_out}'.")


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
    Genera histogramas superpuestos de carga y spin para múltiples sistemas.
    Cada fila corresponde a un átomo y cada color a un sistema distinto.
    """
    if not systems_data:
        print("[AVISO] No se generó figura global: no se encontraron datos para comparar.")
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
    print(f"[OK] Figura global comparativa guardada en '{fig_outname}'.")


# ==========================
# MAIN
# ==========================

def main():
    mode = prompt_numbered_choice(
        "Modo de trabajo:",
        [("Análisis individual", "i"), ("Análisis global de subcarpetas", "g")],
        default_idx=0
    )

    if mode in ("g",):
        global_binning_choice = prompt_numbered_choice(
            "Binning para histogramas globales:",
            [("Automático (Freedman-Diaconis)", "fd"),
             ("Automático (numpy auto)", "auto"),
             ("Automático (Sturges)", "sturges"),
             ("Fijo: elegir cantidad", "fixed_custom")],
            default_idx=0
        )
        global_bins_spec = resolve_histogram_bins_spec(global_binning_choice)

        analysis_kind = prompt_numbered_choice(
            "Análisis a comparar en el modo global:",
            [("Loewdin", "loewdin"), ("Mulliken", "mulliken")],
            default_idx=0
        )

        base_dir = os.getcwd()
        system_dirs = discover_global_analysis_dirs(base_dir)
        if not system_dirs:
            print("Error: no se encontraron subcarpetas con análisis previos en el directorio actual.")
            sys.exit(1)

        system_names = [os.path.basename(system_dir) for system_dir in system_dirs]
        print("Sistemas detectados para el análisis global:")
        for system_name in system_names:
            print("  ", system_name)

        atom_selection_mode = prompt_numbered_choice(
            "Selección de átomos para el modo global:",
            [("Mismos átomos para todos los sistemas", "mismos"),
             ("Átomos específicos para cada sistema", "especificos")],
            default_idx=0
        )

        atom_map = {}
        if atom_selection_mode == "mismos":
            atom_ids_str = input(
                "Ingrese los números de átomo a comparar, separados por espacios (por ejemplo: 88 89 90): "
            ).strip()
            atom_ids = parse_atom_id_list(atom_ids_str)
            if not atom_ids:
                print("No se ingresó ningún átomo para comparar.")
                sys.exit(1)
            for system_name in system_names:
                atom_map[system_name] = list(atom_ids)
        else:
            for system_name in system_names:
                atom_ids_str = input(
                    f"Ingrese los átomos a comparar para {system_name}, separados por espacios (Enter = omitir sistema): "
                ).strip()
                if atom_ids_str == "":
                    atom_map[system_name] = []
                    continue
                atom_ids = parse_atom_id_list(atom_ids_str)
                atom_map[system_name] = atom_ids

        atom_ids = sorted({aid for aids in atom_map.values() for aid in aids})
        if not atom_ids:
            print("No se ingresó ningún átomo para comparar.")
            sys.exit(1)

        atom_labels = {aid: str(aid) for aid in atom_ids}
        use_labels = prompt_numbered_choice(
            "¿Desea asignar nombres personalizados a los átomos?",
            [("No", False), ("Sí", True)],
            default_idx=0
        )
        if use_labels:
            print("Ingrese un nombre para cada átomo; deje vacío para usar el número por defecto.")
            for aid in atom_ids:
                label = input(f"Nombre para el átomo {aid} (dejar vacío para usar '{aid}'): ").strip()
                if label:
                    atom_labels[aid] = label

        systems_data, missing, spin_labels_found = collect_global_hist_data(base_dir, atom_map, analysis_kind)
        if not systems_data:
            print("Error: no se encontraron análisis previos compatibles en las subcarpetas del directorio actual.")
            sys.exit(1)

        all_spin_labels = sorted({label for labels in spin_labels_found.values() for label in labels})
        spin_axis_label = None
        if all_spin_labels:
            if len(all_spin_labels) > 1:
                print("Error: se detectaron análisis globales mezclando representaciones de spin incompatibles:")
                for system_name in sorted(spin_labels_found):
                    labels = ", ".join(sorted(spin_labels_found[system_name]))
                    print(f"  {system_name}: {labels}")
                print("Reprocese las carpetas para que todas usen la misma representación de spin.")
                sys.exit(1)
            if all_spin_labels[0] == "spin_fraction":
                spin_axis_label = f"{analysis_kind.capitalize()} spin fraction"

        print("Sistemas incluidos en el gráfico:")
        for system_name in systems_data:
            atoms_for_system = " ".join(str(aid) for aid in sorted(systems_data[system_name].keys()))
            print(f"  {system_name}: {atoms_for_system}")

        if missing:
            print("[WARN] Faltan algunos archivos de series temporales y se omitieron del gráfico:")
            for system_name, aid, fname in missing[:20]:
                print(f"  {system_name}: átomo {aid} -> {os.path.basename(fname)}")
            if len(missing) > 20:
                print(f"  ... y {len(missing) - 20} casos más.")

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
            analysis_label=analysis_kind.capitalize(),
            spin_axis_label=spin_axis_label,
            fig_outname=fig_outname,
            bins_spec=global_bins_spec,
            percentile=95.0
        )
        return

    # Elegir programa
    prog = prompt_numbered_choice(
        "Programa con el cual se trabajará:",
        [("LIO", "lio"), ("ORCA", "orca")],
        default_idx=1
    )

    have_spin = False
    spin_sign = 1.0
    orca_mode = (prog == "orca")
    population_config = {"mulliken": True, "loewdin": False}
    active_charge_header = "# Mulliken Population Analysis"
    active_spin_header = "# Mulliken Spin Population Analysis"

    if prog == "lio":
        # Archivos de cargas
        mq_files = get_sorted_files("mq")
        if not mq_files:
            print("Error: no se encontraron archivos 'mq_*.dat' (cargas).")
            sys.exit(1)

        print("Archivos de cargas (mq_*.dat) que se van a combinar:")
        for f in mq_files:
            print("  ", f)
        merge_files(mq_files, "mq_full.dat")

        # Archivos de spin (opcional)
        ms_files = get_sorted_files("ms")
        have_spin = bool(ms_files)
        if have_spin:
            print("Archivos de spin (ms_*.dat) que se van a combinar:")
            for f in ms_files:
                print("  ", f)
            merge_files(ms_files, "ms_full.dat")
        else:
            print("[INFO] No se encontraron archivos 'ms_*.dat'. Solo se analizarán cargas.")

        charge_full = "mq_full.dat"
        spin_full = "ms_full.dat"
        spin_sign = -1.0
    else:
        # ORCA: cada <prefijo>_N.out/.dat es un frame
        population_choice = prompt_numbered_choice(
            "Análisis de población a procesar para ORCA:",
            [("Mulliken", "mulliken"),
             ("Loewdin", "loewdin"),
             ("Ambos", "both")],
            default_idx=2
        )
        population_config = get_population_analysis_config(population_choice)

        orca_prefix = input(
            "Ingrese el prefijo de los archivos ORCA antes de _N (por ejemplo, TD o SP; Enter = autodetectar): "
        ).strip()

        orca_files = get_sorted_orca_files(orca_prefix)
        if not orca_files:
            if orca_prefix:
                print(f"Error: no se encontraron archivos '{orca_prefix}_*.out' o '{orca_prefix}_*.dat'.")
            else:
                print("Error: no se encontraron archivos ORCA con formato '<prefijo>_N.out' o '<prefijo>_N.dat'.")
            sys.exit(1)

        if orca_prefix:
            print(f"Archivos ORCA ({orca_prefix}_*.out/.dat) que se van a analizar:")
        else:
            print("Archivos ORCA autodetectados (*_N.out/.dat) que se van a analizar:")
        for f in orca_files:
            print("  ", f)

        if population_config["mulliken"]:
            build_orca_full_files(
                orca_files,
                "mq_orca_todo.dat",
                "ms_orca_todo.dat",
                label="Mulliken",
                header_line="MULLIKEN ATOMIC CHARGES AND SPIN POPULATIONS"
            )
        if population_config["loewdin"]:
            build_orca_full_files(
                orca_files,
                "lq_orca_todo.dat",
                "ls_orca_todo.dat",
                label="Loewdin",
                header_line="LOEWDIN ATOMIC CHARGES AND SPIN POPULATIONS"
            )
        have_spin = True
        if population_config["mulliken"]:
            charge_full = "mq_orca_todo.dat"
            spin_full = "ms_orca_todo.dat"
            active_charge_header = "# Mulliken Population Analysis"
            active_spin_header = "# Mulliken Spin Population Analysis"
        else:
            charge_full = "lq_orca_todo.dat"
            spin_full = "ls_orca_todo.dat"
            active_charge_header = "# Loewdin Population Analysis"
            active_spin_header = "# Loewdin Spin Population Analysis"
        spin_sign = 1.0

    # Preguntar dt y átomos
    dt_str = input("Ingrese el valor del time step en picosegundos (por ejemplo, 0.001): ").strip()
    try:
        dt_ps = float(dt_str)
    except ValueError:
        print("Error: el time step debe ser un número (float).")
        sys.exit(1)

    # Mostrar lista de átomos disponibles
    atoms_list = get_atom_list_from_full(charge_full, active_charge_header, lio=(prog == "lio"))
    if atoms_list:
        print("\nLista de átomos disponibles (id, tipo):")
        for aid, atype in atoms_list:
            print(f"  {aid:4d}  {atype}")
        print("")

    atoms_str = input("Ingrese los números de átomo a trackear, separados por espacios (por ejemplo: 45 46 47): ").strip()
    try:
        atom_ids = [int(x) for x in atoms_str.split()]
    except ValueError:
        print("Error: los números de átomo deben ser enteros separados por espacios.")
        sys.exit(1)

    if not atom_ids:
        print("No se ingresó ningún átomo para trackear.")
        sys.exit(1)

    # Nombres personalizados para los átomos (opcional)
    atom_labels = {aid: str(aid) for aid in atom_ids}
    use_labels = prompt_numbered_choice(
        "¿Desea asignar nombres personalizados a los átomos?",
        [("No", False), ("Sí", True)],
        default_idx=0
    )
    if use_labels:
        print("Ingrese un nombre para cada átomo; deje vacío para usar el número por defecto.")
        for aid in atom_ids:
            label = input(f"Nombre para el átomo {aid} (por ejemplo, 'Fe'; dejar vacío para usar '{aid}'): ").strip()
            if label:
                atom_labels[aid] = label

    make_time_plots = prompt_numbered_choice(
        "¿Desea generar gráficos de carga/spin en función del tiempo?",
        [("No", False), ("Sí", True)],
        default_idx=0
    )

    hist_binning_choice = prompt_numbered_choice(
        "Binning para histogramas del análisis individual:",
        [("Automático (Freedman-Diaconis)", "fd"),
         ("Automático (numpy auto)", "auto"),
         ("Automático (Sturges)", "sturges"),
         ("Fijo: elegir cantidad", "fixed_custom")],
        default_idx=0
    )
    hist_bins_spec = resolve_histogram_bins_spec(hist_binning_choice)

    spin_mode = prompt_numbered_choice(
        "Representación para el análisis de spin:",
        [("Spin directo del output", "raw"),
         ("Spin como fracción del total de los átomos elegidos", "fraction")],
        default_idx=1
    )
    spin_config = get_spin_representation_config(spin_mode)

    spin_keep_mask = None
    if have_spin:
        run_spin_check = prompt_numbered_choice(
            "Chequeo de consistencia de spin por snapshot:",
            [("No realizar chequeo", False),
             ("Comparar la suma de los átomos elegidos con el spin total del sistema", True)],
            default_idx=1
        )
        if run_spin_check:
            tol_str = input(
                "Ingrese la tolerancia permitida para |suma_spines_seleccionados - spin_total| (Enter = 0.20): "
            ).strip()
            if tol_str == "":
                spin_tolerance = 0.20
            else:
                try:
                    spin_tolerance = float(tol_str)
                except ValueError:
                    print("Error: la tolerancia debe ser un número.")
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
                    "Snapshots con mala consistencia de spin detectados:",
                    [("Mantener todos los snapshots", False),
                     ("Eliminar del análisis los snapshots fuera de tolerancia", True)],
                    default_idx=0
                )
                if not filter_bad:
                    spin_keep_mask = None
                elif spin_keep_mask.sum() == 0:
                    print("Error: todos los snapshots quedaron fuera de tolerancia.")
                    sys.exit(1)
                else:
                    n_kept = int(spin_keep_mask.sum())
                    n_removed = int((~spin_keep_mask).sum())
                    print(
                        f"[INFO] Filtrado por consistencia de spin aplicado: "
                        f"{n_removed} snapshots eliminados, {n_kept} snapshots conservados."
                    )
                   

    if orca_mode and population_config["mulliken"]:
        q_ts_out = "mulliken_charge_timeseries.dat"
        q_avg_out = "mulliken_charge_averages.dat"
        q_hist_prefix = "mulliken_charge_hist"
        q_modes_out = "mulliken_charge_modes.dat"
        s_ts_out = "mulliken_spin_timeseries.dat"
        s_avg_out = "mulliken_spin_averages.dat"
        s_hist_prefix = "mulliken_spin_hist"
        s_modes_out = "mulliken_spin_modes.dat"
        mulliken_fig_out = "mulliken_histograms.png"
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

    if not orca_mode or population_config["mulliken"]:
        mulliken_charge_mask = apply_keep_mask_or_warn(
            spin_keep_mask,
            parse_frame_data(charge_full, dt_ps, atom_ids, "charge", active_charge_header)[0].size,
            "Mulliken charge" if orca_mode else "charge"
        )
        # --- Análisis de cargas ---
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

        # --- Análisis de spin, si existe ---
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
            print("[INFO] No hay análisis de spin porque no se encontraron ms_*.dat.")

        for aid in atom_ids:
            q_vals = np.asarray(per_atom_q.get(aid, []), dtype=float)
            if times.size != q_vals.size:
                n = min(times.size, q_vals.size)
                t_use = times[:n]
                q_use = q_vals[:n]
            else:
                t_use = times
                q_use = q_vals

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

            print(f"[OK] Serie temporal conjunta t, q, s para átomo {aid} escrita en '{atom_out}'.")

        make_combined_hist_figure(
            atom_ids,
            hist_charge=hist_q,
            hist_spin=hist_s if have_spin else {},
            atom_labels=atom_labels,
            fig_outname=mulliken_fig_out
        )

        if make_time_plots:
            mulliken_ts_fig_out = "mulliken_timeseries.png" if orca_mode else "qs_timeseries.png"
            make_timeseries_figure(
                times,
                per_atom_q,
                per_atom_s if have_spin else {},
                atom_ids,
                atom_labels=atom_labels,
                fig_outname=mulliken_ts_fig_out,
                spin_ylabel=spin_axis_label
            )

        if orca_mode and mulliken_fig_out != "qs_histograms.png":
            make_combined_hist_figure(
                atom_ids,
                hist_charge=hist_q,
                hist_spin=hist_s if have_spin else {},
                atom_labels=atom_labels,
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
                    spin_ylabel=spin_axis_label
                )

    # --- ORCA: análisis Loewdin adicional ---
    if orca_mode and population_config["loewdin"]:
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

            print(f"[OK] Serie temporal Loewdin para átomo {aid} escrita en '{atom_out}'.")

        make_combined_hist_figure(
            atom_ids,
            hist_charge=hist_q_l,
            hist_spin=hist_s_l,
            atom_labels=atom_labels,
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
                spin_ylabel=spin_axis_label
            )


if __name__ == "__main__":
    main()
