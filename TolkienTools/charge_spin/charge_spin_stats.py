#!/usr/bin/env python3
"""Statistical analysis for charge and spin population time series.

This module computes summaries, histograms, modes, grouped actors and automatic
spin-localization selections from parsed population frames.
"""

import sys

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde

from charge_spin_common import format_summary_stat, sanitize_output_token
from charge_spin_io import get_atom_list_from_full, parse_frame_data


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


def build_terminal_summary_entry(analysis_label, entity_ids, atom_labels, per_atom_charge, per_atom_spin, hist_charge, hist_spin):
    """
    Build one summary section for terminal display.
    """
    rows = []
    for entity_id in entity_ids:
        q_vals = np.asarray(per_atom_charge.get(entity_id, []), dtype=float)
        s_vals = np.asarray(per_atom_spin.get(entity_id, []), dtype=float)
        q_hist = hist_charge.get(entity_id, {})
        s_hist = hist_spin.get(entity_id, {})
        q_peaks = np.asarray(q_hist.get("peaks", []), dtype=float)
        s_peaks = np.asarray(s_hist.get("peaks", []), dtype=float)

        rows.append(
            {
                "entity": atom_labels.get(entity_id, str(entity_id)),
                "q_n": int(q_vals.size),
                "q_avg": float(q_vals.mean()) if q_vals.size > 0 else None,
                "q_peak": float(q_peaks[0]) if q_peaks.size > 0 else None,
                "q_peaks": [float(val) for val in q_peaks],
                "s_n": int(s_vals.size),
                "s_avg": float(s_vals.mean()) if s_vals.size > 0 else None,
                "s_peak": float(s_peaks[0]) if s_peaks.size > 0 else None,
                "s_peaks": [float(val) for val in s_peaks],
            }
        )

    return {"analysis": analysis_label, "rows": rows}


def print_terminal_analysis_summary(summary_entries):
    """
    Print a compact boxed summary at the end of a single-system analysis.
    """
    if not summary_entries:
        return

    lines = ["STATISTICAL SUMMARY"]
    for entry in summary_entries:
        lines.append(f"Analysis: {entry['analysis']}")
        lines.append("entity                q_n     q_avg    q_peak     s_n     s_avg    s_peak")
        for row in entry["rows"]:
            lines.append(
                f"{row['entity'][:20]:20s} "
                f"{row['q_n']:6d} "
                f"{format_summary_stat(row['q_avg']):>9s} "
                f"{format_summary_stat(row['q_peak']):>9s} "
                f"{row['s_n']:6d} "
                f"{format_summary_stat(row['s_avg']):>9s} "
                f"{format_summary_stat(row['s_peak']):>9s}"
            )
            if row["q_peaks"]:
                q_modes = ", ".join(format_summary_stat(val) for val in row["q_peaks"])
                lines.append(f"  q_modes: {q_modes}")
            if row["s_peaks"]:
                s_modes = ", ".join(format_summary_stat(val) for val in row["s_peaks"])
                lines.append(f"  s_modes: {s_modes}")
        lines.append("")

    content_width = max(len(line) for line in lines)
    border = "+" + "-" * (content_width + 2) + "+"
    print("")
    print(border)
    for line in lines:
        print(f"| {line.ljust(content_width)} |")
    print(border)


def build_global_terminal_summary_entry(analysis_label, systems_data, entity_ids, atom_labels):
    """
    Build a global summary section grouped by system and entity.
    """
    rows = []
    for system_name in systems_data:
        for entity_id in entity_ids:
            entity_data = systems_data[system_name].get(entity_id)
            if entity_data is None:
                continue

            q_vals = np.asarray(entity_data.get("charge", []), dtype=float)
            s_vals = np.asarray(entity_data.get("spin", []), dtype=float)
            q_vals = q_vals[~np.isnan(q_vals)]
            s_vals = s_vals[~np.isnan(s_vals)]
            q_peaks = analyze_modes_kde(q_vals)[2] if q_vals.size > 0 else np.array([])
            s_peaks = analyze_modes_kde(s_vals)[2] if s_vals.size > 0 else np.array([])

            rows.append(
                {
                    "system": system_name,
                    "entity": atom_labels.get(entity_id, str(entity_id)),
                    "q_n": int(q_vals.size),
                    "q_avg": float(q_vals.mean()) if q_vals.size > 0 else None,
                    "q_peak": float(q_peaks[0]) if q_peaks.size > 0 else None,
                    "q_peaks": [float(val) for val in q_peaks],
                    "s_n": int(s_vals.size),
                    "s_avg": float(s_vals.mean()) if s_vals.size > 0 else None,
                    "s_peak": float(s_peaks[0]) if s_peaks.size > 0 else None,
                    "s_peaks": [float(val) for val in s_peaks],
                }
            )

    return {"analysis": analysis_label, "rows": rows}


def print_terminal_global_summary(summary_entries):
    """
    Print a compact boxed summary at the end of global analysis.
    """
    if not summary_entries:
        return

    lines = ["GLOBAL STATISTICAL SUMMARY"]
    for entry in summary_entries:
        lines.append(f"Analysis: {entry['analysis']}")
        lines.append("system           entity                q_n     q_avg    q_peak     s_n     s_avg    s_peak")
        for row in entry["rows"]:
            lines.append(
                f"{row['system'][:15]:15s} "
                f"{row['entity'][:20]:20s} "
                f"{row['q_n']:6d} "
                f"{format_summary_stat(row['q_avg']):>9s} "
                f"{format_summary_stat(row['q_peak']):>9s} "
                f"{row['s_n']:6d} "
                f"{format_summary_stat(row['s_avg']):>9s} "
                f"{format_summary_stat(row['s_peak']):>9s}"
            )
            if row["q_peaks"]:
                q_modes = ", ".join(format_summary_stat(val) for val in row["q_peaks"])
                lines.append(f"  q_modes: {q_modes}")
            if row["s_peaks"]:
                s_modes = ", ".join(format_summary_stat(val) for val in row["s_peaks"])
                lines.append(f"  s_modes: {s_modes}")
        lines.append("")

    content_width = max(len(line) for line in lines)
    border = "+" + "-" * (content_width + 2) + "+"
    print("")
    print(border)
    for line in lines:
        print(f"| {line.ljust(content_width)} |")
    print(border)


def build_analysis_entities(atom_ids, atom_labels, actor_config=None):
    """
    Return the entities that will be plotted/analyzed.
    """
    entities = [
        {"id": aid, "label": atom_labels.get(aid, str(aid)), "atom_ids": [aid]}
        for aid in atom_ids
    ]
    if actor_config is not None:
        entities.append(
            {
                "id": actor_config["id"],
                "label": actor_config["label"],
                "atom_ids": list(actor_config["atom_ids"]),
            }
        )
    return entities


def get_analysis_entity_ids(atom_ids, actor_config=None):
    """
    Return the ordered identifiers used in plots/combined outputs.
    """
    entity_ids = list(atom_ids)
    if actor_config is not None:
        entity_ids.append(actor_config["id"])
    return entity_ids


def build_grouped_timeseries_and_stats(
    fullfile,
    dt_ps,
    entities,
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
    Analyze a merged file for arbitrary entities, where each entity can be one atom
    or a grouped sum of several atoms.
    """
    entities = list(entities)
    unique_atom_ids = []
    for entity in entities:
        for aid in entity["atom_ids"]:
            if aid not in unique_atom_ids:
                unique_atom_ids.append(aid)

    times, raw_values, _frame_totals = parse_frame_data(
        fullfile,
        dt_ps,
        unique_atom_ids,
        kind,
        header_start,
        spin_sign=spin_sign
    )

    if keep_mask is not None:
        keep_mask = np.asarray(keep_mask, dtype=bool)
        if keep_mask.size != times.size:
            print(
                f"[WARN] An incompatible snapshot mask was ignored for '{fullfile}': "
                f"mask={keep_mask.size}, frames={times.size}."
            )
        else:
            times = times[keep_mask]
            raw_values = raw_values[keep_mask, :]

    grouped_values = np.full((times.size, len(entities)), np.nan, dtype=float)
    atom_idx_map = {aid: idx for idx, aid in enumerate(unique_atom_ids)}
    for entity_idx, entity in enumerate(entities):
        group_indices = [atom_idx_map[aid] for aid in entity["atom_ids"] if aid in atom_idx_map]
        if not group_indices:
            continue
        group_matrix = raw_values[:, group_indices]
        valid_rows = ~np.isnan(group_matrix).any(axis=1)
        if np.any(valid_rows):
            grouped_values[valid_rows, entity_idx] = np.sum(group_matrix[valid_rows, :], axis=1)

    if kind == "spin" and normalize_spin_fraction:
        grouped_values = normalize_selected_spin_values(grouped_values)

    data = np.column_stack((times, grouped_values)) if times.size > 0 else np.empty((0, len(entities) + 1))

    with open(ts_outname, "w") as out:
        if kind == "spin" and normalize_spin_fraction:
            header = ["time_ps"] + [f"spin_fraction_{sanitize_output_token(entity['id'])}" for entity in entities]
        else:
            header = ["time_ps"] + [f"{kind}_{sanitize_output_token(entity['id'])}" for entity in entities]
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
        values = np.empty((0, len(entities)))
    else:
        data = np.array(data, dtype=float)
        times = data[:, 0]
        values = data[:, 1:]

    per_entity_values = {entity["id"]: np.array([]) for entity in entities}
    for i, entity in enumerate(entities):
        col = values[:, i]
        mask = ~np.isnan(col)
        per_entity_values[entity["id"]] = col[mask]

    with open(avg_outname, "w") as out:
        if kind == "spin" and normalize_spin_fraction:
            out.write("# Spin averages as fractions of the total spin of the displayed entities\n")
        else:
            out.write(f"# Average {kind} values\n")
        out.write("# entity_id  avg_value\n")
        for entity in entities:
            entity_id = entity["id"]
            vals = per_entity_values[entity_id]
            if vals.size > 0:
                out.write(f"{sanitize_output_token(entity_id)}  {vals.mean(): .7f}\n")
            else:
                out.write(f"{sanitize_output_token(entity_id)}  nan\n")

    print(f"[OK] Average {kind} values written to '{avg_outname}'.")

    modes_summary = []
    hist_data_for_plot = {}

    for i, entity in enumerate(entities):
        entity_id = entity["id"]
        vals = per_entity_values[entity_id]
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            print(f"[WARN] No valid {kind} data found for entity {entity['label']}.")
            continue

        vmin, vmax = vals.min(), vals.max()
        safe_id = sanitize_output_token(entity_id)

        if vmin == vmax:
            xs = np.array([vmin])
            dens = np.array([1.0])
            peaks = np.array([vmin])
            peak_heights = np.array([1.0])

            hist_outname = f"{hist_prefix}_entity_{safe_id}.dat"
            with open(hist_outname, "w") as out:
                out.write("# Histogram: bin_center  count\n")
                out.write(f"{vmin: .7f} {vals.size}\n")
                out.write("\n# KDE: x  density\n")
                out.write(f"{vmin: .7f} {dens[0]: .7e}\n")
            print(f"[OK] Trivial {kind} histogram for entity {entity['label']} written to '{hist_outname}'.")

            modes_summary.append((safe_id, peaks))
            hist_data_for_plot[entity_id] = {
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
        xs, dens, peak_positions, peak_heights = analyze_modes_kde(vals)

        hist_outname = f"{hist_prefix}_entity_{safe_id}.dat"
        with open(hist_outname, "w") as out:
            out.write("# Histogram: bin_center  count\n")
            for c, center in zip(counts, bin_centers):
                out.write(f"{center: .7f} {c:d}\n")
            out.write("\n# KDE: x  density\n")
            if xs.size > 0 and dens.size > 0:
                for x_val, d_val in zip(xs, dens):
                    out.write(f"{x_val: .7f} {d_val: .7e}\n")

        print(f"[OK] {kind.capitalize()} histogram for entity {entity['label']} written to '{hist_outname}'.")

        modes_summary.append((safe_id, peak_positions))
        if len(peak_positions) == 1:
            print(f"    Entity {entity['label']}: unimodal {kind} distribution. Peak ≈ {peak_positions[0]: .4f}")
        else:
            print(f"    Entity {entity['label']}: {len(peak_positions)} modes detected in {kind}.")
            for k, mu in enumerate(peak_positions):
                print(f"        Mode {k+1}: {kind} ≈ {mu: .4f}")

        hist_data_for_plot[entity_id] = {
            "bin_centers": bin_centers,
            "counts_norm": counts_norm,
            "xs": xs,
            "dens": dens,
            "peaks": peak_positions,
            "axis_label": "Spin fraction" if kind == "spin" and normalize_spin_fraction else None
        }

    with open(modes_outname, "w") as out:
        out.write(f"# entity_id  n_modes  peak_positions_{kind}(approx)\n")
        for safe_id, peaks in modes_summary:
            if len(peaks) == 0:
                out.write(f"{safe_id}  0\n")
            else:
                peaks_str = " ".join(f"{mu: .6f}" for mu in peaks)
                out.write(f"{safe_id}  {len(peaks):2d}  {peaks_str}\n")

    print(f"[OK] {kind.capitalize()} mode summary written to '{modes_outname}'.")

    return times, per_entity_values, hist_data_for_plot


def build_analysis_timeseries_and_stats(
    fullfile,
    dt_ps,
    atom_ids,
    atom_labels,
    actor_config,
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
    Wrapper that switches to grouped analysis when an extra actor is present.
    """
    if actor_config is None:
        return build_timeseries_and_stats(
            fullfile,
            dt_ps,
            atom_ids,
            kind,
            header_start,
            ts_outname,
            avg_outname,
            hist_prefix,
            modes_outname,
            nbins_hist=nbins_hist,
            spin_sign=spin_sign,
            keep_mask=keep_mask,
            normalize_spin_fraction=normalize_spin_fraction
        )

    entities = build_analysis_entities(atom_ids, atom_labels, actor_config=actor_config)
    return build_grouped_timeseries_and_stats(
        fullfile,
        dt_ps,
        entities,
        kind,
        header_start,
        ts_outname,
        avg_outname,
        hist_prefix,
        modes_outname,
        nbins_hist=nbins_hist,
        spin_sign=spin_sign,
        keep_mask=keep_mask,
        normalize_spin_fraction=normalize_spin_fraction
    )


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
        return [], []

    atom_info = get_atom_list_from_full(spin_full, header_start)
    all_atom_ids = [aid for aid, _atype in atom_info]
    atom_types = {aid: atype for aid, atype in atom_info}
    omitted_atom_ids = [aid for aid in all_atom_ids if aid not in set(selected_atom_ids)]
    if not omitted_atom_ids:
        return [], []

    times_all, all_spin_values, spin_totals = parse_frame_data(
        spin_full,
        dt_ps,
        all_atom_ids,
        kind="spin",
        header_start=header_start,
        spin_sign=spin_sign
    )
    if times_all.size == 0 or times_all.size != bad_mask.size:
        return [], []

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

    chosen_atom_ids = [item["atom_id"] for item in chosen]
    return suggestions, chosen_atom_ids


def select_spin_localization_atoms(
    spin_full,
    header_start,
    spin_sign,
    min_atom_fraction=0.05,
    report_outname="spin_localization_auto_selection.dat",
    lio=False,
):
    """
    Select atoms that carry most of the spin density on average.

    Fractions are computed per snapshot as abs(spin_i) / sum_j abs(spin_j),
    then averaged over snapshots. Atoms above min_atom_fraction get individual
    histograms; all other atoms are grouped into a "resto" actor.
    """
    atom_info = get_atom_list_from_full(spin_full, header_start, lio=lio)
    if not atom_info:
        print("Error: no atoms were found in the spin population file.")
        sys.exit(1)

    all_atom_ids = [aid for aid, _atype in atom_info]
    atom_types = {aid: atype for aid, atype in atom_info}
    _times, spin_values, _spin_totals = parse_frame_data(
        spin_full,
        1.0,
        all_atom_ids,
        kind="spin",
        header_start=header_start,
        spin_sign=spin_sign,
    )
    if spin_values.size == 0:
        print("Error: no spin frames were found for automatic spin-localization analysis.")
        sys.exit(1)

    abs_values = np.abs(spin_values)
    denom = np.nansum(abs_values, axis=1)
    valid_rows = denom > 1e-12
    if not np.any(valid_rows):
        print("Error: all spin frames have zero total absolute spin.")
        sys.exit(1)

    fractions = np.full_like(abs_values, np.nan, dtype=float)
    fractions[valid_rows, :] = abs_values[valid_rows, :] / denom[valid_rows, np.newaxis]
    avg_fraction = np.nanmean(fractions, axis=0)
    mean_spin = np.nanmean(spin_values, axis=0)
    mean_abs_spin = np.nanmean(abs_values, axis=0)

    order = [idx for idx in np.argsort(avg_fraction)[::-1] if not np.isnan(avg_fraction[idx])]
    own_atoms = [
        all_atom_ids[idx] for idx in order
        if float(avg_fraction[idx]) >= min_atom_fraction
    ]
    if not own_atoms:
        own_atoms = [all_atom_ids[order[0]]]

    rest_atoms = [aid for aid in all_atom_ids if aid not in set(own_atoms)]
    actor_config = None
    if rest_atoms:
        actor_config = {
            "id": "actor_resto",
            "label": "resto",
            "atom_ids": rest_atoms,
        }

    with open(report_outname, "w") as out:
        out.write(
            "# atom_id atom_type mean_spin mean_abs_spin avg_abs_spin_fraction "
            "cumulative_fraction own_histogram group\n"
        )
        running = 0.0
        own_set = set(own_atoms)
        for idx in order:
            aid = all_atom_ids[idx]
            frac = float(avg_fraction[idx])
            if np.isnan(frac):
                frac = 0.0
            running += frac
            group = "own" if aid in own_set else "resto"
            out.write(
                f"{aid:d} {atom_types.get(aid, '?')} {mean_spin[idx]:.7f} "
                f"{mean_abs_spin[idx]:.7f} {frac:.7f} {running:.7f} "
                f"{int(aid in own_set)} {group}\n"
            )

    own_labels = [
        f"{aid}({atom_types.get(aid, '?')}, {100.0 * float(avg_fraction[all_atom_ids.index(aid)]):.1f}%)"
        for aid in own_atoms
    ]
    print(
        f"[INFO] Individual spin histograms will be generated for atoms above "
        f"{100.0 * min_atom_fraction:.1f}%: {', '.join(own_labels)}"
    )
    if rest_atoms:
        print(f"[INFO] The remaining {len(rest_atoms)} atoms will be grouped as 'resto'.")
    print(f"[OK] Automatic spin-localization report saved to '{report_outname}'.")

    return own_atoms, actor_config, all_atom_ids, atom_types, avg_fraction
