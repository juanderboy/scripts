#!/usr/bin/env python3
"""Cross-system summaries for previously generated charge/spin analyses.

This module collects per-system time series and builds comparable histograms,
figures and reports for selected atoms or grouped entities.
"""

import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from charge_spin_common import sanitize_output_token
from charge_spin_io import load_atom_timeseries_file
from charge_spin_stats import analyze_modes_kde, get_histogram_edges


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
                "hirshfeld_histograms.png",
                "chelpg_histograms.png",
                "chelpg_loewdin_histograms.png",
                "chelpg_mulliken_histograms.png",
                "chelpg_hirshfeld_histograms.png",
                "qs_histograms_with_coque.png",
                "mulliken_histograms_with_coque.png",
                "loewdin_histograms_with_coque.png",
                "hirshfeld_histograms_with_coque.png",
                "chelpg_histograms_with_coque.png",
                "chelpg_loewdin_histograms_with_coque.png",
                "chelpg_mulliken_histograms_with_coque.png",
                "chelpg_hirshfeld_histograms_with_coque.png",
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
        "hirshfeld": ["hirshfeld"],
        "chelpg": ["chelpg", "chelpg_loewdin"],
        "chelpg_loewdin": ["chelpg_loewdin", "chelpg"],
        "chelpg_mulliken": ["chelpg_mulliken"],
        "chelpg_hirshfeld": ["chelpg_hirshfeld"],
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
                if aid == "coque":
                    candidate = os.path.join(system_dir, f"actor_proposed_spin_pool_{suffix}_timeseries.dat")
                else:
                    candidate = os.path.join(system_dir, f"atom_{aid}_{suffix}_timeseries.dat")
                if os.path.exists(candidate):
                    fname = candidate
                    break
            if fname is None:
                if aid == "coque":
                    expected = os.path.join(system_dir, f"actor_proposed_spin_pool_{suffixes[0]}_timeseries.dat")
                else:
                    expected = os.path.join(system_dir, f"atom_{aid}_{suffixes[0]}_timeseries.dat")
                missing.append((system_name, aid, expected))
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


def get_global_analysis_suffixes(analysis_kind):
    """
    Return the filename suffixes associated with a global analysis kind.
    """
    suffix_map = {
        "mulliken": ["qs"],
        "loewdin": ["loewdin"],
        "hirshfeld": ["hirshfeld"],
        "chelpg": ["chelpg", "chelpg_loewdin"],
        "chelpg_loewdin": ["chelpg_loewdin", "chelpg"],
        "chelpg_mulliken": ["chelpg_mulliken"],
        "chelpg_hirshfeld": ["chelpg_hirshfeld"],
        "all": ["hirshfeld", "loewdin", "qs", "chelpg_hirshfeld", "chelpg_mulliken", "chelpg_loewdin"],
    }
    return suffix_map[analysis_kind]


def infer_entities_from_previous_analysis(system_dir, analysis_kind):
    """
    Infer the entities selected during a previous individual analysis in one system.
    """
    entity_ids = []
    for suffix in get_global_analysis_suffixes(analysis_kind):
        found_for_suffix = []
        pattern = os.path.join(system_dir, f"atom_*_{suffix}_timeseries.dat")
        for fname in sorted(glob.glob(pattern)):
            m = re.match(rf"^atom_(\d+)_{re.escape(suffix)}_timeseries\.dat$", os.path.basename(fname))
            if m:
                found_for_suffix.append(int(m.group(1)))

        actor_fname = os.path.join(system_dir, f"actor_proposed_spin_pool_{suffix}_timeseries.dat")
        if os.path.exists(actor_fname):
            found_for_suffix.append("coque")

        if found_for_suffix:
            deduped = []
            for entity_id in found_for_suffix:
                if entity_id not in deduped:
                    deduped.append(entity_id)
            return deduped

    return []


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
        f_values.write("# system entity property value\n")
        f_ranges.write("# entity property x_min x_max n_total n_used percentile_central\n")

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
                        f_values.write(f"{system_name} {sanitize_output_token(aid)} {prop} {val:.7f}\n")

                if xlim is not None:
                    f_ranges.write(
                        f"{sanitize_output_token(aid)} {prop} {xlim[0]:.7f} {xlim[1]:.7f} {n_total:d} {n_used:d} {percentile:.1f}\n"
                    )
                else:
                    f_ranges.write(
                        f"{sanitize_output_token(aid)} {prop} nan nan {n_total:d} {n_used:d} {percentile:.1f}\n"
                    )

    print(f"[OK] Global histogram values saved to '{values_out}'.")
    print(f"[OK] Global plotting ranges saved to '{ranges_out}'.")


def export_global_snapshot_counts(output_dir, atom_ids, systems_data, analysis_kind):
    """
    Export how many snapshots contribute to each per-system histogram.
    """
    os.makedirs(output_dir, exist_ok=True)
    counts_out = os.path.join(output_dir, f"{analysis_kind}_global_snapshot_counts.dat")

    with open(counts_out, "w") as out:
        out.write("# system entity property n_snapshots\n")
        for system_name in systems_data:
            for aid in atom_ids:
                atom_data = systems_data[system_name].get(aid)
                if atom_data is None:
                    continue
                for prop in ("charge", "spin"):
                    vals = np.asarray(atom_data.get(prop, []), dtype=float)
                    n_snapshots = int(vals[~np.isnan(vals)].size)
                    out.write(
                        f"{system_name} {sanitize_output_token(aid)} {prop} {n_snapshots:d}\n"
                    )

    print(f"[OK] Global snapshot counts saved to '{counts_out}'.")


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
