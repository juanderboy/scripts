#!/usr/bin/env python3
"""Plot generation for charge and spin population analyses.

The routines build combined histogram panels and time-series figures from the
statistics prepared by the analysis layer.
"""

import matplotlib.pyplot as plt
import numpy as np


def make_combined_hist_figure(
    atom_ids,
    hist_charge,
    hist_spin,
    atom_labels=None,
    charge_axis_label="Mulliken charge",
    spin_axis_label="Mulliken spin",
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
                axis_label = info.get("axis_label") or spin_axis_label
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
