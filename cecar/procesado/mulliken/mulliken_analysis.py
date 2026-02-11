#!/usr/bin/env python3
import glob
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


def merge_files(files, outname):
    """
    Concatena los archivos de la lista 'files' en 'outname'.
    """
    with open(outname, "w") as out:
        for fname in files:
            with open(fname, "r") as inp:
                for line in inp:
                    out.write(line)
    print(f"[OK] Archivo combinado escrito en '{outname}'.")


def analyze_modes_kde(vals, n_grid=512, prominence_factor=0.05):
    """
    Usa KDE (gaussian_kde) + find_peaks para detectar modos en la distribución
    de 'vals'.

    Devuelve:
      - xs: grid donde se evaluó la densidad
      - dens: valores de densidad en xs
      - peak_positions: posiciones de los máximos
      - peak_heights: densidad en cada máximo
    """
    vals = np.asarray(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    vmin = vals.min()
    vmax = vals.max()
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


# ==========================
# Lectura y análisis genérico para mq/ms
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
    nbins_hist=50
):
    """
    Analiza un archivo full (mq_full.dat o ms_full.dat).

    kind: 'charge' o 'spin'
    header_start:
        - "# Mulliken Population Analysis"  (cargas)
        - "# Mulliken Spin Population Analysis" (spin)
    """

    atom_ids = list(atom_ids)
    data = []  # [time_ps, val_atom1, val_atom2, ...]
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
                if atom_idx in current_vals:
                    current_vals[atom_idx] = value

    # Serie temporal
    with open(ts_outname, "w") as out:
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

    print(f"[OK] Serie temporal ({kind}) escrita en '{ts_outname}' con {len(data)} frames.")
    print(f"    Columnas: {', '.join(header)}")

    # Listas por átomo
    per_atom_values = {aid: [] for aid in atom_ids}
    for row in data:
        for j, aid in enumerate(atom_ids):
            v = row[1 + j]
            if isinstance(v, float) and not math.isnan(v):
                per_atom_values[aid].append(v)

    # Promedios
    averages = {}
    with open(avg_outname, "w") as out:
        out.write(f"# atom_id average_{kind} (N frames válidos)\n")
        for aid in atom_ids:
            vals = np.asarray(per_atom_values[aid], dtype=float)
            vals = vals[~np.isnan(vals)]
            if vals.size == 0:
                avg = float("nan")
                out.write(f"{aid:4d}   NaN\n")
            else:
                avg = vals.mean()
                out.write(f"{aid:4d}   {avg: .7f}\n")
            averages[aid] = avg

    print(f"[OK] Promedios de {kind} escritos en '{avg_outname}'.")
    for aid in atom_ids:
        avg = averages[aid]
        if math.isnan(avg):
            print(f"      Átomo {aid:4d}: promedio {kind} = NaN")
        else:
            print(f"      Átomo {aid:4d}: promedio {kind} = {avg: .7f}")

    # Histogramas + modos
    modes_summary = []
    hist_data_for_plot = {}

    for aid in atom_ids:
        vals = np.asarray(per_atom_values[aid], dtype=float)
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            print(f"[AVISO] No hay datos válidos de {kind} para el átomo {aid}.")
            modes_summary.append((aid, np.array([]), np.array([])))
            continue

        vmin = vals.min()
        vmax = vals.max()

        if vmin == vmax:
            # Caso degenerado
            hist_outname = f"{hist_prefix}_atom_{aid}.dat"
            with open(hist_outname, "w") as out:
                out.write("# bin_center  count\n")
                out.write(f"{vmin: .7f} {vals.size}\n")
            print(f"[OK] Histograma (trivial) de {kind} para átomo {aid} en '{hist_outname}'.")

            xs = np.array([vmin])
            dens = np.array([1.0])
            peaks = np.array([vmin])
            peak_heights = np.array([1.0])

            modes_summary.append((aid, peaks, peak_heights))
            hist_data_for_plot[aid] = {
                "bin_centers": np.array([vmin]),
                "counts_norm": np.array([1.0]),
                "xs": xs,
                "dens": dens,
                "peaks": peaks
            }
            continue

        counts, bin_edges = np.histogram(vals, bins=nbins_hist, range=(vmin, vmax))
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        counts_norm = counts / counts.sum()

        hist_outname = f"{hist_prefix}_atom_{aid}.dat"
        with open(hist_outname, "w") as out:
            out.write("# bin_center  count\n")
            for c, center in zip(counts, bin_centers):
                out.write(f"{center: .7f} {c:d}\n")

        print(f"[OK] Histograma de {kind} para átomo {aid} escrito en '{hist_outname}'.")

        xs, dens, peak_positions, peak_heights = analyze_modes_kde(vals)
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
            "peaks": peak_positions
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

    return np.array(data), per_atom_values, hist_data_for_plot


def make_combined_hist_figure(
    atom_ids,
    hist_charge,
    hist_spin,
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
        # CARGAS (columna 0)
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
            ax_q.set_ylabel(f"Átomo {aid}")
            ax_q.set_xlabel("Carga de Mulliken")
            ax_q.legend(fontsize=8)
        else:
            ax_q.text(0.5, 0.5, "Sin datos de q", transform=ax_q.transAxes,
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
                ax_s.set_xlabel("Spin de Mulliken")
                ax_s.legend(fontsize=8)
            else:
                ax_s.text(0.5, 0.5, "Sin datos de s", transform=ax_s.transAxes,
                          ha="center", va="center")
                ax_s.set_axis_off()

    plt.tight_layout()
    fig.savefig(fig_outname, dpi=300)
    plt.close(fig)
    print(f"[OK] Figura combinada cargas/spines guardada en '{fig_outname}'.")


# ==========================
# MAIN
# ==========================

def main():
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
        print("[AVISO] No se encontraron archivos 'ms_*.dat'. Solo se analizarán cargas.")

    # Preguntar dt y átomos
    dt_str = input("Ingrese el valor del time step en picosegundos (por ejemplo, 0.001): ").strip()
    try:
        dt_ps = float(dt_str)
    except ValueError:
        print("Error: el time step debe ser un número (float).")
        sys.exit(1)

    atoms_str = input("Ingrese los números de átomo a trackear, separados por espacios (por ejemplo: 45 46 47): ").strip()
    try:
        atom_ids = [int(x) for x in atoms_str.split()]
    except ValueError:
        print("Error: los números de átomo deben ser enteros separados por espacios.")
        sys.exit(1)

    if not atom_ids:
        print("No se ingresó ningún átomo para trackear.")
        sys.exit(1)

    # --- Análisis de cargas (mq_full.dat) ---
    charge_data, per_atom_q, hist_q = build_timeseries_and_stats(
        "mq_full.dat",
        dt_ps,
        atom_ids,
        kind="charge",
        header_start="# Mulliken Population Analysis",
        ts_outname="mq_charge_timeseries.dat",
        avg_outname="mq_charge_averages.dat",
        hist_prefix="mq_charge_hist",
        modes_outname="mq_charge_modes.dat",
        nbins_hist=50
    )

    times = charge_data[:, 0] if len(charge_data) > 0 else np.array([])

    # --- Análisis de spines (ms_full.dat), si existen ---
    hist_s = {}
    per_atom_s = {aid: [] for aid in atom_ids}
    if have_spin:
        spin_data, per_atom_s, hist_s = build_timeseries_and_stats(
            "ms_full.dat",
            dt_ps,
            atom_ids,
            kind="spin",
            header_start="# Mulliken Spin Population Analysis",
            ts_outname="ms_spin_timeseries.dat",
            avg_outname="ms_spin_averages.dat",
            hist_prefix="ms_spin_hist",
            modes_outname="ms_spin_modes.dat",
            nbins_hist=50
        )
        # por simplicidad asumimos misma cantidad de frames y dt
        # (ORCA suele imprimir mq y ms por cada step)
        # si no matchean, se podría hacer un chequeo adicional.
    else:
        print("[INFO] No hay análisis de spin porque no se encontraron ms_*.dat.")

    # --- Archivos por átomo: t, q, s (o t, q si no hay s) ---
    for aid in atom_ids:
        q_vals = np.asarray(per_atom_q.get(aid, []), dtype=float)
        if times.size != q_vals.size:
            # en principio no debería pasar, pero por las dudas:
            n = min(times.size, q_vals.size)
            t_use = times[:n]
            q_use = q_vals[:n]
        else:
            t_use = times
            q_use = q_vals

        atom_out = f"atom_{aid}_qs_timeseries.dat"
        with open(atom_out, "w") as out:
            if have_spin and per_atom_s.get(aid, []):
                s_vals = np.asarray(per_atom_s[aid], dtype=float)
                if s_vals.size != t_use.size:
                    n = min(t_use.size, s_vals.size)
                    t_use = t_use[:n]
                    q_use = q_use[:n]
                    s_vals = s_vals[:n]
                out.write("# time_ps  charge  spin\n")
                for t, q, s in zip(t_use, q_use, s_vals):
                    out.write(f"{t: .7f} {q: .7f} {s: .7f}\n")
            else:
                out.write("# time_ps  charge\n")
                for t, q in zip(t_use, q_use):
                    out.write(f"{t: .7f} {q: .7f}\n")

        print(f"[OK] Serie temporal conjunta t, q, s para átomo {aid} escrita en '{atom_out}'.")

    # --- Figura combinada q/s ---
    make_combined_hist_figure(
        atom_ids,
        hist_charge=hist_q,
        hist_spin=hist_s if have_spin else {},
        fig_outname="qs_histograms.png"
    )


if __name__ == "__main__":
    main()

