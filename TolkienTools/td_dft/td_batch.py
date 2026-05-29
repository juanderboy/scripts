#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv
import glob
import json
import math
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from td_common import (
    HC_EV_NM,
    LN10,
    N_A,
    c_light,
    e_charge,
    eps0,
    hbar,
    m_e,
    open_html_in_browser,
    parse_number_ranges,
    parse_selection_string,
    td_file_sort_key,
)
from td_orca import parse_orca_tddft_eV_fosc
from td_spectrum import compute_epsilon_spectrum


def process_folder(folder, args):
    """
    Modo batch:
      - Usa 'folder' (o '.' si no se pasó nada)
      - Busca TD_*.out
      - Usa todos por defecto (o --select para elegir índices)
      - Aplica --exclude (por número de TD)
      - Para cada TD seleccionado:
          * intenta calcular espectro NEA
          * si falla o queda fuera de rango, avisa y sigue
          * si funciona:
              - guarda espectro_*.dat (lambda) o espectroE_*.dat (energy)
              - opcional: guarda PNG TD_*.out-nea-eps.png con sticks
      - Genera espectros_suma.dat (lambda) o espectrosE_suma.dat (energy)
      - Genera espectro_final.dat con el espectro promedio
      - Genera HTML interactivo con:
          * espectros individuales
          * espectros acumulados
      - Intenta abrir el HTML en el navegador automáticamente
    """

    if not os.path.isdir(folder):
        print(f"ERROR: '{folder}' no es una carpeta válida.")
        sys.exit(1)

    td_files_all = sorted(
        glob.glob(os.path.join(folder, "TD_*.out")),
        key=td_file_sort_key,
    )
    if not td_files_all:
        print(f"No se encontraron archivos 'TD_*.out' en '{folder}'.")
        print("Por favor, recuerde que los archivos de salida de ORCA deben "
              "estar guardados como TD_*.out")
        sys.exit(1)

    n_total = len(td_files_all)
    print(f"Se encontraron {n_total} archivos TD_*.out en '{folder}':")
    for i, f in enumerate(td_files_all, start=1):
        print(f"  {i:2d}: {os.path.basename(f)}")

    # selección: por defecto todos; opcional con --select
    sel_str = args.select if args.select is not None else ""

    try:
        idx_list = parse_selection_string(sel_str, n_total)
    except ValueError as e:
        print(f"Error en la selección: {e}")
        sys.exit(1)

    # aplicar la selección
    td_files_sel = [td_files_all[i] for i in idx_list]

    # procesar --exclude: números de TD_*.out a excluir (por ejemplo '10,13' o '10-20')
    exclude_ids = parse_number_ranges(args.exclude, flag_name="--exclude")

    if exclude_ids:
        print(f"\nSe excluirán los TD con números: {', '.join(str(n) for n in sorted(exclude_ids))}")

    # filtrar td_files_sel según exclude_ids
    td_files = []
    for f in td_files_sel:
        base = os.path.basename(f)
        match = re.search(r"TD_(\d+)", base)
        if match and int(match.group(1)) in exclude_ids:
            print(f"  Excluyendo {base} por --exclude")
            continue
        td_files.append(f)

    if not td_files:
        print("Después de aplicar la selección y --exclude no quedó ningún archivo.")
        sys.exit(1)

    print(f"\nSe usarán {len(td_files)} archivos TD_*.out:")
    for f in td_files:
        print(f"  {os.path.basename(f)}")

    mode = args.mode.lower()

    # ------------------------------------------------------------------
    # Construir espectros individuales, guardar PNGs, exportar espectro_*.dat,
    # armar espectros_suma.dat y las figuras globales.
    # ------------------------------------------------------------------

    all_spectra = []   # lista de (x_i, eps_i, etiqueta, td_file); x_i = λ_nm o E_eV
    x_global_min = float("inf")
    x_global_max = -float("inf")

    for td_file in td_files:
        base = os.path.basename(td_file)

        try:
            energies_eV, foscs = parse_orca_tddft_eV_fosc(td_file)
            if energies_eV.size == 0 or foscs.size == 0:
                raise ValueError("Sin datos de transiciones.")

            E_grid, epsilon_E = compute_epsilon_spectrum(
                energies_eV, foscs, args.linewidth_ev, n_ref=args.nref
            )

            # etiqueta a partir del nombre (número después de TD_)
            match = re.search(r"TD_(\d+)", base)
            label = match.group(1) if match else base

            if mode == "lambda":
                lambda_grid = HC_EV_NM / E_grid
                epsilon_lambda = epsilon_E  # SIN jacobiano

                sort_idx = np.argsort(lambda_grid)
                x_sorted = lambda_grid[sort_idx]
                eps_sorted = epsilon_lambda[sort_idx]

                if args.startx is not None and args.endx is not None:
                    x_min = min(args.startx, args.endx)
                    x_max = max(args.startx, args.endx)
                else:
                    x_min = x_sorted.min()
                    x_max = x_sorted.max()

                mask = (x_sorted >= x_min) & (x_sorted <= x_max)
                x_use = x_sorted[mask]
                eps_use = eps_sorted[mask]

            else:  # energy
                x_sorted = E_grid  # ascendente
                eps_sorted = epsilon_E

                if args.startx is not None and args.endx is not None:
                    lam_min = min(args.startx, args.endx)
                    lam_max = max(args.startx, args.endx)
                    E_max = HC_EV_NM / lam_min
                    E_min = HC_EV_NM / lam_max
                else:
                    E_min = x_sorted.min()
                    E_max = x_sorted.max()

                mask = (x_sorted >= E_min) & (x_sorted <= E_max)
                x_use = x_sorted[mask]
                eps_use = eps_sorted[mask]
                x_min, x_max = x_use.min(), x_use.max()

            # si este TD no tiene puntos en el rango, lo saltamos
            if x_use.size == 0:
                if mode == "lambda":
                    print(
                        f"  Aviso: {base} no tiene puntos en el rango "
                        f"{x_min:.1f}-{x_max:.1f} nm; se omite en el merge."
                    )
                else:
                    print(
                        f"  Aviso: {base} no tiene puntos en el rango de energía "
                        f"seleccionado; se omite en el merge."
                    )
                continue

            x_global_min = min(x_global_min, x_use.min())
            x_global_max = max(x_global_max, x_use.max())

            all_spectra.append((x_use, eps_use, label, base))

            # Nota: PNGs individuales solo si se solicita con --printall.
            if args.printall and not args.nosave:
                fig, ax = plt.subplots()
                ax.plot(x_use, eps_use, lw=1.5)

                w_hwhm = args.linewidth_ev / 2.0
                C0 = math.pi * e_charge * hbar / (2.0 * m_e * c_light * eps0 * args.nref)
                C_eps = C0 * (10.0 * N_A / LN10)
                stick_heights_E = [
                    C_eps * fn / (w_hwhm * math.sqrt(math.pi / math.log(2.0)))
                    for fn in foscs
                ]

                if mode == "lambda":
                    lambda_lines = HC_EV_NM / energies_eV
                    stick_heights = np.array(stick_heights_E)  # SIN jacobiano
                    x_sticks = lambda_lines
                    x_label = r"$\lambda$ / nm"
                    title = f"Absorption spectrum (NEA, λ) - {base}"
                else:
                    x_sticks = energies_eV
                    stick_heights = np.array(stick_heights_E)
                    x_label = "E / eV"
                    title = f"Absorption spectrum (NEA, E) - {base}"

                ax.stem(
                    x_sticks,
                    stick_heights,
                    linefmt="grey",
                    markerfmt=" ",
                    basefmt=" ",
                    use_line_collection=True,
                )

                # marcar picos en este espectro
                peaks, _ = find_peaks(eps_use)
                for p in peaks:
                    xp = x_use[p]
                    yp = eps_use[p]
                    if mode == "lambda":
                        label_txt = f"{xp:.0f}"
                    else:
                        label_txt = f"{xp:.2f}"
                    ax.annotate(
                        label_txt,
                        xy=(xp, yp),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha="center",
                        rotation=90 if mode == "lambda" else 0,
                        fontsize=8,
                    )

                ax.set_xlabel(x_label)
                ax.set_ylabel(r"$\varepsilon$ / (L mol$^{-1}$ cm$^{-1}$)")
                ax.set_xlim(x_min, x_max)
                y_min_local = 0.0
                y_max_local = eps_use.max() * 1.1 if eps_use.size > 0 else 1.0
                ax.set_ylim(y_min_local, y_max_local)
                ax.set_title(title)
                ax.grid(False)
                fig.tight_layout()

                out_png = td_file + "-nea-eps.png"
                fig.savefig(out_png, dpi=300)
                print(f"Espectro guardado en: {out_png}")

                plt.close(fig)

        except Exception:
            print(
                f"  Aviso: no pude leer la información del archivo {base}. "
                "Se omite."
            )
            continue

    if not all_spectra:
        print(
            "\nNingún TD_*.out aportó puntos en el rango seleccionado "
            "o todos fallaron al leerse. Revisá -x0/-x1 o los archivos."
        )
        sys.exit(1)

    # construir una grilla común en x (λ o E)
    max_len = max(len(s[0]) for s in all_spectra)
    x_common = np.linspace(x_global_min, x_global_max, max_len)

    eps_matrix = []
    labels = []
    td_names = []

    for x_i, eps_i, label, td_name in all_spectra:
        eps_interp = np.interp(x_common, x_i, eps_i, left=0.0, right=0.0)
        eps_matrix.append(eps_interp)
        labels.append(label)
        td_names.append(td_name)

        if mode == "lambda":
            out_dat = os.path.join(folder, f"espectro_{label}.dat")
            header = "lambda_nm  epsilon_Lmol-1cm-1"
        else:
            out_dat = os.path.join(folder, f"espectroE_{label}.dat")
            header = "energy_eV  epsilon_Lmol-1cm-1"

        data = np.column_stack([x_common, eps_interp])
        np.savetxt(out_dat, data, header=header)
        print(f"Datos exportados en: {out_dat}")

    eps_matrix = np.array(eps_matrix)
    n_files = eps_matrix.shape[0]

    # lista de máximos por espectro (modo carpeta)
    if args.maxlist:
        if args.maxrange is not None:
            x_min_range, x_max_range = args.maxrange
        else:
            x_min_in = input(
                f"Ingrese XMIN para buscar máximos ({'nm' if mode == 'lambda' else 'eV'}) "
                "[Enter = sin límite inferior]: "
            ).strip()
            x_max_in = input(
                f"Ingrese XMAX para buscar máximos ({'nm' if mode == 'lambda' else 'eV'}) "
                "[Enter = sin límite superior]: "
            ).strip()
            x_min_range = float(x_min_in) if x_min_in else float(x_common.min())
            x_max_range = float(x_max_in) if x_max_in else float(x_common.max())
        if x_min_range >= x_max_range:
            print("ERROR: --maxrange requiere XMIN < XMAX.")
            sys.exit(1)

        if args.maxeps is not None:
            eps_threshold = args.maxeps
        else:
            eps_in = input(
                "Ingrese umbral mínimo de epsilon [Enter = sin umbral]: "
            ).strip()
            eps_threshold = float(eps_in) if eps_in else 0.0
        if eps_threshold < 0:
            print("ERROR: el umbral de epsilon debe ser >= 0.")
            sys.exit(1)

        if args.maxonly and args.allpeaks:
            print("ERROR: --maxonly y --allpeaks son excluyentes.")
            sys.exit(1)
        if args.maxonly is None and args.allpeaks is None:
            choice = input(
                "¿Listar solo el máximo absoluto por espectro? [s/N]: "
            ).strip().lower()
            max_only = choice in ("s", "si", "sí", "y", "yes")
        else:
            max_only = bool(args.maxonly)

        mask = (x_common >= x_min_range) & (x_common <= x_max_range)
        if not np.any(mask):
            print(
                "Aviso: el rango indicado para --maxrange no intersecta "
                "el eje x. Se generará un CSV vacío."
            )

        if mode == "lambda":
            x_col = "lambda_max_nm"
        else:
            x_col = "energy_max_eV"

        peak_rows = []
        max_peaks = 0
        for i in range(n_files):
            y = eps_matrix[i]
            if not np.any(mask):
                peak_rows.append([])
                continue
            y_use = y[mask]
            x_use = x_common[mask]
            peaks, _ = find_peaks(y_use)
            if peaks.size == 0:
                peak_rows.append([])
                continue
            peaks_in = [float(x_use[p]) for p in peaks if y_use[p] >= eps_threshold]
            if max_only and peaks_in:
                p_idx = int(np.argmax([y_use[p] for p in peaks if y_use[p] >= eps_threshold]))
                peaks_in = [peaks_in[p_idx]]
            peak_rows.append(peaks_in)
            if len(peaks_in) > max_peaks:
                max_peaks = len(peaks_in)

        if max_peaks == 0:
            max_peaks = 1

        out_csv = os.path.join(folder, "maximos_individuales.csv")
        with open(out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["label", "td_file"] + [f"peak{i+1}" for i in range(max_peaks)]
            writer.writerow(header)
            for i in range(n_files):
                row_peaks = peak_rows[i] if peak_rows[i] else ["NA"]
                row = [labels[i], td_names[i]] + row_peaks
                if len(row) < len(header):
                    row.extend([""] * (len(header) - len(row)))
                writer.writerow(row)

        print(f"Máximos individuales guardados en: {out_csv}")

    # sumas acumuladas
    cum_sums = np.cumsum(eps_matrix, axis=0)

    # construir espectros_suma.dat o espectrosE_suma.dat
    out_cols = [x_common]
    if mode == "lambda":
        header_cols = ["lambda"]
        out_file = os.path.join(folder, "espectros_suma.dat")
    else:
        header_cols = ["energy_eV"]
        out_file = os.path.join(folder, "espectrosE_suma.dat")

    for i, lbl in enumerate(labels):
        out_cols.append(eps_matrix[i])
        header_cols.append(f"A{i+1}")
    for i in range(n_files):
        out_cols.append(cum_sums[i])
        header_cols.append(f"sum_A1_to_A{i+1}")

    out_array = np.column_stack(out_cols)
    header = "\t".join(header_cols)
    np.savetxt(out_file, out_array, header=header)
    print(f"\nArchivo de salida guardado como: {out_file}")

    # ===== espectro_final.dat = promedio del último sumado =====
    final_sum = cum_sums[-1]            # suma de todos los espectros seleccionados
    final_avg = final_sum / n_files     # promedio auténtico

    if mode == "lambda":
        final_file = os.path.join(folder, "espectro_final.dat")
        final_header = "lambda_nm  epsilon_promedio_Lmol-1cm-1"
    else:
        final_file = os.path.join(folder, "espectro_final.dat")
        final_header = "energy_eV  epsilon_promedio_Lmol-1cm-1"

    final_data = np.column_stack([x_common, final_avg])
    np.savetxt(final_file, final_data, header=final_header)
    print(f"Espectro promedio guardado como: {final_file}")

    if args.html:
        try:
            import plotly.graph_objects as go
        except Exception:
            print("Aviso: no pude importar plotly. Instalá con: pip install plotly")
        else:
            if mode == "lambda":
                x_title = "Wavelength (nm)"
                indiv_title = "Individual Spectra (λ)"
                cum_title = "Cumulative sums of spectra (λ)"
            else:
                x_title = "Energy (eV)"
                indiv_title = "Individual Spectra (E)"
                cum_title = "Cumulative sums of spectra (E)"

            fig_ind = go.Figure()
            trace_colors = []
            for i in range(n_files):
                c = plt.cm.tab20(i % 20)
                color = f"rgb({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)})"
                trace_colors.append(color)
                fig_ind.add_trace(
                    go.Scatter(
                        x=x_common,
                        y=eps_matrix[i],
                        mode="lines",
                        name=f"A{i+1} ({labels[i]})",
                        line=dict(color=color, width=1.5),
                        hovertemplate=(
                            f"{td_names[i]}<br>x=%{{x:.3f}}<br>y=%{{y:.3e}}<extra></extra>"
                        ),
                    )
                )
            fig_ind.update_layout(
                title=indiv_title,
                xaxis_title=x_title,
                yaxis_title="epsilon / (L mol^-1 cm^-1)",
                hovermode="closest",
                showlegend=False,
                autosize=True,
                margin=dict(l=60, r=20, t=60, b=50),
            )

            fig_cum = go.Figure()
            cmap = plt.cm.inferno
            cum_colors = cmap(np.linspace(0.1, 0.9, n_files))
            if n_files > 1:
                for i in range(n_files - 1):
                    c = cum_colors[i]
                    color = f"rgb({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)})"
                    fig_cum.add_trace(
                        go.Scatter(
                            x=x_common,
                            y=cum_sums[i],
                            mode="lines",
                            name=f"sum_A1_to_A{i+1}",
                            line=dict(color=color, width=1.0),
                            opacity=0.9,
                            hovertemplate=(
                                f"sum_A1_to_A{i+1}<br>x=%{{x:.3f}}<br>y=%{{y:.3e}}<extra></extra>"
                            ),
                        )
                    )

            y_last = cum_sums[-1]
            fig_cum.add_trace(
                go.Scatter(
                    x=x_common,
                    y=y_last,
                    mode="lines",
                    name="total sum",
                    line=dict(color="black", width=2.0),
                    hovertemplate="total sum<br>x=%{x:.3f}<br>y=%{y:.3e}<extra></extra>",
                )
            )

            if not args.no_final_peaks:
                peak_indices, _ = find_peaks(y_last)
                if peak_indices.size > 0:
                    x_peak = x_common[peak_indices]
                    y_peak = y_last[peak_indices]
                    if mode == "lambda":
                        peak_text = [f"{v:.1f}" for v in x_peak]
                    else:
                        peak_text = [f"{v:.2f}" for v in x_peak]
                    fig_cum.add_trace(
                        go.Scatter(
                            x=x_peak,
                            y=y_peak,
                            mode="markers+text",
                            text=peak_text,
                            textposition="top center",
                            name="peaks",
                            marker=dict(color="black", size=6),
                            hovertemplate="peak<br>x=%{x:.3f}<br>y=%{y:.3e}<extra></extra>",
                        )
                    )

            cum_sum_trace_count = len(fig_cum.data)
            fig_cum.add_trace(
                go.Scatter(
                    x=x_common,
                    y=final_avg,
                    mode="lines",
                    name="average spectrum",
                    line=dict(color="rgb(35,95,190)", width=2.4),
                    hovertemplate="average<br>x=%{x:.3f}<br>y=%{y:.3e}<extra></extra>",
                    visible=False,
                )
            )
            cum_avg_trace_idx = len(fig_cum.data) - 1

            fig_cum.update_layout(
                title=cum_title,
                xaxis_title=x_title,
                yaxis_title="cumulative epsilon / (L mol^-1 cm^-1)",
                hovermode="closest",
                showlegend=False,
                autosize=True,
                margin=dict(l=60, r=20, t=60, b=50),
            )

            html_path = os.path.join(folder, "a_espectros_interactivos.html")
            fig_div_ind = fig_ind.to_html(
                full_html=False,
                include_plotlyjs="cdn",
                config={"responsive": True},
                default_width="100%",
                default_height="100%",
            )
            fig_div_cum = fig_cum.to_html(
                full_html=False,
                include_plotlyjs=False,
                config={"responsive": True},
                default_width="100%",
                default_height="100%",
            )

            html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Spectra Viewer</title>
  <style>
    html, body {{
      margin: 0;
      padding: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
      font-family: Arial, sans-serif;
      background: #ffffff;
    }}
    .wrap {{
      width: 100%;
      height: 100vh;
      display: flex;
      flex-direction: column;
    }}
    .tabs {{
      display: flex;
      gap: 8px;
      padding: 8px 12px;
      border-bottom: 1px solid #ddd;
      background: #f5f5f5;
      flex: 0 0 auto;
    }}
    .tab-btn {{
      padding: 6px 10px;
      border: 1px solid #888;
      border-radius: 6px;
      background: #fff;
      cursor: pointer;
    }}
    .tab-btn.active {{
      background: #222;
      color: #fff;
      border-color: #222;
    }}
    .toggle-btn.active {{
      background: #222;
      color: #fff;
      border-color: #222;
    }}
    .controls {{
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 8px 12px;
      border-bottom: 1px solid #ddd;
      background: #f7f7f7;
      flex-wrap: wrap;
      flex: 0 0 auto;
    }}
    .controls input {{
      min-width: 260px;
      padding: 6px 8px;
      border: 1px solid #bbb;
      border-radius: 6px;
      font-size: 14px;
    }}
    .controls select {{
      min-width: 240px;
      padding: 6px 8px;
      border: 1px solid #bbb;
      border-radius: 6px;
      font-size: 14px;
      background: #fff;
    }}
    .controls select[multiple] {{
      min-height: 0;
      height: 34px;
    }}
    .controls button {{
      padding: 6px 10px;
      border: 1px solid #888;
      border-radius: 6px;
      background: #fff;
      cursor: pointer;
    }}
    .status {{
      font-size: 13px;
      color: #333;
    }}
    .plot-area {{
      flex: 1 1 auto;
      min-height: 0;
      position: relative;
    }}
    .plot-pane {{
      position: absolute;
      inset: 0;
    }}
    .hidden {{
      display: none;
    }}
    .plot-pane .js-plotly-plot, .plot-pane .plot-container, .plot-pane .svg-container {{
      width: 100% !important;
      height: 100% !important;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="tabs">
      <button id="tab-ind" class="tab-btn" type="button">Espectros individuales</button>
      <button id="tab-cum" class="tab-btn active" type="button">Espectros acumulados</button>
    </div>
    <div class="controls hidden" id="controls-ind">
      <label for="td-filter"><strong>Buscar TD:</strong></label>
      <input id="td-filter" type="text" placeholder="Ej: TD_20.out">
      <label for="td-select"><strong>Seleccionar TD:</strong></label>
      <select id="td-select" multiple size="1" title="Podés seleccionar varios TD (Ctrl/Cmd + click)."></select>
      <button id="td-clear" type="button">Limpiar</button>
      <span class="status" id="td-status"></span>
    </div>
    <div class="controls" id="controls-cum">
      <label><strong>Vista:</strong></label>
      <button id="cum-sum" class="toggle-btn active" type="button">Suma acumulada</button>
      <button id="cum-avg" class="toggle-btn" type="button">Espectro promedio</button>
      <span class="status" id="cum-status">Mostrando suma acumulada</span>
    </div>
    <div class="plot-area">
      <div class="plot-pane hidden" id="pane-ind">
        {fig_div_ind}
      </div>
      <div class="plot-pane" id="pane-cum">
        {fig_div_cum}
      </div>
    </div>
  </div>
  <script>
    (function() {{
      const tdNames = {json.dumps(td_names)};
      const baseColors = {json.dumps(trace_colors)};
      const cumSumTraceCount = {cum_sum_trace_count};
      const cumAvgTraceIndex = {cum_avg_trace_idx};

      const input = document.getElementById("td-filter");
      const select = document.getElementById("td-select");
      const btn = document.getElementById("td-clear");
      const status = document.getElementById("td-status");
      const controlsInd = document.getElementById("controls-ind");
      const controlsCum = document.getElementById("controls-cum");
      const cumStatus = document.getElementById("cum-status");
      const cumSumBtn = document.getElementById("cum-sum");
      const cumAvgBtn = document.getElementById("cum-avg");
      const paneInd = document.getElementById("pane-ind");
      const paneCum = document.getElementById("pane-cum");
      const tabInd = document.getElementById("tab-ind");
      const tabCum = document.getElementById("tab-cum");
      let cumMode = "sum";
      const cumRanges = {{sum: null, avg: null}};

      const gdInd = document.querySelector("#pane-ind .js-plotly-plot");
      const gdCum = document.querySelector("#pane-cum .js-plotly-plot");

      if (!gdInd || !gdCum || !window.Plotly) return;

      function tdSortValue(name) {{
        const m = String(name).match(/TD_(\\d+)/i);
        return m ? parseInt(m[1], 10) : Number.POSITIVE_INFINITY;
      }}

      const uniqueNames = Array.from(new Set(tdNames)).sort(function(a, b) {{
        const va = tdSortValue(a);
        const vb = tdSortValue(b);
        if (va !== vb) return va - vb;
        return String(a).localeCompare(String(b), undefined, {{
          numeric: true,
          sensitivity: "base"
        }});
      }});
      uniqueNames.forEach(function(name) {{
        const opt = document.createElement("option");
        opt.value = name;
        opt.textContent = name;
        select.appendChild(opt);
      }});

      function applyFilter() {{
        const q = input.value.trim().toLowerCase();
        const selectedNames = new Set(
          Array.from(select.selectedOptions).map(function(opt) {{ return opt.value; }})
        );
        const colors = [];
        const widths = [];
        const opacities = [];
        let nMatch = 0;

        for (let i = 0; i < tdNames.length; i++) {{
          const matchesSelect = selectedNames.size > 0 ? selectedNames.has(tdNames[i]) : true;
          const matchesText = q !== "" ? tdNames[i].toLowerCase().includes(q) : true;
          const match = matchesSelect && matchesText;

          if (selectedNames.size === 0 && q === "") {{
            colors.push(baseColors[i]);
            widths.push(1.5);
            opacities.push(1.0);
            continue;
          }}
          if (match) {{
            nMatch += 1;
            colors.push(baseColors[i]);
            widths.push(2.5);
            opacities.push(1.0);
          }} else {{
            colors.push("rgba(170,170,170,0.65)");
            widths.push(1.0);
            opacities.push(0.35);
          }}
        }}

        Plotly.restyle(gdInd, {{
          "line.color": colors,
          "line.width": widths,
          "opacity": opacities
        }}, [...Array(tdNames.length).keys()]);

        if (q === "") {{
          if (selectedNames.size === 0) {{
            status.textContent = "";
          }} else {{
            status.textContent =
              "Seleccionados: " + selectedNames.size + " TD (" + nMatch + " traza/s)";
          }}
        }} else {{
          if (selectedNames.size === 0) {{
            status.textContent = "Coincidencias: " + nMatch + " / " + tdNames.length;
          }} else {{
            status.textContent =
              "Coincidencias: " + nMatch + " / " + tdNames.length +
              " (seleccionados: " + selectedNames.size + ")";
          }}
        }}
      }}

      function setCumulativeMode(mode) {{
        const prevMode = cumMode;
        const currentLayout = gdCum.layout || {{}};
        const hasCurrentRange =
          Array.isArray(currentLayout.xaxis && currentLayout.xaxis.range) &&
          Array.isArray(currentLayout.yaxis && currentLayout.yaxis.range);
        if (hasCurrentRange) {{
          cumRanges[prevMode] = {{
            x: [currentLayout.xaxis.range[0], currentLayout.xaxis.range[1]],
            y: [currentLayout.yaxis.range[0], currentLayout.yaxis.range[1]]
          }};
        }} else {{
          cumRanges[prevMode] = null;
        }}

        cumMode = mode;
        const nTraces = gdCum.data.length;
        const visible = new Array(nTraces).fill(false);
        if (mode === "avg") {{
          if (cumAvgTraceIndex >= 0 && cumAvgTraceIndex < nTraces) {{
            visible[cumAvgTraceIndex] = true;
          }}
          cumStatus.textContent = "Mostrando espectro promedio";
          cumAvgBtn.classList.add("active");
          cumSumBtn.classList.remove("active");
        }} else {{
          for (let i = 0; i < Math.min(cumSumTraceCount, nTraces); i++) {{
            visible[i] = true;
          }}
          cumStatus.textContent = "Mostrando suma acumulada";
          cumSumBtn.classList.add("active");
          cumAvgBtn.classList.remove("active");
        }}
        Plotly.restyle(gdCum, {{"visible": visible}}, [...Array(nTraces).keys()]);

        const targetRange = cumRanges[mode];
        if (targetRange && targetRange.x && targetRange.y) {{
          Plotly.relayout(gdCum, {{
            "xaxis.autorange": false,
            "yaxis.autorange": false,
            "xaxis.range": targetRange.x,
            "yaxis.range": targetRange.y
          }});
        }} else {{
          Plotly.relayout(gdCum, {{
            "xaxis.autorange": true,
            "yaxis.autorange": true
          }});
        }}
      }}

      function setView(view) {{
        if (view === "ind") {{
          paneInd.classList.remove("hidden");
          paneCum.classList.add("hidden");
          controlsInd.classList.remove("hidden");
          controlsCum.classList.add("hidden");
          tabInd.classList.add("active");
          tabCum.classList.remove("active");
        }} else {{
          paneCum.classList.remove("hidden");
          paneInd.classList.add("hidden");
          controlsInd.classList.add("hidden");
          controlsCum.classList.remove("hidden");
          tabCum.classList.add("active");
          tabInd.classList.remove("active");
          setCumulativeMode(cumMode);
        }}
        resizePlots();
        // El plot oculto al cargar puede quedar angosto; forzamos resize al mostrar pestaña.
        window.requestAnimationFrame(function() {{
          Plotly.Plots.resize(gdInd);
          Plotly.Plots.resize(gdCum);
        }});
        setTimeout(function() {{
          Plotly.Plots.resize(gdInd);
          Plotly.Plots.resize(gdCum);
        }}, 80);
      }}

      function resizePlots() {{
        const vh = window.innerHeight || document.documentElement.clientHeight || 800;
        const tabsH = document.querySelector(".tabs").offsetHeight || 0;
        const controlsIndH = controlsInd.classList.contains("hidden") ? 0 : controlsInd.offsetHeight;
        const controlsCumH = controlsCum.classList.contains("hidden") ? 0 : controlsCum.offsetHeight;
        const controlsH = controlsIndH + controlsCumH;
        const target = Math.max(320, vh - tabsH - controlsH - 6);
        const plotW = document.querySelector(".plot-area").clientWidth || window.innerWidth;

        paneInd.style.height = target + "px";
        paneCum.style.height = target + "px";

        Plotly.relayout(gdInd, {{height: target, width: plotW}});
        Plotly.relayout(gdCum, {{height: target, width: plotW}});
      }}

      input.addEventListener("input", applyFilter);
      select.addEventListener("change", applyFilter);
      btn.addEventListener("click", function() {{
        input.value = "";
        Array.from(select.options).forEach(function(opt) {{ opt.selected = false; }});
        applyFilter();
        input.focus();
      }});
      gdCum.on("plotly_relayout", function(ev) {{
        if (!ev) return;
        const xr0 = ev["xaxis.range[0]"];
        const xr1 = ev["xaxis.range[1]"];
        const yr0 = ev["yaxis.range[0]"];
        const yr1 = ev["yaxis.range[1]"];
        if (
          xr0 !== undefined && xr1 !== undefined &&
          yr0 !== undefined && yr1 !== undefined
        ) {{
          cumRanges[cumMode] = {{x: [xr0, xr1], y: [yr0, yr1]}};
        }}
        if (ev["xaxis.autorange"] === true || ev["yaxis.autorange"] === true) {{
          cumRanges[cumMode] = null;
        }}
      }});
      cumSumBtn.addEventListener("click", function() {{ setCumulativeMode("sum"); }});
      cumAvgBtn.addEventListener("click", function() {{ setCumulativeMode("avg"); }});
      tabInd.addEventListener("click", function() {{ setView("ind"); }});
      tabCum.addEventListener("click", function() {{ setView("cum"); }});
      window.addEventListener("resize", resizePlots);

      applyFilter();
      setView("cum");
    }})();
  </script>
</body>
</html>
"""
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_doc)
            print(f"HTML interactivo guardado en: {html_path}")

            open_status = open_html_in_browser(html_path)
            if open_status == "html":
                print("HTML abierto automáticamente en el navegador.")
            elif open_status == "folder":
                print(
                    "Aviso: no pude abrir el HTML automáticamente. "
                    "Se abrió la carpeta con explorer.exe ."
                )
            else:
                print("Aviso: no pude abrir ni el HTML ni la carpeta automáticamente.")
