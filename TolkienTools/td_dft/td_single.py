#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from td_common import HC_EV_NM, LN10, N_A, c_light, e_charge, eps0, hbar, m_e
from td_orca import parse_orca_tddft_eV_fosc
from td_spectrum import compute_epsilon_spectrum


def process_single_file(filename, args):
    """
    Modo archivo único: procesa un solo TD_*.out y opcionalmente exporta espectro_*.dat.
    """
    energies_eV, foscs = parse_orca_tddft_eV_fosc(filename)
    if energies_eV.size == 0 or foscs.size == 0:
        print(f"Aviso: no se pudieron obtener transiciones desde {filename}.")
        return

    # espectro ε(E)
    E_grid, epsilon_E = compute_epsilon_spectrum(
        energies_eV, foscs, args.linewidth_ev, n_ref=args.nref
    )

    mode = args.mode.lower()

    if mode == "lambda":
        # transformar a ε(λ): solo cambio de eje, SIN jacobiano
        lambda_grid = HC_EV_NM / E_grid
        epsilon_lambda = epsilon_E

        sort_idx = np.argsort(lambda_grid)
        x_sorted = lambda_grid[sort_idx]
        eps_sorted = epsilon_lambda[sort_idx]

        # rango en λ
        if args.startx is not None and args.endx is not None:
            x_min = min(args.startx, args.endx)
            x_max = max(args.startx, args.endx)
        else:
            x_min = x_sorted.min()
            x_max = x_sorted.max()

        mask = (x_sorted >= x_min) & (x_sorted <= x_max)
        x_plot = x_sorted[mask]
        eps_plot = eps_sorted[mask]

        x_label = r"$\lambda$ / nm"
        title = "Absorption spectrum (NEA, λ-representation)"

    else:  # mode == "energy"
        x_sorted = E_grid   # ya ascendente
        eps_sorted = epsilon_E

        if args.startx is not None and args.endx is not None:
            # interpretamos -x0/-x1 como rango en λ y lo convertimos a E
            lam_min = min(args.startx, args.endx)
            lam_max = max(args.startx, args.endx)
            E_max = HC_EV_NM / lam_min
            E_min = HC_EV_NM / lam_max
        else:
            E_min = x_sorted.min()
            E_max = x_sorted.max()

        mask = (x_sorted >= E_min) & (x_sorted <= E_max)
        x_plot = x_sorted[mask]
        eps_plot = eps_sorted[mask]

        x_min, x_max = x_plot.min(), x_plot.max()
        x_label = r"E / eV"
        title = "Absorption spectrum (NEA, energy representation)"

    if x_plot.size == 0:
        print(
            f"Aviso: {os.path.basename(filename)} no tiene puntos en el rango "
            f"seleccionado."
        )
        return

    # figura
    fig, ax = plt.subplots()
    ax.plot(x_plot, eps_plot, lw=1.5)

    # sticks: en energía y transformados si se está en λ (solo cambio de eje)
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
    else:
        x_sticks = energies_eV
        stick_heights = np.array(stick_heights_E)

    ax.stem(
        x_sticks,
        stick_heights,
        linefmt="grey",
        markerfmt=" ",
        basefmt=" ",
        use_line_collection=True,
    )

    # detección de picos
    peaks, _ = find_peaks(eps_plot)
    for p in peaks:
        xp = x_plot[p]
        yp = eps_plot[p]
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

    # labels y estilo
    ax.set_xlabel(x_label)
    ax.set_ylabel(r"$\varepsilon$ / (L mol$^{-1}$ cm$^{-1}$)")
    ax.set_xlim(x_min, x_max)
    y_min = 0.0
    y_max = eps_plot.max() * 1.1 if eps_plot.size > 0 else 1.0
    ax.set_ylim(y_min, y_max)
    ax.set_title(title)
    ax.grid(False)
    fig.tight_layout()

    # guardar PNG
    if not args.nosave:
        out_png = filename + "-nea-eps.png"
        fig.savefig(out_png, dpi=300)
        print(f"Espectro guardado en: {out_png}")

    # exportar datos ε como espectro_*.dat (λ) o espectroE_*.dat (E)
    if args.export:
        base = os.path.basename(filename)        # "TD_10.out"
        match = re.search(r"TD_(\d+)", base)
        if match:
            idx = match.group(1)
            if mode == "lambda":
                out_dat = f"espectro_{idx}.dat"
                header = "lambda_nm  epsilon_Lmol-1cm-1"
            else:
                out_dat = f"espectroE_{idx}.dat"
                header = "energy_eV  epsilon_Lmol-1cm-1"
        else:
            if mode == "lambda":
                out_dat = "espectro.dat"
                header = "lambda_nm  epsilon_Lmol-1cm-1"
            else:
                out_dat = "espectroE.dat"
                header = "energy_eV  epsilon_Lmol-1cm-1"

        data = np.column_stack([x_plot, eps_plot])
        np.savetxt(out_dat, data, header=header)
        print(f"Datos exportados en: {out_dat}")

    if args.show:
        plt.show()
