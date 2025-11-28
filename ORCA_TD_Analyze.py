#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genera espectros de absortividad molar ε a partir de TD-DFT de ORCA
usando la formulación NEA de Crespo-Otero & Barbatti.

MODO 1 (archivo ORCA):
    rutina.py TD_10.out [opciones]
        -> ajusta NEA para un solo TD_*.out
        -> grafica ε vs λ (por defecto) o ε vs E según --mode
        -> puede exportar espectro_*.dat / espectroE_*.dat

MODO 2 (carpeta):
    rutina.py [carpeta] [opciones]
        -> si no se indica carpeta, usa la carpeta actual (.)
        -> busca todos los TD_*.out en esa carpeta
        -> te pregunta qué índices usar (1-15, 1,2,4-7, etc.)
        -> aplica --exclude si se indica
        -> genera espectro_*.dat (mode=lambda) o espectroE_*.dat (mode=energy)
        -> genera PNG individual TD_*.out-nea-eps.png para cada uno
        -> combina todos los espectros en espectros_suma.dat / espectrosE_suma.dat
        -> genera espectro_final.dat con el espectro promedio
        -> genera:
             - espectros_individuales.png
             - espectros_sumados.png
"""

import sys
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import math
import os
import glob

# ----------------------------------------------------------------------
# Constantes físicas (SI)
# ----------------------------------------------------------------------
e_charge = 1.602176634e-19      # C
hbar = 1.054571817e-34          # J·s
m_e = 9.1093837015e-31          # kg
c_light = 299792458.0           # m/s
eps0 = 8.8541878128e-12         # F/m
N_A = 6.02214076e23             # mol^-1
LN10 = math.log(10.0)

# hc en unidades eV·nm
HC_EV_NM = 1239.8419843320026

# ----------------------------------------------------------------------
# Funciones auxiliares
# ----------------------------------------------------------------------
def gauss_area(a, m, x, w_hwhm):
    """
    Gaussiana de área 'a' centrada en 'm', con half-width-at-half-maximum (HWHM) = w_hwhm.
    G(x) = A * exp( -ln(2) * ((x - m)/w)**2 )
    A = a / ( w * sqrt(pi / ln(2)) )
    De modo que ∫ G(x) dx = a.
    """
    amplitude = a / (w_hwhm * np.sqrt(np.pi / np.log(2.0)))
    return amplitude * np.exp(-np.log(2.0) * ((x - m) / w_hwhm) ** 2)


def parse_orca_tddft_eV_fosc(filename):
    """
    Parsea la sección de espectro de absorción de ORCA (TD-DFT) y devuelve:
      - energies_eV: array de energías de transición (eV)
      - foscs: array de fuerzas de oscilador

    Asumimos formato ORCA 6.x, en la sección:

         Transition      Energy     Energy  Wavelength fosc(D2)      D2   ...
                          (eV)      (cm-1)    (nm)                 ...

    Ejemplo de línea:
      0-2A  ->  1-2A    0.645453    5205.9  1920.9   0.000058302   0.00369 ...

    Entonces:
      parts[3] = Energy (eV)
      parts[6] = fosc
    """
    spec_start = "ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS"
    spec_end = "ABSORPTION SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS"

    energies = []
    foscs = []
    found_uv_section = False
    version6 = False

    with open(filename, "r") as f:
        for line in f:
            if "Program Version 6" in line:
                version6 = True

            if spec_start in line:
                found_uv_section = True
                # leer hasta el bloque de VELOCITY DIPOLE MOMENTS
                for line in f:
                    if spec_end in line:
                        break
                    # líneas de datos: empiezan con número (índice de transición)
                    if re.search(r"\d\s{1,}\d", line):
                        parts = line.split()
                        if version6:
                            E = float(parts[3])   # eV
                            fosc = float(parts[6])  # fosc
                        else:
                            # fallback para versiones viejas: asumimos el mismo orden
                            E = float(parts[3])
                            fosc = float(parts[6])
                        energies.append(E)
                        foscs.append(fosc)
                break

    if not found_uv_section:
        print(f"'{spec_start}' no encontrado en '{filename}'")
    if len(energies) == 0:
        print(f"No se encontraron transiciones en '{filename}'")
    return np.array(energies), np.array(foscs)


def build_energy_grid(energies_eV, fwhm_eV, start_lambda=None, end_lambda=None):
    """
    Construye un grid de energía (eV) adecuado para la convolución:

    - Si start_lambda y end_lambda están dados (en nm), se convierten a límites en E.
    - Si no, se toma [Emin - 3*FWHM, Emax + 3*FWHM].

    Devuelve:
      E_grid (eV), E_min, E_max
    """
    Emin_data = energies_eV.min()
    Emax_data = energies_eV.max()

    if start_lambda is not None and end_lambda is not None:
        # convertir nm -> eV (E = hc/λ)
        lam_min = min(start_lambda, end_lambda)
        lam_max = max(start_lambda, end_lambda)
        # en energía, λ pequeña = E grande
        Emax = HC_EV_NM / lam_min
        Emin = HC_EV_NM / lam_max
    else:
        Emin = max(Emin_data - 3.0 * fwhm_eV, 1.0e-4)
        Emax = Emax_data + 3.0 * fwhm_eV

    # resolución del grid: paso ~FWHM/50 para líneas suaves
    dE = fwhm_eV / 50.0
    if dE <= 0:
        dE = 0.001

    E_grid = np.arange(Emin, Emax, dE)
    return E_grid, Emin, Emax


def compute_epsilon_spectrum(energies_eV, foscs, fwhm_eV, n_ref=1.33):
    """
    Calcula ε(E) [L mol^-1 cm^-1] a partir de:
      - energies_eV: array de energías de transición en eV
      - foscs: fuerzas de oscilador
      - fwhm_eV: ancho de línea en eV (FWHM)
      - n_ref: índice de refracción (default 1.33 ~ agua)

    Implementa:
      σ(E) = (π e² ħ)/(2 m c ε₀ n_r E) Σ_n ΔE_n f_n g(E-ΔE_n, δ)
    usando g normalizada en eV, y luego:
      ε(E) = σ(E) * (N_A / (1000 ln 10)) * 10^4
    => ε(E) = C_eps * [ S(E) / E ]
    con S(E) = Σ ΔE_n f_n g(E - ΔE_n).
    """
    if energies_eV.size == 0 or foscs.size == 0:
        raise ValueError("Sin datos de energías/fosc para construir el espectro.")

    # prefactor para ε directamente (en L mol^-1 cm^-1)
    C0 = math.pi * e_charge * hbar / (2.0 * m_e * c_light * eps0 * n_ref)
    C_eps = C0 * (10.0 * N_A / LN10)

    # grid de energía
    E_grid, Emin, Emax = build_energy_grid(energies_eV, fwhm_eV)

    # FWHM -> HWHM
    w_hwhm = fwhm_eV / 2.0

    # S(E) = Σ ΔE_n f_n g(E-ΔE_n)
    S = np.zeros_like(E_grid)
    for En, fn in zip(energies_eV, foscs):
        area = En * fn
        S += gauss_area(area, En, E_grid, w_hwhm)

    # ε(E) = C_eps * S(E) / E
    epsilon_E = C_eps * S / E_grid

    return E_grid, epsilon_E


def parse_selection_string(sel, n_files):
    """
    Parsea una selección tipo:
      '1-5'          -> [0,1,2,3,4]
      '1,2,3,7-10'   -> [0,1,2,6,7,8,9]
    Los índices devueltos son 0-based (para indexar listas de Python).

    n_files = cantidad total de archivos (para chequear límites).
    """
    sel = sel.strip()
    if sel == "":
        # vacío = usar todos
        return list(range(n_files))

    indices = set()
    parts = sel.split(",")
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a_str, b_str = part.split("-", 1)
            try:
                a = int(a_str)
                b = int(b_str)
            except ValueError:
                raise ValueError(f"Rango inválido: '{part}'")
            if a < 1 or b < 1 or a > n_files or b > n_files or a > b:
                raise ValueError(f"Rango fuera de límites: '{part}' (1..{n_files})")
            for k in range(a, b + 1):
                indices.add(k - 1)  # 0-based
        else:
            try:
                k = int(part)
            except ValueError:
                raise ValueError(f"Índice inválido: '{part}'")
            if k < 1 or k > n_files:
                raise ValueError(f"Índice fuera de límites: '{part}' (1..{n_files})")
            indices.add(k - 1)

    if not indices:
        raise ValueError("No se seleccionó ningún índice válido.")
    return sorted(indices)


# ----------------------------------------------------------------------
# Lógica para modo "archivo único"
# ----------------------------------------------------------------------
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
        # transformar a ε(λ) con jacobiano dE/dλ = HC_EV_NM / λ^2
        lambda_grid = HC_EV_NM / E_grid
        epsilon_lambda = epsilon_E * (HC_EV_NM / (lambda_grid ** 2))

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

    # sticks: en energía y transformados si se está en λ
    w_hwhm = args.linewidth_ev / 2.0
    C0 = math.pi * e_charge * hbar / (2.0 * m_e * c_light * eps0 * args.nref)
    C_eps = C0 * (10.0 * N_A / LN10)
    stick_heights_E = [
        C_eps * fn / (w_hwhm * math.sqrt(math.pi / math.log(2.0)))
        for fn in foscs
    ]

    if mode == "lambda":
        lambda_lines = HC_EV_NM / energies_eV
        stick_heights = np.array(stick_heights_E) * (HC_EV_NM / (lambda_lines ** 2))
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


# ----------------------------------------------------------------------
# Lógica para modo "carpeta" (batch: varios TD_*.out)
# ----------------------------------------------------------------------
def process_folder(folder, args):
    """
    Modo batch:
      - Usa 'folder' (o '.' si no se pasó nada)
      - Busca TD_*.out
      - Pregunta qué índices usar (1..N, o rangos tipo 1-3,5-7, etc.)
      - Aplica --exclude (por número de TD)
      - Para cada TD seleccionado:
          * intenta calcular espectro NEA
          * si falla o queda fuera de rango, avisa y sigue
          * si funciona:
              - guarda espectro_*.dat (lambda) o espectroE_*.dat (energy)
              - guarda PNG TD_*.out-nea-eps.png con sticks
      - Genera espectros_suma.dat (lambda) o espectrosE_suma.dat (energy)
      - Genera espectro_final.dat con el espectro promedio
      - Grafica:
          * espectros_individuales.png
          * espectros_sumados.png
    """

    if not os.path.isdir(folder):
        print(f"ERROR: '{folder}' no es una carpeta válida.")
        sys.exit(1)

    td_files_all = sorted(glob.glob(os.path.join(folder, "TD_*.out")))
    if not td_files_all:
        print(f"No se encontraron archivos 'TD_*.out' en '{folder}'.")
        print("Por favor, recuerde que los archivos de salida de ORCA deben "
              "estar guardados como TD_*.out")
        sys.exit(1)

    n_total = len(td_files_all)
    print(f"Se encontraron {n_total} archivos TD_*.out en '{folder}':")
    for i, f in enumerate(td_files_all, start=1):
        print(f"  {i:2d}: {os.path.basename(f)}")

    # pedir selección interactiva
    try:
        sel_str = input(
            "\nIndicá el rango que querés usar "
            "(ej: '1-15', '1,2,3,4,7-10'; ENTER = todos): "
        )
    except EOFError:
        sel_str = ""

    try:
        idx_list = parse_selection_string(sel_str, n_total)
    except ValueError as e:
        print(f"Error en la selección: {e}")
        sys.exit(1)

    # aplicar la selección
    td_files_sel = [td_files_all[i] for i in idx_list]

    # procesar --exclude: números de TD_*.out a excluir (por ejemplo '10,13')
    exclude_ids = set()
    if args.exclude is not None:
        for part in args.exclude.split(","):
            p = part.strip()
            if not p:
                continue
            if not p.isdigit():
                print(f"Advertencia: valor de --exclude ignorado: '{p}' (no es entero)")
                continue
            exclude_ids.add(p)

    if exclude_ids:
        print(f"\nSe excluirán los TD con números: {', '.join(sorted(exclude_ids))}")

    # filtrar td_files_sel según exclude_ids
    td_files = []
    for f in td_files_sel:
        base = os.path.basename(f)
        match = re.search(r"TD_(\d+)", base)
        if match and match.group(1) in exclude_ids:
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

    all_spectra = []   # lista de (x_i, eps_i, etiqueta); x_i = λ_nm o E_eV
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
                epsilon_lambda = epsilon_E * (HC_EV_NM / (lambda_grid ** 2))

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

            all_spectra.append((x_use, eps_use, label))

            # --- guardar PNG individual para este TD (espectro + sticks) ---
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
                stick_heights = np.array(stick_heights_E) * (
                    HC_EV_NM / (lambda_lines ** 2)
                )
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

            if not args.nosave:
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

    for x_i, eps_i, label in all_spectra:
        eps_interp = np.interp(x_common, x_i, eps_i, left=0.0, right=0.0)
        eps_matrix.append(eps_interp)
        labels.append(label)

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

    # === GRAFICAR ESPECTROS INDIVIDUALES ===
    plt.figure("Espectros individuales", figsize=(10, 5))
    for i in range(n_files):
        y = eps_matrix[i]
        x = x_common
        plt.plot(x, y, label=f"A{i+1} ({labels[i]})")

    if mode == "lambda":
        plt.title("Espectros individuales (λ)")
        plt.xlabel("Longitud de onda (nm)")
    else:
        plt.title("Espectros individuales (E)")
        plt.xlabel("Energía (eV)")

    plt.ylabel(r"$\varepsilon$ / (L mol$^{-1}$ cm$^{-1}$)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(folder, "espectros_individuales.png"))
    print(f"Figura guardada: {os.path.join(folder, 'espectros_individuales.png')}")

    # === GRAFICAR SUMAS ACUMULADAS ===
    plt.figure("Sumas acumuladas", figsize=(10, 5))

    # degradé de colores para las sumas intermedias
    cmap = plt.cm.inferno
    colors = cmap(np.linspace(0.1, 0.9, n_files))
    x = x_common

    # curvas intermedias: finitas, en degradé
    if n_files > 1:
        for i in range(n_files - 1):
            y = cum_sums[i]
            plt.plot(x, y, color=colors[i], lw=0.7, alpha=0.9)

    # curva final: negra y gruesa
    y_last = cum_sums[-1]
    plt.plot(x, y_last, color="black", lw=2.0, label="suma total")

    # marcar picos en la última suma (sobre la curva negra)
    peak_indices, _ = find_peaks(y_last)
    for idx in peak_indices:
        x_maxp = x[idx]
        A_max = y_last[idx]
        plt.plot(x_maxp, A_max, "o", color="black")
        if mode == "lambda":
            txt = f"{x_maxp:.1f}"
        else:
            txt = f"{x_maxp:.2f}"
        plt.text(
            x_maxp,
            A_max * 1.01,
            txt,
            ha="center",
            va="bottom",
            fontsize=8,
        )

    if mode == "lambda":
        plt.title("Sumas acumuladas de espectros (máximos, λ)")
        plt.xlabel("Longitud de onda (nm)")
    else:
        plt.title("Sumas acumuladas de espectros (máximos, E)")
        plt.xlabel("Energía (eV)")

    plt.ylabel(r"$\varepsilon$ acumulada / (L mol$^{-1}$ cm$^{-1}$)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(folder, "espectros_sumados.png"))
    print(f"Figura guardada: {os.path.join(folder, 'espectros_sumados.png')}")

    if args.show:
        plt.show()


# ----------------------------------------------------------------------
# Programa principal
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        prog="rutina",
        description=(
            "Genera espectros de absortividad molar ε desde TD-DFT de ORCA.\n"
            "MODO 1 (archivo): rutina.py TD_10.out [opciones]\n"
            "MODO 2 (carpeta): rutina.py [carpeta] [opciones]\n"
            "Si no se indica carpeta ni archivo, se usa la carpeta actual."
        ),
        epilog=(
            "FLAGS PRINCIPALES\n"
            "  filename              Archivo TD_*.out o carpeta con TD_*.out (default: '.')\n"
            "  --mode {lambda,energy}\n"
            "      Representación del espectro:\n"
            "        lambda (default) -> ε(λ), aplicando jacobiano dE/dλ\n"
            "        energy          -> ε(E)\n"
            "  -x0, --startx FLOAT   Límite inferior del eje x (en nm). Default: 250 si no hay flags.\n"
            "  -x1, --endx   FLOAT   Límite superior del eje x (en nm). Default: 1000 si no hay flags.\n"
            "  -wev, --linewidth_ev FLOAT\n"
            "      Ancho de línea FWHM en eV para la convolución gaussiana.\n"
            "      Default general: 0.1 eV. Sin flags se fuerza 0.1 eV.\n"
            "  -s, --show            Muestra la ventana de matplotlib.\n"
            "      Sin flags se considera como activado por defecto.\n"
            "  -n, --nosave          No guarda los PNG de espectros individuales.\n"
            "  --nref FLOAT          Índice de refracción (default: 1.33, agua en el visible).\n"
            "  --exclude LIST        En modo carpeta, excluye TD_N.out por número. Ej: --exclude 10,12,15\n"
            "  -e, --export          En modo archivo único, exporta ε como espectro_*.dat o espectroE_*.dat.\n"
            "\n"
            "COMPORTAMIENTO POR DEFECTO (sin ningún flag aparte del filename/carpeta):\n"
            "  Equivalente a usar: -x0 250 -x1 1000 -s --linewidth_ev 0.1\n"
            "  (modo 'lambda' con nref=1.33)\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # filename o carpeta (opcional, default='.')
    parser.add_argument(
        "filename",
        nargs="?",
        default=".",
        help="Archivo de salida de ORCA (TD_*.out) o carpeta con TD_*.out (default: carpeta actual).",
    )

    # modo de representación
    parser.add_argument(
        "--mode",
        choices=["lambda", "energy"],
        default="lambda",
        help="Representación del espectro: 'lambda' (ε(λ), default) o 'energy' (ε(E)).",
    )

    # mostrar ventana
    parser.add_argument(
        "-s",
        "--show",
        default=False,
        action="store_true",
        help="Mostrar la ventana de matplotlib.",
    )

    # no guardar png
    parser.add_argument(
        "-n",
        "--nosave",
   default=False,
        action="store_true",
        help="No guardar los PNG de espectros individuales.",
    )

    # ancho de línea (FWHM) en eV
    parser.add_argument(
        "-wev",
        "--linewidth_ev",
        type=float,
        default=0.1,
        help="Ancho de línea (FWHM) en eV para la convolución gaussiana (default general: 0.1 eV).",
    )

    # rango en λ (nm)
    parser.add_argument(
        "-x0",
        "--startx",
        type=float,
        help="Límite inferior del eje xen nm; se convierte a E en modo 'energy').",
    )
    parser.add_argument(
        "-x1",
        "--endx",
        type=float,
        help="Límite superior del eje x (λ, en nm; se convierte a E en modo 'energy').",
    )

    # índice de refracción
    parser.add_argument(
        "--nref",
        type=float,
        default=1.33,   # agua en el visible
        help="Índice de refracción del medio (default: 1.33, agua en el visible).",
    )

    # excluir algunos TD_* por número (modo carpeta)
    parser.add_argument(
        "--exclude",
        type=str,
        default=None,
        help="Números de TD_*.out a excluir, ej: '10' o '10,12,15' (solo modo carpeta).",
    )

    # exportar datos en modo archivo único
    parser.add_argument(
        "-e",
        "--export",
        default=False,
        action="store_true",
        help="(Modo archivo único) Exportar datos ε como espectro_*.dat (lambda) o espectroE_*.dat (energy).",
    )

    args = parser.parse_args()

    # --- defaults si NO se pningún flag ---
    # (solo rutina.py o rutina.py carpeta)
    has_flags = any(a.startswith("-") for a in sys.argv[1:])
    if not has_flags:
        # default equivalente a: -x0 250 -x1 1000 -s --linewidth_ev 0.1
        args.startx = 250.0
        args.endx = 1000.0
        args.show = True
        args.linewidth_ev = 0.1

    if args.linewidth_ev <= 0:
        print("El ancho de línea (--linewidth_ev) debe ser > 0.")
        sys.exit(1)

    target = args.filename

    if os.path.isdir(target):
       process_folder(target, args)
    else:
        process_single_file(target, args)


if __name__ == "__main__":
    main()

