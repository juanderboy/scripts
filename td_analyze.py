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
        -> opcional: genera PNG individual TD_*.out-nea-eps.png para cada uno
        -> combina todos los espectros en espectros_suma.dat / espectrosE_suma.dat
        -> genera espectro_final.dat con el espectro promedio
        -> genera HTML interactivo con dos vistas:
             - espectros individuales
             - espectros acumulados
        -> abre el HTML automáticamente en el navegador
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
import csv
import json
import shutil
import subprocess
import webbrowser

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


def open_html_in_browser(html_path):
    """
    Intenta abrir el HTML en el navegador por defecto.
    En WSL prioriza `wslview` para abrir en Windows.
    Si no puede abrir el HTML, intenta abrir la carpeta con `explorer.exe .`.
    """
    abs_path = os.path.abspath(html_path)
    folder_path = os.path.dirname(abs_path)
    if shutil.which("wslview"):
        try:
            p = subprocess.run(
                ["wslview", abs_path],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if p.returncode == 0:
                return "html"
        except Exception:
            pass

    if shutil.which("xdg-open"):
        try:
            p = subprocess.run(
                ["xdg-open", abs_path],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if p.returncode == 0:
                return "html"
        except Exception:
            pass

    try:
        if bool(webbrowser.open(f"file://{abs_path}")):
            return "html"
    except Exception:
        pass

    try:
        subprocess.run(
            ["explorer.exe", "."],
            cwd=folder_path,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return "folder"
    except Exception:
        return "none"


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
            "        lambda (default) -> ε(λ)\n"
            "        energy          -> ε(E)\n"
            "  -x0, --startx FLOAT   Límite inferior del eje x (en nm).\n"
            "                        Si no se especifica, se usa 250 nm.\n"
            "  -x1, --endx   FLOAT   Límite superior del eje x (en nm).\n"
            "                        Si no se especifica, se usa 1000 nm.\n"
            "  -wev, --linewidth_ev FLOAT\n"
            "      Ancho de línea FWHM en eV para la convolución gaussiana.\n"
            "      Default: 0.1 eV (a menos que se indique explícitamente).\n"
            "  -s, --show            Muestra la ventana de matplotlib.\n"
            "                        Si no se especifica -s/--show, se asume True.\n"
            "  -n, --nosave          No guarda los PNG que se pudieran generar.\n"
            "  --nref FLOAT          Índice de refracción (default: 1.33, agua en el visible).\n"
            "  --exclude LIST        En modo carpeta, excluye TD_N.out por número. Ej: --exclude 10,12,15\n"
            "  --maxlist             En modo carpeta, genera CSV con máximos individuales.\n"
            "  --maxrange XMIN XMAX  Rango de x para buscar máximos (nm en modo lambda, eV en modo energy).\n"
            "  --maxeps FLOAT        Umbral mínimo de epsilon para listar máximos.\n"
            "  --maxonly             Listar solo el máximo absoluto por espectro (modo carpeta).\n"
            "  --allpeaks            Listar todos los máximos locales (modo carpeta).\n"
            "  --printall            En modo carpeta, guarda PNG individuales TD_*.out-nea-eps.png.\n"
            "  --html                En modo carpeta, genera HTML interactivo (default: activado).\n"
            "  --no-html             En modo carpeta, desactiva el HTML interactivo.\n"
            "  -e, --export          En modo archivo único, exporta ε como espectro_*.dat o espectroE_*.dat.\n"
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
        help="No guardar los PNG que se pudieran generar.",
    )

    # ancho de línea (FWHM) en eV
    parser.add_argument(
        "-wev",
        "--linewidth_ev",
        type=float,
        default=0.1,
        help="Ancho de línea (FWHM) en eV para la convolución gaussiana (default: 0.1 eV).",
    )

    # rango en λ (nm)
    parser.add_argument(
        "-x0",
        "--startx",
        type=float,
        help="Límite inferior del eje x (λ, en nm; se convierte a E en modo 'energy').",
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
        help="Índice de refracción del medilt: 1.33, agua en el visible).",
    )

    # excluir algunos TD_* por número (modo carpeta)
    parser.add_argument(
        "--exclude",
        type=str,
        default=None,
        help="Números de TD_*.out a excluir, ej: '10' o '10,12,15' (solo modo carpeta).",
    )
    # no marcar picos en el espectro final (modo carpeta)
    parser.add_argument(
        "--no-final-peaks",
        default=False,
        action="store_true",
        help="Do not mark/label peaks in the final cumulative spectrum (folder mode).",
    )
    # imprimir PNGs individuales en modo carpeta
    parser.add_argument(
        "--printall",
        default=False,
        action="store_true",
        help="(Modo carpeta) Guardar PNG individuales TD_*.out-nea-eps.png.",
    )

    # lista de máximos por espectro (modo carpeta)
    parser.add_argument(
        "--maxlist",
        default=False,
        action="store_true",
        help="(Modo carpeta) Generar CSV con la localización del máximo de cada espectro.",
    )
    parser.add_argument(
        "--maxrange",
        nargs=2,
        type=float,
        metavar=("XMIN", "XMAX"),
        help="(Modo carpeta) Rango de x para buscar máximos (nm en modo lambda, eV en modo energy).",
    )
    parser.add_argument(
        "--maxeps",
        type=float,
        help="(Modo carpeta) Umbral mínimo de epsilon para listar máximos.",
    )
    parser.add_argument(
        "--maxonly",
        default=None,
        action="store_true",
        help="(Modo carpeta) Listar solo el máximo absoluto dentro del rango.",
    )
    parser.add_argument(
        "--allpeaks",
        default=None,
        action="store_true",
        help="(Modo carpeta) Listar todos los máximos locales dentro del rango.",
    )

    # HTML interactivo (plotly) en modo carpeta: activado por defecto
    html_group = parser.add_mutually_exclusive_group()
    html_group.add_argument(
        "--html",
        dest="html",
        action="store_true",
        help="(Modo carpeta) Generar HTML interactivo (default: activado).",
    )
    html_group.add_argument(
        "--no-html",
        dest="html",
        action="store_false",
        help="(Modo carpeta) No generar HTML interactivo.",
    )
    parser.set_defaults(html=True)

    # exportar datos en modo archivo único
    parser.add_argument(
        "-e",
        "--export",
        default=False,
        action="store_true",
        help="(Modo archivo único) Exportar datos ε como espectro_*.dat (lamb espectroE_*.dat (energy).",
    )

    args = parser.parse_args()

    # --- defaults "suaves" ---
    # Si no se dio -x0/-x1, usar 250–1000 nm
    if args.startx is None:
        args.startx = 250.0
    if args.endx is None:
        args.endx = 1000.0

    # Si el usuario no puso -s/--show, activamos show por defecto
    if not any(a in ("-s", "--show") for a in sys.argv[1:]):
        args.show = True

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
