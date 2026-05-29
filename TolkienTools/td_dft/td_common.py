#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import os
import re
import shutil
import subprocess
import webbrowser

# Physical constants (SI)
e_charge = 1.602176634e-19      # C
hbar = 1.054571817e-34          # J*s
m_e = 9.1093837015e-31          # kg
c_light = 299792458.0           # m/s
eps0 = 8.8541878128e-12         # F/m
N_A = 6.02214076e23             # mol^-1
LN10 = math.log(10.0)

# hc in eV*nm
HC_EV_NM = 1239.8419843320026



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


def td_file_sort_key(path):
    """
    Orden natural para archivos tipo TD_N.out (N numérico).
    """
    base = os.path.basename(path)
    match = re.search(r"TD_(\d+)", base)
    if match:
        return (0, int(match.group(1)), base.lower())
    return (1, 0, base.lower())


def parse_number_ranges(list_str, flag_name="--exclude"):
    """
    Parsea una lista de números y rangos tipo:
      '10'          -> {10}
      '10,12,15'    -> {10,12,15}
      '10-13,20'    -> {10,11,12,13,20}
    """
    if list_str is None:
        return set()

    nums = set()
    parts = list_str.split(",")
    for part in parts:
        p = part.strip()
        if not p:
            continue
        if "-" in p:
            a_str, b_str = p.split("-", 1)
            try:
                a = int(a_str)
                b = int(b_str)
            except ValueError:
                print(f"Advertencia: valor de {flag_name} ignorado: '{p}' (rango inválido)")
                continue
            if a > b:
                print(f"Advertencia: valor de {flag_name} ignorado: '{p}' (rango inválido)")
                continue
            for k in range(a, b + 1):
                nums.add(k)
        else:
            try:
                k = int(p)
            except ValueError:
                print(f"Advertencia: valor de {flag_name} ignorado: '{p}' (no es entero)")
                continue
            nums.add(k)

    return nums


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
