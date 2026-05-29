#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re

import numpy as np


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
