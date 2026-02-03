#!/usr/bin/env python3
import glob
import re
import sys
import math

def get_sorted_ms_files():
    """
    Busca archivos ms_*.dat y los ordena por el número.
    """
    files = []
    for fname in glob.glob("ms_*.dat"):
        m = re.match(r"ms_(\d+)\.dat$", fname)
        if m:
            idx = int(m.group(1))
            files.append((idx, fname))
    files.sort(key=lambda x: x[0])
    return [f for _, f in files]


def merge_files(files, outname="ms_full.dat"):
    """
    Concatena los archivos de la lista 'files' en 'outname'.
    """
    with open(outname, "w") as out:
        for fname in files:
            with open(fname, "r") as inp:
                for line in inp:
                    out.write(line)
    print(f"[OK] Archivo combinado escrito en '{outname}'.")


def build_timeseries(fullfile, dt_ps, atom_ids, outname="ms_spin_timeseries.dat"):
    """
    Lee ms_full.dat y arma la serie temporal de spin para los átomos seleccionados.

    - fullfile: archivo combinado (ms_full.dat)
    - dt_ps: time step en picosegundos (float)
    - atom_ids: lista de números de átomo a trackear (lista de int)
    - outname: nombre del archivo de salida con la serie temporal
    """
    atom_ids = list(atom_ids)
    data = []  # cada fila: [time_ps, spin_atom1, spin_atom2, ...]
    current_spins = {aid: None for aid in atom_ids}
    frame_index = -1
    inside_frame = False

    with open(fullfile, "r") as f:
        for line in f:
            stripped = line.strip()

            # Inicio de bloque de Mulliken Spin
            if stripped.startswith("# Mulliken Spin Population Analysis"):
                frame_index += 1
                inside_frame = True
                current_spins = {aid: None for aid in atom_ids}
                continue

            if not inside_frame:
                continue

            # Fin de bloque: línea de carga total
            if "Total Charge" in stripped:
                time_ps = frame_index * dt_ps
                row = [time_ps]

                # Aseguramos que, si falta algún spin, se escriba NaN
                for aid in atom_ids:
                    spin_val = current_spins.get(aid, None)
                    if spin_val is None:
                        spin_val = float("nan")
                    row.append(spin_val)

                data.append(row)
                inside_frame = False
                continue

            # Saltar comentarios o líneas vacías
            if not stripped or stripped.startswith("#"):
                continue

            # Líneas de átomos: "   1      6     -0.0003077"
            parts = stripped.split()
            if len(parts) >= 3 and parts[0].isdigit():
                atom_idx = int(parts[0])
                try:
                    spin_value = float(parts[2])
                except ValueError:
                    continue

                if atom_idx in current_spins:
                    current_spins[atom_idx] = spin_value

    # Escribir el archivo de salida
    with open(outname, "w") as out:
        header = ["time_ps"] + [f"spin_atom_{aid}" for aid in atom_ids]
        out.write("# " + " ".join(header) + "\n")
        for row in data:
            # Primer valor es el tiempo; el resto son spins
            formatted = []
            for i, val in enumerate(row):
                if isinstance(val, float):
                    formatted.append(f"{val: .7f}")
                else:
                    formatted.append(str(val))
            out.write(" ".join(formatted) + "\n")

    print(f"[OK] Serie temporal escrita en '{outname}' con {len(data)} frames.")
    print(f"    Columnas: {', '.join(header)}")


def main():
    # 1) Buscar y ordenar archivos ms_*.dat
    files = get_sorted_ms_files()
    if not files:
        print("No se encontraron archivos 'ms_*.dat' en el directorio actual.")
        sys.exit(1)

    print("Archivos que se van a combinar (en orden):")
    for fname in files:
        print("  ", fname)

    # 2) Combinar en ms_full.dat
    merge_files(files, outname="ms_full.dat")

    # 3) Preguntar por dt y átomos a trackear
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

    # 4) Construir la serie temporal
    build_timeseries("ms_full.dat", dt_ps, atom_ids, outname="ms_spin_timeseries.dat")


if __name__ == "__main__":
    main()

