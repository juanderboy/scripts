#!/usr/bin/env python3
"""Adapt ORCA QM/MM inputs from an MD snapshot by adding nearby QM waters.

Input:
- AMBER topology (.prmtop) and restart (.rst7) from a molecular dynamics
  snapshot.

Behavior:
- Main goal: identify water residues close to the solute and update ORCA QM/MM
  atom lists (QMAtoms/ActiveAtoms) so those waters are included in the QM region.
- By default, the solute is assumed to be the first residue in the topology.
- If any atom of a water residue is within the cutoff distance from any solute atom,
  the full water residue is selected (all its atoms).
- Outputs selected atom indices in 0-based convention (ORCA/QM/MM friendly).
- Optionally clones an ORCA input and updates QMAtoms/ActiveAtoms blocks.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class AmberTopology:
    natom: int
    residue_labels: list[str]
    residue_pointers: list[int]  # 1-based atom start index per residue


@dataclass
class Residue:
    index: int  # 1-based residue index
    label: str
    start_atom: int  # 1-based inclusive
    end_atom: int  # 1-based inclusive


def _parse_prmtop_flags(prmtop_path: Path) -> dict[str, str]:
    flag_blocks: dict[str, list[str]] = {}
    current_flag: str | None = None

    with prmtop_path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n")
            if line.startswith("%FLAG"):
                current_flag = line.split(maxsplit=1)[1].strip()
                flag_blocks[current_flag] = []
                continue
            if current_flag is None or line.startswith("%FORMAT"):
                continue
            flag_blocks[current_flag].append(line)

    return {name: "\n".join(lines) for name, lines in flag_blocks.items()}


def _int_tokens(text: str) -> list[int]:
    return [int(tok) for tok in text.split()]


def _str_tokens(text: str) -> list[str]:
    return text.split()


def read_prmtop(prmtop_path: Path) -> AmberTopology:
    flags = _parse_prmtop_flags(prmtop_path)

    needed = ("POINTERS", "RESIDUE_LABEL", "RESIDUE_POINTER")
    for flag in needed:
        if flag not in flags:
            raise ValueError(f"No se encontró %FLAG {flag} en {prmtop_path}")

    pointers = _int_tokens(flags["POINTERS"])
    if len(pointers) < 12:
        raise ValueError("El bloque POINTERS no tiene suficientes valores")

    natom = pointers[0]
    nres = pointers[11]

    residue_labels = _str_tokens(flags["RESIDUE_LABEL"])
    residue_pointers = _int_tokens(flags["RESIDUE_POINTER"])

    if len(residue_labels) != nres:
        raise ValueError(
            f"RESIDUE_LABEL inconsistente: esperados {nres}, encontrados {len(residue_labels)}"
        )
    if len(residue_pointers) != nres:
        raise ValueError(
            f"RESIDUE_POINTER inconsistente: esperados {nres}, encontrados {len(residue_pointers)}"
        )

    return AmberTopology(natom=natom, residue_labels=residue_labels, residue_pointers=residue_pointers)


def _parse_fixed_width_floats(lines: Iterable[str], width: int = 12) -> list[float]:
    values: list[float] = []
    for line in lines:
        for i in range(0, len(line), width):
            chunk = line[i : i + width].strip()
            if not chunk:
                continue
            values.append(float(chunk))
    return values


def read_rst7_coords(rst7_path: Path) -> tuple[int, list[tuple[float, float, float]]]:
    with rst7_path.open("r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()

    if len(lines) < 2:
        raise ValueError(f"Archivo rst7 inválido: {rst7_path}")

    second_line_parts = lines[1].split()
    if not second_line_parts:
        raise ValueError(f"No se pudo leer NATOM en la línea 2 de {rst7_path}")

    natom = int(float(second_line_parts[0]))

    numeric_values = _parse_fixed_width_floats(lines[2:], width=12)
    needed = 3 * natom
    if len(numeric_values) < needed:
        raise ValueError(
            f"No hay suficientes coordenadas en {rst7_path}: "
            f"se esperaban al menos {needed} valores y hay {len(numeric_values)}"
        )

    coords_flat = numeric_values[:needed]
    coords = [
        (coords_flat[i], coords_flat[i + 1], coords_flat[i + 2]) for i in range(0, needed, 3)
    ]
    return natom, coords


def build_residues(top: AmberTopology) -> list[Residue]:
    residues: list[Residue] = []
    for i, (label, start) in enumerate(zip(top.residue_labels, top.residue_pointers), start=1):
        if i < len(top.residue_pointers):
            end = top.residue_pointers[i] - 1
        else:
            end = top.natom
        residues.append(Residue(index=i, label=label, start_atom=start, end_atom=end))
    return residues


def find_close_water_atoms(
    coords: list[tuple[float, float, float]],
    residues: list[Residue],
    solute_atom_count: int,
    water_labels: set[str],
    cutoff: float,
) -> tuple[list[int], list[int]]:
    cutoff2 = cutoff * cutoff
    solute_coords = coords[:solute_atom_count]

    close_residue_ids: list[int] = []
    close_atoms: list[int] = []

    for residue in residues:
        if residue.label not in water_labels:
            continue

        close = False
        for atom_1b in range(residue.start_atom, residue.end_atom + 1):
            x, y, z = coords[atom_1b - 1]
            for sx, sy, sz in solute_coords:
                dx = x - sx
                dy = y - sy
                dz = z - sz
                if dx * dx + dy * dy + dz * dz <= cutoff2:
                    close = True
                    break
            if close:
                break

        if close:
            close_residue_ids.append(residue.index)
            close_atoms.extend(range(residue.start_atom, residue.end_atom + 1))

    return close_residue_ids, close_atoms


def to_zero_based(indices_1b: list[int]) -> list[int]:
    return [i - 1 for i in indices_1b]


def to_orca_ranges(atom_indices_0b: list[int]) -> str:
    if not atom_indices_0b:
        return ""

    ranges: list[tuple[int, int]] = []
    start = atom_indices_0b[0]
    prev = atom_indices_0b[0]

    for idx in atom_indices_0b[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        ranges.append((start, prev))
        start = idx
        prev = idx
    ranges.append((start, prev))

    parts: list[str] = []
    for a, b in ranges:
        if a == b:
            parts.append(str(a))
        else:
            parts.append(f"{a}:{b}")
    return " ".join(parts)


def write_orca_with_close_waters(
    orca_input: Path, orca_output: Path, atom_indices_0b: list[int]
) -> tuple[bool, bool]:
    text = orca_input.read_text(encoding="utf-8", errors="replace")
    atom_expr = to_orca_ranges(atom_indices_0b)
    if not atom_expr:
        raise ValueError("No hay átomos para escribir en QMAtoms/ActiveAtoms")

    qmatoms_pattern = re.compile(
        r"(^\s*QMAtoms\s*)\{[^}]*\}(\s*end\b[^\n]*$)", re.MULTILINE
    )
    activeatoms_pattern = re.compile(
        r"(^\s*ActiveAtoms\s*)\{[^}]*\}(\s*end\b[^\n]*$)", re.MULTILINE
    )

    qmatoms_found = bool(qmatoms_pattern.search(text))
    activeatoms_found = bool(activeatoms_pattern.search(text))

    text = qmatoms_pattern.sub(rf"\1{{{atom_expr}}}\2", text, count=1)
    text = activeatoms_pattern.sub(rf"\1{{{atom_expr}}}\2", text, count=1)

    orca_output.write_text(text, encoding="utf-8")
    return qmatoms_found, activeatoms_found


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Adapta un input QM/MM de ORCA a partir de un snapshot de dinámica "
            "molecular (AMBER .prmtop + .rst7).\n"
            "Detecta aguas cercanas al soluto y las incorpora como residuos "
            "completos en la región QM."
        ),
        epilog=(
            "Notas:\n"
            "- Índices reportados: 0-based (compatibles con ORCA).\n"
            "- Si no usás --solute-atoms, el soluto se infiere como el primer residuo.\n"
            "- Etiqueta de agua por defecto: WAT (configurable con --water-labels).\n"
            "- Con --orca-input, el script genera un nuevo input ORCA con QMAtoms y "
            "ActiveAtoms actualizados.\n"
            "\n"
            "Ejemplos:\n"
            "  python3 find_close_waters.py --prmtop sys.prmtop --rst7 snap.rst7 --cutoff 3.5\n"
            "  python3 find_close_waters.py --prmtop sys.prmtop --rst7 snap.rst7 --cutoff 3.5 \\\n"
            "    --output close_atoms.txt --orca-input base.inp --orca-output qm_region.inp"
        ),
    )
    parser.add_argument("--prmtop", required=True, help="Archivo topología AMBER (.prmtop)")
    parser.add_argument("--rst7", required=True, help="Archivo restart AMBER (.rst7)")
    parser.add_argument(
        "--cutoff",
        type=float,
        required=True,
        help="Distancia umbral en Angstrom para definir cercanía",
    )
    parser.add_argument(
        "--solute-atoms",
        type=int,
        default=None,
        help=(
            "Cantidad de átomos de soluto (si no se da, se infiere como el tamaño "
            "del primer residuo del prmtop)"
        ),
    )
    parser.add_argument(
        "--water-labels",
        default="WAT",
        help="Etiquetas de agua separadas por coma (ej: WAT,HOH)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Archivo de salida opcional para los números de átomo (uno por línea)",
    )
    parser.add_argument(
        "--orca-input",
        default=None,
        help="Input ORCA base a clonar y actualizar en QMAtoms/ActiveAtoms",
    )
    parser.add_argument(
        "--orca-output",
        default=None,
        help="Nuevo archivo ORCA de salida (si no se da, usa *_with_close_waters.inp)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    prmtop_path = Path(args.prmtop)
    rst7_path = Path(args.rst7)

    top = read_prmtop(prmtop_path)
    natom_rst, coords = read_rst7_coords(rst7_path)

    if natom_rst != top.natom:
        raise ValueError(
            f"NATOM inconsistente: prmtop={top.natom}, rst7={natom_rst}. "
            "Verificá que sean del mismo sistema."
        )

    residues = build_residues(top)

    if args.solute_atoms is None:
        if not residues:
            raise ValueError("No hay residuos en el prmtop")
        solute_atom_count = residues[0].end_atom - residues[0].start_atom + 1
    else:
        solute_atom_count = args.solute_atoms

    if solute_atom_count <= 0 or solute_atom_count > top.natom:
        raise ValueError(f"Cantidad de átomos de soluto inválida: {solute_atom_count}")

    water_labels = {lbl.strip() for lbl in args.water_labels.split(",") if lbl.strip()}
    if not water_labels:
        raise ValueError("No se indicó ninguna etiqueta de agua válida")

    close_residues, close_atoms = find_close_water_atoms(
        coords=coords,
        residues=residues,
        solute_atom_count=solute_atom_count,
        water_labels=water_labels,
        cutoff=args.cutoff,
    )

    header = [
        f"Soluto (átomos 0-{solute_atom_count - 1})",
        f"Cutoff: {args.cutoff:.3f} A",
        f"Residuos de agua encontrados: {len(close_residues)}",
        f"Átomos de agua reportados: {len(close_atoms)}",
    ]
    print("\n".join(header))

    close_atoms_0b = to_zero_based(close_atoms)

    if close_atoms:
        print("Atom numbers (0-based):")
        print(" ".join(str(a) for a in close_atoms_0b))

    if args.output:
        out = Path(args.output)
        out.write_text(
            "\n".join(str(a) for a in close_atoms_0b) + ("\n" if close_atoms_0b else ""),
            encoding="utf-8",
        )

    if args.orca_input:
        orca_input = Path(args.orca_input)
        if args.orca_output:
            orca_output = Path(args.orca_output)
        else:
            orca_output = orca_input.with_name(f"{orca_input.stem}_with_close_waters{orca_input.suffix}")

        # ORCA indexa desde 0; AMBER en este script está en 1.
        solute_atoms_0b = [i - 1 for i in range(1, solute_atom_count + 1)]
        all_atoms_0b = sorted(set(solute_atoms_0b + close_atoms_0b))

        found_qm, found_active = write_orca_with_close_waters(
            orca_input=orca_input,
            orca_output=orca_output,
            atom_indices_0b=all_atoms_0b,
        )
        print(f"Archivo ORCA generado: {orca_output}")
        if not found_qm:
            print("Aviso: no se encontró línea QMAtoms para actualizar.")
        if not found_active:
            print("Aviso: no se encontró línea ActiveAtoms para actualizar.")


if __name__ == "__main__":
    main()
