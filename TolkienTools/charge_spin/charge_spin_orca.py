#!/usr/bin/env python3
"""ORCA-specific readers and population-table converters.

This module discovers ordered ORCA outputs, extracts available population
analyses and writes normalized intermediate files consumed by the statistics.
"""

import glob
import os
import re
import sys

from charge_spin_common import get_analysis_display_label


def get_sorted_orca_files(prefix=None):
    """
    Find ORCA files named <prefix>_N.out or <prefix>_N.dat
    and sort them by frame number N.

    If prefix is None or "", autodetect any prefix.
    """
    normalized_prefix = prefix.strip() if prefix else None
    patterns = [f"{normalized_prefix}_*.out", f"{normalized_prefix}_*.dat"] if normalized_prefix else ["*_*.out", "*_*.dat"]
    files = {}
    detected_prefixes = set()

    if normalized_prefix:
        regex = re.compile(rf"^{re.escape(normalized_prefix)}_(\d+)\.(out|dat)$")
    else:
        regex = re.compile(r"^(.+?)_(\d+)\.(out|dat)$")

    for pattern in patterns:
        for fname in glob.glob(pattern):
            m = regex.match(fname)
            if not m:
                continue

            if normalized_prefix:
                idx = int(m.group(1))
                ext = m.group(2)
            else:
                detected_prefix = m.group(1)
                idx = int(m.group(2))
                ext = m.group(3)
                detected_prefixes.add(detected_prefix)

            # Prefer .out if duplicates are present
            if idx not in files or ext == "out":
                files[idx] = fname

    if not normalized_prefix and len(detected_prefixes) > 1:
        print("Error: multiple ORCA prefixes were detected in the current directory:")
        for detected_prefix in sorted(detected_prefixes):
            print(f"  {detected_prefix}")
        print("Please enter the desired prefix explicitly to avoid mixing files.")
        sys.exit(1)

    return [files[k] for k in sorted(files.keys())]


def extract_orca_population_block(fname, header_line):
    """
    Extract a charge/spin table (Mulliken or Loewdin) from an ORCA output.
    Return a list of tuples: (atom_idx, element, charge, spin)
    """
    data = []
    in_section = False
    started_rows = False
    with open(fname, "r") as f:
        for line in f:
            if header_line in line:
                in_section = True
                started_rows = False
                continue
            if not in_section:
                continue

            stripped = line.strip()
            if not stripped:
                if started_rows:
                    break
                continue
            if started_rows and not re.match(r"^\s*\d+\s+[A-Za-z]+", line):
                break
            if stripped.startswith("Sum of atomic charges"):
                break
            if stripped.startswith("MULLIKEN REDUCED"):
                break
            if stripped.startswith("Total integrated alpha density"):
                continue
            if stripped.startswith("Total integrated beta density"):
                continue
            if stripped.startswith("ATOM"):
                continue
            if stripped.startswith("---"):
                continue

            m = re.match(r"^\s*(\d+)\s+([A-Za-z]+)\s*(?::\s*)?([-\d\.Ee+]+)\s+([-\d\.Ee+]+)\s*$", line)
            if m:
                started_rows = True
                atom_idx = int(m.group(1))
                element = m.group(2)
                charge = float(m.group(3))
                spin = float(m.group(4))
                data.append((atom_idx, element, charge, spin))

    return data


def detect_orca_multiplicity(fname):
    """
    Extract the spin multiplicity reported in an ORCA output.
    Return an integer or None if it was not found.
    """
    patterns = [
        re.compile(r"^\s*Multiplicity\s+Mult\s+\.\.\.\.\s+(-?\d+)\s*$"),
        re.compile(r"^\s*Multiplicity\s*:\s*(-?\d+)\s*$"),
    ]

    with open(fname, "r") as f:
        for line in f:
            for pattern in patterns:
                m = pattern.match(line)
                if m:
                    return int(m.group(1))

    return None


def extract_orca_charge_only_block(fname, header_line):
    """
    Extract a charge table without a spin column from an ORCA output.
    Return a list of tuples: (atom_idx, element, charge)
    """
    data = []
    in_section = False
    started_rows = False
    with open(fname, "r") as f:
        for line in f:
            if header_line in line:
                in_section = True
                started_rows = False
                continue
            if not in_section:
                continue

            stripped = line.strip()
            if not stripped:
                if started_rows:
                    break
                continue
            if started_rows and not re.match(r"^\s*\d+\s+[A-Za-z]+", line):
                break
            if stripped.startswith("Total charge"):
                break
            if stripped.startswith("CHELPG charges calculated"):
                break
            if stripped.startswith("Sum of atomic charges"):
                break
            if stripped.startswith("MULLIKEN REDUCED"):
                break
            if stripped.startswith("Total integrated alpha density"):
                continue
            if stripped.startswith("Total integrated beta density"):
                continue
            if stripped.startswith("ATOM"):
                continue
            if stripped.startswith("---"):
                continue

            m = re.match(r"^\s*(\d+)\s+([A-Za-z]+)\s*(?::\s*)?([-\d\.Ee+]+)\s*$", line)
            if m:
                started_rows = True
                atom_idx = int(m.group(1))
                element = m.group(2)
                charge = float(m.group(3))
                data.append((atom_idx, element, charge))

    return data


def build_orca_charge_file(
    orca_files,
    out_charge,
    charge_label,
    charge_header_line
):
    """
    Build a merged charge-only file from multiple ORCA <prefix>_N.out/.dat files.
    """
    if not orca_files:
        print("Error: no ORCA files matching '<prefix>_N.out' or '<prefix>_N.dat' were found.")
        sys.exit(1)

    with open(out_charge, "w") as q_out:
        for fname in orca_files:
            charge_block = extract_orca_charge_only_block(fname, charge_header_line)
            if not charge_block:
                print(f"[WARN] {charge_label} table not found in '{fname}'.")
                continue

            q_out.write(f"# {charge_label} Population Analysis\n")
            q_out.write("# Atom   Type   Population\n")

            sum_q = 0.0
            for atom_idx, element, charge in charge_block:
                q_out.write(f"{atom_idx:4d} {element:>3s} {charge: .7f}\n")
                sum_q += charge

            q_out.write(f"  Total Charge = {sum_q: .7f}\n\n")

    print(f"[OK] ORCA charge file ({charge_label}) merged into '{out_charge}'.")


def build_orca_full_files(
    orca_files,
    out_charge,
    out_spin,
    charge_label,
    charge_header_line,
    spin_label=None,
    spin_header_line=None
):
    """
    Build merged charge and spin files from multiple ORCA <prefix>_N.out/.dat files
    using the format expected by build_timeseries_and_stats.
    """
    if not orca_files:
        print("Error: no ORCA files matching '<prefix>_N.out' or '<prefix>_N.dat' were found.")
        sys.exit(1)

    with open(out_charge, "w") as q_out, open(out_spin, "w") as s_out:
        for fname in orca_files:
            if spin_header_line is None:
                block = extract_orca_population_block(fname, charge_header_line)
                if not block:
                    print(f"[WARN] {charge_label} table not found in '{fname}'.")
                    continue
                rows = block
                spin_header_label = spin_label or charge_label
            else:
                charge_block = extract_orca_charge_only_block(fname, charge_header_line)
                spin_block = extract_orca_population_block(fname, spin_header_line)
                if not charge_block:
                    print(f"[WARN] {charge_label} table not found in '{fname}'.")
                    continue
                if not spin_block:
                    spin_source = spin_label or "spin"
                    print(f"[WARN] {spin_source} spin table not found in '{fname}'.")
                    continue

                spin_by_atom = {
                    atom_idx: (element, spin)
                    for atom_idx, element, _charge, spin in spin_block
                }
                rows = []
                missing_spin = False
                for atom_idx, element, charge in charge_block:
                    spin_info = spin_by_atom.get(atom_idx)
                    if spin_info is None:
                        print(
                            f"[WARN] Missing spin for atom {atom_idx} while combining "
                            f"{charge_label} + {spin_label or 'spin'} in '{fname}'."
                        )
                        missing_spin = True
                        break
                    spin_element, spin = spin_info
                    rows.append((atom_idx, spin_element or element, charge, spin))
                if missing_spin:
                    continue
                spin_header_label = spin_label or charge_label

            q_out.write(f"# {charge_label} Population Analysis\n")
            q_out.write("# Atom   Type   Population\n")
            s_out.write(f"# {spin_header_label} Spin Population Analysis\n")
            s_out.write("# Atom   Type   Population\n")

            sum_q = 0.0
            sum_s = 0.0
            for atom_idx, element, charge, spin in rows:
                q_out.write(f"{atom_idx:4d} {element:>3s} {charge: .7f}\n")
                s_out.write(f"{atom_idx:4d} {element:>3s} {spin: .7f}\n")
                sum_q += charge
                sum_s += spin

            # Closing line compatible with the parser
            q_out.write(f"  Total Charge = {sum_q: .7f}\n\n")
            s_out.write(f"  Total Charge = {sum_s: .7f}\n\n")

    spin_print_label = spin_label or charge_label
    print(f"[OK] ORCA charge file ({charge_label}) merged into '{out_charge}'.")
    print(f"[OK] ORCA spin file ({spin_print_label}) merged into '{out_spin}'.")
