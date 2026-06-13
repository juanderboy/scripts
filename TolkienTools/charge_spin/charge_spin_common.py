#!/usr/bin/env python3
"""Shared configuration and prompt helpers for charge/spin analysis.

The functions here parse common user selections, labels, percentages and
output names used by the interactive analysis workflow.
"""

import os
import re
import sys

import numpy as np


def prompt_numbered_choice(title, options, default_idx=None):
    """
    Show a numbered menu and return the value associated with the chosen option.
    options: list of tuples (label, value)
    default_idx: 0-based index of the default option, or None
    """
    print(title)
    for idx, (label, _value) in enumerate(options, start=1):
        default_tag = " [default]" if default_idx is not None and idx - 1 == default_idx else ""
        print(f"  {idx}. {label}{default_tag}")

    prompt = "Select an option by number"
    if default_idx is not None:
        prompt += f" (Enter = {default_idx + 1})"
    prompt += ": "

    choice = input(prompt).strip()
    if choice == "":
        if default_idx is None:
            print("Error: you must select an option.")
            sys.exit(1)
        return options[default_idx][1]

    if not choice.isdigit():
        print("Error: you must enter the number of an option.")
        sys.exit(1)

    selected = int(choice)
    if selected < 1 or selected > len(options):
        print("Error: option out of range.")
        sys.exit(1)

    return options[selected - 1][1]


def find_first_existing_file(candidates):
    for fname in candidates:
        if os.path.isfile(fname):
            return fname
    return None


def print_lio_merge_pop_hint(kind):
    print(f"Falta el archivo consolidado de {kind}.")
    print("Puede construirlo yendo al modulo 1 de Tolkien Tools: Molecular dynamics processing.")
    print("Desde la carpeta raiz de la dinamica fragmentada, use:")
    print("  tolkien-tools 1")
    print("  elegir: inspect-merge")
    print("  aceptar el merge de XYZ/cargas-spines")
    print("Tambien puede correrlo directo con:")
    print("  tolkien-tools md inspect-merge --merge yes")


def print_welcome_banner():
    """
    Print the program welcome banner.
    """
    print("")
    print("=" * 78)
    print("                  CHARGE AND SPIN ENSEMBLE ANALYZER")
    print("=" * 78)
    print("        Statistical analysis of charges and spin populations")
    print("        Time series, KDE modes, histograms, and comparisons")
    print("")
    print("                 Vibecoded by Tolkien, 2026")
    print("=" * 78)
    print("")


def sanitize_output_token(value):
    """
    Convert an identifier into a safe token for headers and filenames.
    """
    text = str(value).strip()
    if not text:
        return "unnamed"
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    sanitized = sanitized.strip("._-")
    return sanitized or "unnamed"


def append_suffix_to_path(path, suffix):
    """
    Insert a suffix before the file extension.
    """
    root, ext = os.path.splitext(path)
    return f"{root}_{suffix}{ext}"


def format_summary_stat(value):
    """
    Format a numeric summary value for terminal output.
    """
    if value is None:
        return "n/a"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isnan(value):
        return "n/a"
    return f"{value:.4f}"


def parse_atom_id_list(atom_ids_str):
    """
    Convert a space-separated string of integers into a list of atom IDs.
    """
    try:
        atom_ids = [int(x) for x in atom_ids_str.split()]
    except ValueError:
        print("Error: atom numbers must be integers separated by spaces.")
        sys.exit(1)
    return atom_ids


def parse_global_entity_list(entity_ids_str):
    """
    Convert a space-separated string into global entity IDs.
    Accept integers and the special token 'coque'.
    """
    entity_ids = []
    for token in entity_ids_str.split():
        lowered = token.lower()
        if lowered == "coque":
            entity_ids.append("coque")
            continue
        try:
            entity_ids.append(int(token))
        except ValueError:
            print("Error: global selections must be atom numbers and may optionally include the token 'coque'.")
            sys.exit(1)
    return entity_ids


def get_spin_representation_config(choice):
    """
    Return the configuration associated with the selected spin representation.
    """
    if choice == "fraction":
        return {
            "normalize": True,
            "column_label": "spin_fraction",
            "axis_label": "Spin fraction",
            "global_axis_label": "Spin fraction"
        }
    return {
        "normalize": False,
        "column_label": "spin",
        "axis_label": "Spin",
        "global_axis_label": None
    }


def get_population_analysis_config(choice):
    """
    Return which population analyses must be processed.
    """
    if choice == "mulliken":
        return {"mulliken": True, "loewdin": False, "hirshfeld": False, "chelpg_loewdin": False, "chelpg_mulliken": False, "chelpg_hirshfeld": False}
    if choice == "loewdin":
        return {"mulliken": False, "loewdin": True, "hirshfeld": False, "chelpg_loewdin": False, "chelpg_mulliken": False, "chelpg_hirshfeld": False}
    if choice == "hirshfeld":
        return {"mulliken": False, "loewdin": False, "hirshfeld": True, "chelpg_loewdin": False, "chelpg_mulliken": False, "chelpg_hirshfeld": False}
    if choice == "chelpg_loewdin":
        return {"mulliken": False, "loewdin": False, "hirshfeld": False, "chelpg_loewdin": True, "chelpg_mulliken": False, "chelpg_hirshfeld": False}
    if choice == "chelpg_mulliken":
        return {"mulliken": False, "loewdin": False, "hirshfeld": False, "chelpg_loewdin": False, "chelpg_mulliken": True, "chelpg_hirshfeld": False}
    if choice == "chelpg_hirshfeld":
        return {"mulliken": False, "loewdin": False, "hirshfeld": False, "chelpg_loewdin": False, "chelpg_mulliken": False, "chelpg_hirshfeld": True}
    return {"mulliken": True, "loewdin": True, "hirshfeld": True, "chelpg_loewdin": False, "chelpg_mulliken": False, "chelpg_hirshfeld": True}


def get_histogram_binning_config(choice):
    """
    Return the bin specification for histograms.
    """
    if choice == "fixed_custom":
        return "fixed_custom"
    if choice == "sturges":
        return "sturges"
    if choice == "auto":
        return "auto"
    return "fd"


def get_analysis_display_label(analysis_kind):
    """
    Return a human-readable label for the population analysis.
    """
    labels = {
        "mulliken": "Mulliken",
        "loewdin": "Loewdin",
        "hirshfeld": "Hirshfeld",
        "chelpg": "CHELPG",
        "chelpg_loewdin": "CHELPG",
        "chelpg_mulliken": "CHELPG",
        "chelpg_hirshfeld": "CHELPG",
    }
    return labels.get(analysis_kind, analysis_kind.capitalize())


def resolve_histogram_bins_spec(choice):
    """
    Resolve the final bin specification, including a user-selected fixed-bin variant.
    """
    bins_spec = get_histogram_binning_config(choice)
    if bins_spec != "fixed_custom":
        return bins_spec

    bins_str = input("Enter the fixed number of bins to use (Enter = 50): ").strip()
    if bins_str == "":
        return 50
    if not bins_str.isdigit():
        print("Error: the number of bins must be a positive integer.")
        sys.exit(1)

    bins_value = int(bins_str)
    if bins_value <= 0:
        print("Error: the number of bins must be greater than zero.")
        sys.exit(1)
    return bins_value


def prompt_additional_atom_selection(available_atom_ids, excluded_atom_ids=None):
    """
    Ask the user for atom IDs to add and validate the selection.
    """
    excluded_atom_ids = set(excluded_atom_ids or [])
    atoms_str = input(
        "Enter the atom numbers to add, separated by spaces: "
    ).strip()
    atom_ids = parse_atom_id_list(atoms_str)
    if not atom_ids:
        print("Error: no atoms were provided to add.")
        sys.exit(1)

    invalid = [aid for aid in atom_ids if aid not in available_atom_ids]
    if invalid:
        print(
            "Error: these atoms are not available in the current analysis: "
            + " ".join(str(aid) for aid in invalid)
        )
        sys.exit(1)

    duplicates = [aid for aid in atom_ids if aid in excluded_atom_ids]
    if duplicates:
        print(
            "Error: these atoms are already part of the analysis: "
            + " ".join(str(aid) for aid in duplicates)
        )
        sys.exit(1)

    deduped = []
    for aid in atom_ids:
        if aid not in deduped:
            deduped.append(aid)
    return deduped


def prompt_float_value(prompt, default_value, min_value=None, max_value=None):
    """
    Read a floating point value from stdin, accepting Enter as a default.
    """
    value_str = input(prompt).strip()
    if value_str == "":
        return float(default_value)
    try:
        value = float(value_str)
    except ValueError:
        print("Error: the value must be a number.")
        sys.exit(1)
    if min_value is not None and value < min_value:
        print(f"Error: the value must be >= {min_value}.")
        sys.exit(1)
    if max_value is not None and value > max_value:
        print(f"Error: the value must be <= {max_value}.")
        sys.exit(1)
    return value


def normalize_percent_input(value):
    """
    Accept either fractions (0.05) or percentages (5) and return a fraction.
    """
    value = float(value)
    if value > 1.0:
        value /= 100.0
    return value
