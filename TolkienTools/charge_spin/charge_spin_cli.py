#!/usr/bin/env python3
import os
import sys

import numpy as np

from charge_spin_common import (
    append_suffix_to_path,
    find_first_existing_file,
    get_analysis_display_label,
    get_population_analysis_config,
    get_spin_representation_config,
    normalize_percent_input,
    parse_global_entity_list,
    print_lio_merge_pop_hint,
    print_welcome_banner,
    prompt_additional_atom_selection,
    prompt_float_value,
    prompt_numbered_choice,
    resolve_histogram_bins_spec,
    sanitize_output_token,
)
from charge_spin_global import (
    collect_global_hist_data,
    discover_global_analysis_dirs,
    export_global_plot_data,
    export_global_snapshot_counts,
    infer_entities_from_previous_analysis,
    make_global_overlay_hist_figure,
)
from charge_spin_io import (
    get_atom_list_from_full,
    get_sorted_files,
    merge_files,
    parse_frame_data,
    write_combined_entity_timeseries,
)
from charge_spin_orca import (
    build_orca_charge_file,
    build_orca_full_files,
    detect_orca_multiplicity,
    get_sorted_orca_files,
)
from charge_spin_plotting import make_combined_hist_figure, make_timeseries_figure
from charge_spin_stats import (
    analyze_spin_consistency,
    apply_keep_mask_or_warn,
    build_analysis_timeseries_and_stats,
    build_global_terminal_summary_entry,
    build_terminal_summary_entry,
    get_analysis_entity_ids,
    print_terminal_analysis_summary,
    print_terminal_global_summary,
    select_spin_localization_atoms,
    suggest_missing_spin_atoms,
)
from charge_spin_viewer import (
    find_default_xyz_for_spin_viewer,
    find_orca_geometry_file_for_viewer,
    open_html_viewer,
    write_orca_spin_localization_viewer,
    write_spin_localization_viewer,
)


def main():
    print_welcome_banner()
    mode = prompt_numbered_choice(
        "Execution mode:",
        [("Single-system analysis", "i"), ("Global analysis across subdirectories", "g")],
        default_idx=0
    )

    if mode in ("g",):
        global_binning_choice = prompt_numbered_choice(
            "Binning for global histograms:",
            [("Automatic (Freedman-Diaconis)", "fd"),
             ("Automatic (numpy auto)", "auto"),
             ("Automatic (Sturges)", "sturges"),
             ("Fixed: choose bin count", "fixed_custom")],
            default_idx=0
        )
        global_bins_spec = resolve_histogram_bins_spec(global_binning_choice)

        analysis_kind = prompt_numbered_choice(
            "Population analysis to compare in global mode:",
            [("Loewdin", "loewdin"),
             ("Mulliken", "mulliken"),
             ("Hirshfeld", "hirshfeld"),
             ("CHELPG (charge) + Hirshfeld (spin)", "chelpg_hirshfeld"),
             ("CHELPG (charge) + Loewdin (spin)", "chelpg_loewdin"),
             ("CHELPG (charge) + Mulliken (spin)", "chelpg_mulliken"),
             ("All", "all")],
            default_idx=0
        )

        base_dir = os.getcwd()
        system_dirs = discover_global_analysis_dirs(base_dir)
        if not system_dirs:
            print("Error: no compatible previous analyses were found in subdirectories of the current directory.")
            sys.exit(1)

        system_names = [os.path.basename(system_dir) for system_dir in system_dirs]
        print("Systems detected for global analysis:")
        for system_name in system_names:
            print("  ", system_name)

        atom_selection_mode = prompt_numbered_choice(
            "Atom selection for global mode:",
            [("Same atoms for all systems", "mismos"),
             ("Specific atoms for each system", "especificos"),
             ("Reuse the entities selected in each previous individual analysis", "previos")],
            default_idx=0
        )

        atom_map = {}
        if atom_selection_mode == "mismos":
            atom_ids_str = input(
                "Enter atoms/entities to compare, separated by spaces (for example: 88 89 coque): "
            ).strip()
            atom_ids = parse_global_entity_list(atom_ids_str)
            if not atom_ids:
                print("No atoms were provided for comparison.")
                sys.exit(1)
            for system_name in system_names:
                atom_map[system_name] = list(atom_ids)
        elif atom_selection_mode == "especificos":
            for system_name in system_names:
                atom_ids_str = input(
                    f"Enter atoms/entities to compare for {system_name}, separated by spaces (Enter = skip system; use 'coque' if available): "
                ).strip()
                if atom_ids_str == "":
                    atom_map[system_name] = []
                    continue
                atom_ids = parse_global_entity_list(atom_ids_str)
                atom_map[system_name] = atom_ids
        else:
            for system_dir in system_dirs:
                system_name = os.path.basename(system_dir)
                inferred_entities = infer_entities_from_previous_analysis(system_dir, analysis_kind)
                atom_map[system_name] = inferred_entities
                if inferred_entities:
                    inferred_str = " ".join(str(entity_id) for entity_id in inferred_entities)
                    print(f"  {system_name}: reusing entities from previous analysis -> {inferred_str}")
                else:
                    print(f"  {system_name}: no previous individual entity selection could be inferred.")

        atom_ids = []
        for aids in atom_map.values():
            for aid in aids:
                if aid not in atom_ids:
                    atom_ids.append(aid)
        if not atom_ids:
            print("No atoms were provided for comparison.")
            sys.exit(1)

        atom_labels = {aid: ("coque" if aid == "coque" else str(aid)) for aid in atom_ids}
        use_labels = prompt_numbered_choice(
            "Do you want to assign custom labels to the atoms?",
            [("No", False), ("Yes", True)],
            default_idx=0
        )
        if use_labels:
                print("Enter a label for each atom; leave blank to use the atom number.")
                for aid in atom_ids:
                    default_label = atom_labels.get(aid, str(aid))
                    label = input(f"Label for entity {aid} (leave blank to use '{default_label}'): ").strip()
                    if label:
                        atom_labels[aid] = label

        global_dir = os.path.join(base_dir, "global")
        os.makedirs(global_dir, exist_ok=True)

        if analysis_kind == "all":
            analysis_kinds_to_run = ["mulliken", "loewdin", "hirshfeld", "chelpg_hirshfeld"]
        else:
            analysis_kinds_to_run = [analysis_kind]

        generated_any = False
        global_summary_entries = []
        for selected_analysis_kind in analysis_kinds_to_run:
            systems_data, missing, spin_labels_found = collect_global_hist_data(base_dir, atom_map, selected_analysis_kind)
            if not systems_data:
                print(
                    f"[WARN] No compatible previous analyses were found for "
                    f"'{selected_analysis_kind}'. This global figure was skipped."
                )
                continue

            all_spin_labels = sorted({label for labels in spin_labels_found.values() for label in labels})
            if selected_analysis_kind in ("chelpg", "chelpg_loewdin"):
                spin_axis_label = "Loewdin spin"
            elif selected_analysis_kind == "chelpg_mulliken":
                spin_axis_label = "Mulliken spin"
            elif selected_analysis_kind == "chelpg_hirshfeld":
                spin_axis_label = "Hirshfeld spin"
            else:
                spin_axis_label = None
            if all_spin_labels:
                if len(all_spin_labels) > 1:
                    print("Error: global analyses with incompatible spin representations were detected:")
                    for system_name in sorted(spin_labels_found):
                        labels = ", ".join(sorted(spin_labels_found[system_name]))
                        print(f"  {system_name}: {labels}")
                    print("Reprocess the directories so that all of them use the same spin representation.")
                    sys.exit(1)
                if all_spin_labels[0] == "spin_fraction":
                    if selected_analysis_kind in ("chelpg", "chelpg_loewdin"):
                        spin_axis_label = "Loewdin spin fraction"
                    elif selected_analysis_kind == "chelpg_mulliken":
                        spin_axis_label = "Mulliken spin fraction"
                    elif selected_analysis_kind == "chelpg_hirshfeld":
                        spin_axis_label = "Hirshfeld spin fraction"
                    else:
                        spin_axis_label = f"{get_analysis_display_label(selected_analysis_kind)} spin fraction"

            print(f"Systems included in the plot for '{selected_analysis_kind}':")
            for system_name in systems_data:
                atoms_for_system = " ".join(str(aid) for aid in systems_data[system_name].keys())
                print(f"  {system_name}: {atoms_for_system}")

            if missing:
                print(f"[WARN] Some time-series files are missing for '{selected_analysis_kind}' and were omitted from the plot:")
                for system_name, aid, fname in missing[:20]:
                    entity_label = "coque" if aid == "coque" else f"atom {aid}"
                    print(f"  {system_name}: {entity_label} -> {os.path.basename(fname)}")
                if len(missing) > 20:
                    print(f"  ... and {len(missing) - 20} more cases.")

            export_global_plot_data(
                global_dir,
                atom_ids,
                systems_data,
                selected_analysis_kind,
                percentile=95.0
            )
            export_global_snapshot_counts(
                global_dir,
                atom_ids,
                systems_data,
                selected_analysis_kind
            )

            fig_outname = os.path.join(global_dir, f"global_{selected_analysis_kind}_histograms.png")
            make_global_overlay_hist_figure(
                atom_ids,
                systems_data,
                atom_labels=atom_labels,
                analysis_label=get_analysis_display_label(selected_analysis_kind),
                spin_axis_label=spin_axis_label,
                fig_outname=fig_outname,
                bins_spec=global_bins_spec,
                percentile=95.0
            )
            global_summary_entries.append(
                build_global_terminal_summary_entry(
                    get_analysis_display_label(selected_analysis_kind),
                    systems_data,
                    atom_ids,
                    atom_labels,
                )
            )
            generated_any = True

        if not generated_any:
            print("Error: no compatible previous analyses were found in subdirectories of the current directory.")
            sys.exit(1)
        print_terminal_global_summary(global_summary_entries)
        return

    # Select program
    prog = prompt_numbered_choice(
        "Program to use:",
        [("LIO", "lio"), ("ORCA", "orca")],
        default_idx=1
    )

    have_spin = False
    spin_sign = 1.0
    orca_mode = (prog == "orca")
    population_config = {"mulliken": True, "loewdin": False, "hirshfeld": False, "chelpg_loewdin": False, "chelpg_mulliken": False, "chelpg_hirshfeld": False}
    primary_analysis_kind = None
    active_charge_header = "# Mulliken Population Analysis"
    active_spin_header = "# Mulliken Spin Population Analysis"
    active_charge_axis_label = "Charge"
    active_spin_axis_label = "Spin"

    if prog == "lio":
        # LIO can consume either already-merged files from `tolkien-tools md
        # merge-pop` or legacy per-segment mq_*.dat/ms_*.dat files.
        charge_full = find_first_existing_file(("mulliken_full.dat", "mq_full.dat"))
        if charge_full:
            print(f"Using merged charge file: {charge_full}")
        else:
            mq_files = get_sorted_files("mq")
            if not mq_files:
                print_lio_merge_pop_hint("cargas")
                print("Error: no se encontro 'mulliken_full.dat' ni 'mq_full.dat'.")
                sys.exit(1)

            print("Charge files (mq_*.dat) to be merged:")
            for f in mq_files:
                print("  ", f)
            merge_files(mq_files, "mq_full.dat")
            charge_full = "mq_full.dat"

        spin_full = find_first_existing_file(("mulliken_spin_full.dat", "ms_full.dat"))
        if spin_full:
            have_spin = True
            print(f"Using merged spin file: {spin_full}")
        else:
            ms_files = get_sorted_files("ms")
            have_spin = bool(ms_files)
            if have_spin:
                print("Spin files (ms_*.dat) to be merged:")
                for f in ms_files:
                    print("  ", f)
                merge_files(ms_files, "ms_full.dat")
                spin_full = "ms_full.dat"
            else:
                spin_full = None
                print_lio_merge_pop_hint("spines")
                print("[INFO] No se encontro 'mulliken_spin_full.dat' ni 'ms_full.dat'. Solo se analizaran cargas.")
        spin_sign = -1.0
    else:
        # ORCA: each <prefix>_N.out/.dat file is a frame
        population_choice = prompt_numbered_choice(
            "Population analysis to process for ORCA:",
            [("Mulliken", "mulliken"),
             ("Loewdin", "loewdin"),
             ("Hirshfeld", "hirshfeld"),
             ("CHELPG (charges) + Loewdin (spins)", "chelpg_loewdin"),
             ("CHELPG (charges) + Mulliken (spins)", "chelpg_mulliken"),
             ("CHELPG (charges) + Hirshfeld (spins)", "chelpg_hirshfeld"),
             ("All", "all")],
            default_idx=6
        )
        population_config = get_population_analysis_config(population_choice)

        orca_prefix = input(
            "Enter the ORCA file prefix before _N (for example, TD or SP; Enter = autodetect): "
        ).strip()

        orca_files = get_sorted_orca_files(orca_prefix)
        if not orca_files:
            if orca_prefix:
                print(f"Error: no files matching '{orca_prefix}_*.out' or '{orca_prefix}_*.dat' were found.")
            else:
                print("Error: no ORCA files matching '<prefix>_N.out' or '<prefix>_N.dat' were found.")
            sys.exit(1)

        if orca_prefix:
            print(f"ORCA files ({orca_prefix}_*.out/.dat) to be analyzed:")
        else:
            print("Autodetected ORCA files (*_N.out/.dat) to be analyzed:")
        for f in orca_files:
            print("  ", f)

        multiplicities = []
        files_without_multiplicity = []
        for fname in orca_files:
            mult = detect_orca_multiplicity(fname)
            if mult is None:
                files_without_multiplicity.append(fname)
            else:
                multiplicities.append(mult)

        if files_without_multiplicity:
            print("[WARN] Spin multiplicity was not found in some ORCA files:")
            for fname in files_without_multiplicity[:10]:
                print(f"  {fname}")
            if len(files_without_multiplicity) > 10:
                print(f"  ... and {len(files_without_multiplicity) - 10} more files.")

        if multiplicities and len(set(multiplicities)) > 1:
            print("Error: inconsistent spin multiplicities were detected across the ORCA files:")
            for mult in sorted(set(multiplicities)):
                count = sum(1 for value in multiplicities if value == mult)
                print(f"  multiplicity {mult}: {count} files")
            sys.exit(1)

        orca_multiplicity = multiplicities[0] if multiplicities else None
        if orca_multiplicity is None:
            print("[WARN] Spin multiplicity could not be determined. The script will assume spin populations are present.")
            have_spin = True
        else:
            have_spin = (orca_multiplicity > 1)
            spin_state_label = "open-shell" if have_spin else "closed-shell"
            print(f"[INFO] ORCA multiplicity detected: {orca_multiplicity} ({spin_state_label}).")
            if not have_spin:
                print("[INFO] Closed-shell system detected (multiplicity = 1). Only charge analyses will be performed.")
                if population_choice in ("chelpg_loewdin", "chelpg_mulliken"):
                    print(
                        "Error: the selected CHELPG + spin analysis requires spin populations, "
                        "but the ORCA outputs correspond to a closed-shell system (multiplicity = 1)."
                    )
                    sys.exit(1)

        mulliken_charge_header_line = (
            "MULLIKEN ATOMIC CHARGES AND SPIN POPULATIONS" if have_spin else "MULLIKEN ATOMIC CHARGES"
        )
        loewdin_charge_header_line = (
            "LOEWDIN ATOMIC CHARGES AND SPIN POPULATIONS" if have_spin else "LOEWDIN ATOMIC CHARGES"
        )

        if population_config["mulliken"]:
            if have_spin:
                build_orca_full_files(
                    orca_files,
                    "mq_orca_todo.dat",
                    "ms_orca_todo.dat",
                    charge_label="Mulliken",
                    charge_header_line=mulliken_charge_header_line
                )
            else:
                build_orca_charge_file(
                    orca_files,
                    "mq_orca_todo.dat",
                    charge_label="Mulliken",
                    charge_header_line=mulliken_charge_header_line
                )
        if population_config["loewdin"]:
            if have_spin:
                build_orca_full_files(
                    orca_files,
                    "lq_orca_todo.dat",
                    "ls_orca_todo.dat",
                    charge_label="Loewdin",
                    charge_header_line=loewdin_charge_header_line
                )
            else:
                build_orca_charge_file(
                    orca_files,
                    "lq_orca_todo.dat",
                    charge_label="Loewdin",
                    charge_header_line=loewdin_charge_header_line
                )
        if population_config["hirshfeld"]:
            if have_spin:
                build_orca_full_files(
                    orca_files,
                    "hq_orca_todo.dat",
                    "hs_orca_todo.dat",
                    charge_label="Hirshfeld",
                    charge_header_line="HIRSHFELD ANALYSIS"
                )
            else:
                build_orca_charge_file(
                    orca_files,
                    "hq_orca_todo.dat",
                    charge_label="Hirshfeld",
                    charge_header_line="HIRSHFELD ANALYSIS"
                )
        if population_config["chelpg_loewdin"]:
            if have_spin:
                build_orca_full_files(
                    orca_files,
                    "cq_loewdin_orca_todo.dat",
                    "cs_loewdin_orca_todo.dat",
                    charge_label="CHELPG",
                    charge_header_line="CHELPG Charges",
                    spin_label="Loewdin",
                    spin_header_line="LOEWDIN ATOMIC CHARGES AND SPIN POPULATIONS"
                )
            else:
                print("[INFO] CHELPG + Loewdin was skipped because the system is closed-shell and has no spin populations.")
        if population_config["chelpg_mulliken"]:
            if have_spin:
                build_orca_full_files(
                    orca_files,
                    "cq_mulliken_orca_todo.dat",
                    "cs_mulliken_orca_todo.dat",
                    charge_label="CHELPG",
                    charge_header_line="CHELPG Charges",
                    spin_label="Mulliken",
                    spin_header_line="MULLIKEN ATOMIC CHARGES AND SPIN POPULATIONS"
                )
            else:
                print("[INFO] CHELPG + Mulliken was skipped because the system is closed-shell and has no spin populations.")
        if population_config["chelpg_hirshfeld"]:
            if have_spin:
                build_orca_full_files(
                    orca_files,
                    "cq_hirshfeld_orca_todo.dat",
                    "cs_hirshfeld_orca_todo.dat",
                    charge_label="CHELPG",
                    charge_header_line="CHELPG Charges",
                    spin_label="Hirshfeld",
                    spin_header_line="HIRSHFELD ANALYSIS"
                )
            else:
                build_orca_charge_file(
                    orca_files,
                    "cq_hirshfeld_orca_todo.dat",
                    charge_label="CHELPG",
                    charge_header_line="CHELPG Charges"
                )

        # The primary analysis controls the spin-consistency check. When several
        # analyses are enabled (for example, "All"), prefer the most robust spin
        # partitioning available: Hirshfeld > Loewdin > Mulliken.
        if population_config["hirshfeld"]:
            primary_analysis_kind = "hirshfeld"
            charge_full = "hq_orca_todo.dat"
            spin_full = "hs_orca_todo.dat" if have_spin else None
            active_charge_header = "# Hirshfeld Population Analysis"
            active_spin_header = "# Hirshfeld Spin Population Analysis"
            active_charge_axis_label = "Hirshfeld charge"
        elif population_config["loewdin"]:
            primary_analysis_kind = "loewdin"
            charge_full = "lq_orca_todo.dat"
            spin_full = "ls_orca_todo.dat" if have_spin else None
            active_charge_header = "# Loewdin Population Analysis"
            active_spin_header = "# Loewdin Spin Population Analysis"
            active_charge_axis_label = "Loewdin charge"
        elif population_config["mulliken"]:
            primary_analysis_kind = "mulliken"
            charge_full = "mq_orca_todo.dat"
            spin_full = "ms_orca_todo.dat" if have_spin else None
            active_charge_header = "# Mulliken Population Analysis"
            active_spin_header = "# Mulliken Spin Population Analysis"
            active_charge_axis_label = "Mulliken charge"
        elif population_config["chelpg_loewdin"]:
            primary_analysis_kind = "chelpg_loewdin"
            charge_full = "cq_loewdin_orca_todo.dat"
            spin_full = "cs_loewdin_orca_todo.dat" if have_spin else None
            active_charge_header = "# CHELPG Population Analysis"
            active_spin_header = "# Loewdin Spin Population Analysis"
            active_charge_axis_label = "CHELPG charge"
        elif population_config["chelpg_mulliken"]:
            primary_analysis_kind = "chelpg_mulliken"
            charge_full = "cq_mulliken_orca_todo.dat"
            spin_full = "cs_mulliken_orca_todo.dat" if have_spin else None
            active_charge_header = "# CHELPG Population Analysis"
            active_spin_header = "# Mulliken Spin Population Analysis"
            active_charge_axis_label = "CHELPG charge"
        else:
            primary_analysis_kind = "chelpg_hirshfeld"
            charge_full = "cq_hirshfeld_orca_todo.dat"
            spin_full = "cs_hirshfeld_orca_todo.dat" if have_spin else None
            active_charge_header = "# CHELPG Population Analysis"
            active_spin_header = "# Hirshfeld Spin Population Analysis"
            active_charge_axis_label = "CHELPG charge"
        spin_sign = 1.0

    # Show available atoms
    atoms_list = get_atom_list_from_full(charge_full, active_charge_header, lio=(prog == "lio"))
    if atoms_list:
        print("\nAvailable atoms (id, type):")
        for aid, atype in atoms_list:
            print(f"  {aid:4d}  {atype}")
        print("")

    atom_type_map = {aid: atype for aid, atype in atoms_list} if atoms_list else {}
    available_atom_ids = {aid for aid, _atype in atoms_list} if atoms_list else set()
    actor_config = None
    spin_consistency_atom_ids = None

    if have_spin:
        atom_selection_mode = prompt_numbered_choice(
            "Atom selection mode:",
            [("Manual atom selection", "manual"),
             ("Automatic spin-localization selection with a grouped 'resto'", "auto_spin")],
            default_idx=0
        )
    else:
        atom_selection_mode = "manual"

    if atom_selection_mode == "auto_spin":
        coverage_value = prompt_float_value(
            "Average spin coverage to explain (Enter = 90%): ",
            90.0,
            min_value=0.0,
        )
        min_atom_value = prompt_float_value(
            "Minimum average spin percentage for an individual histogram (Enter = 5%): ",
            5.0,
            min_value=0.0,
        )
        coverage_fraction = normalize_percent_input(coverage_value)
        min_atom_fraction = normalize_percent_input(min_atom_value)
        if coverage_fraction <= 0.0 or coverage_fraction > 1.0:
            print("Error: spin coverage must be in the interval (0, 100].")
            sys.exit(1)
        if min_atom_fraction < 0.0 or min_atom_fraction > 1.0:
            print("Error: the minimum atom percentage must be in the interval [0, 100].")
            sys.exit(1)

        (
            atom_ids,
            actor_config,
            all_spin_atom_ids,
            atom_type_map,
            avg_spin_fraction,
            coverage_atoms,
        ) = select_spin_localization_atoms(
            spin_full,
            active_spin_header,
            spin_sign,
            coverage_fraction=coverage_fraction,
            min_atom_fraction=min_atom_fraction,
            lio=(prog == "lio"),
        )
        available_atom_ids = set(all_spin_atom_ids)
        spin_consistency_atom_ids = list(all_spin_atom_ids)

        avg_fraction_by_atom = {
            aid: float(avg_spin_fraction[idx])
            for idx, aid in enumerate(all_spin_atom_ids)
        }
        viewer_choice = prompt_numbered_choice(
            "Generate a spin-localization 3D HTML viewer?",
            [("Yes", True), ("No", False)],
            default_idx=0
        )
        if viewer_choice:
            try:
                if orca_mode:
                    viewer_orca_file = find_orca_geometry_file_for_viewer(orca_files)
                    if viewer_orca_file is None:
                        raise ValueError("none of the ORCA files has a CARTESIAN COORDINATES (ANGSTROEM) block")
                    print(
                        f"[INFO] Using one representative ORCA geometry from "
                        f"'{viewer_orca_file}' for the spin-localization viewer."
                    )
                    write_orca_spin_localization_viewer(
                        viewer_orca_file,
                        "spin_localization_viewer.html",
                        coverage_atoms,
                        atom_type_map,
                        avg_fraction_by_atom,
                    )
                else:
                    xyz_path = find_default_xyz_for_spin_viewer()
                    if xyz_path is None:
                        xyz_path = input(
                            "XYZ file for the viewer was not autodetected. Enter path (Enter = skip): "
                        ).strip()
                    if not xyz_path:
                        raise ValueError("no XYZ file selected")
                    if not os.path.isfile(xyz_path):
                        raise ValueError(f"XYZ file '{xyz_path}' was not found")
                    write_spin_localization_viewer(
                        xyz_path,
                        "spin_localization_viewer.html",
                        coverage_atoms,
                        atom_type_map,
                        avg_fraction_by_atom,
                    )
                open_choice = prompt_numbered_choice(
                    "Open the spin-localization viewer in the default browser?",
                    [("Yes", True), ("No", False)],
                    default_idx=0
                )
                if open_choice:
                    open_html_viewer("spin_localization_viewer.html")
            except Exception as exc:
                print(f"[WARN] Spin-localization viewer was skipped: {exc}")
    else:
        atoms_str = input("Enter atom numbers to track, separated by spaces (for example: 45 46 47): ").strip()
        try:
            atom_ids = [int(x) for x in atoms_str.split()]
        except ValueError:
            print("Error: atom numbers must be integers separated by spaces.")
            sys.exit(1)

        if not atom_ids:
            print("No atoms were provided for tracking.")
            sys.exit(1)
        spin_consistency_atom_ids = list(atom_ids)

    # Custom labels for atoms (optional)
    atom_labels = {aid: str(aid) for aid in atom_ids}
    if actor_config is not None:
        atom_labels[actor_config["id"]] = actor_config["label"]
    use_labels = prompt_numbered_choice(
        "Do you want to assign custom labels to the atoms?",
        [("No", False), ("Yes", True)],
        default_idx=0
    )
    if use_labels:
        print("Enter a label for each atom; leave blank to use the default atom number.")
        for aid in atom_ids:
            label = input(f"Label for atom {aid} (for example, 'Fe'; leave blank to use '{aid}'): ").strip()
            if label:
                atom_labels[aid] = label

    make_time_plots = prompt_numbered_choice(
        "Do you want to generate charge/spin time-series plots?",
        [("No", False), ("Yes", True)],
        default_idx=0
    )
    if make_time_plots:
        dt_str = input("Enter the time step in picoseconds (for example, 0.001): ").strip()
        try:
            dt_ps = float(dt_str)
        except ValueError:
            print("Error: the time step must be a number (float).")
            sys.exit(1)
    else:
        # Keep downstream analysis working without prompting for dt when no time plots are requested.
        dt_ps = 1.0

    hist_binning_choice = prompt_numbered_choice(
        "Binning for single-system histograms:",
        [("Automatic (Freedman-Diaconis)", "fd"),
         ("Automatic (numpy auto)", "auto"),
         ("Automatic (Sturges)", "sturges"),
         ("Fixed: choose bin count", "fixed_custom")],
        default_idx=0
    )
    hist_bins_spec = resolve_histogram_bins_spec(hist_binning_choice)

    default_spin_mode_idx = 1 if atom_selection_mode == "auto_spin" else 0
    spin_mode = prompt_numbered_choice(
        "Spin representation for spin statistics, histograms and output files:",
        [("Raw spin values from each snapshot", "raw"),
         ("Spin fraction relative to the selected atoms", "fraction")],
        default_idx=default_spin_mode_idx
    )
    spin_config = get_spin_representation_config(spin_mode)
    if orca_mode and primary_analysis_kind in ("loewdin", "chelpg_loewdin"):
        active_spin_axis_label = "Loewdin spin fraction" if spin_config["normalize"] else "Loewdin spin"
    elif orca_mode and primary_analysis_kind == "hirshfeld":
        active_spin_axis_label = "Hirshfeld spin fraction" if spin_config["normalize"] else "Hirshfeld spin"
    elif orca_mode and primary_analysis_kind == "chelpg_mulliken":
        active_spin_axis_label = "Mulliken spin fraction" if spin_config["normalize"] else "Mulliken spin"
    elif orca_mode and primary_analysis_kind == "chelpg_hirshfeld":
        active_spin_axis_label = "Hirshfeld spin fraction" if spin_config["normalize"] else "Hirshfeld spin"
    else:
        active_spin_axis_label = spin_config["axis_label"]

    spin_keep_mask = None
    terminal_summary_entries = []
    if have_spin:
        run_spin_check = prompt_numbered_choice(
            "Spin consistency check per snapshot:",
            [("Skip the check", False),
             ("Compare the sum over selected atoms with the total system spin", True)],
            default_idx=1
        )
        if run_spin_check:
            tol_str = input(
                "Enter the allowed tolerance for |selected_spin_sum - total_spin| (Enter = 0.20): "
            ).strip()
            if tol_str == "":
                spin_tolerance = 0.20
            else:
                try:
                    spin_tolerance = float(tol_str)
                except ValueError:
                    print("Error: the tolerance must be a number.")
                    sys.exit(1)

            spin_times_check, spin_values_check, spin_totals_check = parse_frame_data(
                spin_full,
                dt_ps,
                spin_consistency_atom_ids,
                kind="spin",
                header_start=active_spin_header,
                spin_sign=spin_sign
            )
            spin_keep_mask = analyze_spin_consistency(
                spin_times_check,
                spin_values_check,
                spin_totals_check,
                tolerance=spin_tolerance,
                report_outname="spin_consistency_report.dat"
            )

            if (~spin_keep_mask).any():
                suggestions, auto_added_atom_ids = suggest_missing_spin_atoms(
                    spin_full,
                    dt_ps,
                    spin_consistency_atom_ids,
                    bad_mask=(~spin_keep_mask),
                    header_start=active_spin_header,
                    spin_sign=spin_sign,
                    report_outname="spin_missing_atom_suggestions.dat"
                )
                resolution_mode = prompt_numbered_choice(
                    "Snapshots with poor spin consistency were detected:",
                    [("Keep all snapshots", "keep_all"),
                     ("Remove snapshots outside the tolerance from the analysis", "filter_bad"),
                     ("Add atoms to the analysis", "add_atoms")],
                    default_idx=0
                )
                if resolution_mode == "keep_all":
                    spin_keep_mask = None
                elif resolution_mode == "filter_bad":
                    if spin_keep_mask.sum() == 0:
                        print("Error: all snapshots were excluded by the tolerance filter.")
                        sys.exit(1)
                    else:
                        n_kept = int(spin_keep_mask.sum())
                        n_removed = int((~spin_keep_mask).sum())
                        print(
                            f"[INFO] Spin-consistency filtering applied: "
                            f"{n_removed} snapshots removed, {n_kept} snapshots kept."
                        )
                else:
                    add_mode_options = [("Add specific atoms manually", "manual")]
                    if auto_added_atom_ids:
                        add_mode_options.append(("Add the automatically suggested atoms individually", "auto_atoms"))
                        add_mode_options.append(("Add the proposed atoms as one grouped actor", "auto_actor"))
                    add_mode_options.append(("Cancel and keep the current atom list", "cancel"))

                    add_mode = prompt_numbered_choice(
                        "How should the analysis be expanded?",
                        add_mode_options,
                        default_idx=0
                    )

                    if add_mode == "manual":
                        new_atom_ids = prompt_additional_atom_selection(
                            available_atom_ids,
                            excluded_atom_ids=atom_ids
                        )
                        atom_ids = list(atom_ids) + new_atom_ids
                        for aid in new_atom_ids:
                            atom_labels.setdefault(aid, str(aid))
                        spin_keep_mask = None
                        print(
                            "[INFO] Added atoms to the analysis: "
                            + ", ".join(
                                f"{aid}({atom_type_map.get(aid, '?')})" for aid in new_atom_ids
                            )
                        )
                    elif add_mode == "auto_atoms":
                        atom_ids = list(atom_ids) + [aid for aid in auto_added_atom_ids if aid not in atom_ids]
                        for aid in auto_added_atom_ids:
                            atom_labels.setdefault(aid, str(aid))
                        spin_keep_mask = None
                        print(
                            "[INFO] Automatically suggested atoms added to the analysis: "
                            + ", ".join(
                                f"{aid}({atom_type_map.get(aid, '?')})" for aid in auto_added_atom_ids
                            )
                        )
                    elif add_mode == "auto_actor":
                        actor_members = [aid for aid in auto_added_atom_ids if aid not in atom_ids]
                        if not actor_members:
                            print("Error: there are no suggested atoms left to build the grouped actor.")
                            sys.exit(1)
                        actor_label = "proposed_spin_pool"
                        actor_config = {
                            "id": f"actor_{sanitize_output_token(actor_label)}",
                            "label": actor_label,
                            "atom_ids": actor_members,
                        }
                        spin_keep_mask = None
                        print(
                            "[INFO] Grouped actor added to the analysis using atoms: "
                            + ", ".join(
                                f"{aid}({atom_type_map.get(aid, '?')})" for aid in actor_members
                            )
                        )
                    else:
                        spin_keep_mask = None
                        print("[INFO] The atom list was left unchanged; all snapshots will be kept.")
                if resolution_mode == "filter_bad" and spin_keep_mask.sum() == 0:
                    print("Error: all snapshots were excluded by the tolerance filter.")
                    sys.exit(1)

    analysis_entity_ids = get_analysis_entity_ids(atom_ids, actor_config=actor_config)
    if actor_config is not None:
        atom_labels[actor_config["id"]] = actor_config["label"]
    if actor_config is None:
        output_variant_suffix = None
    elif actor_config["label"] == "proposed_spin_pool":
        output_variant_suffix = "with_coque"
    else:
        output_variant_suffix = f"with_{sanitize_output_token(actor_config['label'])}"

    if orca_mode and primary_analysis_kind == "mulliken":
        q_ts_out = "mulliken_charge_timeseries.dat"
        q_avg_out = "mulliken_charge_averages.dat"
        q_hist_prefix = "mulliken_charge_hist"
        q_modes_out = "mulliken_charge_modes.dat"
        s_ts_out = "mulliken_spin_timeseries.dat"
        s_avg_out = "mulliken_spin_averages.dat"
        s_hist_prefix = "mulliken_spin_hist"
        s_modes_out = "mulliken_spin_modes.dat"
        mulliken_fig_out = "mulliken_histograms.png"
    elif orca_mode and primary_analysis_kind == "loewdin":
        q_ts_out = "loewdin_charge_timeseries.dat"
        q_avg_out = "loewdin_charge_averages.dat"
        q_hist_prefix = "loewdin_charge_hist"
        q_modes_out = "loewdin_charge_modes.dat"
        s_ts_out = "loewdin_spin_timeseries.dat"
        s_avg_out = "loewdin_spin_averages.dat"
        s_hist_prefix = "loewdin_spin_hist"
        s_modes_out = "loewdin_spin_modes.dat"
        mulliken_fig_out = "loewdin_histograms.png"
    elif orca_mode and primary_analysis_kind == "hirshfeld":
        q_ts_out = "hirshfeld_charge_timeseries.dat"
        q_avg_out = "hirshfeld_charge_averages.dat"
        q_hist_prefix = "hirshfeld_charge_hist"
        q_modes_out = "hirshfeld_charge_modes.dat"
        s_ts_out = "hirshfeld_spin_timeseries.dat"
        s_avg_out = "hirshfeld_spin_averages.dat"
        s_hist_prefix = "hirshfeld_spin_hist"
        s_modes_out = "hirshfeld_spin_modes.dat"
        mulliken_fig_out = "hirshfeld_histograms.png"
    elif orca_mode and primary_analysis_kind == "chelpg_loewdin":
        q_ts_out = "chelpg_loewdin_charge_timeseries.dat"
        q_avg_out = "chelpg_loewdin_charge_averages.dat"
        q_hist_prefix = "chelpg_loewdin_charge_hist"
        q_modes_out = "chelpg_loewdin_charge_modes.dat"
        s_ts_out = "loewdin_spin_timeseries_for_chelpg_loewdin.dat"
        s_avg_out = "loewdin_spin_averages_for_chelpg_loewdin.dat"
        s_hist_prefix = "loewdin_spin_hist_for_chelpg_loewdin"
        s_modes_out = "loewdin_spin_modes_for_chelpg_loewdin.dat"
        mulliken_fig_out = "chelpg_loewdin_histograms.png"
    elif orca_mode and primary_analysis_kind == "chelpg_mulliken":
        q_ts_out = "chelpg_mulliken_charge_timeseries.dat"
        q_avg_out = "chelpg_mulliken_charge_averages.dat"
        q_hist_prefix = "chelpg_mulliken_charge_hist"
        q_modes_out = "chelpg_mulliken_charge_modes.dat"
        s_ts_out = "mulliken_spin_timeseries_for_chelpg_mulliken.dat"
        s_avg_out = "mulliken_spin_averages_for_chelpg_mulliken.dat"
        s_hist_prefix = "mulliken_spin_hist_for_chelpg_mulliken"
        s_modes_out = "mulliken_spin_modes_for_chelpg_mulliken.dat"
        mulliken_fig_out = "chelpg_mulliken_histograms.png"
    elif orca_mode and primary_analysis_kind == "chelpg_hirshfeld":
        q_ts_out = "chelpg_hirshfeld_charge_timeseries.dat"
        q_avg_out = "chelpg_hirshfeld_charge_averages.dat"
        q_hist_prefix = "chelpg_hirshfeld_charge_hist"
        q_modes_out = "chelpg_hirshfeld_charge_modes.dat"
        s_ts_out = "hirshfeld_spin_timeseries_for_chelpg_hirshfeld.dat"
        s_avg_out = "hirshfeld_spin_averages_for_chelpg_hirshfeld.dat"
        s_hist_prefix = "hirshfeld_spin_hist_for_chelpg_hirshfeld"
        s_modes_out = "hirshfeld_spin_modes_for_chelpg_hirshfeld.dat"
        mulliken_fig_out = "chelpg_hirshfeld_histograms.png"
    else:
        q_ts_out = "mq_charge_timeseries.dat"
        q_avg_out = "mq_charge_averages.dat"
        q_hist_prefix = "mq_charge_hist"
        q_modes_out = "mq_charge_modes.dat"
        s_ts_out = "ms_spin_timeseries.dat"
        s_avg_out = "ms_spin_averages.dat"
        s_hist_prefix = "ms_spin_hist"
        s_modes_out = "ms_spin_modes.dat"
        mulliken_fig_out = "qs_histograms.png"

    if output_variant_suffix is not None:
        q_ts_out = append_suffix_to_path(q_ts_out, output_variant_suffix)
        q_avg_out = append_suffix_to_path(q_avg_out, output_variant_suffix)
        q_modes_out = append_suffix_to_path(q_modes_out, output_variant_suffix)
        s_ts_out = append_suffix_to_path(s_ts_out, output_variant_suffix)
        s_avg_out = append_suffix_to_path(s_avg_out, output_variant_suffix)
        s_modes_out = append_suffix_to_path(s_modes_out, output_variant_suffix)
        mulliken_fig_out = append_suffix_to_path(mulliken_fig_out, output_variant_suffix)

    spin_column_label = spin_config["column_label"]
    spin_axis_label = spin_config["axis_label"]
    primary_spin_ylabel = active_spin_axis_label if orca_mode else spin_axis_label

    if not orca_mode or primary_analysis_kind is not None:
        mulliken_charge_mask = apply_keep_mask_or_warn(
            spin_keep_mask,
            parse_frame_data(charge_full, dt_ps, atom_ids, "charge", active_charge_header)[0].size,
            active_charge_axis_label if orca_mode else "charge"
        )
        # --- Charge analysis ---
        times, per_atom_q, hist_q = build_analysis_timeseries_and_stats(
            charge_full,
            dt_ps,
            atom_ids,
            atom_labels,
            actor_config,
            kind="charge",
            header_start=active_charge_header,
            ts_outname=q_ts_out,
            avg_outname=q_avg_out,
            hist_prefix=q_hist_prefix,
            modes_outname=q_modes_out,
            nbins_hist=hist_bins_spec,
            keep_mask=mulliken_charge_mask
        )

        # --- Spin analysis, if available ---
        hist_s = {}
        per_atom_s = {entity_id: np.array([]) for entity_id in analysis_entity_ids}
        if have_spin:
            mulliken_spin_mask = apply_keep_mask_or_warn(
                spin_keep_mask,
                parse_frame_data(spin_full, dt_ps, atom_ids, "spin", active_spin_header, spin_sign=spin_sign)[0].size,
                "Mulliken spin" if orca_mode else "spin"
            )
            _times_spin, per_atom_s, hist_s = build_analysis_timeseries_and_stats(
                spin_full,
                dt_ps,
                atom_ids,
                atom_labels,
                actor_config,
                kind="spin",
                header_start=active_spin_header,
                ts_outname=s_ts_out,
                avg_outname=s_avg_out,
                hist_prefix=s_hist_prefix,
                modes_outname=s_modes_out,
                nbins_hist=hist_bins_spec,
                spin_sign=spin_sign,
                keep_mask=mulliken_spin_mask,
                normalize_spin_fraction=spin_config["normalize"]
            )
        else:
            print("[INFO] No spin analysis was performed because no ms_*.dat files were found.")

        if orca_mode and primary_analysis_kind == "loewdin":
            combined_suffix = "loewdin"
        elif orca_mode and primary_analysis_kind == "hirshfeld":
            combined_suffix = "hirshfeld"
        elif orca_mode and primary_analysis_kind == "chelpg_loewdin":
            combined_suffix = "chelpg_loewdin"
        elif orca_mode and primary_analysis_kind == "chelpg_mulliken":
            combined_suffix = "chelpg_mulliken"
        elif orca_mode and primary_analysis_kind == "chelpg_hirshfeld":
            combined_suffix = "chelpg_hirshfeld"
        else:
            combined_suffix = "qs"

        write_combined_entity_timeseries(
            analysis_entity_ids,
            times,
            per_atom_q,
            per_atom_s if have_spin else {},
            spin_column_label,
            combined_suffix,
            actor_config=actor_config
        )

        make_combined_hist_figure(
            analysis_entity_ids,
            hist_charge=hist_q,
            hist_spin=hist_s if have_spin else {},
            atom_labels=atom_labels,
            charge_axis_label=active_charge_axis_label,
            spin_axis_label=primary_spin_ylabel,
            fig_outname=mulliken_fig_out
        )

        if make_time_plots:
            if orca_mode and primary_analysis_kind == "mulliken":
                mulliken_ts_fig_out = "mulliken_timeseries.png"
            elif orca_mode and primary_analysis_kind == "loewdin":
                mulliken_ts_fig_out = "loewdin_timeseries.png"
            elif orca_mode and primary_analysis_kind == "hirshfeld":
                mulliken_ts_fig_out = "hirshfeld_timeseries.png"
            elif orca_mode and primary_analysis_kind == "chelpg_loewdin":
                mulliken_ts_fig_out = "chelpg_loewdin_timeseries.png"
            elif orca_mode and primary_analysis_kind == "chelpg_mulliken":
                mulliken_ts_fig_out = "chelpg_mulliken_timeseries.png"
            elif orca_mode and primary_analysis_kind == "chelpg_hirshfeld":
                mulliken_ts_fig_out = "chelpg_hirshfeld_timeseries.png"
            else:
                mulliken_ts_fig_out = "qs_timeseries.png"
            if output_variant_suffix is not None:
                mulliken_ts_fig_out = append_suffix_to_path(mulliken_ts_fig_out, output_variant_suffix)
            make_timeseries_figure(
                times,
                per_atom_q,
                per_atom_s if have_spin else {},
                analysis_entity_ids,
                atom_labels=atom_labels,
                fig_outname=mulliken_ts_fig_out,
                spin_ylabel=primary_spin_ylabel
            )

        terminal_summary_entries.append(
            build_terminal_summary_entry(
                get_analysis_display_label(primary_analysis_kind if orca_mode else "mulliken"),
                analysis_entity_ids,
                atom_labels,
                per_atom_q,
                per_atom_s if have_spin else {},
                hist_q,
                hist_s if have_spin else {},
            )
        )

        if orca_mode and primary_analysis_kind == "mulliken" and mulliken_fig_out != "qs_histograms.png":
            make_combined_hist_figure(
                analysis_entity_ids,
                hist_charge=hist_q,
                hist_spin=hist_s if have_spin else {},
                atom_labels=atom_labels,
                charge_axis_label=active_charge_axis_label,
                spin_axis_label=primary_spin_ylabel,
                fig_outname=append_suffix_to_path("qs_histograms.png", output_variant_suffix) if output_variant_suffix is not None else "qs_histograms.png"
            )
            if make_time_plots and mulliken_ts_fig_out != "qs_timeseries.png":
                make_timeseries_figure(
                    times,
                    per_atom_q,
                    per_atom_s if have_spin else {},
                    analysis_entity_ids,
                    atom_labels=atom_labels,
                    fig_outname=append_suffix_to_path("qs_timeseries.png", output_variant_suffix) if output_variant_suffix is not None else "qs_timeseries.png",
                    spin_ylabel=primary_spin_ylabel
                )

    # --- ORCA: additional Loewdin analysis ---
    if orca_mode and population_config["loewdin"] and primary_analysis_kind != "loewdin":
        loewdin_charge_mask = apply_keep_mask_or_warn(
            spin_keep_mask,
            parse_frame_data("lq_orca_todo.dat", dt_ps, atom_ids, "charge", "# Loewdin Population Analysis")[0].size,
            "Loewdin charge"
        )
        times_l, per_atom_q_l, hist_q_l = build_analysis_timeseries_and_stats(
            "lq_orca_todo.dat",
            dt_ps,
            atom_ids,
            atom_labels,
            actor_config,
            kind="charge",
            header_start="# Loewdin Population Analysis",
            ts_outname=append_suffix_to_path("loewdin_charge_timeseries.dat", output_variant_suffix) if output_variant_suffix is not None else "loewdin_charge_timeseries.dat",
            avg_outname=append_suffix_to_path("loewdin_charge_averages.dat", output_variant_suffix) if output_variant_suffix is not None else "loewdin_charge_averages.dat",
            hist_prefix="loewdin_charge_hist",
            modes_outname=append_suffix_to_path("loewdin_charge_modes.dat", output_variant_suffix) if output_variant_suffix is not None else "loewdin_charge_modes.dat",
            nbins_hist=hist_bins_spec,
            keep_mask=loewdin_charge_mask
        )

        hist_s_l = {}
        per_atom_s_l = {entity_id: np.array([]) for entity_id in analysis_entity_ids}
        loewdin_spin_mask = apply_keep_mask_or_warn(
            spin_keep_mask,
            parse_frame_data("ls_orca_todo.dat", dt_ps, atom_ids, "spin", "# Loewdin Spin Population Analysis", spin_sign=spin_sign)[0].size,
            "Loewdin spin"
        )
        _times_spin_l, per_atom_s_l, hist_s_l = build_analysis_timeseries_and_stats(
            "ls_orca_todo.dat",
            dt_ps,
            atom_ids,
            atom_labels,
            actor_config,
            kind="spin",
            header_start="# Loewdin Spin Population Analysis",
            ts_outname=append_suffix_to_path("loewdin_spin_timeseries.dat", output_variant_suffix) if output_variant_suffix is not None else "loewdin_spin_timeseries.dat",
            avg_outname=append_suffix_to_path("loewdin_spin_averages.dat", output_variant_suffix) if output_variant_suffix is not None else "loewdin_spin_averages.dat",
            hist_prefix="loewdin_spin_hist",
            modes_outname=append_suffix_to_path("loewdin_spin_modes.dat", output_variant_suffix) if output_variant_suffix is not None else "loewdin_spin_modes.dat",
            nbins_hist=hist_bins_spec,
            spin_sign=spin_sign,
            keep_mask=loewdin_spin_mask,
            normalize_spin_fraction=spin_config["normalize"]
        )

        write_combined_entity_timeseries(
            analysis_entity_ids,
            times_l,
            per_atom_q_l,
            per_atom_s_l,
            spin_column_label,
            "loewdin",
            actor_config=actor_config
        )

        make_combined_hist_figure(
            analysis_entity_ids,
            hist_charge=hist_q_l,
            hist_spin=hist_s_l,
            atom_labels=atom_labels,
            charge_axis_label="Loewdin charge",
            spin_axis_label="Loewdin spin fraction" if spin_config["normalize"] else "Loewdin spin",
            fig_outname=append_suffix_to_path("loewdin_histograms.png", output_variant_suffix) if output_variant_suffix is not None else "loewdin_histograms.png"
        )

        if make_time_plots:
            make_timeseries_figure(
                times_l,
                per_atom_q_l,
                per_atom_s_l,
                analysis_entity_ids,
                atom_labels=atom_labels,
                fig_outname=append_suffix_to_path("loewdin_timeseries.png", output_variant_suffix) if output_variant_suffix is not None else "loewdin_timeseries.png",
                spin_ylabel="Loewdin spin fraction" if spin_config["normalize"] else "Loewdin spin"
            )

        terminal_summary_entries.append(
            build_terminal_summary_entry(
                "Loewdin",
                analysis_entity_ids,
                atom_labels,
                per_atom_q_l,
                per_atom_s_l,
                hist_q_l,
                hist_s_l,
            )
        )

    # --- ORCA: additional Hirshfeld analysis ---
    if orca_mode and population_config["hirshfeld"] and primary_analysis_kind != "hirshfeld":
        hirshfeld_charge_mask = apply_keep_mask_or_warn(
            spin_keep_mask,
            parse_frame_data("hq_orca_todo.dat", dt_ps, atom_ids, "charge", "# Hirshfeld Population Analysis")[0].size,
            "Hirshfeld charge"
        )
        times_h, per_atom_q_h, hist_q_h = build_analysis_timeseries_and_stats(
            "hq_orca_todo.dat",
            dt_ps,
            atom_ids,
            atom_labels,
            actor_config,
            kind="charge",
            header_start="# Hirshfeld Population Analysis",
            ts_outname=append_suffix_to_path("hirshfeld_charge_timeseries.dat", output_variant_suffix) if output_variant_suffix is not None else "hirshfeld_charge_timeseries.dat",
            avg_outname=append_suffix_to_path("hirshfeld_charge_averages.dat", output_variant_suffix) if output_variant_suffix is not None else "hirshfeld_charge_averages.dat",
            hist_prefix="hirshfeld_charge_hist",
            modes_outname=append_suffix_to_path("hirshfeld_charge_modes.dat", output_variant_suffix) if output_variant_suffix is not None else "hirshfeld_charge_modes.dat",
            nbins_hist=hist_bins_spec,
            keep_mask=hirshfeld_charge_mask
        )

        hist_s_h = {}
        per_atom_s_h = {entity_id: np.array([]) for entity_id in analysis_entity_ids}
        hirshfeld_spin_mask = apply_keep_mask_or_warn(
            spin_keep_mask,
            parse_frame_data("hs_orca_todo.dat", dt_ps, atom_ids, "spin", "# Hirshfeld Spin Population Analysis", spin_sign=spin_sign)[0].size,
            "Hirshfeld spin"
        )
        _times_spin_h, per_atom_s_h, hist_s_h = build_analysis_timeseries_and_stats(
            "hs_orca_todo.dat",
            dt_ps,
            atom_ids,
            atom_labels,
            actor_config,
            kind="spin",
            header_start="# Hirshfeld Spin Population Analysis",
            ts_outname=append_suffix_to_path("hirshfeld_spin_timeseries.dat", output_variant_suffix) if output_variant_suffix is not None else "hirshfeld_spin_timeseries.dat",
            avg_outname=append_suffix_to_path("hirshfeld_spin_averages.dat", output_variant_suffix) if output_variant_suffix is not None else "hirshfeld_spin_averages.dat",
            hist_prefix="hirshfeld_spin_hist",
            modes_outname=append_suffix_to_path("hirshfeld_spin_modes.dat", output_variant_suffix) if output_variant_suffix is not None else "hirshfeld_spin_modes.dat",
            nbins_hist=hist_bins_spec,
            spin_sign=spin_sign,
            keep_mask=hirshfeld_spin_mask,
            normalize_spin_fraction=spin_config["normalize"]
        )

        write_combined_entity_timeseries(
            analysis_entity_ids,
            times_h,
            per_atom_q_h,
            per_atom_s_h,
            spin_column_label,
            "hirshfeld",
            actor_config=actor_config
        )

        make_combined_hist_figure(
            analysis_entity_ids,
            hist_charge=hist_q_h,
            hist_spin=hist_s_h,
            atom_labels=atom_labels,
            charge_axis_label="Hirshfeld charge",
            spin_axis_label="Hirshfeld spin fraction" if spin_config["normalize"] else "Hirshfeld spin",
            fig_outname=append_suffix_to_path("hirshfeld_histograms.png", output_variant_suffix) if output_variant_suffix is not None else "hirshfeld_histograms.png"
        )

        if make_time_plots:
            make_timeseries_figure(
                times_h,
                per_atom_q_h,
                per_atom_s_h,
                analysis_entity_ids,
                atom_labels=atom_labels,
                fig_outname=append_suffix_to_path("hirshfeld_timeseries.png", output_variant_suffix) if output_variant_suffix is not None else "hirshfeld_timeseries.png",
                spin_ylabel="Hirshfeld spin fraction" if spin_config["normalize"] else "Hirshfeld spin"
            )

        terminal_summary_entries.append(
            build_terminal_summary_entry(
                "Hirshfeld",
                analysis_entity_ids,
                atom_labels,
                per_atom_q_h,
                per_atom_s_h,
                hist_q_h,
                hist_s_h,
            )
        )

    # --- ORCA: additional CHELPG + Loewdin analysis ---
    if orca_mode and population_config["chelpg_loewdin"] and primary_analysis_kind != "chelpg_loewdin":
        chelpg_charge_mask = apply_keep_mask_or_warn(
            spin_keep_mask,
            parse_frame_data("cq_loewdin_orca_todo.dat", dt_ps, atom_ids, "charge", "# CHELPG Population Analysis")[0].size,
            "CHELPG charge"
        )
        times_c, per_atom_q_c, hist_q_c = build_analysis_timeseries_and_stats(
            "cq_loewdin_orca_todo.dat",
            dt_ps,
            atom_ids,
            atom_labels,
            actor_config,
            kind="charge",
            header_start="# CHELPG Population Analysis",
            ts_outname=append_suffix_to_path("chelpg_loewdin_charge_timeseries.dat", output_variant_suffix) if output_variant_suffix is not None else "chelpg_loewdin_charge_timeseries.dat",
            avg_outname=append_suffix_to_path("chelpg_loewdin_charge_averages.dat", output_variant_suffix) if output_variant_suffix is not None else "chelpg_loewdin_charge_averages.dat",
            hist_prefix="chelpg_loewdin_charge_hist",
            modes_outname=append_suffix_to_path("chelpg_loewdin_charge_modes.dat", output_variant_suffix) if output_variant_suffix is not None else "chelpg_loewdin_charge_modes.dat",
            nbins_hist=hist_bins_spec,
            keep_mask=chelpg_charge_mask
        )

        hist_s_c = {}
        per_atom_s_c = {entity_id: np.array([]) for entity_id in analysis_entity_ids}
        chelpg_spin_mask = apply_keep_mask_or_warn(
            spin_keep_mask,
            parse_frame_data("cs_loewdin_orca_todo.dat", dt_ps, atom_ids, "spin", "# Loewdin Spin Population Analysis", spin_sign=spin_sign)[0].size,
            "Loewdin spin"
        )
        _times_spin_c, per_atom_s_c, hist_s_c = build_analysis_timeseries_and_stats(
            "cs_loewdin_orca_todo.dat",
            dt_ps,
            atom_ids,
            atom_labels,
            actor_config,
            kind="spin",
            header_start="# Loewdin Spin Population Analysis",
            ts_outname=append_suffix_to_path("loewdin_spin_timeseries_for_chelpg_loewdin.dat", output_variant_suffix) if output_variant_suffix is not None else "loewdin_spin_timeseries_for_chelpg_loewdin.dat",
            avg_outname=append_suffix_to_path("loewdin_spin_averages_for_chelpg_loewdin.dat", output_variant_suffix) if output_variant_suffix is not None else "loewdin_spin_averages_for_chelpg_loewdin.dat",
            hist_prefix="loewdin_spin_hist_for_chelpg_loewdin",
            modes_outname=append_suffix_to_path("loewdin_spin_modes_for_chelpg_loewdin.dat", output_variant_suffix) if output_variant_suffix is not None else "loewdin_spin_modes_for_chelpg_loewdin.dat",
            nbins_hist=hist_bins_spec,
            spin_sign=spin_sign,
            keep_mask=chelpg_spin_mask,
            normalize_spin_fraction=spin_config["normalize"]
        )

        write_combined_entity_timeseries(
            analysis_entity_ids,
            times_c,
            per_atom_q_c,
            per_atom_s_c,
            spin_column_label,
            "chelpg_loewdin",
            actor_config=actor_config
        )

        make_combined_hist_figure(
            analysis_entity_ids,
            hist_charge=hist_q_c,
            hist_spin=hist_s_c,
            atom_labels=atom_labels,
            charge_axis_label="CHELPG charge",
            spin_axis_label="Loewdin spin fraction" if spin_config["normalize"] else "Loewdin spin",
            fig_outname=append_suffix_to_path("chelpg_loewdin_histograms.png", output_variant_suffix) if output_variant_suffix is not None else "chelpg_loewdin_histograms.png"
        )

        if make_time_plots:
            make_timeseries_figure(
                times_c,
                per_atom_q_c,
                per_atom_s_c,
                analysis_entity_ids,
                atom_labels=atom_labels,
                fig_outname=append_suffix_to_path("chelpg_loewdin_timeseries.png", output_variant_suffix) if output_variant_suffix is not None else "chelpg_loewdin_timeseries.png",
                spin_ylabel="Loewdin spin fraction" if spin_config["normalize"] else "Loewdin spin"
            )

        terminal_summary_entries.append(
            build_terminal_summary_entry(
                "CHELPG + Loewdin",
                analysis_entity_ids,
                atom_labels,
                per_atom_q_c,
                per_atom_s_c,
                hist_q_c,
                hist_s_c,
            )
        )

    # --- ORCA: additional CHELPG + Mulliken analysis ---
    if orca_mode and population_config["chelpg_mulliken"] and primary_analysis_kind != "chelpg_mulliken":
        chelpg_charge_mask = apply_keep_mask_or_warn(
            spin_keep_mask,
            parse_frame_data("cq_mulliken_orca_todo.dat", dt_ps, atom_ids, "charge", "# CHELPG Population Analysis")[0].size,
            "CHELPG charge"
        )
        times_c, per_atom_q_c, hist_q_c = build_analysis_timeseries_and_stats(
            "cq_mulliken_orca_todo.dat",
            dt_ps,
            atom_ids,
            atom_labels,
            actor_config,
            kind="charge",
            header_start="# CHELPG Population Analysis",
            ts_outname=append_suffix_to_path("chelpg_mulliken_charge_timeseries.dat", output_variant_suffix) if output_variant_suffix is not None else "chelpg_mulliken_charge_timeseries.dat",
            avg_outname=append_suffix_to_path("chelpg_mulliken_charge_averages.dat", output_variant_suffix) if output_variant_suffix is not None else "chelpg_mulliken_charge_averages.dat",
            hist_prefix="chelpg_mulliken_charge_hist",
            modes_outname=append_suffix_to_path("chelpg_mulliken_charge_modes.dat", output_variant_suffix) if output_variant_suffix is not None else "chelpg_mulliken_charge_modes.dat",
            nbins_hist=hist_bins_spec,
            keep_mask=chelpg_charge_mask
        )

        hist_s_c = {}
        per_atom_s_c = {entity_id: np.array([]) for entity_id in analysis_entity_ids}
        chelpg_spin_mask = apply_keep_mask_or_warn(
            spin_keep_mask,
            parse_frame_data("cs_mulliken_orca_todo.dat", dt_ps, atom_ids, "spin", "# Mulliken Spin Population Analysis", spin_sign=spin_sign)[0].size,
            "Mulliken spin"
        )
        _times_spin_c, per_atom_s_c, hist_s_c = build_analysis_timeseries_and_stats(
            "cs_mulliken_orca_todo.dat",
            dt_ps,
            atom_ids,
            atom_labels,
            actor_config,
            kind="spin",
            header_start="# Mulliken Spin Population Analysis",
            ts_outname=append_suffix_to_path("mulliken_spin_timeseries_for_chelpg_mulliken.dat", output_variant_suffix) if output_variant_suffix is not None else "mulliken_spin_timeseries_for_chelpg_mulliken.dat",
            avg_outname=append_suffix_to_path("mulliken_spin_averages_for_chelpg_mulliken.dat", output_variant_suffix) if output_variant_suffix is not None else "mulliken_spin_averages_for_chelpg_mulliken.dat",
            hist_prefix="mulliken_spin_hist_for_chelpg_mulliken",
            modes_outname=append_suffix_to_path("mulliken_spin_modes_for_chelpg_mulliken.dat", output_variant_suffix) if output_variant_suffix is not None else "mulliken_spin_modes_for_chelpg_mulliken.dat",
            nbins_hist=hist_bins_spec,
            spin_sign=spin_sign,
            keep_mask=chelpg_spin_mask,
            normalize_spin_fraction=spin_config["normalize"]
        )

        write_combined_entity_timeseries(
            analysis_entity_ids,
            times_c,
            per_atom_q_c,
            per_atom_s_c,
            spin_column_label,
            "chelpg_mulliken",
            actor_config=actor_config
        )

        make_combined_hist_figure(
            analysis_entity_ids,
            hist_charge=hist_q_c,
            hist_spin=hist_s_c,
            atom_labels=atom_labels,
            charge_axis_label="CHELPG charge",
            spin_axis_label="Mulliken spin fraction" if spin_config["normalize"] else "Mulliken spin",
            fig_outname=append_suffix_to_path("chelpg_mulliken_histograms.png", output_variant_suffix) if output_variant_suffix is not None else "chelpg_mulliken_histograms.png"
        )

        if make_time_plots:
            make_timeseries_figure(
                times_c,
                per_atom_q_c,
                per_atom_s_c,
                analysis_entity_ids,
                atom_labels=atom_labels,
                fig_outname=append_suffix_to_path("chelpg_mulliken_timeseries.png", output_variant_suffix) if output_variant_suffix is not None else "chelpg_mulliken_timeseries.png",
                spin_ylabel="Mulliken spin fraction" if spin_config["normalize"] else "Mulliken spin"
            )

        terminal_summary_entries.append(
            build_terminal_summary_entry(
                "CHELPG + Mulliken",
                analysis_entity_ids,
                atom_labels,
                per_atom_q_c,
                per_atom_s_c,
                hist_q_c,
                hist_s_c,
            )
        )

    # --- ORCA: additional CHELPG + Hirshfeld analysis ---
    if orca_mode and population_config["chelpg_hirshfeld"] and primary_analysis_kind != "chelpg_hirshfeld":
        chelpg_charge_mask = apply_keep_mask_or_warn(
            spin_keep_mask,
            parse_frame_data("cq_hirshfeld_orca_todo.dat", dt_ps, atom_ids, "charge", "# CHELPG Population Analysis")[0].size,
            "CHELPG charge"
        )
        times_c, per_atom_q_c, hist_q_c = build_analysis_timeseries_and_stats(
            "cq_hirshfeld_orca_todo.dat",
            dt_ps,
            atom_ids,
            atom_labels,
            actor_config,
            kind="charge",
            header_start="# CHELPG Population Analysis",
            ts_outname=append_suffix_to_path("chelpg_hirshfeld_charge_timeseries.dat", output_variant_suffix) if output_variant_suffix is not None else "chelpg_hirshfeld_charge_timeseries.dat",
            avg_outname=append_suffix_to_path("chelpg_hirshfeld_charge_averages.dat", output_variant_suffix) if output_variant_suffix is not None else "chelpg_hirshfeld_charge_averages.dat",
            hist_prefix="chelpg_hirshfeld_charge_hist",
            modes_outname=append_suffix_to_path("chelpg_hirshfeld_charge_modes.dat", output_variant_suffix) if output_variant_suffix is not None else "chelpg_hirshfeld_charge_modes.dat",
            nbins_hist=hist_bins_spec,
            keep_mask=chelpg_charge_mask
        )

        hist_s_c = {}
        per_atom_s_c = {entity_id: np.array([]) for entity_id in analysis_entity_ids}
        chelpg_spin_mask = apply_keep_mask_or_warn(
            spin_keep_mask,
            parse_frame_data("cs_hirshfeld_orca_todo.dat", dt_ps, atom_ids, "spin", "# Hirshfeld Spin Population Analysis", spin_sign=spin_sign)[0].size,
            "Hirshfeld spin"
        )
        _times_spin_c, per_atom_s_c, hist_s_c = build_analysis_timeseries_and_stats(
            "cs_hirshfeld_orca_todo.dat",
            dt_ps,
            atom_ids,
            atom_labels,
            actor_config,
            kind="spin",
            header_start="# Hirshfeld Spin Population Analysis",
            ts_outname=append_suffix_to_path("hirshfeld_spin_timeseries_for_chelpg_hirshfeld.dat", output_variant_suffix) if output_variant_suffix is not None else "hirshfeld_spin_timeseries_for_chelpg_hirshfeld.dat",
            avg_outname=append_suffix_to_path("hirshfeld_spin_averages_for_chelpg_hirshfeld.dat", output_variant_suffix) if output_variant_suffix is not None else "hirshfeld_spin_averages_for_chelpg_hirshfeld.dat",
            hist_prefix="hirshfeld_spin_hist_for_chelpg_hirshfeld",
            modes_outname=append_suffix_to_path("hirshfeld_spin_modes_for_chelpg_hirshfeld.dat", output_variant_suffix) if output_variant_suffix is not None else "hirshfeld_spin_modes_for_chelpg_hirshfeld.dat",
            nbins_hist=hist_bins_spec,
            spin_sign=spin_sign,
            keep_mask=chelpg_spin_mask,
            normalize_spin_fraction=spin_config["normalize"]
        )

        write_combined_entity_timeseries(
            analysis_entity_ids,
            times_c,
            per_atom_q_c,
            per_atom_s_c,
            spin_column_label,
            "chelpg_hirshfeld",
            actor_config=actor_config
        )

        make_combined_hist_figure(
            analysis_entity_ids,
            hist_charge=hist_q_c,
            hist_spin=hist_s_c,
            atom_labels=atom_labels,
            charge_axis_label="CHELPG charge",
            spin_axis_label="Hirshfeld spin fraction" if spin_config["normalize"] else "Hirshfeld spin",
            fig_outname=append_suffix_to_path("chelpg_hirshfeld_histograms.png", output_variant_suffix) if output_variant_suffix is not None else "chelpg_hirshfeld_histograms.png"
        )

        if make_time_plots:
            make_timeseries_figure(
                times_c,
                per_atom_q_c,
                per_atom_s_c,
                analysis_entity_ids,
                atom_labels=atom_labels,
                fig_outname=append_suffix_to_path("chelpg_hirshfeld_timeseries.png", output_variant_suffix) if output_variant_suffix is not None else "chelpg_hirshfeld_timeseries.png",
                spin_ylabel="Hirshfeld spin fraction" if spin_config["normalize"] else "Hirshfeld spin"
            )

        terminal_summary_entries.append(
            build_terminal_summary_entry(
                "CHELPG + Hirshfeld",
                analysis_entity_ids,
                atom_labels,
                per_atom_q_c,
                per_atom_s_c,
                hist_q_c,
                hist_s_c,
            )
        )

    print_terminal_analysis_summary(terminal_summary_entries)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nchau :(")
        sys.exit(130)
