#!/usr/bin/env python3
"""Output writers for final TolKinet fits."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from kinet_common import Experiment, FitResult, MODEL_LABELS, PARAMETER_LABELS
from kinet_fitting import is_near_bound
from kinet_plotting import plot_result


def output_prefix_from_input(path: Path) -> Path:
    """Return the file prefix used for exported fit artifacts."""
    output_dir = path.parent if str(path.parent) else Path(".")
    return output_dir / path.stem



def write_fit_summary_dat(
    path: Path,
    source_input_path: Path,
    analysis_input_path: Path,
    experiment: Experiment,
    result: FitResult,
    final_k_bounds: tuple[float, float],
    args: argparse.Namespace,
    c0: float,
    reaction_start_time: float | None,
    model: str,
) -> None:
    """Write a tab-separated summary of the final accepted fit."""
    k_min, k_max = final_k_bounds
    lines = [
        "# key\tvalue",
        f"source_file\t{source_input_path}",
        f"analysis_file\t{analysis_input_path}",
        f"model\t{MODEL_LABELS[model]}",
        f"fit_method\t{result.method}",
        f"c0_M\t{c0:.10g}",
        f"n_wavelengths\t{experiment.wavelength.size}",
        f"n_spectra\t{experiment.t.size}",
        f"wavelength_min\t{experiment.wavelength[0]:.10g}",
        f"wavelength_max\t{experiment.wavelength[-1]:.10g}",
        f"time_min\t{experiment.t[0]:.10g}",
        f"time_max\t{experiment.t[-1]:.10g}",
        f"fit_error\t{result.error:.10g}",
        f"minimum_recovered_spectrum_value\t{result.spectra.min():.10g}",
        f"k_search_min\t{k_min:.10g}",
        f"k_search_max\t{k_max:.10g}",
    ]
    if reaction_start_time is not None:
        lines.append(f"reaction_start_time_original_units\t{reaction_start_time:.10g}")
    if args.initial_spectrum_weight > 0:
        lines.append(f"initial_spectrum_weight\t{args.initial_spectrum_weight:.10g}")
    for name, value in result.params.items():
        lines.append(f"{PARAMETER_LABELS.get(name, name)}\t{value:.10g}")
    if set(result.params) == {"k"}:
        lines.append(f"half_life\t{np.log(2) / result.params['k']:.10g}")
    lines.append("species\t" + ",".join(result.species_labels))
    lines.append(
        "singular_values\t"
        + ",".join(f"{value:.10g}" for value in result.singular_values)
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")



def export_final_fit_outputs(
    source_input_path: Path,
    analysis_input_path: Path,
    experiment: Experiment,
    result: FitResult,
    final_k_bounds: tuple[float, float],
    args: argparse.Namespace,
    c0: float,
    reaction_start_time: float | None,
    model: str,
) -> None:
    """Save final fit outputs for external plotting and reporting."""
    prefix = output_prefix_from_input(source_input_path)
    concentration_path = prefix.with_name(f"{prefix.name}_concentrations.dat")
    spectra_path = prefix.with_name(f"{prefix.name}_pure_spectra.dat")
    summary_path = prefix.with_name(f"{prefix.name}_fit_summary.dat")
    panel_path = prefix.with_name(f"{prefix.name}_fit_panel.png")

    concentration_table = np.column_stack((experiment.t, result.c.T))
    concentration_header = "\t".join(("time", *result.species_labels))
    np.savetxt(
        concentration_path,
        concentration_table,
        delimiter="\t",
        header=concentration_header,
        comments="# ",
    )

    spectra_table = np.column_stack((experiment.wavelength, result.spectra))
    spectra_header = "\t".join(("wavelength", *result.species_labels))
    np.savetxt(
        spectra_path,
        spectra_table,
        delimiter="\t",
        header=spectra_header,
        comments="# ",
    )

    write_fit_summary_dat(
        summary_path,
        source_input_path,
        analysis_input_path,
        experiment,
        result,
        final_k_bounds,
        args,
        c0,
        reaction_start_time,
        model,
    )
    plot_result(
        experiment,
        result,
        input_filename=source_input_path.name,
        save_path=panel_path,
        show=False,
    )

    print("Archivos exportados:")
    print(f"  {concentration_path}")
    print(f"  {spectra_path}")
    print(f"  {summary_path}")
    print(f"  {panel_path}")

