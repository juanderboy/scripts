#!/usr/bin/env python3
"""Command-line interface and orchestration for TolKinet."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from kinet_common import MODEL_LABELS, MODEL_SPECIES, PARAMETER_LABELS
from kinet_export import export_final_fit_outputs
from kinet_fitting import fit_model_with_auto_k_max, is_near_bound
from kinet_io import read_experiment, resolve_input_file
from kinet_plotting import plot_experiment_overview, plot_result, print_exploration_message
from kinet_preprocessing import crop_wavelengths, drop_spectra, parse_spectrum_selection, preprocess_experiment


def print_startup_banner() -> None:
    """Print the program title and short description."""
    width = 72
    lines = [
        "TOLKinetics",
        "Multiwavelength kinetic analysis",
        "Factor analysis and global fitting of spectral time series",
        "vibecoded by Tolkien, 2026",
    ]

    print()
    print("+" + "-" * (width - 2) + "+")
    for i, line in enumerate(lines):
        if i == 0:
            print("|" + line.center(width - 2) + "|")
            print("|" + " " * (width - 2) + "|")
        else:
            print("|" + line.center(width - 2) + "|")
    print("+" + "-" * (width - 2) + "+")
    print()



def ask_model_choice(default_model: str) -> str:
    """Ask for the kinetic model to use."""
    model_keys = list(MODEL_LABELS)
    if default_model not in MODEL_LABELS:
        raise ValueError(f"Unknown kinetic model: {default_model}")

    print("Modelo cinetico:")
    for i, key in enumerate(model_keys, start=1):
        marker = "default" if key == default_model else ""
        suffix = f" [{marker}]" if marker else ""
        print(f"  {i}. {MODEL_LABELS[key]}{suffix}")

    choice = input("Elegir modelo? Enter = default: ").strip()
    if not choice:
        return default_model

    index = int(choice)
    if index < 1 or index > len(model_keys):
        raise ValueError(f"Model choice must be between 1 and {len(model_keys)}")

    return model_keys[index - 1]



def ask_fit_method_choice(default_method: str, model: str) -> str:
    """Ask how pure spectra should be estimated during fitting."""
    if default_method not in {"nnls", "pinv", "factor"}:
        raise ValueError(f"Unknown fit method: {default_method}")

    direct_methods = [
        ("nnls", "NNLS: espectros no negativos E(lambda) >= 0"),
        ("pinv", "Pseudoinversa: minimos cuadrados sin restricciones"),
    ]
    available_methods = direct_methods.copy()
    if model in {"a_to_b", "a_to_b_to_c", "a_rev_b_to_c"}:
        available_methods.append(
            ("factor", "Factor/SVD: ajuste en el espacio de factores")
        )

    if default_method == "factor" and model not in {"a_to_b", "a_to_b_to_c", "a_rev_b_to_c"}:
        default_method = "nnls"

    print()
    print("Metodo para obtener los espectros puros en cada prueba cinetica:")
    for i, (key, label) in enumerate(available_methods, start=1):
        suffix = " [default]" if key == default_method else ""
        print(f"  {i}. {label}{suffix}")

    choice = input("Elegir metodo? Enter = default: ").strip().lower()
    if not choice:
        return default_method

    for i, (key, _) in enumerate(available_methods, start=1):
        if choice == str(i) or choice == key:
            return key

    raise ValueError(
        "Fit method choice must be one of: "
        + ", ".join(key for key, _ in available_methods)
    )



def print_fit_report(
    input_path: Path,
    experiment: Experiment,
    result: FitResult,
    final_k_bounds: tuple[float, float],
    args: argparse.Namespace,
    c0: float,
    reaction_start_time: float | None,
    model: str,
) -> None:
    """Print the numerical summary for one fit."""
    k_min, k_max = final_k_bounds

    print(f"Input file: {input_path}")
    print(
        "Data matrix: "
        f"{experiment.absorbance.shape[0]} wavelengths x "
        f"{experiment.absorbance.shape[1]} times"
    )
    print(f"Wavelength range: {experiment.wavelength[0]:g} - {experiment.wavelength[-1]:g}")
    print(f"Time range: {experiment.t[0]:g} - {experiment.t[-1]:g}")
    if reaction_start_time is not None:
        print(f"reaction start time: {reaction_start_time:g} original time units")
    print(f"c0: {c0:g} M")
    print(f"model: {MODEL_LABELS[model]}")
    print(f"fit method: {result.method}")
    if args.initial_spectrum_weight > 0:
        print(f"initial spectrum weight: {args.initial_spectrum_weight:g}")
    if k_max == args.k_max:
        print(f"k search range: {k_min:g} - {k_max:g}")
    else:
        print(f"k search range: {k_min:g} - {k_max:g} (expanded from {args.k_max:g})")
    for name, value in result.params.items():
        print(f"{PARAMETER_LABELS.get(name, name)}: {value:.10g}")
        if is_near_bound(value, k_min):
            print(
                "  warning: "
                f"{PARAMETER_LABELS.get(name, name)} reached the lower search bound"
            )
        elif is_near_bound(value, k_max):
            print(
                "  warning: "
                f"{PARAMETER_LABELS.get(name, name)} reached the upper search bound"
            )
    if set(result.params) == {"k"}:
        print(f"half-life: {np.log(2) / result.params['k']:.10g}")
    print(f"fit error: {result.error:.10g}")
    print(f"minimum recovered spectrum value: {result.spectra.min():.10g}")
    print("first singular values:", " ".join(f"{x:.6g}" for x in result.singular_values[:10]))
    print()



def ask_post_fit_spectra_to_drop(experiment: Experiment) -> list[int] | None:
    """Ask whether to remove spectra after inspecting the final residuals panel."""
    while True:
        print("Los indices son los que aparecen en el tooltip del panel de residuos.")
        choice = input(
            "Espectros a eliminar y recalcular? "
            "Ej: 3 o 3,7-9. Enter/n = terminar: "
        ).strip()
        normalized = choice.lower()
        if normalized in {"", "n", "no", "fin", "q", "quit", "salir"}:
            return None

        try:
            drop_indices = parse_spectrum_selection(choice, experiment.t.size)
        except ValueError as exc:
            print(f"Seleccion invalida: {exc}")
            continue

        if not drop_indices:
            return None
        if experiment.t.size - len(drop_indices) < 2:
            print("Deben quedar al menos 2 espectros para recalcular.")
            continue
        return drop_indices



def build_parser() -> argparse.ArgumentParser:
    """Command-line options."""
    parser = argparse.ArgumentParser(
        description="Fit multiwavelength spectrophotometric kinetic data."
    )
    parser.add_argument("input", nargs="?", default="117.txt", help="Input text file")
    parser.add_argument("--lambda-min", type=float, default=320.0)
    parser.add_argument("--lambda-max", type=float, default=820.0)
    parser.add_argument("--baseline-points", type=int, default=20)
    parser.add_argument(
        "--baseline-mode",
        choices=("auto-flat", "region", "points", "none"),
        default="auto-flat",
        help="Baseline correction mode",
    )
    parser.add_argument(
        "--baseline-window",
        type=float,
        default=40.0,
        help="Window width in nm for automatic flat baseline search",
    )
    parser.add_argument(
        "--baseline-lambda-min",
        type=float,
        default=None,
        help="Lower wavelength limit for baseline correction region",
    )
    parser.add_argument(
        "--baseline-lambda-max",
        type=float,
        default=None,
        help="Upper wavelength limit for baseline correction region",
    )
    parser.add_argument(
        "--drop-spectra",
        default="",
        help="1-based spectra to remove before fitting, e.g. '1' or '1,4-6'",
    )
    parser.add_argument(
        "--reaction-start-time",
        type=float,
        default=None,
        help=(
            "Original time where the reaction starts. Spectra with t <= this "
            "value are discarded and remaining times are shifted by this value."
        ),
    )
    parser.add_argument(
        "--kd-lambda-start",
        type=float,
        default=None,
        help="Override first wavelength when converting KD files",
    )
    parser.add_argument(
        "--kd-lambda-step",
        type=float,
        default=None,
        help="Override wavelength step when converting KD files",
    )
    parser.add_argument("--c0", type=float, default=1.0, help="Initial concentration in M")
    parser.add_argument(
        "--k-min",
        type=float,
        default=1e-8,
        help="Lower bound for k during log-space optimization",
    )
    parser.add_argument(
        "--k-max",
        type=float,
        default=1e-1,
        help="Upper bound for k during log-space optimization",
    )
    parser.add_argument(
        "--no-auto-expand-k-max",
        action="store_false",
        dest="auto_expand_k_max",
        help="Disable automatic expansion when a fitted constant reaches --k-max",
    )
    parser.add_argument(
        "--k-max-expand-factor",
        type=float,
        default=10.0,
        help="Multiplier used when automatically expanding --k-max",
    )
    parser.add_argument(
        "--k-max-expand-steps",
        type=int,
        default=4,
        help="Maximum number of automatic --k-max expansions",
    )
    parser.add_argument(
        "--components",
        type=int,
        default=None,
        help="Number of SVD components; defaults to the number of model species",
    )
    parser.add_argument(
        "--fit-method",
        choices=("nnls", "pinv", "factor"),
        default="nnls",
        help=(
            "nnls enforces nonnegative spectra; pinv uses unconstrained "
            "pseudoinverse spectra; factor reproduces the MATLAB factor-space fit"
        ),
    )
    parser.add_argument(
        "--initial-spectrum-weight",
        type=float,
        default=0.0,
        help=(
            "Softly pull species A toward the first measured spectrum in NNLS. "
            "0 disables it; 1 is roughly one extra time point."
        ),
    )
    parser.add_argument(
        "--model",
        choices=tuple(MODEL_LABELS),
        default="a_to_b",
        help="Kinetic model to fit",
    )
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--skip-preprocess-dialog",
        action="store_true",
        help="Do not show the initial raw spectra dialog before preprocessing",
    )
    return parser



def main() -> None:
    """Load data, preprocess, fit the kinetic model and report results."""
    args = build_parser().parse_args()

    print_startup_banner()
    source_input_path = Path(args.input)
    input_path = resolve_input_file(args)
    experiment = read_experiment(input_path)

    model = args.model
    fit_method = args.fit_method
    if not args.no_plot and not args.skip_preprocess_dialog:
        print_exploration_message()
        plot_experiment_overview(
            experiment,
            baseline_window=args.baseline_window,
        )
        model = ask_model_choice(args.model)
        fit_method = ask_fit_method_choice(args.fit_method, model)

    corrected, work_range, c0, reaction_start_time = preprocess_experiment(args, experiment)
    cropped = crop_wavelengths(corrected, work_range[0], work_range[1])
    n_components = args.components
    if n_components is None:
        n_components = len(MODEL_SPECIES[model])

    fit_experiment = cropped
    while True:
        result, final_k_bounds = fit_model_with_auto_k_max(
            model,
            fit_experiment,
            c0=c0,
            n_components=n_components,
            method=fit_method,
            k_bounds=(args.k_min, args.k_max),
            auto_expand=args.auto_expand_k_max,
            expand_factor=args.k_max_expand_factor,
            max_expand_steps=args.k_max_expand_steps,
            initial_spectrum_weight=args.initial_spectrum_weight,
        )

        print_fit_report(
            input_path,
            fit_experiment,
            result,
            final_k_bounds,
            args,
            c0,
            reaction_start_time,
            model,
        )

        if args.no_plot:
            break

        plot_result(fit_experiment, result, input_filename=input_path.name)
        drop_indices = ask_post_fit_spectra_to_drop(fit_experiment)
        if drop_indices is None:
            break

        removed = ", ".join(
            f"{index + 1} (t={fit_experiment.t[index]:.6g})"
            for index in drop_indices
        )
        print(f"Recalculando sin espectros: {removed}")
        print()
        fit_experiment = drop_spectra(fit_experiment, drop_indices)

    export_final_fit_outputs(
        source_input_path,
        input_path,
        fit_experiment,
        result,
        final_k_bounds,
        args,
        c0,
        reaction_start_time,
        model,
    )



if __name__ == "__main__":
    main()
