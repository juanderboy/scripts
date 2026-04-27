#!/usr/bin/env python3
"""Preprocessing and interactive preprocessing choices for TolKinet."""

from __future__ import annotations

import argparse

import numpy as np

from kinet_common import Experiment


def baseline_correct(absorbance: np.ndarray, n_points: int = 20) -> np.ndarray:
    """Subtract the mean of the last n_points from each spectrum.

    Each column is one spectrum at one time. The correction is therefore
    column-wise, matching the original MATLAB loop.
    """
    baseline = absorbance[-n_points:, :].mean(axis=0, keepdims=True)
    return absorbance - baseline



def baseline_correct_region(experiment: Experiment, wavelength_min: float, wavelength_max: float) -> np.ndarray:
    """Subtract the mean absorbance in a selected wavelength interval."""
    mask = (experiment.wavelength >= wavelength_min) & (
        experiment.wavelength <= wavelength_max
    )
    if not np.any(mask):
        raise ValueError(
            f"Baseline region {wavelength_min:g}-{wavelength_max:g} has no data points"
        )
    baseline = experiment.absorbance[mask, :].mean(axis=0, keepdims=True)
    return experiment.absorbance - baseline



def suggest_auto_baseline_region(
    experiment: Experiment,
    window_width: float = 40.0,
    min_points: int = 5,
) -> tuple[float, float]:
    """Suggest a flat, low-activity wavelength interval for baseline correction."""
    wavelength = experiment.wavelength
    candidates: list[tuple[float, float, float, float, float]] = []

    for start_index, start in enumerate(wavelength):
        end = start + window_width
        mask = (wavelength >= start) & (wavelength <= end)
        if np.count_nonzero(mask) < min_points:
            continue

        sub = experiment.absorbance[mask, :]
        temporal_activity = float(np.mean(np.std(sub, axis=1)))
        mean_abs_signal = float(np.mean(np.abs(sub)))
        spectral_slope = float(np.mean(np.abs(np.diff(sub, axis=0)))) if sub.shape[0] > 1 else 0.0
        candidates.append((temporal_activity, mean_abs_signal, spectral_slope, start, wavelength[mask][-1]))

    if not candidates:
        raise ValueError("Could not find an automatic baseline region")

    metrics = np.array([[row[0], row[1], row[2]] for row in candidates])
    scale = np.median(metrics, axis=0)
    scale[scale == 0] = 1.0
    normalized = metrics / scale
    scores = 0.6 * normalized[:, 0] + 0.25 * normalized[:, 1] + 0.15 * normalized[:, 2]
    best_index = int(np.argmin(scores))
    return candidates[best_index][3], candidates[best_index][4]



def drop_spectra(experiment: Experiment, indices: list[int]) -> Experiment:
    """Remove selected spectra by zero-based column indices."""
    if not indices:
        return experiment

    mask = np.ones(experiment.t.size, dtype=bool)
    mask[indices] = False
    return Experiment(
        t=experiment.t[mask],
        wavelength=experiment.wavelength,
        absorbance=experiment.absorbance[:, mask],
    )



def start_reaction_at_time(
    experiment: Experiment,
    reaction_start_time: float | None,
) -> Experiment:
    """Keep spectra with t > reaction_start_time and shift that time to zero."""
    if reaction_start_time is None:
        return experiment
    if not np.isfinite(reaction_start_time):
        raise ValueError("Reaction start time must be finite")

    mask = experiment.t > reaction_start_time
    if not np.any(mask):
        raise ValueError(
            f"Reaction start time {reaction_start_time:g} leaves no spectra with t > t_start"
        )

    return Experiment(
        t=experiment.t[mask] - reaction_start_time,
        wavelength=experiment.wavelength,
        absorbance=experiment.absorbance[:, mask],
    )



def parse_spectrum_selection(selection: str, n_spectra: int) -> list[int]:
    """Parse 1-based spectrum indices such as '1,3-5'."""
    if not selection.strip():
        return []

    indices: set[int] = set()
    for part in selection.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", maxsplit=1)
            start = int(start_text)
            end = int(end_text)
            if start > end:
                start, end = end, start
            values = range(start, end + 1)
        else:
            values = [int(part)]

        for value in values:
            if value < 1 or value > n_spectra:
                raise ValueError(
                    f"Spectrum index {value} is outside the valid range 1-{n_spectra}"
                )
            indices.add(value - 1)

    return sorted(indices)



def parse_wavelength_range(text: str) -> tuple[float, float]:
    """Parse wavelength ranges like '800-820', '800 820' or '800,820'."""
    normalized = text.strip().replace(",", " ")
    if "-" in normalized:
        parts = [part.strip() for part in normalized.split("-", maxsplit=1)]
    else:
        parts = normalized.split()

    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError("Expected wavelength range like 800-820 or 800 820")

    lower, upper = sorted((float(parts[0]), float(parts[1])))
    return lower, upper



def parse_baseline_choice(
    text: str,
    auto_region: tuple[float, float],
    default_points: int,
) -> tuple[str, tuple[float, float] | None, int]:
    """Parse the interactive baseline correction choice."""
    choice = text.strip().lower()
    if not choice or choice in {"auto", "automatic", "a", "1"}:
        return "region", auto_region, default_points

    if choice in {"manual", "m", "2"}:
        region_text = input("Manual baseline region in nm, e.g. 760-820: ").strip()
        return "region", parse_wavelength_range(region_text), default_points

    if choice in {"none", "no", "n", "4"}:
        return "none", None, default_points

    if choice in {"points", "point", "p", "3"}:
        return "points", None, default_points

    point_prefixes = ("points ", "point ", "p ")
    if choice.startswith(point_prefixes):
        _, value_text = choice.split(maxsplit=1)
        n_points = int(value_text)
        if n_points <= 0:
            raise ValueError("Number of baseline points must be positive")
        return "points", None, n_points

    # A bare range such as 760-820 is treated as a manual baseline region.
    return "region", parse_wavelength_range(text), default_points



def parse_reaction_start_time_choice(
    text: str,
    default_reaction_start_time: float | None,
) -> float | None:
    """Parse the interactive reaction-start time choice."""
    choice = text.strip().lower()
    if not choice:
        return default_reaction_start_time
    if choice in {"none", "no", "n"}:
        return None

    reaction_start_time = float(choice)
    if not np.isfinite(reaction_start_time):
        raise ValueError("Reaction start time must be finite")
    return reaction_start_time



def crop_wavelengths(
    experiment: Experiment,
    wavelength_min: float,
    wavelength_max: float,
) -> Experiment:
    """Keep only wavelengths inside the selected interval."""
    mask = (experiment.wavelength >= wavelength_min) & (
        experiment.wavelength <= wavelength_max
    )
    if not np.any(mask):
        raise ValueError(
            f"Wavelength range {wavelength_min:g}-{wavelength_max:g} has no data points"
        )
    return Experiment(
        t=experiment.t,
        wavelength=experiment.wavelength[mask],
        absorbance=experiment.absorbance[mask, :],
    )



def corrected_absorbance_from_baseline_choice(
    experiment: Experiment,
    baseline_mode: str,
    baseline_region: tuple[float, float] | None,
    baseline_points: int,
    baseline_window: float,
) -> tuple[np.ndarray, tuple[float, float] | None]:
    """Apply the selected baseline correction and return the region used."""
    if baseline_mode == "none":
        return experiment.absorbance.copy(), baseline_region
    if baseline_mode == "points":
        return baseline_correct(experiment.absorbance, baseline_points), baseline_region
    if baseline_mode == "auto-flat":
        baseline_region = suggest_auto_baseline_region(
            experiment,
            window_width=baseline_window,
        )
        print(
            "Automatic baseline region: "
            f"{baseline_region[0]:g}-{baseline_region[1]:g} nm"
        )
        return (
            baseline_correct_region(
                experiment,
                baseline_region[0],
                baseline_region[1],
            ),
            baseline_region,
        )
    if baseline_mode == "region":
        if baseline_region is None:
            raise ValueError("Baseline mode 'region' requires a baseline region")
        return (
            baseline_correct_region(
                experiment,
                baseline_region[0],
                baseline_region[1],
            ),
            baseline_region,
        )
    raise ValueError(f"Unknown baseline mode: {baseline_mode}")



def apply_preprocessing_choices(
    args: argparse.Namespace,
    experiment: Experiment,
    drop_indices: list[int],
    reaction_start_time: float | None,
    baseline_mode: str,
    baseline_region: tuple[float, float] | None,
    baseline_points: int,
    work_range: tuple[float, float],
) -> tuple[Experiment, Experiment, tuple[float, float] | None]:
    """Apply spectrum/time pruning, baseline correction and wavelength crop."""
    pruned = drop_spectra(experiment, drop_indices)
    pruned = start_reaction_at_time(pruned, reaction_start_time)
    corrected_absorbance, used_region = corrected_absorbance_from_baseline_choice(
        pruned,
        baseline_mode,
        baseline_region,
        baseline_points,
        args.baseline_window,
    )
    corrected = Experiment(
        t=pruned.t,
        wavelength=pruned.wavelength,
        absorbance=corrected_absorbance,
    )
    cropped = crop_wavelengths(corrected, work_range[0], work_range[1])
    return corrected, cropped, used_region



def ask_preprocessing_choices(
    experiment: Experiment,
    baseline_points: int,
    default_work_range: tuple[float, float],
    default_c0: float,
    baseline_window: float,
    default_reaction_start_time: float | None,
) -> tuple[
    list[int],
    float | None,
    str,
    tuple[float, float] | None,
    int,
    tuple[float, float],
    float,
]:
    """Ask which spectra/baseline region/wavelength range to use."""
    print()
    print(f"El experimento tiene {experiment.t.size} espectros.")
    print("Los indices de espectros son 1-based: el primer espectro es 1.")
    drop_text = input(
        "Espectros a eliminar? Ej: 1 o 1,4-6. Enter = ninguno: "
    )
    drop_indices = parse_spectrum_selection(drop_text, experiment.t.size)

    dropped = drop_spectra(experiment, drop_indices)
    if default_reaction_start_time is None:
        start_default_text = "sin recorte"
    else:
        start_default_text = f"{default_reaction_start_time:g}"
    start_text = input(
        "Tiempo de inicio de reaccion? "
        f"Se usan solo espectros con t > t_inicio. Enter = {start_default_text}: "
    )
    reaction_start_time = parse_reaction_start_time_choice(
        start_text,
        default_reaction_start_time,
    )
    dropped = start_reaction_at_time(dropped, reaction_start_time)

    auto_region = suggest_auto_baseline_region(
        dropped,
        window_width=baseline_window,
    )
    print()
    print(
        "Automatic baseline suggestion: "
        f"{auto_region[0]:g}-{auto_region[1]:g} nm"
    )
    print("Baseline correction options:")
    print(f"  1. Accept automatic region {auto_region[0]:g}-{auto_region[1]:g} nm")
    print("  2. Type a manual wavelength range, e.g. 760-820")
    print(f"  3. Use the last wavelength points, e.g. points 30 (default = {baseline_points})")
    print("  4. Do not correct baseline")
    baseline_text = input(
        "Choice [Enter/1 automatic, range manual, points N, 4 none]: "
    )
    baseline_mode, baseline_region, baseline_points = parse_baseline_choice(
        baseline_text,
        auto_region,
        baseline_points,
    )

    work_text = input(
        "Rango de lambdas para trabajar? Ej: 320-820. "
        f"Enter = {default_work_range[0]:g}-{default_work_range[1]:g}: "
    ).strip()
    if not work_text:
        work_range = default_work_range
    else:
        work_range = parse_wavelength_range(work_text)

    c0_text = input(
        f"Concentracion inicial c0 en molar? Enter = {default_c0:g} M: "
    ).strip()
    if not c0_text:
        c0 = default_c0
    else:
        c0 = float(c0_text)
        if c0 <= 0:
            raise ValueError("c0 must be positive")

    return (
        drop_indices,
        reaction_start_time,
        baseline_mode,
        baseline_region,
        baseline_points,
        work_range,
        c0,
    )



def preprocess_experiment(
    args: argparse.Namespace,
    experiment: Experiment,
) -> tuple[Experiment, tuple[float, float], float, float | None]:
    """Apply optional spectrum removal and baseline correction."""
    drop_indices = parse_spectrum_selection(args.drop_spectra, experiment.t.size)
    work_range = tuple(sorted((args.lambda_min, args.lambda_max)))
    c0 = args.c0
    baseline_mode = args.baseline_mode
    baseline_points = args.baseline_points
    reaction_start_time = args.reaction_start_time

    baseline_region = None
    if args.baseline_lambda_min is not None or args.baseline_lambda_max is not None:
        if args.baseline_lambda_min is None or args.baseline_lambda_max is None:
            raise ValueError(
                "Use both --baseline-lambda-min and --baseline-lambda-max, or neither"
            )
        baseline_mode = "region"
        baseline_region = tuple(
            sorted((args.baseline_lambda_min, args.baseline_lambda_max))
        )

    if args.no_plot or args.skip_preprocess_dialog:
        corrected, _, _ = apply_preprocessing_choices(
            args,
            experiment,
            drop_indices,
            reaction_start_time,
            baseline_mode,
            baseline_region,
            baseline_points,
            work_range,
        )
        return corrected, work_range, c0, reaction_start_time

    while True:
        (
            drop_indices,
            reaction_start_time,
            baseline_mode,
            baseline_region,
            baseline_points,
            work_range,
            c0,
        ) = ask_preprocessing_choices(
            experiment,
            baseline_points,
            default_work_range=work_range,
            default_c0=c0,
            baseline_window=args.baseline_window,
            default_reaction_start_time=reaction_start_time,
        )
        corrected, cropped, used_region = apply_preprocessing_choices(
            args,
            experiment,
            drop_indices,
            reaction_start_time,
            baseline_mode,
            baseline_region,
            baseline_points,
            work_range,
        )
        from kinet_plotting import plot_preprocessed_experiment_preview

        plot_preprocessed_experiment_preview(
            corrected,
            cropped,
            work_range,
            baseline_mode,
            used_region,
            baseline_points,
            reaction_start_time,
        )
        approve = input(
            "Aprobar estos espectros para el ajuste? "
            "Enter/s = si, n = volver a elegir poda/baseline/rango: "
        ).strip().lower()
        if approve in {"", "s", "si", "sí", "y", "yes"}:
            return corrected, work_range, c0, reaction_start_time
        if approve in {"n", "no"}:
            print()
            print("Volviendo a las opciones de preprocesado.")
            continue
        raise ValueError("Approval choice must be yes or no")
