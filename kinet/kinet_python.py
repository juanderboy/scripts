#!/usr/bin/env python3
"""Kinetic-spectral fit for multiwavelength spectral time series.

This is a Python version of the original MATLAB routines kinet.m,
perfil.m and terror.m. The input file is expected to have the same
layout as 117.txt:

    first row:    0 ; t1 ; t2 ; ...
    first column: lambda values
    body:         absorbance(lambda, time)

Convenciones de matrices usadas en todo el programa:

    A_exp      matriz experimental de absorbancias
               filas    = longitudes de onda
               columnas = tiempos

    C          perfiles de concentracion calculados por el modelo cinetico
               filas    = especies
               columnas = tiempos

    E          espectros puros recuperados, expresados como epsilon(lambda)
               si c0 se informa en unidades molares
               filas    = longitudes de onda
               columnas = especies

    A_calc     matriz reconstruida por el modelo
               A_calc = E @ C

Modelos cineticos disponibles:

    A -> B

        A(t) = c0 exp(-k t)
        B(t) = c0 - A(t)

    A -> B -> C

        dA/dt = -k1 A
        dB/dt =  k1 A - k2 B
        dC/dt =  k2 B

    A <-> B -> C

        dA/dt = -k1 A + k_1 B
        dB/dt =  k1 A - (k_1 + k2) B
        dC/dt =  k2 B

Hay dos metodos de ajuste:

    nnls       Metodo por defecto. Para cada k calcula C(t), ajusta E por
               minimos cuadrados no negativos y minimiza ||A_exp - E C||.
               Impone la condicion fisica E(lambda) >= 0.

    factor     Reproduce la logica original de MATLAB en el espacio de SVD.
               Es util para comparar contra las rutinas viejas, pero puede
               dar espectros negativos porque no impone restricciones.

Graficos generados luego del ajuste:

    Se muestra un unico panel grande con:

    1. Baseline-corrected spectra:
       todos los espectros A(lambda) corregidos y recortados, uno por tiempo.
       El color codifica el tiempo: negro al comienzo, naranja al final.

    2. Recovered epsilon spectra:
       espectros puros recuperados para A y B, escalados por c0.

    3. Concentration profiles:
       trazas temporales A(t) y B(t) calculadas con la k ajustada.

    4. Residuals:
       A_exp - A_calc para todos los tiempos.

    5. Singular values:
       valores singulares de la matriz experimental tratada, en escala lineal.

Si se corre con graficos, antes del ajuste se muestra el experimento crudo
para elegir visualmente la region de baseline, el rango de lambdas de trabajo
y, si hace falta, eliminar espectros anomalos antes de continuar. Tambien se
puede indicar un tiempo de inicio de reaccion: se descartan los espectros con
t <= t_inicio y los tiempos restantes se reexpresan como t - t_inicio.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar, nnls

from kd_to_txt import convert_kd


MODEL_LABELS = {
    "a_to_b": "A -> B irreversible de primer orden",
    "a_to_b_to_c": "A -> B -> C irreversible consecutivo",
    "a_rev_b_to_c": "A <-> B -> C",
}

MODEL_SPECIES = {
    "a_to_b": ("A", "B"),
    "a_to_b_to_c": ("A", "B", "C"),
    "a_rev_b_to_c": ("A", "B", "C"),
}

PARAMETER_LABELS = {
    "k": "k",
    "k1": "k1",
    "k_1": "k-1",
    "k2": "k2",
}


@dataclass
class Experiment:
    """Datos experimentales en la convencion lambda x tiempo."""

    t: np.ndarray
    wavelength: np.ndarray
    absorbance: np.ndarray


@dataclass
class FitResult:
    """Resultado completo del ajuste y matrices utiles para diagnostico."""

    method: str
    model: str
    params: dict[str, float]
    species_labels: tuple[str, ...]
    c: np.ndarray
    spectra: np.ndarray
    absorbance_calc: np.ndarray
    residuals: np.ndarray
    singular_values: np.ndarray
    q: np.ndarray
    w: np.ndarray
    error: float


def _parse_float_cell(value: str, path: Path, line_number: int, column_number: int) -> float:
    """Parse one numeric cell from plain TXT or R write.csv2 output."""
    text = value.strip()
    if not text:
        raise ValueError(
            f"{path}: empty numeric value at line {line_number}, column {column_number}"
        )

    try:
        return float(text.replace(",", "."))
    except ValueError as exc:
        raise ValueError(
            f"{path}: cannot parse numeric value {value!r} "
            f"at line {line_number}, column {column_number}"
        ) from exc


def _read_numeric_table(path: Path, delimiter: str) -> np.ndarray:
    """Read a semicolon numeric table, including R write.csv2-style files."""
    rows: list[list[str]] = []
    with path.open(newline="", encoding="utf-8-sig") as file:
        for line_number, row in enumerate(csv.reader(file, delimiter=delimiter), start=1):
            if not row or all(not cell.strip() for cell in row):
                continue
            rows.append(row)

    if not rows:
        raise ValueError(f"{path}: file is empty")

    # R's write.csv2 writes an empty top-left header plus a row-name column.
    # For these files the first data row is duplicated as column names, so skip
    # the CSV header and remove the row-name column.
    has_r_row_names = rows[0][0].strip() == ""
    if has_r_row_names:
        rows = [row[1:] for row in rows[1:]]

    width = len(rows[0])
    if width < 2:
        raise ValueError(f"{path}: expected at least two columns")

    numeric_rows: list[list[float]] = []
    for line_offset, row in enumerate(rows, start=2 if has_r_row_names else 1):
        if len(row) != width:
            raise ValueError(
                f"{path}: line {line_offset} has {len(row)} columns, "
                f"expected {width}"
            )
        numeric_rows.append(
            [
                _parse_float_cell(value, path, line_offset, column_number)
                for column_number, value in enumerate(row, start=1)
            ]
        )

    return np.array(numeric_rows, dtype=float)


def read_experiment(path: str | Path, delimiter: str = ";") -> Experiment:
    """Read semicolon-separated spectrophotometry files.

    Supported layouts:

    - Plain TXT generated by kd_to_txt.py:
      0; t1; t2; ...
      lambda; A(lambda,t1); ...
    - R write.csv2 output with row names and comma decimals.
    """
    path = Path(path)
    data = _read_numeric_table(path, delimiter)
    t = data[0, 1:]
    wavelength = data[1:, 0]
    absorbance = data[1:, 1:]
    return Experiment(t=t, wavelength=wavelength, absorbance=absorbance)


def resolve_input_file(args: argparse.Namespace) -> Path:
    """Convert KD input files to text and return the file to analyze."""
    input_path = Path(args.input)
    if input_path.suffix.lower() != ".kd":
        return input_path

    output_path = input_path.with_name(f"{input_path.stem}-converted.txt")
    n_spectra, n_wavelengths, lambda_first, lambda_last = convert_kd(
        input_path,
        output_path,
        lambda_start=args.kd_lambda_start,
        lambda_step=args.kd_lambda_step,
    )

    print("KD file detected. Converted before analysis.")
    print(f"converted file: {output_path}")
    print(f"spectra / times: {n_spectra}")
    print(f"wavelength points: {n_wavelengths}")
    print(f"wavelength range: {lambda_first:g} - {lambda_last:g}")
    print()

    return output_path


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


def time_gradient_colors(n: int) -> np.ndarray:
    """Return colors from black to orange for n time-ordered spectra."""
    if n <= 1:
        return np.array([[0.0, 0.0, 0.0]])
    start = np.array([0.0, 0.0, 0.0])
    end = np.array([1.0, 0.45, 0.0])
    weights = np.linspace(0.0, 1.0, n)[:, None]
    return start + weights * (end - start)


def plot_time_colored_spectra(
    experiment: Experiment,
    title: str,
    ax=None,
    show_colorbar: bool = False,
    show_hover_labels: bool = False,
) -> None:
    """Plot A(lambda) spectra with time encoded as black-to-orange color."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.cm import ScalarMappable

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    colors = time_gradient_colors(experiment.t.size)
    for j, color in enumerate(colors):
        ax.plot(
            experiment.wavelength,
            experiment.absorbance[:, j],
            color=color,
            linewidth=0.45,
            alpha=0.9,
        )

    ax.set_xlabel("Wavelength")
    ax.set_ylabel("Absorbance")
    ax.set_title(title)

    if show_colorbar:
        cmap = LinearSegmentedColormap.from_list("time_black_orange", ["black", "orange"])
        norm = Normalize(vmin=float(experiment.t[0]), vmax=float(experiment.t[-1]))
        scalar = ScalarMappable(norm=norm, cmap=cmap)
        scalar.set_array([])
        fig.colorbar(scalar, ax=ax, label="Time")

    if show_hover_labels:
        add_spectrum_hover_labels(ax, experiment)


def add_spectrum_hover_labels(ax, experiment: Experiment) -> None:
    """Add hover labels that identify the nearest spectrum line."""
    y_min = float(np.nanmin(experiment.absorbance))
    y_max = float(np.nanmax(experiment.absorbance))
    y_tolerance = max((y_max - y_min) * 0.025, 1e-12)

    annotation = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(12, 12),
        textcoords="offset points",
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.4", "alpha": 0.9},
        arrowprops={"arrowstyle": "->", "color": "0.4"},
    )
    annotation.set_visible(False)

    def on_motion(event) -> None:
        if event.inaxes is not ax or event.xdata is None or event.ydata is None:
            annotation.set_visible(False)
            ax.figure.canvas.draw_idle()
            return

        wavelength_index = int(np.argmin(np.abs(experiment.wavelength - event.xdata)))
        y_values = experiment.absorbance[wavelength_index, :]
        spectrum_index = int(np.argmin(np.abs(y_values - event.ydata)))
        y_nearest = float(y_values[spectrum_index])

        if abs(y_nearest - event.ydata) > y_tolerance:
            annotation.set_visible(False)
            ax.figure.canvas.draw_idle()
            return

        x_nearest = float(experiment.wavelength[wavelength_index])
        annotation.xy = (x_nearest, y_nearest)
        annotation.set_text(
            f"Spectrum {spectrum_index + 1}\n"
            f"t = {experiment.t[spectrum_index]:.6g}"
        )
        annotation.set_visible(True)
        ax.figure.canvas.draw_idle()

    ax.figure.canvas.mpl_connect("motion_notify_event", on_motion)


def add_kinetic_direction_arrows(
    ax,
    experiment: Experiment,
    max_arrows: int = 4,
    min_separation_nm: float = 35.0,
    selected_indices: list[int] | None = None,
) -> None:
    """Mark wavelengths where spectra increase or decrease during the experiment."""
    if experiment.t.size < 2 or experiment.wavelength.size < 3:
        return

    delta = experiment.absorbance[:, -1] - experiment.absorbance[:, 0]
    abs_delta = np.abs(delta)
    max_delta = float(np.nanmax(abs_delta))
    if not np.isfinite(max_delta) or max_delta <= 0:
        return

    if selected_indices is None:
        local_maxima = np.where(
            (abs_delta[1:-1] >= abs_delta[:-2])
            & (abs_delta[1:-1] >= abs_delta[2:])
        )[0] + 1
        if local_maxima.size == 0:
            local_maxima = np.arange(experiment.wavelength.size)

        threshold = 0.15 * max_delta
        candidates = [
            index
            for index in local_maxima
            if abs_delta[index] >= threshold
        ]
        candidates.sort(key=lambda index: abs_delta[index], reverse=True)

        selected: list[int] = []
        for index in candidates:
            wavelength = experiment.wavelength[index]
            if all(
                abs(wavelength - experiment.wavelength[other]) >= min_separation_nm
                for other in selected
            ):
                selected.append(index)
            if len(selected) >= max_arrows:
                break
    else:
        selected = selected_indices[:max_arrows]

    if not selected:
        return

    y_min = float(np.nanmin(experiment.absorbance))
    y_max = float(np.nanmax(experiment.absorbance))
    y_range = max(y_max - y_min, 1e-12)
    arrow_length = 0.12 * y_range
    pad = 0.04 * y_range

    for index in selected:
        x = float(experiment.wavelength[index])
        band_top = float(np.nanmax(experiment.absorbance[index, :]))
        y_low = band_top + pad
        y_high = y_low + arrow_length
        if delta[index] > 0:
            xy = (x, y_high)
            xytext = (x, y_low)
        else:
            xy = (x, y_low)
            xytext = (x, y_high)

        ax.annotate(
            "",
            xy=xy,
            xytext=xytext,
            arrowprops={
                "arrowstyle": "-|>",
                "color": "black",
                "lw": 1.4,
                "mutation_scale": 13,
            },
            annotation_clip=False,
        )

    ax.set_ylim(y_min, y_max + 0.22 * y_range)


def find_isosbestic_points(experiment: Experiment) -> np.ndarray:
    """Estimate isosbestic wavelengths from first/last spectrum crossings."""
    if experiment.t.size < 2:
        return np.array([])

    wavelength = experiment.wavelength
    delta = experiment.absorbance[:, -1] - experiment.absorbance[:, 0]
    crossings: list[float] = []

    for i in range(delta.size - 1):
        y0 = delta[i]
        y1 = delta[i + 1]
        if y0 == 0:
            crossings.append(float(wavelength[i]))
        elif y0 * y1 < 0:
            fraction = abs(y0) / (abs(y0) + abs(y1))
            crossings.append(float(wavelength[i] + fraction * (wavelength[i + 1] - wavelength[i])))

    return np.array(crossings)


def add_two_species_direction_guides(ax, experiment: Experiment) -> None:
    """Add one direction arrow per region separated by estimated isosbestic points."""
    if experiment.t.size < 2 or experiment.wavelength.size < 3:
        return

    delta = experiment.absorbance[:, -1] - experiment.absorbance[:, 0]
    abs_delta = np.abs(delta)
    max_delta = float(np.nanmax(abs_delta))
    if not np.isfinite(max_delta) or max_delta <= 0:
        return

    isosbestic = find_isosbestic_points(experiment)
    boundaries = np.concatenate(
        (
            [float(experiment.wavelength[0])],
            isosbestic,
            [float(experiment.wavelength[-1])],
        )
    )

    selected: list[int] = []
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        mask = (experiment.wavelength >= lower) & (experiment.wavelength <= upper)
        indices = np.where(mask)[0]
        if indices.size == 0:
            continue

        best = int(indices[np.argmax(abs_delta[indices])])
        if abs_delta[best] < 0.15 * max_delta:
            continue
        selected.append(best)

    if selected:
        add_kinetic_direction_arrows(
            ax,
            experiment,
            max_arrows=len(selected),
            min_separation_nm=0.0,
            selected_indices=selected,
        )


def factor_analysis(absorbance: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Q, W and singular values from A = U S Vt, with A ~= Q W.

    Q is the truncated spectral basis. W contains the corresponding temporal
    profiles in factor-analysis coordinates.
    """
    u, singular_values, vt = np.linalg.svd(absorbance, full_matrices=False)
    q = u[:, :n_components]
    w = singular_values[:n_components, None] * vt[:n_components, :]
    return q, w, singular_values


def print_exploration_message() -> None:
    """Explain the exploratory figure shown before preprocessing choices."""
    print("An exploratory figure will open next.")
    print("Use it to inspect the raw spectra, choose a baseline region,")
    print("identify spectra to discard, choose a wavelength range,")
    print("and estimate the number of colored species from diag(S).")
    print("Close the figure window to continue with the questions.")
    print()


def plot_experiment_overview(experiment: Experiment) -> None:
    """Show raw spectra and singular values before preprocessing choices."""
    import sys

    import matplotlib.pyplot as plt

    _, _, singular_values = factor_analysis(
        experiment.absorbance,
        min(experiment.absorbance.shape),
    )

    fig, (ax_spectra, ax_singular) = plt.subplots(
        1,
        2,
        figsize=(14, 5),
        constrained_layout=True,
    )
    fig.suptitle("Experiment overview before preprocessing")

    plot_time_colored_spectra(
        experiment,
        "Raw spectra A(lambda, t)",
        ax=ax_spectra,
        show_hover_labels=True,
    )

    component_indices = np.arange(1, singular_values.size + 1)
    ax_singular.plot(component_indices, singular_values, marker="o", linestyle="none")
    ax_singular.set_xlabel("Component index")
    ax_singular.set_ylabel("Singular value")
    ax_singular.set_title("Factor analysis: diag(S)")

    sys.stdout.flush()
    plt.show()


def plot_baseline_comparison(
    raw: Experiment,
    corrected: Experiment,
    baseline_region: tuple[float, float],
) -> None:
    """Show raw and baseline-corrected spectra side by side."""
    import sys

    import matplotlib.pyplot as plt

    fig, (ax_raw, ax_corrected) = plt.subplots(
        1,
        2,
        figsize=(14, 5),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    fig.suptitle(
        f"Baseline correction preview: {baseline_region[0]:g}-{baseline_region[1]:g} nm"
    )

    plot_time_colored_spectra(raw, "Original spectra", ax=ax_raw)
    plot_time_colored_spectra(corrected, "Baseline-corrected spectra", ax=ax_corrected)
    for ax in (ax_raw, ax_corrected):
        ax.axvspan(
            baseline_region[0],
            baseline_region[1],
            color="0.85",
            alpha=0.45,
            zorder=0,
        )

    sys.stdout.flush()
    plt.show()


def concentration_profile_a_to_b(t: np.ndarray, k: float, c0: float = 1.0) -> np.ndarray:
    """Concentration profiles for A -> B as a 2 x n_times matrix."""
    a = c0 * np.exp(-k * t)
    b = c0 - a
    return np.vstack([a, b])


def concentration_profile_a_to_b_to_c(
    t: np.ndarray,
    k1: float,
    k2: float,
    c0: float = 1.0,
) -> np.ndarray:
    """Concentration profiles for A -> B -> C as a 3 x n_times matrix."""
    if k1 <= 0 or k2 <= 0:
        raise ValueError("All kinetic constants must be positive")

    a = c0 * np.exp(-k1 * t)
    if np.isclose(k1, k2, rtol=1e-10, atol=0.0):
        b = c0 * k1 * t * np.exp(-k1 * t)
    else:
        b = c0 * k1 / (k2 - k1) * (np.exp(-k1 * t) - np.exp(-k2 * t))
    c = c0 - a - b
    return np.vstack([a, b, c])


def concentration_profile_a_rev_b_to_c(
    t: np.ndarray,
    k1: float,
    k_1: float,
    k2: float,
    c0: float = 1.0,
) -> np.ndarray:
    """Concentration profiles for A <-> B -> C as a 3 x n_times matrix."""
    if k1 <= 0 or k_1 <= 0 or k2 <= 0:
        raise ValueError("All kinetic constants must be positive")

    rate_matrix = np.array(
        [
            [-k1, k_1, 0.0],
            [k1, -(k_1 + k2), 0.0],
            [0.0, k2, 0.0],
        ]
    )
    initial = np.array([c0, 0.0, 0.0])
    eigenvalues, eigenvectors = np.linalg.eig(rate_matrix)
    weights = np.linalg.solve(eigenvectors, initial)
    c = eigenvectors @ (weights[:, None] * np.exp(eigenvalues[:, None] * t[None, :]))
    return np.real_if_close(c, tol=1000).real


def factor_space_error(k: float, t: np.ndarray, w: np.ndarray, c0: float) -> float:
    """Equivalent to terror.m for the two-component A -> B model.

    This compares calculated concentration profiles against the experimental
    factor-analysis profiles W. It is kept mainly as a compatibility path with
    the MATLAB routine.
    """
    if k <= 0:
        return np.inf

    c = concentration_profile_a_to_b(t, k, c0=c0)
    r = c @ np.linalg.pinv(w)
    w_calc = np.linalg.solve(r, c)
    diff = (w - w_calc).T
    return np.linalg.norm(diff[:, 0]) + np.linalg.norm(diff[:, 1])


def fit_nonnegative_spectra(
    absorbance: np.ndarray,
    c: np.ndarray,
    initial_spectrum_weight: float = 0.0,
) -> np.ndarray:
    """Fit nonnegative pure spectra for fixed concentration profiles.

    For fixed C(t), each wavelength is an independent least-squares problem:

        A_exp(lambda, :) ~= E(lambda, :) @ C

    nnls enforces E(lambda, species) >= 0.

    If initial_spectrum_weight > 0, add a soft penalty that keeps the pure
    spectrum of species A close to the first measured spectrum. The weight is
    equivalent to adding that many extra time points to the least-squares fit.
    """
    if initial_spectrum_weight < 0:
        raise ValueError("--initial-spectrum-weight must be nonnegative")

    design = c.T
    target = absorbance
    if initial_spectrum_weight > 0:
        penalty_row = np.zeros((1, c.shape[0]))
        penalty_row[0, 0] = np.sqrt(initial_spectrum_weight) * c[0, 0]
        design = np.vstack([design, penalty_row])
        target = np.column_stack(
            [
                absorbance,
                np.sqrt(initial_spectrum_weight) * absorbance[:, 0],
            ]
        )

    spectra = np.empty((absorbance.shape[0], c.shape[0]))
    for i, row in enumerate(target):
        spectra[i, :], _ = nnls(design, row)
    return spectra


def direct_nonnegative_error_for_concentrations(
    c: np.ndarray,
    experiment: Experiment,
    initial_spectrum_weight: float = 0.0,
) -> float:
    """Objective function for a fixed concentration matrix."""
    spectra = fit_nonnegative_spectra(
        experiment.absorbance,
        c,
        initial_spectrum_weight=initial_spectrum_weight,
    )
    residuals = experiment.absorbance - spectra @ c
    error_squared = float(np.sum(residuals**2))
    if initial_spectrum_weight > 0:
        initial_mismatch = c[0, 0] * spectra[:, 0] - experiment.absorbance[:, 0]
        error_squared += initial_spectrum_weight * float(np.sum(initial_mismatch**2))
    return float(np.sqrt(error_squared))


def direct_nonnegative_error(
    k: float,
    experiment: Experiment,
    c0: float,
    initial_spectrum_weight: float = 0.0,
) -> float:
    """Objective function for the A -> B NNLS fit."""
    if k <= 0:
        return np.inf

    c = concentration_profile_a_to_b(experiment.t, k, c0=c0)
    return direct_nonnegative_error_for_concentrations(
        c,
        experiment,
        initial_spectrum_weight=initial_spectrum_weight,
    )


def validate_k_bounds(k_bounds: tuple[float, float]) -> tuple[float, float]:
    """Validate and normalize positive bounds for the kinetic constant."""
    lower, upper = sorted(k_bounds)
    if lower <= 0 or upper <= 0:
        raise ValueError("k bounds must be positive")
    if lower == upper:
        raise ValueError("k bounds must not be equal")
    return lower, upper


def optimize_k(
    objective_for_k,
    k_bounds: tuple[float, float],
) -> float:
    """Minimize an objective as a function of log(k) within positive bounds."""
    k_min, k_max = validate_k_bounds(k_bounds)

    def objective(log_k: float) -> float:
        return objective_for_k(float(np.exp(log_k)))

    opt = minimize_scalar(
        objective,
        bounds=(np.log(k_min), np.log(k_max)),
        method="bounded",
        options={"xatol": 1e-12, "maxiter": 10000},
    )
    if not opt.success:
        raise RuntimeError(f"k optimization failed: {opt.message}")

    return float(np.exp(opt.x))


def optimize_kinetic_parameters(
    objective_for_params,
    parameter_names: tuple[str, ...],
    k_bounds: tuple[float, float],
) -> dict[str, float]:
    """Minimize an objective as a function of multiple positive constants."""
    k_min, k_max = validate_k_bounds(k_bounds)
    log_bounds = [(np.log(k_min), np.log(k_max))] * len(parameter_names)
    lower = np.array([bound[0] for bound in log_bounds])
    upper = np.array([bound[1] for bound in log_bounds])

    center = lower + 0.5 * (upper - lower)
    starts = [center]
    for i in range(len(parameter_names)):
        for fraction in (0.25, 0.75):
            start = center.copy()
            start[i] = lower[i] + fraction * (upper[i] - lower[i])
            starts.append(start)

    best = None

    def objective(log_params: np.ndarray) -> float:
        params = {
            name: float(np.exp(value))
            for name, value in zip(parameter_names, log_params)
        }
        return objective_for_params(params)

    for start in starts:
        clipped_start = np.clip(start, lower, upper)
        opt = minimize(
            objective,
            clipped_start,
            method="Nelder-Mead",
            bounds=log_bounds,
            options={"xatol": 1e-10, "fatol": 1e-8, "maxiter": 3000},
        )
        if best is None or opt.fun < best.fun:
            best = opt

    if best is None or not best.success:
        message = "unknown error" if best is None else best.message
        raise RuntimeError(f"k optimization failed: {message}")

    return {
        name: float(np.exp(value))
        for name, value in zip(parameter_names, best.x)
    }


def fit_a_to_b_factor(
    experiment: Experiment,
    c0: float = 1.0,
    n_components: int = 2,
    k_bounds: tuple[float, float] = (1e-8, 1e-1),
) -> FitResult:
    """Fit k using the original factor-analysis objective."""
    q, w, singular_values = factor_analysis(experiment.absorbance, n_components)

    if n_components != 2:
        raise ValueError("The A -> B factor-space fit requires exactly 2 components.")

    k = optimize_k(
        lambda trial_k: factor_space_error(trial_k, experiment.t, w, c0),
        k_bounds,
    )
    c = concentration_profile_a_to_b(experiment.t, k, c0=c0)
    r = c @ np.linalg.pinv(w)
    spectra = q @ np.linalg.inv(r)
    absorbance_calc = spectra @ c
    residuals = experiment.absorbance - absorbance_calc

    return FitResult(
        method="factor",
        model="a_to_b",
        params={"k": k},
        species_labels=MODEL_SPECIES["a_to_b"],
        c=c,
        spectra=spectra,
        absorbance_calc=absorbance_calc,
        residuals=residuals,
        singular_values=singular_values,
        q=q,
        w=w,
        error=factor_space_error(k, experiment.t, w, c0),
    )


def fit_a_to_b_nnls(
    experiment: Experiment,
    c0: float = 1.0,
    n_components: int = 2,
    k_bounds: tuple[float, float] = (1e-8, 1e-1),
    initial_spectrum_weight: float = 0.0,
) -> FitResult:
    """Fit k by direct reconstruction with nonnegative spectra."""
    q, w, singular_values = factor_analysis(experiment.absorbance, n_components)

    if n_components != 2:
        raise ValueError("The A -> B NNLS fit requires exactly 2 components.")

    k = optimize_k(
        lambda trial_k: direct_nonnegative_error(
            trial_k,
            experiment,
            c0,
            initial_spectrum_weight=initial_spectrum_weight,
        ),
        k_bounds,
    )
    c = concentration_profile_a_to_b(experiment.t, k, c0=c0)
    spectra = fit_nonnegative_spectra(
        experiment.absorbance,
        c,
        initial_spectrum_weight=initial_spectrum_weight,
    )
    absorbance_calc = spectra @ c
    residuals = experiment.absorbance - absorbance_calc
    error = direct_nonnegative_error_for_concentrations(
        c,
        experiment,
        initial_spectrum_weight=initial_spectrum_weight,
    )

    return FitResult(
        method="nnls",
        model="a_to_b",
        params={"k": k},
        species_labels=MODEL_SPECIES["a_to_b"],
        c=c,
        spectra=spectra,
        absorbance_calc=absorbance_calc,
        residuals=residuals,
        singular_values=singular_values,
        q=q,
        w=w,
        error=error,
    )


def fit_a_to_b(
    experiment: Experiment,
    c0: float = 1.0,
    n_components: int = 2,
    method: str = "nnls",
    k_bounds: tuple[float, float] = (1e-8, 1e-1),
    initial_spectrum_weight: float = 0.0,
) -> FitResult:
    """Dispatch between available A -> B fitting methods."""
    if method == "nnls":
        return fit_a_to_b_nnls(
            experiment,
            c0=c0,
            n_components=n_components,
            k_bounds=k_bounds,
            initial_spectrum_weight=initial_spectrum_weight,
        )
    if method == "factor":
        return fit_a_to_b_factor(
            experiment,
            c0=c0,
            n_components=n_components,
            k_bounds=k_bounds,
        )
    raise ValueError(f"Unknown fit method: {method}")


def fit_a_to_b_to_c_nnls(
    experiment: Experiment,
    c0: float = 1.0,
    n_components: int = 3,
    k_bounds: tuple[float, float] = (1e-8, 1e-1),
    initial_spectrum_weight: float = 0.0,
) -> FitResult:
    """Fit A -> B -> C by direct reconstruction with nonnegative spectra."""
    q, w, singular_values = factor_analysis(experiment.absorbance, n_components)

    if n_components != 3:
        raise ValueError("The A -> B -> C NNLS fit requires exactly 3 components.")

    def objective(params: dict[str, float]) -> float:
        c = concentration_profile_a_to_b_to_c(
            experiment.t,
            params["k1"],
            params["k2"],
            c0=c0,
        )
        return direct_nonnegative_error_for_concentrations(
            c,
            experiment,
            initial_spectrum_weight=initial_spectrum_weight,
        )

    params = optimize_kinetic_parameters(
        objective,
        ("k1", "k2"),
        k_bounds,
    )
    c = concentration_profile_a_to_b_to_c(
        experiment.t,
        params["k1"],
        params["k2"],
        c0=c0,
    )
    spectra = fit_nonnegative_spectra(
        experiment.absorbance,
        c,
        initial_spectrum_weight=initial_spectrum_weight,
    )
    absorbance_calc = spectra @ c
    residuals = experiment.absorbance - absorbance_calc
    error = direct_nonnegative_error_for_concentrations(
        c,
        experiment,
        initial_spectrum_weight=initial_spectrum_weight,
    )

    return FitResult(
        method="nnls",
        model="a_to_b_to_c",
        params=params,
        species_labels=MODEL_SPECIES["a_to_b_to_c"],
        c=c,
        spectra=spectra,
        absorbance_calc=absorbance_calc,
        residuals=residuals,
        singular_values=singular_values,
        q=q,
        w=w,
        error=error,
    )


def fit_a_rev_b_to_c_nnls(
    experiment: Experiment,
    c0: float = 1.0,
    n_components: int = 3,
    k_bounds: tuple[float, float] = (1e-8, 1e-1),
    initial_spectrum_weight: float = 0.0,
) -> FitResult:
    """Fit A <-> B -> C by direct reconstruction with nonnegative spectra."""
    q, w, singular_values = factor_analysis(experiment.absorbance, n_components)

    if n_components != 3:
        raise ValueError("The A <-> B -> C NNLS fit requires exactly 3 components.")

    def objective(params: dict[str, float]) -> float:
        c = concentration_profile_a_rev_b_to_c(
            experiment.t,
            params["k1"],
            params["k_1"],
            params["k2"],
            c0=c0,
        )
        return direct_nonnegative_error_for_concentrations(
            c,
            experiment,
            initial_spectrum_weight=initial_spectrum_weight,
        )

    params = optimize_kinetic_parameters(
        objective,
        ("k1", "k_1", "k2"),
        k_bounds,
    )
    c = concentration_profile_a_rev_b_to_c(
        experiment.t,
        params["k1"],
        params["k_1"],
        params["k2"],
        c0=c0,
    )
    spectra = fit_nonnegative_spectra(
        experiment.absorbance,
        c,
        initial_spectrum_weight=initial_spectrum_weight,
    )
    absorbance_calc = spectra @ c
    residuals = experiment.absorbance - absorbance_calc
    error = direct_nonnegative_error_for_concentrations(
        c,
        experiment,
        initial_spectrum_weight=initial_spectrum_weight,
    )

    return FitResult(
        method="nnls",
        model="a_rev_b_to_c",
        params=params,
        species_labels=MODEL_SPECIES["a_rev_b_to_c"],
        c=c,
        spectra=spectra,
        absorbance_calc=absorbance_calc,
        residuals=residuals,
        singular_values=singular_values,
        q=q,
        w=w,
        error=error,
    )


def fit_model(
    model: str,
    experiment: Experiment,
    c0: float,
    n_components: int,
    method: str,
    k_bounds: tuple[float, float],
    initial_spectrum_weight: float = 0.0,
) -> FitResult:
    """Fit the selected kinetic model."""
    if initial_spectrum_weight < 0:
        raise ValueError("--initial-spectrum-weight must be nonnegative")
    if initial_spectrum_weight > 0 and method != "nnls":
        raise ValueError("--initial-spectrum-weight is available only with --fit-method nnls")

    if model == "a_to_b":
        return fit_a_to_b(
            experiment,
            c0=c0,
            n_components=n_components,
            method=method,
            k_bounds=k_bounds,
            initial_spectrum_weight=initial_spectrum_weight,
        )
    if model == "a_rev_b_to_c":
        if method != "nnls":
            raise ValueError("A <-> B -> C is currently implemented only for --fit-method nnls")
        return fit_a_rev_b_to_c_nnls(
            experiment,
            c0=c0,
            n_components=n_components,
            k_bounds=k_bounds,
            initial_spectrum_weight=initial_spectrum_weight,
        )
    if model == "a_to_b_to_c":
        if method != "nnls":
            raise ValueError("A -> B -> C is currently implemented only for --fit-method nnls")
        return fit_a_to_b_to_c_nnls(
            experiment,
            c0=c0,
            n_components=n_components,
            k_bounds=k_bounds,
            initial_spectrum_weight=initial_spectrum_weight,
        )
    raise ValueError(f"Unknown kinetic model: {model}")


def print_startup_banner() -> None:
    """Print the program title and short description."""
    width = 72
    lines = [
        "TolKinet",
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


def is_near_bound(value: float, bound: float) -> bool:
    """Return whether value is numerically close to a search bound."""
    return np.isclose(np.log(value), np.log(bound), atol=1e-5, rtol=0.0)


def parameters_near_upper_bound(result: FitResult, k_max: float) -> list[str]:
    """Return fitted parameters that are effectively at the upper search bound."""
    return [
        name
        for name, value in result.params.items()
        if is_near_bound(value, k_max)
    ]


def fit_model_with_auto_k_max(
    model: str,
    experiment: Experiment,
    c0: float,
    n_components: int,
    method: str,
    k_bounds: tuple[float, float],
    auto_expand: bool,
    expand_factor: float,
    max_expand_steps: int,
    initial_spectrum_weight: float = 0.0,
) -> tuple[FitResult, tuple[float, float]]:
    """Fit a model, expanding the upper k bound when fitted constants hit it."""
    k_min, k_max = validate_k_bounds(k_bounds)
    if max_expand_steps < 0:
        raise ValueError("--k-max-expand-steps must be nonnegative")
    if auto_expand and expand_factor <= 1:
        raise ValueError("--k-max-expand-factor must be greater than 1")

    result = fit_model(
        model,
        experiment,
        c0=c0,
        n_components=n_components,
        method=method,
        k_bounds=(k_min, k_max),
        initial_spectrum_weight=initial_spectrum_weight,
    )

    if not auto_expand:
        return result, (k_min, k_max)

    for _ in range(max_expand_steps):
        bounded_names = parameters_near_upper_bound(result, k_max)
        if not bounded_names:
            break

        next_k_max = k_max * expand_factor
        if not np.isfinite(next_k_max):
            raise ValueError("Expanded k maximum is not finite")

        labels = ", ".join(PARAMETER_LABELS.get(name, name) for name in bounded_names)
        print(
            f"{labels} reached upper search bound ({k_max:g}); "
            f"expanding k max to {next_k_max:g} and refitting."
        )
        k_max = next_k_max
        result = fit_model(
            model,
            experiment,
            c0=c0,
            n_components=n_components,
            method=method,
            k_bounds=(k_min, k_max),
            initial_spectrum_weight=initial_spectrum_weight,
        )

    return result, (k_min, k_max)


def plot_result(
    experiment: Experiment,
    result: FitResult,
    input_filename: str | None = None,
) -> None:
    """Create diagnostic plots for visual inspection of the fit."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(17, 11), constrained_layout=True)
    axes = fig.subplot_mosaic(
        [
            ["spectra", "spectra", "pure"],
            ["fit_overlay", "fit_overlay", "residuals"],
            ["concentrations", "singular", "residuals"],
        ]
    )
    params_text = ", ".join(
        f"{PARAMETER_LABELS.get(name, name)} = {value:.6g}"
        for name, value in result.params.items()
    )
    if input_filename is None:
        title = f"Kinetic fit ({result.method}), {params_text}"
    else:
        title = f"{input_filename} | Kinetic fit ({result.method}), {params_text}"
    fig.suptitle(title)

    plot_time_colored_spectra(
        experiment,
        "Baseline-corrected spectra",
        ax=axes["spectra"],
    )
    if result.model == "a_to_b":
        add_two_species_direction_guides(axes["spectra"], experiment)

    ax_overlay = axes["fit_overlay"]
    ax_overlay.plot(
        experiment.wavelength,
        experiment.absorbance,
        color="black",
        linewidth=0.45,
        alpha=0.35,
    )
    ax_overlay.plot(
        experiment.wavelength,
        result.absorbance_calc,
        color="red",
        linewidth=0.45,
        alpha=0.35,
    )
    ax_overlay.plot([], [], color="black", label="Abs_exp(lambda,t)")
    ax_overlay.plot([], [], color="red", label="A_calc(lambda,t)")
    ax_overlay.set_xlabel("Wavelength")
    ax_overlay.set_ylabel("Absorbance")
    ax_overlay.set_title("Experimental vs reconstructed spectra")
    ax_overlay.legend()

    ax_pure = axes["pure"]
    for i, label in enumerate(result.species_labels):
        ax_pure.plot(experiment.wavelength, result.spectra[:, i], label=label)
    ax_pure.set_xlabel("Wavelength")
    ax_pure.set_ylabel("epsilon(lambda)")
    ax_pure.set_title("Recovered epsilon spectra")
    ax_pure.legend()

    ax_conc = axes["concentrations"]
    for i, label in enumerate(result.species_labels):
        ax_conc.plot(experiment.t, result.c[i, :], label=label)
    ax_conc.set_xlabel("Time")
    ax_conc.set_ylabel("Concentration (M)")
    ax_conc.set_title("Concentration profiles")
    ax_conc.legend()

    ax_residuals = axes["residuals"]
    ax_residuals.plot(experiment.wavelength, result.residuals)
    ax_residuals.set_xlabel("Wavelength")
    ax_residuals.set_ylabel("Residual absorbance")
    ax_residuals.set_title("Residuals")

    ax_singular = axes["singular"]
    component_indices = np.arange(1, result.singular_values.size + 1)
    ax_singular.plot(component_indices, result.singular_values, marker="o", linestyle="none")
    ax_singular.set_xlabel("Component index")
    ax_singular.set_ylabel("Singular value")
    ax_singular.set_title("Singular values")

    plt.show()


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
        choices=("nnls", "factor"),
        default="nnls",
        help="nnls enforces nonnegative spectra; factor reproduces the MATLAB factor-space fit",
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
    auto_corrected = Experiment(
        t=dropped.t,
        wavelength=dropped.wavelength,
        absorbance=baseline_correct_region(dropped, auto_region[0], auto_region[1]),
    )

    print()
    print(
        "Automatic baseline suggestion: "
        f"{auto_region[0]:g}-{auto_region[1]:g} nm"
    )
    print("A comparison figure will open next.")
    print("Close it to accept the suggestion or choose another option.")
    print()
    plot_baseline_comparison(dropped, auto_corrected, auto_region)

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

    if not args.no_plot and not args.skip_preprocess_dialog:
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
            default_c0=args.c0,
            baseline_window=args.baseline_window,
            default_reaction_start_time=args.reaction_start_time,
        )

    experiment = drop_spectra(experiment, drop_indices)
    experiment = start_reaction_at_time(experiment, reaction_start_time)

    if baseline_mode == "none":
        corrected_absorbance = experiment.absorbance.copy()
    elif baseline_mode == "points":
        corrected_absorbance = baseline_correct(
            experiment.absorbance, baseline_points
        )
    elif baseline_mode == "auto-flat":
        baseline_region = suggest_auto_baseline_region(
            experiment,
            window_width=args.baseline_window,
        )
        print(
            "Automatic baseline region: "
            f"{baseline_region[0]:g}-{baseline_region[1]:g} nm"
        )
        corrected_absorbance = baseline_correct_region(
            experiment, baseline_region[0], baseline_region[1]
        )
    elif baseline_mode == "region":
        if baseline_region is None:
            raise ValueError("Baseline mode 'region' requires a baseline region")
        corrected_absorbance = baseline_correct_region(
            experiment, baseline_region[0], baseline_region[1]
        )
    else:
        raise ValueError(f"Unknown baseline mode: {baseline_mode}")

    corrected = Experiment(
        t=experiment.t,
        wavelength=experiment.wavelength,
        absorbance=corrected_absorbance,
    )
    return corrected, work_range, c0, reaction_start_time


def main() -> None:
    """Load data, preprocess, fit the kinetic model and report results."""
    args = build_parser().parse_args()

    print_startup_banner()
    input_path = resolve_input_file(args)
    experiment = read_experiment(input_path)

    model = args.model
    if not args.no_plot and not args.skip_preprocess_dialog:
        print_exploration_message()
        plot_experiment_overview(experiment)
        model = ask_model_choice(args.model)

    corrected, work_range, c0, reaction_start_time = preprocess_experiment(args, experiment)
    cropped = crop_wavelengths(corrected, work_range[0], work_range[1])
    n_components = args.components
    if n_components is None:
        n_components = len(MODEL_SPECIES[model])

    result, final_k_bounds = fit_model_with_auto_k_max(
        model,
        cropped,
        c0=c0,
        n_components=n_components,
        method=args.fit_method,
        k_bounds=(args.k_min, args.k_max),
        auto_expand=args.auto_expand_k_max,
        expand_factor=args.k_max_expand_factor,
        max_expand_steps=args.k_max_expand_steps,
        initial_spectrum_weight=args.initial_spectrum_weight,
    )
    k_min, k_max = final_k_bounds

    print(f"Input file: {input_path}")
    print(f"Data matrix: {cropped.absorbance.shape[0]} wavelengths x {cropped.absorbance.shape[1]} times")
    print(f"Wavelength range: {cropped.wavelength[0]:g} - {cropped.wavelength[-1]:g}")
    print(f"Time range: {cropped.t[0]:g} - {cropped.t[-1]:g}")
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
            print(f"  warning: {PARAMETER_LABELS.get(name, name)} reached the lower search bound")
        elif is_near_bound(value, k_max):
            print(f"  warning: {PARAMETER_LABELS.get(name, name)} reached the upper search bound")
    if set(result.params) == {"k"}:
        print(f"half-life: {np.log(2) / result.params['k']:.10g}")
    print(f"fit error: {result.error:.10g}")
    print(f"minimum recovered spectrum value: {result.spectra.min():.10g}")
    print("first singular values:", " ".join(f"{x:.6g}" for x in result.singular_values[:10]))

    if not args.no_plot:
        plot_result(cropped, result, input_filename=input_path.name)


if __name__ == "__main__":
    main()
