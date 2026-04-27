#!/usr/bin/env python3
"""Plotting helpers for TolKinet."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from kinet_common import Experiment, FitResult, PARAMETER_LABELS
from kinet_linalg import factor_analysis


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
    add_matrix_hover_labels(
        ax,
        x=experiment.wavelength,
        y_matrix=experiment.absorbance,
        times=experiment.t,
        value_label=None,
    )



def add_matrix_hover_labels(
    ax,
    x: np.ndarray,
    y_matrix: np.ndarray,
    times: np.ndarray,
    value_label: str | None = None,
) -> None:
    """Add hover labels that identify the nearest column trace."""
    y_min = float(np.nanmin(y_matrix))
    y_max = float(np.nanmax(y_matrix))
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

        wavelength_index = int(np.argmin(np.abs(x - event.xdata)))
        y_values = y_matrix[wavelength_index, :]
        spectrum_index = int(np.argmin(np.abs(y_values - event.ydata)))
        y_nearest = float(y_values[spectrum_index])

        if abs(y_nearest - event.ydata) > y_tolerance:
            annotation.set_visible(False)
            ax.figure.canvas.draw_idle()
            return

        x_nearest = float(x[wavelength_index])
        annotation.xy = (x_nearest, y_nearest)
        lines = [
            f"Spectrum {spectrum_index + 1}",
            f"t = {times[spectrum_index]:.6g}",
        ]
        if value_label is not None:
            lines.append(f"{value_label} = {y_nearest:.6g}")
        annotation.set_text("\n".join(lines))
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



def print_exploration_message() -> None:
    """Explain the exploratory figure shown before preprocessing choices."""
    print("An exploratory figure will open next.")
    print("Use it to inspect the raw spectra, choose a baseline region,")
    print("identify spectra to discard, choose a wavelength range,")
    print("estimate the number of colored species from diag(S),")
    print("and choose how pure spectra will be estimated during the fit.")
    print("The same figure includes an automatic baseline-correction preview.")
    print("Close the figure window to continue with the questions.")
    print()



def plot_experiment_overview(
    experiment: Experiment,
    baseline_window: float,
) -> None:
    """Show raw spectra, baseline preview and singular values before choices."""
    import sys

    import matplotlib.pyplot as plt
    from kinet_preprocessing import baseline_correct_region, suggest_auto_baseline_region

    _, _, singular_values = factor_analysis(
        experiment.absorbance,
        min(experiment.absorbance.shape),
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(15, 9),
        constrained_layout=True,
    )
    fig.suptitle("Experiment overview before preprocessing")
    ax_spectra = axes[0, 0]
    ax_baseline = axes[0, 1]
    ax_singular = axes[1, 0]
    ax_empty = axes[1, 1]

    plot_time_colored_spectra(
        experiment,
        "Raw spectra A(lambda, t)",
        ax=ax_spectra,
        show_hover_labels=True,
    )

    try:
        auto_region = suggest_auto_baseline_region(
            experiment,
            window_width=baseline_window,
        )
        corrected = Experiment(
            t=experiment.t,
            wavelength=experiment.wavelength,
            absorbance=baseline_correct_region(
                experiment,
                auto_region[0],
                auto_region[1],
            ),
        )
        plot_time_colored_spectra(
            corrected,
            (
                "Automatic baseline preview "
                f"({auto_region[0]:g}-{auto_region[1]:g} nm)"
            ),
            ax=ax_baseline,
            show_hover_labels=True,
        )
        ax_baseline.axvspan(
            auto_region[0],
            auto_region[1],
            color="0.85",
            alpha=0.45,
            zorder=0,
        )
    except ValueError as exc:
        ax_baseline.text(
            0.5,
            0.5,
            f"Automatic baseline preview unavailable:\n{exc}",
            ha="center",
            va="center",
            transform=ax_baseline.transAxes,
        )
        ax_baseline.set_axis_off()

    component_indices = np.arange(1, singular_values.size + 1)
    ax_singular.plot(component_indices, singular_values, marker="o", linestyle="none")
    ax_singular.set_xlabel("Component index")
    ax_singular.set_ylabel("Singular value")
    ax_singular.set_title("Factor analysis: diag(S)")

    ax_empty.text(
        0.0,
        1.0,
        "Use this overview to choose:\n"
        "- spectra to discard\n"
        "- reaction start time\n"
        "- baseline correction mode/region\n"
        "- wavelength range for the fit\n\n"
        "Hover over spectra in the spectral panels to read spectrum number and time.",
        ha="left",
        va="top",
        transform=ax_empty.transAxes,
    )
    ax_empty.set_axis_off()

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



def plot_preprocessed_experiment_preview(
    corrected: Experiment,
    cropped: Experiment,
    work_range: tuple[float, float],
    baseline_mode: str,
    baseline_region: tuple[float, float] | None,
    baseline_points: int,
    reaction_start_time: float | None,
) -> None:
    """Show the spectra that will be passed to the kinetic fit."""
    import sys

    import matplotlib.pyplot as plt

    _, _, singular_values = factor_analysis(
        cropped.absorbance,
        min(cropped.absorbance.shape),
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(17, 5),
        constrained_layout=True,
    )
    baseline_text = baseline_mode
    if baseline_region is not None:
        baseline_text += f" {baseline_region[0]:g}-{baseline_region[1]:g} nm"
    elif baseline_mode == "points":
        baseline_text += f" last {baseline_points} points"
    if reaction_start_time is None:
        start_text = "no reaction-start cut"
    else:
        start_text = f"t_start = {reaction_start_time:g}"
    fig.suptitle(
        "Preprocessed spectra proposed for fitting | "
        f"baseline: {baseline_text} | {start_text}"
    )

    plot_time_colored_spectra(
        corrected,
        "After baseline correction",
        ax=axes[0],
        show_hover_labels=True,
    )
    axes[0].axvspan(
        work_range[0],
        work_range[1],
        color="0.9",
        alpha=0.35,
        zorder=0,
    )
    if baseline_region is not None:
        axes[0].axvspan(
            baseline_region[0],
            baseline_region[1],
            color="tab:blue",
            alpha=0.12,
            zorder=0,
        )

    plot_time_colored_spectra(
        cropped,
        "Final spectra used for fit",
        ax=axes[1],
        show_hover_labels=True,
    )

    component_indices = np.arange(1, singular_values.size + 1)
    axes[2].plot(component_indices, singular_values, marker="o", linestyle="none")
    axes[2].set_xlabel("Component index")
    axes[2].set_ylabel("Singular value")
    axes[2].set_title("diag(S) after preprocessing")

    sys.stdout.flush()
    plt.show()



def plot_result(
    experiment: Experiment,
    result: FitResult,
    input_filename: str | None = None,
    save_path: Path | None = None,
    show: bool = True,
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
    residual_colors = time_gradient_colors(experiment.t.size)
    for j, color in enumerate(residual_colors):
        ax_residuals.plot(
            experiment.wavelength,
            result.residuals[:, j],
            color=color,
            linewidth=0.45,
            alpha=0.9,
        )
    ax_residuals.set_xlabel("Wavelength")
    ax_residuals.set_ylabel("Residual absorbance")
    ax_residuals.set_title("Residuals")
    add_matrix_hover_labels(
        ax_residuals,
        x=experiment.wavelength,
        y_matrix=result.residuals,
        times=experiment.t,
        value_label="residual",
    )

    ax_singular = axes["singular"]
    component_indices = np.arange(1, result.singular_values.size + 1)
    ax_singular.plot(component_indices, result.singular_values, marker="o", linestyle="none")
    ax_singular.set_xlabel("Component index")
    ax_singular.set_ylabel("Singular value")
    ax_singular.set_title("Singular values")

    if save_path is not None:
        fig.savefig(save_path, dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig)
