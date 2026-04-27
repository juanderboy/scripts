#!/usr/bin/env python3
"""Global kinetic fitting routines for TolKinet."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, minimize_scalar, nnls

from kinet_common import Experiment, FitResult, MODEL_SPECIES, PARAMETER_LABELS
from kinet_linalg import factor_analysis
from kinet_models import (
    concentration_profile_a_rev_b_to_c,
    concentration_profile_a_to_b,
    concentration_profile_a_to_b_to_c,
)


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


def factor_space_error_for_concentrations(c: np.ndarray, w: np.ndarray) -> float:
    """Compare kinetic concentration profiles against SVD temporal factors."""
    if c.shape != w.shape:
        raise ValueError("C and W must have the same shape for factor-space fitting")

    try:
        r = c @ np.linalg.pinv(w)
        w_calc = np.linalg.solve(r, c)
    except np.linalg.LinAlgError:
        return np.inf

    diff = w - w_calc
    return float(sum(np.linalg.norm(diff[i, :]) for i in range(diff.shape[0])))


def factor_spectra_from_concentrations(q: np.ndarray, w: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Recover pure spectra from SVD basis and fixed concentration profiles."""
    if c.shape != w.shape:
        raise ValueError("C and W must have the same shape for factor-space fitting")
    r = c @ np.linalg.pinv(w)
    return q @ np.linalg.inv(r)



def fit_direct_spectra(
    absorbance: np.ndarray,
    c: np.ndarray,
    spectra_method: str,
    initial_spectrum_weight: float = 0.0,
) -> np.ndarray:
    """Fit pure spectra for fixed concentration profiles.

    For fixed C(t), each wavelength is an independent least-squares problem:

        A_exp(lambda, :) ~= E(lambda, :) @ C

    With spectra_method="nnls", E(lambda, species) >= 0 is enforced.
    With spectra_method="pinv", the unconstrained pseudoinverse solution is
    used and negative spectrum values are allowed.

    If initial_spectrum_weight > 0, add a soft penalty that keeps the pure
    spectrum of species A close to the first measured spectrum. The weight is
    equivalent to adding that many extra time points to the least-squares fit.
    """
    if initial_spectrum_weight < 0:
        raise ValueError("--initial-spectrum-weight must be nonnegative")
    if spectra_method not in {"nnls", "pinv"}:
        raise ValueError(f"Unknown direct spectra method: {spectra_method}")

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

    if spectra_method == "pinv":
        return (np.linalg.pinv(design) @ target.T).T

    spectra = np.empty((absorbance.shape[0], c.shape[0]))
    for i, row in enumerate(target):
        spectra[i, :], _ = nnls(design, row)
    return spectra



def fit_nonnegative_spectra(
    absorbance: np.ndarray,
    c: np.ndarray,
    initial_spectrum_weight: float = 0.0,
) -> np.ndarray:
    """Fit nonnegative pure spectra for fixed concentration profiles."""
    return fit_direct_spectra(
        absorbance,
        c,
        spectra_method="nnls",
        initial_spectrum_weight=initial_spectrum_weight,
    )



def direct_spectral_error_for_concentrations(
    c: np.ndarray,
    experiment: Experiment,
    spectra_method: str,
    initial_spectrum_weight: float = 0.0,
) -> float:
    """Objective function for a fixed concentration matrix."""
    spectra = fit_direct_spectra(
        experiment.absorbance,
        c,
        spectra_method=spectra_method,
        initial_spectrum_weight=initial_spectrum_weight,
    )
    residuals = experiment.absorbance - spectra @ c
    error_squared = float(np.sum(residuals**2))
    if initial_spectrum_weight > 0:
        initial_mismatch = c[0, 0] * spectra[:, 0] - experiment.absorbance[:, 0]
        error_squared += initial_spectrum_weight * float(np.sum(initial_mismatch**2))
    return float(np.sqrt(error_squared))



def direct_nonnegative_error_for_concentrations(
    c: np.ndarray,
    experiment: Experiment,
    initial_spectrum_weight: float = 0.0,
) -> float:
    """Objective function for a fixed concentration matrix using NNLS."""
    return direct_spectral_error_for_concentrations(
        c,
        experiment,
        spectra_method="nnls",
        initial_spectrum_weight=initial_spectrum_weight,
    )



def direct_spectral_error(
    k: float,
    experiment: Experiment,
    c0: float,
    spectra_method: str,
    initial_spectrum_weight: float = 0.0,
) -> float:
    """Objective function for the A -> B direct spectral fit."""
    if k <= 0:
        return np.inf

    c = concentration_profile_a_to_b(experiment.t, k, c0=c0)
    return direct_spectral_error_for_concentrations(
        c,
        experiment,
        spectra_method=spectra_method,
        initial_spectrum_weight=initial_spectrum_weight,
    )



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



def fit_a_to_b_direct(
    experiment: Experiment,
    c0: float = 1.0,
    n_components: int = 2,
    k_bounds: tuple[float, float] = (1e-8, 1e-1),
    spectra_method: str = "nnls",
    initial_spectrum_weight: float = 0.0,
) -> FitResult:
    """Fit A -> B by direct reconstruction with NNLS or pseudoinverse spectra."""
    q, w, singular_values = factor_analysis(experiment.absorbance, n_components)

    if n_components != 2:
        raise ValueError("The A -> B direct spectral fit requires exactly 2 components.")

    k = optimize_k(
        lambda trial_k: direct_spectral_error(
            trial_k,
            experiment,
            c0,
            spectra_method=spectra_method,
            initial_spectrum_weight=initial_spectrum_weight,
        ),
        k_bounds,
    )
    c = concentration_profile_a_to_b(experiment.t, k, c0=c0)
    spectra = fit_direct_spectra(
        experiment.absorbance,
        c,
        spectra_method=spectra_method,
        initial_spectrum_weight=initial_spectrum_weight,
    )
    absorbance_calc = spectra @ c
    residuals = experiment.absorbance - absorbance_calc
    error = direct_spectral_error_for_concentrations(
        c,
        experiment,
        spectra_method=spectra_method,
        initial_spectrum_weight=initial_spectrum_weight,
    )

    return FitResult(
        method=spectra_method,
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
    if method in {"nnls", "pinv"}:
        return fit_a_to_b_direct(
            experiment,
            c0=c0,
            n_components=n_components,
            k_bounds=k_bounds,
            spectra_method=method,
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



def fit_a_to_b_to_c_direct(
    experiment: Experiment,
    c0: float = 1.0,
    n_components: int = 3,
    k_bounds: tuple[float, float] = (1e-8, 1e-1),
    spectra_method: str = "nnls",
    initial_spectrum_weight: float = 0.0,
) -> FitResult:
    """Fit A -> B -> C by direct reconstruction with NNLS or pseudoinverse spectra."""
    q, w, singular_values = factor_analysis(experiment.absorbance, n_components)

    if n_components != 3:
        raise ValueError("The A -> B -> C direct spectral fit requires exactly 3 components.")

    def objective(params: dict[str, float]) -> float:
        c = concentration_profile_a_to_b_to_c(
            experiment.t,
            params["k1"],
            params["k2"],
            c0=c0,
        )
        return direct_spectral_error_for_concentrations(
            c,
            experiment,
            spectra_method=spectra_method,
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
    spectra = fit_direct_spectra(
        experiment.absorbance,
        c,
        spectra_method=spectra_method,
        initial_spectrum_weight=initial_spectrum_weight,
    )
    absorbance_calc = spectra @ c
    residuals = experiment.absorbance - absorbance_calc
    error = direct_spectral_error_for_concentrations(
        c,
        experiment,
        spectra_method=spectra_method,
        initial_spectrum_weight=initial_spectrum_weight,
    )

    return FitResult(
        method=spectra_method,
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



def fit_a_to_b_to_c_factor(
    experiment: Experiment,
    c0: float = 1.0,
    n_components: int = 3,
    k_bounds: tuple[float, float] = (1e-8, 1e-1),
) -> FitResult:
    """Fit A -> B -> C using a three-component factor-space objective."""
    q, w, singular_values = factor_analysis(experiment.absorbance, n_components)

    if n_components != 3:
        raise ValueError("The A -> B -> C factor-space fit requires exactly 3 components.")

    def objective(params: dict[str, float]) -> float:
        c = concentration_profile_a_to_b_to_c(
            experiment.t,
            params["k1"],
            params["k2"],
            c0=c0,
        )
        return factor_space_error_for_concentrations(c, w)

    params = optimize_kinetic_parameters(
        objective,
        ("k1", "k2"),
        k_bounds,
    )
    candidates = []
    for k1, k2 in (
        (params["k1"], params["k2"]),
        (params["k2"], params["k1"]),
    ):
        c_candidate = concentration_profile_a_to_b_to_c(
            experiment.t,
            k1,
            k2,
            c0=c0,
        )
        try:
            spectra_candidate = factor_spectra_from_concentrations(q, w, c_candidate)
        except np.linalg.LinAlgError:
            continue
        absorbance_calc_candidate = spectra_candidate @ c_candidate
        residuals_candidate = experiment.absorbance - absorbance_calc_candidate
        negative_penalty = float(np.sum(np.minimum(spectra_candidate, 0.0) ** 2))
        reconstruction_error = float(np.linalg.norm(residuals_candidate))
        candidates.append(
            (
                negative_penalty,
                reconstruction_error,
                {"k1": k1, "k2": k2},
                c_candidate,
                spectra_candidate,
                absorbance_calc_candidate,
                residuals_candidate,
            )
        )

    if not candidates:
        raise RuntimeError("A -> B -> C factor fit failed to recover spectra")

    (
        _,
        _,
        params,
        c,
        spectra,
        absorbance_calc,
        residuals,
    ) = min(candidates, key=lambda item: (item[0], item[1]))
    error = factor_space_error_for_concentrations(c, w)

    return FitResult(
        method="factor",
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



def fit_a_to_b_to_c_nnls(
    experiment: Experiment,
    c0: float = 1.0,
    n_components: int = 3,
    k_bounds: tuple[float, float] = (1e-8, 1e-1),
    initial_spectrum_weight: float = 0.0,
) -> FitResult:
    """Fit A -> B -> C by direct reconstruction with nonnegative spectra."""
    return fit_a_to_b_to_c_direct(
        experiment,
        c0=c0,
        n_components=n_components,
        k_bounds=k_bounds,
        spectra_method="nnls",
        initial_spectrum_weight=initial_spectrum_weight,
    )



def fit_a_rev_b_to_c_direct(
    experiment: Experiment,
    c0: float = 1.0,
    n_components: int = 3,
    k_bounds: tuple[float, float] = (1e-8, 1e-1),
    spectra_method: str = "nnls",
    initial_spectrum_weight: float = 0.0,
) -> FitResult:
    """Fit A <-> B -> C by direct reconstruction with NNLS or pseudoinverse spectra."""
    q, w, singular_values = factor_analysis(experiment.absorbance, n_components)

    if n_components != 3:
        raise ValueError("The A <-> B -> C direct spectral fit requires exactly 3 components.")

    def objective(params: dict[str, float]) -> float:
        c = concentration_profile_a_rev_b_to_c(
            experiment.t,
            params["k1"],
            params["k_1"],
            params["k2"],
            c0=c0,
        )
        return direct_spectral_error_for_concentrations(
            c,
            experiment,
            spectra_method=spectra_method,
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
    spectra = fit_direct_spectra(
        experiment.absorbance,
        c,
        spectra_method=spectra_method,
        initial_spectrum_weight=initial_spectrum_weight,
    )
    absorbance_calc = spectra @ c
    residuals = experiment.absorbance - absorbance_calc
    error = direct_spectral_error_for_concentrations(
        c,
        experiment,
        spectra_method=spectra_method,
        initial_spectrum_weight=initial_spectrum_weight,
    )

    return FitResult(
        method=spectra_method,
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



def fit_a_rev_b_to_c_factor(
    experiment: Experiment,
    c0: float = 1.0,
    n_components: int = 3,
    k_bounds: tuple[float, float] = (1e-8, 1e-1),
) -> FitResult:
    """Fit A <-> B -> C using a three-component factor-space objective."""
    q, w, singular_values = factor_analysis(experiment.absorbance, n_components)

    if n_components != 3:
        raise ValueError("The A <-> B -> C factor-space fit requires exactly 3 components.")

    def objective(params: dict[str, float]) -> float:
        c = concentration_profile_a_rev_b_to_c(
            experiment.t,
            params["k1"],
            params["k_1"],
            params["k2"],
            c0=c0,
        )
        return factor_space_error_for_concentrations(c, w)

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
    spectra = factor_spectra_from_concentrations(q, w, c)
    absorbance_calc = spectra @ c
    residuals = experiment.absorbance - absorbance_calc
    error = factor_space_error_for_concentrations(c, w)

    return FitResult(
        method="factor",
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



def fit_a_rev_b_to_c_nnls(
    experiment: Experiment,
    c0: float = 1.0,
    n_components: int = 3,
    k_bounds: tuple[float, float] = (1e-8, 1e-1),
    initial_spectrum_weight: float = 0.0,
) -> FitResult:
    """Fit A <-> B -> C by direct reconstruction with nonnegative spectra."""
    return fit_a_rev_b_to_c_direct(
        experiment,
        c0=c0,
        n_components=n_components,
        k_bounds=k_bounds,
        spectra_method="nnls",
        initial_spectrum_weight=initial_spectrum_weight,
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
    if initial_spectrum_weight > 0 and method == "factor":
        raise ValueError(
            "--initial-spectrum-weight is available only with --fit-method nnls or pinv"
        )

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
        if method == "factor":
            return fit_a_rev_b_to_c_factor(
                experiment,
                c0=c0,
                n_components=n_components,
                k_bounds=k_bounds,
            )
        return fit_a_rev_b_to_c_direct(
            experiment,
            c0=c0,
            n_components=n_components,
            k_bounds=k_bounds,
            spectra_method=method,
            initial_spectrum_weight=initial_spectrum_weight,
        )
    if model == "a_to_b_to_c":
        if method == "factor":
            return fit_a_to_b_to_c_factor(
                experiment,
                c0=c0,
                n_components=n_components,
                k_bounds=k_bounds,
            )
        return fit_a_to_b_to_c_direct(
            experiment,
            c0=c0,
            n_components=n_components,
            k_bounds=k_bounds,
            spectra_method=method,
            initial_spectrum_weight=initial_spectrum_weight,
        )
    raise ValueError(f"Unknown kinetic model: {model}")



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
