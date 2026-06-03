#!/usr/bin/env python3
"""Analytical concentration profiles for kinetic models."""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


def concentration_profile_a_to_b(t: np.ndarray, k: float, c0: float = 1.0) -> np.ndarray:
    """Concentration profiles for A -> B as a 2 x n_times matrix."""
    a = c0 * np.exp(-k * t)
    b = c0 - a
    return np.vstack([a, b])


def concentration_profile_mbfe3_sulfide_autocatalytic(
    t: np.ndarray,
    k_slow: float,
    k_auto: float,
    c0: float = 1.0,
) -> np.ndarray:
    """Global profiles for autocatalytic MbFeIII-SH reduction by sulfide.

    The reduced fraction x follows:

        dx/dt = (k_slow + k_auto * x) * (1 - x)

    Reactive sulfur species are not observed directly. Their accumulation is
    represented phenomenologically by x.
    """
    if k_slow <= 0 or k_auto <= 0:
        raise ValueError("All kinetic constants must be positive")

    decaying_exponent = np.exp(-(k_slow + k_auto) * t)
    remaining_fraction = (
        (k_slow + k_auto)
        * decaying_exponent
        / (k_auto * decaying_exponent + k_slow)
    )
    reduced_fraction = 1.0 - remaining_fraction
    reduced_fraction = np.clip(reduced_fraction, 0.0, 1.0)
    return np.vstack([c0 * (1.0 - reduced_fraction), c0 * reduced_fraction])


def concentration_profile_mbfe3_sulfide_binding_autocatalytic(
    t: np.ndarray,
    k_on: float,
    k_slow: float,
    k_auto: float,
    c0: float = 1.0,
) -> np.ndarray:
    """Profiles for MbFeIII binding HS- before autocatalytic reduction.

    The first step is treated as pseudo-first order for a single experiment:

        MbFeIII -> MbFeIII-HS

    The coordinated complex then reduces to MbFeII with the same apparent
    autocatalytic term used by the two-species sulfide model.
    """
    if k_on <= 0 or k_slow <= 0 or k_auto <= 0:
        raise ValueError("All kinetic constants must be positive")
    if c0 <= 0:
        raise ValueError("c0 must be positive")

    t = np.asarray(t, dtype=float)
    if t.ndim != 1:
        raise ValueError("t must be a one-dimensional array")
    if t.size == 0:
        return np.empty((3, 0))
    if np.any(t < 0):
        raise ValueError("Time values must be nonnegative")
    if np.any(np.diff(t) < 0):
        raise ValueError("Time values must be sorted in ascending order")

    def rhs(_time: float, y: np.ndarray) -> tuple[float, float, float]:
        mbfe3, complexed, reduced = y
        reduced_fraction = np.clip(reduced / c0, 0.0, 1.0)
        binding_rate = k_on * mbfe3
        reduction_rate = (k_slow + k_auto * reduced_fraction) * complexed
        return (
            -binding_rate,
            binding_rate - reduction_rate,
            reduction_rate,
        )

    if t[-1] == 0:
        return np.repeat(np.array([[c0], [0.0], [0.0]]), t.size, axis=1)

    solution = solve_ivp(
        rhs,
        (0.0, float(t[-1])),
        (c0, 0.0, 0.0),
        t_eval=t,
        method="DOP853",
        rtol=1e-8,
        atol=max(c0, 1.0) * 1e-10,
    )
    if not solution.success:
        raise RuntimeError(
            "MbFeIII sulfide binding model integration failed: "
            f"{solution.message}"
        )

    c = np.clip(solution.y, 0.0, c0)
    total = c.sum(axis=0)
    valid = total > 0
    c[:, valid] *= c0 / total[valid]
    return c



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
