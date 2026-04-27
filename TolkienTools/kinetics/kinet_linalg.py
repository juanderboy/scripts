#!/usr/bin/env python3
"""Linear algebra helpers for TolKinet."""

from __future__ import annotations

import numpy as np


def factor_analysis(absorbance: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Q, W and singular values from A = U S Vt, with A ~= Q W.

    Q is the truncated spectral basis. W contains the corresponding temporal
    profiles in factor-analysis coordinates.
    """
    u, singular_values, vt = np.linalg.svd(absorbance, full_matrices=False)
    q = u[:, :n_components]
    w = singular_values[:n_components, None] * vt[:n_components, :]
    return q, w, singular_values

