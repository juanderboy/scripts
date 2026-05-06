#!/usr/bin/env python3
"""Shared data structures and labels for TolKinet."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
    known_species: tuple[str, ...] = ()
    known_spectrum_scales: dict[str, float] | None = None
