#!/usr/bin/env python3
"""Shared data structures and labels for TolKinet."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


GENERAL_MODEL_LABELS = {
    "a_to_b": "A -> B irreversible de primer orden",
    "a_to_b_to_c": "A -> B -> C irreversible consecutivo",
    "a_rev_b_to_c": "A <-> B -> C",
}

SPECIAL_MODEL_LABELS = {
    "mbfe3_sulfide_autocatalytic": "reduccion autocatalitica de MbFe(III) por sulfuros",
    "mbfe3_sulfide_binding_autocatalytic": (
        "reduccion autocatalitica de MbFe(III) por sulfuros con binding inicial"
    ),
}

MODEL_LABELS = {
    **GENERAL_MODEL_LABELS,
    **SPECIAL_MODEL_LABELS,
}


MODEL_SPECIES = {
    "a_to_b": ("A", "B"),
    "a_to_b_to_c": ("A", "B", "C"),
    "a_rev_b_to_c": ("A", "B", "C"),
    "mbfe3_sulfide_autocatalytic": ("MbFeIII-SH", "MbFeII"),
    "mbfe3_sulfide_binding_autocatalytic": ("MbFeIII", "MbFeIII-HS", "MbFeII"),
}


PARAMETER_LABELS = {
    "k": "k",
    "k1": "k1",
    "k_1": "k-1",
    "k2": "k2",
    "k_on": "k_on",
    "k_slow": "k_slow,obs",
    "k_auto": "k_auto",
}


MODEL_PRESENTATIONS = {
    "a_to_b": {
        "scheme": "A -> B",
        "profiles": (
            "[A](t) = c0 * exp(-k * t)",
            "[B](t) = c0 - [A](t)",
        ),
        "parameters": (
            "k: constante de velocidad de primer orden",
        ),
        "notes": (
            "Se ajustan simultaneamente los espectros puros de A y B.",
        ),
    },
    "a_to_b_to_c": {
        "scheme": "A -> B -> C",
        "profiles": (
            "[A](t) = c0 * exp(-k1 * t)",
            "[B](t) = c0 * k1/(k2-k1) * (exp(-k1*t) - exp(-k2*t))",
            "[C](t) = c0 - [A](t) - [B](t)",
        ),
        "parameters": (
            "k1: constante de velocidad para A -> B",
            "k2: constante de velocidad para B -> C",
        ),
        "notes": (
            "Si k1 y k2 coinciden, se usa el limite analitico correspondiente.",
        ),
    },
    "a_rev_b_to_c": {
        "scheme": "A <-> B -> C",
        "profiles": (
            "d[A]/dt = -k1*[A] + k-1*[B]",
            "d[B]/dt = k1*[A] - (k-1+k2)*[B]",
            "d[C]/dt = k2*[B]",
        ),
        "parameters": (
            "k1: constante de velocidad para A -> B",
            "k-1: constante de velocidad para B -> A",
            "k2: constante de velocidad para B -> C",
        ),
        "notes": (
            "Los perfiles temporales se calculan resolviendo el sistema lineal.",
        ),
    },
    "mbfe3_sulfide_autocatalytic": {
        "scheme": "MbFeIII-SH -> MbFeII, con aceleracion autocatalitica aparente",
        "profiles": (
            "x(t) = [MbFeII](t) / [Mb]total",
            "dx/dt = (k_slow + k_auto*x) * (1 - x)",
            "[MbFeIII-SH](t) = c0 * (1 - x)",
            "[MbFeII](t) = c0 * x",
        ),
        "parameters": (
            "k_slow,obs: constante aparente de la fase lenta inicial",
            "k_auto: aceleracion fenomenologica aparente",
        ),
        "notes": (
            "El ajuste usa todos los tiempos y dos especies absorbentes.",
            "x actua como proxy de especies reactivas de azufre no observadas.",
            "La dependencia de k_slow,obs con [HS-] debe determinarse experimentalmente.",
            "k_auto no debe interpretarse directamente como k_Red(HSS-).",
        ),
    },
    "mbfe3_sulfide_binding_autocatalytic": {
        "scheme": (
            "MbFeIII + HS- -> MbFeIII-HS -> MbFeII, "
            "con aceleracion autocatalitica aparente"
        ),
        "profiles": (
            "d[MbFeIII]/dt = -k_on * [MbFeIII]",
            "x(t) = [MbFeII](t) / [Mb]total",
            (
                "d[MbFeIII-HS]/dt = k_on*[MbFeIII] "
                "- (k_slow + k_auto*x)*[MbFeIII-HS]"
            ),
            "d[MbFeII]/dt = (k_slow + k_auto*x)*[MbFeIII-HS]",
        ),
        "parameters": (
            (
                "k_on: constante aparente de coordinacion por HS- "
                "bajo las condiciones del experimento"
            ),
            "k_slow,obs: constante aparente de reduccion lenta del complejo MbFeIII-HS",
            "k_auto: aceleracion fenomenologica aparente",
        ),
        "notes": (
            "El ajuste usa tres especies absorbentes: MbFeIII, MbFeIII-HS y MbFeII.",
            (
                "Si [HS-] se mantiene en exceso, k_on corresponde al k_on "
                "bimolecular multiplicado por [HS-]."
            ),
            "x actua como proxy de especies reactivas de azufre no observadas.",
            "k_auto no debe interpretarse directamente como k_Red(HSS-).",
        ),
    },
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
