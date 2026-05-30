#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Numerical construction of broadened TD-DFT spectra.

This module provides Gaussian broadening and oscillator-strength conversion
used by both single-file and batch analyses.
"""

import math

import numpy as np

from td_common import HC_EV_NM, LN10, N_A, c_light, e_charge, eps0, hbar, m_e


def gauss_area(a, m, x, w_hwhm):
    """
    Gaussiana de área 'a' centrada en 'm', con half-width-at-half-maximum (HWHM) = w_hwhm.
    G(x) = A * exp( -ln(2) * ((x - m)/w)**2 )
    A = a / ( w * sqrt(pi / ln(2)) )
    De modo que ∫ G(x) dx = a.
    """
    amplitude = a / (w_hwhm * np.sqrt(np.pi / np.log(2.0)))
    return amplitude * np.exp(-np.log(2.0) * ((x - m) / w_hwhm) ** 2)


def build_energy_grid(energies_eV, fwhm_eV, start_lambda=None, end_lambda=None):
    """
    Construye un grid de energía (eV) adecuado para la convolución:

    - Si start_lambda y end_lambda están dados (en nm), se convierten a límites en E.
    - Si no, se toma [Emin - 3*FWHM, Emax + 3*FWHM].

    Devuelve:
      E_grid (eV), E_min, E_max
    """
    Emin_data = energies_eV.min()
    Emax_data = energies_eV.max()

    if start_lambda is not None and end_lambda is not None:
        # convertir nm -> eV (E = hc/λ)
        lam_min = min(start_lambda, end_lambda)
        lam_max = max(start_lambda, end_lambda)
        # en energía, λ pequeña = E grande
        Emax = HC_EV_NM / lam_min
        Emin = HC_EV_NM / lam_max
    else:
        Emin = max(Emin_data - 3.0 * fwhm_eV, 1.0e-4)
        Emax = Emax_data + 3.0 * fwhm_eV

    # resolución del grid: paso ~FWHM/50 para líneas suaves
    dE = fwhm_eV / 50.0
    if dE <= 0:
        dE = 0.001

    E_grid = np.arange(Emin, Emax, dE)
    return E_grid, Emin, Emax


def compute_epsilon_spectrum(energies_eV, foscs, fwhm_eV, n_ref=1.33):
    """
    Calcula ε(E) [L mol^-1 cm^-1] a partir de:
      - energies_eV: array de energías de transición en eV
      - foscs: fuerzas de oscilador
      - fwhm_eV: ancho de línea en eV (FWHM)
      - n_ref: índice de refracción (default 1.33 ~ agua)

    Implementa:
      σ(E) = (π e² ħ)/(2 m c ε₀ n_r E) Σ_n ΔE_n f_n g(E-ΔE_n, δ)
    usando g normalizada en eV, y luego:
      ε(E) = σ(E) * (N_A / (1000 ln 10)) * 10^4
    => ε(E) = C_eps * [ S(E) / E ]
    con S(E) = Σ ΔE_n f_n g(E - ΔE_n).
    """
    if energies_eV.size == 0 or foscs.size == 0:
        raise ValueError("Sin datos de energías/fosc para construir el espectro.")

    # prefactor para ε directamente (en L mol^-1 cm^-1)
    C0 = math.pi * e_charge * hbar / (2.0 * m_e * c_light * eps0 * n_ref)
    C_eps = C0 * (10.0 * N_A / LN10)

    # grid de energía
    E_grid, Emin, Emax = build_energy_grid(energies_eV, fwhm_eV)

    # FWHM -> HWHM
    w_hwhm = fwhm_eV / 2.0

    # S(E) = Σ ΔE_n f_n g(E-ΔE_n)
    S = np.zeros_like(E_grid)
    for En, fn in zip(energies_eV, foscs):
        area = En * fn
        S += gauss_area(area, En, E_grid, w_hwhm)

    # ε(E) = C_eps * S(E) / E
    epsilon_E = C_eps * S / E_grid

    return E_grid, epsilon_E
