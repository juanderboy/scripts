#!/usr/bin/env python3
"""Known pure-spectrum readers for TolKinet."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class KnownSpectrumSpec:
    """One user request mapping species labels to one spectra file."""

    species: tuple[str, ...]
    path: Path


def parse_known_spectrum_spec(text: str) -> KnownSpectrumSpec:
    """Parse specs like 'A B spectra.dat' or 'A,B=spectra.dat'."""
    normalized = text.strip()
    if not normalized:
        raise ValueError("Known-spectrum specification is empty")

    if "=" in normalized:
        species_text, path_text = normalized.split("=", maxsplit=1)
        species = tuple(part.strip().upper() for part in species_text.replace(",", " ").split())
        path = Path(path_text.strip())
    else:
        parts = normalized.split()
        if len(parts) < 2:
            raise ValueError("Use species plus file, e.g. 'A B spectra.dat'")
        species = tuple(part.strip().upper() for part in parts[:-1])
        path = Path(parts[-1])

    if not species:
        raise ValueError("At least one species must be listed")
    if any(label not in {"A", "B", "C"} for label in species):
        raise ValueError("Known species must be A, B or C")
    if len(set(species)) != len(species):
        raise ValueError("Known species cannot be repeated in one specification")
    if not str(path):
        raise ValueError("Known-spectrum file path is empty")
    return KnownSpectrumSpec(species=species, path=path)


def _read_header_labels(path: Path) -> tuple[str, ...]:
    """Read the first header line and return data-column labels."""
    with path.open(encoding="utf-8-sig") as file:
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                fields = stripped[1:].strip().replace(";", " ").replace(",", " ").split()
                if len(fields) >= 2 and fields[0].lower() in {"wavelength", "lambda"}:
                    return tuple(field.upper() for field in fields[1:])
                continue
            break
    raise ValueError(f"{path}: no entiendo el formato del archivo. revisarlo")


def read_known_spectrum_file(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Read a pure-spectra table and return wavelength plus spectra by label."""
    try:
        labels = _read_header_labels(path)
        data = np.loadtxt(path, comments="#")
    except ValueError as exc:
        raise ValueError(
            f"{path}: no entiendo el formato del archivo. revisarlo"
        ) from exc
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] != len(labels) + 1:
        raise ValueError(
            f"{path}: header has {len(labels)} spectra but data has {data.shape[1] - 1}"
        )
    wavelength = data[:, 0]
    if wavelength.size < 2:
        raise ValueError(f"{path}: at least two wavelength points are required")
    order = np.argsort(wavelength)
    wavelength = wavelength[order]
    spectra = {
        label: data[order, column + 1]
        for column, label in enumerate(labels)
    }
    return wavelength, spectra


def known_spectra_wavelength_range(specs: list[KnownSpectrumSpec]) -> tuple[float, float] | None:
    """Return the wavelength interval covered by all known spectra files."""
    if not specs:
        return None

    lower_values: list[float] = []
    upper_values: list[float] = []
    for spec in specs:
        file_wavelength, _ = read_known_spectrum_file(spec.path)
        lower_values.append(float(file_wavelength[0]))
        upper_values.append(float(file_wavelength[-1]))

    lower = max(lower_values)
    upper = min(upper_values)
    if lower >= upper:
        raise ValueError("Known spectra files do not share a wavelength overlap")
    return lower, upper


def build_known_spectra_matrix(
    specs: list[KnownSpectrumSpec],
    model_species: tuple[str, ...],
    wavelength: np.ndarray,
) -> tuple[np.ndarray | None, tuple[str, ...]]:
    """Return a matrix with fixed spectra columns and NaN for fitted columns."""
    if not specs:
        return None, ()

    species_to_column = {label: index for index, label in enumerate(model_species)}
    known = np.full((wavelength.size, len(model_species)), np.nan, dtype=float)
    known_labels: list[str] = []

    for spec in specs:
        file_wavelength, file_spectra = read_known_spectrum_file(spec.path)
        for label in spec.species:
            if label not in species_to_column:
                raise ValueError(
                    f"Species {label} is not part of the selected model "
                    f"({', '.join(model_species)})"
                )
            if label not in file_spectra:
                raise ValueError(f"{spec.path}: species {label} was not found in the header")
            if label in known_labels:
                raise ValueError(f"Species {label} was provided more than once")
            if wavelength[0] < file_wavelength[0] or wavelength[-1] > file_wavelength[-1]:
                raise ValueError(
                    f"{spec.path}: wavelength range {file_wavelength[0]:g}-"
                    f"{file_wavelength[-1]:g} does not cover fit range "
                    f"{wavelength[0]:g}-{wavelength[-1]:g}"
                )

            column = species_to_column[label]
            known[:, column] = np.interp(wavelength, file_wavelength, file_spectra[label])
            known_labels.append(label)

    return known, tuple(known_labels)
