#!/usr/bin/env python3
"""Convert HP/Agilent .KD kinetic spectra files to a readable table.

The output table uses the same layout as 117.txt:

    first row:    0 ; t1 ; t2 ; ...
    first column: lambda values
    body:         absorbance(lambda, time)

What this parser assumes about the .KD file:

    - The file is a binary "REGISTER FILE" produced by the HP/Agilent
      spectrophotometer software.
    - Each spectrum block contains the marker:

          Wavelength (nm)\0Absorbance (AU)\0

      followed immediately by the absorbance values as little-endian float64.
    - Each block also contains a RelTime field. For this KD format, the time
      value is stored as a little-endian float64 21 bytes after the "RelTime"
      marker.
    - The wavelength axis is inferred from metadata stored shortly before the
      first spectrum block. If that inference fails, pass --lambda-start and
      --lambda-step explicitly.

This script is intentionally separate from kinet_python.py so KD conversion can
be checked independently before fitting.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np


SPECTRUM_MARKER = b"Wavelength (nm)\0Absorbance (AU)\0"
NEXT_OBJECT_MARKER = b"ObjClass"
RELTIME_MARKER = b"RelTime"


def read_float64_le(data: bytes, offset: int) -> float:
    """Read one little-endian float64 from data."""
    return struct.unpack("<d", data[offset : offset + 8])[0]


def find_spectrum_blocks(data: bytes) -> list[tuple[int, int]]:
    """Return (data_start, n_points) for every spectrum block."""
    blocks: list[tuple[int, int]] = []
    pos = 0

    while True:
        marker_pos = data.find(SPECTRUM_MARKER, pos)
        if marker_pos < 0:
            break

        data_start = marker_pos + len(SPECTRUM_MARKER)
        next_object = data.find(NEXT_OBJECT_MARKER, data_start)
        if next_object < 0:
            raise ValueError(f"Could not find ObjClass after spectrum at byte {marker_pos}")

        # In this KD register format there are 14 bytes between the last
        # absorbance float64 and the next ObjClass marker.
        payload_bytes = next_object - data_start - 14
        if payload_bytes <= 0 or payload_bytes % 8 != 0:
            raise ValueError(
                f"Could not determine spectrum length at byte {marker_pos}: "
                f"payload_bytes={payload_bytes}"
            )

        blocks.append((data_start, payload_bytes // 8))
        pos = marker_pos + 1

    return blocks


def extract_absorbance(data: bytes, blocks: list[tuple[int, int]]) -> np.ndarray:
    """Extract absorbance matrix with shape n_wavelengths x n_times."""
    if not blocks:
        raise ValueError("No spectrum blocks were found")

    n_points = blocks[0][1]
    bad_lengths = [n for _, n in blocks if n != n_points]
    if bad_lengths:
        raise ValueError("Not all spectrum blocks have the same number of points")

    spectra = np.empty((n_points, len(blocks)), dtype=float)
    for j, (start, _) in enumerate(blocks):
        values = struct.unpack(f"<{n_points}d", data[start : start + n_points * 8])
        spectra[:, j] = values

    return spectra


def extract_times(data: bytes, expected_count: int) -> np.ndarray:
    """Extract RelTime values from the KD file."""
    positions: list[int] = []
    pos = 0
    while True:
        marker_pos = data.find(RELTIME_MARKER, pos)
        if marker_pos < 0:
            break
        positions.append(marker_pos)
        pos = marker_pos + 1

    if len(positions) != expected_count:
        raise ValueError(
            f"Found {len(positions)} RelTime fields but {expected_count} spectra"
        )

    times = np.array([read_float64_le(data, p + 21) for p in positions])
    if not np.all(np.isfinite(times)):
        raise ValueError("Some RelTime values are not finite")

    return times


def infer_wavelength_axis(
    data: bytes,
    first_data_start: int,
    n_points: int,
    lambda_start: float | None,
    lambda_step: float | None,
) -> np.ndarray:
    """Infer or construct wavelength axis."""
    if lambda_start is not None or lambda_step is not None:
        if lambda_start is None or lambda_step is None:
            raise ValueError("Use both --lambda-start and --lambda-step, or neither")
        return lambda_start + lambda_step * np.arange(n_points)

    first_marker = first_data_start - len(SPECTRUM_MARKER)
    window_start = max(0, first_marker - 512)
    candidates: list[tuple[int, float, float]] = []

    for offset in range(window_start, first_marker - 15):
        start = read_float64_le(data, offset)
        step = read_float64_le(data, offset + 8)
        end = start + step * (n_points - 1)
        if (
            np.isfinite(start)
            and np.isfinite(step)
            and 100.0 <= start <= 1000.0
            and 0.01 <= step <= 20.0
            and start < end <= 2000.0
        ):
            candidates.append((offset, start, step))

    if not candidates:
        raise ValueError(
            "Could not infer wavelength axis. Try --lambda-start and --lambda-step."
        )

    # Prefer the candidate closest to the first spectrum marker. In JP-117.KD
    # this is start=190 nm, step=2 nm.
    _, start, step = max(candidates, key=lambda item: item[0])
    return start + step * np.arange(n_points)


def write_table(path: Path, times: np.ndarray, wavelength: np.ndarray, absorbance: np.ndarray) -> None:
    """Write the converted table in semicolon-separated text format."""
    with path.open("w", encoding="utf-8") as f:
        f.write("0")
        for t in times:
            f.write(f";{t:.10g}")
        f.write("\n")

        for lam, row in zip(wavelength, absorbance):
            f.write(f"{lam:.10g}")
            for value in row:
                f.write(f";{value:.6f}")
            f.write("\n")


def convert_kd(
    input_path: Path,
    output_path: Path,
    lambda_start: float | None = None,
    lambda_step: float | None = None,
) -> tuple[int, int, float, float]:
    """Convert KD file and return summary information."""
    data = input_path.read_bytes()
    blocks = find_spectrum_blocks(data)
    absorbance = extract_absorbance(data, blocks)
    times = extract_times(data, expected_count=absorbance.shape[1])
    wavelength = infer_wavelength_axis(
        data,
        first_data_start=blocks[0][0],
        n_points=absorbance.shape[0],
        lambda_start=lambda_start,
        lambda_step=lambda_step,
    )
    write_table(output_path, times, wavelength, absorbance)
    return absorbance.shape[1], absorbance.shape[0], wavelength[0], wavelength[-1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert an HP/Agilent .KD kinetic spectra file to text."
    )
    parser.add_argument("input", type=Path, help="Input .KD file")
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        help="Output .txt/.dat file. Defaults to input stem + .txt",
    )
    parser.add_argument("--lambda-start", type=float, default=None)
    parser.add_argument("--lambda-step", type=float, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = args.output if args.output is not None else args.input.with_suffix(".txt")

    n_spectra, n_wavelengths, lambda_first, lambda_last = convert_kd(
        args.input,
        output,
        lambda_start=args.lambda_start,
        lambda_step=args.lambda_step,
    )

    print(f"input: {args.input}")
    print(f"output: {output}")
    print(f"spectra / times: {n_spectra}")
    print(f"wavelength points: {n_wavelengths}")
    print(f"wavelength range: {lambda_first:g} - {lambda_last:g}")


if __name__ == "__main__":
    main()
