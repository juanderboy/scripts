#!/usr/bin/env python3
"""XYZ trajectory readers and writers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from md_common import Segment, count_xyz_frames, parse_float


Coord = tuple[float, float, float]


@dataclass
class XyzFrame:
    natoms_line: str
    comment_line: str
    atom_lines: list[str]
    coords: list[Coord]


def parse_xyz_frames(path: Path) -> list[XyzFrame]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    frames: list[XyzFrame] = []
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        try:
            natoms = int(lines[i].strip())
        except ValueError as exc:
            raise ValueError(f"Formato XYZ invalido en {path}, linea {i + 1}: {lines[i]!r}") from exc
        natoms_line = lines[i]
        i += 1
        if i >= len(lines):
            raise ValueError(f"Falta linea de comentario para frame en {path}")
        comment_line = lines[i]
        i += 1
        if i + natoms > len(lines):
            raise ValueError(f"Frame truncado en {path}, linea {i + 1}")
        atom_lines = lines[i : i + natoms]
        coords: list[Coord] = []
        for atom_idx, line in enumerate(atom_lines, start=1):
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"Linea atomica invalida en {path}, atom {atom_idx}")
            coords.append((parse_float(parts[-3]), parse_float(parts[-2]), parse_float(parts[-1])))
        frames.append(XyzFrame(natoms_line=natoms_line, comment_line=comment_line, atom_lines=atom_lines, coords=coords))
        i += natoms
    return frames


def iter_xyz_frames(path: Path) -> Iterable[tuple[int, str, list[Coord]]]:
    for idx, frame in enumerate(parse_xyz_frames(path), start=1):
        yield idx, frame.comment_line, frame.coords


def merge_segment_xyz(
    segments: list[Segment],
    xyz_name: str,
    output_path: Path,
) -> tuple[int, int, float]:
    total_frames = 0
    processed_segments = 0
    global_time_ps = 0.0

    with output_path.open("w", encoding="utf-8") as out:
        for segment in segments:
            xyz_path = segment.path / xyz_name
            if not xyz_path.exists():
                continue
            frames = parse_xyz_frames(xyz_path)
            if not frames:
                continue
            processed_segments += 1

            if segment.total_ps is not None:
                frame_dt_ps = segment.total_ps / len(frames)
            else:
                frame_dt_ps = 1.0

            for seg_frame, frame in enumerate(frames, start=1):
                out.write(f"{frame.natoms_line.strip()}\n")
                out.write(f"segment={segment.index} frame={seg_frame} time_ps={global_time_ps:.9f}\n")
                for atom_line in frame.atom_lines:
                    out.write(f"{atom_line.rstrip()}\n")
                total_frames += 1
                global_time_ps += frame_dt_ps

    return processed_segments, total_frames, global_time_ps


def xyz_summary(path: Path) -> tuple[int, int]:
    return count_xyz_frames(path)

