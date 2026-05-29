#!/usr/bin/env python3
"""Shared helpers for molecular-dynamics processing tools."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Segment:
    index: int
    path: Path
    dt_ps: float | None
    nstlim: int | None

    @property
    def planned_total_ps(self) -> float | None:
        if self.dt_ps is None or self.nstlim is None:
            return None
        return self.dt_ps * self.nstlim

    @property
    def total_ps(self) -> float | None:
        return self.planned_total_ps

    def duration_from_frames(self, nframes: int) -> float | None:
        if self.dt_ps is None:
            return None
        return self.dt_ps * nframes


def parse_float(text: str) -> float:
    return float(text.replace("d", "e").replace("D", "E"))


def find_numeric_dirs(root: Path) -> list[Path]:
    dirs = [p for p in root.iterdir() if p.is_dir() and p.name.isdigit()]
    return sorted(dirs, key=lambda p: int(p.name))


def parse_d_qm_input(path: Path) -> tuple[float, int]:
    text = path.read_text(encoding="utf-8", errors="ignore")

    dt_match = re.search(r"\bdt\s*=\s*([-+]?\d*\.?\d+(?:[eEdD][-+]?\d+)?)", text, re.IGNORECASE)
    if not dt_match:
        raise ValueError(f"No se encontro 'dt' en {path}")
    dt = parse_float(dt_match.group(1))

    nst_match = re.search(r"\b(?:nstlim|ntslim)\s*=\s*(\d+)", text, re.IGNORECASE)
    if not nst_match:
        raise ValueError(f"No se encontro 'nstlim' o 'ntslim' en {path}")
    nstlim = int(nst_match.group(1))

    return dt, nstlim


def discover_segments(root: Path, input_name: str = "d_QM.in") -> list[Segment]:
    segments: list[Segment] = []
    for seg_dir in find_numeric_dirs(root):
        dt_ps: float | None = None
        nstlim: int | None = None
        input_path = seg_dir / input_name
        if input_path.exists():
            try:
                dt_ps, nstlim = parse_d_qm_input(input_path)
            except ValueError:
                pass
        segments.append(Segment(index=int(seg_dir.name), path=seg_dir, dt_ps=dt_ps, nstlim=nstlim))
    return segments


def count_xyz_frames(path: Path) -> tuple[int, int]:
    nframes = 0
    natoms_ref: int | None = None
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        while True:
            natoms_line = fh.readline()
            if not natoms_line:
                break
            if not natoms_line.strip():
                continue
            natoms = int(natoms_line.strip())
            if natoms_ref is None:
                natoms_ref = natoms
            elif natoms != natoms_ref:
                raise ValueError(f"Numero de atomos variable en {path}: frame {nframes + 1}")
            if not fh.readline():
                raise ValueError(f"Falta linea de comentario en {path}, frame {nframes + 1}")
            for _ in range(natoms):
                if not fh.readline():
                    raise ValueError(f"Frame truncado en {path}, frame {nframes + 1}")
            nframes += 1
    if natoms_ref is None:
        raise ValueError(f"No se encontraron frames XYZ en {path}")
    return natoms_ref, nframes
