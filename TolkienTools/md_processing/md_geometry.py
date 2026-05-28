#!/usr/bin/env python3
"""Geometric analysis for multi-frame XYZ trajectories."""

from __future__ import annotations

import csv
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from md_common import parse_float
from md_xyz import Coord, iter_xyz_frames, xyz_summary


@dataclass(frozen=True)
class MetricSpec:
    label: str
    kind: str
    atoms: tuple[int, ...]


def parse_time_ps(comment: str) -> float | None:
    match = re.search(r"\btime_ps\s*=\s*([-+]?\d*\.?\d+(?:[eEdD][-+]?\d+)?)", comment)
    if not match:
        return None
    return parse_float(match.group(1))


def sub(a: Coord, b: Coord) -> Coord:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def dot(a: Coord, b: Coord) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: Coord, b: Coord) -> Coord:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def norm(v: Coord) -> float:
    return math.sqrt(dot(v, v))


def distance(coords: Sequence[Coord], i: int, j: int) -> float:
    return norm(sub(coords[i - 1], coords[j - 1]))


def angle(coords: Sequence[Coord], i: int, j: int, k: int) -> float:
    v1 = sub(coords[i - 1], coords[j - 1])
    v2 = sub(coords[k - 1], coords[j - 1])
    denom = norm(v1) * norm(v2)
    if denom == 0.0:
        return float("nan")
    cang = max(-1.0, min(1.0, dot(v1, v2) / denom))
    return math.degrees(math.acos(cang))


def dihedral(coords: Sequence[Coord], i: int, j: int, k: int, l: int) -> float:
    b1 = sub(coords[j - 1], coords[i - 1])
    b2 = sub(coords[k - 1], coords[j - 1])
    b3 = sub(coords[l - 1], coords[k - 1])
    n1 = cross(b1, b2)
    n2 = cross(b2, b3)
    b2n = norm(b2)
    if b2n == 0.0:
        return float("nan")
    b2u = (b2[0] / b2n, b2[1] / b2n, b2[2] / b2n)
    m1 = cross(n1, b2u)
    return math.degrees(math.atan2(dot(m1, n2), dot(n1, n2)))


def expected_atom_count(kind: str) -> int:
    counts = {"distance": 2, "angle": 3, "dihedral": 4}
    if kind not in counts:
        raise ValueError(f"Tipo invalido {kind!r}. Usar distance, angle o dihedral")
    return counts[kind]


def parse_metric_spec(text: str) -> MetricSpec:
    parts = text.split(":")
    if len(parts) != 3:
        raise ValueError(f"Metrica invalida {text!r}. Formato: etiqueta:tipo:atomos")
    label = parts[0].strip()
    kind = parts[1].strip().lower()
    atoms = tuple(int(a.strip()) for a in parts[2].split(",") if a.strip())
    if not label:
        raise ValueError("La etiqueta de la metrica no puede estar vacia")
    expected = expected_atom_count(kind)
    if len(atoms) != expected:
        raise ValueError(f"{label}: tipo {kind} requiere {expected} atomos")
    if any(a < 1 for a in atoms):
        raise ValueError(f"{label}: los indices atomicos deben ser >= 1")
    return MetricSpec(label=label, kind=kind, atoms=atoms)


def compute_metric(spec: MetricSpec, coords: Sequence[Coord]) -> float:
    if spec.kind == "distance":
        return distance(coords, spec.atoms[0], spec.atoms[1])
    if spec.kind == "angle":
        return angle(coords, spec.atoms[0], spec.atoms[1], spec.atoms[2])
    if spec.kind == "dihedral":
        return dihedral(coords, spec.atoms[0], spec.atoms[1], spec.atoms[2], spec.atoms[3])
    raise ValueError(f"Tipo de metrica no soportado: {spec.kind}")


def summarize(values: Sequence[float]) -> tuple[str, str, str, str]:
    vals = [v for v in values if not math.isnan(v)]
    if not vals:
        return ("nan", "nan", "nan", "nan")
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return (f"{min(vals):.9f}", f"{max(vals):.9f}", f"{mean:.9f}", f"{math.sqrt(var):.9f}")


def linear_stats(xs: Sequence[float], ys: Sequence[float]) -> tuple[str, str, str]:
    pairs = [(x, y) for x, y in zip(xs, ys) if not math.isnan(x) and not math.isnan(y)]
    if len(pairs) < 2:
        return ("nan", "nan", "nan")
    xvals = [p[0] for p in pairs]
    yvals = [p[1] for p in pairs]
    xmean = sum(xvals) / len(xvals)
    ymean = sum(yvals) / len(yvals)
    sxx = sum((x - xmean) ** 2 for x in xvals)
    syy = sum((y - ymean) ** 2 for y in yvals)
    sxy = sum((x - xmean) * (y - ymean) for x, y in pairs)
    if sxx == 0.0 or syy == 0.0:
        return ("nan", "nan", "nan")
    return (f"{sxy / math.sqrt(sxx * syy):.9f}", f"{sxy / sxx:.9f}", f"{ymean - (sxy / sxx) * xmean:.9f}")


def parse_scatter_pairs(pairs: Sequence[str] | None, specs: Sequence[MetricSpec]) -> list[tuple[str, str]]:
    if not pairs:
        return []
    labels = {spec.label for spec in specs}
    selected: list[tuple[str, str]] = []
    for pair in pairs:
        if "," not in pair:
            raise ValueError(f"Par scatter invalido {pair!r}. Usar etiqueta_x,etiqueta_y")
        left, right = [p.strip() for p in pair.split(",", 1)]
        if left not in labels or right not in labels:
            raise ValueError(f"Par scatter desconocido {pair!r}. Etiquetas: {', '.join(sorted(labels))}")
        selected.append((left, right))
    return selected


def analyze_xyz(
    xyz_path: Path,
    specs: Sequence[MetricSpec],
    output_prefix: str,
    scatter_pairs: Sequence[tuple[str, str]],
    make_plots: bool,
) -> None:
    natoms, nframes = xyz_summary(xyz_path)
    for spec in specs:
        if max(spec.atoms) > natoms:
            raise ValueError(f"{spec.label}: indice atomico fuera de rango, natoms={natoms}")

    rows: list[dict[str, float]] = []
    for frame_idx, comment, coords in iter_xyz_frames(xyz_path):
        row: dict[str, float] = {"frame": float(frame_idx)}
        time_ps = parse_time_ps(comment)
        if time_ps is not None:
            row["time_ps"] = time_ps
        for spec in specs:
            row[spec.label] = compute_metric(spec, coords)
        rows.append(row)

    headers = ["frame"]
    if any("time_ps" in row for row in rows):
        headers.append("time_ps")
    headers.extend(spec.label for spec in specs)

    metrics_path = Path(f"{output_prefix}_metrics.csv")
    with metrics_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({h: format_value(row.get(h)) for h in headers})

    stats_path = Path(f"{output_prefix}_stats.csv")
    with stats_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["label", "kind", "atoms", "n", "min", "max", "mean", "std"])
        for spec in specs:
            values = [row[spec.label] for row in rows]
            writer.writerow([spec.label, spec.kind, "-".join(map(str, spec.atoms)), len(values), *summarize(values)])

    scatter_stats_path = None
    if scatter_pairs:
        scatter_stats_path = Path(f"{output_prefix}_scatter_stats.csv")
        with scatter_stats_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["x", "y", "n", "pearson_r", "slope", "intercept"])
            for x_label, y_label in scatter_pairs:
                xs = [row[x_label] for row in rows]
                ys = [row[y_label] for row in rows]
                writer.writerow([x_label, y_label, len(xs), *linear_stats(xs, ys)])

    if make_plots:
        plot_outputs(rows, specs, output_prefix, scatter_pairs)

    print(f"Archivo XYZ: {xyz_path}")
    print(f"Atomos por frame: {natoms}")
    print(f"Frames: {nframes}")
    print(f"CSV de metricas: {metrics_path}")
    print(f"Resumen estadistico: {stats_path}")
    if scatter_stats_path:
        print(f"Correlaciones scatter: {scatter_stats_path}")
    if make_plots:
        print(f"Figuras PNG: {output_prefix}_*.png")


def format_value(value: float | None) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return f"{value:.9f}"


def axis_label(spec: MetricSpec) -> str:
    if spec.kind == "distance":
        return f"{spec.label} (A)"
    return f"{spec.label} (deg)"


def plot_outputs(
    rows: Sequence[dict[str, float]],
    specs: Sequence[MetricSpec],
    output_prefix: str,
    scatter_pairs: Sequence[tuple[str, str]],
) -> None:
    try:
        mpl_config = Path(".matplotlib-cache").resolve()
        mpl_config.mkdir(exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("No se pudo importar matplotlib. Instalalo o corre con --no-plots.") from exc

    x_axis = "time_ps" if any("time_ps" in row for row in rows) else "frame"
    xs = [row[x_axis] for row in rows]
    for spec in specs:
        ys = [row[spec.label] for row in rows]
        fig, axes = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)
        axes[0].plot(xs, ys, lw=0.8)
        axes[0].set_xlabel(x_axis)
        axes[0].set_ylabel(axis_label(spec))
        axes[0].grid(True, alpha=0.25)
        axes[0].set_title(spec.label)
        axes[1].hist(ys, bins=60, edgecolor="black", linewidth=0.3)
        axes[1].set_xlabel(axis_label(spec))
        axes[1].set_ylabel("count")
        axes[1].grid(True, alpha=0.25)
        fig.savefig(f"{output_prefix}_{spec.label}_timeseries_hist.png", dpi=180)
        plt.close(fig)

    for x_label, y_label in scatter_pairs:
        xvals = [row[x_label] for row in rows]
        yvals = [row[y_label] for row in rows]
        fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
        ax.scatter(xvals, yvals, s=10, alpha=0.45, edgecolors="none")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.25)
        r, slope, intercept = linear_stats(xvals, yvals)
        ax.set_title(f"{y_label} vs {x_label} | r={r}")
        if slope != "nan" and intercept != "nan":
            xmin, xmax = min(xvals), max(xvals)
            ax.plot([xmin, xmax], [float(slope) * xmin + float(intercept), float(slope) * xmax + float(intercept)], lw=1.2)
        fig.savefig(f"{output_prefix}_scatter_{x_label}_vs_{y_label}.png", dpi=180)
        plt.close(fig)

