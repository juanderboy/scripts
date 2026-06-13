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
from md_xyz import Coord, parse_xyz_frames, xyz_summary


ELEMENT_SYMBOLS = {
    7: "N",
    26: "Fe",
}


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


def scale(v: Coord, factor: float) -> Coord:
    return (v[0] * factor, v[1] * factor, v[2] * factor)


def centroid(coords: Sequence[Coord]) -> Coord:
    n = len(coords)
    return (
        sum(coord[0] for coord in coords) / n,
        sum(coord[1] for coord in coords) / n,
        sum(coord[2] for coord in coords) / n,
    )


def normalize(v: Coord) -> Coord:
    vnorm = norm(v)
    if vnorm == 0.0:
        return (float("nan"), float("nan"), float("nan"))
    return (v[0] / vnorm, v[1] / vnorm, v[2] / vnorm)


def element_symbol(atom_line: str) -> str:
    token = atom_line.split()[0]
    try:
        atomic_number = int(token)
    except ValueError:
        return token.strip().capitalize()
    return ELEMENT_SYMBOLS.get(atomic_number, str(atomic_number))


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


def plane_from_four_points(points: Sequence[Coord]) -> tuple[Coord, Coord]:
    """
    Return an order-independent best-fit plane for four points.

    NumPy gives the least-squares normal through SVD. If it is unavailable,
    fall back to the non-collinear triple with the smallest residual.
    """
    center = centroid(points)
    try:
        import numpy as np

        matrix = np.array([[p[0] - center[0], p[1] - center[1], p[2] - center[2]] for p in points], dtype=float)
        _u, _s, vh = np.linalg.svd(matrix, full_matrices=False)
        normal_arr = vh[-1]
        normal = normalize((float(normal_arr[0]), float(normal_arr[1]), float(normal_arr[2])))
        return center, normal
    except Exception:
        pass

    best: tuple[float, Coord] | None = None
    for i in range(len(points) - 2):
        for j in range(i + 1, len(points) - 1):
            for k in range(j + 1, len(points)):
                normal = normalize(cross(sub(points[j], points[i]), sub(points[k], points[i])))
                if math.isnan(normal[0]):
                    continue
                residuals = [point_plane_signed_distance(point, center, normal) for point in points]
                rms = math.sqrt(sum(res * res for res in residuals) / len(residuals))
                if best is None or rms < best[0]:
                    best = (rms, normal)
    if best is None:
        return center, (float("nan"), float("nan"), float("nan"))
    return center, best[1]


def point_plane_signed_distance(point: Coord, plane_point: Coord, plane_normal: Coord) -> float:
    if math.isnan(plane_normal[0]):
        return float("nan")
    return dot(sub(point, plane_point), plane_normal)


def orient_normal_toward_reference(
    plane_point: Coord,
    plane_normal: Coord,
    reference_point: Coord,
) -> Coord:
    if point_plane_signed_distance(reference_point, plane_point, plane_normal) < 0.0:
        return scale(plane_normal, -1.0)
    return plane_normal


def heme_oop(coords: Sequence[Coord], atoms: Sequence[int], signed: bool = True) -> float:
    fe_idx, n1, n2, n3, n4 = atoms[:5]
    fe = coords[fe_idx - 1]
    n_points = [coords[idx - 1] for idx in (n1, n2, n3, n4)]
    plane_point, plane_normal = plane_from_four_points(n_points)
    if signed:
        if len(atoms) < 6:
            raise ValueError("heme_oop firmado requiere NHis/proximal como sexto atomo")
        proximal = coords[atoms[5] - 1]
        plane_normal = orient_normal_toward_reference(plane_point, plane_normal, proximal)
        return point_plane_signed_distance(fe, plane_point, plane_normal)
    return abs(point_plane_signed_distance(fe, plane_point, plane_normal))


def expected_atom_count(kind: str) -> int:
    counts = {"distance": 2, "angle": 3, "dihedral": 4}
    if kind not in counts:
        raise ValueError(f"Tipo invalido {kind!r}. Usar distance, angle, dihedral o heme_oop")
    return counts[kind]


def parse_metric_spec(text: str) -> MetricSpec:
    parts = text.split(":")
    if len(parts) != 3:
        raise ValueError(f"Metrica invalida {text!r}. Formato: etiqueta:tipo:atomos")
    label = parts[0].strip()
    kind = parts[1].strip().lower()
    if kind in {"heme_oop", "heme-oop", "heme-out-of-plane", "heme_out_of_plane"}:
        kind = "heme_oop"
    if kind in {"heme_oop_abs", "heme-oop-abs", "heme_out_of_plane_abs"}:
        kind = "heme_oop_abs"
    atom_text = parts[2].strip()
    if not label:
        raise ValueError("La etiqueta de la metrica no puede estar vacia")
    if kind in {"heme_oop", "heme_oop_abs"}:
        if atom_text.lower() == "auto":
            atoms = ()
        else:
            atoms = tuple(int(a.strip()) for a in atom_text.split(",") if a.strip())
            allowed_counts = {1, 5, 6} if kind == "heme_oop" else {1, 5}
            if len(atoms) not in allowed_counts:
                raise ValueError(
                    f"{label}: tipo {kind} requiere auto, Fe, Fe,N1,N2,N3,N4"
                    + (", o Fe,N1,N2,N3,N4,NHis" if kind == "heme_oop" else "")
                )
    else:
        atoms = tuple(int(a.strip()) for a in atom_text.split(",") if a.strip())
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
    if spec.kind == "heme_oop":
        return heme_oop(coords, spec.atoms, signed=True)
    if spec.kind == "heme_oop_abs":
        return heme_oop(coords, spec.atoms, signed=False)
    raise ValueError(f"Tipo de metrica no soportado: {spec.kind}")


def resolve_metric_specs(specs: Sequence[MetricSpec], atom_lines: Sequence[str], coords: Sequence[Coord]) -> list[MetricSpec]:
    resolved: list[MetricSpec] = []
    for spec in specs:
        if spec.kind in {"heme_oop", "heme_oop_abs"} and len(spec.atoms) in {0, 1}:
            resolved.append(resolve_heme_oop_spec(spec, atom_lines, coords))
        elif spec.kind == "heme_oop" and len(spec.atoms) == 5:
            proximal = resolve_manual_proximal_nitrogen(spec, atom_lines, coords)
            print(f"[INFO] {spec.label}: NHis autodetectado como {proximal}; positivo hacia NHis.")
            resolved.append(MetricSpec(label=spec.label, kind=spec.kind, atoms=(*spec.atoms, proximal)))
        else:
            resolved.append(spec)
    return resolved


def resolve_manual_proximal_nitrogen(spec: MetricSpec, atom_lines: Sequence[str], coords: Sequence[Coord]) -> int:
    symbols = [element_symbol(line) for line in atom_lines]
    fe_idx = spec.atoms[0]
    pyrrolic = set(spec.atoms[1:5])
    candidates = [
        idx
        for idx, symbol in enumerate(symbols, start=1)
        if symbol == "N" and idx not in pyrrolic and distance(coords, fe_idx, idx) <= 3.2
    ]
    if not candidates:
        raise ValueError(
            f"{spec.label}: no se pudo autodetectar NHis; pasar Fe,N1,N2,N3,N4,NHis"
        )
    return min(candidates, key=lambda idx: distance(coords, fe_idx, idx))


def resolve_heme_oop_spec(spec: MetricSpec, atom_lines: Sequence[str], coords: Sequence[Coord]) -> MetricSpec:
    symbols = [element_symbol(line) for line in atom_lines]
    if len(spec.atoms) == 1:
        fe_idx = spec.atoms[0]
    else:
        fe_candidates = [idx for idx, symbol in enumerate(symbols, start=1) if symbol == "Fe"]
        if len(fe_candidates) != 1:
            raise ValueError(
                f"{spec.label}: autodeteccion heme_oop requiere exactamente un Fe "
                f"o pasar el indice del Fe; encontrados {len(fe_candidates)}"
            )
        fe_idx = fe_candidates[0]

    if fe_idx < 1 or fe_idx > len(coords):
        raise ValueError(f"{spec.label}: indice Fe fuera de rango")
    if symbols[fe_idx - 1] != "Fe":
        print(f"[WARN] {spec.label}: el atomo {fe_idx} no parece Fe ({symbols[fe_idx - 1]}).")

    n_candidates = [
        idx
        for idx, symbol in enumerate(symbols, start=1)
        if symbol == "N" and distance(coords, fe_idx, idx) <= 3.2
    ]
    if len(n_candidates) < 4:
        n_candidates = [idx for idx, symbol in enumerate(symbols, start=1) if symbol == "N"]
    if len(n_candidates) < 4:
        raise ValueError(f"{spec.label}: no hay al menos 4 nitrogenos para definir el plano")

    chosen = choose_pyrrolic_nitrogens(coords, fe_idx, n_candidates)
    if spec.kind == "heme_oop":
        proximal = choose_proximal_nitrogen(coords, fe_idx, n_candidates, chosen)
        print(
            f"[INFO] {spec.label}: heme_oop autodetectado con Fe={fe_idx}, "
            f"N4={','.join(map(str, chosen))} y NHis={proximal}; positivo hacia NHis."
        )
        return MetricSpec(label=spec.label, kind=spec.kind, atoms=(fe_idx, *chosen, proximal))

    print(
        f"[INFO] {spec.label}: heme_oop_abs autodetectado con Fe={fe_idx} "
        f"y N4={','.join(map(str, chosen))}"
    )
    return MetricSpec(label=spec.label, kind=spec.kind, atoms=(fe_idx, *chosen))


def choose_pyrrolic_nitrogens(coords: Sequence[Coord], fe_idx: int, candidates: Sequence[int]) -> tuple[int, int, int, int]:
    from itertools import combinations

    fe = coords[fe_idx - 1]
    best: tuple[float, tuple[int, int, int, int]] | None = None
    for combo in combinations(candidates, 4):
        points = [coords[idx - 1] for idx in combo]
        plane_point, plane_normal = plane_from_four_points(points)
        if math.isnan(plane_normal[0]):
            continue
        residuals = [point_plane_signed_distance(point, plane_point, plane_normal) for point in points]
        rms = math.sqrt(sum(res * res for res in residuals) / len(residuals))
        fe_plane_distance = point_plane_signed_distance(fe, plane_point, plane_normal)
        fe_projection = sub(fe, scale(plane_normal, fe_plane_distance))
        center_offset = norm(sub(fe_projection, centroid(points)))
        radial = [norm(sub(point, fe_projection)) for point in points]
        radial_mean = sum(radial) / len(radial)
        radial_std = math.sqrt(sum((value - radial_mean) ** 2 for value in radial) / len(radial))
        score = rms + 0.35 * center_offset + 0.10 * radial_std
        if best is None or score < best[0]:
            best = (score, tuple(combo))
    if best is None:
        raise ValueError("No se pudo detectar un conjunto N4 valido para heme_oop")
    return best[1]


def choose_proximal_nitrogen(
    coords: Sequence[Coord],
    fe_idx: int,
    candidates: Sequence[int],
    pyrrolic: Sequence[int],
) -> int:
    remaining = [idx for idx in candidates if idx not in set(pyrrolic)]
    if not remaining:
        raise ValueError("No se pudo detectar N proximal para orientar heme_oop firmado")
    return min(remaining, key=lambda idx: distance(coords, fe_idx, idx))


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
    frames = parse_xyz_frames(xyz_path)
    if not frames:
        raise ValueError(f"No se encontraron frames en {xyz_path}")
    specs = resolve_metric_specs(specs, frames[0].atom_lines, frames[0].coords)
    for spec in specs:
        if spec.atoms and max(spec.atoms) > natoms:
            raise ValueError(f"{spec.label}: indice atomico fuera de rango, natoms={natoms}")

    rows: list[dict[str, float]] = []
    for frame_idx, frame in enumerate(frames, start=1):
        row: dict[str, float] = {"frame": float(frame_idx)}
        time_ps = parse_time_ps(frame.comment_line)
        if time_ps is not None:
            row["time_ps"] = time_ps
        for spec in specs:
            row[spec.label] = compute_metric(spec, frame.coords)
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
    if spec.kind in {"distance", "heme_oop"}:
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
