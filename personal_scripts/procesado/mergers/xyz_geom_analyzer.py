#!/usr/bin/env python3
"""Analyze geometric parameters from a multi-frame XYZ trajectory."""

import argparse
import csv
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


Coord = Tuple[float, float, float]


@dataclass(frozen=True)
class MetricSpec:
    label: str
    kind: str
    atoms: Tuple[int, ...]


@dataclass
class XyzInfo:
    natoms: int
    nframes: int
    first_comment: str
    last_comment: str
    elements: List[str]


def parse_float(text: str) -> float:
    return float(text.replace("d", "e").replace("D", "E"))


def parse_time_ps(comment: str) -> Optional[float]:
    match = re.search(r"\btime_ps\s*=\s*([-+]?\d*\.?\d+(?:[eEdD][-+]?\d+)?)", comment)
    if not match:
        return None
    return parse_float(match.group(1))


def read_xyz_info(path: Path) -> XyzInfo:
    natoms: Optional[int] = None
    nframes = 0
    first_comment = ""
    last_comment = ""
    elements: List[str] = []

    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        while True:
            natoms_line = fh.readline()
            if not natoms_line:
                break
            if not natoms_line.strip():
                continue

            try:
                current_natoms = int(natoms_line.strip())
            except ValueError as exc:
                raise ValueError(f"Formato XYZ inválido cerca del frame {nframes + 1}: {natoms_line.rstrip()}") from exc

            if natoms is None:
                natoms = current_natoms
            elif current_natoms != natoms:
                raise ValueError(
                    f"Número de átomos variable: frame {nframes + 1} tiene {current_natoms}, esperado {natoms}"
                )

            comment = fh.readline()
            if not comment:
                raise ValueError(f"Falta línea de comentario en frame {nframes + 1}")
            comment = comment.rstrip("\n")
            if nframes == 0:
                first_comment = comment
            last_comment = comment

            frame_elements: List[str] = []
            for atom_idx in range(current_natoms):
                line = fh.readline()
                if not line:
                    raise ValueError(f"Frame truncado en frame {nframes + 1}, átomo {atom_idx + 1}")
                parts = line.split()
                if len(parts) < 4:
                    raise ValueError(f"Línea atómica inválida en frame {nframes + 1}, átomo {atom_idx + 1}")
                frame_elements.append(parts[0])

            if not elements:
                elements = frame_elements
            nframes += 1

    if natoms is None or nframes == 0:
        raise ValueError(f"No se encontraron frames XYZ en {path}")

    return XyzInfo(
        natoms=natoms,
        nframes=nframes,
        first_comment=first_comment,
        last_comment=last_comment,
        elements=elements,
    )


def iter_xyz_frames(path: Path) -> Iterable[Tuple[int, Optional[float], str, List[Coord]]]:
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        frame_idx = 0
        while True:
            natoms_line = fh.readline()
            if not natoms_line:
                break
            if not natoms_line.strip():
                continue
            natoms = int(natoms_line.strip())
            comment = fh.readline()
            if not comment:
                raise ValueError(f"Falta línea de comentario en frame {frame_idx + 1}")
            comment = comment.rstrip("\n")

            coords: List[Coord] = []
            for atom_idx in range(natoms):
                line = fh.readline()
                if not line:
                    raise ValueError(f"Frame truncado en frame {frame_idx + 1}, átomo {atom_idx + 1}")
                parts = line.split()
                if len(parts) < 4:
                    raise ValueError(f"Línea atómica inválida en frame {frame_idx + 1}, átomo {atom_idx + 1}")
                coords.append((parse_float(parts[-3]), parse_float(parts[-2]), parse_float(parts[-1])))

            frame_idx += 1
            yield frame_idx, parse_time_ps(comment), comment, coords


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


def compute_metric(spec: MetricSpec, coords: Sequence[Coord]) -> float:
    if spec.kind == "distance":
        return distance(coords, spec.atoms[0], spec.atoms[1])
    if spec.kind == "angle":
        return angle(coords, spec.atoms[0], spec.atoms[1], spec.atoms[2])
    if spec.kind == "dihedral":
        return dihedral(coords, spec.atoms[0], spec.atoms[1], spec.atoms[2], spec.atoms[3])
    raise ValueError(f"Tipo de métrica no soportado: {spec.kind}")


def expected_atom_count(kind: str) -> int:
    counts = {"distance": 2, "angle": 3, "dihedral": 4}
    if kind not in counts:
        raise ValueError(f"Tipo inválido '{kind}'. Usar: distance, angle o dihedral")
    return counts[kind]


def parse_metric_spec(text: str) -> MetricSpec:
    parts = text.split(":")
    if len(parts) != 3:
        raise ValueError(f"Métrica inválida '{text}'. Formato: etiqueta:tipo:atomos")
    label = parts[0].strip()
    kind = parts[1].strip().lower()
    atoms = tuple(int(a.strip()) for a in parts[2].split(",") if a.strip())
    if not label:
        raise ValueError("La etiqueta de la métrica no puede estar vacía")
    expected = expected_atom_count(kind)
    if len(atoms) != expected:
        raise ValueError(f"{label}: tipo {kind} requiere {expected} átomos")
    if any(a < 1 for a in atoms):
        raise ValueError(f"{label}: los índices atómicos deben ser >= 1")
    return MetricSpec(label=label, kind=kind, atoms=atoms)


def prompt_metric_specs(natoms: int) -> List[MetricSpec]:
    specs: List[MetricSpec] = []
    print("\nDefiní parámetros geométricos. Tipos: distance, angle, dihedral.")
    print("Usá índices 1-based, como en la lista de átomos del XYZ.")
    print("Ejemplo: dNH:distance:2,3")
    print("Dejá la etiqueta vacía para terminar.\n")

    while True:
        label = input("Etiqueta del parámetro: ").strip()
        if not label:
            break
        kind = input("Tipo [distance/angle/dihedral]: ").strip().lower()
        expected = expected_atom_count(kind)
        atoms_text = input(f"Átomos separados por coma ({expected} índices): ").strip()
        atoms = tuple(int(a.strip()) for a in atoms_text.split(",") if a.strip())
        spec = MetricSpec(label=label, kind=kind, atoms=atoms)
        validate_metric_specs([spec], natoms)
        specs.append(spec)

    if not specs:
        raise SystemExit("No se definió ningún parámetro.")
    return specs


def validate_metric_specs(specs: Sequence[MetricSpec], natoms: int) -> None:
    labels = set()
    for spec in specs:
        if spec.label in labels:
            raise ValueError(f"Etiqueta repetida: {spec.label}")
        labels.add(spec.label)
        expected = expected_atom_count(spec.kind)
        if len(spec.atoms) != expected:
            raise ValueError(f"{spec.label}: tipo {spec.kind} requiere {expected} átomos")
        if max(spec.atoms) > natoms:
            raise ValueError(f"{spec.label}: índice atómico fuera de rango, natoms={natoms}")


def select_scatter_pairs(specs: Sequence[MetricSpec], pairs: Optional[Sequence[str]]) -> List[Tuple[str, str]]:
    labels = {s.label for s in specs}
    if pairs:
        selected = []
        for pair in pairs:
            if "," not in pair:
                raise ValueError(f"Par scatter inválido '{pair}'. Usar etiqueta_x,etiqueta_y")
            left, right = [p.strip() for p in pair.split(",", 1)]
            if left not in labels or right not in labels:
                raise ValueError(f"Par scatter desconocido '{pair}'. Etiquetas disponibles: {', '.join(sorted(labels))}")
            selected.append((left, right))
        return selected

    if len(specs) < 2:
        return []

    answer = input("\n¿Querés hacer scatter entre dos parámetros? [s/N]: ").strip().lower()
    if answer not in {"s", "si", "sí", "y", "yes"}:
        return []

    print("Parámetros disponibles:", ", ".join(s.label for s in specs))
    left = input("Eje x: ").strip()
    right = input("Eje y: ").strip()
    if left not in labels or right not in labels:
        raise SystemExit("Alguna etiqueta no existe.")
    return [(left, right)]


def analyze_xyz(
    path: Path,
    specs: Sequence[MetricSpec],
    output_prefix: str,
    scatter_pairs: Sequence[Tuple[str, str]],
    make_plots: bool,
) -> None:
    rows: List[Dict[str, float]] = []

    for frame_idx, time_ps, _comment, coords in iter_xyz_frames(path):
        row: Dict[str, float] = {"frame": float(frame_idx)}
        if time_ps is not None:
            row["time_ps"] = time_ps
        for spec in specs:
            row[spec.label] = compute_metric(spec, coords)
        rows.append(row)

    headers = ["frame"]
    if any("time_ps" in r for r in rows):
        headers.append("time_ps")
    headers.extend(spec.label for spec in specs)

    csv_path = Path(f"{output_prefix}_metrics.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({h: format_value(row.get(h)) for h in headers})

    stats_path = Path(f"{output_prefix}_stats.csv")
    with stats_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["label", "kind", "atoms", "n", "min", "max", "mean", "std"])
        for spec in specs:
            vals = [row[spec.label] for row in rows if not math.isnan(row[spec.label])]
            writer.writerow([spec.label, spec.kind, "-".join(map(str, spec.atoms)), len(vals), *summary(vals)])

    corr_path = None
    if scatter_pairs:
        corr_path = Path(f"{output_prefix}_scatter_stats.csv")
        with corr_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["x", "y", "n", "pearson_r", "slope", "intercept"])
            for x_label, y_label in scatter_pairs:
                xs = [row[x_label] for row in rows]
                ys = [row[y_label] for row in rows]
                writer.writerow([x_label, y_label, len(xs), *linear_stats(xs, ys)])

    if make_plots:
        plot_outputs(rows, specs, output_prefix, scatter_pairs)

    print(f"CSV de métricas: {csv_path}")
    print(f"Resumen estadístico: {stats_path}")
    if corr_path is not None:
        print(f"Correlaciones scatter: {corr_path}")
    if make_plots:
        print(f"Figuras PNG: {output_prefix}_*.png")


def format_value(value: Optional[float]) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return f"{value:.9f}"


def summary(values: Sequence[float]) -> Tuple[str, str, str, str]:
    if not values:
        return ("nan", "nan", "nan", "nan")
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return (f"{min(values):.9f}", f"{max(values):.9f}", f"{mean:.9f}", f"{math.sqrt(var):.9f}")


def linear_stats(xs: Sequence[float], ys: Sequence[float]) -> Tuple[str, str, str]:
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
    r = sxy / math.sqrt(sxx * syy)
    slope = sxy / sxx
    intercept = ymean - slope * xmean
    return (f"{r:.9f}", f"{slope:.9f}", f"{intercept:.9f}")


def plot_outputs(
    rows: Sequence[Dict[str, float]],
    specs: Sequence[MetricSpec],
    output_prefix: str,
    scatter_pairs: Sequence[Tuple[str, str]],
) -> None:
    try:
        mpl_config = Path(".matplotlib-cache").resolve()
        mpl_config.mkdir(exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("No se pudo importar matplotlib. Instalalo o corré con --no-plots.") from exc

    x_axis = "time_ps" if any("time_ps" in r for r in rows) else "frame"
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
            yfit = [float(slope) * xmin + float(intercept), float(slope) * xmax + float(intercept)]
            ax.plot([xmin, xmax], yfit, color="tab:red", lw=1.2)
        fig.savefig(f"{output_prefix}_scatter_{x_label}_vs_{y_label}.png", dpi=180)
        plt.close(fig)


def axis_label(spec: MetricSpec) -> str:
    if spec.kind == "distance":
        return f"{spec.label} (A)"
    return f"{spec.label} (deg)"


def print_atom_table(info: XyzInfo, max_rows: int) -> None:
    print("\nÁtomos del primer frame:")
    for idx, element in enumerate(info.elements[:max_rows], start=1):
        print(f"  {idx:4d}  {element}")
    if len(info.elements) > max_rows:
        print(f"  ... ({len(info.elements) - max_rows} más)")


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Analiza distancias, ángulos y dihedros en un XYZ multi-frame.",
        epilog=(
            "Ejemplos:\n"
            "  Modo interactivo:\n"
            "    python3 xyz_geom_analyzer.py qm_completo.xyz\n\n"
            "  Scatter dNH vs dFeN, usando índices 2-3 y 9-2:\n"
            "    python3 xyz_geom_analyzer.py qm_completo.xyz \\\n"
            "      --metric dNH:distance:2,3 --metric dFeN:distance:9,2 \\\n"
            "      --scatter dNH,dFeN --output his_fe_nh\n"
        ),
    )
    parser.add_argument("xyz", help="Archivo XYZ multi-frame")
    parser.add_argument("--metric", action="append", default=[], help="Formato etiqueta:tipo:atomos. Ej: dNH:distance:2,3")
    parser.add_argument("--scatter", action="append", help="Par etiqueta_x,etiqueta_y. Puede repetirse.")
    parser.add_argument("--output", default="xyz_analysis", help="Prefijo de archivos de salida")
    parser.add_argument("--no-plots", action="store_true", help="No generar PNG")
    parser.add_argument("--show-atoms", type=int, default=80, help="Cantidad de átomos a listar del primer frame")

    args = parser.parse_args()
    xyz_path = Path(args.xyz)
    if not xyz_path.exists():
        raise SystemExit(f"No existe el archivo: {xyz_path}")

    info = read_xyz_info(xyz_path)
    print(f"Archivo: {xyz_path}")
    print(f"Átomos por frame: {info.natoms}")
    print(f"Frames: {info.nframes}")
    print(f"Primer comentario: {info.first_comment}")
    print(f"Último comentario: {info.last_comment}")
    print_atom_table(info, args.show_atoms)

    specs = [parse_metric_spec(text) for text in args.metric]
    if not specs:
        specs = prompt_metric_specs(info.natoms)
    validate_metric_specs(specs, info.natoms)

    scatter_pairs = select_scatter_pairs(specs, args.scatter)
    analyze_xyz(
        path=xyz_path,
        specs=specs,
        output_prefix=args.output,
        scatter_pairs=scatter_pairs,
        make_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
