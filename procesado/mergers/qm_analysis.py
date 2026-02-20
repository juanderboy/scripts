#!/usr/bin/env python3
import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


@dataclass
class Frame:
    natoms_line: str
    comment_line: str
    atom_lines: List[str]
    coords: List[Tuple[float, float, float]]


def parse_float(text: str) -> float:
    return float(text.replace("d", "e").replace("D", "E"))


def find_numeric_dirs(root: Path) -> List[Path]:
    dirs = []
    for p in root.iterdir():
        if p.is_dir() and p.name.isdigit():
            dirs.append(p)
    dirs.sort(key=lambda p: int(p.name))
    return dirs


def parse_d_qm_input(path: Path) -> Tuple[float, int]:
    text = path.read_text(encoding="utf-8", errors="ignore")

    dt_match = re.search(r"\bdt\s*=\s*([-+]?\d*\.?\d+(?:[eEdD][-+]?\d+)?)", text, re.IGNORECASE)
    if not dt_match:
        raise ValueError(f"No se encontró 'dt' en {path}")
    dt = parse_float(dt_match.group(1))

    nst_match = re.search(r"\b(?:nstlim|ntslim)\s*=\s*(\d+)", text, re.IGNORECASE)
    if not nst_match:
        raise ValueError(f"No se encontró 'nstlim' (o 'ntslim') en {path}")
    nstlim = int(nst_match.group(1))

    return dt, nstlim


def parse_xyz_frames(path: Path) -> List[Frame]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    i = 0
    frames: List[Frame] = []

    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue

        try:
            natoms = int(lines[i].strip())
        except ValueError as exc:
            raise ValueError(f"Formato XYZ inválido en {path}, línea {i + 1}: '{lines[i]}'") from exc

        natoms_line = lines[i]
        i += 1

        if i >= len(lines):
            raise ValueError(f"Falta línea de comentario para frame en {path}")
        comment_line = lines[i]
        i += 1

        if i + natoms > len(lines):
            raise ValueError(f"Frame truncado en {path}, línea {i + 1}")

        atom_lines = lines[i : i + natoms]
        coords: List[Tuple[float, float, float]] = []
        for idx, aline in enumerate(atom_lines, start=1):
            parts = aline.split()
            if len(parts) < 4:
                raise ValueError(f"Línea atómica inválida en {path}, frame {len(frames) + 1}, átomo {idx}")
            x = parse_float(parts[-3])
            y = parse_float(parts[-2])
            z = parse_float(parts[-1])
            coords.append((x, y, z))

        frames.append(Frame(natoms_line=natoms_line, comment_line=comment_line, atom_lines=atom_lines, coords=coords))
        i += natoms

    return frames


def vec_sub(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def norm(v: Sequence[float]) -> float:
    return math.sqrt(dot(v, v))


def distance(coords: Sequence[Tuple[float, float, float]], i: int, j: int) -> float:
    a = coords[i - 1]
    b = coords[j - 1]
    return norm(vec_sub(a, b))


def angle(coords: Sequence[Tuple[float, float, float]], i: int, j: int, k: int) -> float:
    a = coords[i - 1]
    b = coords[j - 1]
    c = coords[k - 1]
    v1 = vec_sub(a, b)
    v2 = vec_sub(c, b)
    cang = dot(v1, v2) / (norm(v1) * norm(v2))
    cang = max(-1.0, min(1.0, cang))
    return math.degrees(math.acos(cang))


def dihedral(coords: Sequence[Tuple[float, float, float]], i: int, j: int, k: int, l: int) -> float:
    p1 = coords[i - 1]
    p2 = coords[j - 1]
    p3 = coords[k - 1]
    p4 = coords[l - 1]

    b1 = vec_sub(p2, p1)
    b2 = vec_sub(p3, p2)
    b3 = vec_sub(p4, p3)

    n1 = cross(b1, b2)
    n2 = cross(b2, b3)

    b2u_norm = norm(b2)
    if b2u_norm == 0:
        raise ValueError("No se puede calcular dihedro: vector b2 nulo")
    b2u = (b2[0] / b2u_norm, b2[1] / b2u_norm, b2[2] / b2u_norm)

    m1 = cross(n1, b2u)

    x = dot(n1, n2)
    y = dot(m1, n2)
    angle_deg = math.degrees(math.atan2(y, x))
    # Mantener dihedros en la escala [0, 360) para series temporales continuas.
    if angle_deg < 0.0:
        angle_deg += 360.0
    return angle_deg


def compute_metric(kind: str, atoms: Sequence[int], coords: Sequence[Tuple[float, float, float]]) -> float:
    if kind == "distance":
        return distance(coords, atoms[0], atoms[1])
    if kind == "angle":
        return angle(coords, atoms[0], atoms[1], atoms[2])
    if kind == "dihedral":
        return dihedral(coords, atoms[0], atoms[1], atoms[2], atoms[3])
    raise ValueError(f"Métrica no soportada: {kind}")


def expected_atoms(kind: str) -> int:
    return {"distance": 2, "angle": 3, "dihedral": 4}[kind]


def write_merged_xyz(path: Path, records: Iterable[Tuple[int, int, float, Frame]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for seg, seg_frame, time_ps, frame in records:
            fh.write(f"{frame.natoms_line.strip()}\n")
            fh.write(f"segment={seg} frame={seg_frame} time_ps={time_ps:.9f}\n")
            for line in frame.atom_lines:
                fh.write(f"{line.rstrip()}\n")


def build_histogram(values: Sequence[float], nbins: int) -> Tuple[List[Tuple[float, float, int, float]], float, float]:
    if nbins < 1:
        raise ValueError("El número de bins debe ser >= 1")
    if not values:
        raise ValueError("No hay valores para construir histograma")

    vmin = min(values)
    vmax = max(values)
    if math.isclose(vmin, vmax):
        width = 1.0
    else:
        width = (vmax - vmin) / nbins

    counts = [0 for _ in range(nbins)]
    for v in values:
        if math.isclose(vmin, vmax):
            idx = 0
        else:
            idx = int((v - vmin) / width)
            if idx == nbins:
                idx = nbins - 1
        counts[idx] += 1

    total = len(values)
    rows: List[Tuple[float, float, int, float]] = []
    for i, c in enumerate(counts):
        start = vmin + i * width
        end = start + width
        density = c / total
        rows.append((start, end, c, density))
    return rows, vmin, vmax


def find_histogram_peaks(
    hist_rows: Sequence[Tuple[float, float, int, float]],
    circular: bool,
) -> List[Tuple[float, int, float]]:
    counts = [r[2] for r in hist_rows]
    if len(counts) < 3:
        return []

    peaks: List[Tuple[float, int, float]] = []
    n = len(counts)
    for i in range(n):
        left_i = (i - 1) % n if circular else i - 1
        right_i = (i + 1) % n if circular else i + 1

        if left_i < 0 or right_i >= n:
            continue

        c = counts[i]
        if c >= counts[left_i] and c >= counts[right_i] and (c > counts[left_i] or c > counts[right_i]):
            start, end, _, frac = hist_rows[i]
            center = 0.5 * (start + end)
            peaks.append((center, c, frac))

    peaks.sort(key=lambda p: p[1], reverse=True)
    return peaks


def write_stats_report(
    out_path: Path,
    metric_name: str,
    values: Sequence[float],
    hist_rows: Sequence[Tuple[float, float, int, float]],
    peaks: Sequence[Tuple[float, int, float]],
) -> None:
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(var)
    vmin = min(values)
    vmax = max(values)

    with out_path.open("w", encoding="utf-8") as fh:
        fh.write(f"metric={metric_name}\n")
        fh.write(f"n_frames={n}\n")
        fh.write(f"min={vmin:.9f}\n")
        fh.write(f"max={vmax:.9f}\n")
        fh.write(f"mean={mean:.9f}\n")
        fh.write(f"std={std:.9f}\n")
        fh.write(f"n_bins={len(hist_rows)}\n")
        fh.write(f"n_peaks={len(peaks)}\n")
        fh.write("top_peaks_bin_center,count,fraction\n")
        for center, count, frac in peaks[:10]:
            fh.write(f"{center:.9f},{count},{frac:.9f}\n")


def plot_analysis_panels(
    out_path: Path,
    metric_name: str,
    times_ps: Sequence[float],
    values: Sequence[float],
    hist_rows: Sequence[Tuple[float, float, int, float]],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "No se pudo importar matplotlib. Instalalo para usar --analyze con ploteo."
        ) from exc

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

    ax_time = axes[0]
    ax_time.plot(times_ps, values, lw=0.8, color="tab:blue")
    ax_time.set_xlabel("time (ps)")
    ax_time.set_ylabel(metric_name)
    ax_time.set_title(f"Evolucion temporal: {metric_name}")
    ax_time.grid(True, alpha=0.25)
    if metric_name == "dihedral":
        ax_time.set_ylim(0.0, 360.0)

    ax_hist = axes[1]
    starts = [row[0] for row in hist_rows]
    ends = [row[1] for row in hist_rows]
    counts = [row[2] for row in hist_rows]
    widths = [e - s for s, e in zip(starts, ends)]
    centers = [0.5 * (s + e) for s, e in zip(starts, ends)]
    ax_hist.bar(centers, counts, width=widths, color="tab:orange", edgecolor="black", linewidth=0.3)
    ax_hist.set_xlabel(metric_name)
    ax_hist.set_ylabel("count")
    ax_hist.set_title(f"Histograma: {metric_name}")
    ax_hist.grid(True, alpha=0.25)
    if metric_name == "dihedral":
        ax_hist.set_xlim(0.0, 360.0)

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            "Une qm.xyz de carpetas numéricas (1,2,3,...) con eje temporal desde d_QM.in "
            "y opcionalmente calcula una métrica geométrica a CSV."
        ),
        epilog=(
            "Ejemplos:\n"
            "  1) Solo merge de XYZ:\n"
            "     python3 qm_rutina.py --merged-out qm_completo.xyz\n\n"
            "  2) Distancia 1-2 vs tiempo:\n"
            "     python3 qm_rutina.py --metric distance --atoms 1 2 --csv-out dist_1_2.csv\n\n"
            "  3) Dihedro 1-2-3-4 + análisis de poblaciones:\n"
            "     python3 qm_rutina.py --metric dihedral --atoms 1 2 3 4 \\\n"
            "       --csv-out dih.csv --analyze --hist-bins 72 \\\n"
            "       --hist-out dih_hist.csv --stats-out dih_stats.txt\n"
        ),
    )
    parser.add_argument("--root", default=".", help="Carpeta raíz que contiene subcarpetas numéricas")
    parser.add_argument("--xyz-name", default="qm.xyz", help="Nombre del XYZ dentro de cada carpeta")
    parser.add_argument("--input-name", default="d_QM.in", help="Nombre del input AMBER dentro de cada carpeta")
    parser.add_argument("--merged-out", default="qm_completo.xyz", help="Archivo XYZ de salida combinado")
    parser.add_argument("--skip-merged", action="store_true", help="No escribir el XYZ combinado")
    parser.add_argument("--metric", choices=["distance", "angle", "dihedral"], help="Métrica geométrica para CSV")
    parser.add_argument("--atoms", nargs="+", type=int, help="Índices atómicos 1-based (2, 3 o 4 según métrica)")
    parser.add_argument("--csv-out", default="geom_time.csv", help="CSV de salida para métrica vs tiempo")
    parser.add_argument("--analyze", action="store_true", help="Genera histograma y resumen estadístico del parámetro")
    parser.add_argument("--hist-bins", type=int, default=72, help="Número de bins para histograma")
    parser.add_argument("--hist-out", default="geom_hist.csv", help="CSV de histograma")
    parser.add_argument("--stats-out", default="geom_stats.txt", help="Reporte de estadísticos y picos")
    parser.add_argument("--plot-out", default="geom_panels.png", help="Figura PNG con panel temporal + histograma")

    args = parser.parse_args()

    root = Path(args.root)
    numeric_dirs = find_numeric_dirs(root)
    if not numeric_dirs:
        raise SystemExit(f"No se encontraron carpetas numéricas en: {root}")

    if args.metric:
        nat = expected_atoms(args.metric)
        if not args.atoms or len(args.atoms) != nat:
            raise SystemExit(f"Para metric='{args.metric}' se requieren {nat} índices en --atoms")
        if any(a < 1 for a in args.atoms):
            raise SystemExit("Los índices atómicos deben ser >= 1")

    merged_records: List[Tuple[int, int, float, Frame]] = []
    csv_rows: List[List[object]] = []
    metric_values: List[float] = []
    metric_times_ps: List[float] = []

    global_time_ps = 0.0
    global_frame_idx = 0
    skipped = 0

    for seg_dir in numeric_dirs:
        xyz_path = seg_dir / args.xyz_name
        inp_path = seg_dir / args.input_name
        if not xyz_path.exists() or not inp_path.exists():
            skipped += 1
            continue

        dt_ps, nstlim = parse_d_qm_input(inp_path)
        frames = parse_xyz_frames(xyz_path)
        if not frames:
            skipped += 1
            continue

        nframes = len(frames)
        run_total_ps = dt_ps * nstlim
        frame_dt_ps = run_total_ps / nframes

        for seg_frame, frame in enumerate(frames, start=1):
            time_ps = global_time_ps
            global_frame_idx += 1
            merged_records.append((int(seg_dir.name), seg_frame, time_ps, frame))

            if args.metric:
                if max(args.atoms) > len(frame.coords):
                    raise SystemExit(
                        f"Índice atómico fuera de rango en segmento {seg_dir.name}, frame {seg_frame} "
                        f"(natoms={len(frame.coords)})"
                    )
                value = compute_metric(args.metric, args.atoms, frame.coords)
                csv_rows.append([f"{time_ps:.9f}", f"{value:.9f}"])
                metric_values.append(value)
                metric_times_ps.append(time_ps)

            global_time_ps += frame_dt_ps

    if not merged_records:
        raise SystemExit("No se procesó ningún frame. Revisá nombres de archivos y contenido.")

    if not args.skip_merged:
        write_merged_xyz(Path(args.merged_out), merged_records)

    if args.metric:
        with Path(args.csv_out).open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["time_ps", args.metric])
            writer.writerows(csv_rows)

        if args.analyze:
            hist_rows, _, _ = build_histogram(metric_values, args.hist_bins)
            circular = args.metric == "dihedral"
            peaks = find_histogram_peaks(hist_rows, circular=circular)

            with Path(args.hist_out).open("w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(["bin_start", "bin_end", "count", "fraction"])
                for start, end, count, frac in hist_rows:
                    writer.writerow([f"{start:.9f}", f"{end:.9f}", count, f"{frac:.9f}"])

            write_stats_report(Path(args.stats_out), args.metric, metric_values, hist_rows, peaks)
            plot_analysis_panels(
                Path(args.plot_out),
                args.metric,
                metric_times_ps,
                metric_values,
                hist_rows,
            )

    processed_segments = len({seg for seg, _, _, _ in merged_records})
    print(f"Segmentos procesados: {processed_segments}")
    print(f"Frames totales: {len(merged_records)}")
    print(f"Tiempo total (ps): {global_time_ps:.9f}")
    if skipped:
        print(f"Segmentos omitidos (faltantes/vacíos): {skipped}")
    if not args.skip_merged:
        print(f"XYZ combinado: {args.merged_out}")
    if args.metric:
        print(f"CSV métrica: {args.csv_out}")
        if args.analyze:
            print(f"Histograma: {args.hist_out}")
            print(f"Reporte estadístico: {args.stats_out}")
            print(f"Figura paneles: {args.plot_out}")


if __name__ == "__main__":
    main()
