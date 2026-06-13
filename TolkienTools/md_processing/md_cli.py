#!/usr/bin/env python3
"""Command-line interface for molecular-dynamics processing."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import webbrowser
from dataclasses import dataclass
from pathlib import Path

from md_common import discover_segments, find_numeric_dirs, parse_d_qm_input
from md_geometry import analyze_xyz, parse_metric_spec, parse_scatter_pairs
from md_viewer import write_xyz_viewer
from md_xyz import merge_segment_xyz, parse_xyz_frames, xyz_summary


POPULATION_DEFAULTS = ("mulliken", "mulliken_spin", "lowdin", "lowdin_spin")


@dataclass(frozen=True)
class NcTrajectory:
    segment: int
    path: Path
    frames: int
    ntwx: int
    dt_ps: float
    xyz_path: Path


@dataclass(frozen=True)
class SelectedSnapshot:
    output_index: int
    global_nc_frame: int
    time_ps: float
    trajectory: NcTrajectory
    local_nc_frame: int
    qm_dynamic_frame: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Procesado general de dinamicas moleculares fragmentadas en carpetas numericas.",
        epilog=(
            "Ejemplos:\n"
            "  tolkien-tools md inspect-merge\n"
            "  tolkien-tools md inspect-merge --merge yes --exclude 2,5-7 --out qm_completo.xyz\n"
            "  tolkien-tools md geom --metric dFeN:distance:9,10\n"
            "  tolkien-tools md split-nc --count 100\n"
        ),
    )
    subparsers = parser.add_subparsers(dest="command")

    inspect_p = subparsers.add_parser(
        "inspect-merge",
        help="Inspecciona segmentos y puede mergear XYZ + cargas/spines",
    )
    add_root_args(inspect_p)
    inspect_p.add_argument("--xyz-name", default="qm.xyz")
    inspect_p.add_argument("--input-name", default="d_QM.in")
    inspect_p.add_argument("--out", default="qm_completo.xyz", help="XYZ combinado si se decide mergear")
    inspect_p.add_argument(
        "--merge",
        choices=["ask", "yes", "no"],
        default="ask",
        help="Luego de inspeccionar: preguntar, mergear automaticamente o no mergear",
    )
    inspect_p.add_argument(
        "--exclude",
        default="",
        help="Segmentos a excluir del merge. Ej: 2,5-7",
    )
    inspect_p.add_argument(
        "--pop-sources",
        nargs="+",
        default=list(POPULATION_DEFAULTS),
        help="Archivos de carga/spin a consolidar junto con el XYZ",
    )
    inspect_p.add_argument("--pop-suffix", default="_full.dat", help="Sufijo de salida para cargas/spines consolidados")
    inspect_p.add_argument("--no-pop", action="store_true", help="No consolidar cargas/spines al mergear")

    merge_xyz_p = subparsers.add_parser("merge-xyz", help="Une qm.xyz de carpetas numericas")
    add_root_args(merge_xyz_p)
    merge_xyz_p.add_argument("--xyz-name", default="qm.xyz")
    merge_xyz_p.add_argument("--input-name", default="d_QM.in")
    merge_xyz_p.add_argument("--out", default="qm_completo.xyz")

    geom_p = subparsers.add_parser("geom", help="Analiza distancias, angulos y dihedros en un XYZ multi-frame")
    geom_p.add_argument("xyz", nargs="?", default="qm_completo.xyz", help="Archivo XYZ multi-frame (default: qm_completo.xyz)")
    geom_p.add_argument("--metric", action="append", default=[], help="Formato etiqueta:tipo:atomos. Ej: dFeN:distance:9,10")
    geom_p.add_argument("--scatter", action="append", help="Par etiqueta_x,etiqueta_y. Puede repetirse")
    geom_p.add_argument("--output", default="xyz_analysis", help="Prefijo de archivos de salida")
    geom_p.add_argument("--no-plots", action="store_true", help="No generar PNG")
    geom_p.add_argument("--no-viewer", action="store_true", help="No generar visor 3D HTML del XYZ")
    geom_p.add_argument("--viewer-out", default="xyz_viewer.html", help="HTML de salida para inspeccion 3D")
    geom_p.add_argument("--viewer-frame", type=int, default=1, help="Frame del XYZ para mostrar en el visor")
    geom_p.add_argument("--viewer-labels", choices=["hover", "always"], default="always", help="Mostrar indices siempre o solo en hover")
    geom_p.add_argument("--viewer-backend", choices=["py3dmol", "plotly", "auto"], default="py3dmol", help="Motor del visor 3D")
    geom_p.add_argument("--no-open-viewer", action="store_true", help="No intentar abrir automaticamente el visor HTML")

    split_p = subparsers.add_parser("split-nc", help="Inspecciona NetCDF fragmentados y extrae rst7 con cpptraj")
    split_p.add_argument("prmtop", nargs="?", help="Topologia AMBER .prmtop; si se omite se busca automaticamente")
    split_p.add_argument("nc_pattern", nargs="?", default="QM_*.nc", help="Patron de NetCDF dentro de carpetas numericas")
    split_p.add_argument("frames", nargs="?", help="Modo legacy: all, N, A-B, N-end o lista separada por coma")
    split_p.add_argument("--root", default=".", help="Carpeta raiz con subcarpetas numericas")
    split_p.add_argument("--nc-pattern", dest="nc_pattern_option", help="Patron de NetCDF sin pasar prmtop posicional")
    split_p.add_argument("--cpptraj", help="Ruta a cpptraj; si se omite se busca en PATH")
    split_p.add_argument("--count", type=int, help="Cantidad de rst7 a extraer, distribuidos en toda la trayectoria")
    split_p.add_argument(
        "--skip-initial-ps",
        type=float,
        help="Ignorar frames NC anteriores a este tiempo global en ps antes de muestrear",
    )
    split_p.add_argument("--out-dir", default="restarts", help="Carpeta de salida para rst7 y prmtop")
    split_p.add_argument("--out-prefix", default="QM", help="Prefijo de salida")
    split_p.add_argument("--qm-xyz-out", default="qm_snapshots.xyz", help="XYZ QM de los snapshots elegidos dentro de out-dir")
    split_p.add_argument("--no-qm-xyz", action="store_true", help="No generar XYZ QM de los snapshots elegidos")

    return parser


def add_root_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", default=".", help="Carpeta raiz con subcarpetas numericas")


def cmd_inspect_merge(args: argparse.Namespace) -> None:
    root = Path(args.root)
    segments = discover_segments(root, args.input_name)
    if not segments:
        raise SystemExit(f"No se encontraron carpetas numericas en {root}")

    print(f"Root: {root.resolve()}")
    print(f"Segmentos detectados: {len(segments)}")
    print()
    print("seg  qm.xyz  frames  dt_ps      nstlim   planned_ps  real_ps    estado       archivos poblacion")

    total_frames = 0
    total_real_ps = 0.0
    total_planned_ps = 0.0
    real_segments = 0
    planned_segments = 0
    mismatched_segments = []
    for segment in segments:
        xyz_path = segment.path / args.xyz_name
        xyz_state = "no"
        frames_text = "-"
        frames = None
        if xyz_path.exists():
            try:
                _natoms, frames = xyz_summary(xyz_path)
                frames = max(frames - 1, 0)
                xyz_state = "si"
                frames_text = str(frames)
                total_frames += frames
            except ValueError as exc:
                xyz_state = f"error:{exc}"
        planned_ps = segment.planned_total_ps
        if planned_ps is not None:
            total_planned_ps += planned_ps
            planned_segments += 1
        real_ps = segment.duration_from_frames(frames) if frames is not None else None
        if real_ps is not None:
            total_real_ps += real_ps
            real_segments += 1
        state = segment_frame_state(segment.nstlim, frames)
        if state != "ok" and state != "-":
            mismatched_segments.append(segment.index)
        pop_sources = [name for name in POPULATION_DEFAULTS if (segment.path / name).exists()]
        print(
            f"{segment.index:>3}  {xyz_state:<6} {frames_text:>6}  "
            f"{format_optional(segment.dt_ps):>8}  {format_optional(segment.nstlim):>7}  "
            f"{format_optional(planned_ps):>10}  {format_optional(real_ps):>8}  "
            f"{state:<11}  {','.join(pop_sources) or '-'}"
        )

    print()
    print(f"Frames XYZ totales detectados: {total_frames}")
    if real_segments:
        print(f"Tiempo real desde frames XYZ (ps): {total_real_ps:.9f}")
    if planned_segments:
        print(f"Tiempo planeado desde d_QM.in (ps): {total_planned_ps:.9f}")
    if mismatched_segments:
        text = ", ".join(str(index) for index in mismatched_segments)
        print(f"Segmentos donde frames XYZ != nstlim: {text}")

    maybe_merge_after_inspect_merge(args, segments)


def maybe_merge_after_inspect_merge(args: argparse.Namespace, segments) -> None:
    exclude_indexes = parse_segment_selection(args.exclude) if args.exclude else set()
    if args.merge == "no":
        return
    if args.merge == "ask" and not sys.stdin.isatty():
        return

    should_merge = args.merge == "yes"
    if args.merge == "ask":
        print()
        print("Merge de XYZ inspeccionados")
        print("  Enter/n = no mergear")
        print("  a       = mergear todos los segmentos disponibles")
        print("  e       = mergear excluyendo segmentos")
        choice = input("Que queres hacer? [n/a/e]: ").strip().lower()
        if choice in {"", "n", "no"}:
            return
        if choice in {"a", "all", "t", "todos", "s", "si", "sí", "y", "yes"}:
            should_merge = True
        elif choice in {"e", "exclude", "excluir"}:
            text = input("Segmentos a excluir (ej: 2,5-7): ").strip()
            exclude_indexes = parse_segment_selection(text)
            should_merge = True
        else:
            raise SystemExit(f"Opcion invalida para merge: {choice}")

    if should_merge:
        selected_segments = [segment for segment in segments if segment.index not in exclude_indexes]
        missing = sorted(exclude_indexes - {segment.index for segment in segments})
        if missing:
            print(f"[WARN] Segmentos a excluir no detectados: {', '.join(map(str, missing))}")
        if not selected_segments:
            raise SystemExit("No quedan segmentos para mergear despues de aplicar exclusiones.")
        processed, frames, total_ps = merge_segment_xyz(selected_segments, args.xyz_name, Path(args.out))
        if frames == 0:
            raise SystemExit("No se proceso ningun frame. Revisar nombres de archivos y segmentos.")
        print()
        print(f"Segmentos mergeados: {processed}")
        if exclude_indexes:
            print(f"Segmentos excluidos: {', '.join(map(str, sorted(exclude_indexes)))}")
        print(f"Frames totales: {frames}")
        print(f"Tiempo total (ps): {total_ps:.9f}")
        print(f"XYZ combinado: {args.out}")
        if not args.no_pop:
            merge_populations_after_inspect_merge(selected_segments, args.pop_sources, Path(args.out).parent, args.pop_suffix)


def parse_segment_selection(text: str) -> set[int]:
    indexes: set[int] = set()
    if not text.strip():
        return indexes
    for part in text.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", 1)
            if not left.isdigit() or not right.isdigit():
                raise SystemExit(f"Seleccion de segmentos invalida: {token}")
            start = int(left)
            end = int(right)
            if start > end:
                raise SystemExit(f"Rango de segmentos invalido: {token}")
            indexes.update(range(start, end + 1))
        else:
            if not token.isdigit():
                raise SystemExit(f"Seleccion de segmentos invalida: {token}")
            indexes.add(int(token))
    return indexes


def segment_frame_state(nstlim: int | None, frames: int | None) -> str:
    if frames is None or nstlim is None:
        return "-"
    if frames == nstlim:
        return "ok"
    if frames < nstlim:
        return "incompleta"
    return "mas_frames"


def format_optional(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def cmd_merge_xyz(args: argparse.Namespace) -> None:
    root = Path(args.root)
    segments = discover_segments(root, args.input_name)
    if not segments:
        raise SystemExit(f"No se encontraron carpetas numericas en {root}")
    processed, frames, total_ps = merge_segment_xyz(segments, args.xyz_name, Path(args.out))
    if frames == 0:
        raise SystemExit("No se proceso ningun frame. Revisar nombres de archivos y segmentos.")
    print(f"Segmentos procesados: {processed}")
    print(f"Frames totales: {frames}")
    print(f"Tiempo total (ps): {total_ps:.9f}")
    print(f"XYZ combinado: {args.out}")


def cmd_geom(args: argparse.Namespace) -> None:
    xyz_path = Path(args.xyz)
    if not args.no_viewer:
        try:
            viewer_path = write_xyz_viewer(
                xyz_path,
                Path(args.viewer_out),
                frame_number=args.viewer_frame,
                labels=args.viewer_labels,
                backend=args.viewer_backend,
            )
            print(f"Visor 3D para elegir atomos: {viewer_path.resolve()}")
            if not args.no_open_viewer:
                open_viewer_in_browser(viewer_path)
            print("Usar el visor para rotar la molecula e inspeccionar indices atomicos.")
        except RuntimeError as exc:
            print(f"[WARN] No se genero visor 3D: {exc}")

    metrics_were_prompted = False
    metric_texts = list(args.metric)
    if not metric_texts:
        metric_texts = prompt_metric_specs()
        metrics_were_prompted = True

    specs = [parse_metric_spec(text) for text in metric_texts]
    scatter_texts = args.scatter
    if scatter_texts is None and metrics_were_prompted:
        scatter_texts = prompt_scatter_pairs(specs)
    scatter_pairs = parse_scatter_pairs(scatter_texts, specs)
    analyze_xyz(xyz_path, specs, args.output, scatter_pairs, make_plots=not args.no_plots)


def prompt_metric_specs() -> list[str]:
    print()
    print("Definir parametros geometricos. Formato: etiqueta:tipo:atomos")
    print("Tipos: distance, angle, dihedral")
    print("Ejemplos:")
    print("  dFeN:distance:9,10")
    print("  ang:angle:9,10,11")
    print("  dih:dihedral:1,2,3,4")
    print("Defina todas las metricas que quiera seguir. Cuando termine, apriete Enter sin escribir nada para avanzar.")
    metrics: list[str] = []
    while True:
        text = input(f"Metrica {len(metrics) + 1} a seguir: ").strip()
        if not text:
            break
        parse_metric_spec(text)
        metrics.append(text)
    if not metrics:
        raise SystemExit("No se definio ninguna metrica geometrica.")
    return metrics


def prompt_scatter_pairs(specs) -> list[str]:
    if len(specs) < 2:
        return []

    print()
    print("Metricas definidas:")
    for spec in specs:
        print(f"  {spec.label}: {spec.kind}({','.join(map(str, spec.atoms))})")
    print()
    print("Opcional: definir plots cruzados de dispersion entre metricas.")
    print("Formato: etiqueta_x,etiqueta_y")
    print("Ejemplo: dFeN,dFeO")
    print("Deje vacio para no generar plots cruzados.")

    pairs: list[str] = []
    while True:
        text = input(f"Plot cruzado {len(pairs) + 1}: ").strip()
        if not text:
            break
        # Validate as the user enters the pair, so typos fail early.
        parse_scatter_pairs([text], specs)
        pairs.append(text)
    return pairs


def open_viewer_in_browser(viewer_path: Path) -> None:
    open_status = open_html_in_browser(viewer_path)
    if open_status == "html":
        print("Visor 3D abierto en el navegador.")
    elif open_status == "folder":
        print(
            "[WARN] No pude abrir automaticamente el visor HTML. "
            "Se abrio la carpeta de trabajo con explorer.exe ."
        )
    else:
        print(
            "[WARN] No pude abrir automaticamente el navegador ni la carpeta. "
            "Abrir manualmente el HTML indicado arriba."
        )


def open_html_in_browser(html_path):
    """
    Intenta abrir el HTML en el navegador por defecto.
    En WSL prioriza `wslview` para abrir en Windows.
    Si no puede abrir el HTML, intenta abrir la carpeta con `explorer.exe .`.
    En WSL evita `xdg-open`/`webbrowser`, porque pueden delegar a `gio` y
    reportar exito aunque no haya aplicacion registrada para HTML.
    """
    abs_path = os.path.abspath(html_path)
    folder_path = os.path.dirname(abs_path)
    if shutil.which("wslview"):
        try:
            p = subprocess.run(
                ["wslview", abs_path],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if p.returncode == 0:
                return "html"
        except Exception:
            pass

    if running_under_wsl():
        return open_folder_with_explorer(folder_path)

    if shutil.which("xdg-open"):
        try:
            p = subprocess.run(
                ["xdg-open", abs_path],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if p.returncode == 0:
                return "html"
        except Exception:
            pass

    try:
        if bool(webbrowser.open(f"file://{abs_path}")):
            return "html"
    except Exception:
        pass

    return open_folder_with_explorer(folder_path)


def running_under_wsl() -> bool:
    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        with open("/proc/version", "r", encoding="utf-8", errors="ignore") as fh:
            return "microsoft" in fh.read().lower()
    except OSError:
        return False


def open_folder_with_explorer(folder_path: str) -> str:
    try:
        subprocess.Popen(
            ["explorer.exe", "."],
            cwd=folder_path,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return "folder"
    except Exception:
        return "none"


def merge_populations_after_inspect_merge(segments, sources: list[str], out_dir: Path, suffix: str) -> None:
    existing_sources = [source for source in sources if any((segment.path / source).exists() for segment in segments)]
    if not existing_sources:
        print("Cargas/spines: no se encontraron archivos para consolidar en los segmentos mergeados.")
        return

    print()
    print("Consolidando cargas/spines con la misma seleccion/exclusion del XYZ:")
    merge_population_sources(segments, existing_sources, out_dir, suffix)


def merge_population_sources(segments, sources: list[str], out_dir: Path, suffix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for source in sources:
        output = out_dir / f"{source}{suffix}"
        count = merge_population_source(segments, source, output)
        if count:
            print(f"{source}: {count} segmentos -> {output}")
        else:
            print(f"{source}: no encontrado")


def merge_population_source(segments, source: str, output: Path) -> int:
    count = 0
    with output.open("w", encoding="utf-8") as out:
        for segment in segments:
            path = segment.path / source
            if not path.exists():
                continue
            if write_population_source_without_initial_block(path, out):
                count += 1
                if not line_ends_with_newline(path):
                    out.write("\n")
    if count == 0:
        output.unlink(missing_ok=True)
    return count


def write_population_source_without_initial_block(path: Path, out) -> bool:
    population_blocks = 0
    wrote_line = False
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped.startswith("#") and "Population Analysis" in stripped:
                population_blocks += 1
            if population_blocks >= 2:
                out.write(line)
                wrote_line = True
    return wrote_line


def line_ends_with_newline(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            fh.seek(-1, 2)
            return fh.read(1) == b"\n"
    except OSError:
        return True


def cmd_split_nc(args: argparse.Namespace) -> None:
    cpptraj = resolve_cpptraj(args.cpptraj)
    if not cpptraj:
        raise SystemExit("cpptraj no esta disponible en PATH. Pasar --cpptraj /ruta/cpptraj.")

    root = Path(args.root)
    prmtop = resolve_prmtop(root, args.prmtop)
    nc_pattern = args.nc_pattern_option or args.nc_pattern
    if not prmtop.exists():
        raise SystemExit(f"No existe la topologia: {prmtop}")

    if args.frames is not None:
        cmd_split_nc_legacy(args, cpptraj, prmtop, nc_pattern)
        return

    trajectories = discover_nc_trajectories(root, nc_pattern, cpptraj, prmtop)
    if not trajectories:
        raise SystemExit(f"No se encontraron archivos {nc_pattern!r} en carpetas numericas de {root}")

    print_nc_summary(root, prmtop, trajectories)
    total_frames = sum(item.frames for item in trajectories)
    all_snapshots = resolve_all_snapshots(trajectories)
    skip_initial_ps = resolve_skip_initial_ps(args.skip_initial_ps)
    eligible_snapshots = [snapshot for snapshot in all_snapshots if snapshot.time_ps >= skip_initial_ps]
    if not eligible_snapshots:
        raise SystemExit(
            f"No quedan frames NC despues de aplicar skip inicial de {skip_initial_ps:.9g} ps. "
            f"La trayectoria llega hasta {all_snapshots[-1].time_ps:.9g} ps."
        )
    if skip_initial_ps > 0.0:
        print(
            f"Frames elegibles desde {skip_initial_ps:.9g} ps: "
            f"{len(eligible_snapshots)} de {total_frames}"
        )

    count = args.count if args.count is not None else prompt_restart_count(len(eligible_snapshots))
    if count < 1 or count > len(eligible_snapshots):
        raise SystemExit(f"La cantidad de rst7 debe estar entre 1 y {len(eligible_snapshots)}.")

    selected_snapshots = select_evenly_spaced_snapshots(eligible_snapshots, count)
    selected_frames = [snapshot.global_nc_frame for snapshot in selected_snapshots]
    qm_xyz_frames = [] if args.no_qm_xyz else collect_qm_xyz_frames(selected_snapshots)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    write_sampled_restarts(cpptraj, prmtop, trajectories, selected_frames, out_dir, args.out_prefix)
    shutil.copy2(prmtop, out_dir / prmtop.name)
    if not args.no_qm_xyz:
        qm_xyz_path = out_dir / args.qm_xyz_out
        write_qm_snapshot_xyz(qm_xyz_path, selected_snapshots, qm_xyz_frames)
    print()
    print(f"Listo. Se generaron {count} archivos {args.out_prefix}_*.rst7 en {out_dir}.")
    print(f"Topologia copiada: {out_dir / prmtop.name}")
    if not args.no_qm_xyz:
        print(f"XYZ QM de snapshots: {out_dir / args.qm_xyz_out}")


def resolve_cpptraj(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    conda_env = os.environ.get("CONDA_PREFIX")
    if conda_env:
        conda_cpptraj = Path(conda_env) / "bin" / "cpptraj"
        if conda_cpptraj.exists() and conda_cpptraj.is_file():
            return str(conda_cpptraj)
    return shutil.which("cpptraj")


def resolve_prmtop(root: Path, explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.exists() and not path.is_absolute():
            root_path = root / path
            if root_path.exists():
                return root_path
        return path
    prmtops = natural_sorted(root.glob("*.prmtop"))
    if not prmtops:
        raise SystemExit(f"No se encontraron archivos .prmtop en {root}")
    if len(prmtops) == 1:
        print(f"Topologia detectada: {prmtops[0]}")
        return prmtops[0]
    if not sys.stdin.isatty():
        names = ", ".join(path.name for path in prmtops)
        raise SystemExit(f"Hay mas de un .prmtop en {root}: {names}. Pasar el archivo a usar.")

    print("Los prmtop presentes son:")
    for idx, path in enumerate(prmtops, start=1):
        print(f"  {idx}. {path.name}")
    while True:
        text = input("Escriba el nombre o numero del archivo de topologia a usar: ").strip()
        if text.isdigit():
            index = int(text)
            if 1 <= index <= len(prmtops):
                return prmtops[index - 1]
        for path in prmtops:
            if text == path.name or text == str(path):
                return path
        print("Seleccion invalida.")


def cmd_split_nc_legacy(args: argparse.Namespace, cpptraj: str, prmtop: Path, nc_pattern: str) -> None:
    nc_files = natural_sorted(Path(".").glob(nc_pattern))
    if not nc_files:
        raise SystemExit(f"No se encontraron archivos con patron {nc_pattern!r}")
    if list(Path(".").glob(f"{args.out_prefix}_*.rst7")):
        raise SystemExit(f"Ya existen archivos {args.out_prefix}_*.rst7 en este directorio.")

    selected_frames = resolve_frame_selection(cpptraj, prmtop, nc_files, args.frames or "all")
    with tempfile.TemporaryDirectory(prefix=".md_split_") as tmp:
        tmpdir = Path(tmp)
        input_path = tmpdir / "cpptraj.in"
        out_prefix = tmpdir / "frames.rst7"
        with input_path.open("w", encoding="utf-8") as fh:
            fh.write(f"parm {prmtop}\n")
            for nc in nc_files:
                fh.write(f"trajin {nc}\n")
            if selected_frames is None:
                fh.write(f"trajout {out_prefix} restart multi keepext\n")
            else:
                fh.write(f"trajout {out_prefix} restart multi keepext onlyframes {','.join(map(str, selected_frames))}\n")
            fh.write("run\n")
        result = subprocess.run([cpptraj, "-i", str(input_path)], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            raise SystemExit(result.stdout)
        generated = sorted(tmpdir.glob("frames.*.rst7"))
        if not generated and out_prefix.exists():
            generated = [out_prefix]
        if not generated:
            raise SystemExit("cpptraj no genero frames.\n" + result.stdout[-2000:])
        labels = selected_frames or list(range(1, len(generated) + 1))
        if len(labels) != len(generated):
            raise SystemExit(f"cpptraj genero {len(generated)} frames, pero se esperaban {len(labels)}.")
        for label, frame_path in zip(labels, generated):
            shutil.move(str(frame_path), f"{args.out_prefix}_{label}.rst7")
    print(f"Listo. Se generaron {len(labels)} archivos {args.out_prefix}_*.rst7.")


def discover_nc_trajectories(root: Path, nc_pattern: str, cpptraj: str, prmtop: Path) -> list[NcTrajectory]:
    trajectories: list[NcTrajectory] = []
    for seg_dir in find_numeric_dirs(root):
        input_path = seg_dir / "d_QM.in"
        try:
            dt_ps, _nstlim = parse_d_qm_input(input_path)
        except (OSError, ValueError) as exc:
            raise SystemExit(f"No se pudo leer dt en {input_path}: {exc}") from exc
        ntwx = parse_amber_int_param(input_path, "ntwx") if input_path.exists() else None
        if ntwx is None or ntwx < 1:
            raise SystemExit(f"No se pudo leer ntwx valido en {input_path}. Es necesario para mapear NC -> qm.xyz.")
        xyz_path = seg_dir / "qm.xyz"
        if not xyz_path.exists():
            raise SystemExit(f"No existe {xyz_path}. Es necesario para armar el XYZ QM de snapshots.")
        for nc_path in natural_sorted(seg_dir.glob(nc_pattern)):
            if not nc_path.is_file():
                continue
            frames = get_total_frames(cpptraj, prmtop, [nc_path])
            trajectories.append(
                NcTrajectory(
                    segment=int(seg_dir.name),
                    path=nc_path,
                    frames=frames,
                    ntwx=ntwx,
                    dt_ps=dt_ps,
                    xyz_path=xyz_path,
                )
            )
    return trajectories


def print_nc_summary(root: Path, prmtop: Path, trajectories: list[NcTrajectory]) -> None:
    print(f"Root: {root.resolve()}")
    print(f"Topologia: {prmtop}")
    print()
    print("seg  archivo nc                           frames   ntwx   dt_nc_ps   ultimo frame QM")
    for item in trajectories:
        rel = item.path
        try:
            rel = item.path.relative_to(root)
        except ValueError:
            pass
        print(
            f"{item.segment:>3}  {str(rel):<35}  {item.frames:>6}  {item.ntwx:>5}  "
            f"{item.dt_ps * item.ntwx:>9.6g}  {item.frames * item.ntwx:>15}"
        )
    print()
    print(f"Archivos NC detectados: {len(trajectories)}")
    print(f"Frames totales: {sum(item.frames for item in trajectories)}")


def parse_amber_int_param(path: Path, name: str) -> int | None:
    text = path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(rf"\b{name}\s*=\s*(\d+)", text, re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))


def prompt_restart_count(total_frames: int) -> int:
    if not sys.stdin.isatty():
        raise SystemExit("Para uso no interactivo hay que pasar --count N.")
    while True:
        text = input(f"Cuantos rst7 queres generar? [1-{total_frames}]: ").strip()
        if text.isdigit():
            return int(text)
        print("Ingrese un numero entero.")


def resolve_skip_initial_ps(value: float | None) -> float:
    if value is not None:
        if value < 0.0:
            raise SystemExit("--skip-initial-ps debe ser >= 0.")
        return value
    if not sys.stdin.isatty():
        return 0.0
    text = input("Ignorar frames iniciales antes de cuantos ps? (Enter = 0): ").strip()
    if not text:
        return 0.0
    try:
        parsed = float(text)
    except ValueError as exc:
        raise SystemExit(f"Valor invalido para ps iniciales: {text}") from exc
    if parsed < 0.0:
        raise SystemExit("El tiempo inicial a ignorar debe ser >= 0.")
    return parsed


def resolve_all_snapshots(trajectories: list[NcTrajectory]) -> list[SelectedSnapshot]:
    snapshots: list[SelectedSnapshot] = []
    global_frame = 1
    global_time_ps = 0.0
    for trajectory in trajectories:
        frame_dt_ps = trajectory.dt_ps * trajectory.ntwx
        for local_frame in range(1, trajectory.frames + 1):
            time_ps = global_time_ps + local_frame * frame_dt_ps
            snapshots.append(
                SelectedSnapshot(
                    output_index=0,
                    global_nc_frame=global_frame,
                    time_ps=time_ps,
                    trajectory=trajectory,
                    local_nc_frame=local_frame,
                    qm_dynamic_frame=local_frame * trajectory.ntwx,
                )
            )
            global_frame += 1
        global_time_ps += trajectory.frames * frame_dt_ps
    return snapshots


def select_evenly_spaced_snapshots(
    eligible_snapshots: list[SelectedSnapshot],
    count: int,
) -> list[SelectedSnapshot]:
    if count == 1:
        selected_indexes = [len(eligible_snapshots) // 2]
    else:
        selected_indexes: list[int] = []
        seen: set[int] = set()
        last_index = len(eligible_snapshots) - 1
        for idx in range(count):
            selected_index = round(idx * last_index / (count - 1))
            selected_index = max(0, min(last_index, selected_index))
            while selected_index in seen and selected_index < last_index:
                selected_index += 1
            while selected_index in seen and selected_index > 0:
                selected_index -= 1
            selected_indexes.append(selected_index)
            seen.add(selected_index)
    return [
        SelectedSnapshot(
            output_index=output_index,
            global_nc_frame=snapshot.global_nc_frame,
            time_ps=snapshot.time_ps,
            trajectory=snapshot.trajectory,
            local_nc_frame=snapshot.local_nc_frame,
            qm_dynamic_frame=snapshot.qm_dynamic_frame,
        )
        for output_index, snapshot in enumerate((eligible_snapshots[i] for i in selected_indexes), start=1)
    ]


def collect_qm_xyz_frames(selected_snapshots: list[SelectedSnapshot]):
    frames_by_path = {}
    qm_frames = []
    for snapshot in selected_snapshots:
        xyz_path = snapshot.trajectory.xyz_path
        if xyz_path not in frames_by_path:
            frames_by_path[xyz_path] = parse_xyz_frames(xyz_path)[1:]
        frames = frames_by_path[xyz_path]
        if snapshot.qm_dynamic_frame < 1 or snapshot.qm_dynamic_frame > len(frames):
            raise SystemExit(
                "No se pudo mapear snapshot NC a qm.xyz: "
                f"segmento {snapshot.trajectory.segment}, frame NC local {snapshot.local_nc_frame}, "
                f"frame QM esperado {snapshot.qm_dynamic_frame}, frames QM disponibles {len(frames)}."
            )
        qm_frames.append(frames[snapshot.qm_dynamic_frame - 1])
    return qm_frames


def write_qm_snapshot_xyz(path: Path, selected_snapshots: list[SelectedSnapshot], qm_frames) -> None:
    with path.open("w", encoding="utf-8") as out:
        for snapshot, frame in zip(selected_snapshots, qm_frames):
            out.write(f"{frame.natoms_line.strip()}\n")
            out.write(
                f"restart_index={snapshot.output_index} "
                f"global_nc_frame={snapshot.global_nc_frame} "
                f"time_ps={snapshot.time_ps:.9f} "
                f"segment={snapshot.trajectory.segment} "
                f"nc_file={snapshot.trajectory.path.name} "
                f"nc_frame={snapshot.local_nc_frame} "
                f"qm_dynamic_frame={snapshot.qm_dynamic_frame} "
                f"ntwx={snapshot.trajectory.ntwx}\n"
            )
            for atom_line in frame.atom_lines:
                out.write(f"{atom_line.rstrip()}\n")


def write_sampled_restarts(
    cpptraj: str,
    prmtop: Path,
    trajectories: list[NcTrajectory],
    selected_frames: list[int],
    out_dir: Path,
    out_prefix: str,
) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"La carpeta de salida {out_dir} ya existe y no esta vacia.")
    out_dir.mkdir(parents=True, exist_ok=True)
    if list(out_dir.glob(f"{out_prefix}_*.rst7")):
        raise SystemExit(f"Ya existen archivos {out_prefix}_*.rst7 en {out_dir}.")

    with tempfile.TemporaryDirectory(prefix=".md_split_") as tmp:
        tmpdir = Path(tmp)
        input_path = tmpdir / "cpptraj.in"
        out_prefix_path = tmpdir / "frames.rst7"
        with input_path.open("w", encoding="utf-8") as fh:
            fh.write(f"parm {prmtop}\n")
            for item in trajectories:
                fh.write(f"trajin {item.path}\n")
            fh.write(
                f"trajout {out_prefix_path} restart multi keepext "
                f"onlyframes {','.join(map(str, selected_frames))}\n"
            )
            fh.write("run\n")
        result = subprocess.run([cpptraj, "-i", str(input_path)], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            raise SystemExit(result.stdout)
        generated = natural_sorted(tmpdir.glob("frames.*.rst7"))
        if not generated and out_prefix_path.exists():
            generated = [out_prefix_path]
        if not generated:
            raise SystemExit("cpptraj no genero frames.\n" + result.stdout[-2000:])
        if len(generated) != len(selected_frames):
            raise SystemExit(f"cpptraj genero {len(generated)} frames, pero se esperaban {len(selected_frames)}.")
        for output_index, frame_path in enumerate(generated, start=1):
            shutil.move(str(frame_path), out_dir / f"{out_prefix}_{output_index}.rst7")


def natural_sorted(paths) -> list[Path]:
    def key(path: Path) -> list[object]:
        return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(path))]

    return sorted(paths, key=key)


def resolve_frame_selection(cpptraj: str, prmtop: Path, nc_files: list[Path], frame_spec: str) -> list[int] | None:
    if frame_spec in {"all", "todas", "*"}:
        return None
    total = get_total_frames(cpptraj, prmtop, nc_files)
    frames: list[int] = []
    seen: set[int] = set()
    for part in frame_spec.split(","):
        part = part.strip()
        if part.isdigit():
            candidates = [int(part)]
        elif part.endswith("-end") and part[:-4].isdigit():
            start = int(part[:-4])
            candidates = list(range(start, total + 1))
        elif "-" in part:
            left, right = part.split("-", 1)
            if not left.isdigit() or not right.isdigit():
                raise SystemExit(f"Seleccion de frames invalida: {part}")
            start = int(left)
            end = int(right)
            if start > end:
                raise SystemExit(f"Rango invalido: {part}")
            candidates = list(range(start, end + 1))
        else:
            raise SystemExit(f"Seleccion de frames invalida: {part}")
        for frame in candidates:
            if frame < 1 or frame > total:
                raise SystemExit(f"Frame {frame} fuera de rango. La trayectoria tiene {total} frames.")
            if frame in seen:
                raise SystemExit(f"Frame repetido en seleccion: {frame}")
            seen.add(frame)
            frames.append(frame)
    return frames


def get_total_frames(cpptraj: str, prmtop: Path, nc_files: list[Path]) -> int:
    cmd = [cpptraj, "-p", str(prmtop)]
    for nc in nc_files:
        cmd.extend(["-y", str(nc)])
    cmd.append("-tl")
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in result.stdout.splitlines():
        if line.startswith("Frames:"):
            value = line.split()[1]
            if value.isdigit():
                return int(value)
    raise SystemExit("No pude determinar la cantidad de frames con cpptraj -tl.\n" + result.stdout[-2000:])


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return

    commands = {
        "inspect-merge": cmd_inspect_merge,
        "merge-xyz": cmd_merge_xyz,
        "geom": cmd_geom,
        "split-nc": cmd_split_nc,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main(sys.argv[1:])
