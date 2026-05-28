#!/usr/bin/env python3
"""Command-line interface for molecular-dynamics processing."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from md_common import discover_segments
from md_geometry import analyze_xyz, parse_metric_spec, parse_scatter_pairs
from md_xyz import merge_segment_xyz, xyz_summary


POPULATION_DEFAULTS = ("mulliken", "mulliken_spin", "lowdin", "lowdin_spin")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Procesado general de dinamicas moleculares fragmentadas en carpetas numericas.",
        epilog=(
            "Ejemplos:\n"
            "  tolkien-tools md inspect\n"
            "  tolkien-tools md merge-xyz --out qm_completo.xyz\n"
            "  tolkien-tools md geom qm_completo.xyz --metric dFeN:distance:9,10\n"
            "  tolkien-tools md merge-pop --sources mulliken_spin lowdin_spin\n"
            "  tolkien-tools md spin-ts --source mulliken_spin --atoms 9 10\n"
            "  tolkien-tools md split-nc sistema.prmtop 'QM_*.nc' 250-300\n"
        ),
    )
    subparsers = parser.add_subparsers(dest="command")

    inspect_p = subparsers.add_parser("inspect", help="Resume una corrida fragmentada")
    add_root_args(inspect_p)
    inspect_p.add_argument("--xyz-name", default="qm.xyz")
    inspect_p.add_argument("--input-name", default="d_QM.in")

    merge_xyz_p = subparsers.add_parser("merge-xyz", help="Une qm.xyz de carpetas numericas")
    add_root_args(merge_xyz_p)
    merge_xyz_p.add_argument("--xyz-name", default="qm.xyz")
    merge_xyz_p.add_argument("--input-name", default="d_QM.in")
    merge_xyz_p.add_argument("--out", default="qm_completo.xyz")

    geom_p = subparsers.add_parser("geom", help="Analiza distancias, angulos y dihedros en un XYZ multi-frame")
    geom_p.add_argument("xyz", help="Archivo XYZ multi-frame")
    geom_p.add_argument("--metric", action="append", required=True, help="Formato etiqueta:tipo:atomos. Ej: dFeN:distance:9,10")
    geom_p.add_argument("--scatter", action="append", help="Par etiqueta_x,etiqueta_y. Puede repetirse")
    geom_p.add_argument("--output", default="xyz_analysis", help="Prefijo de archivos de salida")
    geom_p.add_argument("--no-plots", action="store_true", help="No generar PNG")

    merge_pop_p = subparsers.add_parser("merge-pop", help="Une archivos de poblaciones desde segmentos numericos")
    add_root_args(merge_pop_p)
    merge_pop_p.add_argument("--sources", nargs="+", default=list(POPULATION_DEFAULTS), help="Nombres dentro de cada segmento")
    merge_pop_p.add_argument("--input-name", default="d_QM.in")
    merge_pop_p.add_argument("--out-dir", default=".", help="Directorio de salida")
    merge_pop_p.add_argument("--suffix", default="_full.dat", help="Sufijo de salida")

    spin_p = subparsers.add_parser("spin-ts", help="Serie temporal de poblacion para atomos seleccionados")
    add_root_args(spin_p)
    spin_p.add_argument("--source", default="mulliken_spin", help="Archivo fuente dentro de cada segmento o archivo consolidado")
    spin_p.add_argument("--atoms", nargs="+", type=int, required=True, help="Indices atomicos 1-based")
    spin_p.add_argument("--input-name", default="d_QM.in")
    spin_p.add_argument("--dt", type=float, help="Paso temporal entre frames en ps; si se omite se infiere de d_QM.in")
    spin_p.add_argument("--out", default="spin_timeseries.dat")

    split_p = subparsers.add_parser("split-nc", help="Extrae rst7 desde trayectorias NetCDF usando cpptraj")
    split_p.add_argument("prmtop")
    split_p.add_argument("nc_pattern")
    split_p.add_argument("frames", nargs="?", default="all", help="all, N, A-B, N-end o lista separada por coma")
    split_p.add_argument("--cpptraj", help="Ruta a cpptraj; si se omite se busca en PATH")
    split_p.add_argument("--out-prefix", default="QM", help="Prefijo de salida")

    return parser


def add_root_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", default=".", help="Carpeta raiz con subcarpetas numericas")


def cmd_inspect(args: argparse.Namespace) -> None:
    root = Path(args.root)
    segments = discover_segments(root, args.input_name)
    if not segments:
        raise SystemExit(f"No se encontraron carpetas numericas en {root}")

    print(f"Root: {root.resolve()}")
    print(f"Segmentos detectados: {len(segments)}")
    print()
    print("seg  qm.xyz  frames  dt_ps      nstlim   total_ps   archivos poblacion")

    total_frames = 0
    total_ps = 0.0
    for segment in segments:
        xyz_path = segment.path / args.xyz_name
        xyz_state = "no"
        frames_text = "-"
        if xyz_path.exists():
            try:
                _natoms, frames = xyz_summary(xyz_path)
                xyz_state = "si"
                frames_text = str(frames)
                total_frames += frames
            except ValueError as exc:
                xyz_state = f"error:{exc}"
        if segment.total_ps is not None:
            total_ps += segment.total_ps
        pop_sources = [name for name in POPULATION_DEFAULTS if (segment.path / name).exists()]
        print(
            f"{segment.index:>3}  {xyz_state:<6} {frames_text:>6}  "
            f"{format_optional(segment.dt_ps):>8}  {format_optional(segment.nstlim):>7}  "
            f"{format_optional(segment.total_ps):>9}  {','.join(pop_sources) or '-'}"
        )

    print()
    print(f"Frames XYZ totales detectados: {total_frames}")
    print(f"Tiempo total desde d_QM.in (ps): {total_ps:.9f}")


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
    specs = [parse_metric_spec(text) for text in args.metric]
    scatter_pairs = parse_scatter_pairs(args.scatter, specs)
    analyze_xyz(Path(args.xyz), specs, args.output, scatter_pairs, make_plots=not args.no_plots)


def cmd_merge_pop(args: argparse.Namespace) -> None:
    root = Path(args.root)
    segments = discover_segments(root, args.input_name)
    if not segments:
        raise SystemExit(f"No se encontraron carpetas numericas en {root}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for source in args.sources:
        output = out_dir / f"{source}{args.suffix}"
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
            count += 1
            wrote_line = False
            with path.open("r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    wrote_line = True
                    out.write(line)
                if wrote_line and not line_ends_with_newline(path):
                    out.write("\n")
    if count == 0:
        output.unlink(missing_ok=True)
    return count


def line_ends_with_newline(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            fh.seek(-1, 2)
            return fh.read(1) == b"\n"
    except OSError:
        return True


def cmd_spin_ts(args: argparse.Namespace) -> None:
    root = Path(args.root)
    source_path = Path(args.source)
    if source_path.exists() and source_path.is_file():
        if args.dt is None:
            raise SystemExit("Para usar un archivo consolidado en --source hay que pasar --dt.")
        rows = build_population_timeseries_from_file(source_path, args.dt, args.atoms)
    else:
        segments = discover_segments(root, args.input_name)
        if not segments:
            raise SystemExit(f"No se encontraron carpetas numericas en {root}")
        rows = build_population_timeseries_from_segments(segments, args.source, args.atoms, args.dt)

    if not rows:
        raise SystemExit("No se encontraron frames de poblacion para los atomos pedidos.")
    write_timeseries(Path(args.out), args.atoms, rows)
    print(f"Serie temporal escrita en {args.out} con {len(rows)} frames.")


def build_population_timeseries_from_segments(segments, source: str, atom_ids: list[int], dt_override: float | None):
    rows: list[list[float]] = []
    global_time = 0.0
    for segment in segments:
        path = segment.path / source
        if not path.exists():
            continue
        frame_values = parse_population_blocks(path, atom_ids)
        if not frame_values:
            continue
        if dt_override is not None:
            frame_dt = dt_override
        elif segment.total_ps is not None:
            frame_dt = segment.total_ps / len(frame_values)
        else:
            frame_dt = 1.0
        for values in frame_values:
            rows.append([global_time, *values])
            global_time += frame_dt
    return rows


def build_population_timeseries_from_file(path: Path, dt_ps: float, atom_ids: list[int]):
    rows: list[list[float]] = []
    for frame_idx, values in enumerate(parse_population_blocks(path, atom_ids)):
        rows.append([frame_idx * dt_ps, *values])
    return rows


def parse_population_blocks(path: Path, atom_ids: list[int]) -> list[list[float]]:
    frames: list[list[float]] = []
    current = {atom_id: float("nan") for atom_id in atom_ids}
    inside_frame = False
    saw_atom = False
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped.startswith("#") and "Population Analysis" in stripped:
                if inside_frame and saw_atom:
                    frames.append([current[atom_id] for atom_id in atom_ids])
                current = {atom_id: float("nan") for atom_id in atom_ids}
                inside_frame = True
                saw_atom = False
                continue
            if not inside_frame:
                continue
            if "Total Charge" in stripped:
                frames.append([current[atom_id] for atom_id in atom_ids])
                inside_frame = False
                saw_atom = False
                continue
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) >= 3 and parts[0].isdigit():
                atom_idx = int(parts[0])
                if atom_idx in current:
                    try:
                        current[atom_idx] = float(parts[2])
                    except ValueError:
                        pass
                saw_atom = True
    if inside_frame and saw_atom:
        frames.append([current[atom_id] for atom_id in atom_ids])
    return frames


def write_timeseries(path: Path, atom_ids: list[int], rows: list[list[float]]) -> None:
    with path.open("w", encoding="utf-8") as out:
        out.write("# time_ps " + " ".join(f"atom_{atom_id}" for atom_id in atom_ids) + "\n")
        for row in rows:
            out.write(" ".join(f"{value: .9f}" for value in row) + "\n")


def cmd_split_nc(args: argparse.Namespace) -> None:
    cpptraj = args.cpptraj or shutil.which("cpptraj")
    if not cpptraj:
        raise SystemExit("cpptraj no esta disponible en PATH. Pasar --cpptraj /ruta/cpptraj.")

    prmtop = Path(args.prmtop)
    if not prmtop.exists():
        raise SystemExit(f"No existe la topologia: {prmtop}")
    nc_files = sorted(Path(".").glob(args.nc_pattern))
    if not nc_files:
        raise SystemExit(f"No se encontraron archivos con patron {args.nc_pattern!r}")
    if list(Path(".").glob(f"{args.out_prefix}_*.rst7")):
        raise SystemExit(f"Ya existen archivos {args.out_prefix}_*.rst7 en este directorio.")

    selected_frames = resolve_frame_selection(cpptraj, prmtop, nc_files, args.frames)
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
        "inspect": cmd_inspect,
        "merge-xyz": cmd_merge_xyz,
        "geom": cmd_geom,
        "merge-pop": cmd_merge_pop,
        "spin-ts": cmd_spin_ts,
        "split-nc": cmd_split_nc,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main(sys.argv[1:])
