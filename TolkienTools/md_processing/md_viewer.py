#!/usr/bin/env python3
"""Interactive 3D XYZ viewer generation."""

from __future__ import annotations

from pathlib import Path

from md_xyz import parse_xyz_frames


ELEMENT_SYMBOLS = {
    1: "H",
    2: "He",
    3: "Li",
    4: "Be",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    10: "Ne",
    11: "Na",
    12: "Mg",
    13: "Al",
    14: "Si",
    15: "P",
    16: "S",
    17: "Cl",
    18: "Ar",
    26: "Fe",
    29: "Cu",
    30: "Zn",
    44: "Ru",
}

ELEMENT_COLORS = {
    "H": "#f2f2f2",
    "C": "#4a4a4a",
    "N": "#2c6bed",
    "O": "#d62828",
    "S": "#d9a520",
    "P": "#ff7f0e",
    "Fe": "#b7410e",
    "Ru": "#1f9e89",
}


def normalize_element(token: str) -> str:
    try:
        atomic_number = int(token)
    except ValueError:
        return token.strip().capitalize()
    return ELEMENT_SYMBOLS.get(atomic_number, str(atomic_number))


def atom_size(element: str) -> int:
    if element == "H":
        return 5
    if element in {"Fe", "Ru", "Cu", "Zn"}:
        return 10
    return 7


def write_xyz_viewer(
    xyz_path: Path,
    output_path: Path,
    frame_number: int = 1,
    labels: str = "hover",
) -> Path:
    frames = parse_xyz_frames(xyz_path)
    if not frames:
        raise ValueError(f"No se encontraron frames en {xyz_path}")
    if frame_number < 1 or frame_number > len(frames):
        raise ValueError(f"Frame fuera de rango: {frame_number}. El XYZ tiene {len(frames)} frames.")

    frame = frames[frame_number - 1]
    elements: list[str] = []
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    hover: list[str] = []
    text: list[str] = []
    colors: list[str] = []
    sizes: list[int] = []

    for atom_idx, (atom_line, coord) in enumerate(zip(frame.atom_lines, frame.coords), start=1):
        raw_element = atom_line.split()[0]
        element = normalize_element(raw_element)
        x, y, z = coord
        elements.append(element)
        xs.append(x)
        ys.append(y)
        zs.append(z)
        hover.append(f"{atom_idx} {element}<br>x={x:.4f}<br>y={y:.4f}<br>z={z:.4f}")
        text.append(f"{atom_idx} {element}")
        colors.append(ELEMENT_COLORS.get(element, "#7f7f7f"))
        sizes.append(atom_size(element))

    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise RuntimeError(
            "No se pudo importar plotly. Instalar con: python3 -m pip install plotly"
        ) from exc

    mode = "markers+text" if labels == "always" else "markers"
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode=mode,
                text=text,
                hovertext=hover,
                hoverinfo="text",
                textposition="top center",
                marker={
                    "size": sizes,
                    "color": colors,
                    "line": {"color": "#222222", "width": 1},
                    "opacity": 0.92,
                },
            )
        ]
    )
    fig.update_layout(
        title=f"{xyz_path.name} | frame {frame_number} | indices 1-based",
        scene={
            "xaxis_title": "x (A)",
            "yaxis_title": "y (A)",
            "zaxis_title": "z (A)",
            "aspectmode": "data",
        },
        margin={"l": 0, "r": 0, "t": 42, "b": 0},
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path, include_plotlyjs=True, full_html=True)
    return output_path

