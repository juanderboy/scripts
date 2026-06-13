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
    labels: str = "always",
    backend: str = "auto",
) -> Path:
    frames = parse_xyz_frames(xyz_path)
    if not frames:
        raise ValueError(f"No se encontraron frames en {xyz_path}")
    if frame_number < 1 or frame_number > len(frames):
        raise ValueError(f"Frame fuera de rango: {frame_number}. El XYZ tiene {len(frames)} frames.")

    frame = frames[frame_number - 1]
    if backend in {"auto", "py3dmol"}:
        try:
            write_py3dmol_viewer(xyz_path, output_path, frame, frame_number, labels)
            return output_path
        except ImportError:
            if backend == "py3dmol":
                raise RuntimeError("No se pudo importar py3Dmol. Instalar con: python3 -m pip install py3Dmol")
            print("[INFO] py3Dmol no esta instalado en este entorno.")
            print("       Se generara un visor HTML basico con Plotly.")
            print("       Para usar el visor molecular mejorado con esferas/sticks:")
            print("         python3 -m pip install py3Dmol")
        except Exception as exc:
            if backend == "py3dmol":
                raise RuntimeError(f"No se pudo generar el visor py3Dmol: {exc}") from exc
            print(f"[WARN] No se pudo generar el visor py3Dmol: {exc}")
            print("       Se generara un visor HTML basico con Plotly.")

    if backend not in {"auto", "plotly"}:
        raise ValueError(f"Backend de visor no soportado: {backend}")
    write_plotly_viewer(xyz_path, output_path, frame, frame_number, labels)
    return output_path


def frame_to_symbol_xyz(frame) -> str:
    lines = [frame.natoms_line.strip(), frame.comment_line]
    for atom_line, coord in zip(frame.atom_lines, frame.coords):
        raw_element = atom_line.split()[0]
        element = normalize_element(raw_element)
        x, y, z = coord
        lines.append(f"{element:2s} {x: .8f} {y: .8f} {z: .8f}")
    return "\n".join(lines) + "\n"


def write_py3dmol_viewer(
    xyz_path: Path,
    output_path: Path,
    frame,
    frame_number: int,
    labels: str,
) -> None:
    import py3Dmol

    view = py3Dmol.view(width=980, height=720)
    view.addModel(frame_to_symbol_xyz(frame), "xyz")
    view.setStyle({"stick": {"radius": 0.16}, "sphere": {"scale": 0.28}})

    if labels == "always":
        for atom_idx, (atom_line, coord) in enumerate(zip(frame.atom_lines, frame.coords), start=1):
            element = normalize_element(atom_line.split()[0])
            x, y, z = coord
            view.addLabel(
                f"{atom_idx} {element}",
                {
                    "position": {"x": x, "y": y, "z": z},
                    "fontColor": "black",
                    "backgroundColor": "white",
                    "backgroundOpacity": 0.9,
                    "fontSize": 15,
                    "inFront": True,
                },
            )
    else:
        view.startjs += """
	viewer_UNIQUEID.setHoverable({}, true,
		function(atom, viewer, event, container) {
			if (!atom.label) {
				atom.label = viewer.addLabel((atom.index + 1) + " " + atom.elem, {
					position: atom,
					fontColor: "black",
					backgroundColor: "white",
					backgroundOpacity: 0.75,
					fontSize: 14,
					inFront: true
				});
			}
		},
		function(atom, viewer) {
			if (atom.label) {
				viewer.removeLabel(atom.label);
				delete atom.label;
			}
		}
	);
"""

    view.zoomTo()
    title = f"{xyz_path.name} | frame {frame_number} | indices 1-based"
    body = view._make_html()
    html = (
        "<!doctype html>\n"
        "<html>\n"
        "<head><meta charset=\"utf-8\"><title>XYZ viewer</title></head>\n"
        "<body>\n"
        f"<h3 style=\"font-family: sans-serif; margin: 8px 0;\">{title}</h3>\n"
        f"{body}\n"
        "</body>\n"
        "</html>\n"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


def write_plotly_viewer(
    xyz_path: Path,
    output_path: Path,
    frame,
    frame_number: int,
    labels: str,
) -> None:
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
