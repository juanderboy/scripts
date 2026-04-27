#!/usr/bin/env python3
"""Interactive launcher for Tolkien's analysis routines."""

from __future__ import annotations

import subprocess
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    script: Path
    default_args: tuple[str, ...] = ()
    ask_input_file: bool = False


TOOLS = (
    Tool(
        name="TD-DFT spectra",
        description="Build NEA absorption spectra from ORCA TD-DFT outputs.",
        script=ROOT / "TolkienTools" / "td_dft" / "td_analyze.py",
    ),
    Tool(
        name="Charge and spin analysis",
        description="Analyze Mulliken, CHELPG, Lowdin, Hirshfeld and spin populations.",
        script=ROOT / "TolkienTools" / "charge_spin" / "charge_spin_analysis.py",
    ),
    Tool(
        name="Multilambda kinetics",
        description="Fit experimental multiwavelength kinetic spectra.",
        script=ROOT / "TolkienTools" / "kinetics" / "kinet_python.py",
        ask_input_file=True,
    ),
)


REQUIREMENTS_GUIDE = """\
Requisitos para instalar y correr Tolkien Tools

Base comun:
  - Python 3.10 o superior.
  - Paquetes Python:
      numpy
      scipy
      matplotlib
  - En Ubuntu/Debian se pueden instalar con:
      python3 -m pip install numpy scipy matplotlib
    o, si se prefiere usar paquetes del sistema:
      sudo apt install python3 python3-numpy python3-scipy python3-matplotlib

Programas externos:
  - No hace falta tener ORCA ni LIO instalados para ejecutar Tolkien Tools.
    Las rutinas leen archivos ya generados por esos programas.
  - Para abrir automaticamente reportes HTML desde TD-DFT, son utiles
    xdg-open en Linux o wslview en WSL. Si no estan, el HTML igual se genera
    y se puede abrir manualmente.

Archivos esperados por rutina:
  1. TD-DFT spectra
     - Lee salidas TD-DFT de ORCA, normalmente archivos TD_*.out.
     - Usa numpy, scipy.signal.find_peaks y matplotlib.
     - Puede generar PNG, CSV/DAT y un HTML interactivo.

  2. Charge and spin analysis
     - Modo LIO: lee series mq_*.dat y, si existen, ms_*.dat.
     - Modo ORCA: lee archivos <prefijo>_N.out o <prefijo>_N.dat con tablas
       Mulliken, Loewdin, Hirshfeld y/o CHELPG.
     - Usa numpy, scipy.stats.gaussian_kde, scipy.signal.find_peaks y
       matplotlib.

  3. Multilambda kinetics
     - Lee tablas espectrofotometricas lambda x tiempo en .txt/.csv estilo
       semicolon, y puede convertir archivos .KD.
     - Usa numpy, scipy.optimize, scipy.optimize.nnls y matplotlib.

Chequeo rapido:
  python3 -c "import numpy, scipy, matplotlib; print('OK')"
"""


def print_banner() -> None:
    width = 74
    lines = [
        "Tolkien Tools",
        "Computational chemistry and spectroscopy utilities",
        "vibecoded by Tolkien, 2026",
    ]

    print()
    print("=" * width)
    for line in lines:
        print(line.center(width))
    print("=" * width)
    print()


def print_requirements_guide() -> None:
    print()
    print(REQUIREMENTS_GUIDE)


def prompt_tool_choice() -> Tool | None:
    print("Que queres hacer?")
    for index, tool in enumerate(TOOLS, start=1):
        print(f"  {index}. {tool.name}")
        print(f"     {tool.description}")
    print("  i. Instructivo de instalacion y dependencias")
    print("  0. Salir")
    print()

    choice = input("Elegir opcion: ").strip()
    if choice.lower() in {"i", "info", "deps", "dependencias", "requirements", "requisitos"}:
        print_requirements_guide()
        return prompt_tool_choice()
    if choice in {"", "0", "q", "quit", "salir"}:
        return None
    if not choice.isdigit():
        raise ValueError("La opcion debe ser un numero.")

    index = int(choice)
    if index < 1 or index > len(TOOLS):
        raise ValueError(f"La opcion debe estar entre 1 y {len(TOOLS)}.")
    return TOOLS[index - 1]


def tool_from_cli_choice(choice: str) -> Tool:
    normalized = choice.strip().lower()
    aliases = {
        "1": 0,
        "td": 0,
        "td-dft": 0,
        "spectra": 0,
        "espectros": 0,
        "2": 1,
        "charges": 1,
        "charge": 1,
        "spin": 1,
        "cargas": 1,
        "3": 2,
        "kinetics": 2,
        "kinetic": 2,
        "kinet": 2,
        "cinetica": 2,
        "cineticas": 2,
    }
    if normalized not in aliases:
        raise ValueError(
            "La opcion debe ser 1, 2 o 3 "
            "(tambien se aceptan aliases como td, charges o kinetics). "
            "Para dependencias usa: requirements, deps o requisitos."
        )
    return TOOLS[aliases[normalized]]


def is_requirements_choice(choice: str) -> bool:
    return choice.strip().lower() in {
        "i",
        "info",
        "deps",
        "dependencias",
        "requirements",
        "requisitos",
        "--requirements",
        "--deps",
    }


def prompt_extra_args(tool: Tool) -> list[str]:
    args = list(tool.default_args)

    if tool.ask_input_file:
        text = input(
            "Archivo de entrada para la cinetica? "
            "Enter = usar el default de la rutina en la carpeta actual: "
        ).strip()
        if text:
            args.append(text)

    extra = input(
        "Argumentos extra para pasar a la rutina? Enter = ninguno: "
    ).strip()
    if extra:
        args.extend(shlex.split(extra))

    return args


def launch_tool(tool: Tool, args: list[str]) -> int:
    if not tool.script.exists():
        raise FileNotFoundError(f"No existe la rutina: {tool.script}")

    command = [sys.executable, str(tool.script), *args]
    print()
    print(f"Abriendo: {tool.name}")
    print(f"Comando: {' '.join(command)}")
    print()
    sys.stdout.flush()
    return subprocess.run(command, cwd=Path.cwd()).returncode


def main() -> int:
    print_banner()
    try:
        if len(sys.argv) > 1:
            if is_requirements_choice(sys.argv[1]):
                print_requirements_guide()
                return 0
            tool = tool_from_cli_choice(sys.argv[1])
            args = sys.argv[2:]
        else:
            tool = prompt_tool_choice()
            if tool is None:
                return 0
            args = prompt_extra_args(tool)
        return launch_tool(tool, args)
    except KeyboardInterrupt:
        print()
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
