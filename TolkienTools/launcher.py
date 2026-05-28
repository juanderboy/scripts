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
        name="Molecular dynamics processing",
        description="Rebuild and analyze fragmented MD/QMMM runs.",
        script=ROOT / "TolkienTools" / "md_processing" / "md_process.py",
    ),
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

MD_SUBCOMMANDS = (
    (
        "inspect",
        "Revisar segmentos, frames, tiempos y archivos disponibles.",
        "--root /ruta/a/MD_a",
    ),
    (
        "merge-xyz",
        "Unir los qm.xyz de una corrida fragmentada.",
        "--root /ruta/a/MD_a --out qm_completo.xyz",
    ),
    (
        "geom",
        "Analizar distancias, angulos y dihedros en un XYZ multi-frame.",
        "qm_completo.xyz --metric dFeN:distance:9,10",
    ),
    (
        "merge-pop",
        "Unir poblaciones y generar aliases LIO mq_*.dat/ms_*.dat.",
        "--root /ruta/a/MD_a --sources mulliken mulliken_spin",
    ),
    (
        "spin-ts",
        "Extraer serie temporal de poblacion para atomos seleccionados.",
        "--root /ruta/a/MD_a --source mulliken_spin --atoms 9 10",
    ),
    (
        "split-nc",
        "Extraer snapshots rst7 desde QM_*.nc usando cpptraj.",
        "sistema.prmtop 'QM_*.nc' 250-300",
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
  - Para `tolkien-tools md split-nc` hace falta tener cpptraj disponible.
  - Para abrir automaticamente reportes HTML desde TD-DFT, son utiles
    xdg-open en Linux o wslview en WSL. Si no estan, el HTML igual se genera
    y se puede abrir manualmente.

Archivos esperados por rutina:
  1. Molecular dynamics processing
     - Trabaja sobre corridas fragmentadas en subcarpetas numericas 1, 2, 3...
     - Puede inspeccionar segmentos, unir qm.xyz, analizar geometria, unir
       poblaciones y extraer rst7 desde QM_*.nc usando cpptraj.
     - Subcomandos: inspect, merge-xyz, geom, merge-pop, spin-ts, split-nc.

  2. TD-DFT spectra
     - Lee salidas TD-DFT de ORCA, normalmente archivos TD_*.out.
     - Usa numpy, scipy.signal.find_peaks y matplotlib.
     - Puede generar PNG, CSV/DAT y un HTML interactivo.

  3. Charge and spin analysis
     - Modo LIO: lee series mq_*.dat y, si existen, ms_*.dat.
     - Modo ORCA: lee archivos <prefijo>_N.out o <prefijo>_N.dat con tablas
       Mulliken, Loewdin, Hirshfeld y/o CHELPG.
     - Usa numpy, scipy.stats.gaussian_kde, scipy.signal.find_peaks y
       matplotlib.

  4. Multilambda kinetics
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
        "md": 0,
        "dynamics": 0,
        "dinamica": 0,
        "dinamicas": 0,
        "procesado": 0,
        "processing": 0,
        "2": 1,
        "td": 1,
        "td-dft": 1,
        "spectra": 1,
        "espectros": 1,
        "3": 2,
        "charges": 2,
        "charge": 2,
        "spin": 2,
        "cargas": 2,
        "4": 3,
        "kinetics": 3,
        "kinetic": 3,
        "kinet": 3,
        "cinetica": 3,
        "cineticas": 3,
    }
    if normalized not in aliases:
        raise ValueError(
            "La opcion debe ser 1, 2, 3 o 4 "
            "(tambien se aceptan aliases como md, td, charges o kinetics). "
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

    if tool.name == "Molecular dynamics processing":
        return prompt_md_processing_args()

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


def prompt_md_processing_args() -> list[str]:
    print()
    print("Herramientas de procesado de dinamicas:")
    for index, (command, description, example) in enumerate(MD_SUBCOMMANDS, start=1):
        print(f"  {index}. {command}")
        print(f"     {description}")
        print(f"     Ejemplo de argumentos: {example}")
    print("  h. Ayuda general de procesado")
    print("  0. Mostrar ayuda y salir")
    print()

    choice = input("Elegir herramienta (Enter = inspect): ").strip().lower()
    if choice in {"h", "help", "ayuda", "0"}:
        return ["--help"]
    if choice == "":
        command = "inspect"
        example = "--root /ruta/a/MD_a"
    else:
        if not choice.isdigit():
            raise ValueError("La herramienta de procesado debe elegirse por numero.")
        index = int(choice)
        if index < 1 or index > len(MD_SUBCOMMANDS):
            raise ValueError(f"La herramienta debe estar entre 1 y {len(MD_SUBCOMMANDS)}.")
        command, _description, example = MD_SUBCOMMANDS[index - 1]

    print()
    print(f"Seleccionado: {command}")
    print(f"Ejemplo: tolkien-tools md {command} {example}")
    extra = input(
        f"Argumentos para '{command}'? Enter = usar defaults en la carpeta actual: "
    ).strip()

    args = [command]
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


def prompt_continue() -> bool:
    print()
    choice = input("Enter = volver al menu principal; q = salir: ").strip().lower()
    return choice not in {"q", "quit", "salir", "0"}


def main() -> int:
    print_banner()
    try:
        if len(sys.argv) > 1:
            if is_requirements_choice(sys.argv[1]):
                print_requirements_guide()
                return 0
            tool = tool_from_cli_choice(sys.argv[1])
            args = sys.argv[2:]
            return launch_tool(tool, args)
        else:
            last_returncode = 0
            while True:
                tool = prompt_tool_choice()
                if tool is None:
                    return last_returncode
                args = prompt_extra_args(tool)
                last_returncode = launch_tool(tool, args)
                if not prompt_continue():
                    return last_returncode
    except KeyboardInterrupt:
        print()
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
