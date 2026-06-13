#!/usr/bin/env python3
"""Interactive launcher for Tolkien's analysis routines."""

from __future__ import annotations

import subprocess
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
        "inspect-merge",
        "Revisar segmentos, frames y tiempos; despues ofrece mergear XYZ y cargas/spines.",
    ),
    (
        "geom",
        "Generar visor 3D y analizar distancias, angulos y dihedros.",
    ),
    (
        "split-nc",
        "Inspeccionar QM_*.nc fragmentados y generar restarts muestreados.",
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
      py3Dmol, plotly (opcionales, para el visor 3D de geometria)
  - En Ubuntu/Debian se pueden instalar con:
      python3 -m pip install numpy scipy matplotlib py3Dmol plotly
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
     - Puede inspeccionar segmentos y unir qm.xyz junto con poblaciones,
       analizar geometria y generar rst7 muestreados desde QM_*.nc usando cpptraj.
     - El analisis geometrico puede generar un visor 3D HTML con py3Dmol o
       Plotly si estan instalados.
     - Subcomandos: inspect-merge, geom, split-nc.

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

    if tool.ask_input_file:
        text = input(
            "Archivo de entrada para la cinetica? "
            "Enter = usar el default de la rutina en la carpeta actual: "
        ).strip()
        if text:
            args.append(text)

    return args


def prompt_md_processing_choice() -> tuple[str, list[str] | None]:
    print()
    print("Herramientas de procesado de dinamicas:")
    for index, (command, description) in enumerate(MD_SUBCOMMANDS, start=1):
        print(f"  {index}. {command}")
        print(f"     {description}")
    print("  h. Ayuda general de procesado")
    print("  0. Volver al menu principal")
    print("  q. Salir")
    print()

    choice = input("Elegir herramienta (Enter = inspect-merge): ").strip().lower()
    if choice in {"h", "help", "ayuda"}:
        return "launch", ["--help"]
    if choice in {"0", "m", "menu", "principal"}:
        return "main", None
    if choice in {"q", "quit", "salir"}:
        return "exit", None
    if choice == "":
        command = "inspect-merge"
    else:
        if not choice.isdigit():
            raise ValueError("La herramienta de procesado debe elegirse por numero.")
        index = int(choice)
        if index < 1 or index > len(MD_SUBCOMMANDS):
            raise ValueError(f"La herramienta debe estar entre 1 y {len(MD_SUBCOMMANDS)}.")
        command, _description = MD_SUBCOMMANDS[index - 1]

    args = [command]
    if command == "split-nc":
        nc_pattern = input("Patron de trayectorias NetCDF (Enter = QM_*.nc): ").strip() or "QM_*.nc"
        skip_initial_ps = input("Ignorar frames iniciales antes de cuantos ps? (Enter = preguntar despues): ").strip()
        count = input("Cantidad de rst7 a generar (Enter = preguntar despues de inspeccionar): ").strip()
        if nc_pattern != "QM_*.nc":
            args.extend(["--nc-pattern", nc_pattern])
        if skip_initial_ps:
            try:
                if float(skip_initial_ps) < 0.0:
                    raise ValueError
            except ValueError as exc:
                raise ValueError("El tiempo inicial a ignorar debe ser un numero >= 0.") from exc
            args.extend(["--skip-initial-ps", skip_initial_ps])
        if count:
            if not count.isdigit():
                raise ValueError("La cantidad de rst7 debe ser un entero.")
            args.extend(["--count", count])
    return "launch", args


def prompt_md_continue() -> str:
    print()
    choice = input("Enter = volver al panel de dinamicas; m = menu principal; q = salir: ").strip().lower()
    if choice in {"q", "quit", "salir"}:
        return "exit"
    if choice in {"m", "menu", "principal", "0"}:
        return "main"
    return "md"


def run_md_processing_menu(tool: Tool, last_returncode: int = 0) -> tuple[int, bool]:
    while True:
        action, args = prompt_md_processing_choice()
        if action == "exit":
            return last_returncode, True
        if action == "main":
            return last_returncode, False
        if action == "cancel":
            continue
        last_returncode = launch_tool(tool, args or [])
        destination = prompt_md_continue()
        if destination == "exit":
            return last_returncode, True
        if destination == "main":
            return last_returncode, False


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
                if tool.name == "Molecular dynamics processing":
                    last_returncode, should_exit = run_md_processing_menu(tool, last_returncode)
                    if should_exit:
                        return last_returncode
                    continue
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
