#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Command-line interface for TD-DFT spectrum analysis.

It defines terminal options and dispatches processing to either the single-file
or folder workflow.
"""

import argparse
import os
import sys

from td_batch import process_folder
from td_single import process_single_file


def main():
    parser = argparse.ArgumentParser(
        prog="rutina",
        description=(
            "Genera espectros de absortividad molar ε desde TD-DFT de ORCA.\n"
            "MODO 1 (archivo): rutina.py TD_10.out [opciones]\n"
            "MODO 2 (carpeta): rutina.py [carpeta] [opciones]\n"
            "Si no se indica carpeta ni archivo, se usa la carpeta actual."
        ),
        epilog=(
            "FLAGS PRINCIPALES\n"
            "  filename              Archivo TD_*.out o carpeta con TD_*.out (default: '.')\n"
            "  --mode {lambda,energy}\n"
            "      Representación del espectro:\n"
            "        lambda (default) -> ε(λ)\n"
            "        energy          -> ε(E)\n"
            "  -x0, --startx FLOAT   Límite inferior del eje x (en nm).\n"
            "                        Si no se especifica, se usa 200 nm.\n"
            "  -x1, --endx   FLOAT   Límite superior del eje x (en nm).\n"
            "                        Si no se especifica, se usa 1000 nm.\n"
            "  -wev, --linewidth_ev FLOAT\n"
            "      Ancho de línea FWHM en eV para la convolución gaussiana.\n"
            "      Default: 0.1 eV (a menos que se indique explícitamente).\n"
            "  -s, --show            Muestra la ventana de matplotlib.\n"
            "                        Si no se especifica -s/--show, se asume True.\n"
            "  -n, --nosave          No guarda los PNG que se pudieran generar.\n"
            "  --nref FLOAT          Índice de refracción (default: 1.33, agua en el visible).\n"
            "  --exclude LIST        En modo carpeta, excluye TD_N.out por número o rango. Ej: --exclude 10,12,15 o 10-20\n"
            "  --select LIST         En modo carpeta, selecciona índices 1..N o rangos (ej: 1-3,5,7).\n"
            "  --maxlist             En modo carpeta, genera CSV con máximos individuales.\n"
            "  --maxrange XMIN XMAX  Rango de x para buscar máximos (nm en modo lambda, eV en modo energy).\n"
            "  --maxeps FLOAT        Umbral mínimo de epsilon para listar máximos.\n"
            "  --maxonly             Listar solo el máximo absoluto por espectro (modo carpeta).\n"
            "  --allpeaks            Listar todos los máximos locales (modo carpeta).\n"
            "  --printall            En modo carpeta, guarda PNG individuales TD_*.out-nea-eps.png.\n"
            "  --html                En modo carpeta, genera HTML interactivo (default: activado).\n"
            "  --no-html             En modo carpeta, desactiva el HTML interactivo.\n"
            "  -e, --export          En modo archivo único, exporta ε como espectro_*.dat o espectroE_*.dat.\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # filename o carpeta (opcional, default='.')
    parser.add_argument(
        "filename",
        nargs="?",
        default=".",
        help="Archivo de salida de ORCA (TD_*.out) o carpeta con TD_*.out (default: carpeta actual).",
    )

    # modo de representación
    parser.add_argument(
        "--mode",
        choices=["lambda", "energy"],
        default="lambda",
        help="Representación del espectro: 'lambda' (ε(λ), default) o 'energy' (ε(E)).",
    )

    # mostrar ventana
    parser.add_argument(
        "-s",
        "--show",
        default=False,
        action="store_true",
        help="Mostrar la ventana de matplotlib.",
    )

    # no guardar png
    parser.add_argument(
        "-n",
        "--nosave",
        default=False,
        action="store_true",
        help="No guardar los PNG que se pudieran generar.",
    )

    # ancho de línea (FWHM) en eV
    parser.add_argument(
        "-wev",
        "--linewidth_ev",
        type=float,
        default=0.1,
        help="Ancho de línea (FWHM) en eV para la convolución gaussiana (default: 0.1 eV).",
    )

    # rango en λ (nm)
    parser.add_argument(
        "-x0",
        "--startx",
        type=float,
        help="Límite inferior del eje x (λ, en nm; se convierte a E en modo 'energy').",
    )
    parser.add_argument(
        "-x1",
        "--endx",
        type=float,
        help="Límite superior del eje x (λ, en nm; se convierte a E en modo 'energy').",
    )

    # índice de refracción
    parser.add_argument(
        "--nref",
        type=float,
        default=1.33,   # agua en el visible
        help="Índice de refracción del medilt: 1.33, agua en el visible).",
    )

    # excluir algunos TD_* por número (modo carpeta)
    parser.add_argument(
        "--exclude",
        type=str,
        default=None,
        help="Números de TD_*.out a excluir, ej: '10', '10,12,15' o '10-20' (solo modo carpeta).",
    )
    # selección explícita por índice (modo carpeta)
    parser.add_argument(
        "--select",
        type=str,
        default=None,
        help="Selecciona índices 1..N o rangos, ej: '1-15' o '1,2,3,7-10' (solo modo carpeta).",
    )
    # no marcar picos en el espectro final (modo carpeta)
    parser.add_argument(
        "--no-final-peaks",
        default=False,
        action="store_true",
        help="Do not mark/label peaks in the final cumulative spectrum (folder mode).",
    )
    # imprimir PNGs individuales en modo carpeta
    parser.add_argument(
        "--printall",
        default=False,
        action="store_true",
        help="(Modo carpeta) Guardar PNG individuales TD_*.out-nea-eps.png.",
    )

    # lista de máximos por espectro (modo carpeta)
    parser.add_argument(
        "--maxlist",
        default=False,
        action="store_true",
        help="(Modo carpeta) Generar CSV con la localización del máximo de cada espectro.",
    )
    parser.add_argument(
        "--maxrange",
        nargs=2,
        type=float,
        metavar=("XMIN", "XMAX"),
        help="(Modo carpeta) Rango de x para buscar máximos (nm en modo lambda, eV en modo energy).",
    )
    parser.add_argument(
        "--maxeps",
        type=float,
        help="(Modo carpeta) Umbral mínimo de epsilon para listar máximos.",
    )
    parser.add_argument(
        "--maxonly",
        default=None,
        action="store_true",
        help="(Modo carpeta) Listar solo el máximo absoluto dentro del rango.",
    )
    parser.add_argument(
        "--allpeaks",
        default=None,
        action="store_true",
        help="(Modo carpeta) Listar todos los máximos locales dentro del rango.",
    )

    # HTML interactivo (plotly) en modo carpeta: activado por defecto
    html_group = parser.add_mutually_exclusive_group()
    html_group.add_argument(
        "--html",
        dest="html",
        action="store_true",
        help="(Modo carpeta) Generar HTML interactivo (default: activado).",
    )
    html_group.add_argument(
        "--no-html",
        dest="html",
        action="store_false",
        help="(Modo carpeta) No generar HTML interactivo.",
    )
    parser.set_defaults(html=True)

    # exportar datos en modo archivo único
    parser.add_argument(
        "-e",
        "--export",
        default=False,
        action="store_true",
        help="(Modo archivo único) Exportar datos ε como espectro_*.dat (lamb espectroE_*.dat (energy).",
    )

    args = parser.parse_args()

    # --- defaults "suaves" ---
    # Si no se dio -x0/-x1, usar 200–1000 nm
    if args.startx is None:
        args.startx = 200.0
    if args.endx is None:
        args.endx = 1000.0

    # Si el usuario no puso -s/--show, activamos show por defecto
    if not any(a in ("-s", "--show") for a in sys.argv[1:]):
        args.show = True

    if args.linewidth_ev <= 0:
        print("El ancho de línea (--linewidth_ev) debe ser > 0.")
        sys.exit(1)

    target = args.filename

    if os.path.isdir(target):
        process_folder(target, args)
    else:
        process_single_file(target, args)


if __name__ == "__main__":
    main()
