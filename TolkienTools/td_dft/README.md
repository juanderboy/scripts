# TD-DFT NEA Spectra

Rutina conectada actualmente:

```text
/home/juanderboy/scripts/TolkienTools/td_dft/td_analyze.py
```

Uso desde el menu maestro:

```bash
tolkien-tools
```

Este modulo contiene la rutina de espectros de absorcion por Nuclear Ensemble
Approximation (NEA) a partir de calculos TD-DFT de ORCA.

Estructura interna:

- `td_analyze.py`: entrada compatible usada por el launcher.
- `td_cli.py`: argumentos de linea de comandos y dispatch archivo/carpeta.
- `td_common.py`: constantes fisicas, rangos, selecciones y apertura de HTML.
- `td_orca.py`: parsing de salidas TD-DFT de ORCA.
- `td_spectrum.py`: grilla de energia, gaussianas y calculo NEA.
- `td_single.py`: procesamiento de un unico archivo `TD_*.out`.
- `td_batch.py`: procesamiento de carpetas con muchos `TD_*.out`.
