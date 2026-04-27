# Tolkien Tools

Menu maestro para abrir rutinas grandes desde cualquier carpeta.

Comando principal:

```bash
tolkien-tools
```

Instructivo de instalacion y dependencias:

```bash
tolkien-tools requirements
```

Atajos directos:

```bash
tolkien-tools 1   # TD-DFT NEA spectra
tolkien-tools 2   # charge and spin analysis
tolkien-tools 3   # multilambda kinetics
```

Los argumentos despues del numero se pasan directo a la rutina elegida, por
ejemplo:

```bash
tolkien-tools 3 117.txt --fit-method pinv --no-plot
```

Rutinas conectadas inicialmente:

- `td_dft/td_analyze.py`: espectros de absorcion por Nuclear Ensemble
  Approximation (NEA) desde calculos TD-DFT de ORCA.
- `charge_spin/charge_spin_analysis.py`: analisis de cargas Mulliken, CHELPG, Lowdin,
  Hirshfeld y poblaciones de spin.
- `kinetics/kinet_python.py`: cineticas experimentales multilambda.

El launcher llama las rutinas por ruta absoluta y mantiene como carpeta de
trabajo el directorio desde donde se corre `tolkien-tools`, que es lo que
necesitan las rutinas que buscan archivos en la carpeta actual.

## Requisitos

Base comun:

- Python 3.10 o superior.
- Paquetes Python: `numpy`, `scipy`, `matplotlib`.

Instalacion con `pip`:

```bash
python3 -m pip install numpy scipy matplotlib
```

Alternativa Ubuntu/Debian con paquetes del sistema:

```bash
sudo apt install python3 python3-numpy python3-scipy python3-matplotlib
```

No hace falta tener ORCA ni LIO instalados para ejecutar Tolkien Tools: las
rutinas leen archivos ya generados por esos programas. Para abrir
automaticamente reportes HTML desde TD-DFT son utiles `xdg-open` en Linux o
`wslview` en WSL, pero no son obligatorios.

Dependencias por rutina:

- TD-DFT spectra: lee salidas TD-DFT de ORCA, normalmente `TD_*.out`; usa
  `numpy`, `scipy.signal.find_peaks` y `matplotlib`.
- Charge and spin analysis: en modo LIO lee `mq_*.dat` y opcionalmente
  `ms_*.dat`; en modo ORCA lee `<prefijo>_N.out` o `<prefijo>_N.dat`; usa
  `numpy`, `scipy.stats.gaussian_kde`, `scipy.signal.find_peaks` y
  `matplotlib`.
- Multilambda kinetics: lee tablas espectrofotometricas lambda x tiempo y
  puede convertir `.KD`; usa `numpy`, `scipy.optimize`,
  `scipy.optimize.nnls` y `matplotlib`.

Chequeo rapido:

```bash
python3 -c "import numpy, scipy, matplotlib; print('OK')"
```
