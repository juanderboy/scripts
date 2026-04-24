# Tolkien Tools

Menu maestro para abrir rutinas grandes desde cualquier carpeta.

Comando principal:

```bash
tolkien-tools
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
