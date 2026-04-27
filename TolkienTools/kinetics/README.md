# Multilambda Kinetics

Rutina conectada actualmente:

```text
/home/juanderboy/scripts/TolkienTools/kinetics/kinet_python.py
```

Uso desde el menu maestro:

```bash
tolkien-tools
```

Este modulo contiene la rutina de cineticas experimentales multilambda y el
conversor `kd_to_txt.py`.

Por ahora `kinet/` conserva datos de prueba, archivos `.KD/.txt` y scripts
MATLAB historicos.

## Organizacion interna

`kinet_python.py` se mantiene como punto de entrada compatible con llamadas
viejas, pero la implementacion esta separada en modulos dentro de esta misma
carpeta:

- `kinet_cli.py`: parser de argumentos, preguntas interactivas y flujo general.
- `kinet_common.py`: dataclasses y etiquetas compartidas.
- `kinet_io.py`: lectura de tablas y conversion previa de `.KD`.
- `kinet_preprocessing.py`: baseline, recortes de tiempo/lambda y descarte de espectros.
- `kinet_models.py`: perfiles analiticos de concentracion para los modelos cineticos.
- `kinet_fitting.py`: NNLS, pseudoinversa, factor analysis y optimizacion de constantes.
- `kinet_linalg.py`: helpers de algebra lineal/SVD.
- `kinet_plotting.py`: figuras exploratorias, diagnosticos y panel final.
- `kinet_export.py`: archivos finales de concentraciones, espectros puros, resumen y PNG.

El metodo `factor` esta implementado para `A -> B`, `A -> B -> C` y
`A <-> B -> C`.
Para `A -> B -> C`, el ajuste en espacio de factores identifica los dos
exponentes cineticos y luego elige el orden `k1/k2` que da espectros
recuperados con menor contribucion negativa. En el mecanismo reversible, las
constantes tienen roles distintos (`k1`, `k-1`, `k2`) y no se permutan. Este
ultimo caso es mas sensible a correlaciones entre parametros; conviene usarlo
como diagnostico y comparar siempre contra `nnls`.

Backup previo a la separacion:

```text
kinet_python.py.backup_before_split
```
