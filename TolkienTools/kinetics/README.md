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

Cada analisis escribe sus artefactos en una subcarpeta
`results_<nombre-del-archivo>` junto al archivo de entrada, para no mezclar
resultados con experimentos pendientes de procesar.

Una vez elegido el modelo, la rutina imprime una presentacion basica con el
esquema cinetico, las especies absorbentes, la evolucion temporal ajustada, los
parametros y notas de interpretacion.

El metodo `factor` esta implementado para `A -> B`, `A -> B -> C` y
`A <-> B -> C`.
Para `A -> B -> C`, el ajuste en espacio de factores identifica los dos
exponentes cineticos y luego elige el orden `k1/k2` que da espectros
recuperados con menor contribucion negativa. En el mecanismo reversible, las
constantes tienen roles distintos (`k1`, `k-1`, `k2`) y no se permutan. Este
ultimo caso es mas sensible a correlaciones entre parametros; conviene usarlo
como diagnostico y comparar siempre contra `nnls`.

## Mecanismos especiales

El menu de modelos incluye una categoria separada para analisis especificos de
un sistema quimico. Actualmente contiene:

- `reduccion autocatalitica de MbFe(III) por sulfuros`: ajusta globalmente
  todos los tiempos como `dx/dt = (k_slow + k_auto * x) * (1 - x)`, donde `x`
  es la fraccion de `MbFeII`. `k_slow,obs` describe la fase lenta inicial y
  `k_auto` cuantifica la aceleracion aparente por especies reactivas de azufre.
  La dependencia de `k_slow,obs` con `[HS-]` se debe determinar
  experimentalmente antes de asignarle una interpretacion mecanistica.

Ejemplo no interactivo:

```bash
tolkien-tools 4 experimento.dat \
  --model mbfe3_sulfide_autocatalytic \
  --baseline-mode none \
  --no-plot
```

Backup previo a la separacion:

```text
kinet_python.py.backup_before_split
```
