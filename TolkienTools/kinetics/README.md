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
- `kinet_models.py`: perfiles de concentracion para los modelos cineticos.
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

- `reduccion autocatalitica de MbFe(III) por sulfuros con binding inicial`:
  agrega el paso `MbFeIII + HS- -> MbFeIII-HS` antes de la reduccion. Ajusta
  tres espectros absorbentes (`MbFeIII`, `MbFeIII-HS`, `MbFeII`) y tres
  constantes (`k_on`, `k_slow,obs`, `k_auto`). En un experimento individual,
  `k_on` se trata como constante aparente pseudo-primer orden; para estimar un
  `k_on` bimolecular hay que considerar la concentracion efectiva de `HS-`.

## Detalles tecnicos de los modelos

### `mbfe3_sulfide_autocatalytic`

Este modelo asume que la especie inicial observable ya es el complejo
coordinado `MbFeIII-HS`. Es apropiado cuando `[HS-]` es alta y el binding es
rapido frente a la ventana temporal observada. La variable interna
`x = [MbFeII] / [Mb]total` representa la fraccion reducida y se usa como proxy
fenomenologico de especies reactivas de azufre no observadas.

La evolucion es:

```text
dx/dt = (k_slow + k_auto*x) * (1 - x)
[MbFeIII-HS](t) = c0 * (1 - x)
[MbFeII](t) = c0 * x
```

Este caso tiene una solucion analitica compacta y se evalua directamente en
`kinet_models.py`, sin integracion numerica.

### `mbfe3_sulfide_binding_autocatalytic`

Este modelo agrega el proceso inicial de coordinacion:

```text
MbFeIII + HS- -> MbFeIII-HS -> MbFeII
```

Las especies absorbentes ajustadas son `MbFeIII`, `MbFeIII-HS` y `MbFeII`.
Las constantes ajustadas son `k_on`, `k_slow,obs` y `k_auto`.

Para un unico experimento, `k_on` se modela como constante aparente
pseudo-primer orden:

```text
d[MbFeIII]/dt    = -k_on * [MbFeIII]
x                = [MbFeII] / [Mb]total
d[MbFeIII-HS]/dt = k_on*[MbFeIII] - (k_slow + k_auto*x)*[MbFeIII-HS]
d[MbFeII]/dt     = (k_slow + k_auto*x)*[MbFeIII-HS]
```

Si `[HS-]` esta en exceso y se conoce su concentracion efectiva, el `k_on`
bimolecular se debe estimar fuera del ajuste dividiendo la constante aparente
por `[HS-]`.

Este sistema se resuelve con `scipy.integrate.solve_ivp`. `solve_ivp` es un
integrador numerico de ecuaciones diferenciales ordinarias: el codigo define
las derivadas del modelo y el integrador calcula las concentraciones en los
tiempos experimentales. Es conveniente aca porque el mecanismo incluye un
intermedio observable y una velocidad de reduccion que depende de la fraccion
reducida acumulada. Esa dependencia autocatalitica acopla las ecuaciones y una
solucion analitica cerrada seria mas dificil de mantener y mas propensa a
errores.

La ventaja practica de `solve_ivp` es que el codigo sigue de cerca el mecanismo
quimico: tasas de binding, reduccion y acumulacion de producto. Tambien facilita
extender el modelo mas adelante, por ejemplo para agregar binding reversible,
dependencia explicita de `[HS-]`, consumo de sulfuro u otros intermediarios. La
desventaja es que cada prueba de parametros durante el ajuste requiere integrar
el sistema, por lo que es mas costoso que una formula analitica. Para tres
especies y una cantidad normal de tiempos experimentales, el costo deberia ser
razonable.

Ejemplo no interactivo:

```bash
tolkien-tools 4 experimento.dat \
  --model mbfe3_sulfide_autocatalytic \
  --baseline-mode none \
  --no-plot
```

```bash
tolkien-tools 4 experimento.dat \
  --model mbfe3_sulfide_binding_autocatalytic \
  --baseline-mode none \
  --no-plot
```

Backup previo a la separacion:

```text
kinet_python.py.backup_before_split
```
