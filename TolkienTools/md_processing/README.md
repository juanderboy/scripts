# Molecular Dynamics Processing

Herramientas para reconstruir y analizar dinamicas moleculares fragmentadas en
subcarpetas numericas (`1/`, `2/`, `3/`, ...).

Uso desde el menu maestro:

```bash
tolkien-tools 1
tolkien-tools md
```

Subcomandos principales:

```bash
tolkien-tools md inspect-merge
tolkien-tools md inspect-merge --merge yes --out qm_completo.xyz
tolkien-tools md inspect-merge --merge yes --exclude 2,5-7 --out qm_completo.xyz
tolkien-tools md geom
tolkien-tools md geom qm_completo.xyz --metric dFeN:distance:9,10
tolkien-tools md split-nc
tolkien-tools md split-nc --count 100
```

La rama asume el formato habitual de corridas QMMM/MD fragmentadas: una carpeta
raiz con segmentos numericos, cada uno con archivos como `qm.xyz`, `d_QM.in`,
`mulliken`, `mulliken_spin`, `lowdin`, `lowdin_spin` y trayectorias `QM_*.nc`.

Para tiempos de dinamica, `d_QM.in` se usa para leer el `dt`; el `nstlim` queda
como referencia del tiempo planeado. Los tiempos reales se calculan a partir de
la cantidad de frames efectivamente presentes en `qm.xyz`, de modo que segmentos
cortados antes de tiempo no inflen la duracion total.

`inspect-merge` muestra el panel de segmentos y, si se corre desde una terminal
interactiva, ofrece mergear el XYZ y consolidar cargas/spines inmediatamente.
Se puede mergear todo o excluir segmentos concretos con listas/rangos como
`2,5-7`. Para scripts o uso no interactivo:

```bash
tolkien-tools md inspect-merge --merge yes --exclude 2,5-7 --out qm_completo.xyz
tolkien-tools md inspect-merge --merge no
```

Cuando `inspect-merge` mergea el XYZ, tambien busca `mulliken`,
`mulliken_spin`, `lowdin` y `lowdin_spin` en los mismos segmentos seleccionados
y genera los consolidados correspondientes (`*_full.dat`) junto al XYZ
combinado. Las exclusiones indicadas para las geometrías se aplican igual a
cargas/spines. Si solo se quiere el XYZ:

```bash
tolkien-tools md inspect-merge --merge yes --exclude 2,5-7 --no-pop
```

El subcomando `merge-xyz` sigue disponible como atajo compatible con versiones
anteriores, pero el flujo recomendado es inspeccionar y decidir el merge desde
`inspect-merge`.

Al procesar los `qm.xyz` segmentados, Tolkien Tools descarta siempre el primer
bloque de coordenadas de cada segmento y cuenta la dinamica desde el segundo.
`merge-xyz` tampoco incluye esos bloques iniciales en el XYZ consolidado.

El mismo criterio se aplica a los bloques segmentados de poblaciones
electronicas: `inspect-merge` descarta el primer bloque de cada segmento antes
de escribir los consolidados. Genera archivos compatibles con el modo LIO de la
rama de analisis de cargas y spines:

- `mulliken` -> `mulliken_full.dat`
- `mulliken_spin` -> `mulliken_spin_full.dat`

Esto permite correr despues `tolkien-tools 3` en modo LIO sobre la misma carpeta
de salida.

`split-nc` busca automaticamente el `.prmtop` en la carpeta raiz y los NetCDF
en subcarpetas numericas (`1/`, `2/`, ...). Si hay mas de una topologia,
pregunta cual usar. Luego cuenta frames por archivo `.nc`, muestra el total y
pregunta cuantos `rst7` generar. Los snapshots se distribuyen a lo largo de toda
la trayectoria concatenada y se escriben en `restarts/` como `QM_1.rst7`,
`QM_2.rst7`, etc.; tambien copia ahi el `.prmtop`.

Antes de elegir los snapshots, `split-nc` puede descartar un tramo inicial de la
trayectoria en picosegundos. En modo interactivo lo pregunta despues del panel
de NetCDF; para scripts se usa `--skip-initial-ps`. El filtro conserva frames
con `time_ps >= valor`, calculando el tiempo global desde `dt * ntwx` de cada
segmento.

Ademas genera `restarts/qm_snapshots.xyz` con las coordenadas QM
correspondientes a los mismos snapshots. Para mapear un frame del NetCDF al
`qm.xyz` del segmento usa `ntwx` de `d_QM.in`: el frame local `k` del NetCDF se
asocia al frame dinamico `k * ntwx` del `qm.xyz`. Como los `qm.xyz` de LIO
repiten dos veces la geometria inicial, se descarta el primer bloque y se indexa
sobre la secuencia dinamica restante, igual que en `inspect-merge`/`merge-xyz`. Cada
comentario del XYZ registra `restart_index`, frame global del NC, segmento,
archivo NC, frame NC local y frame QM usado.

Para uso no interactivo:

```bash
tolkien-tools md split-nc --count 100
tolkien-tools md split-nc --count 100 --skip-initial-ps 0.2
tolkien-tools md split-nc --root corrida --nc-pattern 'QM_*.nc' --count 100
```

Para omitir el XYZ QM:

```bash
tolkien-tools md split-nc --count 100 --no-qm-xyz
```

El modo legacy sigue disponible si se pasan topologia, patron y seleccion de
frames explicitamente:

```bash
tolkien-tools md split-nc sistema.prmtop 'QM_*.nc' 250-300
```

`geom` genera automaticamente un visor 3D HTML del primer frame antes de pedir o
calcular metricas e intenta abrirlo en el navegador. Por default usa `py3Dmol`,
con esferas, sticks y colores por elemento. Plotly queda disponible solo si se
pide explicitamente con `--viewer-backend plotly` o `--viewer-backend auto`.
Por default muestra los indices atomicos siempre, porque el objetivo principal
es elegir atomos para definir metricas. Para verlos solo al pasar el mouse:

```bash
tolkien-tools md geom qm_completo.xyz --viewer-labels hover
```

Para elegir otro backend:

```bash
tolkien-tools md geom qm_completo.xyz --viewer-backend py3dmol
tolkien-tools md geom qm_completo.xyz --viewer-backend plotly
tolkien-tools md geom qm_completo.xyz --viewer-backend auto
```

Para generar el HTML sin abrirlo automaticamente:

```bash
tolkien-tools md geom qm_completo.xyz --no-open-viewer
```
