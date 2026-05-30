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
tolkien-tools md inspect
tolkien-tools md merge-xyz --out qm_completo.xyz
tolkien-tools md geom
tolkien-tools md geom qm_completo.xyz --metric dFeN:distance:9,10
tolkien-tools md merge-pop --sources mulliken mulliken_spin
tolkien-tools md spin-ts --source mulliken_spin --atoms 9 10
tolkien-tools md split-nc sistema.prmtop 'QM_*.nc' 250-300
```

La rama asume el formato habitual de corridas QMMM/MD fragmentadas: una carpeta
raiz con segmentos numericos, cada uno con archivos como `qm.xyz`, `d_QM.in`,
`mulliken`, `mulliken_spin`, `lowdin`, `lowdin_spin` y trayectorias `QM_*.nc`.

Para tiempos de dinamica, `d_QM.in` se usa para leer el `dt`; el `nstlim` queda
como referencia del tiempo planeado. Los tiempos reales se calculan a partir de
la cantidad de frames efectivamente presentes en `qm.xyz`, de modo que segmentos
cortados antes de tiempo no inflen la duracion total.

Al procesar los `qm.xyz` segmentados, Tolkien Tools descarta siempre el primer
bloque de coordenadas de cada segmento y cuenta la dinamica desde el segundo.
`merge-xyz` tampoco incluye esos bloques iniciales en el XYZ consolidado.

El mismo criterio se aplica a los bloques segmentados de poblaciones electronicas:
`merge-pop`, `spin-ts` y el merge legacy de `mq_N.dat`/`ms_N.dat` descartan el
primer bloque de cada segmento antes del analisis.

`merge-pop` genera archivos consolidados compatibles con el modo LIO de la rama
de analisis de cargas y spines:

- `mulliken` -> `mulliken_full.dat`
- `mulliken_spin` -> `mulliken_spin_full.dat`

Esto permite correr despues `tolkien-tools 3` en modo LIO sobre la misma carpeta
de salida.

Si por compatibilidad con rutinas viejas se necesitan archivos fragmentados
`mq_N.dat`/`ms_N.dat`, se pueden pedir explicitamente con `--lio-aliases`.

`geom` genera automaticamente un visor 3D HTML del primer frame antes de pedir o
calcular metricas e intenta abrirlo en el navegador. Si `py3Dmol` esta
instalado, usa un visor molecular con esferas, sticks y colores por elemento; si
no, usa Plotly como fallback. Por default muestra los indices atomicos siempre,
porque el objetivo principal es elegir atomos para definir metricas. Para verlos
solo al pasar el mouse:

```bash
tolkien-tools md geom qm_completo.xyz --viewer-labels hover
```

Para forzar un backend:

```bash
tolkien-tools md geom qm_completo.xyz --viewer-backend py3dmol
tolkien-tools md geom qm_completo.xyz --viewer-backend plotly
```

Para generar el HTML sin abrirlo automaticamente:

```bash
tolkien-tools md geom qm_completo.xyz --no-open-viewer
```
