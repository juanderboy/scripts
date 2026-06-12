# Charge And Spin Analysis

Rutina conectada actualmente:

```text
/home/juanderboy/scripts/TolkienTools/charge_spin/charge_spin_analysis.py
```

Uso desde el menu maestro:

```bash
tolkien-tools
```

Este modulo contiene la rutina de analisis de cargas Mulliken, CHELPG, Lowdin,
Hirshfeld y poblaciones de spin.

Estructura interna:

- `charge_spin_analysis.py`: entrada compatible usada por el launcher.
- `charge_spin_cli.py`: flujo interactivo y orquestacion.
- `charge_spin_common.py`: utilidades, prompts y configuraciones.
- `charge_spin_io.py`: lectura/escritura de archivos consolidados y series.
- `charge_spin_orca.py`: extraccion de poblaciones desde outputs ORCA.
- `charge_spin_stats.py`: KDE, histogramas, normalizacion y checks de spin.
- `charge_spin_plotting.py`: figuras de histogramas y series temporales.
- `charge_spin_viewer.py`: visor HTML de localizacion de spin y geometria.
- `charge_spin_global.py`: comparaciones globales entre subdirectorios.

Para poblaciones de spin, el flujo interactivo permite elegir como se arman la
estadistica, los histogramas y los archivos de salida:

- `Raw spin values from each snapshot`: usa directamente el valor de spin que
  sale de cada foto/output. Esta es la opcion por defecto.
- `Spin fraction relative to the selected atoms`: normaliza cada snapshot por
  la suma de spin de los atomos/entidades elegidos.

Si hay poblaciones de spin, la seleccion de atomos puede hacerse manualmente o
automaticamente por localizacion de spin. El modo automatico calcula, para cada
snapshot, `abs(spin_atom) / sum(abs(spin_todos_los_atomos))`, promedia esa
fraccion en la dinamica y genera histogramas individuales solo para los atomos
que superan el umbral minimo pedido (5% por defecto). El resto de los atomos se
agrupa como una entidad `resto`, con su propio histograma.

Tambien existe un modo de fragmentos moleculares para analizar zonas como
Fe/porfirina/agua/histidina. Antes de pedir la composicion de cada fragmento,
el flujo genera `spin_fragment_numbering_viewer.html` con todos los atomos
numerados. Despues se ingresan nombres de fragmentos y listas de atomos; cada
fragmento se analiza como una entidad cuyo valor de spin por snapshot es la suma
de los spines atomicos que lo forman. Las listas aceptan atomos separados por
espacios o comas, rangos como `10-18`, y el token `remaining` para tomar todos
los atomos aun no asignados a fragmentos previos. Las definiciones usadas se
guardan en `spin_fragment_definitions.dat` y las salidas principales llevan el
sufijo `with_fragments`. Si quedan atomos sin asignar a ningun fragmento, el
programa avisa cuales son y permite redefinir los fragmentos; si se acepta
continuar, los agrupa automaticamente como `resto` y genera su histograma.

En modo ORCA, aunque el histograma se arme con muchos `SP_*.out`, el visor
`spin_localization_viewer.html` usa una unica geometria representativa: toma el
primer output ORCA ordenado que contenga un bloque
`CARTESIAN COORDINATES (ANGSTROEM)`. En modo LIO, si existe un XYZ en la carpeta
(`qm_completo.xyz`, `qm.xyz` u otro `*.xyz`), el mismo visor se genera desde ese
XYZ. El visor escribe conectividad explicita inferida por distancias
conservadoras para evitar enlaces largos espurios del auto-bonding de 3Dmol. En
ambos casos, el flujo puede intentar abrir el HTML en el navegador por defecto.
