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
tolkien-tools md geom qm_completo.xyz --metric dFeN:distance:9,10
tolkien-tools md merge-pop --sources mulliken_spin lowdin_spin
tolkien-tools md spin-ts --source mulliken_spin --atoms 9 10
tolkien-tools md split-nc sistema.prmtop 'QM_*.nc' 250-300
```

La rama asume el formato habitual de corridas QMMM/MD fragmentadas: una carpeta
raiz con segmentos numericos, cada uno con archivos como `qm.xyz`, `d_QM.in`,
`mulliken`, `mulliken_spin`, `lowdin`, `lowdin_spin` y trayectorias `QM_*.nc`.

