#!/bin/bash

#el script está armado para entrar en las carpetas ICOND_00001 hasta ICOND_01000 y en cada una correr el run.sh
#te va avisando lo que va haciendo.
# si son muchas carpetas a explorar, adaptar esto como un slurm de corrida en cecar porque demora un rato largo y diego va a putear.

for i in $(seq -f "%05g" 2 1000); do
    dir="ICOND_${i}"
    script="${dir}/run.sh"

    echo "Procesando: $dir"

    if [ -x "$script" ]; then
        (cd "$dir" && ./run.sh)
        echo "✔ Finalizado: $dir"
    else
        echo "⚠ No se puede ejecutar $script (no existe o no tiene permisos)"
    fi
done

