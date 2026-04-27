#!/bin/bash

## $-N traj_00003

#SBATCH --job-name="RuCO_P"
#SBATCH --nodes=1
#SBATCH --error="RUN.err"
#SBATCH --output="RUN.out"
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=8
#SBATCH --time=72:00:00
#SBATCH --partition=cpu-x
#SBATCH --mem-per-cpu=8000MB
##SBATCH --nodelist=n02
#SBATCH --exclude=n09,n05,n01,n19,n20,n06,n07,n04,n15,xeon01,xeon02

#----------------------------------------------------------------------------
# EJEMPLO DE SCRIPT DE CORRIDA CPU
# El siguiente script permite correr un programa CPU 
# (Gaussian 16 en este caso) en paralelo con 20 procesadores en la cola cpu-x
# usando los nuevos nodos Xeon.
# MODIFICAR SEGUN EL CASO.
# El script consta de las siguientes secciones:
# 0) Timestamp de inicio de script.
# 1) Carga de modulos.
# 2) Setup de escritura en disco local.
# 3) Ejecucion de comandos de corrida (gaussian en este caso).
# 4) Copia de resultados al home del usuario y borrado de archivos en el nodo.
# 5) Timestamp de finalizacion del script.
#----------------------------------------------------------------------------

# 0) Timestamp de inicio de script.
echo "    Trabajo \"${SLURM_JOB_NAME}\""
echo "    Id: ${SLURM_JOB_ID}"
echo "    Partición: ${SLURM_JOB_PARTITION}"
echo "    Nodos: ${SLURM_JOB_NODELIST}"
echo PRIMARY_DIR=$dir_at_node
date +"Inicio %F - %T"

# 1) Carga de modulos y variables de ambiente.
module purge

export SHARC=/home/jmarcolongo/Soft/sharc4-4.0/bin
export ORCADIR=/home/jmarcolongo/Soft/orca_6_0_1_linux_x86-64_shared_openmpi416
export PATH="/home/jmarcolongo/Soft/orca_6_0_1_linux_x86-64_shared_openmpi416:$PATH"
export LD_LIBRARY_PATH="/home/jmarcolongo/Soft/orca_6_0_1_linux_x86-64_shared_openmpi416:$LD_LIBRARY_PATH"

__conda_setup="$('/home/jmarcolongo/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/jmarcolongo/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/jmarcolongo/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/jmarcolongo/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup

conda activate sharc4.0
module load foss/2021a

# 2) NO EDITAR ESTA SECCION! Setup para escritura en disco local. 
#----------------------------------------------------------------------------
workdir="$(pwd)" 
user="$(whoami)"
dir_at_storage=$workdir
dir_at_node="/tmp/${user}/$SLURM_JOB_ID"
mkdir -p $dir_at_node
cp -r ${dir_at_storage}/* $dir_at_node
cd $dir_at_node
#----------------------------------------------------------------------------

#crea la carpeta de scratch especifica para los n/p.
mkdir -p /tmp/jmarcolongo/sharc_scratch_RuCO_LVC_p

# 4) Ejecucion de comandos de corrida (parametrizaciones de sharc4 en este caso).

lambda () {
    printf "%03d" "$1"
}

for i in $(seq 7 15); do
    PRIMARY_DIR="$dir_at_node/DSPL_$(lambda "$i")_p/"

    echo "Entrando a: $PRIMARY_DIR"
        cd $PRIMARY_DIR
	cp ../ORCA.resources.p ORCA.resources
	if [ -d ../DSPL_000_eq/SAVE ];
        then
          if [ -d ./SAVE ];
          then
            rm -r ./SAVE
          fi
          cp -r ../DSPL_000_eq/SAVE ./
        else
          echo "Should do a reference overlap calculation, but the reference data in ../../DSPL_RESULTS/DSPL_000_eq/ seems not OK."
          exit 1
        fi

        $SHARC/SHARC_ORCA.py QM.in > QM.log 2> QM.er
done

# 5) NO EDITAR ESTA SECCION! Copia de resultados al home del usuario y borrado 
#    de archivos en el nodo.
#----------------------------------------------------------------------------
cp -r ${dir_at_node}/* $dir_at_storage
rm -rf $dir_at_node
rm -rf /tmp/jmarcolongo/sharc_scratch_RuCO_LVC_p
#----------------------------------------------------------------------------

# 6) Timestamp de finalizacion del script.
echo
date +"fin %F - %T"

