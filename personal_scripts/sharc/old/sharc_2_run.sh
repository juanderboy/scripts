#!/bin/bash

## $-N traj_00001

#SBATCH --job-name="Ru_10"
#SBATCH --nodes=1
#SBATCH --error="RUN.err"
#SBATCH --output="RUN.out"
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=8
#SBATCH --time=72:00:00
#SBATCH --partition=cpu
#SBATCH --mem-per-cpu=8000MB
##SBATCH --nodelist=n02
#SBATCH --exclude=n09,n01,n03

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
echoPRIMARY_DIR=$dir_at_node

date +"Inicio %F - %T"

# 1) Carga de modulos y variables de ambiente.
export SHARC=/home/jmarcolongo/Soft/sharc-2.1.2/bin
export ORCADIR=/usr/local/easybuild/software/ORCA/4.2.1-gompi-2019b
export PATH="/usr/local/easybuild/software/ORCA/4.2.1-gompi-2019b:$PATH"
export LD_LIBRARY_PATH="/usr/local/easybuild/software/ORCA/4.2.1-gompi-2019b:$LD_LIBRARY_PATH"


module load ORCA/4.2.1-gompi-2019b ncurses/6.1-GCCcore-8.3.0 HDF5/1.10.5-gompi-2019b netCDF/4.7.1-gompi-2019b OpenBLAS/0.3.7-GCC-8.3.0 libjpeg-turbo/2.0.3-GCCcore-8.3.0 LAPACK/3.8.0-GCC-8.3.0 FFTW/3.3.8-gompi-2019b Python/3.7.4-GCCcore-8.3.0 matplotlib/3.1.1-foss-2019b-Python-3.7.4 Python/2.7.16-GCCcore-8.3.0 GCCcore/8.3.0


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

# 4) Ejecucion de comandos de corrida (gaussian 16 en este caso).

PRIMARY_DIR=$dir_at_node

cd $PRIMARY_DIR

$SHARC/sharc.x input

# 5) NO EDITAR ESTA SECCION! Copia de resultados al home del usuario y borrado 
#    de archivos en el nodo.
#----------------------------------------------------------------------------
cp -r ${dir_at_node}/* $dir_at_storage
rm -rf $dir_at_node
#----------------------------------------------------------------------------

# 6) Timestamp de finalizacion del script.
echo
date +"fin %F - %T"

