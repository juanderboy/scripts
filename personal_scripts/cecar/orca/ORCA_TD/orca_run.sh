#!/bin/bash
#SBATCH --job-name="td_acheto"
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --error="slurm-%j.err"
#SBATCH --output="slurm-%j.out"
#SBATCH --partition=cpu-x
#SBATCH --mem=8000
#SBATCH --time=72:00:00
#SBATCH --exclude=n09,n01,n19,n20,n04,n05,n06,n07,n12,n13,xg01

#----------------------------------------------------------------------------
# EJEMPLO DE SCRIPT DE CORRIDA CPU
# El siguiente script permite correr un programa CPU 
# (ORCA en este caso) en paralelo con 8 procesadores en la cola cpu
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
echo
date +"Inicio %F - %T"

# 1) Carga de modulos.
module purge
module load foss/2021a
#module load ORCA/4.2.1-gompi-2019b ncurses/6.1-GCCcore-8.3.0 HDF5/1.10.5-gompi-2019b netCDF/4.7.1-gompi-2019b OpenBLAS/0.3.7-GCC-8.3.0 libjpeg-turbo/2.0.3-GCCcore-8.3.0 LAPACK/3.8.0-GCC-8.3.0 FFTW/3.3.8-gompi-2019b Python/3.7.4-GCCcore-8.3.0 matplotlib/3.1.1-foss-2019b-Python-3.7.4 Python/2.7.16-GCCcore-8.3.0
export ORCADIR=/home/jmarcolongo/Soft/orca_6_0_1_linux_x86-64_shared_openmpi416

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

# 4) Ejecucion de comandos de corrida
${ORCADIR}/orca RuCO_TD_acheto.inp > RuCO_TD_acheto.out

#/home/jmarcolongo/Soft/orca_6_0_1_linux_x86-64_shared_openmpi416/orca RuCO_opt.inp > RuCO_opt.out

# 5) NO EDITAR ESTA SECCION! Copia de resultados al home del usuario y borrado 
#    de archivos en el nodo.
#----------------------------------------------------------------------------
cp -r ${dir_at_node}/* $dir_at_storage
rm -rf $dir_at_node
#----------------------------------------------------------------------------

# 6) Timestamp de finalizacion del script.
echo
date +"fin %F - %T"
