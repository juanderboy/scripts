#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30GB
#SBATCH --partition=rtx4070
#SBATCH --gres=gpu:1
###SBATCH --exclude=n01,n06,n19,n20
###SBATCH --nodelist=xg03
#SBATCH --err=gs.err
#SBATCH --out=gs.out
#SBATCH --time=72:00:00


#----------------------------------------------------------------------------
# EJEMPLO DE SCRIPT DE CORRIDA GPU
# El siguiente script permite correr un programa GPU
# (AMBER 20 en este caso) en GPU con 1 procesador en la cola RTX2080.
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
module load Amber/20.11-fosscuda-2020a-AmberTools-21.3-Python-3.8.2
#Amber/22-fosscuda-2020a-AmberTools-23-Python-3.8.2
#Amber/24-fosscuda-2020a-AmberTools-24-Python-3.8.2-lio1.2-libint-2.6.0
#module load Amber/20.11-fosscuda-2020b-AmberTools-21.3
# 2) NO EDITAR ESTA SECCION! Setup para escritura en disco local del nodo.
#----------------------------------------------------------------------------
workdir="$(pwd)"
user="$(whoami)"
dir_at_storage=$workdir
dir_at_node="/tmp/${user}/$SLURM_JOB_ID"
mkdir -p $dir_at_node
cp -r ${dir_at_storage}/* $dir_at_node
cd $dir_at_node
#----------------------------------------------------------------------------

# 4) Ejecucion de comandos de corrida (pmemd.cuda en este caso).
pmemd.cuda -O -i npt.in -o npt.out -p S2r_DMF.prmtop -c heat1b.rst7 -r QM_0.rst7 -x npt.nc &> amber_npt.err
#pmemd.cuda -O -i heat0.in -o heat0.out -p S2r_DMF.prmtop -c S2r_inicial.rst7 -r heat0.rst7 -x heat0.nc
#pmemd.cuda -O -i heat1.in -o heat1b.out -p S2r_DMF.prmtop -c heat1.rst7 -r heat1b.rst7 -x heat1b.nc &> amber_1b.err

# 5) NO EDITAR ESTA SECCION! Copia de resultados al home del usuario y borrado
#    de archivos en el nodo.
#----------------------------------------------------------------------------
cp -r ${dir_at_node}/* $dir_at_storage
rm -rf $dir_at_node
#----------------------------------------------------------------------------

# 6) Timestamp de finalizacion del script.
echo
date +"fin %F - %T"
