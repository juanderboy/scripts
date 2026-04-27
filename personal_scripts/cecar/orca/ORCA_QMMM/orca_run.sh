#!/bin/bash
#SBATCH --job-name="OH_NTO_27"
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --error="slurm-%j.err"
#SBATCH --output="slurm-%j.out"
#SBATCH --partition=cpu
#SBATCH --mem=32000
#SBATCH --time=72:00:00
#SBATCH --exclude=n01,n06,n09,n19,n20

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
module load Amber/20.11-fosscuda-2020b-AmberTools-21.3

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

# 4) Ejecucion de comandos de corrida. va a cada carpeta en que se corrio una
# dinámica, y arma con cpptraj y con orca_mm todo lo necesario para hacer
# QM-MM con orca. Luego corre un td.

for j in {33..33}
do
mkdir ${j}
cd ${j}
cp ../QM_${j}.rst7 .
cp ../orca_td.inp .
cp ../1wla_HID_OH_solv.prmtop .
cpptraj -p 1wla_HID_OH_solv.prmtop -y QM_${j}.rst7 -x orca.pdb
orca_mm -convff -AMBER 1wla_HID_OH_solv.prmtop
${ORCADIR}/orca orca_td.inp > TD_${j}.out
mv orca.pdb QM_${j}.pdb
cp TD_${j}.out ../.
cp TD_${j}.out /home/jmarcolongo/sims/meli-andressa/FeOH/TD_propionatos/.
cd ..
cp -r ${j} /home/jmarcolongo/sims/meli-andressa/FeOH/TD_propionatos/.
done


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
