#!/bin/bash
#SBATCH --job-name=FeSSH_60
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=490GB
#SBATCH --partition=mem
###SBATCH --exclude=n01,n06,n19,n20
#SBATCH --nodelist=e03
#SBATCH --err=/home/jmarcolongo/sims/fran/gs.err
#SBATCH --out=/home/jmarcolongo/sims/fran/gs.out

# Aqui van todos los module load que usaste para instalar todos los programas, lio, libxc, libint, amber
module load OpenBLAS/0.3.7-GCC-8.3.0
module load LAPACK/3.8.0-GCC-8.3.0
module load CMake/3.15.3-GCCcore-8.3.0
module load Autoconf/2.69-GCCcore-8.3.0
# module load Eigen/3.3.7-GCCcore-8.3.0
# module load Boost/1.71.0-gompi-2019b
module load GMP/6.1.2-GCCcore-8.3.0
module load FFTW/3.3.8-gompi-2019b
export GFORTRAN_UNBUFFERED_ALL=1
export OMP_NUM_THREADS=8

# Aqui tenes que hacer el source del archivo ese librerias_github.sh
source /home/jmarcolongo/Soft/lio/liohome.sh
source /home/fvieyra/programas/lio/librerias_CPU.sh

# AMBER
# aqui tenes q hacer el source al archivo de variables de amber o poner las variables a mano.
source /home/jmarcolongo/Soft/amber22/amber.sh

sander -O -i d_LR.in  -p 1wla_ssh.prmtop -c QM_0.rst7 -r LR_0.rst7 -o LR_0.out -x LR_0.nc

# ESTE SCRIPT SIRVE PARA CORRER LA VERSION LIO-OPENLR DE GONZA EN CPU, Y ESTÁ ACOPLADO A UN AMBER COMPILADO EXCLUSIVAMENTE PARA ESTO, QUE FUNCA EN CPU TB.
# ESTA VERSIÓN DE LIO ES ANÁLOGA A LA DEV PERO CAPAZ DE HACER LINEAR RESPONSE EN CAPA ABIERTA.
