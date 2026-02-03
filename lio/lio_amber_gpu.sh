#!/bin/bash
#SBATCH --job-name=FeSSH_60
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=20GB
#SBATCH --partition=rtx4070
###SBATCH --exclude=n01,n06,n19,n20
###SBATCH --nodelist=xg03
#SBATCH --err=gs.err
#SBATCH --out=gs.out

# Aqui van todos los module load que usaste para instalar todos los programas, lio, libxc, libint, amber
module load OpenBLAS/0.3.7-GCC-8.3.0
module load LAPACK/3.8.0-GCC-8.3.0
module load CMake/3.15.3-GCCcore-8.3.0
module load Autoconf/2.69-GCCcore-8.3.0
# module load Eigen/3.3.7-GCCcore-8.3.0
# module load Boost/1.71.0-gompi-2019b
module load GMP/6.1.2-GCCcore-8.3.0
module load FFTW/3.3.8-gompi-2019b
module load fosscuda/2019b OpenBLAS/0.3.7-GCC-8.3.0

export GFORTRAN_UNBUFFERED_ALL=1
export OMP_NUM_THREADS=6

# Aqui tenes que hacer el source del archivo ese librerias_github.sh
source /home/jmarcolongo/Soft/lio-gpu/liohome.sh
source /home/fvieyra/programas/lio/librerias_CPU.sh

# AMBER
# aqui tenes q hacer el source al archivo de variables de amber o poner las variables a mano.
source /home/jmarcolongo/Soft/amber_lio_gpu/amber22/amber.sh

sander -O -i d_QM.in -p 1wla_ssh.prmtop -c MM_2.rst7 -r QM_2.rst7 -o QM_2.out -x QM_2.nc

##lio dev para gpu con su amber22 compilado exclusivo. hace dinamicas qmmmm capa abierta bien :). Este lio necesita un "lio.in"
# que debe tener todo lo de lio. para que lea bien la topología, dentro del lio.in hay que meter al fondo lo siguiente:

#&lionml
# ndyn_steps = 0,
# edyn_steps = 0,
#&end

