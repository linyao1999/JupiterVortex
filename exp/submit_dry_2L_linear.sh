#!/bin/bash
SBATCH -J ivp
SBATCH -N 1 
SBATCH -n 12
SBATCH -t 12:00:00
SBATCH -e ./log/error.%j
SBATCH -o ./log/stdout.%j


echo run begins at: `date`
echo Directory: `pwd`

echo 'node: '
echo $SLURM_JOB_NODELIST
echo 'cores per node: '
echo $SLURM_TASKS_PER_NODE

module add anaconda3/2023.07
source activate dedalus3

mpirun -n 12 python3 dry_2L_linearEQ_noNu.py

