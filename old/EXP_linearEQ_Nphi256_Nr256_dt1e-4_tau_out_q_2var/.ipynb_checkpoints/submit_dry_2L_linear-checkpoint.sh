#!/bin/bash
#SBATCH -J ivp
#SBATCH -N 1 
#SBATCH -n 13
#SBATCH -p edr,fdr
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -e ./error.%j
#SBATCH -o ./stdout.%j
#SBATCH --no-requeue

echo run begins at: `date`
echo Directory: `pwd`

source /etc/profile.d/modules.sh

echo 'node: '
echo $SLURM_JOB_NODELIST
echo 'cores per node: '
echo $SLURM_TASKS_PER_NODE

source activate dedalus3

mpirun -n 32 python3 dry_2L_linear.py
