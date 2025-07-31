#!/bin/bash
#SBATCH -J L_low_m27
#SBATCH -N 1 
#SBATCH -n 32
#SBATCH -p edr,fdr
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -e ./log/error.%j
#SBATCH -o ./log/stdout.%j
#SBATCH --no-requeue

source /etc/profile.d/modules.sh
source activate dedalus3

export init_m=27
echo "Running ivp.py with init_m = $init_m at $(date) on $(hostname)"
mpirun -n 32 python3 $HOME/JupiterVortex/Linear_lowF/ivp.py
