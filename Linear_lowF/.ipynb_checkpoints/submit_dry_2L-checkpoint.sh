#!/bin/bash
#SBATCH -J L_low
#SBATCH -N 1 
#SBATCH -n 13
#SBATCH -p edr,fdr
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -e ./error.%j
#SBATCH -o ./stdout.%j
#SBATCH --no-requeue

source /etc/profile.d/modules.sh
source activate dedalus3

echo "Running job at $(date) on $(hostname)"

for init_m in {1..31}
do 
    export init_m
    echo "Running ivp.py with init_m = $init_m"
    mpirun -n 32 python3 ivp.py
done