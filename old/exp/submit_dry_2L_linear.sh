#!/bin/bash
#SBATCH -J ivp
#SBATCH -N 1 
#SBATCH -n 13
#SBATCH -p edr,fdr
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -e ./log/error.%j
#SBATCH -o ./log/stdout.%j
#SBATCH --no-requeue

# if test -f /etc/profile.d/modules.sh    ; then . /etc/profile.d/modules.sh    ; fi
# if test -f /etc/profile.d/zz_modules.sh ; then . /etc/profile.d/zz_modules.sh ; fi

echo run begins at: `date`
echo Directory: `pwd`

# source /etc/profile.d/modules.sh

# echo 'node: '
# echo $SLURM_JOB_NODELIST
# echo 'cores per node: '
# echo $SLURM_TASKS_PER_NODE

# module purge
# module add anaconda3/2023.07
# source activate dedalus3
# echo "list of loaded modules:"
# module list
# echo " "


echo 'Your job is running on node(s):'
echo $SLURM_JOB_NODELIST
echo 'Cores per node:'
echo $SLURM_TASKS_PER_NODE

module add ffmpeg

if test -f /etc/profile.d/modules.sh    ; then . /etc/profile.d/modules.sh    ; fi
if test -f /etc/profile.d/zz_modules.sh ; then . /etc/profile.d/zz_modules.sh ; fi
if [[ ${SLURM_JOB_PARTITION} == *"hdr"* ]]; then
  module purge
  module load intel/2021.4.0_rhel8
  module load openmpi/3.1.6
elif [[ ${SLURM_JOB_PARTITION} == *"fdr"* || ${SLURM_JOB_PARTITION} == *"edr"* ]]; then
  module add anaconda3/2023.07
fi
echo "list of loaded modules:"
module list
echo " "

source activate dedalus3

mpirun -n 12 python3 dry_2L_linearEQ_noNu.py

