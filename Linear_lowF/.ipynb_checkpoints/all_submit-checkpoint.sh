#!/bin/bash

mkdir -p jobs
mkdir -p log

for init_m in {1..31}
do
    job_script="jobs/job_init_m_${init_m}.sh"

    cat << EOF > "$job_script"
#!/bin/bash
#SBATCH -J L_low_m${init_m}
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

export init_m=${init_m}
echo "Running ivp.py with init_m = \$init_m at \$(date) on \$(hostname)"
mpirun -n 32 python3 \$HOME/JupiterVortex/Linear_lowF/ivp.py
EOF

    chmod +x "$job_script"
    sbatch "$job_script"
done
