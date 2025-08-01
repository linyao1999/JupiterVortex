#!/bin/bash

source /etc/profile.d/modules.sh
source activate dedalus3

# Custom experiment parameters
Nphi=320
Nr=128
stop_sim_time=400
Ld=1e7
U=100
Re_list=(1e16 1e14)

mkdir -p log

for Re in "${Re_list[@]}"; do
    exp_dir="$HOME/fs06/GFD_Polar_vortex/ddloutput/NL_Ld/Nphi${Nphi}_Nr${Nr}_Re${Re}_Ld${Ld}_U${U}"
    mkdir -p "$exp_dir"
    
    alias_dir="output/Nphi${Nphi}_Nr${Nr}_Re${Re}_Ld${Ld}_U${U}"
    mkdir -p output 
    rm -f "$alias_dir"
    ln -s "$exp_dir" "$alias_dir"
    
    rm -f "$exp_dir"/*.py
    cp ./ivp.py "$exp_dir"
    
    # Create job script
    jobscript="${exp_dir}/run_job.sh"
    cat << EOF > "$jobscript"
#!/bin/bash
#SBATCH -J Re${Re}Ld${Ld}
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -p edr,fdr
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -e log/error_Re${Re}_Ld${Ld}.%j
#SBATCH -o log/stdout_Re${Re}_Ld${Ld}.%j
#SBATCH --no-requeue

source /etc/profile.d/modules.sh
source activate dedalus3

echo "Running job at \$(date) on \$(hostname)"
cd "$exp_dir" || exit 1
export Nphi=${Nphi}
export Nr=${Nr}
export Re=${Re}
export F=${F}
export Ld=${Ld}
export U=${U}
export stop_sim_time=${stop_sim_time}
echo "Nphi=\${Nphi}, Nr=\${Nr}, Re=\${Re}, Ld=\${Ld}, U=\${U}, stop_sim_time=\${stop_sim_time}"
mpirun -n 32 python3 ivp.py
echo "Finished job at \$(date)"
EOF

    chmod +x "$jobscript"
    sbatch "$jobscript"
done
