#!/bin/bash

source /etc/profile.d/modules.sh
source activate dedalus3

# Custom experiment parameters
Nphi=128
Nr=256

# Re_list=(200)
# F_list=(5.18)

# Re_list=(10)
# F_list=(51.8)

Re_list=(4)
F_list=(100)

# Re_list=(2)
# F_list=(400 900)

mkdir -p log

for Re in "${Re_list[@]}"; do
  for F in "${F_list[@]}"; do

    exp_dir="$HOME/fs06/GFD_Polar_vortex/ddloutput/NL_findRe/Nphi${Nphi}_Nr${Nr}_Re${Re}_F${F}"
    mkdir -p "$exp_dir"

    alias_dir="out_Nphi${Nphi}_Nr${Nr}_Re${Re}_F${F}"
    rm -f "$alias_dir"
    ln -s "$exp_dir" "$alias_dir"

    rm -f "$exp_dir"/*.py
    cp ./ivp.py ./IVP*.ipynb "$exp_dir"

    # Create job script
    jobscript="${exp_dir}/run_job.sh"
    cat << EOF > "$jobscript"
#!/bin/bash
#SBATCH -J Re${Re}_F${F}
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -p edr,fdr
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -e log/error_Re${Re}_F${F}.%j
#SBATCH -o log/stdout_Re${Re}_F${F}.%j
#SBATCH --no-requeue

source /etc/profile.d/modules.sh
source activate dedalus3

echo "Running job at \$(date) on \$(hostname)"
cd "$exp_dir" || exit 1
export Nphi=${Nphi}
export Nr=${Nr}
export Re=${Re}
export F=${F}
echo "Nphi=\${Nphi}, Nr=\${Nr}, Re=\${Re}, F=\${F}"
mpirun -n 32 python3 ivp.py
echo "Finished job at \$(date)"
EOF

    chmod +x "$jobscript"
    sbatch "$jobscript"

  done
done
