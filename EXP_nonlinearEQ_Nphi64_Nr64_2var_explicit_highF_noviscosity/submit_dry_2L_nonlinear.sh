#!/bin/bash
#SBATCH -J evp_ivp
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

# JupiterVortex/EXP_nonlinearEQ_Nphi64_Nr64_2var_explicit_highF_noviscosity/ivp.py

exp_dir="$HOME/fs06/GFD_Polar_vortex/ddloutput/EXP_nonlinearEQ_Nphi64_Nr64_2var_explicit_highF_noviscosity"

echo "Creating directory: $exp_dir"
mkdir -p "$exp_dir"

alias_dir="output"
rm -f "$alias_dir"
ln -s "$exp_dir" "$alias_dir"

echo "Copying files to: $exp_dir"
rm -f "$exp_dir"/*.py
cp ./ivp.py ./IVP*.ipynb "$exp_dir"

cd "$exp_dir" || exit 1

echo "Running job at $(date) on $(hostname)"
mpirun -n 32 python3 ivp.py