#!/bin/bash
#SBATCH -J evp_ivp
#SBATCH -N 1 
#SBATCH -n 13
#SBATCH -p edr,fdr
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -e ./error.%j
#SBATCH -o ./stdout.%j
#SBATCH --no-requeue

source /etc/profile.d/modules.sh
source activate dedalus3

set -e 

Nphi=144
Nr=64
expname="evpivp_Nphi${Nphi}_Nr${Nr}"
pythonfile="dry_2L_linear.py"
base_dir="/home/linyao/fs06/GFD_Polar_vortex/ddloutput"
exp_dir="${base_dir}/${expname}"

mkdir -p "$exp_dir"
cd "$exp_dir" || exit 1

# === Create unified Python script ===
cat << EOF > "$pythonfile"
import os
import numpy as np
import dedalus.public as d3
import h5py
import logging
logger = logging.getLogger(__name__)

# Dynamically set mode
prob_class = os.environ.get('PROB_CLASS', 'EVP')
restart = False
init_pattern = 'evp_max_growth'

# Parameters
a = 6.99e7
P = 9.925 * 3600
R_dim = a / 2
gamma_dim = 2 * (2 * np.pi / P) / a**2
L = 1e7
U = 100
T = L / U
F = 51.8
gamma = gamma_dim * L**2 * T
R = R_dim / L

Nphi = $Nphi
Nr = $Nr
m_max = 35
kn_zeros = 5
init_pattern_file = './EVP.h5'

coords = d3.PolarCoordinates('phi', 'r')
dist = d3.Distributor(coords, dtype=np.complex128 if prob_class == 'EVP' else np.float64)
disk = d3.DiskBasis(coords, shape=(Nphi, Nr), radius=R, dtype=dist.dtype)
phi, r = dist.local_grids(disk)

psi1 = dist.Field(name='psi1', bases=disk)
psi2 = dist.Field(name='psi2', bases=disk)
tau_psi1 = dist.Field(name='tau_psi1', bases=disk.edge)
tau_psi2 = dist.Field(name='tau_psi2', bases=disk.edge)
s = dist.Field()  # only used in EVP

psi2u = lambda A: d3.Skew(d3.Gradient(A))
lift = lambda A: d3.Lift(A, disk, -1)
q1 = d3.lap(psi1) - F * (psi1 - psi2) + lift(tau_psi1)
q2 = d3.lap(psi2) + F * (psi1 - psi2) + lift(tau_psi2)
u1 = psi2u(psi1)
u2 = psi2u(psi2)

r_sq = dist.Field(bases=disk.radial_basis)
r_sq['g'] = r**2
Psi1 = r_sq / 2
Psi2 = - r_sq / 2
Q1 = d3.lap(Psi1) - F * (Psi1 - Psi2) - 0.5 * gamma * r_sq
Q2 = d3.lap(Psi2) + F * (Psi1 - Psi2) - 0.5 * gamma * r_sq
U1 = psi2u(Psi1)
U2 = psi2u(Psi2)

# === EVP ===
if prob_class == 'EVP':
    problem = d3.EVP([psi1, psi2, tau_psi1, tau_psi2], eigenvalue=s, namespace=locals())
    problem.add_equation("s*q1 = 0")
    problem.add_equation("s*q2 = 0")
    problem.add_equation("psi1(r=R) = 0")
    problem.add_equation("psi2(r=R) = 0")

    def custom_key(z, tol=1e-6):
        return (-z.real if np.abs(z.real) >= tol else 0, np.abs(z.imag))

    evals_list, psi1_list, psi2_list = [], [], []
    for m in range(1, m_max+1):
        solver = problem.build_solver()
        sp = solver.subproblems_by_group[(m, None)]
        solver.solve_dense(sp)
        evals = solver.eigenvalues[np.isfinite(solver.eigenvalues)]
        evals = sorted(evals, key=lambda z: custom_key(z))
        evals_list.append(np.copy(evals[0:kn_zeros*2]))

        psi1_eigen, psi2_eigen = [], []
        for nk in range(kn_zeros*2):
            solver.set_state(np.argmin(np.abs(solver.eigenvalues - evals[nk])), sp.subsystems[0])
            psi1_eigen.append(np.copy(psi1['g'].real))
            psi2_eigen.append(np.copy(psi2['g'].real))
        psi1_list.append(np.copy(psi1_eigen))
        psi2_list.append(np.copy(psi2_eigen))

    with h5py.File(init_pattern_file, 'w') as f:
        tasks = f.create_group('tasks')
        tasks.create_dataset('evals', data=evals_list)
        tasks.create_dataset('psi1', data=psi1_list)
        tasks.create_dataset('psi2', data=psi2_list)
        tasks.create_dataset('phi', data=phi)
        tasks.create_dataset('r', data=r)

# === IVP ===
elif prob_class == 'IVP':
    import time
    timestep = 1e-4
    stop_sim_time = 20
    timestepper = d3.RK443
    file_handler_mode = 'overwrite'

    problem = d3.IVP([psi1, psi2, tau_psi1, tau_psi2], namespace=locals())
    problem.add_equation("dt(q1) + u1@grad(Q1) + U1@grad(q1) = 0")
    problem.add_equation("dt(q2) + u2@grad(Q2) + U2@grad(q2) = 0")
    problem.add_equation("psi1(r=R) = 0")
    problem.add_equation("psi2(r=R) = 0")

    solver = problem.build_solver(timestepper)
    solver.stop_sim_time = stop_sim_time

    if not restart:
        if init_pattern == 'random':
            psi1.fill_random('g', seed=42)
            psi2.fill_random('g', seed=24)
        elif init_pattern == 'evp_max_growth':
            with h5py.File(init_pattern_file, 'r') as f:
                psi1['g'] = f["tasks/psi1"][18,0,:,:]
                psi2['g'] = f["tasks/psi2"][18,0,:,:]
    else:
        solver.load_state('./checkpoints/checkpoints_s4.h5')

    snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.1, max_writes=500, mode=file_handler_mode)
    snapshots.add_task(psi1, name='psi1')
    snapshots.add_task(psi2, name='psi2')

    flow = d3.GlobalFlowProperty(solver, cadence=10)
    flow.add_property((psi1**2 + psi2**2), name='energy')

    try:
        while solver.proceed:
            solver.step(timestep)
            if solver.iteration % 100 == 0:
                print(f"[{time.ctime()}] iter={solver.iteration} t={solver.sim_time:.3f} maxE={flow.max('energy'):.4e}")
    except:
        print("Exception raised, exiting.")
    finally:
        solver.log_stats()
EOF

# === Run both modes ===
echo "===== EVP Phase ====="
PROB_CLASS=EVP mpirun -n 13 python3 "$pythonfile"

echo "===== IVP Phase ====="
PROB_CLASS=IVP mpirun -n 13 python3 "$pythonfile"
