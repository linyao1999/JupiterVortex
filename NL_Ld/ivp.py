import numpy as np
import dedalus.public as d3
import matplotlib.pyplot as plt
import logging
import os
import h5py
logger = logging.getLogger(__name__)

restart = False 

# Parameters
# -------- Choosen parameters ---------
Ld = float(os.environ.get("Ld")) # deformation varying from 350 km to 1300 km based on obs
U = float(os.environ.get("U"))  # 80m/s in Siegelman et al. 2018
# -------- fixed parameters -----------
a = 6.99e7 # radius (m)
P = 9.925 * 3600 # rotation period (s)
delta = 1  # H1/H2
# -------- derived parameters -----------
R_dim = a / 2
gamma_dim = 4 * np.pi / P / (a**2)

# Dimensionless parameters
R = R_dim / Ld
gamma = gamma_dim * Ld**3 / U

print(f"Ld    = {Ld}")
print(f"U     = {U}")
print(f"gamma = {gamma}")
print(f"R     = {R}")

# Numerical parameters
Nphi = int(os.environ.get("Nphi"))
Nr = int(os.environ.get("Nr"))
Re = float(os.environ.get("Re"))
timestepper = "RK443"
timestep = 1e-5
initial_timestep = timestep
max_timestep = 1e-3
stop_sim_time = float(os.environ.get("stop_sim_time"))
dtype = np.float64
initv = 1e-3
dealias = 3/2
nu = 1.0 / Re

# Bases
coords = d3.PolarCoordinates('phi', 'r')
dist = d3.Distributor(coords, dtype=dtype)
disk = d3.DiskBasis(coords, shape=(Nphi, Nr), radius=R, dealias=dealias, dtype=dtype)
phi, r = dist.local_grids(disk)

# Fields
psi1 = dist.Field(name='psi1', bases=disk)
psi2 = dist.Field(name='psi2', bases=disk)
q1 = dist.Field(name='q1', bases=disk)
q2 = dist.Field(name='q2', bases=disk)

tau_q1 = dist.Field(name='tau_q1', bases=disk.edge)
tau_q2 = dist.Field(name='tau_q2', bases=disk.edge)
tau_q3 = dist.Field(name='tau_q3', bases=disk.edge)
tau_q4 = dist.Field(name='tau_q4', bases=disk.edge)

tau_bc = dist.Field(name='tau_bc')  # force the integ(psi1 - psi2) = 0; internal energy is conserved

# Substitutions
psi2u = lambda A: d3.Skew(d3.Gradient(A))
lift_basis = disk
lift = lambda A,n: d3.Lift(A, lift_basis, n)
u1 = psi2u(psi1)
u2 = psi2u(psi2)

# Background
r_sq = dist.Field(bases=disk.radial_basis)
r_sq['g'] = r**2  # radial coordinate
Psi1 = r_sq / 2 / R
Psi2 = - r_sq / 2 / R
Q1 = d3.lap(Psi1) - (Psi1 - Psi2) - 0.5 * gamma * r_sq
Q2 = d3.lap(Psi2) + delta * (Psi1 - Psi2) - 0.5 * gamma * r_sq
U1 = psi2u(Psi1)
U2 = psi2u(Psi2)

# Problem
problem = d3.IVP([psi1, psi2, q1, q2, tau_q1, tau_q2, tau_q3, tau_q4, tau_bc], namespace=locals())

problem.add_equation("q1 - (lap(psi1) - (psi1 - psi2)) + lift(tau_q1,-1)= 0")
problem.add_equation("q2 - (lap(psi2) + delta * (psi1 - psi2)) + lift(tau_q2,-1)= 0")
problem.add_equation("dt(q1) - nu*lap(q1) + u1@grad(Q1) + lift(tau_q3,-1) = -(U1@grad(q1) + u1@grad(q1))")
problem.add_equation("dt(q2) - nu*lap(q2) + u2@grad(Q2) + lift(tau_q4,-1) = -(U2@grad(q2) + u2@grad(q2))")

problem.add_equation("psi1(r=R) + tau_bc = 0")
problem.add_equation("psi2(r=R) - tau_bc = 0")
problem.add_equation("q1(r=R) = 0")
problem.add_equation("q2(r=R) = 0")
problem.add_equation("integ(psi1 - psi2) = 0")

# Solver
solver = problem.build_solver(timestepper)
solver.print_subproblem_ranks(solver.subproblems, timestep)
solver.stop_sim_time = stop_sim_time

# Initial conditions
if not restart:
    psi1.fill_random('g', seed=42, distribution='normal', scale=1e-3)
    psi2.fill_random('g', seed=42, distribution='normal', scale=1e-3)
    q1.fill_random('g', seed=42, distribution='normal', scale=1e-3)
    q2.fill_random('g', seed=42, distribution='normal', scale=1e-3)
    psi1.low_pass_filter(scales=0.1) # Keep only the lower modes
    psi2.low_pass_filter(scales=0.1) 
    q1.low_pass_filter(scales=0.1)
    q2.low_pass_filter(scales=0.1) 
    # m, n = dist.coeff_layout.local_group_arrays(psi1.domain, scales=1)
    # psi1['c'] *= (m == 5)
    # psi2['c'] *= (m == 5)
    psi1['g'] *= initv
    psi2['g'] *= initv
    file_handler_mode = 'overwrite'
else:
    import glob
    import re
    checkpoint_path = 'checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_path, 'checkpoints')

    def get_num(filename):
        match = re.search(r'_s(\d+)\.h5$', filename)
        return int(match.group(1)) if match else -1
    
    checkpoint_file = sorted(glob.glob(f"{checkpoint_prefix}_s*.h5"), key=get_num)[0]
    print('checkpoint file is: ', checkpoint_file)
    write, initial_timestep = solver.load_state(checkpoint_file)
    file_handler_mode = 'append'

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.1, max_writes=100, mode=file_handler_mode)
snapshots.add_task(psi1, name='psi1', scales=(1,1))
snapshots.add_task(psi2, name='psi2', scales=(1,1))
snapshots.add_task(q1, name='q1', scales=(1,1))
snapshots.add_task(q2, name='q2', scales=(1,1))
snapshots.add_task(psi1, name='psi1_coef', layout='c', scales=(1,1))
snapshots.add_task(psi2, name='psi2_coef', layout='c', scales=(1,1))
snapshots.add_task(q1, name='q1_coef', layout='c', scales=(1,1))
snapshots.add_task(q2, name='q2_coef', layout='c', scales=(1,1))

scalars = solver.evaluator.add_file_handler('scalars', sim_dt=0.1, max_writes=1000, mode=file_handler_mode)
scalars.add_task(d3.Integrate(u1@u1 + u2@u2)*(U**2), name='KE') 
scalars.add_task(d3.Integrate((psi1-psi2)**2 * (U**2) / 4), name='APE') 

checkpoints = solver.evaluator.add_file_handler('checkpoints', sim_dt=1, max_writes=1, mode=file_handler_mode)
checkpoints.add_tasks(solver.state, scales=(1,1))

# CFL
U1_cfl = u1.evaluate().copy()
U1_cfl['g'] = U1['g']
U2_cfl = u2.evaluate().copy()
U2_cfl['g'] = U2['g']
CFL = d3.CFL(solver, initial_dt=initial_timestep, cadence=10, safety=0.2, threshold=0.1,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u1+U1_cfl)
CFL.add_velocity(u2+U2_cfl)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=100)
flow.add_property(d3.Integrate(u1@u1 + u2@u2), name='KE')

# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        # print(timestep)
        solver.step(timestep)
        # psi1['c'] *= (m == 5)
        # psi2['c'] *= (m == 5)
        if (solver.iteration-1) % 1000 == 0:
            max_KE = flow.max('KE')
            logger.info('Iteration=%i, Time=%e, dt=%e, integ(KE)=%.3e' %(solver.iteration, solver.sim_time, timestep, max_KE))
        # if (solver.iteration-1) % 100 == 0:
        #     psi1['c'] /= max_KE**0.5
        #     psi2['c'] /= max_KE**0.5
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
