import numpy as np
import dedalus.public as d3
import matplotlib.pyplot as plt
import logging
import os 
import h5py
logger = logging.getLogger(__name__)

# Parameters
# Physical paramters 
F1 = 51.8  # L**2/Ld1**2
F2 = 51.8 
a = 6.99e7
T = 9.925*3600
L = 1e7
U = 100
gamma = 4 * np.pi / T / a / a * (L**3) / U
# gamma = 0.7198 # 2 omega / a**2 * L**3 / U
a_norm = a / L / 2
timestep = 1e-3
timestepper = d3.RK443
stop_sim_time = 10
initv = 1

snapshots_dir = f'/net/fs06/d0/linyao/GFD_Polar_vortex/ddloutput/'
snapshots_name = f'snapshots_F51_U_{U}_linear_noNu_2var'

snapshots_file = snapshots_dir + snapshots_name

os.makedirs(snapshots_dir, exist_ok=True)

# numerical parameters
m = 10
Nphi = 24
Nr = 64
dtype = np.float64

# Bases
coords = d3.PolarCoordinates('phi', 'r')
dist = d3.Distributor(coords, dtype=dtype)
disk = d3.DiskBasis(coords, shape=(Nphi, Nr), radius=3.5, dtype=dtype)
phi, r = dist.local_grids(disk)

# Fields
s = dist.Field(name='s')
psi1 = dist.Field(name='psi1', bases=disk)
psi2 = dist.Field(name='psi2', bases=disk)
tau_psi1 = dist.Field(name='tau_psi1', bases=disk.edge)
tau_psi2 = dist.Field(name='tau_psi2', bases=disk.edge)

# Substitutions
# dt = lambda A: s*A
psi2u = lambda A: -d3.Skew(d3.Gradient(A))
lift_basis = disk.derivative_basis(2)
lift = lambda A: d3.Lift(A, lift_basis, -1)
q1 = d3.lap(psi1) - F1 * (psi1 - psi2)
q2 = d3.lap(psi2) + F2 * (psi1 - psi2)

# Background 
r_field = dist.Field(bases=disk.radial_basis)
r_field['g'] = r**2  # radial coordinate
psi1_0 = 0.5 * (r_field)
psi2_0 = - 0.5 * (r_field)
Q1 = d3.Laplacian(psi1_0) - F1 * (psi1_0 - psi2_0) - 0.5 * gamma * (r_field)
Q2 = d3.Laplacian(psi2_0) + F2 * (psi1_0 - psi2_0) - 0.5 * gamma * (r_field)

# Problem
problem = d3.IVP([psi1, psi2, tau_psi1, tau_psi2], namespace=locals())
problem.add_equation("dt(lap(psi1) - F1 * (psi1 - psi2)) " \
                        "+ (psi2u(psi1)) @ grad(Q1) " \
                        "+ psi2u(psi1_0) @ grad(lap(psi1) - F1 * (psi1 - psi2))" \
                        "+ lift(tau_psi1) = 0")
problem.add_equation("dt(lap(psi2) + F2 * (psi1 - psi2)) " \
                        "+ (psi2u(psi2)) @ grad(Q2) " \
                        "+ psi2u(psi2_0) @ grad(lap(psi2) + F2 * (psi1 - psi2))" \
                        "+ lift(tau_psi2) = 0")
problem.add_equation("psi1(r=a_norm) = 0") # 7 is a/L
problem.add_equation("psi2(r=a_norm) = 0")

# Solver
# solver = problem.build_solver()
solver = problem.build_solver(timestepper)
solver.print_subproblem_ranks(solver.subproblems, timestep)
solver.stop_sim_time = stop_sim_time

# Initial conditions
psi1.fill_random('g', seed=42, distribution='standard_normal') # Random noise
psi1['g'] *= initv
psi1.low_pass_filter(scales=0.9) # Keep only lower fourth of the modes

psi2.fill_random('g', seed=42, distribution='standard_normal') # Random noise
psi2['g'] *= initv
psi2.low_pass_filter(scales=0.9) # Keep only lower fourth of the modes

# Analysis
snapshots = solver.evaluator.add_file_handler(snapshots_file, sim_dt=0.1, max_writes=500, mode='overwrite')
snapshots.add_task(psi1, scales=(16, 1), name='psi1')
# snapshots.add_task(tau1, scales=(16, 1), name='tau1')


# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=1)
flow.add_property((psi2u(psi1))**2, name='ke')  # could be a vector; to be check 

# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
        # timestep=1e-3
        #timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % 100 == 0:
            # max_u = np.sqrt(flow.max('u2'))
            max_psi1 = np.sqrt(flow.max('ke'))
            logger.info("Iteration=%i, Time=%e, dt=%e, max(psi1)=%e" %(solver.iteration, solver.sim_time, timestep, max_psi1))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()


