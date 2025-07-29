"""
Dedalus script simulating librational instability in a disk by solving the
incompressible Navier-Stokes equations linearized around a background librating
flow. This script demonstrates solving an initial value problem in the disk.
It can be ran serially or in parallel, and uses the built-in analysis framework
to save data snapshots to HDF5 files. The `plot_disk.py` and `plot_scalars.py`
scripts can be used to produce plots from the saved data. The simulation should
take roughly 20 cpu-minutes to run.

The problem is non-dimesionalized using the disk radius and librational frequency,
so the resulting viscosity is related to the Ekman number as:

    nu = Ekman

For incompressible hydro in the disk, we need one tau term for the velocity.
Here we lift to the original (k=0) basis.

To run and plot using e.g. 4 processes:
    $ mpiexec -n 4 python3 libration.py
    $ mpiexec -n 4 python3 plot_disk.py snapshots/*.h5
    $ python3 plot_scalars.py scalars/*.h5
"""
# salloc --nodes 1 --qos interactive --time 02:00:00 --constraint cpu --account=m3312

import numpy as np
import dedalus.public as d3
from scipy.special import jv
import logging
logger = logging.getLogger(__name__)

# Parameters
# Physical paramters 
F1 = 51.8  # L**2/Ld1**2
F2 = 51.8
a = 6.99e7
T = 9.925*3600
L = 1e7
U = 100
gamma = 4 * np.pi / a / a / T * (L**3) / U
# gamma = 0.7198 # 2 omega / a**2 * L**3 / U
a_norm = a / L / 2 

# kphi = 10
Nphi, Nr = 32, 64
# Ekman = 1 / 2 / 20**2
# Ro = 40
dealias = 3/2
stop_sim_time = 10
# timestepper = d3.SBDF2
timestepper = d3.RK443
# timestep = 1e-3
timestep =1e-3
max_timestep = 1e-3
dtype = np.float64
nu = 1e-11 # 1e-11

# output names
snapshots_dir = f'snapshots_Fhigh_U_{U}_2tau_linearRHS_lapNu_q'

# Bases
coords = d3.PolarCoordinates('phi', 'r')
dist = d3.Distributor(coords, dtype=dtype)
disk = d3.DiskBasis(coords, shape=(Nphi, Nr), radius=a_norm, dealias=dealias, dtype=dtype)
edge = disk.edge

# Fields
psi1 = dist.Field(name='psi1', bases=disk)
psi2 = dist.Field(name='psi2', bases=disk)
q1 = dist.Field(name='q1', bases=disk)
q2 = dist.Field(name='q2', bases=disk)
tau_psi1 = dist.Field(name='tau_psi1', bases=disk.edge)
tau_psi2 = dist.Field(name='tau_psi2', bases=disk.edge)
tau_psi3 = dist.Field(name='tau_psi3', bases=disk.edge)
tau_psi4 = dist.Field(name='tau_psi4', bases=disk.edge)
taus = [tau_psi1, tau_psi2, tau_psi3, tau_psi4]

# Substitutions
phi, r = dist.local_grids(disk)
lift = lambda A,n: d3.Lift(A, disk, n)
psi2u = lambda A: -d3.Skew(d3.Gradient(A))
r_field = dist.Field(bases=disk.radial_basis)
r_field['g'] = r  # radial coordinate


# # Background librating flow
t = dist.Field()
psi1_0 = 0.5 * (r_field**2)
psi2_0 = - 0.5 * (r_field**2)
Q1 = d3.Laplacian(psi1_0) - F1 * (psi1_0 - psi2_0) - 0.5 * gamma * (r_field**2)
Q2 = d3.Laplacian(psi2_0) + F2 * (psi1_0 - psi2_0) - 0.5 * gamma * (r_field**2)


# Problem
problem = d3.IVP([q1, q2, psi1, psi2]+taus, time=t, namespace=locals())
problem.add_equation("q1 - (lap(psi1) - F1 * (psi1 - psi2)) + lift(tau_psi1,-1) = 0")
problem.add_equation("q2 - (lap(psi2) + F2 * (psi1 - psi2)) + lift(tau_psi2,-1) = 0")
problem.add_equation("dt(q1) " \
                        "+ (psi2u(psi1)) @ grad(Q1) " \
                        "+ (psi2u(psi1_0)) @ grad(q1)" \
                        "- nu * lap(q1)" \
                        "+ lift(tau_psi3,-1)= 0")
problem.add_equation("dt(q2) " \
                        "+ (psi2u(psi2)) @ grad(Q2) " \
                        "+ (psi2u(psi2_0)) @ grad(q2)" \
                        "- nu * lap(q2)" \
                        "+ lift(tau_psi4,-1)= 0")

problem.add_equation("q1(r=a_norm) = 0") # 7 is a/L
problem.add_equation("q2(r=a_norm) = 0")
problem.add_equation("psi1(r=a_norm) = 0") # 7 is a/L
problem.add_equation("psi2(r=a_norm) = 0")

# Solver
solver = problem.build_solver(timestepper)
solver.print_subproblem_ranks(solver.subproblems, timestep)
solver.stop_sim_time = stop_sim_time

# Initial conditions
psi1.fill_random('g', seed=42, distribution='standard_normal') # Random noise
psi1['g'] *= 1e-5
psi1.low_pass_filter(scales=0.25) # Keep only lower fourth of the modes

psi2.fill_random('g', seed=42, distribution='standard_normal') # Random noise
psi2['g'] *= 1e-5
psi2.low_pass_filter(scales=0.25) # Keep only lower fourth of the modes

q1.fill_random('g', seed=42, distribution='standard_normal') # Random noise
q1['g'] *= 1e-5
q1.low_pass_filter(scales=0.25) # Keep only lower fourth of the modes

q2.fill_random('g', seed=42, distribution='standard_normal') # Random noise
q2['g'] *= 1e-5
q2.low_pass_filter(scales=0.25) # Keep only lower fourth of the modes

# Analysis
snapshots = solver.evaluator.add_file_handler(snapshots_dir, sim_dt=timestep*50, max_writes=100, mode='overwrite')
snapshots.add_task(psi1, scales=(16, 1), name='psi1')
snapshots.add_task(psi2, scales=(16, 1), name='psi2')
# snapshots.add_task(q1, scales=(16, 1), name='q1')
# snapshots.add_task(q2, scales=(16, 1), name='q2')
# scalars = solver.evaluator.add_file_handler('scalars', sim_dt=0.01)
# scalars.add_task(d3.integ(0.5*u@u), name='KE')

# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep*1e-3, cadence=5, safety=0.1, threshold=0.02,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(psi2u(psi1))

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=1)
# flow.add_property(u@u, name='u2')
flow.add_property(psi1, name='psi1')  # could be a vector; to be check 

# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
        # timestep=1e-3
        #timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            # max_u = np.sqrt(flow.max('u2'))
            max_psi1 = np.sqrt(flow.max('psi1'))
            logger.info("Iteration=%i, Time=%e, dt=%e, max(psi1)=%e" %(solver.iteration, solver.sim_time, timestep, max_psi1))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
