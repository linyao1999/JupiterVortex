import numpy as np
import dedalus.public as d3
import matplotlib.pyplot as plt
import logging
import os
import h5py
logger = logging.getLogger(__name__)

# Parameters
# Physical paramters
a = 6.99e7 # radius (m)
P = 9.925 * 3600 # rotation period (s)
R_dim = a / 2
gamma_dim = 2 * (2 * np.pi / P) / a**2
# Dimensionless parameters
L = 1e7 # m
U = 10 # m/s
T = L / U # s
F = 5.18 # = L**2 / Ld**2
gamma = gamma_dim * L**2 * T
R = R_dim / L

print(f"gamma = {gamma}")
print(f"R     = {R}")

# Numerical parameters
Nphi = 64
Nr = 64
timestepper = "RK443"
timestep = 0.01
stop_sim_time = 20
dtype = np.float64
initv = 1e-6 

# Bases
coords = d3.PolarCoordinates('phi', 'r')
dist = d3.Distributor(coords, dtype=dtype)
disk = d3.DiskBasis(coords, shape=(Nphi, Nr), radius=R, dtype=dtype)
phi, r = dist.local_grids(disk)

# Fields
psi1 = dist.Field(name='psi1', bases=disk)
psi2 = dist.Field(name='psi2', bases=disk)
tau_psi1 = dist.Field(name='tau_psi1', bases=disk.edge)
tau_psi2 = dist.Field(name='tau_psi2', bases=disk.edge)

# Substitutions
psi2u = lambda A: d3.Skew(d3.Gradient(A))
lift_basis = disk
lift = lambda A: d3.Lift(A, lift_basis, -1)
q1 = d3.lap(psi1) - F * (psi1 - psi2)
q2 = d3.lap(psi2) + F * (psi1 - psi2)
u1 = psi2u(psi1)
u2 = psi2u(psi2)

# Background
r_sq = dist.Field(bases=disk.radial_basis)
r_sq['g'] = r**2  # radial coordinate
Psi1 = r_sq / 2
Psi2 = - r_sq / 2
Q1 = d3.lap(Psi1) - F * (Psi1 - Psi2) - 0.5 * gamma * r_sq
Q2 = d3.lap(Psi2) + F * (Psi1 - Psi2) - 0.5 * gamma * r_sq
U1 = psi2u(Psi1)
U2 = psi2u(Psi2)

# Problem
problem = d3.IVP([psi1, psi2, tau_psi1, tau_psi2], namespace=locals())
problem.add_equation("dt(q1) + lift(tau_psi1) = -(u1@grad(Q1) + U1@grad(q1))")
problem.add_equation("dt(q2) + lift(tau_psi2) = -(u2@grad(Q2) + U2@grad(q2))")
problem.add_equation("psi1(r=R) = 0")
problem.add_equation("psi2(r=R) = 0")

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
psi1.fill_random('g', seed=42, distribution='normal', scale=1e-3)
psi2.fill_random('g', seed=42, distribution='normal', scale=1e-3)
# m, n = dist.coeff_layout.local_group_arrays(psi1.domain, scales=1)
# psi1['c'] *= (m == 5)
# psi2['c'] *= (m == 5)
psi1['g'] *= initv
psi2['g'] *= initv

# Analysis , sim_dt=1
# snapshots = solver.evaluator.add_file_handler('snapshots', iter=100, max_writes=10)
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.1, max_writes=1000)
snapshots.add_tasks(solver.state, scales=(8,1))

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(u1@u1 + u2@u2, name='KE')

# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
        solver.step(timestep)
        # psi1['c'] *= (m == 5)
        # psi2['c'] *= (m == 5)
        if (solver.iteration-1) % 10 == 0:
            max_KE = flow.max('KE')
            logger.info('Iteration=%i, Time=%e, dt=%e, max(KE)=%.3e' %(solver.iteration, solver.sim_time, timestep, max_KE))
        # if (solver.iteration-1) % 100 == 0:
        #     psi1['c'] /= max_KE**0.5
        #     psi2['c'] /= max_KE**0.5
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
