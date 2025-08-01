import numpy as np
import dedalus.public as d3
import matplotlib.pyplot as plt
import logging
import os
import h5py
logger = logging.getLogger(__name__)

restart = False 

# Parameters
# Physical parameters
a = 6.99e7 # radius (m)
P = 9.925 * 3600 # rotation period (s)
R_dim = a / 2
gamma_dim = 2 * (2 * np.pi / P) / a**2
# Dimensionless parameters
L = 1e7 # m
U = 100 # m/s
T = L / U # s
# F = 51.8 # = L**2 / Ld**2
F = 5.18
gamma = gamma_dim * L**2 * T
R = R_dim / L

print(f"gamma = {gamma}")
print(f"R     = {R}")

# Numerical parameters
Nphi = 32
Nr = 64
Re = 10
timestepper = "RK443"
timestep = 1e-2
initial_timestep = timestep
max_timestep = 1e-2
stop_sim_time = 2
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
# q1 = d3.lap(psi1) - F * (psi1 - psi2)
# q2 = d3.lap(psi2) + F * (psi1 - psi2)
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
problem = d3.IVP([psi1, psi2, q1, q2, tau_q1, tau_q2, tau_q3, tau_q4, tau_bc], namespace=locals())

problem.add_equation("q1 - (lap(psi1) - F * (psi1 - psi2)) + lift(tau_q1,-1)= 0")
problem.add_equation("q2 - (lap(psi2) + F * (psi1 - psi2)) + lift(tau_q2,-1)= 0")
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
    psi1.low_pass_filter(scales=0.25) # Keep only lower fourth of the modes
    psi2.low_pass_filter(scales=0.25) # Keep only lower fourth of the modes
    q1.low_pass_filter(scales=0.25) # Keep only lower fourth of the modes
    q2.low_pass_filter(scales=0.25) # Keep only lower fourth of the modes
    # m, n = dist.coeff_layout.local_group_arrays(psi1.domain, scales=1)
    # psi1['c'] *= (m == 5)
    # psi2['c'] *= (m == 5)
    psi1['g'] *= initv
    psi2['g'] *= initv
    file_handler_mode = 'overwrite'
else:
    write, initial_timestep = solver.load_state('snapshots/snapshots_s2.h5')
    file_handler_mode = 'append'

# Analysis , sim_dt=1

snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.1, max_writes=1, mode=file_handler_mode)
snapshots.add_task(psi1, name='psi1', layout='c', scales=(1,1))
snapshots.add_task(psi2, name='psi2', scales=(1,1))

# snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=1, max_writes=1, mode=file_handler_mode)
# snapshots.add_tasks(solver.state, scales=(1,1))


# scalars = solver.evaluator.add_file_handler('scalars', sim_dt=0.01)
# scalars.add_tasks(d3.Average(u1@u1 + u2@u2), name='KE')

# CFL
# print('line 124')
# print(initial_timestep)
U1_cfl = u1.evaluate().copy()
U1_cfl['g'] = U1['g']
CFL = d3.CFL(solver, initial_dt=initial_timestep, cadence=10, safety=0.2, threshold=0.1,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u1+U1_cfl)
# CFL.add_velocity(u1+U1_cfl)
# CFL.add_velocity(u1)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=100)
# flow.add_property(d3.Integrate(u1@u1 + u2@u2), layout='g', name='KE')
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
        if (solver.iteration-1) % 100 == 0:
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
