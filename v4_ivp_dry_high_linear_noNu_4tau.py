import numpy as np
import dedalus.public as d3
from scipy.special import jv
import logging
logger = logging.getLogger(__name__)

# salloc --nodes 1 --qos interactive --time 04:00:00 --constraint cpu --account=m3312

# Parameters
# Physical paramters 
F1 = 51.8  # L**2/Ld1**2
F2 = 51.8
a = 6.99e7
T = 9.925*3600
L = 1e7
U = 10
gamma = 4 * np.pi / a / a / T * (L**3) / U  # gamma = 0.7198 # 2 omega / a**2 * L**3 / U
a_norm = a / L / 2

snapshots_dir = f'/pscratch/sd/l/linyaoly/GFD_Polar_vortex/ddloutput/snapshots_F51_U_{U}_noNu_linear_4var_4tau_2taup'

Nphi, Nr = 64, 128
dealias = 3/2
stop_sim_time = 100
timestepper = d3.RK443
timestep =1e-3
# max_timestep = 1e-1
dtype = np.float64
nu = 1e-11
initv = 1e-1

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
# u1 = dist.VectorField(coords, name='u1', bases=disk)
# u2 = dist.VectorField(coords, name='u2', bases=disk)

tau1 = dist.Field(name='tau1', bases=edge)
tau2 = dist.Field(name='tau2', bases=edge)
# tau3 = dist.Field(name='tau3', bases=edge)
# tau4 = dist.Field(name='tau4', bases=edge)
taup = dist.Field(name='taup')
# taup2 = dist.Field(name='taup2')
# taus = [tau1, tau2, tau3, tau4, taup]
taus = [tau1, tau2, taup]

# Substitutions
phi, r = dist.local_grids(disk)
lift = lambda A,n: d3.Lift(A, disk, n)
# psi2u = lambda A: -d3.Skew(d3.Gradient(A))
r_field = dist.Field(bases=disk.radial_basis)
r_field['g'] = r  # radial coordinate

# Background
psi1_0 = 0.5 * (r_field**2)
psi2_0 = - 1.0 * psi1_0
Q1 = d3.Laplacian(psi1_0) - F1 * (psi1_0 - psi2_0) - 0.5 * gamma * (r_field**2)
Q2 = d3.Laplacian(psi2_0) + F2 * (psi1_0 - psi2_0) - 0.5 * gamma * (r_field**2)
U1 = -d3.Skew(d3.Gradient(psi1_0))  # (d/dr, -1/r d/dphi)
U2 = -d3.Skew(d3.Gradient(psi1_0))
t = dist.Field()

# Problem 
# q1 = d3.lap(psi1) - F1 * (psi1 - psi2) + taup + lift(tau1, -1)
# q2 = d3.lap(psi2) + F2 * (psi1 - psi2) + taup + lift(tau2, -1)
# u1 = -d3.Skew(d3.Gradient(psi1))
# u2 = -d3.Skew(d3.Gradient(psi2))

problem = d3.IVP([psi1, psi2, q1, q2] + taus, time=t, namespace=locals())
problem.add_equation("q1 - lap(psi1) + F1 * (psi1 - psi2) + taup + lift(tau1, -1) = 0")
problem.add_equation("q2 - lap(psi2) - F2 * (psi1 - psi2) + taup + lift(tau2, -1) = 0")
# problem.add_equation("u1 + skew(grad(psi1)) = 0")
# problem.add_equation("u2 + skew(grad(psi2)) = 0")
problem.add_equation(" dt(q1) " \
                        "+ U1 @ grad(q1)" \
                        "- skew(grad(psi1)) @ grad(Q1)" \
                        "= 0" )
problem.add_equation(" dt(q2) " \
                        "+ U2 @ grad(q2)" \
                        "- skew(grad(psi2)) @ grad(Q2)" \
                        "= 0" )
problem.add_equation("psi1(r=a_norm) = 0") # 7 is a/L
problem.add_equation("psi2(r=a_norm) = 0")
# problem.add_equation("q1(r=a_norm) = 0") # 7 is a/L
# problem.add_equation("q2(r=a_norm) = 0")
problem.add_equation("integ(psi1) = 0")


# Solver
solver = problem.build_solver(timestepper)
solver.print_subproblem_ranks(solver.subproblems, timestep)
solver.stop_sim_time = stop_sim_time

# Initial conditions
psi1.fill_random('g', seed=42, distribution='standard_normal') # Random noise
psi1['g'] *= initv
psi1.low_pass_filter(scales=0.25) # Keep only lower fourth of the modes

psi2.fill_random('g', seed=42, distribution='standard_normal') # Random noise
psi2['g'] *= initv
psi2.low_pass_filter(scales=0.25) # Keep only lower fourth of the modes

q1.fill_random('g', seed=42, distribution='standard_normal') # Random noise
q1['g'] *= initv
q1.low_pass_filter(scales=0.25) # Keep only lower fourth of the modes

q2.fill_random('g', seed=42, distribution='standard_normal') # Random noise
q2['g'] *= initv
q2.low_pass_filter(scales=0.25) # Keep only lower fourth of the modes

# u1.fill_random('g', seed=42, distribution='standard_normal') # Random noise
# u1['g'] *= initv
# u1.low_pass_filter(scales=0.25) # Keep only lower fourth of the modes

# u2.fill_random('g', seed=42, distribution='standard_normal') # Random noise
# u2['g'] *= initv
# u2.low_pass_filter(scales=0.25) # Keep only lower fourth of the modes

# Analysis
snapshots = solver.evaluator.add_file_handler(snapshots_dir, sim_dt=5, max_writes=500, mode='overwrite')
snapshots.add_task(psi1, scales=(16, 1), name='psi1')
snapshots.add_task(psi2, scales=(16, 1), name='psi2')


# # CFL
# CFL = d3.CFL(solver, initial_dt=max_timestep*1e-3, cadence=5, safety=0.1, threshold=0.02,
#              max_change=1.5, min_change=0.5, max_dt=max_timestep)
# CFL.add_velocity(u1)

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
        # Check for NaNs in psi1
        # print(solver.state[0])
        if solver.state[0]['g'].any() > 1e5:
            logger.error("NaN detected in psi1. Exiting loop.")
            break

        if (solver.iteration-1) % 1000 == 0:
            # max_u = np.sqrt(flow.max('u2'))
            max_psi1 = np.sqrt(flow.max('psi1'))
            logger.info("Iteration=%i, Time=%e, dt=%e, max(psi1)=%e" %(solver.iteration, solver.sim_time, timestep, max_psi1))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
