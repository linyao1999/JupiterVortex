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

import numpy as np
import dedalus.public as d3
from scipy.special import jv
import logging
logger = logging.getLogger(__name__)

# Parameters
# Physical paramters 
F1 = 1  # L**2/Ld1**2
F2 = 1 
a = 6.99e7
# T = 9.925*3600
L = 1e7
# gamma = 4 * np.pi / a / a / T * (L**3) / 100
gamma = 0.7198 # 2 omega / a**2 * L**3 / U

# kphi = 10
Nphi, Nr = 32, 128
# Ekman = 1 / 2 / 20**2
# Ro = 40
dealias = 3/2
stop_sim_time = 2
timestepper = d3.SBDF2
# timestep = 1e-3
timestep =1e-2
dtype = np.float64

# Bases
coords = d3.PolarCoordinates('phi', 'r')
dist = d3.Distributor(coords, dtype=dtype)
disk = d3.DiskBasis(coords, shape=(Nphi, Nr), radius=3.5, dealias=dealias, dtype=dtype)
edge = disk.edge

# Fields
# u = dist.VectorField(coords, name='u', bases=disk)
# p = dist.Field(name='p', bases=disk)
# tau_u = dist.VectorField(coords, name='tau_u', bases=edge)
# tau_p = dist.Field(name='tau_p')
psi1 = dist.Field(name='psi1', bases=disk)
psi2 = dist.Field(name='psi2', bases=disk)
tau_psi1 = dist.Field(name='tau_psi1', bases=edge)
tau_psi2 = dist.Field(name='tau_psi2', bases=edge)


# Substitutions
phi, r = dist.local_grids(disk)
# nu = Ekman
lift = lambda A: d3.Lift(A, disk, -1)
# lift_basis = disk.derivative_basis(2)
# lift = lambda A: d3.Lift(A, lift_basis, -1)
ephi = dist.VectorField(coords, name='ephi', bases=disk)
ephi['g'][0] = 1

r_field = dist.Field(bases=disk)
r_field['g'] = np.meshgrid(phi, r, indexing='ij')[1]  # radial coordinate
# dphi = lambda A: d3.grad(A)@ephi*r 
dphi = lambda A: r_field * d3.grad(A)@ephi

# # Background librating flow
# u0_real = dist.VectorField(coords, bases=disk)
# u0_imag = dist.VectorField(coords, bases=disk)
# u0_real['g'][0] = Ro * np.real(jv(1, (1-1j)*r/np.sqrt(2*Ekman)) / jv(1, (1-1j)/np.sqrt(2*Ekman)))
# u0_imag['g'][0] = Ro * np.imag(jv(1, (1-1j)*r/np.sqrt(2*Ekman)) / jv(1, (1-1j)/np.sqrt(2*Ekman)))
t = dist.Field()
# u0 = np.cos(t) * u0_real - np.sin(t) * u0_imag

# Problem
problem = d3.IVP([psi1, psi2, tau_psi1, tau_psi2], time=t, namespace=locals())
# problem.add_equation("div(u) + tau_p = 0")
# problem.add_equation("dt(u) - nu*lap(u) + grad(p) + lift(tau_u) = - u@grad(u0) - u0@grad(u)")
# problem.add_equation("u(r=1) = 0")
# problem.add_equation("integ(p) = 0")
# q1 = lap(psi1) - F1 * (psi1 - psi2) + tau_psi1
# q2 = lap(psi2) + F2 * (psi1 - psi2) + tau_psi2
problem.add_equation("dt(lap(psi1) - F1 * (psi1 - psi2)) + dphi(lap(psi1) - F1 * (psi1 - psi2)) + (gamma + 2 * F1) * dphi(psi1) + lift(tau_psi1) = 0")
problem.add_equation("dt(lap(psi2) + F2 * (psi1 - psi2)) + dphi(lap(psi2) + F2 * (psi1 - psi2)) + (gamma - 2 * F2) * dphi(psi2) + lift(tau_psi2) = 0")
problem.add_equation("psi1(r=3.5) = 0") # 7 is a/L
problem.add_equation("psi2(r=3.5) = 0")
# problem.add_equation("integ(psi1) = 0")
# problem.add_equation("integ(psi2) = 0")

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
# u.fill_random('g', seed=42, distribution='standard_normal') # Random noise
# u.low_pass_filter(scales=0.25) # Keep only lower fourth of the modes
psi1.fill_random('g', seed=42, distribution='standard_normal') # Random noise
psi1.low_pass_filter(scales=0.25) # Keep only lower fourth of the modes
psi2.fill_random('g', seed=42, distribution='standard_normal') # Random noise
psi2.low_pass_filter(scales=0.25) # Keep only lower fourth of the modes

# Analysis
# snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.1, max_writes=20)
# snapshots.add_task(-d3.div(d3.skew(u)), scales=(4, 1), name='vorticity')
# scalars = solver.evaluator.add_file_handler('scalars', sim_dt=0.01)
# scalars.add_task(d3.integ(0.5*u@u), name='KE')
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.1, max_writes=20)
snapshots.add_task(psi1, scales=(4, 1), name='psi1')
# scalars = solver.evaluator.add_file_handler('scalars', sim_dt=0.01)
# scalars.add_task(d3.integ(0.5*u@u), name='KE')

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=100)
# flow.add_property(u@u, name='u2')
flow.add_property(d3.Laplacian(psi1) - F1 * (psi1 - psi2), name='q1')  # could be a vector; to be check 

# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
        solver.step(timestep)
        if (solver.iteration-1) % 100 == 0:
            # max_u = np.sqrt(flow.max('u2'))
            max_q1 = np.sqrt(flow.max('q1'))
            logger.info("Iteration=%i, Time=%e, dt=%e, max(q1)=%e" %(solver.iteration, solver.sim_time, timestep, max_q1))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
