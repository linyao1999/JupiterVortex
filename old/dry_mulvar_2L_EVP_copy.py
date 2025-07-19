"""
Dedalus script solving the linear stability eigenvalue problem for pipe flow.
This script demonstrates solving an eigenvalue problem in the periodic cylinder
using the disk basis and a parametrized axial wavenumber. It should take just
a few seconds to run (serial only).

The radius of the pipe is R = 1, and the problem is non-dimensionalized using
the radius and laminar velocity, such that the background flow is w0 = 1 - r**2.

No-slip boundary conditions are implemented on the velocity perturbations.
For incompressible hydro with one boundary, we need one tau term each for the
scalar axial velocity and vector horizontal (in-disk) velocity. Here we choose
to left the tau terms to the original (k=0) basis.

The eigenvalues are compared to the results of Vasil et al. (2016) [1] in Table 3.

To run, print, and plot the slowest decaying mode:
    $ python3 pipe_flow.py

References:
    [1]: G. M. Vasil, K. J. Burns, D. Lecoanet, S. Olver, B. P. Brown, J. S. Oishi,
         "Tensor calculus in polar coordinates using Jacobi polynomials," Journal
         of Computational Physics (2016).
"""


import numpy as np
import dedalus.public as d3
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger(__name__)


# Parameters
# physical parameters
L_scale = 6.99e7/2/np.pi  # 6.99e7 m is the radius of Jupiter, divided by 2π sets a natural length scale in units 2π.
Omega_scale = 2 * np.pi /(9.925*3600)/4/(np.pi)**2
f0 = 8 * np.pi 
gprime = 24 /(L_scale*Omega_scale**2)  # g0 = 24; g = g0/(L_scale*Omega_scale^2)     # gravity
H1 = 1e4 / L_scale
H2 = 1e4 / L_scale
gamma = 2.0
u0 = 50  / L_scale / Omega_scale
F1 = f0**2 / gprime / H1
F2 = f0**2 / gprime / H2

kz = 1.2 
m = 6
Nphi = 2 * m + 2
Nr = 64
dtype = np.complex128

# Bases
coords = d3.PolarCoordinates('phi', 'r')
dist = d3.Distributor(coords, dtype=dtype)
disk = d3.DiskBasis(coords, shape=(Nphi, Nr), radius=1, dtype=dtype)
phi, r = dist.local_grids(disk)

# Fields
s = dist.Field(name='s')
psi1 = dist.Field(name='psi1', bases=disk)
psi2 = dist.Field(name='psi2', bases=disk)
tau_psi1 = dist.Field(name='tau_psi1', bases=disk.edge)
tau_psi2 = dist.Field(name='tau_psi2', bases=disk.edge)

# Substitutions
dt = lambda A: s*A
lift_basis = disk.derivative_basis(2)
lift = lambda A: d3.Lift(A, lift_basis, -1)
# define a unit vector in the azimuthal direction
ephi = dist.VectorField(coords, name='ephi', bases=disk)
ephi['g'][0] = 0  
ephi['g'][1] = 1  
dphi = lambda A: d3.skew(d3.grad(A)) @ ephi  # azimuthal derivative dphi(A)
r_field = dist.Field(name='r_field', bases=disk)
r_field['g'] = r  # assign the grid values to the field

# Background
Qr1 = dist.Field(name='Qr1', bases=disk.radial_basis)
Qr2 = dist.Field(name='Qr2', bases=disk.radial_basis)
Qr1['g'] = - gamma * r - 2 * F1 * u0 * r
Qr2['g'] = - gamma * r + 2 * F2 * u0 * r

# Problem
problem = d3.EVP([psi1, psi2, tau_psi1, tau_psi2], eigenvalue=s, namespace=locals())
# q1 = lap(psi1) - F1 * (psi1 - psi2)
# q2 = lap(psi2) + F2 * (psi1 - psi2)
problem.add_equation("dt((lap(psi1))-F1*(psi1-psi2))+ u0 * r_field * ((lap(psi1))-F1*(psi1-psi2)) - Qr1 * (psi1) +lift(tau_psi1)=0")
problem.add_equation("dt((lap(psi2))+F2*(psi1-psi2))- u0 * r_field * ((lap(psi2))+F2*(psi1-psi2)) - Qr2 * (psi2) +lift(tau_psi2)=0")
problem.add_equation("psi1(r=1) = 0")
problem.add_equation("psi2(r=1) = 0")
problem.add_equation("psi1(r=0) = 0")
problem.add_equation("psi2(r=0) = 0")
# problem.add_equation("dphi(psi1)(r=1) = 0")
# problem.add_equation("dphi(psi2)(r=1) = 0")
# Solver
# # the Fourier (φ) basis is the first basis inside `disk`
# solver = problem.build_solver()
# # Create the subproblem for mode m
# solver.solve_dense(kphi=m)

# # Sort eigenvalues
# evals = solver.eigenvalues[np.isfinite(solver.eigenvalues)]
# evals = evals[np.argsort(-evals.real)]
# print(f"Slowest-decaying mode for m={m}: λ = {evals[0]}")

# # Set state to that eigenmode
# eig_index = np.argmin(np.abs(solver.eigenvalues - evals[0]))
# solver.set_state(eig_index, kphi=m)

solver = problem.build_solver()
sp = solver.subproblems_by_group[(m, None)]
# sp = solver.subproblems[0]
solver.solve_dense(sp)
# solver.solve_dense(solver.subproblems[0])
evals = solver.eigenvalues[np.isfinite(solver.eigenvalues)]
evals = evals[np.argsort(-evals.real)]
print(f"Slowest decaying mode: λ = {evals[0]}")
solver.set_state(np.argmin(np.abs(solver.eigenvalues - evals[0])), sp.subsystems[0])
# solver.set_state(np.argmin(np.abs(solver.eigenvalues - evals[0])), 0)

# Plot eigenfunction
scales = (32, 4)
# ω = d3.div(d3.skew(psi1)).evaluate()
# ω.change_scales(scales)
psi1.change_scales(scales)
psi2.change_scales(scales)
phi, r = dist.local_grids(disk, scales=scales)
x, y = coords.cartesian(phi, r)

cmap = 'RdBu_r'
fig, ax = plt.subplots(2, 2, figsize=(6, 6))
ax[0,0].pcolormesh(x, y, psi1['g'].real, cmap=cmap)
ax[0,0].set_title(r"$\psi_1$")
ax[0,1].pcolormesh(x, y, psi2['g'].real, cmap=cmap)
ax[0,1].set_title(r"$\psi_2$")

for axi in ax.flatten():
    axi.set_aspect('equal')
    axi.set_axis_off()
fig.tight_layout()
fig.savefig("pipe_eigenfunctions.png", dpi=200)

