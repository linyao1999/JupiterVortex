import numpy as np
import dedalus.public as d3
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger(__name__)

# Parameters
# Physical paramters 
F1 = 5.18  # L**2/Ld1**2
F2 = 5.18 
a = 6.99e7
T = 9.925*3600
L = 1e7
U = 50
gamma = 4 * np.pi / T / a / a * (L**3) / U
# gamma = 0.7198 # 2 omega / a**2 * L**3 / U
a_norm = a / L

# numerical parameters
kphi = 2
m = 10
Nphi = 2 * m + 2
Nr = 64
dtype = np.complex128

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
dt = lambda A: s*A
dphi = lambda A: 1j*kphi*A 
lift_basis = disk.derivative_basis(2)
lift = lambda A: d3.Lift(A, lift_basis, -1)

# Problem
problem = d3.EVP([psi1, psi2, tau_psi1, tau_psi2], eigenvalue=s, namespace=locals())
problem.add_equation("dt(lap(psi1) - F1 * (psi1 - psi2)) + dphi(lap(psi1) - F1 * (psi1 - psi2)) + (gamma + 2 * F1) * dphi(psi1) + lift(tau_psi1) = 0")
problem.add_equation("dt(lap(psi2) + F2 * (psi1 - psi2)) + dphi(lap(psi2) + F2 * (psi1 - psi2)) + (gamma - 2 * F2) * dphi(psi2) + lift(tau_psi2) = 0")
problem.add_equation("psi1(r=a_norm) = 0") # 7 is a/L
problem.add_equation("psi2(r=a_norm) = 0")
# problem.add_equation("integ(psi1) = 0")
# problem.add_equation("integ(psi2) = 0")

# Solver

solver = problem.build_solver()

for m in range(20):
    # sp = solver.subproblems[]
    sp = solver.subproblems_by_group[(m, None)]
    solver.solve_dense(sp)
    evals = solver.eigenvalues[np.isfinite(solver.eigenvalues)]
    evals = evals[np.argsort(-evals.real)]
    print(f"Slowest decaying mode: λ = {evals[0]}")
    solver.set_state(np.argmin(np.abs(solver.eigenvalues - evals[0])), sp.subsystems[0])


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
    # ax[0,0].plot(psi1['g'].real)
    # ax[0,0].set_title(r"$\psi_1$")
    # ax[0,1].plot(psi2['g'].real)
    # ax[0,1].set_title(r"$\psi_2$")

    # print(psi1['g'].real[:10] * L_scale)

    for axi in ax.flatten():
        axi.set_aspect('equal')
        axi.set_axis_off()
    fig.tight_layout()
    fig.savefig(f'pipe_eigenfunctions.png', dpi=200)
