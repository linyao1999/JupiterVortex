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

evp_dir = f'/net/fs06/d0/linyao/GFD_Polar_vortex/ddloutput/EVP/'
os.makedirs(evp_dir, exist_ok=True)

# numerical parameters
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
psi2u = lambda A: -d3.Skew(d3.Gradient(A))
lift_basis = disk.derivative_basis(2)
lift = lambda A: d3.Lift(A, lift_basis, -1)
q1 = d3.lap(psi1) - F1 * (psi1 - psi2)
q2 = d3.lap(psi2) + F2 * (psi1 - psi2)

# Background 
r_field = dist.Field(bases=disk.radial_basis)
r_field['g'] = r  # radial coordinate
psi1_0 = 0.5 * (r_field**2)
psi2_0 = - 0.5 * (r_field**2)
Q1 = d3.Laplacian(psi1_0) - F1 * (psi1_0 - psi2_0) - 0.5 * gamma * (r_field**2)
Q2 = d3.Laplacian(psi2_0) + F2 * (psi1_0 - psi2_0) - 0.5 * gamma * (r_field**2)

# Problem
problem = d3.EVP([psi1, psi2, tau_psi1, tau_psi2], eigenvalue=s, namespace=locals())
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
solver = problem.build_solver()

for kphi in range(1,7):
    sp = solver.subproblems_by_group[(kphi, None)]
    solver.solve_dense(sp)
    evals = solver.eigenvalues[np.isfinite(solver.eigenvalues)]
    evals = evals[np.argsort(-evals.real)]
    print(f"Slowest decaying mode: λ = {evals[0]}")
    solver.set_state(np.argmin(np.abs(solver.eigenvalues - evals[0])), sp.subsystems[0])

    # # Plot eigenfunction
    scales = (32, 4)
    psi1.change_scales(scales)
    psi2.change_scales(scales)
    phi, r = dist.local_grids(disk, scales=scales)
    x, y = coords.cartesian(phi, r)

    print(psi1['g'].real.shape)

    cmap = 'RdBu_r'
    fig, ax = plt.subplots(1, 2, figsize=(6, 6))
    ax[0].pcolormesh(x, y, psi1['g'].real, cmap=cmap)
    ax[0].set_title(r"$\psi_1$")
    ax[1].pcolormesh(x, y, psi2['g'].real, cmap=cmap)
    ax[1].set_title(r"$\psi_2$")

    for axi in ax.flatten():
        axi.set_aspect('equal')
        axi.set_axis_off()
    fig.tight_layout()
    fig.savefig(f'./plots/EVP_dry_phi_m{kphi}_F{F1}_U{U}.png', dpi=200)

    hfile = h5py.File(f'{evp_dir}EVP_dry_phi_m{kphi}_F{F1}_U{U}.h5', 'w')
    tasks = hfile.create_group('tasks')
    tasks.create_dataset('psi1', data=psi1['g'].real)
    tasks.create_dataset('psi2', data=psi2['g'].real)
    tasks.create_dataset('x', data=x)
    tasks.create_dataset('y', data=y)
    tasks.create_dataset('phi', data=phi[:])
    tasks.create_dataset('r', data=r[:])    
    # # plot PV
    # scales = (32, 4)
    # q1.change_scales(scales)
    # q2.change_scales(scales)
    # phi, r = dist.local_grids(disk, scales=scales)
    # x, y = coords.cartesian(phi, r)

    # cmap = 'RdBu_r'
    # fig, ax = plt.subplots(1, 2, figsize=(6, 6))
    # ax[0].pcolormesh(x, y, q1['g'].real, cmap=cmap)
    # ax[0].set_title(r"$q_1$")
    # ax[1].pcolormesh(x, y, q2['g'].real, cmap=cmap)
    # ax[1].set_title(r"$q_2$")

    # for axi in ax.flatten():
    #     axi.set_aspect('equal')
    #     axi.set_axis_off()
    # fig.tight_layout()
    # fig.savefig(f'q_m{kphi}.png', dpi=200)