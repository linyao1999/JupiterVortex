import numpy as np
import dedalus.public as d3
import matplotlib.pyplot as plt
import logging
import os
import h5py
logger = logging.getLogger(__name__)
loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger in loggers:
    logger.setLevel(logging.WARNING)

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

# evp_dir = f'/net/fs06/d0/linyao/GFD_Polar_vortex/ddloutput/EVP/'
# os.makedirs(evp_dir, exist_ok=True)

# numerical parameters
m_max = 10
# Nphi = 2 * (m_max + 1)
# Nphi = 4 * m_max
Nphi = 144
Nr = 64
dtype = np.complex128

# Bases
coords = d3.PolarCoordinates('phi', 'r')
dist = d3.Distributor(coords, dtype=dtype)
disk = d3.DiskBasis(coords, shape=(Nphi, Nr), radius=R, dtype=dtype)
phi, r = dist.local_grids(disk)

# Fields
s = dist.Field(name='s')
psi1 = dist.Field(name='psi1', bases=disk)
psi2 = dist.Field(name='psi2', bases=disk)
tau_psi1 = dist.Field(name='tau_psi1', bases=disk.edge)
tau_psi2 = dist.Field(name='tau_psi2', bases=disk.edge)

# Substitutions
dt = lambda A: s*A
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
problem = d3.EVP([psi1, psi2, tau_psi1, tau_psi2], eigenvalue=s, namespace=locals())
problem.add_equation("dt(q1) + u1@grad(Q1) + U1@grad(q1) + lift(tau_psi1) = 0")
problem.add_equation("dt(q2) + u2@grad(Q2) + U2@grad(q2) + lift(tau_psi2) = 0")
problem.add_equation("psi1(r=R) = 0")
problem.add_equation("psi2(r=R) = 0")

# Solver
solver = problem.build_solver(ncc_cutoff=1e-6, entry_cutoff=1e-6)

m_range = np.arange(1, m_max+1)
evals_list = []
for m in m_range:
    # Solve
    sp = solver.subproblems_by_group[(m, None)]
    solver.solve_dense(sp)
    evals = solver.eigenvalues[np.isfinite(solver.eigenvalues)]
    evals = evals[np.argsort(-evals.real)]
    evals_list.append(evals[0])
    print(f"m={m}, λ_max={evals[0]}")
    solver.set_state(np.argmin(np.abs(solver.eigenvalues - evals[0])), sp.subsystems[0])

    # # Plot eigenfunction
    scales = (1, 1)
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
    fig.savefig(f'EVP_dry_phi_m{m}_F{F}_U{U}.png', dpi=200)

    # hfile = h5py.File(f'{evp_dir}EVP_dry_phi_m{kphi}_F{F}_U{U}.h5', 'w')
    # tasks = hfile.create_group('tasks')
    # tasks.create_dataset('psi1', data=psi1['g'].real)
    # tasks.create_dataset('psi2', data=psi2['g'].real)
    # tasks.create_dataset('x', data=x)
    # tasks.create_dataset('y', data=y)
    # tasks.create_dataset('phi', data=phi[:])
    # tasks.create_dataset('r', data=r[:])

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

# Plot dispersion relation
from scipy.special import jn_zeros
k = np.array([jn_zeros(mi,1)[0] / R for mi in m_range])
a1 = k**4 + 2*F*k**2
a2 = gamma*(F + k**2)
a3 = k**8 + F**2*(gamma**2 - 4*k**4)
c_mathematica = (a2 + np.sqrt(a3+0j)) / a1
s_mathematica = -1j * m_range * c_mathematica
s_dedalus = np.array(evals_list)

plt.figure(figsize=(6,4))
plt.plot(m_range, s_dedalus.real, '.', color='C0', label='Re(s) (Dedalus)')
# plt.plot(m_range, 10*s_dedalus.imag, '.', color='C1', label='10*Im(s) (Dedalus)')
plt.plot(m_range, s_mathematica.real, '-', color='C0', label='Re(s) (Mathematica)')
# plt.plot(m_range, 10*s_mathematica.imag, '-', color='C1', label='10*Im(s) (Mathematica)')
plt.legend()
plt.xlabel('m')
plt.ylabel('s')
plt.tight_layout()
plt.savefig('dispersion_relation.pdf')