import numpy as np 
import dedalus.public as d3
import h5py 
import matplotlib.pyplot as plt
import logging

# --------- CHOOSE THE PROBLEM -----------
prob_class = 'EVP'
# prob_class = 'IVP'

# ----------- Physical parameters -----------------
# ======= fixed ==========
L = 1e7 # horizontal length scale; appro radius of polar vortex
T = 9.925*3600  # period of Jupiter rotation
a = 6.99e7 
# ======= derived =========
omega = 2 * np.pi / T  # rotation speed
gamma = 2 * omega / a / a # dimensional gamma
a_norm = a / L / 2.0   # dimensionless disk radius
# ======= defined/changed ==========
# F1 = 5.18   # L**2/Ld1**2; can vary between 0.1 and 100
F1 = 51.8   # L**2/Ld1**2; can vary between 0.1 and 100
delta = 1.0 # H2/H1
F2 = delta**2 * F1 # L**2/Ld1**2;
U = 100
Gamma = gamma * (L**3) / U

# ------------- Numerical parameters --------------------
Nphi = 64
Nr = 128
dtype = np.complex128
max_kphi = 10
output_dir = f'/net/fs06/d0/linyao/GFD_Polar_vortex/ddloutput/{prob_class}/'

# --------------- Bases ------------------------
coords = d3.PolarCoordinates('phi', 'r')
dist = d3.Distributor(coords, dtype=dtype)
disk = d3.DiskBasis(coords, shape=(Nphi, Nr), radius=a_norm, dtype=dtype)
phi, r = dist.local_grids(disk)

# --------------- Fields -----------------------
s = dist.Field(name='s')
psi1 = dist.Field(name='psi1', bases=disk)
psi2 = dist.Field(name='psi2', bases=disk)
tau1 = dist.Field(name='tau1', bases=disk.edge)
tau2 = dist.Field(name='tau2', bases=disk.edge)

# ----------------- Substitutions ----------------------
dt = lambda A: s*A
# psi2u = lambda A: d3.Skew(d3.Gradient(A))
lift_basis = disk.derivative_basis(2)
lift = lambda A: d3.Lift(A, lift_basis, -1)
q1 = - d3.lap(psi1) - F1 * (psi1 - psi2)
q2 = - d3.lap(psi2) + F2 * (psi1 - psi2)
u1 = - d3.Skew(d3.Gradient(psi1))
u2 = - d3.Skew(d3.Gradient(psi2))

# ------------------ Background ------------------------
r2 = dist.Field(bases=disk.radial_basis)
r2['g'] = 0.5 * (r**2)
Psi1 = r2 
Psi2 = - r2
U1 = - d3.Skew(d3.Gradient(Psi1))
U2 = - d3.Skew(d3.Gradient(Psi2))
Q1 = - d3.lap(Psi1) - F1 * (Psi1 - Psi2) - Gamma * r2
Q2 = - d3.lap(Psi2) + F2 * (Psi1 - Psi2) - Gamma * r2

# ------------------ Problem ------------------------------
if prob_class == 'EVP':
    print(prob_class)
    problem = d3.EVP([psi1, psi2, tau1, tau2], eigenvalue=s, namespace=locals())
    problem.add_equation(" dt(q1) + u1 @ grad(Q1) + U1 @ grad(q1) + lift(tau1) = 0 ")
    problem.add_equation(" dt(q2) + u2 @ grad(Q2) + U2 @ grad(q2) + lift(tau2) = 0 ")
    problem.add_equation("psi1(r=a_norm) = 0")
    problem.add_equation("psi2(r=a_norm) = 0")
else:
    print(prob_class)

solver = problem.build_solver()

if prob_class == 'EVP':
    for kphi in range(1, max_kphi+1):
        sp = solver.subproblems_by_group[(kphi, None)]
        solver.solve_dense(sp)
        evals = solver.eigenvalues[np.isfinite(solver.eigenvalues)]
        evals = evals[np.argsort(-evals.real)]
        print(f"m={kphi}; slowest decaying mode: λ = {evals[0]}")
        solver.set_state(np.argmin(np.abs(solver.eigenvalues - evals[0])), sp.subsystems[0]) 

        # hfile = h5py.File(f'{output_dir}{prob_class}_dry_2L_linearEQ_noNu_F{F1}_U{U}_m{kphi}.h5', 'w')
        # tasks = hfile.create_group('tasks')
        # tasks.create_dataset('psi1', data=psi1['g'].real)
        # tasks.create_dataset('psi2', data=psi2['g'].real)
        # tasks.create_dataset('phi', data=phi)
        # tasks.create_dataset('r', data=r)
        
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
        fig.savefig(f'./plots/EVP_dry_phi_F{F1}_U{U}_m{kphi}.png', dpi=200)

    phi, r = dist.local_grids(disk)
    hfile = h5py.File(f'{output_dir}{prob_class}_dry_2L_linearEQ_noNu_F{F1}_U{U}_background.h5', 'w')
    tasks = hfile.create_group('tasks')
    tasks.create_dataset('Psi1', data=Psi1['g'].real)
    tasks.create_dataset('Psi2', data=Psi2['g'].real)
    tasks.create_dataset('U1', data=U1['g'].real)
    tasks.create_dataset('U2', data=U2['g'].real)
    tasks.create_dataset('Q1', data=Q1['g'].real)
    tasks.create_dataset('Q2', data=Q2['g'].real)
    tasks.create_dataset('phi', data=phi)
    tasks.create_dataset('r', data=r)


