import numpy as np 
import dedalus.public as d3
import h5py 

# Parameters
# Physical paramters 
F1 = 51.8  # L**2/Ld1**2
F2 = 51.8 
a = 6.99e7
T = 9.925*3600
L = 1e7
U = 100
gamma = 4 * np.pi / T / a / a * (L**3) / U
a_norm = a / L / 2

# numerical parameters
m = 10
Nphi = 2 * m + 2
Nr = 64
dtype = np.complex128

coords = d3.PolarCoordinates('phi', 'r')
dist = d3.Distributor(coords, dtype=dtype)
disk = d3.DiskBasis(coords, shape=(Nphi, Nr), radius=a_norm, dtype=dtype)
phi, r = dist.local_grids(disk)

r2 = dist.Field(bases=disk.radial_basis)
r2['g'] = r**2  # radial coordinate
Psi1 = 0.5 * (r2)
Psi2 = - 0.5 * (r2)
Q1 = d3.Laplacian(Psi1) - F1 * (Psi1 - Psi2) - 0.5 * gamma * r2
Q2 = d3.Laplacian(Psi2) + F2 * (Psi1 - Psi2) - 0.5 * gamma * r2
U1 = d3.Skew(d3.Gradient(Psi1))
U2 = d3.Skew(d3.Gradient(Psi2))

x, y = coords.cartesian(phi, r)

hfile = h5py.File("debug.h5", "w")
tasks = hfile.create_group('tasks')
tasks.create_dataset('Psi1', data=Psi1['g'].real)
tasks.create_dataset('Psi2', data=Psi2['g'].real)
tasks.create_dataset('U1', data=U1['g'].real)
tasks.create_dataset('U2', data=U2['g'].real)
tasks.create_dataset('Q1', data=Q1['g'].real)
tasks.create_dataset('Q2', data=Q2['g'].real)
tasks.create_dataset('r', data = r)








