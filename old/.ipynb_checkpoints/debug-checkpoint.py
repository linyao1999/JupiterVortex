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
Nphi = 32
Nr = 64
dtype = np.complex128

coords = d3.PolarCoordinates('phi', 'r')
dist = d3.Distributor(coords, dtype=dtype)
disk = d3.DiskBasis(coords, shape=(Nphi, Nr), radius=a_norm, dtype=dtype)
phi, r = dist.local_grids(disk)

# r3 = dist.Field(bases=disk)
r3 = dist.Field(bases=disk.radial_basis)
r3['g'] = r**4  # radial coordinate
Psi1 = r3
Psi2 = - r3
gradPsi1 = d3.Gradient(Psi1)
skewgradPsi1 = d3.Skew(d3.Gradient(Psi1))
lapPsi1 = d3.Laplacian(Psi1)
x, y = coords.cartesian(phi, r)

hfile = h5py.File("debug.h5", "w")
tasks = hfile.create_group('tasks')
tasks.create_dataset('Psi1', data=Psi1['g'].real)
tasks.create_dataset('Psi2', data=Psi2['g'].real)
tasks.create_dataset('gradPsi1', data=gradPsi1['g'].real)
tasks.create_dataset('skewgradPsi1', data=skewgradPsi1['g'].real)
tasks.create_dataset('lapPsi1', data=lapPsi1['g'].real)
tasks.create_dataset('r', data = r)
tasks.create_dataset('phi', data = phi)
tasks.create_dataset('x', data = x)
tasks.create_dataset('y', data = y)


