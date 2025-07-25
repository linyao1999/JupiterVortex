import numpy as np 
import xarray as xr 
import matplotlib.pyplot as plt 

import h5py
F1 = 51.8
U = 100
s = 1
st = -1
prob_class = 'IVP'
output_dir = f'/net/fs06/d0/linyao/GFD_Polar_vortex/ddloutput/{prob_class}/'
snapshots_name = f'snapshots_F{int(np.floor(F1))}_U_{U}_linear_noNu'
snapshots_file = output_dir + snapshots_name + '/' + snapshots_name
snapshots_file = '/home/linyao/fs06/GFD_Polar_vortex/ddloutput/IVP/snapshots_F51_U_100_linear_noNu_initEVP/snapshots_F51_U_100_linear_noNu_initEVP'

with h5py.File(f'{snapshots_file}_s{s}.h5', "r") as f:
    print("Top-level keys:", list(f.keys()))        # likely includes 'scales' and 'tasks'
    print("Task variables:", list(f["tasks"].keys()))  # e.g., ['psi1', 'psi2']
    print("Task variables:", list(f["scales"].keys()))  # e.g., ['psi1', 'psi2']

    psi1 = f["tasks/psi1"][:]   # read full psi1 data
    print(psi1.shape)           # e.g., (time, x, y) or (time, r, phi)
    # tau1 = f["tasks/tau1"][:]
    # psi2 = f["tasks/psi2"][:]
    # q1 = f["tasks/q1"][:]
    # q2 = f["tasks/q2"][:]
    dset = f['tasks']['psi1']
    phi = np.reshape(dset.dims[1][0][:].ravel(), (-1, 1))
    r = np.reshape(dset.dims[2][0][:].ravel(), (1,-1))

x = r * np.cos(phi)
y = r * np.sin(phi)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter

# Assume psi1.shape = (nt, ny, nx)
# psi1 = psi1[:-1,:,:]
nt = psi1.shape[0]

fig = plt.figure(figsize=(7, 5))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.05], wspace=0.1)
ax = fig.add_subplot(gs[0, 0])
ax.set_aspect('equal')
ax.set_adjustable('box')

# Initialize first frame
levels = 21
contour = ax.contourf(x, y, psi1[0], levels=levels, cmap='RdBu_r')
# cbar = fig.colorbar(contour, cax=fig.add_subplot(gs[0, 1]))

# Store current contour so we can remove it
def update(frame):
    global contour
    # Remove previous contour plots
    for c in contour.collections:
        c.remove()
    # Plot new frame
    contour = ax.contourf(x, y, psi1[frame], levels=levels, cmap='RdBu_r')
    # plt.colorbar(contour)
    ax.set_title(f'Time step {frame}')
    return contour.collections

ani = animation.FuncAnimation(fig, update, frames=nt, interval=100, blit=False)

from matplotlib.animation import PillowWriter
writer = PillowWriter(fps=10)
ani.save('out.gif', writer=writer)