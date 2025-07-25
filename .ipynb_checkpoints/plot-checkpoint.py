import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter

import h5py
import numpy as np 
import xarray as xr 
import matplotlib.pyplot as plt 

F1 = 51.8
U = 100
s = 1
st = -1
prob_class = 'IVP'
output_dir = f'/net/fs06/d0/linyao/GFD_Polar_vortex/ddloutput/{prob_class}/'
snapshots_name = f'snapshots_F51_U_{U}_linear_noNu'

snapshots_file = output_dir + snapshots_name + '/' + snapshots_name

with h5py.File(f'{snapshots_file}_s{s}.h5', "r") as f:
    print("Top-level keys:", list(f.keys()))        # likely includes 'scales' and 'tasks'
    print("Task variables:", list(f["tasks"].keys()))  # e.g., ['psi1', 'psi2']
    print("Task variables:", list(f["scales"].keys()))  # e.g., ['psi1', 'psi2']

    psi1 = f["tasks/psi1"][:]   # read full psi1 data
    print(psi1.shape)           # e.g., (time, x, y) 
    dset = f['tasks']['psi1']
    phi = np.reshape(dset.dims[1][0][:].ravel(), (-1, 1))
    r = np.reshape(dset.dims[2][0][:].ravel(), (1,-1))

psi1 = psi1[:6,:,:]

x = r * np.cos(phi)
y = r * np.sin(phi)

nt = psi1.shape[0]

fig = plt.figure(figsize=(7, 5))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.05], wspace=0.1)
ax = fig.add_subplot(gs[0, 0])
ax.set_aspect('equal')
ax.set_adjustable('box')

# Initialize first frame
levels = np.linspace(np.min(psi1), np.max(psi1), 21)
contour = ax.contourf(x, y, psi1[0], levels=levels, cmap='RdBu_r')
cbar = fig.colorbar(contour, cax=fig.add_subplot(gs[0, 1]))

# Store current contour so we can remove it
def update(frame):
    global contour
    # Remove previous contour plots
    for c in contour.collections:
        c.remove()
    # Plot new frame
    contour = ax.contourf(x, y, psi1[frame], levels=levels, cmap='RdBu_r')
    ax.set_title(f'Time step {frame}')
    return contour.collections

ani = animation.FuncAnimation(fig, update, frames=nt, interval=100, blit=False)

# Save using FFMpegWriter
writer = FFMpegWriter(fps=10, metadata=dict(artist='Lin'), bitrate=1800)
ani.save('psi1_evolution.mp4', writer=writer, dpi=150)
