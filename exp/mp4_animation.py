import matplotlib
matplotlib.use('Agg')

import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter

# Load data
F1 = 51.8
U = 100
s = 1
prob_class = 'IVP'
output_dir = f'/net/fs06/d0/linyao/GFD_Polar_vortex/ddloutput/{prob_class}/'
snapshots_name = f'snapshots_F{int(np.floor(F1))}_U_{U}_linear_noNu'
snapshots_file = output_dir + snapshots_name + '/' + snapshots_name

with h5py.File(f'{snapshots_file}_s{s}.h5', "r") as f:
    psi1 = f["tasks/psi1"][:]
    dset = f['tasks']['psi1']
    phi = np.reshape(dset.dims[1][0][:].ravel(), (-1, 1))
    r = np.reshape(dset.dims[2][0][:].ravel(), (1, -1))

x = r * np.cos(phi)
y = r * np.sin(phi)
psi1 = psi1[:200, :, :]
nt = psi1.shape[0]

# Set up figure
fig = plt.figure(figsize=(7, 5))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.05], wspace=0.1)
ax = fig.add_subplot(gs[0, 0])
ax.set_aspect('equal')
ax.set_adjustable('box')
levels = 21

def update(frame):
    ax.clear()
    ax.set_aspect('equal')
    ax.set_adjustable('box')
    ax.contourf(x, y, psi1[frame], levels=levels, cmap='RdBu_r')
    ax.set_title(f'Time step {frame}')
    return []

# Animate and save
writer = FFMpegWriter(fps=10, metadata=dict(artist='Lin Yao'), bitrate=1800)
ani = animation.FuncAnimation(fig, update, frames=nt, interval=100, blit=False)
ani.save(f'JupiterVortex/plots/{snapshots_name}.mp4', writer=writer)
