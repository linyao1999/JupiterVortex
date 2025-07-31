import numpy as np 
import xarray as xr 
import matplotlib.pyplot as plt 
import os 
import glob
import h5py
snapshots_path = './snapshots'
snapshots_prefix = os.path.join(snapshots_path, 'snapshots')
files = sorted(glob.glob(f"{snapshots_prefix}_s*.h5"))
psi1_list = []
psi2_list = []
q1_list = []
q2_list = []

for fname in files:
    with h5py.File(fname, 'r') as f:
        psi1 = f["tasks/psi1"][:]
        psi2 = f["tasks/psi2"][:]
        q1 = f["tasks/q1"][:]
        q2 = f["tasks/q2"][:] 

        dset = f['tasks']['psi1']
        phi = np.reshape(dset.dims[1][0][:].ravel(), (-1, 1))
        r = np.reshape(dset.dims[2][0][:].ravel(), (1,-1))

        psi1_list.append(np.copy(psi1))
        psi2_list.append(np.copy(psi2))
        q1_list.append(np.copy(q1))
        q2_list.append(np.copy(q2))

psi1 = np.concatenate(psi1_list, axis=0)
psi2 = np.concatenate(psi2_list, axis=0)
q1 = np.concatenate(q1_list, axis=0)
q2 = np.concatenate(q2_list, axis=0)
print(psi1.shape)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

x = r * np.cos(phi)
y = r * np.sin(phi)

# Assume psi1.shape = psi2.shape = (nt, ny, nx)
psi1 = psi1[:1000, :, :]
psi2 = psi2[:1000, :, :]
nt = psi1.shape[0]

fig = plt.figure(figsize=(12, 5))
gs = fig.add_gridspec(1, 4, width_ratios=[1, 0.05, 1, 0.05], wspace=0.3)

# Axes for psi1 and its colorbar
ax1 = fig.add_subplot(gs[0, 0])
cax1 = fig.add_subplot(gs[0, 1])
ax1.set_aspect('equal')
ax1.set_adjustable('box')

# Axes for psi2 and its colorbar
ax2 = fig.add_subplot(gs[0, 2])
cax2 = fig.add_subplot(gs[0, 3])
ax2.set_aspect('equal')
ax2.set_adjustable('box')

# First frame setup
vmin1, vmax1 = psi1[0].min(), psi1[0].max()
vmin2, vmax2 = psi2[0].min(), psi2[0].max()
levels1 = np.linspace(vmin1, vmax1, 21)
levels2 = np.linspace(vmin2, vmax2, 21)

contour1 = ax1.contourf(x, y, psi1[0], levels=levels1, cmap='RdBu_r')
cbar1 = fig.colorbar(contour1, cax=cax1)

contour2 = ax2.contourf(x, y, psi2[0], levels=levels2, cmap='RdBu_r')
cbar2 = fig.colorbar(contour2, cax=cax2)

def update(frame):
    global contour1, cbar1, contour2, cbar2

    # Clear previous contours
    for c in contour1.collections:
        c.remove()
    for c in contour2.collections:
        c.remove()

    # Clear colorbar axes
    cax1.cla()
    cax2.cla()

    # Update color levels dynamically
    vmin1, vmax1 = psi1[frame].min(), psi1[frame].max()
    vmin2, vmax2 = psi2[frame].min(), psi2[frame].max()
    levels1 = np.linspace(vmin1, vmax1, 21)
    levels2 = np.linspace(vmin2, vmax2, 21)

    # Plot new contours
    contour1 = ax1.contourf(x, y, psi1[frame], levels=levels1, cmap='RdBu_r')
    contour2 = ax2.contourf(x, y, psi2[frame], levels=levels2, cmap='RdBu_r')

    # Add new colorbars
    cbar1 = fig.colorbar(contour1, cax=cax1)
    cbar2 = fig.colorbar(contour2, cax=cax2)

    ax1.set_title(f'psi1 — time {frame}')
    ax2.set_title(f'psi2 — time {frame}')

    return contour1.collections + contour2.collections

ani = animation.FuncAnimation(fig, update, frames=nt, interval=100, blit=False)

from matplotlib.animation import FFMpegWriter

# Save as MP4 using FFMpegWriter
writer = FFMpegWriter(fps=10, metadata=dict(artist='Lin Yao'), bitrate=1800)
ani.save('psi1_psi2.mp4', writer=writer)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

x = r * np.cos(phi)
y = r * np.sin(phi)

# Assume psi1.shape = psi2.shape = (nt, ny, nx)
q1 = q1[:1000, :, :]
q2 = q2[:1000, :, :]
nt = q1.shape[0]

fig = plt.figure(figsize=(12, 5))
gs = fig.add_gridspec(1, 4, width_ratios=[1, 0.05, 1, 0.05], wspace=0.3)

# Axes for psi1 and its colorbar
ax1 = fig.add_subplot(gs[0, 0])
cax1 = fig.add_subplot(gs[0, 1])
ax1.set_aspect('equal')
ax1.set_adjustable('box')

# Axes for psi2 and its colorbar
ax2 = fig.add_subplot(gs[0, 2])
cax2 = fig.add_subplot(gs[0, 3])
ax2.set_aspect('equal')
ax2.set_adjustable('box')

# First frame setup
vmin1, vmax1 = q1[0].min(), q1[0].max()
vmin2, vmax2 = q2[0].min(), q2[0].max()
levels1 = np.linspace(vmin1, vmax1, 21)
levels2 = np.linspace(vmin2, vmax2, 21)

contour1 = ax1.contourf(x, y, q1[0], levels=levels1, cmap='RdBu_r')
cbar1 = fig.colorbar(contour1, cax=cax1)

contour2 = ax2.contourf(x, y, q2[0], levels=levels2, cmap='RdBu_r')
cbar2 = fig.colorbar(contour2, cax=cax2)

def update(frame):
    global contour1, cbar1, contour2, cbar2

    # Clear previous contours
    for c in contour1.collections:
        c.remove()
    for c in contour2.collections:
        c.remove()

    # Clear colorbar axes
    cax1.cla()
    cax2.cla()

    # Update color levels dynamically
    vmin1, vmax1 = q1[frame].min(), q1[frame].max()
    vmin2, vmax2 = q2[frame].min(), q2[frame].max()
    levels1 = np.linspace(vmin1, vmax1, 21)
    levels2 = np.linspace(vmin2, vmax2, 21)

    # Plot new contours
    contour1 = ax1.contourf(x, y, q1[frame], levels=levels1, cmap='RdBu_r')
    contour2 = ax2.contourf(x, y, q2[frame], levels=levels2, cmap='RdBu_r')

    # Add new colorbars
    cbar1 = fig.colorbar(contour1, cax=cax1)
    cbar2 = fig.colorbar(contour2, cax=cax2)

    ax1.set_title(f'q1 — time {frame}')
    ax2.set_title(f'q2 — time {frame}')

    return contour1.collections + contour2.collections

ani = animation.FuncAnimation(fig, update, frames=nt, interval=100, blit=False)

from matplotlib.animation import FFMpegWriter

# Save as MP4 using FFMpegWriter
writer = FFMpegWriter(fps=10, metadata=dict(artist='Lin Yao'), bitrate=1800)
ani.save('q1_q2.mp4', writer=writer)
