import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

data = xr.open_dataset("state_vectors.nc")
data = data["__xarray_dataarray_variable__"]

pos = data.sel(component=["x", "y", "z"])
pos = pos.transpose("time", "object", "component")
pos_np = pos.values

T, N, _ = pos_np.shape

fig = plt.figure(facecolor="black")
plt.style.use('dark_background')
ax = fig.add_subplot(projection='3d')

COMET_IDX = -1

scat = ax.scatter(
    pos_np[0,:,0],
    pos_np[0,:,1],
    pos_np[0,:,2]
)
ax.set_box_aspect([1,1,1])
ax.set_position([0, 0, 1, 1])
ax.grid(False)
ax.set_axis_off()

labels = list(pos.coords["object"].values)

texts = [
    ax.text(
        pos_np[0,i,0],
        pos_np[0,i,1],
        pos_np[0,i,2],
        labels[i],
        fontsize=6,
    )

    for i in range(N)
]

def update(frame):
    coords = pos_np[frame]

    # --- update scatter ---
    scat._offsets3d = (
        coords[:,0],
        coords[:,1],
        coords[:,2]
    )

    # --- target position ---
    target = coords[COMET_IDX]

    # --- distances to others ---
    diff = coords - target
    dists = np.linalg.norm(diff, axis=1)

    # exclude self (distance = 0)
    dists[COMET_IDX] = np.inf

    nearest = np.min(dists)

    radius = 1.5 * nearest

    ax.set_xlim(target[0] - radius, target[0] + radius)
    ax.set_ylim(target[1] - radius, target[1] + radius)
    ax.set_zlim(target[2] - radius, target[2] + radius)

    for i, txt in enumerate(texts):
        txt.set_position((coords[i,0], coords[i,1]))
        txt.set_3d_properties(coords[i,2])

    return (scat, *texts)

ani = FuncAnimation(fig, update, frames=T, interval=30)

ani.save("solar_system.mp4", writer="ffmpeg", fps=30)