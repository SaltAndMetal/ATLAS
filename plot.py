import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Extract the time-series data
data= xr.open_dataset("state_vectors.nc")
data = data["__xarray_dataarray_variable__"]
pos = data.sel(component=["x", "y", "z"])
pos = pos.transpose("time", "object", "component")
pos_np = pos.values

T, N, _ = pos_np.shape

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# initial frame
scat = ax.scatter(
    pos_np[0,:,0],
    pos_np[0,:,1],
    pos_np[0,:,2]
)

# fix bounds of the axis
ax.set_xlim(pos_np[:,:,0].min(), pos_np[:,:,0].max())
ax.set_ylim(pos_np[:,:,1].min(), pos_np[:,:,1].max())
ax.set_zlim(pos_np[:,:,2].min(), pos_np[:,:,2].max())

def update(frame):
    scat._offsets3d = (
        pos_np[frame,:,0],
        pos_np[frame,:,1],
        pos_np[frame,:,2]
    )
    return (scat,)

ani = FuncAnimation(fig, update, frames=T, interval=30)

ani.save("solar_system.mp4", writer="ffmpeg", fps=30)