from astropy.wcs import WCS
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

data = xr.open_dataset("state_vectors.nc")
data = data["__xarray_dataarray_variable__"]

pos = data.sel(component=["RA", "DEC", "SMIA_3sigma", "SMAA_3sigma", "Theta_3sigma"])
pos = pos.transpose("time", "object", "component")
pos_np = pos.values

T, _, _ = pos_np.shape

wcs = WCS(naxis=2)

# Reference point (center of the plot in RA/Dec)
wcs.wcs.crval = [np.mean(pos[:, -1, 0]), np.mean(pos[:, -1, 1])]

# Reference pixel (center of the image)
wcs.wcs.crpix = [0, 0]

# Pixel scale (degrees per pixel)
scale = 0.01  # adjust
wcs.wcs.cdelt = np.array([-scale, scale])

# Projection type (gnomonic / tangent plane)
wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

from astropy.visualization.wcsaxes import WCSAxes

fig = plt.figure()
plt.style.use('dark_background')
ax = fig.add_subplot(111, projection=wcs)

from matplotlib.patches import Ellipse

scatter = ax.scatter([], [], transform=ax.get_transform('world'))
pad = 5
ax.set_xlim(pos_np[:, -1, 0].min() - pad, pos_np[:, -1, 0].max() + pad)
ax.set_ylim(pos_np[:,-1, 1].min() - pad, pos_np[:, -1, 1].max() + pad)

ellipse = Ellipse((0, 0), width=0, height=0, angle=0, fill=False)
ax.add_patch(ellipse)

def update(frame):
    x = pos_np[frame, -1, 0]
    y = pos_np[frame, -1, 1]

    scatter.set_offsets([[x, y]])

    ellipse.center = (x, y)
    ellipse.width = 2 * pos_np[frame, -1, 2]
    ellipse.height = 2 * pos_np[frame, -1, 3]
    ellipse.angle = pos_np[frame, -1, 4]

    return scatter, ellipse

from matplotlib.animation import FuncAnimation

ani = FuncAnimation(fig, update, frames=T, interval=50, blit=False)

ani.save("sky.mp4", writer="ffmpeg", fps=30)