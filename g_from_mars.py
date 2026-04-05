import numpy as np
import xarray as xr

data = xr.open_dataset("state_vectors.nc")
data = data["__xarray_dataarray_variable__"]

pos = data.sel(component=["x", "y", "z"])
pos = pos.transpose("time", "object", "component")
pos_np = pos.values

T, _, _ = pos_np.shape

COMET_IDX = -1
MARS_IDX = 4
MARS_MASS = 6.417E23
AU_IN_M = 1.496E11
COMET_MASS = 4.4E10
SOLAR_MASS = 1.989E30
G = 6.67E-11
def distance(pos):
    return np.linalg.norm(pos, axis=-1)

pos_from_mars = pos_np[:, COMET_IDX, :] - pos_np[:, MARS_IDX, :]
closest_mars_t = np.argmin(distance(pos_from_mars))
mars_distance = distance(pos_from_mars[closest_mars_t])
sun_distance = distance(pos_np[closest_mars_t, COMET_IDX, :])

print(f"Gravitational force from Mars is {G*COMET_MASS*MARS_MASS/(mars_distance*AU_IN_M)**2}")
print(f"Gravitational force from the Sun is {G*COMET_MASS*SOLAR_MASS/(sun_distance*AU_IN_M)**2}")