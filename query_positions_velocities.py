# REQUIREMENTS
# pip install astroquery astropy numpy pandas
#
# Script: query_3I_ATLAS_horizons.py

from astroquery.jplhorizons import Horizons
from astropy.time import Time, TimeDelta
import astropy.units as u
import numpy as np
import pandas as pd
import xarray as xr

OBJECTS    = ["Sun", "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "3I/ATLAS"]
CENTER     = "@sun"
START_DATE = Time("2025-05-07")
TIME_STEP  = TimeDelta(1, format='jd') # How finely spaced the time axis is
END_DATE   = Time("2027-01-01")
SAVE_FILENAME   = "state_vectors.nc"

# Calculate how many timestamps we have and generate a numpy array of them
N_TIME_STAMPS = int(np.floor((END_DATE - START_DATE)/TIME_STEP)) + 1
TIME_STAMPS   = START_DATE + np.arange(N_TIME_STAMPS) * TIME_STEP

KM_PER_AU = 149597870.7            # km per AU
S_PER_DAY = 86400.0                # seconds per day

# Our data array, with dimensions [objects, timestamps, components]
data = xr.DataArray(
    np.zeros((len(OBJECTS), N_TIME_STAMPS, 6)),
    dims=["object", "time", "component"],
    coords={
        "object": OBJECTS,
        "time": np.arange(1000),
        "component": ["x","y","z","vx","vy","vz"] # Units are AU and days
    }
)

#Epoch array in the format NASA Horizons wants, only in the current chunk
jd_epochs = TIME_STAMPS.tbd.jd

# Query Horizons vectors at these epochs
for object in OBJECTS:
    object_data = Horizons(id=object, location=CENTER, epochs=jd_epochs)
    vec = object_data.vectors()   # returns astropy.table.Table

    # extract the relevant columns (x,y,z in AU; vx,vy,vz in AU/day)
    for component, _ in data.groupby("component"):
        data[object, :, component] = np.array(vec[component], dtype=float)

# save to CSV
data.to_netcdf(SAVE_FILENAME)