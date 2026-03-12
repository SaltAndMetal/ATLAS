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
from tqdm import tqdm

import warnings
from astropy.utils.exceptions import ErfaWarning

# Ignore all ErfaWarnings
warnings.simplefilter('ignore', category=ErfaWarning)

OBJECT_IDS    = {"Sun":20,
                 "Mercury":199,
                 "Venus":299,
                 "Earth":399,
                 "Mars":499,
                 "Jupiter":599,
                 "Saturn":699,
                 "Uranus":799,
                 "Neptune":899,
                 "3I/ATLAS":"C/2025 N1"
                }
CENTER     = "@sun"
START_DATE = Time("2025-05-07")
TIME_STEP  = TimeDelta(1, format='jd') # How finely spaced the time axis is
END_DATE   = Time("2027-01-01")
SAVE_FILENAME   = "state_vectors.nc"

# Calculate how many timestamps we have and generate a numpy array of them
N_TIME_STAMPS = int(np.floor((END_DATE - START_DATE)/TIME_STEP)) + 1
TIME_STAMPS   = START_DATE + np.arange(N_TIME_STAMPS) * TIME_STEP

CHUNK_SIZE = 50 # How many timestamps to request at once
N_CHUNKS = int(np.ceil(N_TIME_STAMPS / CHUNK_SIZE))

print(f"Requesting {N_TIME_STAMPS} time stamps from {START_DATE} to {END_DATE}. Making {N_CHUNKS} requests of maximum size {CHUNK_SIZE} timestamps each ")

# Our data array, with dimensions [objects, timestamps, components]
data = xr.DataArray(
    np.zeros((len(OBJECT_IDS), N_TIME_STAMPS, 6)),
    dims=["object", "time", "component"],
    coords={
        "object": list(OBJECT_IDS.keys()),
        "time": TIME_STAMPS.tdb.jd,
        "component": ["x","y","z","vx","vy","vz"] # Units are AU and days
    }
)

# Loading bar
with tqdm(total=len(OBJECT_IDS)*N_CHUNKS) as pbar:
    # Query Horizons vectors at these epochs
    for object, id in OBJECT_IDS.items():

        # Query and get all the vectors, once for each chunk
        vec = np.array((N_TIME_STAMPS,))
        for start_date in range(0, N_TIME_STAMPS, CHUNK_SIZE):
            time_range = TIME_STAMPS[start_date:start_date+CHUNK_SIZE].tdb.jd

            # Make the query
            if object == "3I/ATLAS":
                object_data = Horizons(id=id, location=CENTER, epochs=time_range, id_type="smallbody")
            else:
                object_data = Horizons(id=id, location=CENTER, epochs=time_range)
                
            vec = object_data.vectors()

            # Put the results into the DataArray
            for component, _ in data.groupby("component"):
                data.loc[object, time_range, component] = np.array(vec[component], dtype=float)
            
            pbar.update(1)

# save to CSV
data.to_netcdf(SAVE_FILENAME)