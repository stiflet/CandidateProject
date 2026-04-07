from pyproj import Transformer
import rioxarray  # for .rio accessor
import xarray as xr
import pandas as pd
from dask.diagnostics import ProgressBar


ds = xr.open_zarr("https://data.dynamical.org/noaa/hrrr/forecast-48-hour/latest.zarr")

lon_lat_to_ds = Transformer.from_crs("EPSG:4326", ds.rio.crs, always_xy=True)
lon, lat = -73.8727, 40.7761

x, y = lon_lat_to_ds.transform(lon, lat)

print(x, y)

point = ds.sel(x=x, y=y, method="nearest")

df = point
cols = df.to_array().coords.get("variable").values

da = df.to_dataarray()

with ProgressBar():
    aa = da.compute().values

bb = pd.DataFrame(aa.T, columns=cols)
bb["init_time"] = df.init_time.values
bb["valid_time"] = df.valid_time.values
bb["lead_time"] = df.lead_time.values
bb.set_index(["init_time", "valid_time"], inplace=True)

bb.to_csv("hrrr_forecast.csv")