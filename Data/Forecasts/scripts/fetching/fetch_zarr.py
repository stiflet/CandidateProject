from pyproj import Transformer
import rioxarray
import xarray as xr
import pandas as pd
from dask.diagnostics import ProgressBar

def fetch_forecast(zarr_url, lat, lon, filename):
  ds = xr.open_zarr(zarr_url)

  lon_lat_to_ds = Transformer.from_crs("EPSG:4326", ds.rio.crs, always_xy=True)
  

  x, y = lon_lat_to_ds.transform(lon, lat)
  point = ds.sel(x=x, y=y, method="nearest")


  cols = point.to_array().coords.get("variable").values

  da = point.to_dataarray()

  with ProgressBar():
      vals = da.compute().values

  df = pd.DataFrame(vals.T, columns=cols)
  df["init_time"] = df.init_time.values
  df["valid_time"] = df.valid_time.values
  df["lead_time"] = df.lead_time.values
  df.set_index(["init_time", "valid_time"], inplace=True)

  df.to_csv(filename)


if __name__ == '__main__':

  zarr_url = "https://data.dynamical.org/noaa/hrrr/forecast-48-hour/latest.zarr"
  lon, lat = -73.8727, 40.7761
  filename = "hrrr_forecast.csv"

  fetch_forecast(zarr_url, lat, lon, filename)




