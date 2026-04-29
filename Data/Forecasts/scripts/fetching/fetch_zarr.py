from pyproj import Transformer
import rioxarray
import xarray as xr
import pandas as pd
import dask
import time
from dask.diagnostics import ProgressBar
from tqdm.auto import tqdm

def open_store(zarr_url):
    try:
        return xr.open_zarr(zarr_url, consolidated=True, chunks="auto")
    except Exception:
        return xr.open_zarr(zarr_url, chunks="auto")

def load_var_with_retry(da, max_retries=3, sleep_seconds=3):
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            with dask.config.set(scheduler="threads"):
                with ProgressBar():
                    return da.load()
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(sleep_seconds)
            else:
                raise last_err

def fetch_forecast(zarr_url, lat, lon, filename, variables=None, max_retries=3, **kwargs):
    ds = open_store(zarr_url)

    transformer = Transformer.from_crs("EPSG:4326", ds.rio.crs, always_xy=True)
    x, y = transformer.transform(lon, lat)

    # subset early
    sub = ds.sel(x=x, y=y, method="nearest")
    if kwargs:
        sub = sub.sel(**kwargs)

    # choose variables
    if variables is None:
        variables = list(sub.data_vars)
    else:
        variables = [v for v in variables if v in sub.data_vars]

    dfs = []
    failed = []

    for var in tqdm(variables, desc="Variables"):
        try:
            da = load_var_with_retry(sub[var], max_retries=max_retries)

            df_var = (
                da.to_dataframe(name=var)
                .reset_index()
            )

            # keep only the useful columns for merging
            keep_cols = [c for c in ["init_time", "valid_time", "lead_time", var] if c in df_var.columns]
            df_var = df_var[keep_cols]

            dfs.append(df_var)

        except Exception as e:
            failed.append((var, str(e)))
            print(f"\nSkipping {var} after {max_retries} failed attempts:\n{e}\n")

    if not dfs:
        raise RuntimeError("No variables could be downloaded successfully.")

    # merge all successful variables
    df = dfs[0]
    merge_keys = [c for c in ["init_time", "valid_time", "lead_time"] if c in df.columns]

    for df_var in dfs[1:]:
        df = df.merge(df_var, on=merge_keys, how="outer")

    if {"init_time", "valid_time"}.issubset(df.columns):
        df.set_index(["init_time", "valid_time"], inplace=True)

    
    df.to_parquet(filename, engine="pyarrow", compression="snappy")

    if failed:
        failed_df = pd.DataFrame(failed, columns=["variable", "error"])
        failed_name = filename.replace(".parquet", "_failed_variables.csv")
        failed_df.to_csv(failed_name, index=False)
        print(f"Some variables failed. Details written to: {failed_name}")

    return df


if __name__ == "__main__":
    dates = ("2020-10-01", "2026-04-05")
    zarr_url = "https://data.dynamical.org/noaa/hrrr/forecast-48-hour/latest.zarr"
    lon, lat = -87.91, 41.98
    filename = "hrrr_forecast.parquet"

    df = fetch_forecast(
        zarr_url,
        lat,
        lon,
        filename,
        init_time=slice(*dates),
        max_retries=3,
    )
