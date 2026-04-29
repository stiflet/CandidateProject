# pip install boto3 botocore numcodecs xarray s3fs cartopy pandas numpy tqdm

from __future__ import annotations

import dataclasses
import datetime as dt
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

import boto3
import cartopy.crs as ccrs
import numcodecs as ncd
import numpy as np
import pandas as pd
import s3fs
import xarray as xr
from botocore import UNSIGNED
from botocore.config import Config
from tqdm.auto import tqdm


# ============================================================
# Data classes
# ============================================================

@dataclasses.dataclass(frozen=True)
class ZarrId:
    run_hour: dt.datetime
    level_type: str       # "sfc" or "prs"
    var_level: str        # e.g. "2m_above_ground"
    var_name: str         # e.g. "TMP"
    model_type: str       # "fcst" or "anl"

    def format_chunk_id(self, chunk_id: str) -> str:
        # Forecast arrays have an extra leading dimension for lead time.
        return f"0.{chunk_id}" if self.model_type == "fcst" else chunk_id


@dataclasses.dataclass(frozen=True)
class VariableSpec:
    """
    Describes one HRRR variable to fetch.
    """
    level_type: str
    var_level: str
    var_name: str
    output_name: str
    model_type: str = "fcst"
    convert: str | None = None   # None, "K_to_C", "Pa_to_hPa"


# ============================================================
# Variable alias catalog
# ============================================================

VARIABLE_ALIASES: dict[str, list[VariableSpec]] = {
    # Single variables
    "temp": [
        VariableSpec("sfc", "2m_above_ground", "TMP", "tmp_2m_c", convert="K_to_C"),
    ],
    "dewpoint": [
        VariableSpec("sfc", "2m_above_ground", "DPT", "dpt_2m_c", convert="K_to_C"),
    ],
    "rh": [
        VariableSpec("sfc", "2m_above_ground", "RH", "rh_2m"),
    ],
    "u10": [
        VariableSpec("sfc", "10m_above_ground", "UGRD", "u10"),
    ],
    "v10": [
        VariableSpec("sfc", "10m_above_ground", "VGRD", "v10"),
    ],
    "cloud": [
        VariableSpec("sfc", "entire_atmosphere", "TCDC", "tcdc"),
    ],
    "skin_temp": [
        VariableSpec("sfc", "surface", "SKT", "skin_temp_c", convert="K_to_C"),
    ],
    "pressure": [
        VariableSpec("sfc", "surface", "PRES", "pres_hpa", convert="Pa_to_hPa"),
    ],

    # Bundles
    "wind": [
        VariableSpec("sfc", "10m_above_ground", "UGRD", "u10"),
        VariableSpec("sfc", "10m_above_ground", "VGRD", "v10"),
    ],
    "ngboost_core": [
        VariableSpec("sfc", "2m_above_ground", "TMP", "tmp_2m_c", convert="K_to_C"),
        VariableSpec("sfc", "2m_above_ground", "DPT", "dpt_2m_c", convert="K_to_C"),
        VariableSpec("sfc", "2m_above_ground", "RH", "rh_2m"),
        VariableSpec("sfc", "10m_above_ground", "UGRD", "u10"),
        VariableSpec("sfc", "10m_above_ground", "VGRD", "v10"),
        VariableSpec("sfc", "entire_atmosphere", "TCDC", "tcdc"),
        VariableSpec("sfc", "surface", "SKT", "skin_temp_c", convert="K_to_C"),
    ],
    "ngboost_core_with_pressure": [
        VariableSpec("sfc", "2m_above_ground", "TMP", "tmp_2m_c", convert="K_to_C"),
        VariableSpec("sfc", "2m_above_ground", "DPT", "dpt_2m_c", convert="K_to_C"),
        VariableSpec("sfc", "2m_above_ground", "RH", "rh_2m"),
        VariableSpec("sfc", "10m_above_ground", "UGRD", "u10"),
        VariableSpec("sfc", "10m_above_ground", "VGRD", "v10"),
        VariableSpec("sfc", "entire_atmosphere", "TCDC", "tcdc"),
        VariableSpec("sfc", "surface", "SKT", "skin_temp_c", convert="K_to_C"),
        VariableSpec("sfc", "surface", "PRES", "pres_hpa", convert="Pa_to_hPa"),
    ],
}


def resolve_variables(
    variables: list[VariableSpec] | None = None,
    aliases: list[str] | None = None,
) -> list[VariableSpec]:
    """
    Combine explicit VariableSpec entries and alias-based selections.
    Deduplicates by output_name.
    """
    combined: list[VariableSpec] = []

    if aliases:
        for alias in aliases:
            if alias not in VARIABLE_ALIASES:
                raise ValueError(f"Unknown variable alias: {alias}")
            combined.extend(VARIABLE_ALIASES[alias])

    if variables:
        combined.extend(variables)

    if not combined:
        raise ValueError("No variables requested.")

    deduped: dict[str, VariableSpec] = {}
    for var in combined:
        deduped[var.output_name] = var

    return list(deduped.values())


# ============================================================
# URL / S3 helpers
# ============================================================

def create_s3_group_url(zarr_id: ZarrId, prefix: bool = False) -> str:
    url = "s3://hrrrzarr/" if prefix else ""
    url += zarr_id.run_hour.strftime(
        f"{zarr_id.level_type}/%Y%m%d/%Y%m%d_%Hz_{zarr_id.model_type}.zarr/"
    )
    url += f"{zarr_id.var_level}/{zarr_id.var_name}"
    return url


def create_s3_subgroup_url(zarr_id: ZarrId, prefix: bool = False) -> str:
    return create_s3_group_url(zarr_id, prefix=prefix) + f"/{zarr_id.var_level}"


def create_s3_chunk_url(zarr_id: ZarrId, chunk_id: str, prefix: bool = False) -> str:
    return create_s3_subgroup_url(zarr_id, prefix=prefix) + f"/{zarr_id.var_name}/{zarr_id.format_chunk_id(chunk_id)}"


def get_s3_resource():
    return boto3.resource(
        service_name="s3",
        region_name="us-west-1",
        config=Config(
            signature_version=UNSIGNED,
            retries={"max_attempts": 10, "mode": "standard"},
            max_pool_connections=64,
        ),
    )


def retrieve_object_bytes(s3, key: str, bucket: str = "hrrrzarr") -> bytes:
    return s3.Object(bucket, key).get()["Body"].read()


# ============================================================
# Dtype / decode helpers
# ============================================================

def get_hrrr_chunk_dtype(zarr_id: ZarrId) -> np.dtype:
    """
    Practical dtype rule for HRRR Zarr forecast chunks.
    """
    change_time = dt.datetime(2024, 6, 1, 0)

    dtype = np.dtype("<f2")
    if zarr_id.run_hour >= change_time:
        dtype = np.dtype("<f4")

    if zarr_id.var_level == "surface" and zarr_id.var_name == "PRES":
        dtype = np.dtype("<f4")

    return dtype


def decompress_fcst_chunk(compressed_data: bytes, dtype: np.dtype) -> np.ndarray:
    """
    Forecast chunk shape is (n_leads, 150, 150).
    """
    buffer = ncd.blosc.decompress(compressed_data)
    flat = np.frombuffer(buffer, dtype=dtype)

    entry_size = 150 * 150
    if len(flat) % entry_size != 0:
        raise ValueError(f"Unexpected chunk size {len(flat)} for dtype {dtype}")

    n_leads = len(flat) // entry_size
    return flat.reshape(n_leads, 150, 150)


# ============================================================
# Grid / nearest point helpers
# ============================================================

def get_hrrr_projection():
    return ccrs.LambertConformal(
        central_longitude=262.5,
        central_latitude=38.5,
        standard_parallels=(38.5, 38.5),
        globe=ccrs.Globe(semimajor_axis=6371229, semiminor_axis=6371229),
    )


def load_chunk_index():
    fs = s3fs.S3FileSystem(anon=True)
    return xr.open_zarr(s3fs.S3Map("s3://hrrrzarr/grid/HRRR_chunk_index.zarr", s3=fs))


def get_nearest_point(chunk_index: xr.Dataset, lon: float, lat: float) -> xr.Dataset:
    projection = get_hrrr_projection()
    x, y = projection.transform_point(lon, lat, ccrs.PlateCarree())
    return chunk_index.sel(x=x, y=y, method="nearest")


# ============================================================
# Utility helpers
# ============================================================

def normalize_lead_hours(lead_hours):
    """
    Accepts:
      - None               -> use all available leads
      - int                -> one lead
      - list/tuple/set     -> selected leads
      - range              -> selected leads
      - slice              -> handled later after chunk read
    """
    if lead_hours is None:
        return None
    if isinstance(lead_hours, int):
        return np.array([lead_hours], dtype=np.int16)
    if isinstance(lead_hours, slice):
        return lead_hours
    return np.array(sorted(set(int(x) for x in lead_hours)), dtype=np.int16)


def apply_conversion(values: np.ndarray, convert: str | None) -> np.ndarray:
    if convert is None:
        return values
    if convert == "K_to_C":
        return values - 273.15
    if convert == "Pa_to_hPa":
        return values / 100.0
    raise ValueError(f"Unknown conversion: {convert}")


# ============================================================
# Variable probing / dynamic selection
# ============================================================

def probe_variable(
    run_hour: dt.datetime,
    var: VariableSpec,
    chunk_id: str,
    s3,
) -> tuple[bool, str]:
    """
    Test whether a variable appears readable for one run and one chunk.
    """
    zid = ZarrId(
        run_hour=run_hour,
        level_type=var.level_type,
        var_level=var.var_level,
        var_name=var.var_name,
        model_type=var.model_type,
    )
    key = create_s3_chunk_url(zid, chunk_id)
    dtype = get_hrrr_chunk_dtype(zid)

    try:
        compressed = retrieve_object_bytes(s3, key)
        _ = decompress_fcst_chunk(compressed, dtype=dtype)
        return True, "ok"
    except Exception as e:
        return False, str(e)


def filter_available_variables(
    variables: list[VariableSpec],
    probe_run_hour: dt.datetime,
    chunk_id: str,
    s3,
    on_missing: str = "drop",   # "drop", "keep", "error"
) -> tuple[list[VariableSpec], list[VariableSpec]]:
    """
    Preflight-check variables on one sample run.

    on_missing:
      - "drop": exclude variables that fail probe
      - "keep": keep them, let per-run logic fill NaN later
      - "error": raise immediately
    """
    available = []
    missing = []

    for var in variables:
        ok, msg = probe_variable(probe_run_hour, var, chunk_id, s3)
        if ok:
            available.append(var)
        else:
            missing.append(var)
            print(f"Variable unavailable in probe: {var.output_name} ({var.var_name}/{var.var_level}) -> {msg}")

    if missing and on_missing == "error":
        raise RuntimeError("Some requested variables failed preflight probe.")

    if on_missing == "drop":
        return available, missing

    return variables, missing


# ============================================================
# Single-variable fetch for one run
# ============================================================

def fetch_one_variable_for_run(
    run_hour: dt.datetime,
    var: VariableSpec,
    chunk_id: str,
    in_chunk_x: int,
    in_chunk_y: int,
    s3,
    lead_hours=None,
    sleep_seconds: float = 1.5,
    max_retries: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        selected_leads, values
    """
    lead_hours = normalize_lead_hours(lead_hours)

    zid = ZarrId(
        run_hour=run_hour,
        level_type=var.level_type,
        var_level=var.var_level,
        var_name=var.var_name,
        model_type=var.model_type,
    )
    key = create_s3_chunk_url(zid, chunk_id)
    dtype = get_hrrr_chunk_dtype(zid)

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            compressed = retrieve_object_bytes(s3, key)
            arr = decompress_fcst_chunk(compressed, dtype=dtype)
            point_vals = arr[:, in_chunk_y, in_chunk_x].astype(np.float32)

            if isinstance(lead_hours, slice):
                selected_idx = np.arange(len(point_vals), dtype=np.int16)[lead_hours]
                point_vals = point_vals[lead_hours]
            elif lead_hours is None:
                selected_idx = np.arange(len(point_vals), dtype=np.int16)
            else:
                selected_idx = lead_hours[lead_hours < len(point_vals)]
                point_vals = point_vals[selected_idx]

            point_vals = apply_conversion(point_vals, var.convert)
            return selected_idx, point_vals

        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(sleep_seconds * attempt)
            else:
                raise RuntimeError(
                    f"Failed for run={run_hour:%Y-%m-%d %H:%MZ}, "
                    f"var={var.var_name}, level={var.var_level}"
                ) from last_err


# ============================================================
# Multi-variable single-run fetch with dynamic failure handling
# ============================================================

def fetch_one_run_multi_dynamic(
    run_hour: dt.datetime,
    variables: list[VariableSpec],
    chunk_id: str,
    in_chunk_x: int,
    in_chunk_y: int,
    s3,
    lead_hours=None,
    variable_fail_policy: str = "nan",   # "nan", "skip_variable", "error"
) -> pd.DataFrame | None:
    """
    Per run:
    - if one variable fails, do NOT lose the whole run unless policy='error'
    - the first successful variable defines the lead-hour index

    variable_fail_policy:
      - "nan": failed variable becomes NaN column for that run
      - "skip_variable": ignore failed variable for that run
      - "error": raise immediately
    """
    base_df = None
    successful_cols = 0
    requested_outputs = [v.output_name for v in variables]

    for var in variables:
        try:
            selected_idx, vals = fetch_one_variable_for_run(
                run_hour=run_hour,
                var=var,
                chunk_id=chunk_id,
                in_chunk_x=in_chunk_x,
                in_chunk_y=in_chunk_y,
                s3=s3,
                lead_hours=lead_hours,
            )

            if base_df is None:
                valid_times = [run_hour + dt.timedelta(hours=int(h)) for h in selected_idx]
                base_df = pd.DataFrame(
                    {
                        "run_time": run_hour,
                        "lead_hour": selected_idx.astype(np.int16),
                        "valid_time": valid_times,
                    }
                )

            base_df[var.output_name] = vals
            successful_cols += 1

        except Exception as e:
            if variable_fail_policy == "error":
                raise

            print(f"Missing variable for run {run_hour:%Y-%m-%d %H:%MZ}: {var.output_name} -> {e}")

            if variable_fail_policy == "skip_variable":
                continue

            if variable_fail_policy == "nan" and base_df is not None:
                base_df[var.output_name] = np.nan

    if base_df is None:
        return None

    for col in requested_outputs:
        if col not in base_df.columns:
            base_df[col] = np.nan

    if successful_cols == 0:
        return None

    return base_df


# ============================================================
# Main fetcher
# ============================================================

def fetch_hrrr_point_forecasts_dynamic(
    start: str | dt.datetime,
    end: str | dt.datetime,
    lat: float,
    lon: float,
    variables: list[VariableSpec] | None = None,
    aliases: list[str] | None = None,
    lead_hours=None,
    max_workers: int = 16,
    run_frequency: str = "1H",
    preflight_policy: str = "drop",       # "drop", "keep", "error"
    variable_fail_policy: str = "nan",    # "nan", "skip_variable", "error"
) -> pd.DataFrame:
    """
    Dynamic HRRR point fetcher.

    Parameters
    ----------
    start, end:
        inclusive start, exclusive end
    lat, lon:
        target point
    variables:
        explicit VariableSpec list
    aliases:
        optional alias bundles like ["ngboost_core"] or ["temp", "wind", "cloud"]
    lead_hours:
        None -> all available leads
        int -> one lead
        list/range -> specific lead hours
        slice -> selected slice
    max_workers:
        parallel workers across runs
    run_frequency:
        usually "1H"
    preflight_policy:
        "drop", "keep", or "error"
    variable_fail_policy:
        "nan", "skip_variable", or "error"

    Returns
    -------
    DataFrame with:
        run_time, lead_hour, valid_time, <requested variable columns>
    """
    start = pd.Timestamp(start).to_pydatetime()
    end = pd.Timestamp(end).to_pydatetime()

    variables = resolve_variables(variables=variables, aliases=aliases)

    chunk_index = load_chunk_index()
    nearest = get_nearest_point(chunk_index, lon, lat)

    chunk_id = str(nearest.chunk_id.values)
    in_chunk_x = int(nearest.in_chunk_x.values)
    in_chunk_y = int(nearest.in_chunk_y.values)
    grid_lat = float(nearest.latitude.values)
    grid_lon = float(nearest.longitude.values)

    print("Nearest HRRR point:")
    print(f"  requested lat/lon : {lat:.4f}, {lon:.4f}")
    print(f"  HRRR grid lat/lon : {grid_lat:.4f}, {grid_lon:.4f}")
    print(f"  chunk_id          : {chunk_id}")
    print(f"  in_chunk_x/y      : {in_chunk_x}, {in_chunk_y}")

    run_hours = pd.date_range(
        start=start,
        end=end,
        freq=run_frequency,
        inclusive="left",
    ).to_pydatetime().tolist()

    if not run_hours:
        raise ValueError("No run hours generated.")

    s3 = get_s3_resource()

    active_variables, missing_probe = filter_available_variables(
        variables=variables,
        probe_run_hour=run_hours[0],
        chunk_id=chunk_id,
        s3=s3,
        on_missing=preflight_policy,
    )

    print("\nActive variables:")
    for v in active_variables:
        print(f"  {v.output_name}: {v.level_type}/{v.var_level}/{v.var_name}/{v.model_type}")

    if missing_probe:
        print("\nVariables that failed probe:")
        for v in missing_probe:
            print(f"  {v.output_name}: {v.level_type}/{v.var_level}/{v.var_name}/{v.model_type}")

    out = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(
                fetch_one_run_multi_dynamic,
                run_hour=run_hour,
                variables=active_variables,
                chunk_id=chunk_id,
                in_chunk_x=in_chunk_x,
                in_chunk_y=in_chunk_y,
                s3=s3,
                lead_hours=lead_hours,
                variable_fail_policy=variable_fail_policy,
            ): run_hour
            for run_hour in run_hours
        }

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Fetching HRRR runs"):
            res = fut.result()
            if res is not None:
                out.append(res)

    if not out:
        raise RuntimeError("No runs were successfully fetched.")

    df = (
        pd.concat(out, ignore_index=True)
        .sort_values(["run_time", "lead_hour"])
        .reset_index(drop=True)
    )

    for v in variables:
        if v.output_name not in df.columns:
            df[v.output_name] = np.nan

    ordered_cols = ["run_time", "lead_hour", "valid_time"] + [v.output_name for v in variables]
    df = df[[c for c in ordered_cols if c in df.columns]]

    df.attrs["requested_lat"] = lat
    df.attrs["requested_lon"] = lon
    df.attrs["grid_lat"] = grid_lat
    df.attrs["grid_lon"] = grid_lon
    df.attrs["chunk_id"] = chunk_id

    return df


# ============================================================
# Feature engineering helpers for downstream modeling
# ============================================================

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add common time-based predictors for forecasting models like NGBoost.
    """
    df = df.copy()
    df["lead_time"] = df["lead_hour"]
    df["hour"] = df["valid_time"].dt.hour
    df["dayofyear"] = df["valid_time"].dt.dayofyear
    df["month"] = df["valid_time"].dt.month
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add useful derived meteorological features if the needed columns exist.
    """
    df = df.copy()

    if {"u10", "v10"}.issubset(df.columns):
        df["wind_speed_10m"] = np.sqrt(df["u10"] ** 2 + df["v10"] ** 2)

    if "tmp_2m_c" in df.columns:
        df["tmp_2m_lag1"] = df.groupby("run_time")["tmp_2m_c"].shift(1)
        df["tmp_2m_change"] = df["tmp_2m_c"] - df["tmp_2m_lag1"]

    return df


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    variables = [
        # --- Temperature (most important) ---
        VariableSpec("sfc", "2m_above_ground",  "TMP",  "tmp_2m_c"),
    
        # --- Moisture ---
        VariableSpec("sfc", "2m_above_ground",  "DPT",  "dpt_2m_c"),   # dewpoint
        VariableSpec("sfc", "2m_above_ground",  "RH",   "rh_2m"),      # relative humidity
    
        # --- Wind ---
        VariableSpec("sfc", "10m_above_ground", "UGRD", "u10"),
        VariableSpec("sfc", "10m_above_ground", "VGRD", "v10"),
    
        # --- Clouds (very important for temp uncertainty) ---
        VariableSpec("sfc", "entire_atmosphere","TCDC", "tcdc"),
    ]

    df = fetch_hrrr_point_forecasts_dynamic(
        start="2020-01-01 00:00",
        end="2026-02-01 00:00",
        lat=40.7769,
        lon=-73.8740,
        variables=variables,
        max_workers=32,
        lead_hours=range(0, 19),
    )
