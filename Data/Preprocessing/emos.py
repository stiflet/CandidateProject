from huggingface_hub import upload_file, delete_file
from scipy.constants import convert_temperature
import pandas as pd
import time
import os

forecasts = pd.read_parquet('/content/WeatherData/Forecasts/gefs/klga.parquet').set_index('valid_time').tz_localize('utc')
observations = pd.read_parquet('/content/WeatherData/Observations/metar_klga.parquet').set_index('valid')

obs = observations.tmpf.copy()
obs = obs.apply(convert_temperature, args=('f','c'))

emos_subset = forecasts[['init_time', 'lead_time', 'ensemble_member', 'maximum_temperature_2m']].copy()


lead_times = [f"{i}h" for i in range(3, 13, 3)]

for lead_time in lead_times:
  obs = obs.resample(lead_time).max()
  emos_leadtime = emos_subset[emos_subset.lead_time == lead_time][['ensemble_member', 'maximum_temperature_2m']].pivot(columns = 'ensemble_member', values = 'maximum_temperature_2m')
  emos_leadtime['obs'] = obs

  emos_leadtime.to_parquet(f'emos_{lead_time}.parquet')

  upload_file(
    path_or_fileobj=f'emos_{lead_time}.parquet',
    path_in_repo=f"Preprocessing/EMOS/emos_{lead_time}.parquet",
    repo_id="Stiflet/WeatherData",
    repo_type="dataset")
  
  os.remove(f'emos_{lead_time}.parquet')
  time.sleep(5)
