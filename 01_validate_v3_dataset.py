
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
files=[
    ROOT/"data/POWER_Point_Hourly_20240728_20251231_040d54N_081d30E_UTC.csv",
    ROOT/"data/greenhouse_model_master_v3_correct_nasa.csv",
    ROOT/"data/greenhouse_model_normalized_v3_correct_nasa.csv",
    ROOT/"data/normalization_parameters_v3_correct_nasa.csv",
]
for p in files:
    if not p.exists(): raise FileNotFoundError(p)
df=pd.read_csv(files[1],usecols=["timestamp_sensor_UTC6","split","core_row_valid",
                                 "nasa_forcing_latitude","nasa_forcing_longitude"])
df=df[df.core_row_valid==1]
print("Valid rows:",len(df))
print(df.groupby("split").size())
print("Forcing coordinate:",
      df.nasa_forcing_latitude.dropna().iloc[0],
      df.nasa_forcing_longitude.dropna().iloc[0])
assert abs(df.nasa_forcing_latitude.dropna().iloc[0]-40.54)<1e-9
assert abs(df.nasa_forcing_longitude.dropna().iloc[0]-81.30)<1e-9
