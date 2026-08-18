# Data dictionary

## Canonical processed model data

### `greenhouse_model_master_v3_correct_nasa.csv`

Key fields:

| Field | Meaning |
|---|---|
| `timestamp_sensor_UTC6` | Greenhouse logger-clock timestamp after the empirical UTC+6 alignment convention |
| `timestamp_UTC` | Corresponding UTC timestamp |
| `split` | `train`, `validation`, or `test` |
| `dt_prev_min` | Actual elapsed minutes since the preceding retained observation |
| `sequence_break_before` | Indicates a recurrent-sequence break before the row |
| `segment_id` | Contiguous sequence segment identifier |
| `core_row_valid` | Whether the row is retained for the final v3 protocol |
| `endpoint_eligible_1h/6h/24h/72h` | Whether a target exists within ±7 min of the requested horizon in the same contiguous segment/split |
| `avg_temp` | Greenhouse mean air temperature, °C |
| `avg_hum` | Greenhouse mean relative humidity, % |
| `土温` | Soil temperature, °C |
| `土湿` | Soil moisture |
| `greenhouse_light_mean` | Mean greenhouse light signal |
| `NASA_T2M_C` | NASA POWER 2-m air temperature, °C |
| `NASA_WS2M_m_s` | NASA POWER 2-m wind speed, m/s |
| `NASA_WS10M_m_s` | NASA POWER 10-m wind speed, m/s |
| `NASA_ALLSKY_SFC_SW_DWN_W_m2_equiv` | Hourly solar radiation converted to W/m²-equivalent |
| `NASA_RH2M_pct` | NASA POWER 2-m relative humidity, % |
| `NASA_T2MDEW_C` | NASA POWER 2-m dew-point temperature, °C |
| `NASA_PRECTOTCORR_mm_day` | NASA POWER corrected precipitation, mm/day |
| `nasa_forcing_latitude` | 40.54 |
| `nasa_forcing_longitude` | 81.30 |

### `greenhouse_model_normalized_v3_correct_nasa.csv`

Contains the canonical model features plus train-only Min-Max normalized versions.
Normalization parameters are stored in `normalization_parameters_v3_correct_nasa.csv`.

## NASA source file

`POWER_Point_Hourly_20240728_20251231_040d54N_081d30E_UTC.csv`

- Point: 40.54°N, 81.30°E
- Native resolution: hourly
- Time standard: UTC
- Date range: 2024-07-28 through 2025-12-31

## Time convention

UTC+6 is an empirical logger-clock alignment supported by greenhouse-light / NASA-solar
correspondence. It must not be described as the official civil timezone of Alar.

## Obsolete data

Do not use NASA forcing at 37.2944°N, 79.8547°E or NASA-dependent v2 result files.
