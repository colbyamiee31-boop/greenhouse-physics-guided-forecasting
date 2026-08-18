# Greenhouse Physics-Guided Forecasting Reproducibility Package

Final v3 pipeline for the manuscript on physics-guided recurrent greenhouse forecasting.

## Canonical forcing and protocol
- Greenhouse / NASA POWER point: 40.54°N, 81.30°E, Alar, Xinjiang, China
- NASA POWER: hourly UTC
- Empirical logger alignment: UTC+6 h
- Train: 2024-07-29 to 2025-08-31
- Validation: 2025-09-01 to 2025-10-31
- Winter test: 2025-11-01 to 2025-12-31
- Sequence length: 72 observed records
- Forecast horizons: 1, 6, 24, 72 h
- Target tolerance: ±7 min
- Train-only Min-Max scaling
- No long-gap interpolation
- Invalid 2025-02-29 records quarantined

## Seeds
20260817, 42, 1234, 777, 31415

## Run order
1. `python src/01_validate_v3_dataset.py`
2. `python src/02_train_modern_baselines.py`
3. `python src/03_fit_greybox_multiscale.py`
4. `python src/04_gradient_diagnostics_reference.py`
5. `python src/05_nasa_uncertainty_scenarios.py`
6. `python src/06_multiseed_statistics_reference.py`

## Provenance warning
All NASA-dependent results generated with 37.2944°N, 79.8547°E are obsolete.

Exact wall-clock runtimes depend on hardware and software versions. Fixed seeds reduce
initialization variability but do not guarantee bitwise-identical neural-network weights.
