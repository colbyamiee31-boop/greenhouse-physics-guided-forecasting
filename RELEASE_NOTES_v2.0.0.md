# Release v2.0.0 — canonical v3 reproducibility package

## Major corrections
- Corrected NASA POWER point to **40.54°N, 81.30°E**.
- Invalidated all NASA-dependent results generated from **37.2944°N, 79.8547°E**.
- Retained the empirically supported **UTC+6 h logger-clock alignment** after revalidation.
- Rebuilt the canonical v3 model-ready and normalized datasets.

## Reproducibility coverage
- Modern LSTM/GRU/TCN/Transformer baseline definitions.
- Multi-timescale grey-box diagnostics.
- Fixed-physics LSTM and gradient-conflict diagnostics.
- NASA boundary-forcing stress tests.
- Five-seed mean ± SD and nested seed/day paired inference.
- One-command analysis workflow through `run_all.py`.

## Verified headline results
- Grey-box validation R² peaks at 6 h (0.786).
- Practical thermal-mass equifinality remains substantial.
- Fixed physics improves 6 h physics consistency while degrading 6 h forecast accuracy.
- Archived five-seed inference reproduces the robust +0.687°C 6 h temperature RMSE penalty
  of the fixed-physics LSTM relative to LSTM.
- Solar perturbations dominate the tested grey-box NASA-boundary sensitivity.

## Release policy
The code repository is intended to remain lightweight. The complete processed-data/results
archive should be deposited in Zenodo or another permanent research repository.
