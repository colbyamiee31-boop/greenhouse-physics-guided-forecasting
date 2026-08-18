# Repository validation status

## Verified exactly in the current reconstruction environment

1. Canonical v3 data:
   - 70,398 valid rows
   - Train 52,993
   - Validation 8,685
   - Test 8,720
   - NASA forcing point 40.54°N, 81.30°E

2. Grey-box multi-timescale relation:
   reproduced validation/test R² values with maximum absolute difference **0.0**
   relative to the frozen v3 result CSV.

3. Five-seed statistics:
   recomputation from the archived five-seed prediction arrays reproduced the final
   robust paired conclusions, including:
   - GRU 1 h temperature: ΔRMSE = -0.275°C
   - Fixed-physics LSTM 6 h temperature: ΔRMSE = +0.687°C,
     nested CI [+0.454, +0.925]

4. NASA grey-box stress test:
   - T2M ±2°C: mean |6 h endpoint shift| ≈ 0.182°C
   - Solar ±20%: ≈ 0.286°C
   - Wind ±50%: 0 under the current reduced relation because the direct wind/outdoor
     term is practically non-identifiable.

## Independent full-training smoke validation

A fresh 14-epoch LSTM / fixed-physics LSTM reconstruction produced:

- 6 h temperature RMSE:
  - LSTM: 2.489°C
  - Fixed physics: 3.165°C
  - Δ = +0.676°C

- Physics-consistency RMSE:
  - LSTM: 0.580°C/h
  - Fixed physics: 0.469°C/h
  - Δ = -0.111°C/h

Thus the central qualitative result is independently reproduced:
**physics consistency improves while 6 h predictive generalization deteriorates.**

A 1024-origin gradient smoke diagnostic on the independently trained fixed-physics checkpoint gave:
- negative cosine fraction = 50%
- median raw physics/data gradient norm ratio ≈ 200×

The manuscript-grade archived diagnostic uses 4096 representative origins and reports
56.25% negative batches and a median norm ratio of approximately 168×.

## Why neural-network point estimates can differ

Random seeds are fixed, but CPU thread scheduling, BLAS/PyTorch versions and kernel implementations
can alter optimization trajectories. The manuscript therefore relies on five-seed aggregate results
rather than the numerical value of one independently reconstructed checkpoint.
