# Reproducibility notes

## Neural models
Random seeds are fixed, but exact floating-point training trajectories can differ across
PyTorch versions, CPU/GPU kernels, thread settings and hardware. Manuscript claims should
therefore be based on the five-seed aggregate and paired uncertainty analysis, not a single
checkpoint hash.

## Physics
The hidden thermal-mass parameters are effective model parameters, not uniquely identified
thermophysical constants. Practical equifinality is a reported result, not a bug to suppress.

## NASA forcing
Stress-test perturbations are sensitivity scenarios. They are not empirical estimates of
NASA POWER bias relative to a colocated weather station.

## Time derivative
The final methodology uses finite differences over explicit observation intervals and
coarse-grained horizons. It does not claim automatic differentiation with respect to time
for a discrete LSTM.
