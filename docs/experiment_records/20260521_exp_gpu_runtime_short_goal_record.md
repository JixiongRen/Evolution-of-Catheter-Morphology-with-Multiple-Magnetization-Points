# GPU Runtime Short Goal Record

## Experiment Objective

Reduce unnecessary GPU memory pressure and per-iteration I/O overhead in the single-layer IK workflow, then check whether disabling global JAX x64 is numerically acceptable for a small comparison case.

## Experiment Plan

1. Force experiment child processes to run with `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
2. Reduce intermediate pose storage frequency by adding a configurable `pose_save_every` option.
3. Add an environment-controlled JAX x64 switch through `MCRS_JAX_ENABLE_X64`.
4. Run one short x64/x32 comparison with the same target and solver settings.

## Execution Log

- Added shared JAX x64 configuration helpers for the single-layer IK package and JAX nondimensional modules.
- Updated experiment launchers to pass `XLA_PYTHON_CLIENT_PREALLOCATE=false` explicitly to child processes.
- Added `--pose_save_every`, defaulting to 10, while preserving `iter_0000.npz` and `final.npz`.
- Ran a four-callback comparison case at target `[-0.04, 0.07, 0.02]` with the mixed magnetization pattern.

## Results

| Mode | x64 enabled | Iterations | Position error mm | Raw equilibrium norm | Pose files |
| --- | --- | ---: | ---: | ---: | --- |
| x64 | true | 4 | 168.9953198334 | 0.0002098861 | `iter_0000.npz`, `final.npz` |
| x32 | false | 4 | 168.9953079543 | 0.0002099163 | `iter_0000.npz`, `final.npz` |

## Data Analysis

- Position error differed by about `1.19e-05 mm` in this short run.
- Raw equilibrium norm differed by about `3.03e-08`.
- The storage change worked as intended: no per-callback centerline NPZ files were written when `pose_save_every=10` and only four callbacks occurred.
- Runtime was not improved in this short test because compilation and setup dominated the run.

## Conclusion

The float32 path can run and produced nearly identical numerical results for this small comparison. It is reasonable to test longer IK runs with `MCRS_JAX_ENABLE_X64=0` to evaluate memory and throughput effects under steady-state workload.

## Next Step Plan

Run one medium-length representative IK case with `MCRS_JAX_ENABLE_X64=0`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`, and `pose_save_every=10`, then compare GPU memory, utilization, runtime, and final convergence quality against the default x64 run.
