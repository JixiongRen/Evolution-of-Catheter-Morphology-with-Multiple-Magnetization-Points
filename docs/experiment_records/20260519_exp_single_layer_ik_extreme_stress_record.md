# 20260519 Exp Single Layer IK Extreme Stress Record

## Experiment Objective

- Stress-test the current single-layer IK pipeline under mixed magnetization and extreme targets.
- Modify the code based on observed failure modes until at least 10 extreme scenarios solve correctly and stably.

## Experiment Plan

- Build a repeatable stress-suite runner based on `single_layer_ik.auto_run_joint_ik`.
- Use a fixed mixed magnetization pattern:
  - `0,0,-0.005301; 0,0,0.005301; 0.005301,0,0`
- Evaluate at least 10 extreme targets with common solver settings.
- Inspect failures, patch the pipeline, rerun, and repeat by experiment round.
- Success criterion for the final round:
  - at least 10 extreme cases succeed
  - stop reason is acceptable
  - final position error stays within the configured stopping tolerance

## Execution Log

### Environment

- Branch: `dev`
- Working directory: `/home/ub22/Evolution-of-Catheter-Morphology-with-Multiple-Magnetization-Points`
- Key scripts:
  - `single_layer_ik/auto_run_joint_ik.py`
  - `single_layer_ik/experiment_auto_joint_ik_variants.py`
- Key output directory:
  - `ik_out/stress_suite_mixedmag_extreme_config`

### Parameters

- Target task: extreme mixed-magnetization stress suite
- Monitoring interval: `30 min` for this goal
- Mixed magnetization:
  - `0,0,-0.005301; 0,0,0.005301; 0.005301,0,0`

### Timeline

- `2026-05-19` Round 1 started
- Added batch stress-suite entrypoint and config
- Observed JAX x64 truncation warnings during auto current initialization
- Patched x64 configuration earlier in the auto current-init path

## Results

- Round 1 pending

## Data Analysis

- Round 1 pending

## Conclusion

- Round 1 pending

## Next Step Plan

- Run the full 12-case stress suite.
- Analyze failures and cluster them by initialization, reachability, or solver stability.
- Patch the most likely root cause and launch the next round.
