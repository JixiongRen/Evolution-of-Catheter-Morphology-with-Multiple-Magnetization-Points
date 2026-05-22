# Project Memory

## Operational Agreements

- This repository is used for long-running unattended experiments.
- After the experiment goal is set, execution should continue autonomously without routine interruptions.
- Long-running jobs should be monitored every 10 minutes.
- Waiting periods should stay quiet and use low-frequency polling to reduce token usage.
- Each large experiment round must end with:
  - an updated experiment record in `docs/experiment_records/`
  - a local commit on branch `dev`
- No remote push is allowed during this workflow.
- File modifications must stay inside the repository workspace.

## Experiment Documentation Convention

- Directory: `docs/experiment_records/`
- File pattern: `{YYYYMMDD}_exp_{exp_name}_record.md`
- Minimum sections:
  - Experiment Objective
  - Experiment Plan
  - Execution Log
  - Results
  - Data Analysis
  - Conclusion
  - Next Step Plan

## Autonomous Follow-Through

- After each round, choose the next justified large experiment from the latest evidence and continue when the path is clear.
- Stop only for real blockers, missing prerequisites, or contradictory instructions.

## Optimization Priority Notes

- Neural networks are reserved as a future helper for IK initialization and attempt selection.
- The preferred NN role is to predict better `I_init`, `I_ref`, and `x_init`, then let the physics-based LM solver validate and refine the result.
- Do not prioritize full NN replacement of FK, residuals, or LM in the near term because the current dataset is too small and physical consistency risk is high.
- The current high-priority work is major LM/FK solver acceleration: reducing full Jacobian cost, avoiding repeated diagnostics, reusing compilation, improving coarse-to-fine solving, and exploring matrix-free or reduced-control updates.

## Efficiency Optimization Experiment Convention

- Use the `efficiency-optimization-experiment` skill for large FK/IK runtime optimization rounds.
- Store efficiency records and matched logs under `docs/efficiency_optimization_experiment/`.
- Each large round must use one unique ID: `eff_opt_exp_YYYYMMDDHHMMSS`.
- Record file pattern: `{eff_opt_exp_YYYYMMDDHHMMSS}_{exp_name}_record.md`.
- Log-data directory pattern: `{eff_opt_exp_YYYYMMDDHHMMSS}_{exp_name}_logs_data/`.
- Record actual runtime for each round and use it to estimate the next comparable wait interval.
- After each large efficiency experiment, commit relevant changes locally on branch `dev`; never push.
