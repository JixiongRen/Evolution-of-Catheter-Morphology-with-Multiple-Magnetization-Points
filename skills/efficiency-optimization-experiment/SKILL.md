---
name: efficiency-optimization-experiment
description: Run large FK/IK efficiency optimization experiments in this repository, with unique experiment IDs, matched records and log-data directories, adaptive wait intervals, local dev commits, and documentation under docs/efficiency_optimization_experiment.
---

# Efficiency Optimization Experiment

Use this workflow for large experiments that aim to reduce FK/IK runtime.

## Scope

- Target: reduce IK runtime from 30-60 min per case toward about 1 min, with 3 min as the tolerated upper bound.
- Main technical path:
  - optimize FK solve time
  - restructure FK linearization and how IK uses FK
  - prefer matrix-free, reduced-space, coarse-to-fine, and compilation-reuse strategies before neural-network replacement
- Neural networks are a future helper for initialization or attempt selection only.

## Required ID And Paths

At the start of every large efficiency experiment, create one unique ID:

```text
eff_opt_exp_YYYYMMDDHHMMSS
```

Use the same ID for all artifacts in that round:

- Record:
  - `docs/efficiency_optimization_experiment/{eff_opt_exp_YYYYMMDDHHMMSS}_{exp_name}_record.md`
- Logs and data:
  - `docs/efficiency_optimization_experiment/{eff_opt_exp_YYYYMMDDHHMMSS}_{exp_name}_logs_data/`

Do not reuse an ID across rounds.

## Record Sections

Maintain the record from experiment start to finish. Include at least:

- Experiment Objective
- Baseline And Motivation
- Experiment Plan
- Implementation Changes
- Execution Log
- Runtime Measurement
- Results
- Data Analysis
- Risks And Regressions
- Conclusion
- Next Step Plan

## Runtime Tracking

- Record wall-clock runtime for every run and for the full round.
- Use the previous comparable experiment's runtime as the next wait estimate.
- Adjust the wait estimate when the new task is clearly smaller or larger.
- During waits, stay quiet and poll at low frequency to conserve tokens.

## Git Rules

- After each large experiment round finishes, commit all relevant code, records, and generated experiment documentation locally on branch `dev`.
- Never push.
- Do not commit unrelated generated bulk data unless it is intentionally part of the round.
- If not on `dev`, stop before committing and report the blocker.

## Execution Loop

1. Define `exp_name` and create a unique `eff_opt_exp_YYYYMMDDHHMMSS`.
2. Create the record file and log-data directory.
3. Write the objective, baseline, plan, expected runtime, and success criteria.
4. Make scoped code or parameter changes.
5. Launch the experiment with explicit runtime environment settings when needed.
6. Wait based on the estimated runtime, with quiet low-frequency checks.
7. Collect wall time, solver iteration counts, FK call counts, convergence quality, and output paths.
8. Update analysis, conclusion, and next-step plan.
9. Commit locally on `dev`.

## Quality Metrics

For FK/IK speed work, track both speed and physical quality:

- total wall time
- compile time if measurable
- steady-state solve time
- FK solve count
- IK iteration count
- final tip error
- raw equilibrium residual
- interface gap
- smoothness or curvature anomaly metrics
- current utilization and saturation
- target reachability or control Jacobian rank if available

Speedups are valid only when physical quality does not regress beyond the experiment's stated tolerance.
