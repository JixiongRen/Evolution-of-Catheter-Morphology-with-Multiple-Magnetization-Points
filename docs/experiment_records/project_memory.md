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
