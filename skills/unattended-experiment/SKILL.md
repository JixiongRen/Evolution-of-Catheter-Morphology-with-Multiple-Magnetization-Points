---
name: unattended-experiment
description: Run and supervise long unattended experiments in this repository with low-noise monitoring, mandatory experiment records, and local-only dev branch commits.
---

# Unattended Experiment Skill

Use this workflow for large experiment rounds in this repository.

## Rules

1. Continue autonomously after the experiment goal is defined.
2. Do not interrupt a healthy run for routine confirmations.
3. Poll long-running jobs every 10 minutes.
4. Stay quiet during waits. Prefer sleeping or low-frequency checks.
5. Keep all modifications inside the repository workspace.
6. After each large experiment round:
   - update the experiment record
   - commit locally on branch `dev`
   - do not push

## Experiment Record

- Directory: `docs/experiment_records/`
- File name: `{YYYYMMDD}_exp_{exp_name}_record.md`
- Required sections:
  - Experiment Objective
  - Experiment Plan
  - Execution Log
  - Results
  - Data Analysis
  - Conclusion
  - Next Step Plan

## Recommended Execution Loop

1. Define the current experiment objective.
2. Create the experiment record from the template.
3. Apply the planned code or parameter changes.
4. Launch the experiment.
5. Poll every 10 minutes with minimal output.
6. When the run finishes, summarize the evidence and update the record.
7. Choose the next justified experiment step.
8. Commit the round locally on `dev`.

## Escalation

Escalate only when blocked by:

- missing data or calibration assets
- contradictory user requirements
- repository state that makes a safe local commit impossible
