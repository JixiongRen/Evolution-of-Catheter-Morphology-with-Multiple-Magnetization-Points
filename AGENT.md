# Project Agent Rules

Read this file before any non-trivial work in this repository.

## Core Operating Rules

1. Treat this project as a long-running unattended experiment environment.
2. Once an experiment goal is defined, continue execution autonomously until the current large experiment finishes or a real blocker appears.
3. Do not stop in the middle of an experiment for routine confirmations.
4. While a long-running algorithm is executing, check progress every 10 minutes.
5. During the 10-minute wait window, stay mostly silent and prefer sleeping or low-frequency polling to conserve tokens.
6. After each large experiment round, prepare an experiment record under `docs/experiment_records/` and commit the round locally on branch `dev`.
7. Commits must stay local. Never push to any remote.
8. Only modify files inside this workspace. Do not touch files outside the repository directory.

## Experiment Monitoring

- Use low-frequency monitoring for unattended runs.
- Default polling interval during steady execution is 10 minutes.
- Avoid verbose progress output unless a failure, completion, or major state change occurs.
- If the process crashes, collect the failure evidence, update the experiment record, and continue with the next justified step when possible.

## Experiment Records

Every large experiment round must produce one record file:

- Location: `docs/experiment_records/`
- File name: `{YYYYMMDD}_exp_{exp_name}_record.md`

Each record must include at least these sections:

1. Experiment Objective
2. Experiment Plan
3. Execution Log
4. Results
5. Data Analysis
6. Conclusion
7. Next Step Plan

Keep the record updated during the round and finalize it before the local commit.

## Git Rules

- Use only local Git operations.
- Commit completed experiment rounds only on local branch `dev`.
- Do not push.
- Do not rewrite history unless explicitly requested.

## Execution Policy

- After one large experiment finishes, infer the next most reasonable large experiment from the latest results and continue implementation when the path is clear.
- Escalate only for real blockers such as missing assets, contradictory requirements, or unsafe repository state.
- Keep edits scoped, reversible, and documented.
