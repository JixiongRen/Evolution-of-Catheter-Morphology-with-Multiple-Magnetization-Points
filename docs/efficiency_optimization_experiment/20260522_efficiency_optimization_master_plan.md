# FK/IK Efficiency Optimization Master Plan

Created: 2026-05-22

This document is the master plan for reducing single-layer IK runtime. It is not an experiment record. Each large experiment must still create its own uniquely numbered record and matched log-data directory under this folder.

## 1. Objective

The current IK workflow can take about 30-60 minutes per target case. The optimization target is:

- Target runtime: about 1 minute per case.
- Tolerated upper bound: about 3 minutes per case.
- Required quality: no meaningful regression in physical equilibrium, tip accuracy, interface consistency, smoothness, or current feasibility.

The main route is:

1. Optimize FK solve time.
2. Rebuild FK linearization and the way IK consumes FK sensitivity information.
3. Use neural networks only later as initialization or attempt-selection helpers, not as the primary solver.

## 2. Current Diagnosis

The present performance issue is not expected to be solved by parameter tuning alone. The likely high-cost operations are:

- FK LM iterations that explicitly build dense Jacobians.
- Dense normal-equation construction through `J.T @ J`.
- Dense direct linear solves inside the equilibrium solver.
- Repeated FK calls from IK, especially finite-difference sensitivity calls for `x` or protrusion-related variables.
- Expensive or unnecessary diagnostic output, especially full centerline and compressed pose data saved too frequently.
- Recompilation and shape changes when JAX inputs or solver structures are not kept stable.
- Low-quality initial guesses that force many attempts or many outer iterations.

Previous short runtime work already showed:

- `XLA_PYTHON_CLIENT_PREALLOCATE=false` should be used explicitly for experiment subprocesses.
- Sparse pose saving is preferable to per-iteration `np.savez_compressed`.
- Float32 produced nearly identical results in a short comparison, but longer representative cases still need validation.

The most important unresolved correctness issue is modeling consistency. In particular, the definition of the controlled tip must be made consistent across old IK code, FK code, and single-layer IK code. A fast solver is not useful if different paths optimize different physical points.

## 3. Modeling Assumptions

The target device is a permanent magnet-based magnetic continuum robot. The relevant modeling hierarchy should remain:

- Use high-fidelity Cosserat or GL-style equilibrium solves for final verification and final IK refinement.
- Use lower-cost approximations, coarser discretization, or continuation only to find a good basin.
- Treat permanent magnet sections as physical actuation regions with fixed body-frame magnetization vectors. Their world-frame force and torque effects must remain consistent with the local frame convention and magnetic field model.

The optimization work must preserve the following physical checks:

- The chosen tip or task point is the same in every solver path.
- Magnet centers, rigid segment endpoints, and flexible segment endpoints have explicit conventions.
- `m_body` to world-frame mapping is consistent.
- Field interpolation stays within calibrated field bounds or reports out-of-range behavior.
- Final reported success requires both low task error and acceptable equilibrium residual.

## 4. Strategy

The recommended main line is:

1. Correctness calibration.
2. Dense FK profiling.
3. Matrix-free FK.
4. Implicit adjoint sensitivity.
5. Reduced-space IK.
6. Coarse-to-fine and load/target continuation.
7. Batch regression and fallback policy.

Single-layer penalty IK can remain as a fallback or research branch, but the main production route should move toward reduced-space IK:

```text
minimize task_cost(I, x, z)
subject to E(z, I, x) = 0
```

This keeps equilibrium as a hard inner solve instead of allowing a penalty formulation to trade physical validity for lower tip error.

## 5. Success Metrics

Every large efficiency experiment should report at least:

- Total wall time.
- Compile time if measurable.
- Steady-state solve time.
- FK solve count.
- IK outer iteration count.
- FK LM iteration count.
- Final tip error.
- Raw equilibrium residual.
- Interface gap.
- Smoothness or curvature anomaly metric.
- Current utilization and saturation.
- Control Jacobian rank or singular values when available.
- Whether float32 or float64 is used.
- Whether the run includes first-call compilation.

Speedup is valid only when physical quality remains within the experiment's stated tolerance.

## 6. Two-Month Roadmap

| Phase | Time Window | Goal | Main Deliverable | Success Standard |
| --- | --- | --- | --- | --- |
| P0 | Immediate | Establish workflow and master plan | Skill, directory, this master plan | Future experiments have consistent IDs, records, logs, runtime tracking, and local dev commits |
| P1 | Week 1 | Baseline and modeling consistency | Dense FK/IK profiling experiment | 10-12 fixed cases have runtime decomposition and quality metrics |
| P2 | Week 2 | Low-risk runtime cleanup | Config and I/O optimization | 2-5x speedup without convergence or quality regression |
| P3 | Weeks 3-4 | Matrix-free FK prototype | JVP/VJP Krylov FK path with dense fallback | Per-step FK cost reduced by 5-10x on representative cases |
| P4 | Weeks 4-5 | Implicit adjoint sensitivity | Sensitivity path without explicit full `J` or finite-difference FK loops | IK linearization cost becomes small relative to FK equilibrium solve |
| P5 | Weeks 5-6 | Reduced-space IK | Physical-variable IK over `I` and `x` with strict FK equilibrium | Stable convergence on representative near and stress targets |
| P6 | Weeks 6-7 | Continuation and initialization | Coarse-to-fine mesh, load, and target continuation | Typical IK iteration count drops below 50-80 |
| P7 | Week 8 | Regression and stabilization | 10+ extreme-case benchmark suite | p50 near 1 minute and p95 or extreme cases within about 3 minutes if feasible |

## 7. Phase Details

### P1: Baseline And Modeling Consistency

Goal:

- Find the real bottleneck before changing solver structure.
- Ensure all IK paths optimize the same physical task point.

Tasks:

- Audit tip extraction in old IK, FK engine, and single-layer IK.
- Define one canonical task point convention.
- Audit rigid segment endpoint, magnet center, and flexible node conventions.
- Add or run profiling that separates residual evaluation, Jacobian construction, `J.T @ J`, dense solve, line search residual calls, stats packing, and I/O.
- Measure first-call compile time separately from steady-state runtime.

Expected experiment:

- `exp_name=fk_dense_baseline`
- Record path: `docs/efficiency_optimization_experiment/{ID}_fk_dense_baseline_record.md`
- Log-data path: `docs/efficiency_optimization_experiment/{ID}_fk_dense_baseline_logs_data/`

Go decision:

- If dense Jacobian plus dense solve dominates, proceed to matrix-free FK.
- If another component dominates, optimize that component first.

### P2: Low-Risk Runtime Cleanup

Goal:

- Remove known avoidable overhead before heavy algorithm work.

Tasks:

- Use `XLA_PYTHON_CLIENT_PREALLOCATE=false` in all experiment launch commands.
- Keep `pose_save_every` sparse by default.
- Avoid compressed centerline writes inside tight loops unless the run is explicitly diagnostic.
- Run medium-length float32 versus float64 comparisons.
- Keep JAX PyTree shapes stable where possible.
- Avoid expensive final Jacobian stats unless explicitly requested.

Success standard:

- Runtime improves by 2-5x on comparable cases.
- Final tip error, raw equilibrium residual, and shape quality remain acceptable.

### P3: Matrix-Free FK

Goal:

- Stop building and solving dense LM systems as the default FK path.

Tasks:

- Implement a matrix-free linear operator for `J v` with JVP.
- Implement a matrix-free adjoint operator for `J.T v` with VJP.
- Solve damped normal or augmented systems with CG, LSMR, GMRES, or MINRES depending on conditioning.
- Add a simple preconditioner based on diagonal, block diagonal, or previous-step curvature.
- Keep the current dense path as a correctness fallback.
- Compare matrix-free and dense FK on identical seeds and controls.

Success standard:

- Per-LM-step FK cost decreases by 5-10x on representative cases.
- Final equilibrium residual and task-point geometry match dense FK within tolerance.
- Memory pressure drops enough to avoid GPU memory saturation on stress cases.

### P4: Implicit Adjoint Sensitivity

Goal:

- Make IK sensitivity computation cheap and physically consistent.

Tasks:

- Use the equilibrium condition `E(z, I, x) = 0`.
- Compute sensitivities through implicit differentiation instead of repeated FK finite differences.
- Avoid requiring full stored `J` in `lm_stats`.
- Include protrusion or `x` sensitivity in the same framework.
- Reuse matrix-free FK operators where possible.

Success standard:

- IK outer linearization no longer performs multiple full FK solves per small perturbation.
- Sensitivity cost is predictable and substantially lower than the old finite-difference path.

### P5: Reduced-Space IK

Goal:

- Make the main IK path optimize physical controls while preserving strict equilibrium.

Tasks:

- Define the reduced variables, likely currents `I` and protrusion or segment parameter `x`.
- Use FK as an inner equilibrium solve with warm start.
- Use trust-region or line-search globalization.
- Enforce current and protrusion bounds in physical variables instead of relying on saturated parameter transforms.
- Track current utilization and saturation explicitly.
- Keep single-layer penalty IK as a fallback for comparison.

Success standard:

- Representative near targets solve reliably.
- Stress targets either converge or fail with clear reachability diagnostics.
- Equilibrium residual does not degrade while tip error improves.

### P6: Coarse-To-Fine And Continuation

Goal:

- Reduce iterations and avoid poor local branches.

Tasks:

- Start with coarse `M_list` to find a basin.
- Prolong the coarse solution to finer mesh.
- Use load continuation for magnetic field or current magnitude when needed.
- Use target continuation for far target jumps.
- Use tighter tolerances only in final refinement.

Success standard:

- Typical total IK iteration count falls below 50-80.
- Fine solve starts close enough that few expensive iterations are needed.
- Branch jumps and unnatural shapes are reduced.

### P7: Regression Suite

Goal:

- Prove the optimized solver is fast and stable beyond one hand-picked case.

Tasks:

- Build at least 10 extreme but meaningful targets.
- Include mixed magnetization:

```text
m_body_list = "0,0,-0.005301;0,0,0.005301;0.005301,0,0"
```

- Include near, diagonal, lateral, high-curvature, boundary, and negative-axis targets.
- Run optimized solver and dense or previous solver baselines when feasible.
- Plot convergence, target point, start point, final point, and pose curves.

Success standard:

- p50 runtime approaches 1 minute if algorithmic changes perform as expected.
- p95 or hard cases stay within about 3 minutes if reachable.
- All successful cases meet physical quality tolerances.
- Failures are classified as reachability, conditioning, numerical, or implementation failures.

## 8. First Formal Experiment Design

The first large experiment should be a profiling baseline rather than an optimization experiment.

Experiment name:

```text
fk_dense_baseline
```

Cases:

| Case Group | Configuration | Purpose |
| --- | --- | --- |
| FK micro | `M_list=3,3,3`, `5,5,5`, `8,8,8` | Measure scaling of dense FK |
| FK repeated call | Same shape, changed currents | Separate compile, warm-start, and steady-state behavior |
| FK repeated protrusion | Same shape if possible, changed `x` or protrusion | Detect recompilation and finite-difference cost |
| IK macro near | Near target, low outer cap | Measure overhead for easy case |
| IK macro stress | Far or diagonal target, higher outer cap | Measure FK call amplification in hard case |

Required timing components:

- Residual evaluation time.
- Jacobian construction time.
- `J.T @ J` construction time.
- Linear solve time.
- Line-search residual time.
- Stats and diagnostics time.
- I/O time if measurable.
- Compile time or first-call time.
- Steady-state time excluding compile.

Dense FK bottleneck is confirmed if:

- `jacobian + J.T @ J + linear solve` is at least 70% of steady-state FK time.
- FK time is at least 80% of IK wall time.
- FK per-step time grows strongly as `M_list` grows.
- Outer 9D IK solve time is small compared with FK time.

If these conditions are not met, the next experiment should target the actual dominant component instead of starting matrix-free FK immediately.

## 9. Engineering Rules For This Track

- Every large efficiency experiment must use one ID: `eff_opt_exp_YYYYMMDDHHMMSS`.
- Every large efficiency experiment must create one record file and one log-data directory using the same ID.
- Records and logs for this track must stay under `docs/efficiency_optimization_experiment/`.
- Every large experiment must record wall-clock runtime.
- The next experiment's wait estimate should use the previous comparable runtime.
- During long waits, keep output quiet and poll at low frequency.
- After each large experiment, commit relevant changes locally on branch `dev`.
- Never push.
- Do not commit unrelated large generated data unless it is intentionally part of the experiment record.

## 10. Risk Register

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Tip definition mismatch | Fast solver optimizes wrong point | Resolve before major solver work |
| Matrix-free solve instability | FK may fail despite lower memory | Keep dense fallback and compare residuals |
| Float32 numerical drift | Silent quality regression | Use representative float32/float64 benchmarks |
| Coarse mesh branch mismatch | Fine refinement starts on wrong branch | Add shape and residual checks before prolongation |
| Saturated current transform | Vanishing derivatives and slow IK | Prefer physical bounds and trust-region steps |
| Over-logging | Runtime dominated by I/O | Sparse saves by default |
| Recompilation | Misleading runtime measurements | Separate compile and steady-state timing |
| Overfitting to a small suite | Poor generalization | Use at least 10 diverse stress cases |

## 11. Expected Milestones

- End of Week 1: exact bottleneck known; tip and magnet convention issues documented or fixed.
- End of Week 2: low-risk runtime cleanup complete; a representative medium case has float32/float64 evidence.
- End of Week 4: matrix-free FK prototype produces comparable equilibrium solutions.
- End of Week 5: IK no longer depends on expensive finite-difference FK loops for core sensitivities.
- End of Week 6: reduced-space IK can solve representative cases with strict equilibrium.
- End of Week 7: coarse-to-fine strategy reduces iteration counts on hard targets.
- End of Week 8: benchmark suite reports whether 1 minute is realistic and whether 3 minutes is reached for hard cases.

## 12. Immediate Next Step

Start the first large experiment:

```text
exp_name = fk_dense_baseline
```

The experiment should be launched only after creating:

```text
docs/efficiency_optimization_experiment/{eff_opt_exp_YYYYMMDDHHMMSS}_fk_dense_baseline_record.md
docs/efficiency_optimization_experiment/{eff_opt_exp_YYYYMMDDHHMMSS}_fk_dense_baseline_logs_data/
```

The first experiment's conclusion should decide whether the next round is:

- Matrix-free FK implementation.
- I/O and diagnostics cleanup.
- Magnetic interpolation optimization.
- IK sensitivity restructuring.
- Initialization and continuation improvement.

