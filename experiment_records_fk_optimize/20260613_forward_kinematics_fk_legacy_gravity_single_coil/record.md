# forward_kinematics FK legacy rerun record

## Scope

- Entry scripts: `forward_kinematics/fk_example.py` and `forward_kinematics/fk.py`.
- Solver settings: `tol=1e-5`, `max_iter=6000`, `lm_damping=0.1`, `jac_method=fwd`.
- Geometry: `flex_lengths=0.041,0.03,0.03`, `rigid_lengths=0.003,0.003,0.003`, `M_list=5,5,5`.
- Initial pose: `p0=0,0,0`, `Q0=0.70710678,0,0.70710678,0`, `axis_body=0,0,1`.
- Gravity: `g_world=0,0,-9.81`.
- Single-coil case: coil 6 current `20 A`, current vector `0,0,0,0,0,20,0,0`.
- Single-magnet case: axial tip magnet, `m_body_list=0,0,0; 0,0,0; 0,0,0.005301`.

## Data retention

- `fk_example.py` cases retain `stdout.log`, `stderr.log`, and `command.txt`.
- `fk.py` callback cases retain `stdout.log`, `stderr.log`, `command.txt`, `final_solution.npz`, `final_centerline_samples.npz`, and per-20-iteration callback snapshots under `callback_snaps/`.

## Results

- `gravity_only`: converged, final residual `9.994e-06` at iteration 3549.
- `gravity_only_with_callback`: converged, final residual `9.994e-06` at iteration 3549; saved 178 callback NPZ/PNG snapshot pairs.
- `single_tip_axial_coil6_20A`: reached max iteration, final residual `2.317e-05` at iteration 6000; strict convergence flag false.
- `single_tip_axial_coil6_20A_with_callback`: reached max iteration, final residual `2.317e-05` at iteration 6000; strict convergence flag false; saved 301 callback NPZ/PNG snapshot pairs.

## Geometry check from `final_solution.npz`

- `gravity_only_with_callback`: final tip `[0.033677, 0.000000, -0.100115] m`; maximum absolute y displacement `0.000000 m`.
- `single_tip_axial_coil6_20A_with_callback`: final tip `[0.040585, -0.025629, -0.082302] m`; maximum absolute y displacement `0.025629 m`.

The single-coil result does not satisfy the expected qualitative check of staying in the xOz plane and lifting the tip upward. This suggests the coil index/sign, calibration-frame convention, or magnet/body-frame convention should be checked before using this result as the report figure.
