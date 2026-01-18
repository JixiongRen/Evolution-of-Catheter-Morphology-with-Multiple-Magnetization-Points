import numpy as np
import inspect

# ========== robust imports aligned with your project ==========
# Prefer nondim re-export module, fallback to external_wrench directly.
try:
    from pose_modules import external_wrench_nondim as ew
    SRC = "pose_modules.external_wrench_nondim"
except Exception:
    from pose_modules import external_wrench as ew
    SRC = "pose_modules.external_wrench"

# quat_to_rotmat is imported inside ew (in your external_wrench.py it is from .segments)
# For the test we will directly call ew.quat_to_rotmat via that import path.
quat_to_rotmat = ew.quat_to_rotmat

GravityLineDensity = ew.GravityLineDensity
make_external_wrench_density_flexible = ew.make_external_wrench_density_flexible


# ========== small utilities ==========
def random_unit_quat(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=4)
    q = q / np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    return q

def quat_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float).reshape(3)
    axis = axis / np.linalg.norm(axis)
    half = 0.5 * angle_rad
    w = np.cos(half)
    xyz = axis * np.sin(half)
    q = np.array([w, xyz[0], xyz[1], xyz[2]], dtype=float)
    return q / np.linalg.norm(q)

def make_state(p: np.ndarray, Q: np.ndarray) -> np.ndarray:
    # x = [p(3), Q(4), f(3), tau(3)]
    p = np.asarray(p, dtype=float).reshape(3)
    Q = np.asarray(Q, dtype=float).reshape(4)
    Q = Q / np.linalg.norm(Q)
    return np.concatenate([p, Q, np.zeros(3), np.zeros(3)], axis=0)

def assert_allclose(a, b, tol=1e-10, name=""):
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    err = np.max(np.abs(a - b))
    if err > tol:
        raise AssertionError(
            f"[FAIL] {name} max|diff|={err:.3e} > tol={tol:.3e}\n"
            f"  a={a}\n"
            f"  b={b}"
        )
    print(f"[PASS] {name} max|diff|={err:.3e}")


# ========== tests ==========
def test_gravity_is_world_invariant():
    """
    With your updated world-frame definition:
      gravity.force_world() returns rhoA * g_vec (N/m) in WORLD.
    So fext_density(x,Q) should be independent of Q.
    """
    rhoA = 2.0
    g_vec = np.array([0.0, 0.0, -1.0])
    gravity = GravityLineDensity(rhoA=rhoA, g_vec=g_vec)

    fext, tauext = make_external_wrench_density_flexible(
        gravity=gravity,
        magnetic_density_fun=None,
        magnetic_density_frame="world",
    )

    expected_f_world = gravity.force_world()  # rhoA*g_vec
    expected_tau_world = np.zeros(3)

    rng = np.random.default_rng(0)
    p = np.array([0.01, -0.02, 0.03])
    sigma = 0.123

    Q_list = [
        np.array([1.0, 0.0, 0.0, 0.0]),
        quat_from_axis_angle([1, 0, 0], np.deg2rad(90.0)),
    ] + [random_unit_quat(rng) for _ in range(20)]

    for k, Q in enumerate(Q_list):
        x = make_state(p, Q)
        f = fext(x, sigma)
        tau = tauext(x, sigma)

        assert_allclose(f, expected_f_world, tol=1e-10, name=f"gravity fext invariant (case {k})")
        assert_allclose(tau, expected_tau_world, tol=1e-10, name=f"gravity tauext invariant (case {k})")


def test_magnetic_density_frame_conversion_aligned():
    """
    Aligned to your implementation in external_wrench.py:
      if magnetic_density_frame=="body": f_world = R @ f_body (R = world_from_body)
      if magnetic_density_frame=="world": f_world = f_world (no rotation)
    """
    gravity = GravityLineDensity(rhoA=0.0, g_vec=np.array([0.0, 0.0, -1.0]))

    f_body_const = np.array([1.0, 2.0, 3.0])     # N/m in body
    tau_body_const = np.array([-1.0, 0.5, 0.0])  # N*m/m in body

    def mag_body(_x, _sigma):
        return f_body_const, tau_body_const

    f_world_const = np.array([4.0, -2.0, 1.0])
    tau_world_const = np.array([0.1, 0.2, 0.3])

    def mag_world(_x, _sigma):
        return f_world_const, tau_world_const

    rng = np.random.default_rng(1)
    Q = random_unit_quat(rng)
    x = make_state(np.zeros(3), Q)
    sigma = 0.5
    R = quat_to_rotmat(Q)  # world_from_body (per your file)

    # Case A: body -> world conversion should happen (R @ v_body)
    fext_A, tauext_A = make_external_wrench_density_flexible(
        gravity=gravity,
        magnetic_density_fun=mag_body,
        magnetic_density_frame="body",
    )
    fA = fext_A(x, sigma)
    tA = tauext_A(x, sigma)
    assert_allclose(fA, R @ f_body_const, tol=1e-10, name="mag body->world fext (R @ f_body)")
    assert_allclose(tA, R @ tau_body_const, tol=1e-10, name="mag body->world tauext (R @ tau_body)")

    # Case B: world should remain world (no Q dependence)
    fext_B, tauext_B = make_external_wrench_density_flexible(
        gravity=gravity,
        magnetic_density_fun=mag_world,
        magnetic_density_frame="world",
    )
    fB = fext_B(x, sigma)
    tB = tauext_B(x, sigma)
    assert_allclose(fB, f_world_const, tol=1e-10, name="mag world fext (invariant)")
    assert_allclose(tB, tau_world_const, tol=1e-10, name="mag world tauext (invariant)")


if __name__ == "__main__":
    print(f"Using module: {SRC}")
    print("make_external_wrench_density_flexible defined in:")
    try:
        print(inspect.getsource(make_external_wrench_density_flexible)[:400], "...\n")
    except Exception:
        print("(source not available via inspect)\n")

    print("Running aligned frame-consistency sanity checks...")
    test_gravity_is_world_invariant()
    test_magnetic_density_frame_conversion_aligned()
    print("All aligned sanity checks passed.")
