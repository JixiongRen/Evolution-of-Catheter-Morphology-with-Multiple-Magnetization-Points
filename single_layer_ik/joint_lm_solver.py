from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp


jax.config.update("jax_enable_x64", True)
Array = jnp.ndarray


@dataclass
class JointLMStats:
    y_star: Array
    ok: bool
    normR: float
    cost: float
    lam: float
    stop_reason: str
    n_iter: int


class JointLMSolver:
    def __init__(
        self,
        *,
        residual_fn: Callable[[Array], Array],
        jac_method: str = "fwd",
        step_norm_clip: Optional[float] = 0.0,
        backtrack_max: int = 6,
        backtrack_factor: float = 0.5,
    ) -> None:
        if jac_method not in ("fwd", "rev"):
            raise ValueError("jac_method must be 'fwd' or 'rev'")

        self._residual = residual_fn
        self._residual_jit = jax.jit(residual_fn)

        if jac_method == "fwd":
            self._jac_jit = jax.jit(jax.jacfwd(residual_fn))
        else:
            self._jac_jit = jax.jit(jax.jacrev(residual_fn))

        self._step_norm_clip = float(step_norm_clip) if step_norm_clip is not None else 0.0
        self._backtrack_max = int(backtrack_max)
        self._backtrack_factor = float(backtrack_factor)

    def solve_lm(
        self,
        y0: Array,
        *,
        max_iter: int = 100,
        tol: float = 1e-6,
        lm_damping: float = 1e-3,
        lam_max: float = 1e10,
        gtol: float = 1e-12,
        xtol: float = 1e-12,
        max_damping_attempts: int = 12,
        verbose: bool = True,
        return_history: bool = True,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Tuple[Array, JointLMStats, Dict[str, Any]]:
        y = jnp.asarray(y0, dtype=jnp.float64).reshape(-1,)
        lam = float(lm_damping)

        R = self._residual_jit(y)
        normR = float(jnp.linalg.norm(R))
        cost = 0.5 * normR * normR

        history: Dict[str, Any] = {"outer": []}

        if verbose:
            print(f"[Joint-LM] iter=0 ||R||={normR:.3e} lam={lam:.3e}")

        if (not jnp.isfinite(normR)) or (not jnp.isfinite(cost)):
            stats = JointLMStats(
                y_star=jnp.array(y),
                ok=False,
                normR=float(normR),
                cost=float(cost),
                lam=float(lam),
                stop_reason="nan_init",
                n_iter=0,
            )
            return jnp.array(y), stats, history

        for it in range(int(max_iter)):
            if normR < float(tol):
                if verbose:
                    print(f"[Joint-LM] converged: iter={it} ||R||={normR:.3e}")
                stats = JointLMStats(
                    y_star=jnp.array(y),
                    ok=True,
                    normR=float(normR),
                    cost=float(cost),
                    lam=float(lam),
                    stop_reason="converged",
                    n_iter=int(it),
                )
                return jnp.array(y), stats, history

            J = self._jac_jit(y)
            g = jnp.transpose(J) @ R
            g_norm_inf = float(jnp.max(jnp.abs(g))) if g.size else 0.0
            if g_norm_inf < float(gtol):
                if verbose:
                    print(f"[Joint-LM] stop: small grad ||J^T R||_inf={g_norm_inf:.3e}")
                stats = JointLMStats(
                    y_star=jnp.array(y),
                    ok=False,
                    normR=float(normR),
                    cost=float(cost),
                    lam=float(lam),
                    stop_reason="small_grad",
                    n_iter=int(it),
                )
                return jnp.array(y), stats, history

            H = jnp.transpose(J) @ J
            diagH = jnp.clip(jnp.diag(H), a_min=1e-12)
            D = jnp.diag(diagH)

            step_accepted = False
            last_rho = 0.0
            last_pred = 0.0
            last_act = 0.0
            dy_norm = float("nan")

            for _ in range(int(max_damping_attempts)):
                A = H + float(lam) * D
                try:
                    dy = -jnp.linalg.solve(A, g)
                except Exception:
                    lam = min(float(lam) * 3.0, float(lam_max))
                    continue

                # optional step clipping (global 2-norm)
                dy_norm = float(jnp.linalg.norm(dy))
                if self._step_norm_clip and dy_norm > self._step_norm_clip:
                    scale = self._step_norm_clip / max(dy_norm, 1e-18)
                    dy = dy * scale
                    dy_norm = float(jnp.linalg.norm(dy))
                y_norm = float(jnp.linalg.norm(y))
                if dy_norm < float(xtol) * (float(xtol) + y_norm):
                    if verbose:
                        print(f"[Joint-LM] stop: tiny step ||dy||={dy_norm:.3e}")
                    stats = JointLMStats(
                        y_star=jnp.array(y),
                        ok=False,
                        normR=float(normR),
                        cost=float(cost),
                        lam=float(lam),
                        stop_reason="tiny_step",
                        n_iter=int(it),
                    )
                    return jnp.array(y), stats, history

                R_lin = R + J @ dy
                pred = 0.5 * float(jnp.dot(R, R) - jnp.dot(R_lin, R_lin))

                # try step and simple backtracking if needed
                y_new = y + dy
                R_new = self._residual_jit(y_new)
                normR_new = float(jnp.linalg.norm(R_new))
                cost_new = 0.5 * normR_new * normR_new

                attempt = 0
                while (not jnp.isfinite(normR_new)) or (not jnp.isfinite(cost_new)) or (cost_new >= cost):
                    if attempt >= self._backtrack_max:
                        break
                    dy = dy * self._backtrack_factor
                    y_new = y + dy
                    R_new = self._residual_jit(y_new)
                    normR_new = float(jnp.linalg.norm(R_new))
                    cost_new = 0.5 * normR_new * normR_new
                    attempt += 1

                act = float(cost - cost_new)
                rho = act / max(pred, 1e-18)

                last_rho, last_pred, last_act = float(rho), float(pred), float(act)

                if rho > 0.0 and cost_new < cost:
                    y, R, normR, cost = y_new, R_new, normR_new, cost_new
                    step_accepted = True

                    if rho > 0.75:
                        lam = max(float(lam) * 0.3, 1e-12)
                        status = "↓↓"
                    elif rho > 0.25:
                        lam = max(float(lam) * 0.5, 1e-12)
                        status = "↓"
                    else:
                        status = "→"

                    if verbose:
                        print(
                            f"[Joint-LM] iter={it+1} ||R||={normR:.3e} lam={lam:.3e} {status} "
                            f"rho={rho:.2f} pred={pred:.3e} act={act:.3e} ||dy||={dy_norm:.3e}"
                        )

                    break

                lam = min(float(lam) * 3.0, float(lam_max))

            if return_history:
                rec = {
                    "iter": int(it + 1),
                    "accepted": bool(step_accepted),
                    "lam": float(lam),
                    "rho": float(last_rho),
                    "pred": float(last_pred),
                    "act": float(last_act),
                    "dy_norm": float(dy_norm),
                    "normR": float(normR),
                    "cost": float(cost),
                }
                history["outer"].append(rec)

                if callback is not None:
                    callback({"y": y, **rec})

            if not step_accepted:
                if verbose:
                    print(f"[Joint-LM] iter={it+1} reject, lam={lam:.3e}, ||R||={normR:.3e}")
                if lam >= float(lam_max) * 0.999:
                    stats = JointLMStats(
                        y_star=jnp.array(y),
                        ok=False,
                        normR=float(normR),
                        cost=float(cost),
                        lam=float(lam),
                        stop_reason="lam_max",
                        n_iter=int(it + 1),
                    )
                    return jnp.array(y), stats, history

        if verbose:
            print(f"[Joint-LM] max_iter reached: ||R||={normR:.3e}")

        stats = JointLMStats(
            y_star=jnp.array(y),
            ok=False,
            normR=float(normR),
            cost=float(cost),
            lam=float(lam),
            stop_reason="max_iter",
            n_iter=int(max_iter),
        )
        return jnp.array(y), stats, history
