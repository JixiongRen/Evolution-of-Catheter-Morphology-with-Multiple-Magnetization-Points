from __future__ import annotations

import os
import warnings

import jax


_FALSE_VALUES = {"0", "false", "no", "off"}


def configure_jax_enable_x64(env_var: str = "MCRS_JAX_ENABLE_X64", *, default: bool = True) -> bool:
    """Configure JAX x64 mode from an environment variable.

    Args:
        env_var: Environment variable that controls x64 mode.
        default: Value used when the variable is unset or empty.

    Returns:
        The configured x64 state.
    """

    raw = os.environ.get(env_var)
    enabled = bool(default) if raw is None or raw.strip() == "" else raw.strip().lower() not in _FALSE_VALUES
    jax.config.update("jax_enable_x64", enabled)
    if not enabled:
        warnings.filterwarnings(
            "ignore",
            message="Explicitly requested dtype .* is not available.*",
            category=UserWarning,
        )
    return enabled
