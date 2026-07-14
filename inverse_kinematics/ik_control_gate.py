"""Control-channel gating for supervised IK outer updates."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any, Iterable


PRODUCTION_CLASS = "可生产使用"
LIMITED_CLASS = "可有限使用"
LIMITED_DECISION_CLASS = "有限使用候选"
BASIN_CLASS = "入盆可用"
SINGLE_STEP_CLASS = "单步可用"
WEAK_RESPONSE_CLASS = "弱响应"
BLOCKED_ROLE = "禁用"
REFINE_ROLE = "精修可用"
BASIN_ROLE = "入盆可用"


@dataclass(frozen=True)
class ControlGate:
    """Machine-readable control-channel gate used by the IK outer loop."""

    enabled: bool
    source: str | None
    key: str | None
    missing_policy: str
    allow_limited: bool
    allowed_mask: tuple[bool, ...]
    basin_mask: tuple[bool, ...]
    channel_roles: tuple[str, ...]
    channel_reasons: tuple[str, ...]
    safe_step_abs_max: tuple[float | None, ...]
    basin_step_abs_max: tuple[float | None, ...]
    matched_rows: int

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "enabled": bool(self.enabled),
            "source": self.source,
            "key": self.key,
            "missing_policy": self.missing_policy,
            "allow_limited": bool(self.allow_limited),
            "allowed_mask": [bool(v) for v in self.allowed_mask],
            "basin_mask": [bool(v) for v in self.basin_mask],
            "channel_roles": list(self.channel_roles),
            "allowed_channels": channel_names_from_mask(self.allowed_mask),
            "basin_channels": channel_names_from_mask(self.basin_mask),
            "blocked_channels": channel_names_from_mask(tuple(not v for v in self.allowed_mask)),
            "channel_reasons": list(self.channel_reasons),
            "safe_step_abs_max": [None if v is None else float(v) for v in self.safe_step_abs_max],
            "basin_step_abs_max": [None if v is None else float(v) for v in self.basin_step_abs_max],
            "matched_rows": int(self.matched_rows),
        }


def channel_names_from_mask(mask: Iterable[bool]) -> list[str]:
    """Return control channel names from a boolean mask."""
    values = list(mask)
    names: list[str] = []
    for idx, flag in enumerate(values):
        if not flag:
            continue
        if idx == len(values) - 1:
            names.append("x")
        else:
            names.append(f"I{idx}")
    return names


def _default_allowed_mask(n_coils: int, missing_policy: str) -> list[bool]:
    if missing_policy == "allow_all":
        return [True] * (int(n_coils) + 1)
    if missing_policy == "allow_x_only":
        return [False] * int(n_coils) + [True]
    if missing_policy == "block_all":
        return [False] * (int(n_coils) + 1)
    raise ValueError("missing_policy must be 'block_all', 'allow_x_only', or 'allow_all'")


def _default_reason(channel_idx: int, n_coils: int, missing_policy: str) -> str:
    name = "x" if channel_idx == int(n_coils) else f"I{channel_idx}"
    return f"{name}:missing_policy:{missing_policy}"


def _row_matches_key(row: dict[str, str], key: str) -> bool:
    run = str(row.get("run", row.get("point_key", ""))).strip()
    run_norm = _canonical_gate_key(run)
    key_norm = _canonical_gate_key(str(key).strip())
    return bool(run_norm == key_norm or run_norm.startswith(f"{key_norm}_"))


def _canonical_gate_key(value: str) -> str:
    """Normalize point-key variants such as A1_pt13 and A1_pt_13."""
    return re.sub(r"_pt_?(\d+)", lambda match: f"_pt{int(match.group(1)):02d}", str(value).strip())


def _row_channel_index(row: dict[str, str], n_coils: int) -> int | None:
    kind = str(row.get("kind", "")).strip()
    if kind == "x":
        return int(n_coils)
    if kind == "I":
        try:
            idx = int(str(row.get("channel", "")).strip())
        except ValueError:
            return None
        if 0 <= idx < int(n_coils):
            return idx
    return None


def _is_allowed_class(classification: str, *, allow_limited: bool) -> bool:
    classification = str(classification).strip()
    if classification == PRODUCTION_CLASS:
        return True
    if allow_limited and classification in (LIMITED_CLASS, LIMITED_DECISION_CLASS):
        return True
    return False


def _is_basin_class(row: dict[str, str], *, channel_idx: int, n_coils: int, allow_limited: bool) -> bool:
    """Return whether an audited channel can be used during basin seeking."""
    labels = {
        str(row.get("classification", "")).strip(),
        str(row.get("decision", "")).strip(),
        str(row.get("role", "")).strip(),
        str(row.get("stage_role", "")).strip(),
    }
    if any(_is_allowed_class(label, allow_limited=allow_limited) for label in labels):
        return True
    if BASIN_CLASS in labels or SINGLE_STEP_CLASS in labels:
        return True
    if channel_idx == int(n_coils) and WEAK_RESPONSE_CLASS in labels:
        return True
    return False


def _row_classification(row: dict[str, str]) -> str:
    """Return a channel classification from detailed or summary audit rows."""
    return str(row.get("classification", row.get("decision", ""))).strip()


def _row_safe_step_abs(row: dict[str, str]) -> float | None:
    """Return the conservative per-channel safe step from an audit row."""
    values: list[float] = []
    for key in (
        "safe_step_abs",
        "safe_step_abs_max",
        "basin_safe_step_abs_max",
        "basin_step_abs_max",
        "limited_safe_step_abs_max",
        "limited_safe_step_abs_min",
    ):
        raw = str(row.get(key, "")).strip()
        if not raw:
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        if math.isfinite(value) and value > 0.0:
            values.append(value)
    if not values:
        return None
    return float(min(values))


def load_control_gate(
    path: str | Path | None,
    *,
    key: str | None,
    n_coils: int = 8,
    allow_limited: bool = True,
    missing_policy: str = "block_all",
) -> ControlGate:
    """Load a control-channel gate from a channel-classification CSV file.

    Channel indices in the CSV are zero-based for coil currents. The last
    control channel is the insertion coordinate ``x``.
    """
    n_coils = int(n_coils)
    if path is None or str(path).strip() == "":
        allowed = tuple([True] * (n_coils + 1))
        return ControlGate(
            enabled=False,
            source=None,
            key=key,
            missing_policy=missing_policy,
            allow_limited=bool(allow_limited),
            allowed_mask=allowed,
            basin_mask=allowed,
            channel_roles=tuple(REFINE_ROLE for _ in allowed),
            channel_reasons=tuple("gate_disabled" for _ in allowed),
            safe_step_abs_max=tuple(None for _ in allowed),
            basin_step_abs_max=tuple(None for _ in allowed),
            matched_rows=0,
        )
    if key is None or str(key).strip() == "":
        raise ValueError("--channel_gate_key is required when --channel_gate_file is provided")

    gate_path = Path(path).resolve()
    if not gate_path.exists():
        raise FileNotFoundError(str(gate_path))
    if gate_path.suffix.lower() != ".csv":
        raise ValueError("Only CSV channel gate files are currently supported")

    matched: list[dict[str, str]] = []
    with gate_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if _row_matches_key(row, str(key)):
                matched.append(row)

    allowed = _default_allowed_mask(n_coils, missing_policy)
    basin_allowed = _default_allowed_mask(n_coils, missing_policy)
    reasons = [_default_reason(idx, n_coils, missing_policy) for idx in range(n_coils + 1)]
    roles = [REFINE_ROLE if flag else BLOCKED_ROLE for flag in allowed]
    safe_steps: list[float | None] = [None for _ in range(n_coils + 1)]
    basin_safe_steps: list[float | None] = [None for _ in range(n_coils + 1)]
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in matched:
        idx = _row_channel_index(row, n_coils)
        if idx is None:
            continue
        grouped.setdefault(idx, []).append(row)

    for idx, rows in grouped.items():
        classes = [_row_classification(row) for row in rows]
        allowed_count = sum(1 for cls in classes if _is_allowed_class(cls, allow_limited=allow_limited))
        basin_count = sum(
            1
            for row in rows
            if _is_basin_class(row, channel_idx=idx, n_coils=n_coils, allow_limited=allow_limited)
        )
        if rows and allowed_count == len(rows):
            allowed[idx] = True
            basin_allowed[idx] = True
            roles[idx] = REFINE_ROLE
            reasons[idx] = f"audited_allowed:{'|'.join(sorted(set(classes)))}"
            row_safe_steps = [value for row in rows if (value := _row_safe_step_abs(row)) is not None]
            safe_steps[idx] = min(row_safe_steps) if row_safe_steps else None
            basin_safe_steps[idx] = safe_steps[idx]
        elif rows and basin_count == len(rows):
            allowed[idx] = False
            basin_allowed[idx] = True
            roles[idx] = BASIN_ROLE
            reasons[idx] = f"audited_basin_only:{'|'.join(sorted(set(classes)))}"
            row_safe_steps = [value for row in rows if (value := _row_safe_step_abs(row)) is not None]
            basin_safe_steps[idx] = min(row_safe_steps) if row_safe_steps else None
        else:
            allowed[idx] = False
            basin_allowed[idx] = False
            roles[idx] = BLOCKED_ROLE
            reasons[idx] = f"audited_blocked:{'|'.join(sorted(set(classes)))}"
            safe_steps[idx] = None
            basin_safe_steps[idx] = None

    return ControlGate(
        enabled=True,
        source=str(gate_path),
        key=str(key),
        missing_policy=str(missing_policy),
        allow_limited=bool(allow_limited),
        allowed_mask=tuple(bool(v) for v in allowed),
        basin_mask=tuple(bool(v) for v in basin_allowed),
        channel_roles=tuple(roles),
        channel_reasons=tuple(reasons),
        safe_step_abs_max=tuple(safe_steps),
        basin_step_abs_max=tuple(basin_safe_steps),
        matched_rows=len(matched),
    )
