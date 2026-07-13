"""Tests for audited IK control-channel gates."""

from __future__ import annotations

from pathlib import Path

from inverse_kinematics.ik_control_gate import load_control_gate


def test_control_gate_blocks_unaudited_channels(tmp_path: Path) -> None:
    table = tmp_path / "gate.csv"
    table.write_text(
        "\n".join(
            [
                "run,kind,channel,classification",
                "A1_pt13_ch3_4_best_accept_seg1,I,3,同分支误差超限",
                "A1_pt13_ch3_4_best_accept_seg1,I,4,同分支误差超限",
                "A1_pt13_ch3_4_best_accept_seg1,x,-1,可生产使用",
            ]
        ),
        encoding="utf-8",
    )

    gate = load_control_gate(table, key="A1_pt13", n_coils=8)

    assert gate.enabled
    assert gate.matched_rows == 3
    assert gate.allowed_mask == (False, False, False, False, False, False, False, False, True)
    assert "x" in gate.as_dict()["allowed_channels"]
    assert "I3" in gate.as_dict()["blocked_channels"]


def test_control_gate_limited_channels_are_configurable(tmp_path: Path) -> None:
    table = tmp_path / "gate.csv"
    table.write_text(
        "\n".join(
            [
                "run,kind,channel,classification",
                "A2_pt17_ch4_last_accept_seg1,I,4,可有限使用",
            ]
        ),
        encoding="utf-8",
    )

    allowed = load_control_gate(table, key="A2_pt17", n_coils=8, allow_limited=True)
    blocked = load_control_gate(table, key="A2_pt17", n_coils=8, allow_limited=False)

    assert allowed.allowed_mask[4]
    assert not blocked.allowed_mask[4]


def test_control_gate_disabled_allows_legacy_full_update() -> None:
    gate = load_control_gate(None, key=None, n_coils=8)

    assert not gate.enabled
    assert gate.allowed_mask == (True, True, True, True, True, True, True, True, True)
