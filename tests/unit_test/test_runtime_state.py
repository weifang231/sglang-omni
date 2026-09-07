# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

from sglang_omni.utils.runtime_state import (
    RUNTIME_CONTROL_FILE_ENV,
    RUNTIME_METRICS_DIR_ENV,
    RuntimeStateChannel,
    bounded_float,
    bounded_int,
)
from sglang_omni.utils.runtime_timing import (
    RUNTIME_TIMING_DIR_ENV,
    flush_all_runtime_timing_recorders,
)


def test_bounded_runtime_control_values() -> None:
    assert bounded_int(3, minimum=1, maximum=4) == 3
    assert bounded_int(9, minimum=1, maximum=4) == 4
    assert bounded_int(True, minimum=1, maximum=4) is None
    assert bounded_int(1.5, minimum=1, maximum=4) is None
    assert bounded_float(-1, minimum=0, maximum=2) == 0
    assert bounded_float(float("nan"), minimum=0, maximum=2) is None


def test_runtime_state_channel_is_opt_in_and_atomic(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv(RUNTIME_METRICS_DIR_ENV, raising=False)
    monkeypatch.delenv(RUNTIME_CONTROL_FILE_ENV, raising=False)
    monkeypatch.delenv(RUNTIME_TIMING_DIR_ENV, raising=False)
    disabled = RuntimeStateChannel(engine="test", component="scheduler")
    assert disabled.enabled is False
    assert disabled.maybe_publish(lambda: {"running_requests": 1}) is False

    metrics_dir = tmp_path / "metrics"
    control_file = tmp_path / "control.json"
    timing_dir = tmp_path / "timing"
    monkeypatch.setenv(RUNTIME_METRICS_DIR_ENV, str(metrics_dir))
    monkeypatch.setenv(RUNTIME_CONTROL_FILE_ENV, str(control_file))
    monkeypatch.setenv(RUNTIME_TIMING_DIR_ENV, str(timing_dir))
    channel = RuntimeStateChannel(
        engine="sglang-omni",
        component="scheduler",
        stage_id="talker/ar",
        publish_interval_s=0,
        control_poll_interval_s=0,
    )

    assert channel.maybe_publish(lambda: {"running_requests": 2}) is True
    snapshot = json.loads(channel.snapshot_path.read_text())
    assert snapshot["running_requests"] == 2
    assert snapshot["stage_id"] == "talker/ar"
    assert list(metrics_dir.glob("*.tmp")) == []

    control_file.write_text('{"generation_cap":3}', encoding="utf-8")
    assert channel.read_control_if_changed() == {"generation_cap": 3}
    assert channel.read_control_if_changed() is None
    control_file.unlink()
    assert channel.read_control_if_changed() == {}

    channel.record_timing_ns("unit_loop_ns", 1234)
    flush_all_runtime_timing_recorders()
    timing_files = list(timing_dir.glob("*.timing.json"))
    assert len(timing_files) == 1
    timing = json.loads(timing_files[0].read_text())
    assert timing["stage_id"] == "talker/ar"
    assert timing["families"]["runtime_publish_ns"]
    assert timing["families"]["runtime_control_read_ns"]
    assert timing["families"]["unit_loop_ns"] == [1234]
    assert list(timing_dir.glob("*.tmp")) == []
