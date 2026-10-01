# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import random
import types

import pytest

from sglang_omni.admission import load_admission_policy
from sglang_omni.admission_policies import capacity_table as ct
from sglang_omni.admission_policies.fit_capacity_table import fit, main, read_events
from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.proto import CompleteMessage, OmniRequest, StreamMessage
from tests.unit_test.fixtures.pipeline_fakes import RecordingCoordinatorControlPlane


def profile_document(capacity=3, deadline=1.0, prices=(0.1, 0.5, 0.9), rates=None):
    features = []
    for n in range(capacity):
        # P[first_s <= 1.0] = 1 - n/capacity-ish: 10 rows, the first few slow
        rows = [{"first_s": 2.0 if i < 10 * n // capacity else 0.5} for i in range(10)]
        features.append(rows)
    doc = {
        "capacity": capacity,
        "kind": "asr",
        "deadline_s": deadline,
        "guard_s": 0.0,
        "features_by_occupancy": features,
        "prices": list(prices),
    }
    if rates is not None:
        doc["departure_rates"] = list(rates)
    else:
        pass
    return doc


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def request(**metadata) -> OmniRequest:
    return OmniRequest(inputs="hello", metadata=metadata)


# --- solver and profile ------------------------------------------------------------------------


def test_solve_prices_satisfies_the_average_reward_equations() -> None:
    q, mu = [0.95, 0.9, 0.7, 0.4], [10.0, 18.0, 24.0, 28.0]
    solved = ct.solve_prices(q, 30.0, mu)
    gains = []
    for n in range(len(q) + 1):
        value = mu[n - 1] * solved["prices"][n - 1] if n else 0.0
        if n < len(q):
            value += 30.0 * max(q[n] - solved["prices"][n], 0.0)
        gains.append(value)
    assert max(gains) - min(gains) < 1e-9
    assert solved["bellman_rate_spread"] < 1e-9
    with pytest.raises(ValueError):
        ct.solve_prices(q, 0.0, mu)


def test_prices_grow_with_the_operating_rate() -> None:
    mu = [10.0, 18.0, 24.0]
    base = ct.CapacityTableProfile.from_dict(profile_document(rates=mu))
    low, high = base.repriced(8.0), base.repriced(40.0)
    assert sum(high.prices) > sum(low.prices)
    assert high.repriced(8.0).prices == pytest.approx(low.prices)
    with pytest.raises(ValueError, match="cannot re-price"):
        ct.CapacityTableProfile.from_dict(profile_document()).repriced(10.0)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d.update(capacity=0), "capacity"),
        (lambda d: d.update(kind="omni"), "kind"),
        (lambda d: d.update(deadline_s=-1), "deadline_s"),
        (lambda d: d.update(prices=[0.1]), "prices"),
        (lambda d: d["features_by_occupancy"].__setitem__(0, []), "non-empty"),
        (lambda d: d.update(departure_rates=[1.0, 0.0, 2.0]), "departure_rates"),
    ],
)
def test_invalid_profiles_are_rejected(mutate, message) -> None:
    doc = profile_document()
    mutate(doc)
    with pytest.raises(ValueError, match=message):
        ct.CapacityTableProfile.from_dict(doc)


@pytest.mark.parametrize(
    "remaining_seconds, expected",
    [(0.1, 0.0), (0.6, 0.5), (0.85, 0.75), (2.1, 1.0), (float("nan"), 0.0)],
)
def test_probability_includes_equal_and_duplicate_latencies(
    remaining_seconds: float, expected: float
) -> None:
    document = profile_document(capacity=1, prices=(0.1,))
    document["guard_s"] = 0.1
    document["features_by_occupancy"] = [
        [{"first_s": seconds} for seconds in (2.0, 0.5, 0.75, 0.5)]
    ]
    profile = ct.CapacityTableProfile.from_dict(document)
    assert profile.probability(0, remaining_seconds) == expected
    assert profile.document == document


def test_decision_rule() -> None:
    profile = ct.CapacityTableProfile.from_dict(profile_document())
    assert profile.q_table() == pytest.approx([1.0, 0.7, 0.4])
    assert profile.decide(0, 1.0).admission == "admit"  # q 1.0 > price 0.1
    assert profile.decide(2, 1.0).reason == "deadline_or_price"  # q 0.4 < price 0.9
    assert profile.decide(1, 0.2).admission == "reject"  # budget below every first_s
    assert profile.decide(3, 1.0).reason == "capacity"
    assert profile.decide(0, 0.0).admission == "reject"


# --- policy ------------------------------------------------------------------------------------


def test_apply_mode_rejects_and_tracks_occupancy(tmp_path) -> None:
    clock = FakeClock()
    recorder = ct.EventRecorder(tmp_path / "events.jsonl")
    policy = ct.CapacityTablePolicy(
        ct.CapacityTableProfile.from_dict(profile_document()),
        mode="apply",
        recorder=recorder,
        clock=clock,
    )
    assert policy.admit("a", request()) is True  # occupancy 0
    assert policy.admit("b", request()) is True  # occupancy 1: q 0.7 > 0.5
    assert policy.admit("c", request()) is False  # occupancy 2: q 0.4 < 0.9
    clock.now += 0.3
    policy.first_output("a")
    clock.now += 0.2
    policy.completed("a")
    assert policy.admit("d", request()) is True  # back at occupancy 1
    policy.aborted("b")
    policy.aborted("unknown")  # ignored
    assert policy.snapshot()["in_flight"] == 1
    assert policy.counts == {"admitted": 3, "rejected": 1, "shadow_rejected": 0}
    # deadline from the request wins over the profile's
    assert policy.admit("late", request(remaining_budget_s=0.1)) is False
    policy.close()
    events = read_events(tmp_path / "events.jsonl")
    assert [e["event"] for e in events] == [
        "admit",
        "admit",
        "admit",
        "first_output",
        "completed",
        "admit",
        "aborted",
        "admit",
    ]
    assert events[3]["first_s"] == pytest.approx(0.3)
    assert events[4]["total_s"] == pytest.approx(0.5)
    assert events[2]["enforced"] is True and events[2]["admission"] == "reject"


def test_shadow_mode_decides_but_never_rejects() -> None:
    policy = ct.CapacityTablePolicy(
        ct.CapacityTableProfile.from_dict(profile_document()),
        mode="shadow",
        clock=FakeClock(),
    )
    assert all(policy.admit(str(i), request()) for i in range(5))
    assert policy.counts["shadow_rejected"] == 3 and policy.counts["rejected"] == 0


def test_record_mode_needs_no_profile_but_needs_a_path(tmp_path) -> None:
    with pytest.raises(ValueError, match="record_path"):
        ct.CapacityTablePolicy(None, mode="record")
    with pytest.raises(ValueError, match="needs a profile"):
        ct.CapacityTablePolicy(None, mode="apply")
    policy = ct.CapacityTablePolicy(
        None,
        mode="record",
        recorder=ct.EventRecorder(tmp_path / "e.jsonl"),
        clock=FakeClock(),
    )
    assert all(policy.admit(str(i), request()) for i in range(50))


# --- fit round trip ---------------------------------------------------------------------------


def simulate(path, *, requests=800, rate=20.0, service=0.04, seed=1):
    """Poisson arrivals into a stable box whose first-output delay grows mildly with occupancy."""
    rng = random.Random(seed)
    clock = FakeClock()
    policy = ct.CapacityTablePolicy(
        None, mode="record", recorder=ct.EventRecorder(path), clock=clock
    )
    pending = []  # (time, event, request_id)
    t = clock.now
    for i in range(requests):
        t += rng.expovariate(rate)
        pending.append((t, "arrive", str(i)))
    while pending:
        pending.sort()
        t, event, rid = pending.pop(0)
        clock.now = t
        if event == "arrive":
            occupancy = len(policy.in_flight)
            policy.admit(rid, request())
            delay = service * (1 + 0.15 * occupancy) * rng.uniform(0.6, 1.4)
            pending.append((t + delay, "first", rid))
            pending.append((t + delay + service * rng.uniform(0.5, 1.5), "done", rid))
        elif event == "first":
            policy.first_output(rid)
        else:
            policy.completed(rid)
    policy.close()


def test_fit_produces_a_usable_profile_and_cli_round_trips(tmp_path) -> None:
    events_path = tmp_path / "events.jsonl"
    simulate(events_path)
    events = read_events(events_path)
    document = fit(
        events, kind="asr", deadline_s=0.06, arrival_rate_rps=20.0, min_samples=10
    )
    profile = ct.CapacityTableProfile.from_dict(document)
    assert profile.capacity >= 3
    assert profile.q_table()[0] > profile.q_table()[-1]  # slower at higher occupancy
    assert all(r > 0 for r in profile.departure_rates)
    assert document["diagnostics"]["samples_by_occupancy"]["0"] >= 10
    with pytest.raises(ValueError, match="lack"):
        fit(events, kind="asr", deadline_s=0.06, arrival_rate_rps=20.0, capacity=64)
    out = tmp_path / "profile.json"
    assert (
        main(
            [
                str(events_path),
                "--kind",
                "asr",
                "--deadline-s",
                "0.25",
                "--arrival-rate-rps",
                "20",
                "--min-samples",
                "10",
                "--output",
                str(out),
            ]
        )
        == 0
    )
    assert json.loads(out.read_text())["capacity"] == profile.capacity
    # the fitted profile, applied, rejects at high occupancy and admits at low occupancy
    policy = ct.CapacityTablePolicy(profile, mode="apply", clock=FakeClock())
    assert policy.admit("x", request()) is True
    assert profile.decide(profile.capacity, 0.06).reason == "capacity"


# --- factory and coordinator integration ------------------------------------------------------


def config_with(**options) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        admission_policy=f"{ct.__name__}.make_policy", admission_policy_options=options
    )


def test_make_policy_validates_options(tmp_path) -> None:
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile_document(rates=[10.0, 18.0, 24.0])))
    with pytest.raises(ValueError, match="unknown keys"):
        ct.make_policy(config=config_with(bogus=1))
    with pytest.raises(ValueError, match="mode must be"):
        ct.make_policy(config=config_with(mode="dry-run"))
    with pytest.raises(ValueError, match="profile is required"):
        ct.make_policy(config=config_with(mode="apply"))
    with pytest.raises(FileNotFoundError):
        ct.make_policy(config=config_with(profile=str(tmp_path / "missing.json")))
    policy = load_admission_policy(
        config_with(mode="apply", profile=str(profile_path), arrival_rate_rps=40)
    )
    assert isinstance(policy, ct.CapacityTablePolicy)
    assert policy.profile.planning_arrival_rate_rps == 40.0
    recorder = load_admission_policy(
        config_with(mode="record", record_path=str(tmp_path / "events.jsonl"))
    )
    assert recorder.profile is None and recorder.mode == "record"
    recorder.close()


def test_policy_through_the_coordinator_hook(tmp_path) -> None:
    async def run() -> None:
        clock = FakeClock()
        policy = ct.CapacityTablePolicy(
            ct.CapacityTableProfile.from_dict(profile_document()),
            mode="apply",
            recorder=ct.EventRecorder(tmp_path / "events.jsonl"),
            clock=clock,
        )
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="asr",
            admission_policy=policy,
        )
        coordinator.control_plane = RecordingCoordinatorControlPlane()
        coordinator.register_stage("asr", "inproc://asr")

        queue: asyncio.Queue = asyncio.Queue()
        await coordinator.submit_request("a", "hello", stream_queue=queue)
        await coordinator.submit_request("b", "hello")
        from sglang_omni.admission import AdmissionRejectedError

        with pytest.raises(AdmissionRejectedError):
            await coordinator.submit_request("c", "hello")
        clock.now += 0.1
        await coordinator.handle_stream(StreamMessage("a", "asr", chunk={}, chunk_id=0))
        await coordinator.handle_completion(
            CompleteMessage("a", "asr", True, result={})
        )
        await coordinator.handle_completion(
            CompleteMessage("b", "asr", True, result={})
        )
        await coordinator.stop()
        events = read_events(tmp_path / "events.jsonl")
        kinds = [(e["event"], e["request_id"]) for e in events]
        assert ("first_output", "a") in kinds and ("completed", "b") in kinds
        assert next(e for e in events if e["event"] == "first_output")[
            "first_s"
        ] == pytest.approx(0.1)
        assert policy.in_flight == {}

    asyncio.run(run())
