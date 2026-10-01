# SPDX-License-Identifier: Apache-2.0
"""Coordinator submit-dispatch cost of the admission policy hook.

Measures only the coordinator (fake control plane, no IPC, no model). Each mode
submits ``--requests`` requests after ``--warmup`` discarded ones and records the
wall time of ``submit_request``; the completion is fed back immediately so the
in-flight set stays at one. Modes are shuffled per round so clock drift and
GC land on all of them alike.

    python -m benchmarks.eval.benchmark_admission_overhead
    python -m benchmarks.eval.benchmark_admission_overhead --base-ref <sha>

``--base-ref`` additionally measures the coordinator of another git revision
(for example upstream main before the hook) loaded from ``git show``, so a PR
can report base / unset / always-admit side by side.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import logging
import random
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
COORDINATOR_PATH = "sglang_omni/pipeline/coordinator.py"


class AlwaysAdmit:
    """The cheapest possible policy: a method call that returns True."""

    def admit(self, request_id, request):
        return True

    def completed(self, request_id):
        return None

    aborted = completed


class RejectAll(AlwaysAdmit):
    def admit(self, request_id, request):
        return False


def load_coordinator_from_ref(ref: str):
    """Import ``Coordinator`` of another git revision next to the current tree."""
    source = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "show", f"{ref}:{COORDINATOR_PATH}"], text=True
    )
    directory = tempfile.mkdtemp(prefix="admission-overhead-")
    path = Path(directory) / "coordinator_at_ref.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("coordinator_at_ref", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Coordinator


async def measure(coordinator_cls, policy, requests: int, warmup: int, reject: bool):
    from sglang_omni.admission import AdmissionRejectedError
    from sglang_omni.proto import CompleteMessage
    from tests.unit_test.fixtures.pipeline_fakes import RecordingCoordinatorControlPlane

    options = {"admission_policy": policy} if policy is not None else {}
    coordinator = coordinator_cls(
        "inproc://complete", "inproc://abort", "entry", **options
    )
    coordinator.control_plane.close()
    coordinator.control_plane = RecordingCoordinatorControlPlane()
    coordinator.register_stage("entry", "inproc://entry")
    samples = []
    for index in range(requests + warmup):
        request_id = str(index)
        started = time.perf_counter_ns()
        if reject:
            try:
                await coordinator.submit_request(request_id, "hello")
            except AdmissionRejectedError:
                pass
            elapsed = time.perf_counter_ns() - started
        else:
            await coordinator.submit_request(request_id, "hello")
            elapsed = time.perf_counter_ns() - started
            await coordinator.handle_completion(
                CompleteMessage(request_id, "entry", True, result={})
            )
            coordinator.completion_futures.pop(request_id, None)
            coordinator.control_plane.submitted.clear()
        if index >= warmup:
            samples.append(elapsed / 1000)
    ordered = sorted(samples)
    summary = {
        f"p{p}_microseconds": ordered[int((len(ordered) - 1) * p / 100)]
        for p in (50, 95, 99)
    }
    summary["dispatches_per_second"] = 1e6 / statistics.mean(samples)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--requests", type=int, default=10000)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument(
        "--base-ref", default=None, help="git revision to measure as 'base'"
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="write the full report as JSON"
    )
    parser.add_argument("--seed", type=int, default=20261001)
    args = parser.parse_args()
    sys.path.insert(0, str(REPO_ROOT))
    from sglang_omni.pipeline.coordinator import Coordinator

    # The coordinator logs one warning per rejection (as it does for the in-flight cap);
    # keep that out of the timed loop.
    logging.getLogger("sglang_omni.pipeline.coordinator").setLevel(logging.ERROR)

    modes = {}
    if args.base_ref:
        modes["base"] = (load_coordinator_from_ref(args.base_ref), None, False)
    modes["unset"] = (Coordinator, None, False)
    modes["always_admit"] = (Coordinator, AlwaysAdmit(), False)
    modes["reject_all"] = (Coordinator, RejectAll(), True)
    results = {name: [] for name in modes}
    rng = random.Random(args.seed)
    for _ in range(args.rounds):
        order = list(modes)
        rng.shuffle(order)
        for name in order:
            cls, policy, reject = modes[name]
            results[name].append(
                asyncio.run(measure(cls, policy, args.requests, args.warmup, reject))
            )
    medians = {
        name: {
            metric: statistics.median(r[metric] for r in rounds) for metric in rounds[0]
        }
        for name, rounds in results.items()
    }
    report = {
        "base_ref": args.base_ref,
        "python": sys.version,
        "requests_per_round": args.requests,
        "warmup_per_round": args.warmup,
        "rounds": args.rounds,
        "scope": "coordinator submit dispatch with a fake control plane; no IPC, completion or model cost",
        "results": results,
        "median_across_rounds": medians,
    }
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("| mode | p50 (us) | p95 (us) | p99 (us) | dispatches/s |")
    print("|---|---:|---:|---:|---:|")
    for name, m in medians.items():
        print(
            f"| {name} | {m['p50_microseconds']:.2f} | {m['p95_microseconds']:.2f} | "
            f"{m['p99_microseconds']:.2f} | {m['dispatches_per_second']:,.0f} |"
        )


if __name__ == "__main__":
    main()
