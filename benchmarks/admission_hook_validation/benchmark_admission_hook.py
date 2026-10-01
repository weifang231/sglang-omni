"""Compare coordinator-only dispatch against the PR base; no model or IPC latency."""

import argparse
import asyncio
import importlib.util
import json
import random
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--base", default="3a52817258b63d772c3e3c19dc9f1f234972b4cd")
    parser.add_argument("--requests", type=int, default=10000)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    sys.path.insert(0, str(arguments.repo.resolve()))
    from sglang_omni.pipeline.coordinator import Coordinator
    from sglang_omni.proto import CompleteMessage
    from tests.unit_test.fixtures.pipeline_fakes import RecordingCoordinatorControlPlane

    class AlwaysAdmit:
        def admit(self, request_id, request):
            return True

        def completed(self, request_id):
            pass

        aborted = completed

    async def measure(coordinator_type, policy, use_policy):
        options = {"admission_policy": policy} if use_policy else {}
        coordinator = coordinator_type(
            "inproc://complete", "inproc://abort", "entry", **options
        )
        coordinator.control_plane.close()
        coordinator.control_plane = RecordingCoordinatorControlPlane()
        coordinator.register_stage("entry", "inproc://entry")
        samples = []
        for index in range(arguments.requests + 1000):
            request_id = str(index)
            started = time.perf_counter_ns()
            await coordinator.submit_request(request_id, "hello")
            elapsed = time.perf_counter_ns() - started
            await coordinator.handle_completion(
                CompleteMessage(request_id, "entry", True, result={})
            )
            coordinator.completion_futures.pop(request_id)
            coordinator.control_plane.submitted.clear()
            if index >= 1000:
                samples.append(elapsed / 1000)
        ordered = sorted(samples)
        return {
            f"p{percentile}_microseconds": ordered[
                int((len(ordered) - 1) * percentile / 100)
            ]
            for percentile in (50, 95, 99)
        } | {"dispatches_per_second": 1e6 / statistics.mean(samples)}

    baseline_source = subprocess.check_output(
        [
            "git",
            "-C",
            str(arguments.repo),
            "show",
            f"{arguments.base}:sglang_omni/pipeline/coordinator.py",
        ],
        text=True,
    )
    with tempfile.TemporaryDirectory() as temporary_directory:
        baseline_file = Path(temporary_directory) / "baseline_coordinator.py"
        baseline_file.write_text(baseline_source)
        specification = importlib.util.spec_from_file_location(
            "baseline_coordinator", baseline_file
        )
        baseline_module = importlib.util.module_from_spec(specification)
        sys.modules[specification.name] = baseline_module
        specification.loader.exec_module(baseline_module)
        modes = {
            "base": (baseline_module.Coordinator, None, False),
            "unset": (Coordinator, None, True),
            "always_admit": (Coordinator, AlwaysAdmit(), True),
        }
        results = {name: [] for name in modes}
        random_generator = random.Random(20261001)
        for _ in range(arguments.rounds):
            order = list(modes)
            random_generator.shuffle(order)
            for name in order:
                results[name].append(asyncio.run(measure(*modes[name])))
        report = {
            "base_commit": arguments.base,
            "python": sys.version,
            "requests_per_round": arguments.requests,
            "rounds": arguments.rounds,
            "scope": "Fake control plane, submit dispatch only; excludes model, real IPC and completion cost.",
            "results": results,
            "median_across_rounds": {
                name: {
                    metric: statistics.median(result[metric] for result in rounds)
                    for metric in rounds[0]
                }
                for name, rounds in results.items()
            },
        }
        arguments.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report["median_across_rounds"], indent=2))


if __name__ == "__main__":
    main()
