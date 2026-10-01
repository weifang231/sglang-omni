# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

import numpy as np

RUN = Path(__file__).resolve().parent
OUTPUT = RUN / "tts-seeded-control"


def main():
    summaries = json.loads((OUTPUT / "summaries.json").read_text())
    requests = {}
    report = {
        "purpose": "Fixed-output-work diagnostic; greedy semantic and subtalker decoding, paired per-input seeds, max_new_tokens=32.",
        "seeds": [8201, 8202, 8203],
        "requests_per_mode": 384,
        "limitations": [
            "Models remain on fixed GPUs 3/4/5 in this supplement; primary campaign rotates GPUs.",
            "32-frame truncated outputs are a performance control, not an audio quality evaluation.",
            "Three independent arrival seeds do not establish statistical equivalence.",
        ],
        "modes": {},
    }
    for mode in ("base", "unset", "always"):
        requests[mode] = [
            request | {"arrival_seed": seed}
            for seed in (8201, 8202, 8203)
            for request in json.loads(
                (OUTPUT / f"tts-{mode}-seeded-{seed}-results.json").read_text()
            )["requests"]
        ]
        good = [request for request in requests[mode] if request["success"]]
        rounds = [summary for summary in summaries if summary["mode"] == mode]
        report["modes"][mode] = {
            "successes": len(good),
            "count": len(requests[mode]),
            "first_ms": {
                f"p{q}": float(
                    np.percentile([r["first_seconds"] for r in good], q) * 1000
                )
                for q in (50, 95, 99)
            },
            "total_ms": {
                f"p{q}": float(
                    np.percentile([r["total_seconds"] for r in good], q) * 1000
                )
                for q in (50, 95, 99)
            },
            "output_bytes": {
                f"p{q}": float(np.percentile([r["bytes"] for r in good], q))
                for q in (50, 95, 99)
            },
            "median_completion_throughput_rps": float(
                np.median([r["completion_throughput_rps"] for r in rounds])
            ),
            "per_wave": rounds,
        }
    baseline = {(r["arrival_seed"], r["request_number"]): r for r in requests["base"]}
    report["paired_output_size_match"] = {
        mode: sum(
            r["bytes"] == baseline[r["arrival_seed"], r["request_number"]]["bytes"]
            for r in requests[mode]
        )
        / len(requests[mode])
        for mode in ("unset", "always")
    }
    OUTPUT.joinpath("report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
