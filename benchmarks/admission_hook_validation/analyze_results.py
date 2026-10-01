# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

import numpy as np

RUN = Path(__file__).resolve().parent


def analyze():
    summaries = json.loads((RUN / "summaries.json").read_text())
    output = {
        "scope": "Normal-load HTTP model serving, 3 seeds with Latin-square GPU rotation; not saturation throughput or statistical equivalence.",
        "models": {},
        "rejection": json.loads((RUN / "http-rejection-verification.json").read_text()),
    }
    generator = np.random.default_rng(20261001)
    for model in ("whisper", "tts"):
        rows = {}
        for mode in ("base", "unset", "always"):
            rounds = [
                summary
                for summary in summaries
                if summary["model"] == model and summary["mode"] == mode
            ]
            requests = []
            for wave in range(3):
                requests.extend(
                    json.loads(
                        (RUN / f"{model}-{mode}-wave{wave}-results.json").read_text()
                    )["requests"]
                )
            good = [request for request in requests if request["success"]]
            rows[mode] = {
                "requests": len(requests),
                "successes": len(good),
                "pooled_first_ms": {
                    f"p{percentile}": float(
                        np.percentile(
                            [request["first_seconds"] for request in good], percentile
                        )
                        * 1000
                    )
                    for percentile in (50, 95, 99)
                },
                "pooled_total_ms": {
                    f"p{percentile}": float(
                        np.percentile(
                            [request["total_seconds"] for request in good], percentile
                        )
                        * 1000
                    )
                    for percentile in (50, 95, 99)
                },
                "output_bytes_or_text_characters": {
                    f"p{percentile}": float(
                        np.percentile(
                            [request["bytes"] for request in good], percentile
                        )
                    )
                    for percentile in (50, 95, 99)
                },
                "median_completion_throughput_rps": float(
                    np.median(
                        [summary["completion_throughput_rps"] for summary in rounds]
                    )
                ),
                "per_wave": rounds,
            }
        pairs = {}
        for mode in ("unset", "always"):
            comparisons = {}
            for metric in (
                "first_seconds",
                "total_seconds",
                "completion_throughput_rps",
            ):
                ratios = []
                for seed in (8101, 8102, 8103):
                    base = next(
                        row for row in rows["base"]["per_wave"] if row["seed"] == seed
                    )
                    candidate = next(
                        row for row in rows[mode]["per_wave"] if row["seed"] == seed
                    )
                    if metric.endswith("seconds"):
                        ratios.append(candidate[metric]["p50"] / base[metric]["p50"])
                    else:
                        ratios.append(candidate[metric] / base[metric])
                resamples = generator.choice(
                    ratios, size=(10000, 3), replace=True
                ).mean(axis=1)
                comparisons[metric] = {
                    "mean_ratio": float(np.mean(ratios)),
                    "per_wave_ratios": ratios,
                    "paired_wave_bootstrap_95_interval": np.percentile(
                        resamples, [2.5, 97.5]
                    ).tolist(),
                    "caution": "Only 3 independent wave pairs; interval cannot establish equivalence.",
                }
            pairs[mode] = comparisons
        output["models"][model] = {"modes": rows, "paired_comparison_to_base": pairs}
    (RUN / "report.json").write_text(json.dumps(output, indent=2) + "\n")
    for model, results in output["models"].items():
        for mode, result in results["modes"].items():
            print(
                model,
                mode,
                result["successes"],
                result["pooled_first_ms"],
                result["pooled_total_ms"],
                result["median_completion_throughput_rps"],
            )


if __name__ == "__main__":
    analyze()
