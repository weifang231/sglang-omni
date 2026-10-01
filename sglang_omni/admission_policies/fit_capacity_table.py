# SPDX-License-Identifier: Apache-2.0
"""Fit a capacity-table profile from events recorded by the capacity_table policy.

    python -m sglang_omni.admission_policies.fit_capacity_table events.jsonl \\
        --kind asr --deadline-s 0.5 --arrival-rate-rps 48 --capacity 24 --output profile.json

The recorded traffic defines, per admission occupancy ``n``: the first-output latencies of
requests admitted at ``n`` (``features_by_occupancy[n]``), the completion rate while ``n+1``
requests are in flight (``departure_rates[n]``, departures divided by time spent in that
state), ``q[n]`` (share of those latencies within ``deadline_s - guard_s``) and the prices
solved for ``--arrival-rate-rps``. ``--capacity auto`` takes the largest occupancy every
state below which has at least ``--min-samples`` latencies and one departure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from sglang_omni.admission_policies.capacity_table import (
    CapacityTableProfile,
    solve_prices,
)


def read_events(path: str | Path) -> list[dict[str, Any]]:
    events = []
    with open(path, "r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                events.append(json.loads(line))
            else:
                pass
    events.sort(key=lambda e: e["t"])
    return events


def fit(
    events: list[dict[str, Any]],
    *,
    kind: str,
    deadline_s: float,
    arrival_rate_rps: float,
    capacity: int | str = "auto",
    guard_s: float = 0.01,
    min_samples: int = 20,
) -> dict[str, Any]:
    admitted: dict[str, dict[str, Any]] = {}
    first_by_occupancy: dict[int, list[float]] = {}
    time_in_state: dict[int, float] = {}
    departures: dict[int, int] = {}
    occupancy = 0
    last_t = events[0]["t"] if events else 0.0
    for e in events:
        time_in_state[occupancy] = time_in_state.get(occupancy, 0.0) + (e["t"] - last_t)
        last_t = e["t"]
        rid, kind_of_event = e["request_id"], e["event"]
        if kind_of_event == "admit":
            if e.get("enforced"):
                continue
            else:
                pass
            admitted[rid] = {"t": e["t"], "occupancy": occupancy}
            occupancy += 1
        elif kind_of_event == "first_output":
            row = admitted.get(rid)
            if row is not None:
                row["first_s"] = e["t"] - row["t"]
            else:
                pass
        elif kind_of_event in ("completed", "aborted"):
            row = admitted.pop(rid, None)
            if row is None:
                continue
            else:
                pass
            departures[occupancy] = departures.get(occupancy, 0) + 1
            occupancy = max(occupancy - 1, 0)
            if kind_of_event == "completed" and "first_s" in row:
                first_by_occupancy.setdefault(row["occupancy"], []).append(
                    row["first_s"]
                )
            else:
                pass
        else:
            pass

    def rate(n: int) -> float | None:
        if departures.get(n, 0) == 0 or time_in_state.get(n, 0.0) <= 0:
            return None
        else:
            return departures[n] / time_in_state[n]

    def usable(n: int) -> bool:
        return (
            len(first_by_occupancy.get(n, [])) >= min_samples
            and rate(n + 1) is not None
        )

    if capacity == "auto":
        cap = 0
        while usable(cap):
            cap += 1
        if cap == 0:
            raise ValueError(
                f"no occupancy level has {min_samples} first-output samples and a departure"
            )
        else:
            pass
    else:
        cap = int(capacity)
        short = [n for n in range(cap) if not usable(n)]
        if short:
            raise ValueError(
                f"occupancies {short} lack {min_samples} first-output samples or a "
                f"departure; record longer or lower --capacity (auto would pick "
                f"{next((n for n in range(cap) if not usable(n)), cap)})"
            )
        else:
            pass
    features = [
        [{"first_s": round(x, 6)} for x in sorted(first_by_occupancy[n])]
        for n in range(cap)
    ]
    departure_rates = [rate(n + 1) for n in range(cap)]
    q = [
        sum(1 for r in rows if r["first_s"] <= deadline_s - guard_s) / len(rows)
        for rows in features
    ]
    solved = solve_prices(q, arrival_rate_rps, departure_rates)
    document = {
        "schema_version": 1,
        "kind": kind,
        "capacity": cap,
        "deadline_s": deadline_s,
        "guard_s": guard_s,
        "planning_arrival_rate_rps": arrival_rate_rps,
        "features_by_occupancy": features,
        "departure_rates": departure_rates,
        "q": q,
        "prices": solved["prices"],
        "surrogate_gain_rps": solved["surrogate_gain_rps"],
        "bellman_rate_spread": solved["bellman_rate_spread"],
        "diagnostics": {
            "events": len(events),
            "samples_by_occupancy": {
                str(n): len(first_by_occupancy.get(n, [])) for n in range(cap)
            },
            "time_in_state_s": {
                str(n): time_in_state.get(n, 0.0) for n in range(cap + 1)
            },
            "departures_by_state": {
                str(n): departures.get(n, 0) for n in range(1, cap + 1)
            },
            "max_observed_occupancy": max(time_in_state) if time_in_state else 0,
        },
    }
    CapacityTableProfile.from_dict(document)  # validate
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "events", help="JSONL written by the policy in record/shadow/apply mode"
    )
    parser.add_argument("--kind", choices=("asr", "tts"), required=True)
    parser.add_argument(
        "--deadline-s", type=float, required=True, help="first-output SLO"
    )
    parser.add_argument(
        "--arrival-rate-rps", type=float, required=True, help="operating rate"
    )
    parser.add_argument("--capacity", default="auto", help="int, or 'auto'")
    parser.add_argument("--guard-s", type=float, default=0.01)
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    document = fit(
        read_events(args.events),
        kind=args.kind,
        deadline_s=args.deadline_s,
        arrival_rate_rps=args.arrival_rate_rps,
        capacity=args.capacity if args.capacity == "auto" else int(args.capacity),
        guard_s=args.guard_s,
        min_samples=args.min_samples,
    )
    Path(args.output).write_text(json.dumps(document, indent=2) + "\n")
    q = document["q"]
    print(
        f"capacity={document['capacity']} rate={args.arrival_rate_rps} "
        f"q[0]={q[0]:.3f} q[-1]={q[-1]:.3f} prices[-1]={document['prices'][-1]:.3f} -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
else:
    pass
