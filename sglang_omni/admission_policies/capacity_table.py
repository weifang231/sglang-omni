# SPDX-License-Identifier: Apache-2.0
"""Capacity-table admission: reject early the requests unlikely to meet their first-output deadline.

Enable with::

    admission_policy: sglang_omni.admission_policies.capacity_table.make_policy
    admission_policy_options:
      mode: apply            # record | shadow | apply
      profile: profile.json  # shadow/apply
      arrival_rate_rps: 48   # optional: re-solve the prices for this operating rate
      record_path: events.jsonl   # optional in shadow/apply, required in record mode

Workflow: run ``mode: record`` under representative load (admits everything, writes one
JSONL event per admit / first output / completion), fit a profile from that file with
``python -m sglang_omni.admission_policies.fit_capacity_table``, check it with
``mode: shadow`` (decides and records, never rejects), then switch to ``mode: apply``.

Decision rule, for a request arriving at execution occupancy ``n`` with ``remaining`` seconds of
budget: admit iff ``n < capacity`` and ``q(n, remaining) > prices[n]``, where ``q`` is the
empirical probability, from the recorded traffic, that a request admitted at occupancy ``n``
produces its first output within ``remaining - guard_s``; ``prices[n]`` is the average-reward
value of an execution slot solved from the fitted birth/death chain at the operating arrival
rate. The remaining budget is ``request.metadata["remaining_budget_s"]`` when the deployment
sets it (seconds left when the request was handed to the pipeline), else the profile's ``deadline_s``.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from bisect import bisect_right
from dataclasses import asdict, dataclass
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

BUDGET_KEY = "remaining_budget_s"
MODES = ("record", "shadow", "apply")


def solve_prices(probabilities, arrival_rate_rps, departure_rates):
    """Average-reward prices of the finite birth/death admission chain.

    State ``n`` (0..capacity-1) admits an arrival with reward ``q[n]`` and pays ``prices[n]``;
    completions leave state ``n+1`` at ``departure_rates[n]``. For a candidate gain ``g``:
    ``prices[c-1] = g / mu[c-1]`` and ``prices[n-1] = (g - lambda * max(q[n] - prices[n], 0)) / mu[n-1]``;
    the gain is the root of the state-0 residual (bisection in exact decimals).
    """
    q_values = list(probabilities)
    rates = list(departure_rates)
    if not rates or len(q_values) != len(rates):
        raise ValueError(
            "every admissible occupancy needs a probability and a departure rate"
        )
    else:
        pass
    for x in q_values:
        if not isinstance(x, (int, float)) or not math.isfinite(x) or not 0 <= x <= 1:
            raise ValueError("probabilities must lie in [0, 1]")
        else:
            pass
    for x in rates:
        if not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0:
            raise ValueError("departure rates must be positive finite numbers")
        else:
            pass
    if (
        not isinstance(arrival_rate_rps, (int, float))
        or not math.isfinite(arrival_rate_rps)
        or arrival_rate_rps <= 0
    ):
        raise ValueError("arrival rate must be a positive finite number")
    else:
        pass
    with localcontext() as context:
        context.prec = 60
        q = [Decimal(str(x)) for x in q_values]
        mu = [Decimal(str(x)) for x in rates]
        arrival = Decimal(str(arrival_rate_rps))

        def recurse(gain):
            prices = [Decimal(0)] * len(mu)
            prices[-1] = gain / mu[-1]
            for n in range(len(mu) - 1, 0, -1):
                prices[n - 1] = (
                    gain - arrival * max(q[n] - prices[n], Decimal(0))
                ) / mu[n - 1]
            return gain - arrival * max(q[0] - prices[0], Decimal(0)), prices

        lo, hi, gain = Decimal(0), arrival, Decimal(0)
        if any(q):
            for _ in range(400):
                gain = (lo + hi) / 2
                residual, _ = recurse(gain)
                if abs(residual) <= Decimal("1e-35") * arrival:
                    break
                elif residual > 0:
                    hi = gain
                else:
                    lo = gain
            else:
                raise ArithmeticError("admission price solve did not converge")
        else:
            pass
        _, prices = recurse(gain)
        stored = [float(x) for x in prices]
    gains = []
    for n in range(len(stored) + 1):
        value = rates[n - 1] * stored[n - 1] if n else 0.0
        if n < len(stored):
            value += arrival_rate_rps * max(q_values[n] - stored[n], 0.0)
        else:
            pass
        gains.append(value)
    return {
        "prices": stored,
        "surrogate_gain_rps": float(gain),
        "bellman_rate_spread": max(gains) - min(gains),
    }


@dataclass(frozen=True)
class Decision:
    admission: str  # "admit" | "reject"
    reason: str  # "probability_exceeds_price" | "deadline_or_price" | "capacity"
    occupancy: int
    remaining_s: float | None
    probability: float | None
    price: float | None


def finite_positive(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value) and value > 0


class CapacityTableProfile:
    """Validated in-memory form of a fitted profile (see ``fit_capacity_table``)."""

    def __init__(self, document: dict[str, Any]):
        self.document = document
        self.capacity: int = document["capacity"]
        self.kind: str = document["kind"]
        self.deadline_s: float = float(document["deadline_s"])
        self.guard_s: float = float(document.get("guard_s") or 0.0)
        self.features: list[list[dict[str, float]]] = document["features_by_occupancy"]
        self.first_output_seconds: list[list[float]] = [
            sorted(row["first_s"] for row in rows) for rows in self.features
        ]
        self.prices: list[float] = [float(p) for p in document["prices"]]
        rates = document.get("departure_rates")
        self.departure_rates: list[float] | None = (
            [float(x) for x in rates] if isinstance(rates, list) else None
        )
        planning = document.get("planning_arrival_rate_rps")
        self.planning_arrival_rate_rps: float | None = (
            float(planning) if isinstance(planning, (int, float)) else None
        )

    @classmethod
    def from_dict(cls, document: dict[str, Any]) -> "CapacityTableProfile":
        if not isinstance(document, dict):
            raise ValueError("profile must be a JSON object")
        else:
            pass
        capacity = document.get("capacity")
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        else:
            pass
        if document.get("kind") not in ("asr", "tts"):
            raise ValueError("kind must be 'asr' or 'tts'")
        else:
            pass
        if not finite_positive(document.get("deadline_s")):
            raise ValueError("deadline_s must be a positive finite number")
        else:
            pass
        for key in ("features_by_occupancy", "prices"):
            value = document.get(key)
            if not isinstance(value, list) or len(value) != capacity:
                raise ValueError(f"{key} must be a list of length capacity={capacity}")
            else:
                pass
        for n, rows in enumerate(document["features_by_occupancy"]):
            if not isinstance(rows, list) or not rows:
                raise ValueError(f"features_by_occupancy[{n}] must be a non-empty list")
            else:
                pass
            for row in rows:
                first = row.get("first_s") if isinstance(row, dict) else None
                if not isinstance(first, (int, float)) or not math.isfinite(first):
                    raise ValueError(
                        f"features_by_occupancy[{n}] rows need a finite first_s"
                    )
                else:
                    pass
        for p in document["prices"]:
            if not isinstance(p, (int, float)) or not math.isfinite(p):
                raise ValueError("prices must be finite numbers")
            else:
                pass
        rates = document.get("departure_rates")
        if rates is not None:
            if not isinstance(rates, list) or len(rates) != capacity:
                raise ValueError(
                    f"departure_rates must be a list of length capacity={capacity}"
                )
            else:
                pass
            for x in rates:
                if not finite_positive(x):
                    raise ValueError("departure_rates must be positive finite numbers")
                else:
                    pass
        else:
            pass
        planning = document.get("planning_arrival_rate_rps")
        if planning is not None and not finite_positive(planning):
            raise ValueError(
                "planning_arrival_rate_rps must be a positive finite number"
            )
        else:
            pass
        profile = cls(document)
        if rates is not None and planning is not None:
            fitted = solve_prices(profile.q_table(), planning, profile.departure_rates)[
                "prices"
            ]
            drift = max(abs(a - b) for a, b in zip(fitted, profile.prices))
            if drift > 1e-9:
                logger.warning(
                    "capacity-table profile: stored prices differ from a re-solve at "
                    "planning_arrival_rate_rps=%s by %.3g",
                    planning,
                    drift,
                )
            else:
                pass
        else:
            pass
        return profile

    @classmethod
    def load(cls, path: str | Path) -> "CapacityTableProfile":
        with open(path, "r", encoding="utf-8") as stream:
            return cls.from_dict(json.load(stream))

    def probability(self, occupancy: int, remaining_s: float) -> float:
        """P[first output within ``remaining_s - guard_s``] for admission at ``occupancy``."""
        budget = remaining_s - self.guard_s
        if math.isnan(budget):
            return 0.0
        else:
            latencies = self.first_output_seconds[occupancy]
            return bisect_right(latencies, budget) / len(latencies)

    def q_table(self) -> list[float]:
        return [self.probability(n, self.deadline_s) for n in range(self.capacity)]

    def repriced(self, arrival_rate_rps: float) -> "CapacityTableProfile":
        """Copy whose prices are re-solved for ``arrival_rate_rps`` (needs departure_rates)."""
        if self.departure_rates is None:
            raise ValueError("profile has no departure_rates; cannot re-price")
        else:
            pass
        solved = solve_prices(self.q_table(), arrival_rate_rps, self.departure_rates)
        document = dict(self.document)
        document.update(
            prices=solved["prices"],
            planning_arrival_rate_rps=arrival_rate_rps,
            surrogate_gain_rps=solved["surrogate_gain_rps"],
            bellman_rate_spread=solved["bellman_rate_spread"],
        )
        return CapacityTableProfile.from_dict(document)

    def decide(self, occupancy: int, remaining_s: float) -> Decision:
        if occupancy >= self.capacity:
            return Decision("reject", "capacity", occupancy, remaining_s, None, None)
        else:
            pass
        q = self.probability(occupancy, remaining_s)
        price = self.prices[occupancy]
        if remaining_s > 0 and q > price:
            return Decision(
                "admit", "probability_exceeds_price", occupancy, remaining_s, q, price
            )
        else:
            return Decision(
                "reject", "deadline_or_price", occupancy, remaining_s, q, price
            )


class EventRecorder:
    """Appends one JSON object per line: {"event", "request_id", "t", "occupancy", ...}."""

    def __init__(self, path: str | Path, *, flush_every: int = 256):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a", encoding="utf-8")
        self.flush_every = flush_every
        self.pending = 0

    def write(self, event: dict[str, Any]) -> None:
        self.stream.write(json.dumps(event, separators=(",", ":")) + "\n")
        self.pending += 1
        if self.pending >= self.flush_every:
            self.stream.flush()
            self.pending = 0
        else:
            pass

    def close(self) -> None:
        self.stream.flush()
        self.stream.close()


class CapacityTablePolicy:
    """The admission policy object (see the module docstring)."""

    def __init__(
        self,
        profile: CapacityTableProfile | None,
        *,
        mode: str = "apply",
        recorder: EventRecorder | None = None,
        clock=time.monotonic,
    ):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        else:
            pass
        if mode != "record" and profile is None:
            raise ValueError(f"mode {mode!r} needs a profile")
        else:
            pass
        if mode == "record" and recorder is None:
            raise ValueError("record mode needs a record_path")
        else:
            pass
        self.profile = profile
        self.mode = mode
        self.recorder = recorder
        self.clock = clock
        self.lock = threading.Lock()
        self.in_flight: dict[str, float] = {}  # request_id -> admit time
        self.decisions: dict[str, Decision] = {}
        self.counts = {"admitted": 0, "rejected": 0, "shadow_rejected": 0}

    def record(self, event: str, request_id: str, now: float, **fields: Any) -> None:
        if self.recorder is None:
            return
        else:
            pass
        self.recorder.write(
            {"event": event, "request_id": request_id, "t": now, **fields}
        )

    def remaining_budget(self, request: Any, now: float) -> float:
        metadata = getattr(request, "metadata", None) or {}
        budget = metadata.get(BUDGET_KEY)
        if isinstance(budget, (int, float)) and math.isfinite(budget):
            return float(budget)
        elif self.profile is not None:
            return self.profile.deadline_s
        else:
            return math.nan  # record mode: no deadline, nothing to decide

    def admit(self, request_id: str, request: Any) -> bool:
        now = self.clock()
        with self.lock:
            occupancy = len(self.in_flight)
            remaining = self.remaining_budget(request, now)
            if self.profile is None:
                decision = Decision("admit", "record", occupancy, None, None, None)
            else:
                decision = self.profile.decide(occupancy, remaining)
            reject = decision.admission == "reject" and self.mode == "apply"
            if decision.admission == "reject":
                self.counts[
                    "shadow_rejected" if self.mode != "apply" else "rejected"
                ] += 1
            else:
                self.counts["admitted"] += 1
            self.record("admit", request_id, now, **asdict(decision), enforced=reject)
            if reject:
                return False
            else:
                pass
            self.in_flight[request_id] = now
            self.decisions[request_id] = decision
        return True

    def first_output(self, request_id: str) -> None:
        now = self.clock()
        with self.lock:
            if request_id not in self.in_flight:
                return
            else:
                pass
            self.record(
                "first_output",
                request_id,
                now,
                occupancy=len(self.in_flight),
                first_s=now - self.in_flight[request_id],
            )

    def completed(self, request_id: str) -> None:
        self.release(request_id, "completed")

    def aborted(self, request_id: str) -> None:
        self.release(request_id, "aborted")

    def release(self, request_id: str, event: str) -> None:
        now = self.clock()
        with self.lock:
            admitted_at = self.in_flight.pop(request_id, None)
            self.decisions.pop(request_id, None)
            if admitted_at is None:
                return
            else:
                pass
            self.record(
                event,
                request_id,
                now,
                occupancy=len(self.in_flight) + 1,
                total_s=now - admitted_at,
            )

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "policy": "capacity_table",
                "mode": self.mode,
                "capacity": self.profile.capacity if self.profile else None,
                "deadline_s": self.profile.deadline_s if self.profile else None,
                "planning_arrival_rate_rps": (
                    self.profile.planning_arrival_rate_rps if self.profile else None
                ),
                "in_flight": len(self.in_flight),
                "counts": dict(self.counts),
            }

    def close(self) -> None:
        if self.recorder is not None:
            self.recorder.close()
        else:
            pass


def make_policy(*, config) -> CapacityTablePolicy:
    """Factory for ``PipelineConfig.admission_policy``; reads ``admission_policy_options``."""
    options = dict(getattr(config, "admission_policy_options", None) or {})
    unknown = set(options) - {"mode", "profile", "arrival_rate_rps", "record_path"}
    if unknown:
        raise ValueError(f"admission_policy_options: unknown keys {sorted(unknown)}")
    else:
        pass
    mode = options.get("mode", "apply")
    if mode not in MODES:
        raise ValueError(
            f"admission_policy_options.mode must be one of {MODES}, got {mode!r}"
        )
    else:
        pass
    profile = None
    if mode != "record":
        path = options.get("profile")
        if not path:
            raise ValueError(
                f"admission_policy_options.profile is required in mode {mode!r}"
            )
        else:
            pass
        profile = CapacityTableProfile.load(path)
        rate = options.get("arrival_rate_rps")
        if rate is not None:
            if not finite_positive(rate):
                raise ValueError(
                    "admission_policy_options.arrival_rate_rps must be positive"
                )
            else:
                pass
            profile = profile.repriced(float(rate))
        else:
            pass
    else:
        pass
    record_path = options.get("record_path")
    recorder = EventRecorder(record_path) if record_path else None
    policy = CapacityTablePolicy(profile, mode=mode, recorder=recorder)
    logger.info(
        "capacity-table admission: mode=%s capacity=%s deadline_s=%s rate_rps=%s record=%s",
        mode,
        profile.capacity if profile else None,
        profile.deadline_s if profile else None,
        profile.planning_arrival_rate_rps if profile else None,
        record_path,
    )
    return policy
