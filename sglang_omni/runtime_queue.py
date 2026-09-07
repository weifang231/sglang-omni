# SPDX-License-Identifier: Apache-2.0
"""Runtime-owned FIFO/EDF queues with global and per-class WIP credits."""

from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from collections import Counter, deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from types import MappingProxyType
from typing import Any, Generic, Literal, TypeVar

from sglang_omni.admission import QueueFullError
from sglang_omni.utils.runtime_timing import RuntimeTimingRecorder

REQUEST_CLASS_METADATA_KEY = "sglang_omni.request_class"
FIRST_OUTPUT_DEADLINE_METADATA_KEY = "sglang_omni.first_output_deadline_unix_s"
ADMISSION_CORRELATION_ID_METADATA_KEY = "sglang_omni.admission_correlation_id"
PLAYBACK_BUFFER_MS_METADATA_KEY = "sglang_omni.playback_buffer_ms"
DEFAULT_REQUEST_CLASS = "default"

QueueDiscipline = Literal["fifo", "edf"]
ClassLimitMode = Literal["hard_limit", "soft_reservation"]
AdmissionScoreMethod = Literal["erlang_empirical", "erlang_empirical_threshold"]
ADMISSION_DECISION_HISTORY_LIMIT = 128
ADMISSION_SCORE_REFERENCE_TOLERANCE = 1e-12
ADMISSION_THRESHOLD_TABLE_SCHEMA_VERSION = 1
DEFAULT_ADMISSION_MAX_REQUIRED_RETURNS = 2048
ItemT = TypeVar("ItemT")


def _label(value: Any, *, default: str, field_name: str) -> str:
    if value is None:
        return default
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty")
    if len(normalized) > 128 or any(ord(character) < 32 for character in normalized):
        raise ValueError(f"{field_name} must be at most 128 printable characters")
    return normalized


def _optional_limit(value: Any, *, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer or null")
    return value


def _finite_float(
    value: Any,
    *,
    field_name: str,
    minimum: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a finite number")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field_name} must be a finite number") from exc
    if not math.isfinite(parsed) or (minimum is not None and parsed < minimum):
        qualifier = (
            f" greater than or equal to {minimum:g}" if minimum is not None else ""
        )
        raise ValueError(f"{field_name} must be a finite number{qualifier}")
    return parsed


def _sha256_fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validated_sha256(value: Any, *, field_name: str) -> str:
    fingerprint = _label(value, default="", field_name=field_name)
    if len(fingerprint) != 64 or any(
        character not in "0123456789abcdef" for character in fingerprint
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")
    return fingerprint


@dataclass(frozen=True, slots=True)
class AdmissionClassConfig:
    """Calibrated empirical-Erlang inputs for one request class."""

    effective_k: int
    mu: float
    service_samples_s: tuple[float, ...]
    gamma: float
    max_required_returns: int = DEFAULT_ADMISSION_MAX_REQUIRED_RETURNS
    compiled_thresholds_s: tuple[float, ...] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    threshold_profile_fingerprint: str | None = field(default=None, repr=False)
    threshold_table_digest: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if (
            isinstance(self.effective_k, bool)
            or not isinstance(self.effective_k, int)
            or self.effective_k < 0
        ):
            raise ValueError("effective_k must be a non-negative integer")
        if (
            isinstance(self.max_required_returns, bool)
            or not isinstance(self.max_required_returns, int)
            or self.max_required_returns < 0
        ):
            raise ValueError("max_required_returns must be a non-negative integer")
        mu = float(self.mu)
        gamma = float(self.gamma)
        if not math.isfinite(mu) or mu < 0.0:
            raise ValueError("mu must be finite and non-negative")
        if self.effective_k > 0 and mu <= 0.0:
            raise ValueError("mu must be positive when effective_k is positive")
        if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be finite and between 0 and 1")
        samples = tuple(float(sample) for sample in self.service_samples_s)
        if not samples or any(
            not math.isfinite(sample) or sample < 0.0 for sample in samples
        ):
            raise ValueError(
                "service_samples_s must contain finite non-negative values"
            )
        object.__setattr__(self, "mu", mu)
        object.__setattr__(self, "gamma", gamma)
        object.__setattr__(self, "service_samples_s", samples)

        threshold_artifacts = (
            self.compiled_thresholds_s,
            self.threshold_profile_fingerprint,
            self.threshold_table_digest,
        )
        if any(value is None for value in threshold_artifacts) and any(
            value is not None for value in threshold_artifacts
        ):
            raise ValueError(
                "compiled_thresholds_s, threshold_profile_fingerprint, and "
                "threshold_table_digest must be provided together"
            )
        if self.compiled_thresholds_s is None:
            return
        thresholds = tuple(float(threshold) for threshold in self.compiled_thresholds_s)
        expected_entries = self.max_required_returns + 1
        if len(thresholds) != expected_entries:
            raise ValueError(
                f"compiled_thresholds_s must contain {expected_entries} entries"
            )
        previous = -math.inf
        for index, threshold in enumerate(thresholds):
            if not math.isfinite(threshold) or threshold < 0.0:
                raise ValueError(
                    f"compiled_thresholds_s[{index}] must be finite and non-negative"
                )
            if threshold < previous:
                raise ValueError("compiled_thresholds_s must be non-decreasing")
            previous = threshold
        object.__setattr__(self, "compiled_thresholds_s", thresholds)
        assert self.threshold_profile_fingerprint is not None
        _validated_sha256(
            self.threshold_profile_fingerprint,
            field_name="threshold_profile_fingerprint",
        )
        assert self.threshold_table_digest is not None
        _validated_sha256(
            self.threshold_table_digest,
            field_name="threshold_table_digest",
        )

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        field_name: str,
    ) -> AdmissionClassConfig:
        effective_k = _optional_limit(
            raw.get("effective_k"),
            field_name=f"{field_name}.effective_k",
        )
        if effective_k is None:
            raise ValueError(f"{field_name}.effective_k is required")
        raw_samples = raw.get("service_samples_s")
        if not isinstance(raw_samples, (list, tuple)) or not raw_samples:
            raise ValueError(
                f"{field_name}.service_samples_s must be a non-empty JSON array"
            )
        samples = tuple(
            _finite_float(
                sample,
                field_name=f"{field_name}.service_samples_s[{index}]",
                minimum=0.0,
            )
            for index, sample in enumerate(raw_samples)
        )
        max_required_returns = _optional_limit(
            raw.get("max_required_returns", DEFAULT_ADMISSION_MAX_REQUIRED_RETURNS),
            field_name=f"{field_name}.max_required_returns",
        )
        assert max_required_returns is not None
        raw_thresholds = raw.get("compiled_thresholds_s")
        compiled_thresholds_s = None
        threshold_profile_fingerprint = None
        threshold_table_digest = None
        if raw_thresholds is not None:
            if not isinstance(raw_thresholds, (list, tuple)):
                raise ValueError(
                    f"{field_name}.compiled_thresholds_s must be a JSON array"
                )
            compiled_thresholds_s = tuple(
                _finite_float(
                    threshold,
                    field_name=f"{field_name}.compiled_thresholds_s[{index}]",
                    minimum=0.0,
                )
                for index, threshold in enumerate(raw_thresholds)
            )
            threshold_profile_fingerprint = _validated_sha256(
                raw.get("threshold_profile_fingerprint"),
                field_name=f"{field_name}.threshold_profile_fingerprint",
            )
            threshold_table_digest = _validated_sha256(
                raw.get("threshold_table_digest"),
                field_name=f"{field_name}.threshold_table_digest",
            )
        elif (
            raw.get("threshold_profile_fingerprint") is not None
            or raw.get("threshold_table_digest") is not None
        ):
            raise ValueError(
                f"{field_name}.compiled_thresholds_s, "
                f"{field_name}.threshold_profile_fingerprint, and "
                f"{field_name}.threshold_table_digest must be provided together"
            )
        return cls(
            effective_k=effective_k,
            mu=_finite_float(raw.get("mu"), field_name=f"{field_name}.mu", minimum=0.0),
            service_samples_s=samples,
            gamma=_finite_float(
                raw.get("gamma"),
                field_name=f"{field_name}.gamma",
                minimum=0.0,
            ),
            max_required_returns=max_required_returns,
            compiled_thresholds_s=compiled_thresholds_s,
            threshold_profile_fingerprint=threshold_profile_fingerprint,
            threshold_table_digest=threshold_table_digest,
        )


@dataclass(frozen=True, slots=True)
class AdmissionThresholdTable:
    """One immutable deadline-threshold table installed for a request class."""

    request_class: str
    profile_fingerprint: str
    table_digest: str
    thresholds_s: tuple[float, ...]

    @property
    def max_required_returns(self) -> int:
        return len(self.thresholds_s) - 1

    def to_config_fields(self) -> dict[str, Any]:
        return {
            "max_required_returns": self.max_required_returns,
            "compiled_thresholds_s": list(self.thresholds_s),
            "threshold_profile_fingerprint": self.profile_fingerprint,
            "threshold_table_digest": self.table_digest,
        }


@dataclass(frozen=True, slots=True)
class AdmissionControlConfig:
    """Opt-in ingress admission settings nested under queue_control."""

    enabled: bool = False
    score_method: AdmissionScoreMethod = "erlang_empirical"
    classes: dict[str, AdmissionClassConfig] = field(default_factory=dict)
    enforce: bool = True
    _threshold_tables: Mapping[str, AdmissionThresholdTable] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.score_method not in {"erlang_empirical", "erlang_empirical_threshold"}:
            raise ValueError(
                "score_method must be 'erlang_empirical' or "
                "'erlang_empirical_threshold'"
            )
        if self.enabled and not self.classes:
            raise ValueError(
                "admission classes must not be empty when admission is enabled"
            )
        tables: dict[str, AdmissionThresholdTable] = {}
        if self.score_method == "erlang_empirical_threshold":
            for request_class, class_config in self.classes.items():
                if class_config.gamma + ADMISSION_SCORE_REFERENCE_TOLERANCE >= 1.0:
                    raise ValueError(
                        "gamma plus the numerical tie guard must be less than 1 "
                        "for score_method='erlang_empirical_threshold'"
                    )
                if class_config.compiled_thresholds_s is None:
                    raise ValueError(
                        "compiled_thresholds_s is required for "
                        "score_method='erlang_empirical_threshold'"
                    )
                profile_fingerprint = admission_threshold_profile_fingerprint(
                    request_class,
                    effective_k=class_config.effective_k,
                    mu=class_config.mu,
                    service_samples_s=class_config.service_samples_s,
                    gamma=class_config.gamma,
                    max_required_returns=class_config.max_required_returns,
                )
                if class_config.threshold_profile_fingerprint != profile_fingerprint:
                    raise ValueError(
                        "compiled admission threshold profile fingerprint does "
                        f"not match class {request_class!r}"
                    )
                thresholds_s = class_config.compiled_thresholds_s
                table_digest = admission_threshold_table_digest(
                    profile_fingerprint=profile_fingerprint,
                    thresholds_s=thresholds_s,
                )
                if class_config.threshold_table_digest != table_digest:
                    raise ValueError(
                        "compiled admission threshold table digest does not "
                        f"match class {request_class!r}"
                    )
                validate_erlang_empirical_admission_threshold_table(
                    request_class=request_class,
                    effective_k=class_config.effective_k,
                    mu=class_config.mu,
                    service_samples_s=class_config.service_samples_s,
                    gamma=class_config.gamma,
                    profile_fingerprint=profile_fingerprint,
                    table_digest=table_digest,
                    thresholds_s=thresholds_s,
                )
                tables[request_class] = AdmissionThresholdTable(
                    request_class=request_class,
                    profile_fingerprint=profile_fingerprint,
                    table_digest=table_digest,
                    thresholds_s=thresholds_s,
                )
        elif any(
            class_config.compiled_thresholds_s is not None
            for class_config in self.classes.values()
        ):
            raise ValueError(
                "compiled admission thresholds require "
                "score_method='erlang_empirical_threshold'"
            )
        object.__setattr__(self, "_threshold_tables", MappingProxyType(tables))

    @property
    def threshold_tables(self) -> Mapping[str, AdmissionThresholdTable]:
        return self._threshold_tables

    @classmethod
    def from_mapping(cls, raw: Any) -> AdmissionControlConfig:
        if isinstance(raw, AdmissionControlConfig):
            return raw
        if raw is None or raw is False:
            return cls()
        if not isinstance(raw, Mapping):
            raise ValueError("queue_control.admission must be a JSON object or false")
        enabled = raw.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError("queue_control.admission.enabled must be boolean")
        enforce = raw.get("enforce", True)
        if not isinstance(enforce, bool):
            raise ValueError("queue_control.admission.enforce must be boolean")
        score_method = str(raw.get("score_method", "erlang_empirical")).strip().lower()
        if score_method not in {"erlang_empirical", "erlang_empirical_threshold"}:
            raise ValueError(
                "queue_control.admission.score_method must be "
                "'erlang_empirical' or 'erlang_empirical_threshold'"
            )
        raw_classes = raw.get("classes", {})
        if not isinstance(raw_classes, Mapping):
            raise ValueError("queue_control.admission.classes must be a JSON object")
        classes: dict[str, AdmissionClassConfig] = {}
        for raw_class, raw_config in raw_classes.items():
            request_class = _label(
                raw_class,
                default=DEFAULT_REQUEST_CLASS,
                field_name="queue_control.admission class",
            )
            if not isinstance(raw_config, Mapping):
                raise ValueError(
                    f"queue_control.admission.classes[{raw_class!r}] must be "
                    "a JSON object"
                )
            classes[request_class] = AdmissionClassConfig.from_mapping(
                raw_config,
                field_name=f"queue_control.admission.classes[{raw_class!r}]",
            )
        if enabled and not classes:
            raise ValueError(
                "queue_control.admission.classes must not be empty when "
                "admission is enabled"
            )
        return cls(
            enabled=enabled,
            enforce=enforce,
            score_method=score_method,  # type: ignore[arg-type]
            classes=classes,
        )


def erlang_wait_cdf(
    wait_budget_s: float,
    *,
    effective_k: int,
    mu: float,
    required_returns: int,
) -> float:
    """Return the Erlang waiting-time CDF used by D1 admission."""

    if wait_budget_s < 0:
        return 0.0
    if required_returns <= 0:
        return 1.0
    if effective_k <= 0 or mu <= 0:
        return 0.0
    x = effective_k * mu * wait_budget_s
    if x <= 0:
        return 0.0

    last = required_returns - 1
    mode = min(math.floor(x), last)
    max_log_probability = -x + mode * math.log(x) - math.lgamma(mode + 1)
    scaled_sum = 0.0
    log_x = math.log(x)
    for count in range(required_returns):
        log_probability = -x + count * log_x - math.lgamma(count + 1)
        scaled_sum += math.exp(log_probability - max_log_probability)
    poisson_cdf = math.exp(max_log_probability) * scaled_sum
    return min(max(1.0 - poisson_cdf, 0.0), 1.0)


def _erlang_empirical_score_for_required_returns(
    remaining_budget_s: float,
    *,
    effective_k: int,
    mu: float,
    required_returns: int,
    samples: tuple[float, ...],
) -> float:
    if required_returns <= 0:
        return sum(sample <= remaining_budget_s for sample in samples) / len(samples)
    return sum(
        erlang_wait_cdf(
            remaining_budget_s - sample,
            effective_k=effective_k,
            mu=mu,
            required_returns=required_returns,
        )
        for sample in samples
    ) / len(samples)


def erlang_empirical_admission_score(
    remaining_budget_s: float,
    *,
    effective_k: int,
    mu: float,
    active_count: int,
    queue_position: int,
    service_samples_s: tuple[float, ...],
) -> float:
    """Compute the empirical-Erlang D1 score."""

    if remaining_budget_s < 0 or effective_k <= 0 or active_count > effective_k:
        return 0.0
    if not service_samples_s:
        raise ValueError("service_samples_s must not be empty")
    required_returns = max(active_count + queue_position - effective_k + 1, 0)
    return _erlang_empirical_score_for_required_returns(
        remaining_budget_s,
        effective_k=effective_k,
        mu=mu,
        required_returns=required_returns,
        samples=service_samples_s,
    )


def admission_threshold_profile_fingerprint(
    request_class: str,
    *,
    effective_k: int,
    mu: float,
    service_samples_s: tuple[float, ...],
    gamma: float,
    max_required_returns: int,
) -> str:
    return _sha256_fingerprint(
        {
            "schema_version": ADMISSION_THRESHOLD_TABLE_SCHEMA_VERSION,
            "request_class": request_class,
            "score_method": "erlang_empirical_threshold",
            "effective_k": int(effective_k),
            "mu": float(mu),
            "service_samples_s": [float(sample) for sample in service_samples_s],
            "gamma": float(gamma),
            "max_required_returns": int(max_required_returns),
            "numerical_tie_guard": ADMISSION_SCORE_REFERENCE_TOLERANCE,
        }
    )


def admission_threshold_table_digest(
    *,
    profile_fingerprint: str,
    thresholds_s: tuple[float, ...] | list[float],
) -> str:
    return _sha256_fingerprint(
        {
            "schema_version": ADMISSION_THRESHOLD_TABLE_SCHEMA_VERSION,
            "profile_fingerprint": profile_fingerprint,
            "thresholds_hex": [float(threshold).hex() for threshold in thresholds_s],
        }
    )


@lru_cache(maxsize=128)
def _compile_erlang_empirical_admission_thresholds_cached(
    effective_k: int,
    mu: float,
    service_samples_s: tuple[float, ...],
    gamma: float,
    max_required_returns: int,
) -> tuple[float, ...]:
    if max_required_returns < 0:
        raise ValueError("max_required_returns must be non-negative")
    if not (0.0 <= gamma and gamma + ADMISSION_SCORE_REFERENCE_TOLERANCE < 1.0):
        raise ValueError(
            "gamma must be non-negative and gamma plus the numerical tie guard "
            "must be less than 1"
        )
    if not service_samples_s:
        raise ValueError("service_samples_s must not be empty")
    if any(not math.isfinite(sample) or sample < 0.0 for sample in service_samples_s):
        raise ValueError("service_samples_s must contain finite non-negative values")
    if effective_k <= 0 or mu <= 0.0 or gamma == 0.0:
        return (0.0,) * (max_required_returns + 1)

    sorted_samples = tuple(sorted(service_samples_s))
    required_sample_count = 1
    while required_sample_count / len(sorted_samples) < gamma:
        required_sample_count += 1
    thresholds = [float(sorted_samples[required_sample_count - 1])]
    rate = effective_k * mu
    maximum_sample = float(sorted_samples[-1])
    guarded_gamma = gamma + ADMISSION_SCORE_REFERENCE_TOLERANCE
    for required_returns in range(1, max_required_returns + 1):
        low = 0.0
        high = max(maximum_sample + required_returns / rate + 1.0, 1.0)
        while (
            _erlang_empirical_score_for_required_returns(
                high,
                effective_k=effective_k,
                mu=mu,
                required_returns=required_returns,
                samples=sorted_samples,
            )
            <= guarded_gamma
        ):
            high *= 2.0
            if not math.isfinite(high):
                raise ValueError("could not find a finite admission threshold")
        while True:
            midpoint = low + (high - low) / 2.0
            if midpoint == low or midpoint == high:
                break
            score = _erlang_empirical_score_for_required_returns(
                midpoint,
                effective_k=effective_k,
                mu=mu,
                required_returns=required_returns,
                samples=sorted_samples,
            )
            if score > guarded_gamma:
                high = midpoint
            else:
                low = midpoint
        thresholds.append(high)
    return tuple(thresholds)


def compile_erlang_empirical_admission_thresholds(
    *,
    effective_k: int,
    mu: float,
    service_samples_s: tuple[float, ...],
    gamma: float,
    max_required_returns: int = DEFAULT_ADMISSION_MAX_REQUIRED_RETURNS,
) -> tuple[float, ...]:
    parsed_mu = _finite_float(mu, field_name="mu", minimum=0.0)
    parsed_gamma = _finite_float(gamma, field_name="gamma", minimum=0.0)
    return _compile_erlang_empirical_admission_thresholds_cached(
        effective_k,
        parsed_mu,
        tuple(float(sample) for sample in service_samples_s),
        parsed_gamma,
        max_required_returns,
    )


def compile_erlang_empirical_admission_threshold_table(
    request_class: str,
    *,
    effective_k: int,
    mu: float,
    service_samples_s: tuple[float, ...],
    gamma: float,
    max_required_returns: int = DEFAULT_ADMISSION_MAX_REQUIRED_RETURNS,
) -> AdmissionThresholdTable:
    normalized_class = _label(
        request_class,
        default=DEFAULT_REQUEST_CLASS,
        field_name="request_class",
    )
    thresholds_s = compile_erlang_empirical_admission_thresholds(
        effective_k=effective_k,
        mu=mu,
        service_samples_s=service_samples_s,
        gamma=gamma,
        max_required_returns=max_required_returns,
    )
    profile_fingerprint = admission_threshold_profile_fingerprint(
        normalized_class,
        effective_k=effective_k,
        mu=float(mu),
        service_samples_s=tuple(float(sample) for sample in service_samples_s),
        gamma=float(gamma),
        max_required_returns=max_required_returns,
    )
    return AdmissionThresholdTable(
        request_class=normalized_class,
        profile_fingerprint=profile_fingerprint,
        table_digest=admission_threshold_table_digest(
            profile_fingerprint=profile_fingerprint,
            thresholds_s=thresholds_s,
        ),
        thresholds_s=thresholds_s,
    )


@lru_cache(maxsize=128)
def _validate_erlang_empirical_admission_threshold_table_cached(
    effective_k: int,
    mu: float,
    service_samples_s: tuple[float, ...],
    gamma: float,
    thresholds_s: tuple[float, ...],
) -> None:
    if effective_k <= 0 or mu <= 0.0 or gamma == 0.0:
        if any(threshold != 0.0 for threshold in thresholds_s):
            raise ValueError(
                "zero-capacity or zero-gamma admission threshold table must "
                "contain only zero thresholds"
            )
        return
    guarded_gamma = gamma + ADMISSION_SCORE_REFERENCE_TOLERANCE
    for required_returns, threshold in enumerate(thresholds_s):
        score = _erlang_empirical_score_for_required_returns(
            threshold,
            effective_k=effective_k,
            mu=mu,
            required_returns=required_returns,
            samples=service_samples_s,
        )
        target = gamma if required_returns == 0 else guarded_gamma
        if score + ADMISSION_SCORE_REFERENCE_TOLERANCE < target:
            raise ValueError(
                "compiled admission threshold does not pass its predicate at "
                f"required_returns={required_returns}"
            )


def validate_erlang_empirical_admission_threshold_table(
    *,
    request_class: str,
    effective_k: int,
    mu: float,
    service_samples_s: tuple[float, ...],
    gamma: float,
    profile_fingerprint: str,
    table_digest: str,
    thresholds_s: tuple[float, ...],
) -> None:
    expected_profile_fingerprint = admission_threshold_profile_fingerprint(
        request_class,
        effective_k=effective_k,
        mu=mu,
        service_samples_s=service_samples_s,
        gamma=gamma,
        max_required_returns=len(thresholds_s) - 1,
    )
    if profile_fingerprint != expected_profile_fingerprint:
        raise ValueError(
            "compiled admission threshold profile fingerprint does not match inputs"
        )
    expected_table_digest = admission_threshold_table_digest(
        profile_fingerprint=profile_fingerprint,
        thresholds_s=thresholds_s,
    )
    if table_digest != expected_table_digest:
        raise ValueError(
            "compiled admission threshold table digest does not match payload"
        )
    _validate_erlang_empirical_admission_threshold_table_cached(
        effective_k,
        mu,
        tuple(float(sample) for sample in service_samples_s),
        gamma,
        tuple(float(threshold) for threshold in thresholds_s),
    )


@dataclass(frozen=True, slots=True)
class OnlineAllocatorMetadata:
    """Versioned provenance for an externally computed queue-control update."""

    revision: int
    source_runtime_id: str
    source_snapshot_sequence: int
    source_config_generation: int
    source_config_fingerprint: str
    target_config_fingerprint: str
    profile_fingerprint: str

    @classmethod
    def from_mapping(cls, raw: Any) -> OnlineAllocatorMetadata | None:
        if isinstance(raw, OnlineAllocatorMetadata):
            return raw
        if raw is None:
            return None
        if not isinstance(raw, Mapping):
            raise ValueError("queue_control.online_allocator must be a JSON object")
        if raw.get("schema_version") != 2:
            raise ValueError("queue_control.online_allocator.schema_version must be 2")
        revision = _optional_limit(
            raw.get("revision"),
            field_name="queue_control.online_allocator.revision",
        )
        source_snapshot_sequence = _optional_limit(
            raw.get("source_snapshot_sequence"),
            field_name="queue_control.online_allocator.source_snapshot_sequence",
        )
        source_config_generation = _optional_limit(
            raw.get("source_config_generation"),
            field_name="queue_control.online_allocator.source_config_generation",
        )
        if revision is None or revision < 1:
            raise ValueError(
                "queue_control.online_allocator.revision must be at least 1"
            )
        if source_snapshot_sequence is None or source_snapshot_sequence < 1:
            raise ValueError(
                "queue_control.online_allocator.source_snapshot_sequence must be "
                "at least 1"
            )
        if source_config_generation is None:
            raise ValueError(
                "queue_control.online_allocator.source_config_generation is required"
            )
        source_runtime_id = _label(
            raw.get("source_runtime_id"),
            default="",
            field_name="queue_control.online_allocator.source_runtime_id",
        )
        if not source_runtime_id:
            raise ValueError(
                "queue_control.online_allocator.source_runtime_id is required"
            )
        return cls(
            revision=revision,
            source_runtime_id=source_runtime_id,
            source_snapshot_sequence=source_snapshot_sequence,
            source_config_generation=source_config_generation,
            source_config_fingerprint=_validated_sha256(
                raw.get("source_config_fingerprint"),
                field_name="queue_control.online_allocator.source_config_fingerprint",
            ),
            target_config_fingerprint=_validated_sha256(
                raw.get("target_config_fingerprint"),
                field_name="queue_control.online_allocator.target_config_fingerprint",
            ),
            profile_fingerprint=_validated_sha256(
                raw.get("profile_fingerprint"),
                field_name="queue_control.online_allocator.profile_fingerprint",
            ),
        )

    def to_snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "revision": self.revision,
            "source_runtime_id": self.source_runtime_id,
            "source_snapshot_sequence": self.source_snapshot_sequence,
            "source_config_generation": self.source_config_generation,
            "source_config_fingerprint": self.source_config_fingerprint,
            "target_config_fingerprint": self.target_config_fingerprint,
            "profile_fingerprint": self.profile_fingerprint,
        }


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    """One calibrated ingress-admission evaluation."""

    admitted: bool
    would_admit: bool
    enforced: bool
    request_id: str
    admission_correlation_id: str | None
    request_class: str
    phase: Literal["arrival", "recheck"]
    score: float | None
    gamma: float | None
    reason: str
    effective_k: int | None
    mu: float | None
    active_count: int
    queue_position: int
    remaining_budget_s: float | None
    required_returns: int | None = None
    threshold_s: float | None = None
    threshold_slack_s: float | None = None
    score_method: AdmissionScoreMethod | None = None
    threshold_table_digest: str | None = None


@dataclass(frozen=True, slots=True)
class RuntimeQueueAdmissionRejection(Generic[ItemT]):
    """A waiting request removed by an authoritative admission recheck."""

    item: RuntimeQueueItem[ItemT]
    decision: AdmissionDecision


@dataclass
class RuntimeQueueItem(Generic[ItemT]):
    """One accepted request waiting for or holding a runtime credit."""

    request_id: str
    value: ItemT
    request_class: str
    deadline_unix_s: float | None
    sequence: int
    enqueued_ns: int
    admission_correlation_id: str | None = None


@dataclass(frozen=True)
class RuntimeQueueCancellation(Generic[ItemT]):
    """Result of removing a waiting or active request."""

    item: RuntimeQueueItem[ItemT] | None
    was_active: bool
    dispatched: tuple[RuntimeQueueItem[ItemT], ...]


class RuntimeCreditQueue(Generic[ItemT]):
    """Non-preemptive WIP-credit queue.

    A request acquires one global credit and, when configured, one credit for
    its class. Decreasing a limit never preempts active work; it only prevents
    new dispatches until the active count falls below the new limit. Decreasing
    the waiting limit does not evict accepted waiters; it rejects new requests
    that would have to wait until the queue falls below the new limit.
    """

    def __init__(
        self,
        *,
        max_active_requests: int | None = None,
        max_waiting_requests: int | None = None,
        class_limits: Mapping[str, int] | None = None,
        class_limit_mode: ClassLimitMode = "hard_limit",
        discipline: QueueDiscipline = "fifo",
        trust_request_metadata: bool = False,
        class_metadata_key: str = REQUEST_CLASS_METADATA_KEY,
        deadline_metadata_key: str = FIRST_OUTPUT_DEADLINE_METADATA_KEY,
        admission_correlation_id_metadata_key: str = (
            ADMISSION_CORRELATION_ID_METADATA_KEY
        ),
        admission: Any | None = None,
        online_allocator: Any | None = None,
        clock: Callable[[], float] = time.time,
        runtime_id: str | None = None,
    ) -> None:
        normalized_class_limits = self._normalize_limits(
            max_active_requests,
            class_limits or {},
        )
        if isinstance(max_waiting_requests, bool) or (
            max_waiting_requests is not None
            and (not isinstance(max_waiting_requests, int) or max_waiting_requests < 0)
        ):
            raise ValueError("max_waiting_requests must be a non-negative integer")
        if class_limit_mode not in {"hard_limit", "soft_reservation"}:
            raise ValueError(
                "class_limit_mode must be 'hard_limit' or 'soft_reservation'"
            )
        if discipline not in {"fifo", "edf"}:
            raise ValueError("discipline must be 'fifo' or 'edf'")
        if not isinstance(trust_request_metadata, bool):
            raise ValueError("trust_request_metadata must be a boolean")
        if not trust_request_metadata and discipline == "edf":
            raise ValueError(
                "EDF requires trust_request_metadata=true because deadlines "
                "otherwise remain untrusted"
            )
        if not trust_request_metadata and any(
            request_class != DEFAULT_REQUEST_CLASS
            for request_class in normalized_class_limits
        ):
            raise ValueError(
                "non-default class limits require trust_request_metadata=true"
            )
        if not class_metadata_key.strip():
            raise ValueError("class_metadata_key must not be empty")
        if not deadline_metadata_key.strip():
            raise ValueError("deadline_metadata_key must not be empty")
        if not admission_correlation_id_metadata_key.strip():
            raise ValueError("admission_correlation_id_metadata_key must not be empty")

        admission_config = AdmissionControlConfig.from_mapping(admission)
        if (
            admission_config.enabled
            and admission_config.enforce
            and discipline != "edf"
        ):
            raise ValueError(
                "queue_control discipline must be 'edf' when admission "
                "enforcement is enabled"
            )
        if admission_config.enabled and not trust_request_metadata:
            raise ValueError(
                "queue_control admission requires trust_request_metadata=true"
            )
        if class_limit_mode == "soft_reservation":
            self._validate_soft_reservations(
                max_active_requests,
                normalized_class_limits,
            )
        online_allocator_metadata = OnlineAllocatorMetadata.from_mapping(
            online_allocator
        )

        self.max_active_requests = max_active_requests
        self.max_waiting_requests = max_waiting_requests
        self.class_limits = normalized_class_limits
        self.class_limit_mode: ClassLimitMode = class_limit_mode
        self.discipline: QueueDiscipline = discipline
        self.trust_request_metadata = trust_request_metadata
        self.class_metadata_key = class_metadata_key
        self.deadline_metadata_key = deadline_metadata_key
        self.admission_correlation_id_metadata_key = (
            admission_correlation_id_metadata_key
        )
        self.admission = admission_config
        self.online_allocator = online_allocator_metadata
        self._clock = clock
        self._runtime_id = runtime_id or uuid.uuid4().hex
        self._snapshot_sequence = 0
        self._config_generation = 0
        self._waiting: list[RuntimeQueueItem[ItemT]] = []
        self._active: dict[str, RuntimeQueueItem[ItemT]] = {}
        self._active_by_class: Counter[str] = Counter()
        self._sequence = 0
        self._waiting_rejected_total = 0
        self._last_rejection_reason: str | None = None
        self._admission_admitted_total = 0
        self._admission_rejected_total = 0
        self._admission_recheck_passed_total = 0
        self._admission_would_admit_decisions_total = 0
        self._admission_would_reject_decisions_total = 0
        self._admission_shadow_would_reject_decisions_total = 0
        self._admission_decision_sequence = 0
        self._admission_reason_counts: Counter[str] = Counter()
        self._recent_admission_decisions: deque[dict[str, Any]] = deque(
            maxlen=ADMISSION_DECISION_HISTORY_LIMIT
        )
        self._recheck_rejections_by_class_total: Counter[str] = Counter()
        self._soft_reservation_reserved_dispatch_total = 0
        self._soft_reservation_borrowed_dispatch_total = 0
        self._soft_reservation_blocked_total = 0
        self._config_fingerprint = self._fingerprint_current_config()
        self._timing_recorder = RuntimeTimingRecorder(component="runtime_queue")
        if (
            self.online_allocator is not None
            and self.online_allocator.target_config_fingerprint
            != self._config_fingerprint
        ):
            raise ValueError(
                "queue_control.online_allocator.target_config_fingerprint does "
                "not match the requested config"
            )

    @classmethod
    def from_config(cls, config: Any) -> RuntimeCreditQueue[Any]:
        """Build from a Pydantic config object or an equivalent mapping."""
        values = config.model_dump() if hasattr(config, "model_dump") else dict(config)
        return cls(**values)

    @staticmethod
    def _config_value(value: Any) -> Any:
        if hasattr(value, "model_dump"):
            return value.model_dump()
        return value

    @staticmethod
    def _normalize_limits(
        max_active_requests: int | None,
        class_limits: Mapping[str, int],
    ) -> dict[str, int]:
        if isinstance(max_active_requests, bool) or (
            max_active_requests is not None
            and (not isinstance(max_active_requests, int) or max_active_requests < 0)
        ):
            raise ValueError("max_active_requests must be a non-negative integer")
        normalized: dict[str, int] = {}
        for request_class, limit in class_limits.items():
            if not isinstance(request_class, str) or not request_class.strip():
                raise ValueError("class limit keys must be non-empty strings")
            if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
                raise ValueError("class limits must be non-negative integers")
            request_class = request_class.strip()
            if request_class in normalized:
                raise ValueError(
                    "class limit keys must remain unique after trimming whitespace"
                )
            normalized[request_class] = limit
        return normalized

    @staticmethod
    def _validate_soft_reservations(
        max_active_requests: int | None,
        class_limits: Mapping[str, int],
    ) -> None:
        if max_active_requests is None:
            raise ValueError("soft_reservation requires max_active_requests")
        if not class_limits:
            raise ValueError("soft_reservation requires class_limits")
        if sum(class_limits.values()) > max_active_requests:
            raise ValueError(
                "soft_reservation class shares must not exceed max_active_requests"
            )

    def semantic_mapping(self) -> dict[str, Any]:
        """Return the normalized queue-control config excluding allocator metadata."""

        return {
            "discipline": self.discipline,
            "max_active_requests": self.max_active_requests,
            "max_waiting_requests": self.max_waiting_requests,
            "class_limits": dict(sorted(self.class_limits.items())),
            "class_limit_mode": self.class_limit_mode,
            "trust_request_metadata": self.trust_request_metadata,
            "class_metadata_key": self.class_metadata_key,
            "deadline_metadata_key": self.deadline_metadata_key,
            "admission_correlation_id_metadata_key": (
                self.admission_correlation_id_metadata_key
            ),
            "admission": {
                "enabled": self.admission.enabled,
                "enforce": self.admission.enforce,
                "score_method": self.admission.score_method,
                "classes": {
                    request_class: {
                        "effective_k": class_config.effective_k,
                        "mu": class_config.mu,
                        "service_samples_s": list(class_config.service_samples_s),
                        "gamma": class_config.gamma,
                        **(
                            {
                                "max_required_returns": (
                                    class_config.max_required_returns
                                ),
                                "threshold_profile_fingerprint": (
                                    self.admission.threshold_tables[
                                        request_class
                                    ].profile_fingerprint
                                ),
                                "threshold_table_digest": (
                                    self.admission.threshold_tables[
                                        request_class
                                    ].table_digest
                                ),
                            }
                            if self.admission.score_method
                            == "erlang_empirical_threshold"
                            else {}
                        ),
                    }
                    for request_class, class_config in sorted(
                        self.admission.classes.items()
                    )
                },
            },
        }

    def _admission_semantic(
        self,
        admission: AdmissionControlConfig,
    ) -> dict[str, Any]:
        return {
            "enabled": admission.enabled,
            "enforce": admission.enforce,
            "score_method": admission.score_method,
            "classes": {
                request_class: {
                    "effective_k": class_config.effective_k,
                    "mu": class_config.mu,
                    "service_samples_s": list(class_config.service_samples_s),
                    "gamma": class_config.gamma,
                    **(
                        {
                            "max_required_returns": class_config.max_required_returns,
                            "threshold_profile_fingerprint": (
                                admission.threshold_tables[
                                    request_class
                                ].profile_fingerprint
                            ),
                            "threshold_table_digest": (
                                admission.threshold_tables[request_class].table_digest
                            ),
                        }
                        if admission.score_method == "erlang_empirical_threshold"
                        else {}
                    ),
                }
                for request_class, class_config in sorted(admission.classes.items())
            },
        }

    def _validate_online_allocator_update(
        self,
        incoming: OnlineAllocatorMetadata | None,
        *,
        new_fingerprint: str,
    ) -> None:
        if incoming is None:
            return
        if incoming.target_config_fingerprint != new_fingerprint:
            raise ValueError(
                "queue_control.online_allocator.target_config_fingerprint does "
                "not match the requested config"
            )
        current = self.online_allocator
        if current is None:
            return
        if incoming == current and new_fingerprint == self._config_fingerprint:
            return
        if incoming.revision < current.revision:
            raise ValueError(
                "queue_control.online_allocator.revision must not decrease "
                f"({incoming.revision} < {current.revision})"
            )
        if incoming.revision == current.revision and (
            new_fingerprint != self._config_fingerprint or incoming != current
        ):
            raise ValueError(
                "queue_control fields changed without advancing "
                "queue_control.online_allocator.revision"
            )
        if (
            incoming.source_runtime_id == current.source_runtime_id
            and incoming.source_snapshot_sequence <= current.source_snapshot_sequence
        ):
            raise ValueError(
                "queue_control.online_allocator.source_snapshot_sequence must "
                "advance between revisions for one runtime"
            )

    def config_fingerprint(self) -> str:
        return self._config_fingerprint

    @property
    def config_generation(self) -> int:
        return self._config_generation

    @property
    def runtime_id(self) -> str:
        return self._runtime_id

    def _fingerprint_current_config(self) -> str:
        return _sha256_fingerprint(self.semantic_mapping())

    def enqueue(
        self,
        request_id: str,
        value: ItemT,
        metadata: Mapping[str, Any] | None,
        *,
        enqueued_ns: int | None = None,
    ) -> tuple[RuntimeQueueItem[ItemT], ...]:
        start_ns = self._timing_recorder.start_ns()
        try:
            return self._enqueue_impl(
                request_id,
                value,
                metadata,
                enqueued_ns=enqueued_ns,
            )
        finally:
            self._timing_recorder.record_elapsed_ns("queue_enqueue_ns", start_ns)

    def _enqueue_impl(
        self,
        request_id: str,
        value: ItemT,
        metadata: Mapping[str, Any] | None,
        *,
        enqueued_ns: int | None = None,
    ) -> tuple[RuntimeQueueItem[ItemT], ...]:
        if request_id in self._active or any(
            queued.request_id == request_id for queued in self._waiting
        ):
            raise ValueError(f"request {request_id!r} is already queued or active")

        request_class, deadline_unix_s, admission_correlation_id = self._parse_metadata(
            metadata
        )
        item = RuntimeQueueItem(
            request_id=request_id,
            value=value,
            request_class=request_class,
            deadline_unix_s=deadline_unix_s,
            sequence=self._sequence,
            enqueued_ns=time.monotonic_ns() if enqueued_ns is None else enqueued_ns,
            admission_correlation_id=admission_correlation_id,
        )
        self._sequence += 1
        decision = self._evaluate_arrival_admission(item)
        if decision is not None:
            self._record_admission_decision(decision)
            if not decision.admitted:
                self._admission_rejected_total += 1
                self._last_rejection_reason = decision.reason
                raise QueueFullError()
            self._admission_admitted_total += 1
        if (
            self.max_waiting_requests is not None
            and len(self._waiting) >= self.max_waiting_requests
            and not self._can_dispatch_immediately(request_class)
        ):
            self._waiting_rejected_total += 1
            self._last_rejection_reason = "waiting_queue_full"
            raise QueueFullError()

        self._waiting.append(item)
        return self._drain()

    def attributes(
        self, metadata: Mapping[str, Any] | None
    ) -> tuple[str, float | None]:
        """Return the normalized class and absolute deadline metadata."""
        request_class, deadline_unix_s, _ = self._parse_metadata(metadata)
        return request_class, deadline_unix_s

    def release(
        self, request_id: str
    ) -> tuple[RuntimeQueueItem[ItemT], tuple[RuntimeQueueItem[ItemT], ...]] | None:
        start_ns = self._timing_recorder.start_ns()
        try:
            return self._release_impl(request_id)
        finally:
            self._timing_recorder.record_elapsed_ns("queue_release_ns", start_ns)
            if start_ns is not None and not self._active and not self._waiting:
                self._timing_recorder.flush()

    def _release_impl(
        self, request_id: str
    ) -> tuple[RuntimeQueueItem[ItemT], tuple[RuntimeQueueItem[ItemT], ...]] | None:
        item = self._active.pop(request_id, None)
        if item is None:
            return None
        self._active_by_class[item.request_class] -= 1
        if self._active_by_class[item.request_class] <= 0:
            self._active_by_class.pop(item.request_class, None)
        return item, self._drain()

    def cancel(self, request_id: str) -> RuntimeQueueCancellation[ItemT]:
        active = self._active.pop(request_id, None)
        if active is not None:
            self._active_by_class[active.request_class] -= 1
            if self._active_by_class[active.request_class] <= 0:
                self._active_by_class.pop(active.request_class, None)
            return RuntimeQueueCancellation(active, True, self._drain())

        for index, item in enumerate(self._waiting):
            if item.request_id == request_id:
                self._waiting.pop(index)
                return RuntimeQueueCancellation(item, False, self._drain())
        return RuntimeQueueCancellation(None, False, ())

    def update(
        self,
        *,
        max_active_requests: int | None,
        class_limits: Mapping[str, int],
        discipline: QueueDiscipline,
        max_waiting_requests: int | None = None,
        class_limit_mode: ClassLimitMode | None = None,
        admission: Any | None = None,
        online_allocator: Any | None = None,
    ) -> tuple[RuntimeQueueItem[ItemT], ...]:
        normalized_class_limits = self._normalize_limits(
            max_active_requests,
            class_limits,
        )
        if class_limit_mode is None:
            class_limit_mode = self.class_limit_mode
        if class_limit_mode not in {"hard_limit", "soft_reservation"}:
            raise ValueError(
                "class_limit_mode must be 'hard_limit' or 'soft_reservation'"
            )
        if discipline not in {"fifo", "edf"}:
            raise ValueError("discipline must be 'fifo' or 'edf'")
        if max_waiting_requests is None:
            max_waiting_requests = self.max_waiting_requests
        if isinstance(max_waiting_requests, bool) or (
            max_waiting_requests is not None
            and (not isinstance(max_waiting_requests, int) or max_waiting_requests < 0)
        ):
            raise ValueError("max_waiting_requests must be a non-negative integer")
        if not self.trust_request_metadata and discipline == "edf":
            raise ValueError(
                "EDF requires trust_request_metadata=true because deadlines "
                "otherwise remain untrusted"
            )
        if not self.trust_request_metadata and any(
            request_class != DEFAULT_REQUEST_CLASS
            for request_class in normalized_class_limits
        ):
            raise ValueError(
                "non-default class limits require trust_request_metadata=true"
            )
        admission_config = (
            self.admission
            if admission is None
            else AdmissionControlConfig.from_mapping(self._config_value(admission))
        )
        if (
            admission_config.enabled
            and admission_config.enforce
            and discipline != "edf"
        ):
            raise ValueError(
                "queue_control discipline must be 'edf' when admission "
                "enforcement is enabled"
            )
        if admission_config.enabled and not self.trust_request_metadata:
            raise ValueError(
                "queue_control admission requires trust_request_metadata=true"
            )
        if class_limit_mode == "soft_reservation":
            self._validate_soft_reservations(
                max_active_requests,
                normalized_class_limits,
            )
        online_allocator_metadata = (
            self.online_allocator
            if online_allocator is None
            else OnlineAllocatorMetadata.from_mapping(
                self._config_value(online_allocator)
            )
        )
        new_fingerprint = _sha256_fingerprint(
            {
                **self.semantic_mapping(),
                "discipline": discipline,
                "max_active_requests": max_active_requests,
                "max_waiting_requests": max_waiting_requests,
                "class_limits": dict(sorted(normalized_class_limits.items())),
                "class_limit_mode": class_limit_mode,
                "admission": self._admission_semantic(admission_config),
            }
        )
        self._validate_online_allocator_update(
            online_allocator_metadata,
            new_fingerprint=new_fingerprint,
        )
        self.max_active_requests = max_active_requests
        self.max_waiting_requests = max_waiting_requests
        self.class_limits = normalized_class_limits
        self.class_limit_mode = class_limit_mode
        self.discipline = discipline
        self.admission = admission_config
        self.online_allocator = online_allocator_metadata
        if new_fingerprint != self._config_fingerprint:
            self._config_generation += 1
            self._config_fingerprint = new_fingerprint
        return self._drain()

    def clear(self) -> tuple[RuntimeQueueItem[ItemT], ...]:
        items = tuple(self._active.values()) + tuple(self._waiting)
        self._active.clear()
        self._active_by_class.clear()
        self._waiting.clear()
        return items

    def is_active(self, request_id: str) -> bool:
        return request_id in self._active

    def is_waiting(self, request_id: str) -> bool:
        return any(item.request_id == request_id for item in self._waiting)

    @property
    def last_rejection_reason(self) -> str | None:
        return self._last_rejection_reason

    def snapshot(self) -> dict[str, Any]:
        waiting_by_class = Counter(item.request_class for item in self._waiting)
        self._snapshot_sequence += 1
        recent_admission_decisions = list(self._recent_admission_decisions)
        return {
            "runtime_id": self._runtime_id,
            "snapshot_sequence": self._snapshot_sequence,
            "config_generation": self._config_generation,
            "queue_control_config_fingerprint": self._config_fingerprint,
            "discipline": self.discipline,
            "max_active_requests": self.max_active_requests,
            "max_waiting_requests": self.max_waiting_requests,
            "class_limits": dict(self.class_limits),
            "class_limit_mode": self.class_limit_mode,
            "trust_request_metadata": self.trust_request_metadata,
            "online_allocator": (
                None
                if self.online_allocator is None
                else self.online_allocator.to_snapshot()
            ),
            "active_requests": len(self._active),
            "waiting_requests": len(self._waiting),
            "active_by_class": dict(self._active_by_class),
            "waiting_by_class": dict(waiting_by_class),
            "waiting_rejected_total": self._waiting_rejected_total,
            "recheck_rejections_by_class_total": dict(
                self._recheck_rejections_by_class_total
            ),
            "soft_reservation": {
                "reserved_dispatch_total": self._soft_reservation_reserved_dispatch_total,
                "borrowed_dispatch_total": self._soft_reservation_borrowed_dispatch_total,
                "blocked_total": self._soft_reservation_blocked_total,
            },
            "admission": {
                "enabled": self.admission.enabled,
                "enforce": self.admission.enforce,
                "score_method": self.admission.score_method,
                "classes": {
                    request_class: {
                        "effective_k": class_config.effective_k,
                        "mu": class_config.mu,
                        "gamma": class_config.gamma,
                        "service_sample_count": len(class_config.service_samples_s),
                        **(
                            {
                                "max_required_returns": (
                                    class_config.max_required_returns
                                ),
                                "threshold_entry_count": len(
                                    self.admission.threshold_tables[
                                        request_class
                                    ].thresholds_s
                                ),
                                "threshold_profile_fingerprint": (
                                    self.admission.threshold_tables[
                                        request_class
                                    ].profile_fingerprint
                                ),
                                "threshold_table_digest": (
                                    self.admission.threshold_tables[
                                        request_class
                                    ].table_digest
                                ),
                            }
                            if self.admission.score_method
                            == "erlang_empirical_threshold"
                            else {}
                        ),
                    }
                    for request_class, class_config in sorted(
                        self.admission.classes.items()
                    )
                },
                "admitted_total": self._admission_admitted_total,
                "rejected_total": self._admission_rejected_total,
                "actual_rejected_total": self._admission_rejected_total,
                "recheck_passed_total": self._admission_recheck_passed_total,
                "would_admit_decisions_total": (
                    self._admission_would_admit_decisions_total
                ),
                "would_reject_decisions_total": (
                    self._admission_would_reject_decisions_total
                ),
                "shadow_would_reject_decisions_total": (
                    self._admission_shadow_would_reject_decisions_total
                ),
                "decision_reason_counts": dict(
                    sorted(self._admission_reason_counts.items())
                ),
                "decision_sequence": self._admission_decision_sequence,
                "recent_decision_capacity": ADMISSION_DECISION_HISTORY_LIMIT,
                "recent_decision_first_sequence": (
                    recent_admission_decisions[0]["decision_sequence"]
                    if recent_admission_decisions
                    else None
                ),
                "recent_decision_last_sequence": (
                    recent_admission_decisions[-1]["decision_sequence"]
                    if recent_admission_decisions
                    else None
                ),
                "recent_decision_overwritten_total": max(
                    self._admission_decision_sequence - len(recent_admission_decisions),
                    0,
                ),
                "recent_decisions": recent_admission_decisions,
            },
        }

    def _can_dispatch_immediately(self, request_class: str) -> bool:
        """Return whether a new request can acquire a credit without waiting.

        The queue is drained after every state transition, so an available
        global and class credit cannot be owed to an older eligible waiter.
        """
        return self._global_credit_available() and self._class_credit_available(
            request_class,
            candidate=None,
        )

    def _drain(self) -> tuple[RuntimeQueueItem[ItemT], ...]:
        dispatched: list[RuntimeQueueItem[ItemT]] = []
        while self._global_credit_available():
            index = self._next_eligible_index()
            if index is None:
                break
            item = self._waiting.pop(index)
            self._record_class_dispatch(item)
            self._active[item.request_id] = item
            self._active_by_class[item.request_class] += 1
            dispatched.append(item)
        return tuple(dispatched)

    def _global_credit_available(self) -> bool:
        return (
            self.max_active_requests is None
            or len(self._active) < self.max_active_requests
        )

    def _class_credit_available(
        self,
        request_class: str,
        *,
        candidate: RuntimeQueueItem[ItemT] | None,
    ) -> bool:
        limit = self.class_limits.get(request_class)
        if self.class_limit_mode == "hard_limit":
            return limit is None or self._active_by_class[request_class] < limit

        reservation = 0 if limit is None else limit
        if self._active_by_class[request_class] < reservation:
            return True
        if self._has_under_reserved_waiter(excluding=candidate):
            self._soft_reservation_blocked_total += 1
            return False
        return True

    def _has_under_reserved_waiter(
        self,
        *,
        excluding: RuntimeQueueItem[ItemT] | None,
    ) -> bool:
        if self.class_limit_mode != "soft_reservation":
            return False
        excluded_sequence = None if excluding is None else excluding.sequence
        for item in self._waiting:
            if item.sequence == excluded_sequence:
                continue
            reservation = self.class_limits.get(item.request_class, 0)
            if reservation <= 0:
                continue
            if self._active_by_class[item.request_class] < reservation:
                return True
        return False

    def _record_class_dispatch(self, item: RuntimeQueueItem[ItemT]) -> None:
        if self.class_limit_mode != "soft_reservation":
            return
        reservation = self.class_limits.get(item.request_class, 0)
        if self._active_by_class[item.request_class] < reservation:
            self._soft_reservation_reserved_dispatch_total += 1
        else:
            self._soft_reservation_borrowed_dispatch_total += 1

    def _next_eligible_index(self) -> int | None:
        eligible = [
            (index, item)
            for index, item in enumerate(self._waiting)
            if self._class_credit_available(item.request_class, candidate=item)
        ]
        if not eligible:
            return None
        if self.discipline == "fifo":
            return min(eligible, key=lambda pair: pair[1].sequence)[0]
        return min(
            eligible,
            key=lambda pair: (
                math.inf
                if pair[1].deadline_unix_s is None
                else pair[1].deadline_unix_s,
                pair[1].sequence,
            ),
        )[0]

    @staticmethod
    def _admission_order_key(item: RuntimeQueueItem[Any]) -> tuple[float, int]:
        deadline = item.deadline_unix_s
        return (math.inf if deadline is None else deadline, item.sequence)

    def _admission_candidates(
        self, request_class: str
    ) -> list[RuntimeQueueItem[ItemT]]:
        return sorted(
            (
                item
                for item in self._waiting
                if item.request_class == request_class
                and item.request_id not in self._active
            ),
            key=self._admission_order_key,
        )

    def _evaluate_admission(
        self,
        item: RuntimeQueueItem[ItemT],
        *,
        queue_position: int,
        phase: Literal["arrival", "recheck"],
        now_unix_s: float | None = None,
    ) -> AdmissionDecision | None:
        admission = self.admission
        if not admission.enabled:
            return None
        class_config = admission.classes.get(item.request_class)
        if class_config is None:
            return None

        active_count = self._active_by_class[item.request_class]
        deadline = item.deadline_unix_s
        now = self._clock() if now_unix_s is None else now_unix_s
        remaining_budget_s = None if deadline is None else deadline - now
        required_returns = None
        threshold_s = None
        threshold_slack_s = None
        threshold_table_digest = (
            admission.threshold_tables[item.request_class].table_digest
            if admission.score_method == "erlang_empirical_threshold"
            else None
        )

        if class_config.effective_k == 0:
            score = 0.0
            reason = "zero_effective_k"
            would_admit = False
        elif active_count > class_config.effective_k:
            score = 0.0
            reason = "active_above_effective_k"
            would_admit = False
        elif deadline is None:
            score = None
            reason = "no_deadline"
            would_admit = True
        elif remaining_budget_s is not None and remaining_budget_s < 0:
            score = 0.0
            reason = "deadline_expired"
            would_admit = False
        else:
            assert remaining_budget_s is not None
            required_returns = max(
                active_count + queue_position - class_config.effective_k + 1,
                0,
            )
            if admission.score_method == "erlang_empirical":
                score = erlang_empirical_admission_score(
                    remaining_budget_s,
                    effective_k=class_config.effective_k,
                    mu=class_config.mu,
                    active_count=active_count,
                    queue_position=queue_position,
                    service_samples_s=class_config.service_samples_s,
                )
                would_admit = score >= class_config.gamma
                reason = "score_pass" if would_admit else "score_below_gamma"
            else:
                threshold_table = admission.threshold_tables[item.request_class]
                score = None
                if class_config.gamma == 0.0:
                    threshold_s = 0.0
                    threshold_slack_s = remaining_budget_s
                    would_admit = True
                    reason = "score_pass"
                elif required_returns > threshold_table.max_required_returns:
                    would_admit = False
                    reason = "threshold_table_exhausted"
                else:
                    threshold_s = threshold_table.thresholds_s[required_returns]
                    would_admit = remaining_budget_s >= threshold_s
                    threshold_slack_s = remaining_budget_s - threshold_s
                    reason = "score_pass" if would_admit else "score_below_gamma"

        return AdmissionDecision(
            admitted=would_admit or not admission.enforce,
            would_admit=would_admit,
            enforced=admission.enforce,
            request_id=item.request_id,
            admission_correlation_id=item.admission_correlation_id,
            request_class=item.request_class,
            phase=phase,
            score=score,
            gamma=class_config.gamma,
            reason=reason,
            effective_k=class_config.effective_k,
            mu=class_config.mu,
            active_count=active_count,
            queue_position=queue_position,
            remaining_budget_s=remaining_budget_s,
            required_returns=required_returns,
            threshold_s=threshold_s,
            threshold_slack_s=threshold_slack_s,
            score_method=admission.score_method,
            threshold_table_digest=threshold_table_digest,
        )

    def _evaluate_arrival_admission(
        self,
        item: RuntimeQueueItem[ItemT],
    ) -> AdmissionDecision | None:
        request_class = item.request_class
        item_key = self._admission_order_key(item)
        now = self._clock()
        queue_position = sum(
            1
            for candidate in self._waiting
            if candidate.request_class == request_class
            and self._admission_order_key(candidate) < item_key
        )
        decision = self._evaluate_admission(
            item,
            queue_position=queue_position,
            phase="arrival",
            now_unix_s=now,
        )
        if decision is None or decision.admitted:
            return decision

        queue_position = 0
        for candidate in sorted(
            (
                candidate
                for candidate in self._waiting
                if candidate.request_class == request_class
                and self._admission_order_key(candidate) < item_key
            ),
            key=self._admission_order_key,
        ):
            prior_decision = self._evaluate_admission(
                candidate,
                queue_position=queue_position,
                phase="recheck",
                now_unix_s=now,
            )
            if prior_decision is None or prior_decision.admitted:
                queue_position += 1
        return self._evaluate_admission(
            item,
            queue_position=queue_position,
            phase="arrival",
            now_unix_s=now,
        )

    def _record_admission_decision(self, decision: AdmissionDecision) -> None:
        self._admission_decision_sequence += 1
        if decision.would_admit:
            self._admission_would_admit_decisions_total += 1
        else:
            self._admission_would_reject_decisions_total += 1
            if not decision.enforced:
                self._admission_shadow_would_reject_decisions_total += 1
        self._admission_reason_counts[decision.reason] += 1
        self._recent_admission_decisions.append(
            {
                "decision_sequence": self._admission_decision_sequence,
                "request_id": decision.request_id,
                "admission_correlation_id": decision.admission_correlation_id,
                "request_class": decision.request_class,
                "phase": decision.phase,
                "admitted": decision.admitted,
                "would_admit": decision.would_admit,
                "enforced": decision.enforced,
                "score": decision.score,
                "gamma": decision.gamma,
                "reason": decision.reason,
                "effective_k": decision.effective_k,
                "mu": decision.mu,
                "active_count": decision.active_count,
                "queue_position": decision.queue_position,
                "remaining_budget_s": decision.remaining_budget_s,
                "required_returns": decision.required_returns,
                "threshold_s": decision.threshold_s,
                "threshold_slack_s": decision.threshold_slack_s,
                "score_method": decision.score_method,
                "threshold_table_digest": decision.threshold_table_digest,
            }
        )

    def recheck_admission(self) -> list[RuntimeQueueAdmissionRejection[ItemT]]:
        """Re-evaluate waiting requests after queue/config changes."""

        start_ns = self._timing_recorder.start_ns()
        try:
            return self._recheck_admission_impl()
        finally:
            self._timing_recorder.record_elapsed_ns("queue_recheck_ns", start_ns)

    def _recheck_admission_impl(self) -> list[RuntimeQueueAdmissionRejection[ItemT]]:
        """Re-evaluate waiting requests after queue/config changes."""

        if not self.admission.enabled:
            return []
        rejected: list[RuntimeQueueAdmissionRejection[ItemT]] = []
        rejected_sequences: set[int] = set()
        classes = {item.request_class for item in self._waiting}
        for request_class in sorted(classes):
            queue_position = 0
            now = self._clock()
            for item in self._admission_candidates(request_class):
                decision = self._evaluate_admission(
                    item,
                    queue_position=queue_position,
                    phase="recheck",
                    now_unix_s=now,
                )
                if decision is None:
                    queue_position += 1
                    continue
                self._record_admission_decision(decision)
                if decision.admitted:
                    self._admission_recheck_passed_total += 1
                    queue_position += 1
                    continue
                self._admission_rejected_total += 1
                self._recheck_rejections_by_class_total[request_class] += 1
                rejected_sequences.add(item.sequence)
                rejected.append(
                    RuntimeQueueAdmissionRejection(item=item, decision=decision)
                )
        if rejected_sequences:
            self._waiting = [
                item
                for item in self._waiting
                if item.sequence not in rejected_sequences
            ]
        return rejected

    def _parse_metadata(
        self, metadata: Mapping[str, Any] | None
    ) -> tuple[str, float | None, str | None]:
        metadata = metadata or {}
        if not self.trust_request_metadata:
            return DEFAULT_REQUEST_CLASS, None, None
        raw_class = metadata.get(self.class_metadata_key, DEFAULT_REQUEST_CLASS)
        if not isinstance(raw_class, str) or not raw_class.strip():
            raise ValueError(
                f"request metadata {self.class_metadata_key!r} must be a "
                "non-empty string"
            )
        request_class = raw_class.strip()

        raw_deadline = metadata.get(self.deadline_metadata_key)
        raw_correlation_id = metadata.get(self.admission_correlation_id_metadata_key)
        admission_correlation_id = (
            None
            if raw_correlation_id is None
            else _label(
                raw_correlation_id,
                default="",
                field_name=self.admission_correlation_id_metadata_key,
            )
        )
        if raw_deadline is None:
            return request_class, None, admission_correlation_id
        if isinstance(raw_deadline, bool):
            raise ValueError(
                f"request metadata {self.deadline_metadata_key!r} must be a "
                "finite Unix timestamp in seconds"
            )
        try:
            deadline_unix_s = float(raw_deadline)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"request metadata {self.deadline_metadata_key!r} must be a "
                "finite Unix timestamp in seconds"
            ) from exc
        if not math.isfinite(deadline_unix_s):
            raise ValueError(
                f"request metadata {self.deadline_metadata_key!r} must be a "
                "finite Unix timestamp in seconds"
            )
        return request_class, deadline_unix_s, admission_correlation_id
