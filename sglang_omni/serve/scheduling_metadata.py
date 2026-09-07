# SPDX-License-Identifier: Apache-2.0
"""Trusted scheduling header parsing for OpenAI-compatible endpoints."""

from __future__ import annotations

import math
import os
import time
from collections.abc import Mapping
from typing import Any

from sglang_omni.runtime_queue import (
    ADMISSION_CORRELATION_ID_METADATA_KEY,
    FIRST_OUTPUT_DEADLINE_METADATA_KEY,
    PLAYBACK_BUFFER_MS_METADATA_KEY,
    REQUEST_CLASS_METADATA_KEY,
)

REQUEST_CLASS_HEADER = "x-sglang-omni-request-class"
FIRST_OUTPUT_DEADLINE_MS_HEADER = "x-sglang-omni-first-output-deadline-ms"
ADMISSION_CORRELATION_ID_HEADER = "x-sglang-omni-admission-correlation-id"
PLAYBACK_BUFFER_MS_HEADER = "x-sglang-omni-playback-buffer-ms"
TRUST_SCHEDULING_HEADERS_ENV = "SGLANG_OMNI_TRUST_SCHEDULING_HEADERS"
MAX_PLAYBACK_BUFFER_MS = 5000.0


def scheduling_headers_trusted() -> bool:
    return os.environ.get(TRUST_SCHEDULING_HEADERS_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _label_header(value: Any, *, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty")
    if len(normalized) > 128 or any(ord(character) < 32 for character in normalized):
        raise ValueError(f"{field_name} must be at most 128 printable characters")
    return normalized


def _non_negative_header_float(
    value: Any,
    *,
    field_name: str,
    maximum: float | None = None,
) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field_name} must be a number") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{field_name} must be finite and non-negative")
    if maximum is not None:
        parsed = min(parsed, maximum)
    return parsed


def scheduling_metadata_from_headers(
    headers: Mapping[str, str] | None,
) -> dict[str, Any]:
    if headers is None or not scheduling_headers_trusted():
        return {}
    normalized_headers = {str(key).lower(): value for key, value in headers.items()}
    metadata: dict[str, Any] = {}
    if REQUEST_CLASS_HEADER in normalized_headers:
        metadata[REQUEST_CLASS_METADATA_KEY] = _label_header(
            normalized_headers[REQUEST_CLASS_HEADER],
            field_name=REQUEST_CLASS_HEADER,
        )
    if FIRST_OUTPUT_DEADLINE_MS_HEADER in normalized_headers:
        deadline_ms = _non_negative_header_float(
            normalized_headers[FIRST_OUTPUT_DEADLINE_MS_HEADER],
            field_name=FIRST_OUTPUT_DEADLINE_MS_HEADER,
        )
        metadata[FIRST_OUTPUT_DEADLINE_METADATA_KEY] = (
            time.time() + deadline_ms / 1000.0
        )
    if ADMISSION_CORRELATION_ID_HEADER in normalized_headers:
        metadata[ADMISSION_CORRELATION_ID_METADATA_KEY] = _label_header(
            normalized_headers[ADMISSION_CORRELATION_ID_HEADER],
            field_name=ADMISSION_CORRELATION_ID_HEADER,
        )
    if PLAYBACK_BUFFER_MS_HEADER in normalized_headers:
        metadata[PLAYBACK_BUFFER_MS_METADATA_KEY] = _non_negative_header_float(
            normalized_headers[PLAYBACK_BUFFER_MS_HEADER],
            field_name=PLAYBACK_BUFFER_MS_HEADER,
            maximum=MAX_PLAYBACK_BUFFER_MS,
        )
    return metadata
