# SPDX-License-Identifier: Apache-2.0
"""Shared streaming response helpers for OpenAI-compatible endpoints."""

from __future__ import annotations

import asyncio
import logging
import inspect
from collections.abc import AsyncIterator
from typing import Any

from fastapi.responses import StreamingResponse
from starlette.types import Receive, Scope, Send

logger = logging.getLogger(__name__)

STREAM_DONE_SENTINEL = "[DONE]"


class ClosableStreamingResponse(StreamingResponse):
    """Close the response body iterator at the ASGI ownership boundary."""

    def __init__(self, *args: Any, on_close: Any = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.on_close = on_close

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            try:
                await close_async_iterator_if_supported(self.body_iterator)
            except asyncio.CancelledError:
                logger.warning("Cancelled while closing streaming response body")
            except Exception:
                logger.warning("Failed to close streaming response body", exc_info=True)
            finally:
                if self.on_close is not None:
                    result = self.on_close()
                    if inspect.isawaitable(result):
                        await result


async def close_async_iterator_if_supported(stream: AsyncIterator[Any]) -> None:
    try:
        close = stream.aclose
    except AttributeError:
        return
    await close()
