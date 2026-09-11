"""Buffer audio until its policy boundary while continuing to deliver text."""

import asyncio
from contextlib import aclosing, suppress
import time


async def policy_audio_stream(stream, policy, request_id):
    pending = None
    buffered, duration = [], 0.0
    released = False
    ended = False
    async with aclosing(stream):
        try:
            while not ended or buffered:
                if buffered:
                    decision = policy.audio_release(request_id, duration, stream_ended=ended)
                    if decision.release:
                        released = True
                        for chunk in buffered:
                            yield chunk
                        buffered.clear()
                        duration = 0.0
                if ended:
                    break
                if pending is None:
                    pending = asyncio.create_task(anext(stream))
                deadline = policy.audio_deadline(request_id) if buffered else None
                timeout = max(0.0, deadline - time.monotonic()) if deadline is not None else None
                done, _ = await asyncio.wait({pending}, timeout=timeout)
                if not done:
                    continue
                try:
                    chunk = pending.result()
                except StopAsyncIteration:
                    ended = True
                    continue
                finally:
                    pending = None
                if chunk.modality == "audio" and chunk.audio_data is not None and not released:
                    if not chunk.sample_rate:
                        raise ValueError("Playback requires an explicit decoder sample rate")
                    buffered.append(chunk)
                    duration += chunk.audio_data.size / chunk.sample_rate
                else:
                    yield chunk
        finally:
            if pending is not None:
                pending.cancel()
                with suppress(asyncio.CancelledError, StopAsyncIteration):
                    await pending
