"""Exercise text delivery and audio deadlines while engine output is idle."""

import asyncio
from contextlib import aclosing
from types import SimpleNamespace
import time

import numpy as np
import unittest

from sglang_omni.client.playback import policy_audio_stream
from sglang_omni.client.types import GenerateChunk


class Policy:
    def __init__(self):
        self.deadline = time.monotonic() + .05

    def audio_release(self, request_id, duration, *, stream_ended=False):
        return SimpleNamespace(release=stream_ended or time.monotonic() >= self.deadline)

    def audio_deadline(self, request_id):
        return self.deadline


class PlaybackTests(unittest.IsolatedAsyncioTestCase):
    async def test_text_passes_held_audio_and_timer_fires_without_another_chunk(self):
        finish = asyncio.Event()
        closed = []

        async def source():
            try:
                yield GenerateChunk(request_id="x", modality="audio", audio_data=np.ones(10), sample_rate=100)
                yield GenerateChunk(request_id="x", modality="text", text="answer")
                await finish.wait()
            finally:
                closed.append(True)

        policy = Policy()
        async with aclosing(policy_audio_stream(source(), policy, "x")) as stream:
            assert (await anext(stream)).text == "answer"
            assert time.monotonic() < policy.deadline
            assert (await asyncio.wait_for(anext(stream), 1)).modality == "audio"
            assert time.monotonic() >= policy.deadline
            finish.set()
        assert closed == [True]


    async def test_early_eos_flushes_whole_chunks_in_order(self):
        async def source():
            for i in (1, 2):
                yield GenerateChunk(request_id="x", modality="audio", audio_data=np.ones(i), sample_rate=100)

        policy = Policy()
        policy.deadline = time.monotonic() + 100
        result = [x async for x in policy_audio_stream(source(), policy, "x")]
        assert [x.audio_data.size for x in result] == [1, 2]
