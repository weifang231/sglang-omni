"""Audio message contents must reach the encoder, including API inline media."""

import asyncio
import base64
import io
import wave
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sglang_omni.models.ming_omni.components import preprocessor as module
from sglang_omni.proto import OmniRequest, StagePayload


class Tokenizer:
    def convert_tokens_to_ids(self, token):
        return 12 if token == "<audioPatch>" else 99

    def encode(self, text, add_special_tokens=False):
        return [99] * len(text)


@pytest.fixture
def processor(monkeypatch):
    config = SimpleNamespace(
        audio_config=SimpleNamespace(ds_kernel_size=1, ds_stride=1),
        vision_config=SimpleNamespace(),
    )
    monkeypatch.setattr(module, "load_ming_config", lambda _: config)
    monkeypatch.setattr(module, "load_ming_tokenizer", lambda _: Tokenizer())
    monkeypatch.setattr(
        module, "_compute_mel_features_for_waveform",
        lambda *_: (torch.zeros(4, 80), 4, 2),
    )
    return module.MingPreprocessor("test-model")


@pytest.mark.parametrize("style", ["audio_url", "input_audio", "top_level"])
def test_audio_reaches_encoder_with_matching_placeholders(processor, monkeypatch, style):
    url = "data:audio/wav;base64,AA=="
    loaded = []

    def load(value, *, target_sample_rate, source_name):
        loaded.append(value)
        return np.zeros(16000, dtype=np.float32)

    monkeypatch.setattr(module, "load_audio", load)
    content = [{"type": "text", "text": "Answer the spoken question."}]
    if style == "audio_url":
        content.append({"type": style, style: {"url": url}})
    elif style == "input_audio":
        content.append({"type": style, style: {"data": "AA==", "format": "wav"}})
    inputs = [{"role": "user", "content": content}]
    if style == "top_level":
        inputs = {"messages": inputs, "audios": [url]}
    payload = StagePayload(request_id="audio", request=OmniRequest(inputs=inputs), data={})

    result = asyncio.run(processor(payload))

    assert loaded == [url]
    audio = result.data["encoder_inputs"]["audio_encoder"]
    assert tuple(audio["audio_feats"].shape) == (1, 4, 80)
    assert audio["audio_feats_lengths"].tolist() == [[4]]
    assert audio["audio_placeholder_loc_lens"][0, 0, 1].item() == 2
    assert result.data["prompt"]["input_ids"].eq(12).sum().item() == 2
    assert audio["cache_key"] == module.compute_audio_cache_key([url])


def test_failed_audio_decode_is_not_a_text_only_request(processor, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("decoder unavailable")

    monkeypatch.setattr(module, "load_audio", fail)
    inputs = [{"role": "user", "content": [
        {"type": "audio_url", "audio_url": {"url": "audio://missing"}},
    ]}]
    payload = StagePayload(request_id="failed", request=OmniRequest(inputs=inputs), data={})
    with pytest.raises(RuntimeError, match="decoder unavailable"):
        asyncio.run(processor(payload))


def test_inline_wav_uses_the_audio_decoder(processor, monkeypatch):
    samples = np.array([0, 8192, -8192, 16384], dtype="<i2")
    encoded = io.BytesIO()
    with wave.open(encoded, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(samples.tobytes())
    url = "data:audio/wav;base64," + base64.b64encode(encoded.getvalue()).decode()
    seen = []

    def mel(waveform, *args):
        seen.append(waveform)
        return torch.zeros(4, 80), 4, 2

    monkeypatch.setattr(module, "_compute_mel_features_for_waveform", mel)
    payload = StagePayload(
        request_id="wav",
        request=OmniRequest(inputs=[{"role": "user", "content": [
            {"type": "audio_url", "audio_url": {"url": url}},
        ]}]),
        data={},
    )
    asyncio.run(processor(payload))
    assert len(seen) == 1
    np.testing.assert_array_equal(seen[0], samples.astype(np.float32) / 32768)
