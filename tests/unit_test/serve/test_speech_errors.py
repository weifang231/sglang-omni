# SPDX-License-Identifier: Apache-2.0
"""Speech API error mapping helpers."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from sglang_omni.admission import AdmissionRejectedError, QueueFullError
from sglang_omni.client.types import ClientError
from sglang_omni.serve import create_app
from sglang_omni.serve.speech_errors import speech_generation_error


@pytest.mark.parametrize(
    "exc",
    [QueueFullError(), RuntimeError(QueueFullError.MESSAGE)],
)
def test_speech_generation_error_maps_queue_full_to_503(exc: BaseException) -> None:
    err = speech_generation_error(exc)
    assert err.status_code == 503
    assert QueueFullError.MESSAGE in err.message


def test_speech_generation_error_keeps_other_failures_as_500() -> None:
    err = speech_generation_error(RuntimeError("cuda out of memory"))
    assert err.status_code == 500
    assert "cuda out of memory" in err.message


@pytest.mark.parametrize(
    "message",
    [
        "The request is longer than the model's context length",
        "Requested token count exceeds the model's maximum context length",
        "Request requires more tokens than the thinker KV cache can hold",
        "Request req-1 exceeds the maximum number of tokens: 8193 > 8192",
        "Request req-1 requires too many SWA KV tokens for decode preallocation",
    ],
)
def test_speech_generation_error_maps_context_rejection_to_400(
    message: str,
) -> None:
    err = speech_generation_error(RuntimeError(message))

    assert err.status_code == 400
    assert err.error_type == "BadRequestError"
    assert err.code == 400
    assert err.message == message


def test_speech_generation_error_does_not_match_unrelated_token_message() -> None:
    err = speech_generation_error(
        RuntimeError(
            "kernel assertion: Request req-1 exceeds the maximum number of "
            "tokens: temporary buffer"
        )
    )

    assert err.status_code == 500
    assert err.error_type == "server_error"


@pytest.mark.parametrize("params", [{}, {"nfe": 1}, {"gen_seconds": -1}])
def test_auk_validation_reaches_http_as_bad_request(params, caplog):
    from fastapi.testclient import TestClient

    from sglang_omni.client.client import Client
    from sglang_omni.client.types import ClientError
    from sglang_omni.models.auk.hf_config import AuKRuntimeConfig
    from sglang_omni.models.auk.request_builders import build_auk_state
    from sglang_omni.proto import StagePayload
    from sglang_omni.serve import create_app

    class PreprocessingClient:
        async def speech(self, request, *, request_id, **kwargs):
            payload = StagePayload(
                request_id=request_id,
                request=Client.build_omni_request(request),
                data={},
            )
            try:
                build_auk_state(payload, AuKRuntimeConfig(model_path="unused"))
            except ValueError as error:
                raise ClientError(str(error)) from error
            raise AssertionError("invalid request reached generation")

    client = TestClient(create_app(PreprocessingClient(), model_name="tencent/AuK"))
    response = client.post(
        "/v1/audio/speech",
        json={
            "input": "Hello.",
            "stage_params": {"auk_engine": params},
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "BadRequestError"
    assert "AuK" in response.json()["error"]["message"]
    assert not any(record.exc_info for record in caplog.records)


@pytest.mark.parametrize(
    "exc",
    [
        AdmissionRejectedError(),
        AdmissionRejectedError("capacity"),
        RuntimeError(AdmissionRejectedError.MESSAGE),
    ],
)
def test_admission_rejection_maps_to_429(exc: BaseException) -> None:
    mapped = speech_generation_error(exc)
    assert mapped.status_code == 429
    assert mapped.error_type == "rate_limit_error"
    assert mapped.code == "admission_rejected"
    assert AdmissionRejectedError.MESSAGE in mapped.message


class RejectingSpeechClient:
    async def speech(self, request, **kwargs):
        raise ClientError(str(AdmissionRejectedError("capacity")))

    async def generate(self, request, **kwargs):
        raise ClientError(str(AdmissionRejectedError("capacity")))
        yield

    async def abort(self, request_id):
        return None


@pytest.mark.parametrize("stream_format", [None, "audio", "sse"])
def test_admission_rejection_reaches_speech_http(stream_format, caplog) -> None:
    client = TestClient(create_app(RejectingSpeechClient(), model_name="test"))
    payload = {"input": "Hello.", "response_format": "pcm"}
    if stream_format is not None:
        payload.update(stream=True, stream_format=stream_format)
    response = client.post("/v1/audio/speech", json=payload)
    assert response.status_code == 429
    assert response.json()["error"]["type"] == "rate_limit_error"
    assert response.json()["error"]["code"] == "admission_rejected"
    assert "capacity" in response.json()["error"]["message"]
    assert not any(record.exc_info for record in caplog.records)


def test_admission_rejection_reaches_each_batch_http_item() -> None:
    client = TestClient(create_app(RejectingSpeechClient(), model_name="test"))
    response = client.post(
        "/v1/audio/speech/batch",
        json={"items": [{"input": "Hello."}, {"input": "World."}]},
    )
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 2
    for result in results:
        assert result["status"] == "error"
        assert result["error"]["type"] == "rate_limit_error"
        assert result["error"]["code"] == "admission_rejected"


def silence_wav(duration_s: float = 0.5, sample_rate: int = 16000) -> bytes:
    import io
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * int(duration_s * sample_rate))
    return buffer.getvalue()


class RejectingTranscriptionClient:
    def health(self):
        return {"running": True}

    async def completion(self, request, *, request_id, audio_format="wav"):
        raise ClientError(str(AdmissionRejectedError("capacity")))

    async def generate(self, request, request_id=None, **kwargs):
        raise ClientError(str(AdmissionRejectedError("capacity")))
        yield

    async def abort(self, request_id):
        return None


@pytest.mark.parametrize("stream", [False, True])
def test_admission_rejection_reaches_transcription_http(stream, caplog) -> None:
    client = TestClient(create_app(RejectingTranscriptionClient(), model_name="asr"))
    data = {"model": "asr"}
    if stream:
        data["stream"] = "true"
    else:
        pass
    response = client.post(
        "/v1/audio/transcriptions",
        data=data,
        files={"file": ("short.wav", silence_wav(), "audio/wav")},
    )
    assert response.status_code == 429, response.text
    assert "Admission rejected" in response.json()["detail"]
    assert not any(record.exc_info for record in caplog.records)


class QueueFullTranscriptionClient(RejectingTranscriptionClient):
    async def completion(self, request, *, request_id, audio_format="wav"):
        raise ClientError(QueueFullError.MESSAGE)

    async def generate(self, request, request_id=None, **kwargs):
        raise ClientError(QueueFullError.MESSAGE)
        yield


@pytest.mark.parametrize("stream", [False, True])
def test_queue_full_reaches_transcription_http_as_503(stream) -> None:
    client = TestClient(create_app(QueueFullTranscriptionClient(), model_name="asr"))
    data = {"model": "asr", **({"stream": "true"} if stream else {})}
    response = client.post(
        "/v1/audio/transcriptions",
        data=data,
        files={"file": ("short.wav", silence_wav(), "audio/wav")},
    )
    assert response.status_code == 503, response.text
    assert QueueFullError.MESSAGE in response.json()["detail"]
