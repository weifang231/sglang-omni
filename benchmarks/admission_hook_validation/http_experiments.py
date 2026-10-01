# SPDX-License-Identifier: Apache-2.0
import argparse
import asyncio
import base64
import hashlib
import json
import os
import random
import signal
import time
from pathlib import Path

import httpx
import numpy as np
from start_servers import ROOT, RUN, start

MODES = ("base", "unset", "always")
metadata = [
    line.split("|")
    for line in (ROOT / "datasets/current/seed-tts/en/meta.lst")
    .read_text()
    .splitlines()
]
unique = {row[2]: row for row in metadata}
samples = list(unique.values())[:64]
audio = [
    (ROOT / "datasets/current/seed-tts/en" / row[2]).read_bytes() for row in samples
]
reference = "data:audio/wav;base64," + base64.b64encode(audio[0]).decode()


async def ready(port, timeout=1800):
    deadline = time.monotonic() + timeout
    async with httpx.AsyncClient(timeout=3) as client:
        while time.monotonic() < deadline:
            try:
                response = await client.get(f"http://127.0.0.1:{port}/health")
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            receipts = [
                json.loads(path.read_text())
                for path in sorted(
                    RUN.glob("*-process.json"),
                    key=lambda path: path.stat().st_mtime,
                    reverse=True,
                )
            ]
            receipt = next(
                (receipt for receipt in receipts if receipt["port"] == port), None
            )
            if receipt is not None and not Path(f'/proc/{receipt["pid"]}').exists():
                raise RuntimeError(f'Server {port} exited; inspect {receipt["log"]}')
            await asyncio.sleep(2)
    raise RuntimeError(f"Server {port} did not become ready")


def speech_payload(index=0, stream=True):
    return {
        "input": samples[index % len(samples)][3],
        "ref_audio": reference,
        "ref_text": samples[0][1],
        "response_format": "pcm",
        "stream": stream,
        "language": "english",
        "temperature": 0.0,
    }


async def request(client, port, model, index):
    started = time.perf_counter()
    first = None
    header = None
    content_bytes = 0
    error = None
    kwargs = (
        {"json": speech_payload(index)}
        if model == "tts"
        else {
            "files": {"file": ("audio.wav", audio[index % len(audio)], "audio/wav")},
            "data": {
                "model": "openai/whisper-large-v3",
                "language": "en",
                "stream": "true",
                "response_format": "json",
                "temperature": "0",
            },
        }
    )
    endpoint = "/v1/audio/speech" if model == "tts" else "/v1/audio/transcriptions"
    status = None
    try:
        async with client.stream(
            "POST", f"http://127.0.0.1:{port}{endpoint}", **kwargs
        ) as response:
            status = response.status_code
            header = time.perf_counter() - started
            if status != 200:
                error = (await response.aread()).decode()[:2000]
            elif model == "tts":
                async for chunk in response.aiter_bytes():
                    if chunk:
                        content_bytes += len(chunk)
                        if first is None:
                            first = time.perf_counter() - started
            else:
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        continue
                    event = json.loads(payload)
                    if event.get("type") == "error" or "error" in event:
                        error = json.dumps(event)
                    delta = event.get("delta") or (
                        event.get("text")
                        if event.get("type") == "transcript.text.done"
                        else None
                    )
                    if delta:
                        content_bytes += len(str(delta))
                        if first is None:
                            first = time.perf_counter() - started
    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"
    elapsed = time.perf_counter() - started
    return {
        "index": index,
        "status": status,
        "headers_seconds": header,
        "first_seconds": first,
        "total_seconds": elapsed,
        "bytes": content_bytes,
        "error": error,
        "success": status == 200 and first is not None and error is None,
    }


async def rejection():
    await ready(22106)
    results = []
    async with httpx.AsyncClient(timeout=60) as client:
        for name, payload in [
            ("nonstream", speech_payload(stream=False)),
            ("stream_pcm", speech_payload()),
            ("stream_sse", speech_payload() | {"stream_format": "sse"}),
        ]:
            response = await client.post(
                "http://127.0.0.1:22106/v1/audio/speech", json=payload
            )
            results.append(
                {"case": name, "status": response.status_code, "body": response.json()}
            )
        item = speech_payload(stream=False)
        response = await client.post(
            "http://127.0.0.1:22106/v1/audio/speech/batch", json={"items": [item, item]}
        )
        results.append(
            {"case": "batch", "status": response.status_code, "body": response.json()}
        )
    (RUN / "http-rejection.json").write_text(json.dumps(results, indent=2) + "\n")
    print("HTTP_REJECTION", json.dumps(results), flush=True)


async def bench(receipt, seed, count=256):
    port = receipt["port"]
    model = receipt["model"]
    rate = 8 if model == "whisper" else 2
    await ready(port)
    async with httpx.AsyncClient(
        timeout=120,
        limits=httpx.Limits(max_connections=256, max_keepalive_connections=256),
    ) as client:
        for index in range(24):
            outcome = await request(client, port, model, index)
            if not outcome["success"]:
                raise RuntimeError(f'Warmup failed: {receipt["label"]}: {outcome}')
        random_generator = random.Random(seed)
        cumulative = 0.0
        arrivals = []
        indexes = []
        for index in range(count):
            cumulative += random_generator.expovariate(rate)
            arrivals.append(cumulative)
            indexes.append(random_generator.randrange(len(samples)))
        started = time.perf_counter()

        async def scheduled(index):
            target = started + arrivals[index]
            await asyncio.sleep(max(0, target - time.perf_counter()))
            sent = time.perf_counter()
            outcome = await request(client, port, model, indexes[index])
            return outcome | {
                "request_number": index,
                "scheduled_seconds": arrivals[index],
                "dispatch_lag_seconds": sent - target,
                "input_index": indexes[index],
            }

        outcomes = await asyncio.gather(*(scheduled(index) for index in range(count)))
        duration = time.perf_counter() - started
    good = [outcome for outcome in outcomes if outcome["success"]]
    summary = {
        "model": model,
        "mode": receipt["mode"],
        "gpu": receipt["gpu"],
        "seed": seed,
        "count": count,
        "rate_rps": rate,
        "successes": len(good),
        "duration_including_drain_seconds": duration,
        "completion_throughput_rps": len(good) / duration,
        "arrival_window_seconds": arrivals[-1],
        "successful_requests_per_arrival_window": len(good) / arrivals[-1],
    }
    for metric in (
        "first_seconds",
        "total_seconds",
        "headers_seconds",
        "dispatch_lag_seconds",
    ):
        summary[metric] = (
            {
                f"p{percentile}": float(
                    np.percentile([outcome[metric] for outcome in good], percentile)
                )
                for percentile in (50, 95, 99)
            }
            if good
            else {}
        )
    output = RUN / f'{receipt["label"]}-results.json'
    output.write_text(
        json.dumps(
            {"receipt": receipt, "summary": summary, "requests": outcomes}, indent=2
        )
        + "\n"
    )
    print("BENCH_DONE", json.dumps(summary), flush=True)
    return summary


async def stop(receipts):
    for receipt in receipts:
        try:
            os.killpg(receipt["pid"], signal.SIGTERM)
        except ProcessLookupError:
            pass
    await asyncio.sleep(10)
    for receipt in receipts:
        try:
            os.killpg(receipt["pid"], signal.SIGKILL)
        except ProcessLookupError:
            pass
    await asyncio.sleep(3)


async def campaign():
    summaries = []
    manifest = {
        "base_commit": "3a52817258b63d772c3e3c19dc9f1f234972b4cd",
        "head_commit": "04fb452d",
        "sglang_tag": "v0.5.20",
        "seed_tts_audio_count": len(samples),
        "audio_sha256": [hashlib.sha256(content).hexdigest() for content in audio],
        "protocol": {
            "whisper": "multipart SSE, first nonempty delta or final transcript; final-only responses are not token-level TTFT",
            "tts": "PCM streaming, first nonempty bytes",
        },
        "warmup_count": 24,
        "requests_per_model_mode_wave": 256,
        "seeds": [8101, 8102, 8103],
        "gpu_rotation": "Each mode runs once on each of GPUs 0-2 (ASR) or 3-5 (TTS).",
        "throughput_definition": "successful full completions / first dispatch timer to final completion, includes drain",
        "warnings": [
            "Fixed normal loads; this is not a saturation throughput or equivalence proof.",
            "Three waves share a host; CPU contention can affect tails.",
            "Repeated inputs and warm cache are identical across modes.",
        ],
    }
    (RUN / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for wave in range(3):
        if wave == 0:
            receipts = [
                json.loads((RUN / f"{model}-{mode}-wave0-process.json").read_text())
                for model in ("whisper", "tts")
                for mode in MODES
            ]
        else:
            receipts = [
                start(
                    model,
                    mode,
                    offset + (mode_index + wave) % 3,
                    22100 + offset + (mode_index + wave) % 3,
                    f"{model}-{mode}-wave{wave}",
                )
                for model, offset in [("whisper", 0), ("tts", 3)]
                for mode_index, mode in enumerate(MODES)
            ]
        try:
            await asyncio.gather(*(ready(receipt["port"]) for receipt in receipts))
            await asyncio.sleep(10)
            summaries += await asyncio.gather(
                *(bench(receipt, 8101 + wave) for receipt in receipts)
            )
            (RUN / "summaries.json").write_text(json.dumps(summaries, indent=2) + "\n")
        finally:
            await stop(receipts)
    print("CAMPAIGN_COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=["rejection", "campaign"])
    arguments = parser.parse_args()
    asyncio.run(rejection() if arguments.experiment == "rejection" else campaign())
