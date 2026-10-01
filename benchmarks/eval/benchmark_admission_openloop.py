# SPDX-License-Identifier: Apache-2.0
"""Open-loop (Poisson arrival) load against the speech endpoints, scored by SLO-goodput.

The closed-loop sweeps in this directory hold a fixed concurrency, so they cannot show
what happens when offered load exceeds capacity: every request still completes, only
later. This script sends ``--requests`` requests at a fixed Poisson rate regardless of
how the server keeps up, and reports how many of them produced their first output
within ``--slo-s`` (attainment), that count per second of offered load (goodput), and
how many were rejected with HTTP 429 (admission policy) or 503 (queue full).

    python -m benchmarks.eval.benchmark_admission_openloop --task asr --port 8000 \\
        --model-path openai/whisper-large-v3 --dataset seedtts-50 \\
        --rate 48 --requests 512 --slo-s 0.5 --output run.json

``--task asr`` uploads clips to ``/v1/audio/transcriptions`` (``--stream`` measures text
TTFT over SSE; otherwise first output = completion); ``--task tts`` streams PCM from
``/v1/audio/speech`` and measures time to the first streamed chunk, or, with ``--no-stream``,
the complete response (a completion deadline). ``--seed`` fixes the arrival
times and the clip order so native and policy runs see the same traffic.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time

import aiohttp

from benchmarks.dataset.prepare import DATASETS
from benchmarks.dataset.seedtts import load_seedtts_samples
from benchmarks.tasks.asr import make_asr_send_fn
from benchmarks.tasks.tts import make_tts_send_fn


def classify(result) -> str:
    if result.is_success:
        return "ok"
    elif result.error.startswith("HTTP 429"):
        return "rejected"
    elif result.error.startswith("HTTP 503"):
        return "queue_full"
    else:
        return "error"


def first_output_s(result, task: str, stream: bool) -> float | None:
    if not result.is_success:
        return None
    elif task == "tts" and stream:
        return result.audio_ttfp_s
    elif task == "tts":
        return result.latency_s
    elif stream:
        return result.text_ttft_s
    else:
        return result.latency_s


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    else:
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, int(round((len(ordered) - 1) * p)))]


async def run(args) -> dict:
    source = DATASETS.get(args.dataset, args.dataset)
    samples = load_seedtts_samples(source, args.max_samples or None, split=args.lang)
    if not samples:
        raise RuntimeError(f"no samples from {args.dataset!r}")
    else:
        pass
    api = f"http://{args.host}:{args.port}/v1/audio/"
    if args.task == "asr":
        send = make_asr_send_fn(
            args.model_path, api + "transcriptions", lang=args.lang, stream=args.stream
        )
    else:
        send = make_tts_send_fn(
            args.model_path,
            api + "speech",
            stream=not args.no_stream,
            response_format="wav",
        )
    rng = random.Random(args.seed)
    order = [rng.randrange(len(samples)) for _ in range(args.requests)]
    gaps = [rng.expovariate(args.rate) for _ in range(args.requests)]
    timeout = aiohttp.ClientTimeout(total=args.timeout_s)
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        for sample in samples[: args.warmup]:
            await send(session, sample)
        records: list[dict] = [None] * args.requests
        started = time.perf_counter()

        async def one(index: int, due: float) -> None:
            await asyncio.sleep(max(0.0, due - (time.perf_counter() - started)))
            sent = time.perf_counter() - started
            result = await send(session, samples[order[index]])
            records[index] = {
                "index": index,
                "sample_id": result.request_id,
                "arrival_s": round(due, 6),
                "sent_s": round(sent, 6),
                "completed_s": round(time.perf_counter() - started, 6),
                "status": classify(result),
                "first_output_s": first_output_s(
                    result,
                    args.task,
                    args.stream if args.task == "asr" else not args.no_stream,
                ),
                "latency_s": round(result.latency_s, 6),
                "error": result.error[:200] if result.error else None,
            }

        due, tasks = 0.0, []
        for index in range(args.requests):
            due += gaps[index]
            tasks.append(asyncio.create_task(one(index, due)))
        await asyncio.gather(*tasks)
    stream_mode = args.stream if args.task == "asr" else not args.no_stream
    statuses = [r["status"] for r in records]
    firsts = [r["first_output_s"] for r in records if r["first_output_s"] is not None]
    attained = sum(1 for f in firsts if f <= args.slo_s)
    span = max(r["completed_s"] for r in records) - min(r["arrival_s"] for r in records)
    summary = {
        "task": args.task,
        "stream": stream_mode,
        "rate_rps": args.rate,
        "requests": args.requests,
        "slo_s": args.slo_s,
        "seed": args.seed,
        "offered_span_s": round(span, 3),
        "ok": statuses.count("ok"),
        "rejected": statuses.count("rejected"),
        "queue_full": statuses.count("queue_full"),
        "error": statuses.count("error"),
        "attained": attained,
        "attainment": attained / args.requests,
        "goodput_rps": attained / span if span > 0 else 0.0,
        "first_output_p50_s": percentile(firsts, 0.5),
        "first_output_p95_s": percentile(firsts, 0.95),
        "late_rate": (len(firsts) - attained) / args.requests,
    }
    return {"summary": summary, "config": vars(args), "requests": records}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--task", choices=("asr", "tts"), default="asr")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model-path", required=True, help="served model name")
    parser.add_argument(
        "--dataset", default="seedtts-50", help="registered alias, repo id or meta.lst"
    )
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--lang", default="en")
    parser.add_argument(
        "--rate", type=float, required=True, help="offered load, requests per second"
    )
    parser.add_argument("--requests", type=int, default=512)
    parser.add_argument(
        "--slo-s", type=float, required=True, help="first-output deadline"
    )
    parser.add_argument("--stream", action="store_true", help="asr: SSE and text TTFT")
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="tts: non-streaming, first output = completion",
    )
    parser.add_argument(
        "--warmup", type=int, default=4, help="sequential untimed requests first"
    )
    parser.add_argument("--seed", type=int, default=701)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    report = asyncio.run(run(args))
    s = report["summary"]
    print(
        f"rate={s['rate_rps']:g} req/s n={s['requests']} slo={s['slo_s']}s | ok={s['ok']} "
        f"rejected={s['rejected']} queue_full={s['queue_full']} error={s['error']} | "
        f"attainment={s['attainment']:.3f} goodput={s['goodput_rps']:.2f} req/s | "
        f"first-output p50={s['first_output_p50_s']} p95={s['first_output_p95_s']}"
    )
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=1)
    else:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
else:
    pass
