# SPDX-License-Identifier: Apache-2.0
import argparse
import asyncio
import json

import http_experiments as experiments
from start_servers import RUN, start

original_payload = experiments.speech_payload


def controlled_payload(index=0, stream=True):
    return original_payload(index, stream) | {
        "seed": 910000 + index,
        "max_new_tokens": 32,
        "stage_params": {
            "tts_engine": {"do_sample": False, "subtalker_dosample": False}
        },
    }


async def main(reuse_servers=False):
    output = RUN / "tts-seeded-control"
    output.mkdir(exist_ok=True)
    experiments.RUN = output
    experiments.speech_payload = controlled_payload
    if reuse_servers:
        receipts = [
            json.loads((RUN / f"tts-{mode}-seeded-control-process.json").read_text())
            for mode in experiments.MODES
        ]
    else:
        receipts = [
            start("tts", mode, 3 + index, 22103 + index, f"tts-{mode}-seeded-control")
            for index, mode in enumerate(experiments.MODES)
        ]
    summaries = []
    try:
        await asyncio.gather(
            *(experiments.ready(receipt["port"]) for receipt in receipts)
        )
        await asyncio.sleep(10)
        for seed in (8201, 8202, 8203):
            for receipt in receipts:
                receipt["label"] = f'tts-{receipt["mode"]}-seeded-{seed}'
            summaries += await asyncio.gather(
                *(experiments.bench(receipt, seed, count=128) for receipt in receipts)
            )
            (output / "summaries.json").write_text(
                json.dumps(summaries, indent=2) + "\n"
            )
        print("CONTROL_COMPLETE", flush=True)
    finally:
        await experiments.stop(receipts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-servers", action="store_true")
    arguments = parser.parse_args()
    asyncio.run(main(arguments.reuse_servers))
