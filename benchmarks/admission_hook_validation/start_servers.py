# SPDX-License-Identifier: Apache-2.0
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs/pr1-validation"


def start(model, mode, gpu, port, label):
    source = RUN / "sources" / ("base" if mode == "base" else "head")
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": f"{source}:{RUN}/sources/sglang/python:{RUN}",
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "CUDA_HOME": "/usr/local/cuda-13.2",
            "LD_LIBRARY_PATH": f"{ROOT}/.venv-sglang/lib/python3.12/site-packages/nvidia/cudnn/lib:{ROOT}/.venv-sglang/lib/python3.12/site-packages/nvidia/cu13/lib",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "SGLANG_OMNI_STARTUP_TIMEOUT": "1800",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    if model == "whisper":
        config = {
            "config_cls": "WhisperASRPipelineConfig",
            "model_path": str(ROOT / "models/whisper-large-v3"),
            "stages": {
                "asr": {
                    "gpu": 0,
                    "engine": {
                        "mem_fraction_static": 0.3,
                        "cuda_graph_max_bs": 8,
                        "max_running_requests": 8,
                    },
                }
            },
        }
        served_model = "openai/whisper-large-v3"
    else:
        config = {
            "config_cls": "Qwen3TTSPipelineConfig",
            "model_path": str(ROOT / "models/qwen3-tts-base"),
            "stages": {
                "tts_engine": {
                    "gpu": 0,
                    "engine": {
                        "mem_fraction_static": 0.3,
                        "cuda_graph_max_bs": 8,
                        "max_running_requests": 8,
                    },
                },
                "vocoder": {"gpu": 0},
            },
        }
        served_model = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    if mode in ("always", "reject"):
        config["admission_policy"] = "validation_policies." + (
            "always_admit" if mode == "always" else "reject_all"
        )
    path = RUN / "configs" / f"{label}.json"
    path.write_text(json.dumps(config, indent=2) + "\n")
    argv = [
        str(ROOT / ".venv-sglang/bin/python"),
        "-m",
        "sglang_omni.cli",
        "serve",
        "--config",
        str(path),
        "--model-name",
        served_model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    log_path = RUN / "logs" / f"{label}.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(
            argv,
            cwd=source,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    receipt = {
        "pid": process.pid,
        "model": model,
        "mode": mode,
        "gpu": gpu,
        "port": port,
        "label": label,
        "argv": argv,
        "config": config,
        "source": str(source),
        "log": str(log_path),
    }
    (RUN / f"{label}-process.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    print(
        json.dumps(
            [
                start(
                    model,
                    mode,
                    offset + index,
                    22100 + offset + index,
                    f"{model}-{mode}-wave0",
                )
                for model, offset in [("whisper", 0), ("tts", 3)]
                for index, mode in enumerate(("base", "unset", "always"))
            ],
            indent=2,
        )
    )
