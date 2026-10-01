#!/bin/bash
# usage: serve_tts.sh <label> <gpu> <port> <mode: native|record|shadow|apply> [profile.json] [rate_rps]
set -u
LABEL=$1; GPU=$2; PORT=$3; MODE=$4; PROFILE=${5:-}; RATE=${6:-}
ROOT=/scr/rucnyz/projects/aproj/omni; RUN=$ROOT/runs/pr2-demo; SRC=$ROOT/sglang-omni-upstream; SGL=$ROOT/runs/pr1-validation/sources/sglang/python
mkdir -p $RUN/logs $RUN/configs $RUN/events
python3 - "$LABEL" "$MODE" "$PROFILE" "$RATE" <<'PY'
import json, sys
label, mode, profile, rate = sys.argv[1:5]
root = "/scr/rucnyz/projects/aproj/omni"
cfg = {"config_cls": "Qwen3TTSPipelineConfig", "model_path": f"{root}/models/qwen3-tts-base",
       "stages": {"tts_engine": {"gpu": 0, "engine": {"mem_fraction_static": 0.3, "cuda_graph_max_bs": 32, "max_running_requests": 32}}, "vocoder": {"gpu": 0}}}
if mode != "native":
    opts = {"mode": mode, "record_path": f"{root}/runs/pr2-demo/events/{label}.jsonl"}
    if profile: opts["profile"] = profile
    if rate: opts["arrival_rate_rps"] = float(rate)
    cfg["admission_policy"] = "sglang_omni.admission_policies.capacity_table.make_policy"
    cfg["admission_policy_options"] = opts
json.dump(cfg, open(f"{root}/runs/pr2-demo/configs/{label}.json", "w"), indent=1)
PY
export PYTHONPATH=$SRC:$SGL CUDA_VISIBLE_DEVICES=$GPU CUDA_HOME=/usr/local/cuda-13.2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 SGLANG_OMNI_STARTUP_TIMEOUT=1800 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH=$ROOT/.venv-sglang/lib/python3.12/site-packages/nvidia/cudnn/lib:$ROOT/.venv-sglang/lib/python3.12/site-packages/nvidia/cu13/lib
cd $SRC
$ROOT/.venv-sglang/bin/python -m sglang_omni.cli serve --config $RUN/configs/$LABEL.json --model-name Qwen/Qwen3-TTS-12Hz-1.7B-Base --host 127.0.0.1 --port $PORT > $RUN/logs/server-$LABEL.log 2>&1 &
echo $! > $RUN/logs/server-$LABEL.pid
for i in $(seq 1 900); do curl -sf http://127.0.0.1:$PORT/health >/dev/null && break; kill -0 $(cat $RUN/logs/server-$LABEL.pid) 2>/dev/null || { echo "server $LABEL died"; exit 1; }; sleep 2; done
echo "[$(date -u +%H:%M:%SZ)] $LABEL ready on :$PORT gpu $GPU mode=$MODE"
