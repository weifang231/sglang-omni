#!/bin/bash
# usage: serve_cap.sh <label> <gpu> <port> <max_running_requests> <max_queued_requests>  — native Whisper with an explicit engine cap
set -u
LABEL=$1; GPU=$2; PORT=$3; R=$4; Q=$5
ROOT=/scr/rucnyz/projects/aproj/omni; RUN=$ROOT/runs/pr2-demo; SRC=$ROOT/sglang-omni-upstream; SGL=$ROOT/runs/pr1-validation/sources/sglang/python
mkdir -p $RUN/logs $RUN/configs
python3 - "$LABEL" "$R" "$Q" <<'PY'
import json, sys
label, r, q = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
root = "/scr/rucnyz/projects/aproj/omni"
cfg = {"config_cls": "WhisperASRPipelineConfig", "model_path": f"{root}/models/whisper-large-v3",
       "audio_chunking": {"max_concurrent_long_audio_requests": 64},
       "stages": {"asr": {"gpu": 0, "engine": {"mem_fraction_static": 0.3, "cuda_graph_max_bs": min(r, 64), "max_running_requests": r, "max_queued_requests": q}}}}
json.dump(cfg, open(f"{root}/runs/pr2-demo/configs/{label}.json", "w"), indent=1)
PY
export PYTHONPATH=$SRC:$SGL CUDA_VISIBLE_DEVICES=$GPU CUDA_HOME=/usr/local/cuda-13.2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 SGLANG_OMNI_STARTUP_TIMEOUT=1800 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH=$ROOT/.venv-sglang/lib/python3.12/site-packages/nvidia/cudnn/lib:$ROOT/.venv-sglang/lib/python3.12/site-packages/nvidia/cu13/lib
cd $SRC
$ROOT/.venv-sglang/bin/python -m sglang_omni.cli serve --config $RUN/configs/$LABEL.json --model-name openai/whisper-large-v3 --host 127.0.0.1 --port $PORT > $RUN/logs/server-$LABEL.log 2>&1 &
echo $! > $RUN/logs/server-$LABEL.pid
for i in $(seq 1 900); do curl -sf http://127.0.0.1:$PORT/health >/dev/null && break; kill -0 $(cat $RUN/logs/server-$LABEL.pid) 2>/dev/null || { echo "server $LABEL died"; exit 1; }; sleep 2; done
echo "[$(date -u +%H:%M:%SZ)] $LABEL ready on :$PORT gpu $GPU cap=$R+$Q"
