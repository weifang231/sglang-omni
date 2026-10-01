#!/bin/bash
# four engine caps in parallel, native Whisper, rates above the knee: caps.sh
set -u
ROOT=/scr/rucnyz/projects/aproj/omni; RUN=$ROOT/runs/pr2-demo; SRC=$ROOT/sglang-omni-upstream; SGL=$ROOT/runs/pr1-validation/sources/sglang/python
PY="$ROOT/.venv-sglang/bin/python"; export PYTHONPATH=$SGL:$SRC
RATES=${RATES:-"48 64 96 128"}; SEEDS=${SEEDS:-"701 702 703"}; N=512; SLO=0.5
CAPS=("16 16" "24 16" "32 16" "24 1024"); GPUS=(0 1 2 6)
one() { r=$1; q=$2; gpu=$3; port=$((23600+gpu)); label=cap-r${r}q${q}
  $RUN/serve_cap.sh $label $gpu $port $r $q || { echo "$label: server failed"; return 1; }
  for rate in $RATES; do for s in $SEEDS; do echo "$label rate$rate s$s: $(cd $SRC && $PY -m benchmarks.eval.benchmark_admission_openloop --task asr --port $port --model-path openai/whisper-large-v3 --dataset seedtts-50 --rate $rate --requests $N --slo-s $SLO --seed $s --output $RUN/results/$label-r$rate-s$s.json 2>&1 | grep -vE "^W[0-9]|_pytree|torchada" | tail -n 1)"; done; done
  $RUN/stop.sh $label >/dev/null; }
for i in "${!CAPS[@]}"; do one ${CAPS[$i]} ${GPUS[$i]} & done; wait; echo "CAPS DONE"
