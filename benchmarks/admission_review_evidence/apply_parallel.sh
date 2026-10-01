#!/bin/bash
# one rate per GPU, in parallel: apply_parallel.sh "48 56 64 80 96 128" "0 1 2 3 4 5"
set -u
RATES=($1); GPUS=($2)
ROOT=/scr/rucnyz/projects/aproj/omni; RUN=$ROOT/runs/pr2-demo; SRC=$ROOT/sglang-omni-upstream; SGL=$ROOT/runs/pr1-validation/sources/sglang/python
PY="$ROOT/.venv-sglang/bin/python"; export PYTHONPATH=$SGL:$SRC
SEEDS=${SEEDS:-"701 702 703"}; N=${N:-512}; SLO=${SLO:-0.5}
one() { r=$1; gpu=$2; port=$((23300+gpu))
  $RUN/serve.sh apply-r$r $gpu $port apply $RUN/profile-whisper.json $r || { echo "apply r$r: server failed"; return 1; }
  for s in $SEEDS; do echo "apply r$r s$s: $(cd $SRC && $PY -m benchmarks.eval.benchmark_admission_openloop --task asr --port $port --model-path openai/whisper-large-v3 --dataset seedtts-50 --rate $r --requests $N --slo-s $SLO --seed $s --output $RUN/results/apply-r$r-s$s.json 2>&1 | grep -vE "^W[0-9]|_pytree|torchada" | tail -n 1)"; done
  $RUN/stop.sh apply-r$r >/dev/null; }
for i in "${!RATES[@]}"; do one ${RATES[$i]} ${GPUS[$i]} & done; wait; echo "APPLY PARALLEL DONE"
