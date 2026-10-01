#!/bin/bash
set -u
GPU=$1; PORT=$2
ROOT=/scr/rucnyz/projects/aproj/omni; RUN=$ROOT/runs/pr2-demo; SRC=$ROOT/sglang-omni-upstream; SGL=$ROOT/runs/pr1-validation/sources/sglang/python
PY="$ROOT/.venv-sglang/bin/python"; export PYTHONPATH=$SGL:$SRC
RATES=${RATES:-"48 56 64 80 96 128"}; SEEDS=${SEEDS:-"701 702 703"}; N=${N:-512}; SLO=${SLO:-0.5}
bench() { (cd $SRC && $PY -m benchmarks.eval.benchmark_admission_openloop --task asr --port $PORT --model-path openai/whisper-large-v3 --dataset seedtts-50 --rate $1 --requests $N --slo-s $SLO --seed $2 --output $RUN/results/$3.json 2>&1 | grep -vE "^W[0-9]|_pytree|torchada" | tail -n 1); }
for r in $RATES; do
  $RUN/serve.sh apply-r$r $GPU $PORT apply $RUN/profile-whisper.json $r || exit 1
  for s in $SEEDS; do echo "apply r$r s$s: $(bench $r $s apply-r$r-s$s)"; done
  $RUN/stop.sh apply-r$r
done
echo "APPLY DONE"
