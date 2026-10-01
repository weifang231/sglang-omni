#!/bin/bash
# Whisper demo on the PR branch: record -> fit -> native vs apply at rates above the knee, seeded traffic.
# usage: campaign.sh <gpu> <port>
set -u
GPU=$1; PORT=$2
ROOT=/scr/rucnyz/projects/aproj/omni; RUN=$ROOT/runs/pr2-demo; SRC=$ROOT/sglang-omni-upstream; SGL=$ROOT/runs/pr1-validation/sources/sglang/python
PY="$ROOT/.venv-sglang/bin/python"; export PYTHONPATH=$SGL:$SRC
RATES=${RATES:-"24 32 40 48 56 64"}; SEEDS=${SEEDS:-"701 702 703"}; N=${N:-512}; SLO=${SLO:-0.5}; RECORD_RATES=${RECORD_RATES:-"24 32 40"}
bench() { (cd $SRC && $PY -m benchmarks.eval.benchmark_admission_openloop --task asr --port $PORT --model-path openai/whisper-large-v3 --dataset seedtts-50 --rate $1 --requests $N --slo-s $SLO --seed $2 --output $RUN/results/$3.json 2>&1 | grep -vE "^W[0-9]|_pytree|torchada" | tail -n 1); }
mkdir -p $RUN/results
echo "== 1. record at $RECORD_RATES req/s (one server, events appended)"
$RUN/serve.sh record $GPU $PORT record || exit 1
for r in $RECORD_RATES; do for s in 801 802; do echo "record r$r s$s: $(bench $r $s record-r$r-s$s)"; done; done
$RUN/stop.sh record
echo "== 2. fit (planning rate 48)"
(cd $SRC && $PY -m sglang_omni.admission_policies.fit_capacity_table $RUN/events/record.jsonl --kind asr --deadline-s $SLO --arrival-rate-rps 48 --output $RUN/profile-whisper.json 2>&1 | grep -vE "^W[0-9]|_pytree" | tail -n 2)
echo "== 3. native"
$RUN/serve.sh native $GPU $PORT native || exit 1
for r in $RATES; do for s in $SEEDS; do echo "native r$r s$s: $(bench $r $s native-r$r-s$s)"; done; done
$RUN/stop.sh native
echo "== 4. apply, prices re-solved per rate"
for r in $RATES; do
  $RUN/serve.sh apply-r$r $GPU $PORT apply $RUN/profile-whisper.json $r || exit 1
  for s in $SEEDS; do echo "apply r$r s$s: $(bench $r $s apply-r$r-s$s)"; done
  $RUN/stop.sh apply-r$r
done
echo "CAMPAIGN DONE"
