# Reference-token replay

`reference_replay.py` is an opt-in latency-testing adapter for SGLang's official
`register_sampler_backend` factory. It calls the loaded runner's original sampler
and then selects a reference token. The original model forward and sampling
computation still run. Reference tensors live on the model device.

The wrapper keeps the original runtime sampling-backend name, so native
FlashInfer stochastic sampling still uses its ordinary backend. It does not set
`SamplingParams.custom_params`, install a custom logit processor, or change
overlap, attention, or CUDA-graph flags.

## Loaded-runner API

Install and remove the adapter only while the runner has no work in flight:

```python
from sglang_omni.benchmarks.reference_replay import (
    ReferencePlan, ReferenceSequence, install_reference_replay,
)

plan = ReferencePlan(
    {request_id: ReferenceSequence(actual_prompt_length, tuple(reference_ids))},
    vocab_size=runner.model_config.vocab_size,
    device=f"cuda:{runner.gpu_id}",
)
installation = install_reference_replay(runner, plan, mode="forced")
# Run the already-bound request cohort through the ordinary runner.
installation.close()
```

`mode="noop"` performs the same reference lookup and returns the native token.
`close()` restores the original sampler object and the original `sample` method;
the restored native path has no replay wrapper. The wrapper rejects another
installation or an unexpected owner changing those objects.

The caller must bind the exact model, tokenizer, input token IDs, request IDs and
reference output. The plan checks request IDs, prompt lengths, vocabulary bounds
and output positions; it does not establish the provenance of prompt contents.
The latency harness records the full input and output IDs before timing.
References must retain sampled stop/EOS IDs needed for termination; trimmed
response text alone is insufficient. The real early-stop check uses
`no_stop_trim=True` to preserve its actual stop token.

## Optional Engine control RPC

The local benchmark exposes `register_benchmark_control` through a task-private
`sglang.srt.plugins` entry point named `reference_replay`. Enable that entry point
explicitly with `SGLANG_PLUGINS=reference_replay`; the repository does not register
it for normal serving.

```python
engine.collective_rpc(
    "configure_reference_replay",
    mode="replay",  # "noop" or "native" are the other arms
    sequences={request_id: {
        "prompt_length": actual_prompt_length,
        "token_ids": reference_ids,
    }},
    receipt_path=absolute_unique_receipt_path,
)
```

The RPC requires a fully idle scheduler, the reviewed dense Qwen3 model, and
non-speculative decoding. An
Engine response can arrive before its last overlap result leaves the scheduler,
so the harness retries only the explicit idle error, for at most 10 seconds,
outside every timed measurement. Each formal configuration has its own receipt
and SHA-256 reference in the raw measurement. `mode="native"` removes the adapter.

The idle RPC also records the worker's PID, selected plugin, Python path, thread
setting, and the actual installed plan's `bind/select` code locations and source hashes. It rejects
loaded diagnostic modules or unexpected method files before generation. Use a
private entry-point directory containing only `reference_replay`; the formal
launcher hashes that metadata and its source before starting the worker. These
identity checks do not run in the per-token path.

## Position and termination behavior

Each forward binds its actual `ForwardBatch.rids` and reads the current CPU
sequence-length mirror. The reviewed dense runner derives its GPU positions
from those lengths. When the full reference cohort has a common valid output
offset, the adapter returns a pre-created immutable GPU frame. Request membership
and prompt/reference lengths are cached by row ordering; changing sequence
lengths are not cached. The returned frame retains its backing storage across
subsequent steps and cache eviction.

Mixed offsets, partial cohorts and discarded boundary samples use the existing
GPU-position Triton selector. Both paths use actual positions rather than a
shared call counter. The CPU path supports focused binding tests. Fixed-cohort
decoder results do not establish latency equivalence for the mixed-position path,
which still adds a kernel launch and per-step host work.

The final adapter preloads reference lengths while configuring the idle worker.
After the CPU proves a discarded row, the GPU compares its offset with that
preloaded length. It does not construct a host boolean mask at termination.
The earlier mask construction issued a blocking H2D copy; its retained traces
and measurements remain part of the experiment evidence.

The first occurrence of an uncached request-ID ordering still allocates a device
row-index tensor. The fixed-cohort benchmark warms those orderings before timing,
and its plan cache survives arm changes. This test does not establish latency
equivalence for arbitrary continuous-batch cache misses.

Normal overlap may launch one extra forward before the previous result finishes
the request. The Engine adapter permits an exhausted reference only when the
actual request is finished, its actual length cap proves that step is discarded,
or its previous pending result contains an actual stop/EOS token. For the last
case it waits on the engine's existing copy event and reads its existing CPU
result only at the exhausted boundary. Ordinary steps add no token D2H copy.
The extra forward still calls the native sampler and returns its natural result;
the engine discards that result. The receipt records this boundary. Unproved
exhaustion and unknown IDs fail instead of generating unconstrained output.

Forced replay rejects native-token log probabilities and sampling-support masks,
because those values do not describe an overwritten token. Reference replay
must not be used to report model-quality scores.

## Local test scope

The local environment is SGLang `0.5.18.dev249+gcea16bf22`, PyTorch
`2.13.0+cu130`, Triton `3.7.1` and Transformers `5.12.1`. This is a development
checkout, not a reproduction of the `0.5.18` release. The real decoder test uses
cached Qwen3-0.6B weights on physical GB300 GPU 2, with external NUMA 1 binding.
The installed checkout contains pre-existing DP research changes; the paired
test uses the same loaded source with DP attention disabled and DP size one.
It retains default overlap, CUDA graphs, TRTLLM attention and FlashInfer sampling;
prefix caching is disabled in all arms before reference preparation.

CPU checks cover binding, reordered and mixed rows, invalid references, ownership
restoration, thread-local binding and terminal proofs. Explicit GPU checks cover
the fused gather for batches 1, 8 and 32 with int32/int64 positions and mixed
discarded rows. The real decoder checks visible token IDs and finish reasons,
including a stop-token row whose peer continues decoding.
The frame check also changes positions, reorders rows, and evicts cached mappings
while retaining an earlier output tensor, then verifies that its contents remain
unchanged.

These checks cover the tested dense, non-speculative decoder. They do not cover
DLLM, multimodal rotary-position layouts, arbitrary runner callers, the full Omni
speech pipeline, or changes in model quality. The nonstreaming decoder test
records cohort wall time and the engine's per-request completion latency; it
does not establish TTFT or TPOT. Performance conclusions belong to the retained
paired measurements, not to the correctness tests.
