# SPDX-License-Identifier: Apache-2.0
"""Reference-token replay after the unmodified SGLang sampler.

Install explicitly on a loaded Shared ModelRunner. The official factory creates
the wrapper without changing the runtime's original sampling backend. Reference
tokens are resident on the model device before measurement. No custom logit
processor or SamplingParams.custom_params is involved.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from sglang.srt.layers.sampler import (
    Sampler,
    create_sampler,
    register_sampler_backend,
)
from torch import nn

BACKEND_NAME = "omni_reference_replay_v1"
_factory_args = ContextVar("reference_replay_factory", default=None)
_registered = False


@triton.jit
def _select_kernel(
    rows,
    positions,
    prompts,
    tokens,
    natural,
    discard,
    output,
    count: tl.constexpr,
    width: tl.constexpr,
    has_discard: tl.constexpr,
    block: tl.constexpr,
):
    i = tl.arange(0, block)
    valid = i < count
    row = tl.load(rows + i, valid, 0)
    offset = tl.load(positions + i, valid, 0) - tl.load(prompts + row, valid, 0) + 1
    if has_discard:
        dropped = offset == tl.load(discard + row, valid, -1)
        offset = tl.where(dropped, 0, offset)
    selected = tl.load(tokens + row * width + offset, valid, 0)
    if has_discard:
        selected = tl.where(dropped, tl.load(natural + i, valid, 0), selected)
    tl.store(output + i, selected, valid)


@dataclass(frozen=True)
class ReferenceSequence:
    prompt_length: int
    token_ids: tuple[int, ...]


class ReferencePlan:
    """Finite request-ID map; positions, not call counts, select output tokens."""

    def __init__(
        self,
        sequences: Mapping[str, ReferenceSequence],
        *,
        vocab_size: int,
        device: str | torch.device,
        batch_cache_limit: int = 64,
    ):
        if not sequences or vocab_size <= 0 or batch_cache_limit <= 0:
            raise ValueError("a nonempty finite plan and positive limits are required")
        self._sequences = dict(sequences)
        for rid, seq in self._sequences.items():
            if not isinstance(rid, str) or not rid:
                raise ValueError("request IDs must be nonempty strings")
            if type(seq.prompt_length) is not int or seq.prompt_length < 1:
                raise ValueError("prompt length must be a positive integer")
            if not isinstance(seq.token_ids, tuple) or not seq.token_ids:
                raise ValueError("reference token IDs must be a nonempty tuple")
            if any(
                type(t) is not int or t < 0 or t >= vocab_size for t in seq.token_ids
            ):
                raise ValueError(
                    "reference token IDs must be inside the model vocabulary"
                )
        self.device = torch.device(device)
        self._row_by_rid = {rid: i for i, rid in enumerate(self._sequences)}
        width = max(len(s.token_ids) for s in self._sequences.values())
        self._tokens = torch.tensor(
            [
                list(s.token_ids) + [0] * (width - len(s.token_ids))
                for s in self._sequences.values()
            ],
            device=self.device,
            dtype=torch.int64,
        )
        self._prompt_lengths = torch.tensor(
            [s.prompt_length for s in self._sequences.values()],
            device=self.device,
            dtype=torch.int64,
        )
        self._reference_lengths = torch.tensor(
            [len(s.token_ids) for s in self._sequences.values()],
            device=self.device,
            dtype=torch.int64,
        )
        self._batch_cache_limit = batch_cache_limit
        self._batch_rows = OrderedDict()

    def bind(self, forward_batch, terminal_resolver=None):
        rids = tuple(forward_batch.rids or ())
        lengths = forward_batch.seq_lens_cpu
        if not rids or len(set(rids)) != len(rids):
            raise ValueError("each batch row needs a unique request ID")
        if not isinstance(lengths, torch.Tensor) or lengths.device.type != "cpu":
            raise ValueError("the existing CPU sequence-length mirror is required")
        if lengths.ndim != 1 or len(lengths) != len(rids):
            raise ValueError("request IDs and sequence lengths do not align")
        if lengths.dtype not in {torch.int32, torch.int64}:
            raise ValueError("sequence lengths must be integers")
        # Read the existing host mirror, never synchronize device positions to CPU.
        discarded = []
        for rid, length in zip(rids, lengths.tolist()):
            if rid not in self._sequences:
                raise ValueError(f"unregistered replay request: {rid}")
            seq = self._sequences[rid]
            offset = length - seq.prompt_length
            if not 0 <= offset < len(seq.token_ids):
                proof = (
                    terminal_resolver(rid, offset, len(seq.token_ids))
                    if offset == len(seq.token_ids) and terminal_resolver
                    else None
                )
                if not proof:
                    raise ValueError(
                        f"reference position out of bounds for {rid}: {offset}"
                    )
                discarded.append((rid, offset, proof))
        rows = self._batch_rows.get(rids)
        if rows is None:
            rows = torch.tensor(
                [self._row_by_rid[rid] for rid in rids],
                device=self.device,
                dtype=torch.int64,
            )
            self._batch_rows[rids] = rows
            if len(self._batch_rows) > self._batch_cache_limit:
                self._batch_rows.popitem(last=False)
        return rows, True if discarded else None, discarded

    def select(self, rows, positions, natural, discarded=None):
        if positions.ndim != 1 or positions.shape != rows.shape:
            raise ValueError("only one sampled token per request is supported")
        if positions.device != rows.device:
            raise ValueError("positions and reference table must share a device")
        if rows.is_cuda:
            selected = torch.empty_like(natural)
            _select_kernel[(1,)](
                rows,
                positions,
                self._prompt_lengths,
                self._tokens,
                natural,
                self._reference_lengths if discarded is not None else rows,
                selected,
                rows.numel(),
                self._tokens.shape[1],
                discarded is not None,
                triton.next_power_of_2(rows.numel()),
                num_warps=1,
            )
            return selected
        offsets = positions.to(torch.int64) - self._prompt_lengths[rows] + 1
        if discarded is not None:
            discarded = offsets == self._reference_lengths[rows]
            offsets = torch.where(discarded, 0, offsets)
        selected = self._tokens[rows, offsets].to(natural.dtype)
        return (
            torch.where(discarded, natural, selected)
            if discarded is not None
            else selected
        )


class ReferenceReplaySampler(Sampler):
    def __init__(self, delegate: Sampler, plan: ReferencePlan, mode: str):
        # Delegate owns the initialized TP group and all native sampling policy.
        nn.Module.__init__(self)
        if mode not in {"noop", "forced"}:
            raise ValueError("mode must be noop or forced")
        self.delegate = delegate
        self.plan = plan
        self.mode = mode
        self._rows = ContextVar(f"replay_rows_{id(self)}", default=None)

    def forward(
        self,
        logits_output,
        sampling_info,
        return_logprob,
        top_logprobs_nums,
        token_ids_logprobs,
        positions,
    ):
        bound = self._rows.get()
        if bound is None:
            raise RuntimeError("replay sampler requires a bound ForwardBatch")
        rows, discarded = bound
        if self.mode == "forced" and (
            return_logprob
            or any(getattr(sampling_info, "return_sampling_masks", None) or [])
        ):
            raise ValueError(
                "forced replay does not expose native-token logprobs or support masks"
            )
        natural = self.delegate(
            logits_output,
            sampling_info,
            return_logprob,
            top_logprobs_nums,
            token_ids_logprobs,
            positions,
        )
        if natural.ndim != 1 or natural.shape != rows.shape:
            raise ValueError("native output and reference batch shape differ")
        reference = self.plan.select(rows, positions, natural, discarded)
        if natural.shape != reference.shape:
            raise ValueError("native output and reference batch shape differ")
        if self.mode == "noop":
            return natural
        return reference

    def compute_logprobs_only(self, *args, **kwargs):
        if self.mode == "forced":
            raise ValueError("logprob-only replay is outside this benchmark")
        return self.delegate.compute_logprobs_only(*args, **kwargs)


def _factory():
    args = _factory_args.get()
    if args is None:
        raise RuntimeError("use install_reference_replay on a loaded ModelRunner")
    return ReferenceReplaySampler(*args)


class ReplayInstallation:
    def __init__(self, runner, plan: ReferencePlan, mode: str, terminal_resolver=None):
        global _registered
        if isinstance(runner.sampler, ReferenceReplaySampler):
            raise ValueError("reference replay is already installed")  # noqa: TRY004
        if not _registered:
            register_sampler_backend(BACKEND_NAME, _factory)
            _registered = True
        self.runner = runner
        self.original_sampler = runner.sampler
        self.original_sample = runner.sample
        self.discarded_rows = []
        self._had_sample_attribute = "sample" in vars(runner)
        token = _factory_args.set((self.original_sampler, plan, mode))
        try:
            self.sampler = create_sampler(BACKEND_NAME)
        finally:
            _factory_args.reset(token)

        def sample_with_reference(logits_output, forward_batch):
            rows, discarded, evidence = plan.bind(forward_batch, terminal_resolver)
            self.discarded_rows.extend(evidence)
            token = self.sampler._rows.set((rows, discarded))
            try:
                return self.original_sample(logits_output, forward_batch)
            finally:
                self.sampler._rows.reset(token)

        self.bound_sample = sample_with_reference
        runner.sampler = self.sampler
        runner.sample = self.bound_sample
        self.closed = False

    def close(self):
        if self.closed:
            return
        if (
            self.runner.sampler is not self.sampler
            or self.runner.sample is not self.bound_sample
        ):
            raise RuntimeError("another owner changed the installed replay hooks")
        self.runner.sampler = self.original_sampler
        if self._had_sample_attribute:
            self.runner.sample = self.original_sample
        else:
            del self.runner.sample
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def install_reference_replay(
    runner, plan: ReferencePlan, *, mode: str, terminal_resolver=None
):
    """Attach to an existing worker without changing its backend or overlap flags."""
    return ReplayInstallation(runner, plan, mode, terminal_resolver)


def _pending_stop_proof(scheduler, rid, offset, prompt_length):
    for batch, result in getattr(scheduler, "result_queue", ()):
        for row, req in enumerate(batch.reqs):
            if req.rid != rid or req.sampling_params.ignore_eos:
                continue
            lengths = batch.seq_lens_cpu
            if lengths is None or int(lengths[row]) != prompt_length + offset - 1:
                continue
            tokens = result.next_token_ids
            if (
                not isinstance(tokens, torch.Tensor)
                or tokens.device.type != "cpu"
                or tokens.ndim != 1
                or len(tokens) != len(batch.reqs)
                or result.copy_done is None
            ):
                continue
            # Only at an exhausted reference, inspect the engine's existing D2H.
            # No new token copy or synchronization is added to ordinary steps.
            result.copy_done.synchronize()
            token = int(tokens[row])
            stop = set(req.sampling_params.stop_token_ids or ()) | set(
                req.eos_token_ids or ()
            )
            if req.tokenizer is not None:
                stop.add(req.tokenizer.eos_token_id)
                stop.update(req.tokenizer.additional_stop_token_ids or ())
            if token in stop:
                return f"actual_pending_stop_token:{token}"
    return None


def _benchmark_runtime_identity():
    """Validate task plugin identity outside all measured generation calls."""
    import os
    import sys
    from pathlib import Path

    diagnostic_modules = {
        name: name in sys.modules for name in ("engine_profile", "mask_device_control")
    }
    methods = {}
    source = Path(__file__).resolve()
    for name in ("bind", "select"):
        method = getattr(ReferencePlan, name)
        code = getattr(method, "__code__", None)
        methods[name] = {
            "code_filename": code.co_filename if code is not None else None,
            "qualname": method.__qualname__,
            "module": method.__module__,
        }
        if code is None or Path(code.co_filename).resolve() != source:
            raise RuntimeError(f"unexpected reference {name} implementation")
    if any(diagnostic_modules.values()):
        raise RuntimeError("diagnostic modules are loaded in the benchmark worker")
    if os.environ.get("SGLANG_PLUGINS") != "reference_replay":
        raise RuntimeError("benchmark requires only the reference_replay plugin")
    return {
        "environment": {
            name: os.environ.get(name)
            for name in ("SGLANG_PLUGINS", "PYTHONPATH", "OMP_NUM_THREADS")
        },
        "reference_methods": methods,
        "diagnostic_modules": diagnostic_modules,
        "pid": os.getpid(),
    }


def register_benchmark_control():
    """Optional general-plugin entry point for idle-only Engine.collective_rpc.

    This adds a control method; native sampling has no per-step instrumentation.
    Enable only in a task-owned benchmark process, never in ordinary serving.
    """
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    def after_scheduler_init(result, scheduler, *args, **kwargs):
        installation = None
        plans = {}

        def terminal_resolver(rid, offset, reference_length):
            batches = [scheduler.running_batch, scheduler.last_batch]
            batches.extend(b for b, _ in getattr(scheduler, "result_queue", ()))
            for batch in batches:
                for req in getattr(batch, "reqs", ()):
                    if req.rid != rid:
                        continue
                    if req.finished():
                        return "actual_request_finished"
                    if req.sampling_params.max_new_tokens == reference_length == offset:
                        return "actual_request_length_cap"
            if installation is not None:
                seq = installation.sampler.plan._sequences[rid]
                return _pending_stop_proof(scheduler, rid, offset, seq.prompt_length)
            return None

        def configure_reference_replay(*, mode, sequences=None, receipt_path=None):
            nonlocal installation
            if not scheduler.is_fully_idle():
                raise RuntimeError("replay configuration requires an idle scheduler")
            if mode not in {"native", "noop", "replay"}:
                raise ValueError("unknown benchmark arm")
            runtime_identity = _benchmark_runtime_identity()
            runner = scheduler.tp_worker.model_runner
            if not scheduler.spec_algorithm.is_none():
                raise ValueError(
                    "speculative decoding is outside this replay benchmark"
                )
            previous_discarded = []
            if installation is not None:
                previous_discarded = installation.discarded_rows
                installation.close()
                installation = None
            if mode != "native":
                key = tuple(
                    (rid, s["prompt_length"], tuple(s["token_ids"]))
                    for rid, s in sorted(sequences.items())
                )
                if key not in plans:
                    if len(plans) >= 64:
                        raise ValueError("benchmark plan limit reached")
                    plans[key] = ReferencePlan(
                        {
                            rid: ReferenceSequence(length, tokens)
                            for rid, length, tokens in key
                        },
                        vocab_size=runner.model_config.vocab_size,
                        device=f"cuda:{runner.gpu_id}",
                    )
                plan = plans[key]
                installation = install_reference_replay(
                    runner,
                    plan,
                    mode="forced" if mode == "replay" else "noop",
                    terminal_resolver=terminal_resolver,
                )
            torch.cuda.synchronize(runner.gpu_id)
            if receipt_path is not None:
                import json
                from pathlib import Path

                from sglang.srt.runtime_context import get_exec

                Path(receipt_path).write_text(
                    json.dumps(
                        {
                            "arm": mode,
                            "backend": get_exec().kernel.sampling_backend,
                            "enable_overlap": scheduler.enable_overlap,
                            "disable_cuda_graph": runner.server_args.disable_cuda_graph,
                            "attention_backend": runner.server_args.attention_backend,
                            "sampler_class": type(runner.sampler).__qualname__,
                            "sampler_module": type(runner.sampler).__module__,
                            "runtime_identity": runtime_identity,
                            "reference_bound": installation is not None,
                            "previous_discarded_rows": previous_discarded,
                        },
                        indent=2,
                    )
                    + "\n"
                )

        scheduler.configure_reference_replay = configure_reference_replay

    HookRegistry.register(
        "sglang.srt.managers.scheduler.Scheduler.__init__",
        after_scheduler_init,
        HookType.AFTER,
    )
