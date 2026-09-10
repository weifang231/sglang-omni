# SPDX-License-Identifier: Apache-2.0
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.layers.sampler import Sampler
from sglang_omni.benchmarks.reference_replay import (
    ReferencePlan,
    ReferenceSequence,
    _benchmark_runtime_identity,
    _pending_stop_proof,
    install_reference_replay,
)
from torch import nn


def test_benchmark_identity_records_current_worker(monkeypatch):
    monkeypatch.setenv("SGLANG_PLUGINS", "reference_replay")
    monkeypatch.setenv("PYTHONPATH", "/task/plugin_site_only_replay:/task/repo")
    actual = _benchmark_runtime_identity()
    assert actual["pid"] == os.getpid()
    assert actual["environment"]["SGLANG_PLUGINS"] == "reference_replay"
    assert actual["diagnostic_modules"] == {
        "engine_profile": False,
        "mask_device_control": False,
    }
    assert actual["reference_methods"]["bind"]["qualname"] == "ReferencePlan.bind"


@pytest.mark.parametrize("module", ["engine_profile", "mask_device_control"])
def test_benchmark_identity_rejects_diagnostic_module(monkeypatch, module):
    monkeypatch.setenv("SGLANG_PLUGINS", "reference_replay")
    monkeypatch.setitem(sys.modules, module, SimpleNamespace())
    with pytest.raises(RuntimeError, match="diagnostic modules"):
        _benchmark_runtime_identity()


def test_benchmark_identity_rejects_unexpected_method(monkeypatch):
    monkeypatch.setenv("SGLANG_PLUGINS", "reference_replay")
    monkeypatch.setattr(ReferencePlan, "bind", lambda *args: None)
    with pytest.raises(RuntimeError, match="unexpected reference bind"):
        _benchmark_runtime_identity()


def test_benchmark_identity_rejects_broad_plugin_selection(monkeypatch):
    monkeypatch.setenv("SGLANG_PLUGINS", "reference_replay,reference_profile")
    with pytest.raises(RuntimeError, match="only the reference_replay"):
        _benchmark_runtime_identity()


@pytest.fixture(autouse=True)
def factory_runtime_context(monkeypatch):
    # CPU binding tests do not initialize a distributed model worker.
    monkeypatch.setattr(
        "sglang.srt.layers.sampler.get_server_args", lambda: SimpleNamespace()
    )


class Delegate(Sampler):
    def __init__(self):
        nn.Module.__init__(self)
        self.calls = 0

    def forward(self, logits, info, return_logprob, top, token_ids, positions):
        self.calls += 1
        return logits.argmax(-1)


class Runner:
    def __init__(self):
        self.sampler = Delegate()

    def sample(self, logits, batch):
        return self.sampler(
            logits, batch.info, batch.return_logprob, [], [], batch.positions
        )


def batch(rids, lengths, *, logprob=False):
    return SimpleNamespace(
        rids=rids,
        seq_lens_cpu=torch.tensor(lengths),
        positions=torch.tensor(lengths) - 1,
        info=SimpleNamespace(return_sampling_masks=None),
        return_logprob=logprob,
    )


def plan():
    return ReferencePlan(
        {
            "a": ReferenceSequence(3, (4, 5, 2)),
            "b": ReferenceSequence(6, (7, 1)),
            "c": ReferenceSequence(2, (3,)),
        },
        vocab_size=10,
        device="cpu",
    )


def test_official_factory_native_noop_forced_and_restore():
    runner = Runner()
    original = runner.sampler
    logits = torch.arange(20.0).reshape(2, 10)
    b = batch(["a", "b"], [3, 6])
    native = runner.sample(logits, b)
    with install_reference_replay(runner, plan(), mode="noop"):
        assert torch.equal(runner.sample(logits, b), native)
    with install_reference_replay(runner, plan(), mode="forced"):
        assert runner.sample(logits, b).tolist() == [4, 7]
    assert runner.sampler is original and original.calls == 3
    assert "sample" not in vars(runner)


def test_row_moves_additions_removals_and_eos():
    runner = Runner()
    with install_reference_replay(runner, plan(), mode="forced"):
        for rids, lengths, expected in [
            (["a", "b"], [3, 6], [4, 7]),
            (["b", "a"], [7, 4], [1, 5]),
            (["c", "a"], [2, 5], [3, 2]),
            (["a"], [5], [2]),
        ]:
            assert (
                runner.sample(torch.zeros(len(rids), 10), batch(rids, lengths)).tolist()
                == expected
            )


@pytest.mark.parametrize(
    "rids,lengths",
    [
        (["unknown"], [3]),
        (["a", "a"], [3, 3]),
        (["a"], [2]),
        (["a"], [6]),
        (["b"], [8]),
    ],
)
def test_invalid_binding_fails_without_fallback(rids, lengths):
    runner = Runner()
    original = runner.sampler
    with (
        install_reference_replay(runner, plan(), mode="forced"),
        pytest.raises(ValueError),
    ):
        runner.sample(torch.zeros(len(rids), 10), batch(rids, lengths))
    assert original.calls == 0


def test_context_reset_after_failure_and_double_install():
    runner = Runner()
    with install_reference_replay(runner, plan(), mode="forced"):
        with pytest.raises(ValueError):
            runner.sample(torch.zeros(1, 10), batch(["a"], [3], logprob=True))
        with pytest.raises(ValueError):
            install_reference_replay(runner, plan(), mode="noop")
        assert runner.sample(torch.zeros(1, 10), batch(["a"], [3])).item() == 4
        with pytest.raises(RuntimeError, match="bound ForwardBatch"):
            runner.sampler(
                torch.zeros(1, 10),
                batch(["a"], [3]).info,
                False,
                [],
                [],
                torch.tensor([2]),
            )


def test_thread_local_request_binding():
    runner = Runner()
    with install_reference_replay(runner, plan(), mode="forced"):

        def run(item):
            rid, length = item
            return runner.sample(torch.zeros(1, 10), batch([rid], [length])).item()

        with ThreadPoolExecutor(2) as pool:
            assert list(pool.map(run, [("a", 4), ("b", 6)] * 12)) == [5, 7] * 12


def test_only_proven_terminal_lookahead_passes_through_native():
    runner = Runner()

    def terminal(rid, offset, length):
        return (
            "actual_request_length_cap"
            if rid == "b" and offset == length == 2
            else None
        )

    with install_reference_replay(
        runner, plan(), mode="forced", terminal_resolver=terminal
    ) as installed:
        logits = torch.arange(20.0).reshape(2, 10)
        assert runner.sample(logits, batch(["b", "a"], [8, 4])).tolist() == [9, 5]
        assert installed.discarded_rows == [("b", 2, "actual_request_length_cap")]
        with pytest.raises(ValueError):
            runner.sample(torch.zeros(1, 10), batch(["b"], [9]))
        with pytest.raises(ValueError):
            runner.sample(torch.zeros(1, 10), batch(["a"], [6]))


def test_warmed_terminal_binding_has_no_host_tensor_allocation(monkeypatch):
    p = plan()
    p.bind(batch(["b", "a"], [6, 3]))
    final_batch = batch(["b", "a"], [8, 4])

    def reject_allocation(*args, **kwargs):
        raise AssertionError(
            "terminal binding must not construct a host-to-device mask"
        )

    monkeypatch.setattr(torch, "tensor", reject_allocation)
    rows, discard, evidence = p.bind(final_batch, lambda *args: "actual_length_cap")
    assert rows.tolist() == [1, 0]
    assert discard is True
    assert evidence == [("b", 2, "actual_length_cap")]


@pytest.mark.parametrize("tokens", [(), (-1,), (10,), (True,)])
def test_reference_vocabulary_validation(tokens):
    with pytest.raises(ValueError):
        ReferencePlan({"a": ReferenceSequence(1, tokens)}, vocab_size=10, device="cpu")


@pytest.mark.parametrize(
    "token,ignore,length,event_present,expected",
    [
        (2, False, 5, True, "actual_pending_stop_token:2"),
        (7, False, 5, True, "actual_pending_stop_token:7"),
        (4, False, 5, True, None),
        (2, True, 5, True, None),
        (2, False, 4, True, None),
        (2, False, 5, False, None),
    ],
)
def test_pending_eos_requires_actual_previous_result(
    token, ignore, length, event_present, expected
):
    event = SimpleNamespace(synchronize=lambda: None)
    req = SimpleNamespace(
        rid="a",
        sampling_params=SimpleNamespace(ignore_eos=ignore, stop_token_ids={7}),
        eos_token_ids={2},
        tokenizer=None,
    )
    previous = SimpleNamespace(reqs=[req], seq_lens_cpu=torch.tensor([length]))
    result = SimpleNamespace(
        next_token_ids=torch.tensor([token]), copy_done=event if event_present else None
    )
    scheduler = SimpleNamespace(result_queue=[(previous, result)])
    assert _pending_stop_proof(scheduler, "a", 3, 3) == expected


@pytest.mark.skipif(
    os.environ.get("REFERENCE_REPLAY_GPU_TEST") != "1",
    reason="explicit task GPU required",
)
@pytest.mark.parametrize("size", [1, 8, 32])
@pytest.mark.parametrize("position_dtype", [torch.int32, torch.int64])
def test_fused_cuda_gather_real_row_moves_and_discard(size, position_dtype):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "2"
    sequences = {
        f"r{i}": ReferenceSequence(i + 3, (i + 1, i + 2, i + 3)) for i in range(size)
    }
    p = ReferencePlan(sequences, vocab_size=100, device="cuda:0")
    runner = Runner()
    terminal = lambda rid, offset, length: (
        "actual_test_length_cap" if offset == length == 3 else None
    )
    with install_reference_replay(runner, p, mode="forced", terminal_resolver=terminal):
        for offset in [0, 1, 2, 3]:
            indices = list(reversed(range(size)))
            lengths = [
                i + 3 + (offset if i % 2 == 0 else min(offset, 2)) for i in indices
            ]
            b = batch([f"r{i}" for i in indices], lengths)
            b.positions = b.positions.to(device="cuda:0", dtype=position_dtype)
            logits = torch.arange(size * 100.0, device="cuda:0").reshape(size, 100)
            actual = runner.sample(logits, b).cpu().tolist()
            expected = [
                99 if offset == 3 and i % 2 == 0 else i + 1 + min(offset, 2)
                for i in indices
            ]
            assert actual == expected
