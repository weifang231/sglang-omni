# SPDX-License-Identifier: Apache-2.0
"""Ming stage factories must apply the GPU selected by replica placement."""

import sys
from types import ModuleType

import pytest

from sglang_omni.models.ming_omni import stages


@pytest.mark.parametrize("factory,module,constructor", [
    ("create_audio_encoder_executor", "audio_encoder", "MingAudioEncoder"),
    ("create_image_encoder_executor", "image_encoder", "MingImageEncoder"),
    ("create_talker_executor", "talker_executor", "MingTalkerExecutor"),
    ("create_streaming_talker_executor", "streaming_talker", "MingStreamingTalkerScheduler"),
])
@pytest.mark.parametrize("device,gpu_id,expected", [("cuda", 3, "cuda:3"), ("cuda:1", None, "cuda:1"), ("cpu", None, "cpu")])
def test_factory_uses_replica_device(monkeypatch, factory, module, constructor, device, gpu_id, expected):
    seen = {}
    component = ModuleType(module)
    def build(**kwargs):
        seen.update(kwargs)
        return object()
    setattr(component, constructor, build)
    monkeypatch.setitem(sys.modules, "sglang_omni.models.ming_omni.components." + module, component)
    from sglang_omni.models import weight_loader
    monkeypatch.setattr(weight_loader, "resolve_model_path", lambda path: path)
    getattr(stages, factory)("test-model", device=device, gpu_id=gpu_id)
    assert seen["device"] == expected
