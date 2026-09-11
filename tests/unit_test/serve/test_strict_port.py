# SPDX-License-Identifier: Apache-2.0
"""SGLANG_OMNI_STRICT_PORT turns the port fallback into a hard error."""

from __future__ import annotations

import socket

import pytest

from sglang_omni.serve.launcher import _find_available_port


def test_free_port_is_returned_unchanged(monkeypatch):
    monkeypatch.delenv("SGLANG_OMNI_STRICT_PORT", raising=False)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        free = probe.getsockname()[1]
    assert _find_available_port("127.0.0.1", free) == free


def test_busy_port_falls_back_by_default(monkeypatch):
    monkeypatch.delenv("SGLANG_OMNI_STRICT_PORT", raising=False)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as holder:
        holder.bind(("127.0.0.1", 0))
        busy = holder.getsockname()[1]
        assert _find_available_port("127.0.0.1", busy) != busy


def test_busy_port_hard_errors_under_strict(monkeypatch):
    monkeypatch.setenv("SGLANG_OMNI_STRICT_PORT", "1")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as holder:
        holder.bind(("127.0.0.1", 0))
        busy = holder.getsockname()[1]
        with pytest.raises(RuntimeError, match="STRICT_PORT"):
            _find_available_port("127.0.0.1", busy)


def test_closed_uvicorn_connection_does_not_change_the_requested_port(monkeypatch):
    monkeypatch.setenv("SGLANG_OMNI_STRICT_PORT", "1")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        listener.listen(1)
        with socket.create_connection(("127.0.0.1", port), timeout=2) as client:
            peer, _ = listener.accept()
            peer.close()
            assert client.recv(1) == b""
    assert _find_available_port("127.0.0.1", port) == port
