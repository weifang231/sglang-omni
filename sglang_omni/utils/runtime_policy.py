# SPDX-License-Identifier: Apache-2.0
"""Explicit experiment-owned runtime extension; absent factory keeps Native."""

import importlib
import os


class PolicyAdmissionRejected(RuntimeError):
    """An explicitly infeasible policy decision; unavailable is not rejection."""

    def __init__(self, message, receipt):
        super().__init__(message)
        self.receipt = receipt


def create_runtime_policy(*, role, **context):
    path = os.environ.get("SGLANG_OMNI_RUNTIME_POLICY_FACTORY")
    if not path:
        return None
    module, separator, name = path.partition(":")
    if not separator or not module or not name:
        raise ValueError("Runtime policy factory must be module:callable")
    factory = getattr(importlib.import_module(module), name)
    return factory(role=role, **context)
