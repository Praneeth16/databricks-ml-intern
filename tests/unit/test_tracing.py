"""Unit tests for agent.core.tracing.

Tracing must be fail-soft: when MLflow can't reach the workspace (no creds /
network during unit tests) span helpers should silently no-op instead of
breaking the agent.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agent.core import tracing


def setup_function():
    tracing.reset_for_tests()


def test_init_returns_false_without_experiment_path():
    assert tracing.init_tracing(None) is False
    assert tracing.init_tracing("") is False


def test_init_returns_false_when_mlflow_raises():
    with patch("mlflow.set_tracking_uri", side_effect=RuntimeError("no creds")):
        ok = tracing.init_tracing("/Shared/ml-intern")
    assert ok is False


def test_init_idempotent_after_first_success():
    with patch("mlflow.set_tracking_uri") as set_uri, \
         patch("mlflow.set_registry_uri"), \
         patch("mlflow.set_experiment"):
        assert tracing.init_tracing("/Shared/ml-intern") is True
        # Second call returns True without re-invoking MLflow setters.
        assert tracing.init_tracing("/Shared/ml-intern") is True
        assert set_uri.call_count == 1


def test_trace_span_yields_none_when_uninitialised():
    with tracing.trace_span("foo") as span:
        assert span is None


def test_trace_span_swallows_mlflow_failure():
    with patch("mlflow.set_tracking_uri"), patch("mlflow.set_registry_uri"), \
         patch("mlflow.set_experiment"):
        tracing.init_tracing("/Shared/ml-intern")

    with patch("mlflow.start_span", side_effect=RuntimeError("boom")):
        with tracing.trace_span("foo") as span:
            assert span is None  # swallowed, not raised


@pytest.mark.asyncio
async def test_traced_decorator_passthrough_when_uninitialised():
    @tracing.traced("op")
    async def fn(x):
        return x * 2

    assert await fn(3) == 6


def test_traced_sync_decorator_passthrough_when_uninitialised():
    @tracing.traced("op")
    def fn(x):
        return x + 1

    assert fn(4) == 5
