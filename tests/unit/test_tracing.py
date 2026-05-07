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


def test_init_binds_experiment_id_for_trace_export():
    """The v3 trace exporter logs ``trace_info.mlflow_experiment.experiment_id
    is missing`` whenever a span flushes outside an active run. We sidestep
    by exporting ``MLFLOW_EXPERIMENT_ID`` from the experiment that
    ``set_experiment`` returns."""
    import os

    fake_experiment = type("E", (), {"experiment_id": "424242"})()
    with patch("mlflow.set_tracking_uri"), \
         patch("mlflow.set_registry_uri"), \
         patch("mlflow.set_experiment", return_value=fake_experiment):
        os.environ.pop("MLFLOW_EXPERIMENT_ID", None)
        assert tracing.init_tracing("/Shared/ml-intern") is True
        assert os.environ.get("MLFLOW_EXPERIMENT_ID") == "424242"


def test_reset_for_tests_clears_experiment_id_env_var():
    import os

    os.environ["MLFLOW_EXPERIMENT_ID"] = "stale-from-prior-test"
    tracing.reset_for_tests()
    assert "MLFLOW_EXPERIMENT_ID" not in os.environ


def test_trace_span_stamps_experiment_id_attribute():
    """Per-span ``mlflow.experimentId`` attribute makes the v3 exporter's
    experiment lookup deterministic — without it traces flushed during
    process exit can race the active-experiment globals and the backend
    rejects them as missing experiment_id."""
    fake_experiment = type("E", (), {"experiment_id": "9001"})()
    captured: list[dict] = []

    class _FakeSpan:
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake_start_span(name, attributes=None, **_):
        captured.append({"name": name, "attributes": attributes})
        return _FakeSpan()

    with patch("mlflow.set_tracking_uri"), \
         patch("mlflow.set_registry_uri"), \
         patch("mlflow.set_experiment", return_value=fake_experiment):
        tracing.init_tracing("/Shared/ml-intern")

    with patch("mlflow.start_span", side_effect=fake_start_span):
        with tracing.trace_span("foo", {"k": "v"}):
            pass

    assert captured, "trace_span never invoked mlflow.start_span"
    attrs = captured[0]["attributes"]
    assert attrs["mlflow.experimentId"] == "9001"
    # Caller's existing attributes survive.
    assert attrs["k"] == "v"


@pytest.mark.asyncio
async def test_traced_decorator_stamps_experiment_id_attribute():
    fake_experiment = type("E", (), {"experiment_id": "9002"})()
    captured: list[dict] = []

    class _FakeSpan:
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake_start_span(name, attributes=None, **_):
        captured.append({"name": name, "attributes": attributes})
        return _FakeSpan()

    with patch("mlflow.set_tracking_uri"), \
         patch("mlflow.set_registry_uri"), \
         patch("mlflow.set_experiment", return_value=fake_experiment):
        tracing.init_tracing("/Shared/ml-intern")

    @tracing.traced("op")
    async def fn(x):
        return x + 1

    with patch("mlflow.start_span", side_effect=fake_start_span):
        assert await fn(2) == 3

    assert captured[0]["attributes"]["mlflow.experimentId"] == "9002"


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
