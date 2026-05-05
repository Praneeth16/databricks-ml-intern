"""MLflow Tracing for agent observability.

Replaces the previous HF-dataset trajectory upload + custom Delta KPI tables
with native MLflow Tracing. Each user turn becomes one trace; tool calls and
LLM calls hang off it as child spans. The MLflow UI under the configured
experiment shows the full agent loop tree, and Lakeview dashboards roll the
spans up into cost / latency / tool-mix charts.

Usage::

    init_tracing(settings.experiment_path)              # once per session

    with trace_span("agent_turn", {"turn_index": 4}):    # span every turn
        ...

    @traced(name="tool.run")                             # span every tool call
    async def call_tool(...): ...

All helpers are fail-soft: if MLflow init can't reach the workspace (e.g.
during unit tests), trace_span yields None and traced returns the function
unwrapped. The agent never breaks because telemetry can't connect.
"""

from __future__ import annotations

import contextlib
import functools
import logging
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

_INITIALIZED = False


def init_tracing(experiment_path: str | None) -> bool:
    """Configure MLflow tracking URI + experiment. Idempotent.

    Returns True if the configuration succeeded so callers can decide whether
    to skip span emission entirely. Failure (no host, no auth, network error)
    is logged at WARNING once and never raised.
    """
    global _INITIALIZED
    if _INITIALIZED:
        return True
    if not experiment_path:
        return False
    try:
        import mlflow

        mlflow.set_tracking_uri("databricks")
        mlflow.set_registry_uri("databricks-uc")
        mlflow.set_experiment(experiment_path)
        _INITIALIZED = True
        logger.info("MLflow Tracing initialised at experiment=%s", experiment_path)
        return True
    except Exception as e:
        logger.warning("MLflow Tracing init failed (%s) — running without traces.", e)
        return False


@contextlib.contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None):
    """Start an MLflow span. Yields the span (or None on failure)."""
    if not _INITIALIZED:
        yield None
        return
    try:
        import mlflow

        with mlflow.start_span(name=name, attributes=attributes or {}) as span:
            yield span
    except Exception as e:
        logger.debug("trace_span(%s) suppressed: %s", name, e)
        yield None


def traced(name: str | None = None):
    """Decorator wrapping a sync or async function in an MLflow span.

    Falls back to a passthrough wrapper when tracing isn't initialised so
    decorated functions never crash from a missing MLflow session.
    """
    def deco(fn: Callable[..., Any]):
        if _is_coro(fn):
            @functools.wraps(fn)
            async def awrapper(*args, **kwargs):
                if not _INITIALIZED:
                    return await fn(*args, **kwargs)
                try:
                    import mlflow
                    with mlflow.start_span(name=name or fn.__name__):
                        return await fn(*args, **kwargs)
                except Exception:
                    return await fn(*args, **kwargs)
            return awrapper

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if not _INITIALIZED:
                return fn(*args, **kwargs)
            try:
                import mlflow
                with mlflow.start_span(name=name or fn.__name__):
                    return fn(*args, **kwargs)
            except Exception:
                return fn(*args, **kwargs)
        return wrapper
    return deco


def _is_coro(fn: Callable[..., Any]) -> bool:
    import inspect
    return inspect.iscoroutinefunction(fn)


def reset_for_tests() -> None:
    """Test hook: forget initialisation so the next call re-runs."""
    global _INITIALIZED
    _INITIALIZED = False
