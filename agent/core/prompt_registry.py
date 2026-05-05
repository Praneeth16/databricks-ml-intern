"""System prompt source: MLflow Prompt Registry first, YAML fallback.

The agent's system prompt lives in two places:

1. **MLflow Prompt Registry** under ``ml_intern.agent.system_prompt`` —
   the source of truth in production. Versions are tied to bundle deploys
   so a rollback is one ``databricks bundle deploy --target prod
   --prompt-version <n>`` away.

2. **YAML at ``agent/prompts/system_prompt_v3.yaml``** — the fallback used
   in local CLI runs, unit tests, and any environment where the registry
   isn't reachable. Always shipped with the package so the agent has a
   prompt to start with even when the workspace is unreachable.

The registry takes precedence so prompt iteration doesn't require a code
deploy. ``load_system_prompt(name, version=...)`` tries the registry; on
any failure it falls back to the YAML and logs a warning.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_YAML = (
    Path(__file__).resolve().parent.parent / "prompts" / "system_prompt_v3.yaml"
)


def load_from_registry(name: str, version: str | int | None = None) -> Optional[str]:
    """Pull the named prompt from the MLflow Prompt Registry.

    ``version`` may be a version number (str / int), an alias ("production",
    "champion"), or None for the latest. Returns the rendered template
    string, or None on any failure (network, auth, not-found).
    """
    try:
        import mlflow

        mlflow.set_registry_uri("databricks-uc")
        # MLflow 2.21+ exposes ``mlflow.genai.load_prompt``; older builds
        # have ``mlflow.load_prompt``. Try both.
        loader = None
        try:
            from mlflow.genai import load_prompt as loader  # type: ignore
        except Exception:
            loader = getattr(mlflow, "load_prompt", None)
        if loader is None:
            logger.info("MLflow build has no load_prompt(); using YAML fallback.")
            return None

        if version is None:
            uri = f"prompts:/{name}/latest"
        else:
            uri = f"prompts:/{name}/{version}"
        prompt = loader(uri)
        # Both APIs return an object with a ``.template`` attribute.
        text = getattr(prompt, "template", None)
        if not text:
            text = str(prompt) if prompt is not None else None
        return text
    except Exception as e:
        logger.warning("Prompt registry lookup for %s failed: %s", name, e)
        return None


def load_from_yaml(path: Path | str | None = None) -> str:
    """Load the system_prompt template from the bundled YAML."""
    p = Path(path) if path else _DEFAULT_YAML
    with open(p, "r") as f:
        data = yaml.safe_load(f)
    return data.get("system_prompt", "")


def load_system_prompt(
    name: str,
    version: str | int | None = None,
    yaml_path: Path | str | None = None,
) -> str:
    """Resolve the system prompt, preferring the registry."""
    text = load_from_registry(name, version=version)
    if text:
        return text
    return load_from_yaml(yaml_path)
