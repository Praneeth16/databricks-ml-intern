"""Unit tests for agent.core.prompt_registry.

Behavior under test:
  - Registry hit → returns the registered template.
  - Registry miss / failure → falls back to YAML.
  - YAML loader returns ``system_prompt`` value.
"""

from __future__ import annotations

import textwrap
from unittest.mock import MagicMock, patch

from agent.core import prompt_registry


def test_load_from_yaml(tmp_path):
    p = tmp_path / "system_prompt.yaml"
    p.write_text("system_prompt: |\n  Hello {{ tools }}\n")
    out = prompt_registry.load_from_yaml(p)
    assert "Hello {{ tools }}" in out


def test_load_system_prompt_prefers_registry(tmp_path):
    p = tmp_path / "system_prompt.yaml"
    p.write_text("system_prompt: 'YAML-FALLBACK'\n")
    fake = MagicMock()
    fake.template = "REGISTRY-WIN"
    with patch.object(prompt_registry, "load_from_registry", return_value="REGISTRY-WIN"):
        out = prompt_registry.load_system_prompt("ml_intern.agent.system_prompt", yaml_path=p)
    assert out == "REGISTRY-WIN"


def test_load_system_prompt_falls_back_to_yaml(tmp_path):
    p = tmp_path / "system_prompt.yaml"
    p.write_text("system_prompt: 'YAML-FALLBACK'\n")
    with patch.object(prompt_registry, "load_from_registry", return_value=None):
        out = prompt_registry.load_system_prompt("ml_intern.agent.system_prompt", yaml_path=p)
    assert out == "YAML-FALLBACK"


def test_load_from_registry_returns_none_on_mlflow_import_failure():
    with patch("builtins.__import__", side_effect=ImportError("no mlflow")):
        out = prompt_registry.load_from_registry("anything")
    assert out is None


def test_load_from_registry_returns_template_attribute_when_present():
    fake_prompt = MagicMock()
    fake_prompt.template = "tpl-text"
    fake_loader = MagicMock(return_value=fake_prompt)
    fake_genai = MagicMock(load_prompt=fake_loader)
    with patch("mlflow.set_registry_uri"), \
         patch.dict(
             "sys.modules",
             {"mlflow.genai": fake_genai},
         ):
        out = prompt_registry.load_from_registry("name", version=3)
    assert out == "tpl-text"
    fake_loader.assert_called_once_with("prompts:/name/3")
