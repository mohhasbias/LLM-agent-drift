"""Controlled drift injection for learning-agent evaluation."""

from __future__ import annotations

import copy
from typing import Any


def _arg_rename_key(tool_name: str, arg_name: str) -> str:
    return f"{arg_name}_v2"


def _tool_rename_name(tool_name: str) -> str:
    return f"{tool_name}__new"


def _rename_tool_in_schema(tools: list[dict[str, Any]], old: str, new: str) -> None:
    for tool in tools:
        fn = tool.get("function", {})
        if fn.get("name") == old:
            fn["name"] = new


def _ensure_confirm_tool(tools: list[dict[str, Any]]) -> None:
    tools.append(
        {
            "type": "function",
            "function": {
                "name": "confirm_policy_compliance",
                "description": "Acknowledge new policy before acting.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
    )


def apply_drift(scenario: dict[str, Any], drift_type: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a deep-copied scenario with injected drift and drift metadata.

    Supported drift types:
      - none
      - schema: rename argument keys in expected actions and note in task text.
      - toolset: rename expected tool function names and tool schema names.
      - policy: enforce pre-action policy confirmation tool call.
    """
    out = copy.deepcopy(scenario)
    metadata: dict[str, Any] = {"drift_type": drift_type}

    if drift_type == "none":
        return out, metadata

    if not out.get("english_answer_list"):
        return out, metadata

    # Current datasets in this experiment are one-task scenarios.
    answer_steps: list[dict[str, Any]] = out["english_answer_list"][0]
    tools: list[dict[str, Any]] = out.get("english_tools", [])
    task_text = out["english_tasks"][0] if out.get("english_tasks") else ""

    if drift_type == "schema":
        rename_map: dict[str, dict[str, str]] = {}
        for step in answer_steps:
            action = step.get("action", {})
            tool_name = str(action.get("name", ""))
            args = action.get("arguments")
            if not isinstance(args, dict) or not args:
                continue
            arg_map: dict[str, str] = {}
            for key in list(args.keys()):
                new_key = _arg_rename_key(tool_name, key)
                args[new_key] = args.pop(key)
                arg_map[key] = new_key
            if arg_map:
                rename_map[tool_name] = arg_map
        out["english_tasks"][0] = (
            f"{task_text}\n\n[Drift Notice] Tool schemas changed; use updated argument keys."
        )
        metadata["schema_rename_map"] = rename_map
        return out, metadata

    if drift_type == "toolset":
        rename_map: dict[str, str] = {}
        for step in answer_steps:
            action = step.get("action", {})
            old_name = action.get("name")
            if not isinstance(old_name, str) or not old_name:
                continue
            new_name = _tool_rename_name(old_name)
            action["name"] = new_name
            rename_map[old_name] = new_name
        for old_name, new_name in rename_map.items():
            _rename_tool_in_schema(tools, old_name, new_name)
        out["english_tasks"][0] = (
            f"{task_text}\n\n[Drift Notice] Tool names have changed to new API versions."
        )
        metadata["tool_rename_map"] = rename_map
        return out, metadata

    if drift_type == "policy":
        _ensure_confirm_tool(tools)
        confirm_step = {
            "idx": -1,
            "dependency_list": [],
            "action": {"name": "confirm_policy_compliance", "arguments": {}},
            "observation": "Policy acknowledged.",
        }
        for step in answer_steps:
            deps = step.get("dependency_list")
            if isinstance(deps, list):
                step["dependency_list"] = [dep + 1 for dep in deps]
            idx = step.get("idx")
            if isinstance(idx, int):
                step["idx"] = idx + 1
        answer_steps.insert(0, confirm_step)
        out["english_tasks"][0] = (
            f"{task_text}\n\n[Policy Update] You must call "
            "`confirm_policy_compliance` once before any other tool call."
        )
        metadata["required_first_tool"] = "confirm_policy_compliance"
        return out, metadata

    raise ValueError(f"Unsupported drift_type: {drift_type}")
