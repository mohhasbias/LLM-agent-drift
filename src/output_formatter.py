"""Output formatter — produces WildToolBench eval_runner-compatible JSONL.

CRITICAL: output file must be named Wild-Tool-Bench_result.jsonl.

Each record schema:
  {id, model_name, result: [
    {action_name_label, is_optimal,
     inference_log: {task_idx, begin_of_current_task,
                     step_N: {inference_input, inference_output, inference_answer}},
     latency, input_token_count, output_token_count}
  ]}

tool_calls[i].function.arguments MUST be a JSON string (not a dict).
"""
import copy
import json
import os
from typing import Any

RESULT_FILENAME = "Wild-Tool-Bench_result.jsonl"


def _serialize_arguments(turn_result: dict[str, Any]) -> dict[str, Any]:
    """Return a deep copy of turn_result with all tool_call arguments as JSON strings."""
    turn_result = copy.deepcopy(turn_result)
    log = turn_result.get("inference_log", {})
    for key, step in log.items():
        if not key.startswith("step_"):
            continue
        tool_calls = step.get("inference_output", {}).get("tool_calls") or []
        for tc in tool_calls:
            args = tc.get("function", {}).get("arguments")
            if isinstance(args, dict):
                tc["function"]["arguments"] = json.dumps(args)
    return turn_result


def format_scenario_result(
    scenario_id: str,
    model_name: str,
    turn_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Wrap turn results into a top-level eval_runner scenario record.

    Ensures tool_calls[*].function.arguments are JSON strings.
    """
    return {
        "id": scenario_id,
        "model_name": model_name,
        "result": [_serialize_arguments(t) for t in turn_results],
    }


def write_condition_results(records: list[dict[str, Any]], output_dir: str) -> None:
    """Write records to {output_dir}/Wild-Tool-Bench_result.jsonl (one JSON per line)."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, RESULT_FILENAME)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
