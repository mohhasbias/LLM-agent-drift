"""Config-driven progress tracker for experiment execution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow `python src/progress_tracker.py ...` without pre-setting PYTHONPATH.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment_config import (
    load_experiment_config,
    resolve_conditions,
    resolve_model_keys,
    resolve_output_root,
    resolve_scenario_index_path,
)


def load_checkpoint(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as f:
        return sum(1 for _ in f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Track experiment progress from checkpoint/results.")
    parser.add_argument("--config", default="experiment.yaml")
    parser.add_argument(
        "--scenario-set",
        default=None,
        help="Scenario set key from config.scenarios.available. Defaults to scenarios.active.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write machine-readable summary JSON.",
    )
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    models = resolve_model_keys(cfg, scenario_set=args.scenario_set)

    scenario_index_path = Path(resolve_scenario_index_path(cfg, scenario_set=args.scenario_set))
    with scenario_index_path.open(encoding="utf-8") as f:
        scenario_ids = json.load(f)
    scenario_count = len(scenario_ids)
    conditions = resolve_conditions(cfg, scenario_set=args.scenario_set)
    output_root = Path(resolve_output_root(cfg, scenario_set=args.scenario_set))

    summary: dict[str, object] = {
        "config": cfg.get("_config_path"),
        "scenarios": scenario_count,
        "conditions": len(conditions),
        "models": {},
    }

    print(f"Config: {cfg.get('_config_path')}")
    print(f"Scenarios: {scenario_count} | Conditions: {len(conditions)}")
    print("")

    for model_key in models:
        checkpoint_path = output_root / model_key / "checkpoint.json"
        checkpoint = load_checkpoint(checkpoint_path)
        target_pairs = scenario_count * len(conditions)
        completed_pairs = sum(len(v) for v in checkpoint.values())

        per_condition = []
        for cond in conditions:
            result_file = output_root / model_key / cond / "Wild-Tool-Bench_result.jsonl"
            per_condition.append(
                {
                    "condition": cond,
                    "rows": count_lines(result_file),
                    "result_file": str(result_file),
                }
            )

        pct = (completed_pairs / target_pairs * 100.0) if target_pairs else 0.0
        print(f"[{model_key}] pairs: {completed_pairs}/{target_pairs} ({pct:.2f}%)")
        for row in per_condition:
            print(f"  {row['condition']}: {row['rows']} rows")
        print("")

        summary["models"][model_key] = {
            "completed_pairs": completed_pairs,
            "target_pairs": target_pairs,
            "completion_pct": pct,
            "checkpoint_path": str(checkpoint_path),
            "per_condition_rows": per_condition,
        }

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote summary JSON: {out_path}")


if __name__ == "__main__":
    main()
