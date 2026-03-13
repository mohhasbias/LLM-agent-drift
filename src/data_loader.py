import json, random
from collections import defaultdict
from typing import Any


def load_scenarios(jsonl_path: str) -> list[dict[str, Any]]:
    with open(jsonl_path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _dominant_task_type(scenario: dict) -> str:
    types = scenario.get("english_task_types", [])
    if not types:
        return "unknown"
    counts: dict[str, int] = defaultdict(int)
    for t in types:
        counts[t] += 1
    return max(counts, key=counts.get)


def stratified_sample(
    scenarios: list[dict[str, Any]], n: int, seed: int = 42
) -> list[dict[str, Any]]:
    by_type: dict[str, list] = defaultdict(list)
    for s in scenarios:
        by_type[_dominant_task_type(s)].append(s)

    rng = random.Random(seed)
    total = len(scenarios)
    sampled: list[dict[str, Any]] = []
    for group in by_type.values():
        k = max(1, round(n * len(group) / total))
        sampled.extend(rng.sample(group, min(k, len(group))))

    rng.shuffle(sampled)
    result = sampled[:n]
    if len(result) < n:
        raise ValueError(
            f"Could only collect {len(result)} samples across strata; "
            f"requested {n}. Check stratum sizes."
        )
    return result


def save_sample_index(scenario_ids: list[str], path: str) -> None:
    with open(path, "w") as f:
        json.dump(scenario_ids, f, indent=2)


def load_sample_index(path: str) -> list[str]:
    with open(path) as f:
        return json.load(f)
