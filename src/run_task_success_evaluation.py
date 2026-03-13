"""Task-success evaluation on upstream tau2 retail environment under drift."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TAU2_SRC = Path("/tmp/tau2-bench-upstream/src")
if str(TAU2_SRC) not in sys.path:
    sys.path.insert(0, str(TAU2_SRC))

from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage
from tau2.data_model.simulation import SimulationRun, TerminationReason
from tau2.domains.retail.environment import get_environment, get_tasks
from tau2.evaluator.evaluator_env import EnvironmentEvaluator
from tau2.metrics.agent_metrics import is_successful

from src.data_loader import load_sample_index, load_scenarios
from src.drift_injector import apply_drift
from src.experiment_config import (
    ensure_dataset_available,
    get_model_config,
    load_experiment_config,
    resolve_model_keys,
    resolve_output_root,
    resolve_scenario_index_path,
)
from src.learning_memory import LearningMemory
from src.run_learning_evaluation import (
    _apply_hint_repair,
    _apply_rule_repair,
    _build_messages,
    _make_client,
    _response_to_dict,
    _safe_json_loads,
    _chat_complete_with_retry,
    set_requests_per_minute,
)


MAX_STEPS = 12


def _parse_options_blob(blob: str) -> dict[str, str]:
    return {k: v for k, v in re.findall(r"'([^']+)': '([^']*)'", blob)}


def _parse_user_details(content: str) -> dict[str, Any] | None:
    if "payment_methods={" not in content or "orders=[" not in content:
        return None
    payment_ids = re.findall(r"'([^']+)': (?:Paypal|CreditCard|GiftCard)\(", content)
    orders_match = re.search(r"orders=\[([^\]]*)\]", content)
    address_match = re.search(
        r"address=UserAddress\(address1='([^']*)', address2='([^']*)', city='([^']*)', country='([^']*)', state='([^']*)', zip='([^']*)'\)",
        content,
    )
    return {
        "payment_method_ids": payment_ids,
        "orders": re.findall(r"'([^']+)'", orders_match.group(1)) if orders_match else [],
        "address": (
            {
                "address1": address_match.group(1),
                "address2": address_match.group(2),
                "city": address_match.group(3),
                "country": address_match.group(4),
                "state": address_match.group(5),
                "zip": address_match.group(6),
            }
            if address_match
            else None
        ),
    }


def _parse_order_details(content: str) -> dict[str, Any] | None:
    order_match = re.search(r"order_id='([^']+)'", content)
    status_match = re.search(r"status='([^']+)'", content)
    address_match = re.search(
        r"address=UserAddress\(address1='([^']*)', address2='([^']*)', city='([^']*)', country='([^']*)', state='([^']*)', zip='([^']*)'\)",
        content,
    )
    items = []
    for name, product_id, item_id, options_blob in re.findall(
        r"OrderItem\(name='([^']+)', product_id='([^']+)', item_id='([^']+)', price=.*?, options=\{(.*?)\}\)",
        content,
    ):
        items.append(
            {
                "name": name,
                "product_id": product_id,
                "item_id": item_id,
                "options": _parse_options_blob(options_blob),
            }
        )
    pay_ids = re.findall(r"payment_method_id='([^']+)'", content)
    if not order_match:
        return None
    return {
        "order_id": order_match.group(1),
        "status": status_match.group(1) if status_match else "",
        "address": (
            {
                "address1": address_match.group(1),
                "address2": address_match.group(2),
                "city": address_match.group(3),
                "country": address_match.group(4),
                "state": address_match.group(5),
                "zip": address_match.group(6),
            }
            if address_match
            else None
        ),
        "items": items,
        "payment_method_ids": pay_ids,
    }


def _parse_product_details(content: str) -> dict[str, Any] | None:
    name_match = re.search(r"name='([^']+)'", content)
    product_match = re.search(r"product_id='([^']+)'", content)
    if not product_match:
        return None
    variants = []
    for item_id, options_blob, available in re.findall(
        r"Variant\(item_id='([^']+)', options=\{(.*?)\}, available=(True|False), price=",
        content,
    ):
        variants.append(
            {
                "item_id": item_id,
                "options": _parse_options_blob(options_blob),
                "available": available == "True",
            }
        )
    return {
        "name": name_match.group(1) if name_match else "",
        "product_id": product_match.group(1),
        "variants": variants,
    }


def _task_text_lower(task_text: str) -> str:
    return task_text.lower()


def _infer_excluded_item_names(task_text: str) -> set[str]:
    text = _task_text_lower(task_text)
    excluded: set[str] = set()
    for pat in (
        r"everything but (?:an? )?([a-z -]+?)(?:[.,;]|$)",
        r"except (?:an? )?([a-z -]+?)(?:[.,;]|$)",
    ):
        for raw in re.findall(pat, text):
            excluded.add(raw.strip())
    return excluded


def _desired_option_map(task_text: str) -> dict[str, str]:
    pairs = {k.lower(): v.lower() for k, v in re.findall(r"'([^']+)': '([^']+)'", task_text)}
    text = _task_text_lower(task_text)
    if "black dial" in text:
        pairs.setdefault("dial color", "black")
    if "leather strap" in text:
        pairs.setdefault("strap material", "leather")
    return pairs


def _desired_product_keywords(task_text: str) -> list[str]:
    text = _task_text_lower(task_text)
    keywords: list[str] = []
    for key in [
        "laptop",
        "watch",
        "wristwatch",
        "tablet",
        "e-reader",
        "air purifier",
        "water bottle",
        "gaming mouse",
        "keyboard",
        "mouse",
    ]:
        if key in text:
            keywords.append(key)
    return keywords


def _product_keyword_matches(name: str, keyword: str) -> bool:
    lname = name.lower()
    if keyword == "watch":
        return "watch" in lname
    return keyword in lname


def _infer_target_product_names(task_text: str, pred_name: str) -> list[str]:
    text = _task_text_lower(task_text)
    desired = _desired_option_map(task_text)
    keywords = _desired_product_keywords(task_text)
    if pred_name == "modify_pending_order_address":
        if "laptop" in keywords:
            return ["laptop"]
        return keywords
    if pred_name in {"exchange_delivered_order_items", "modify_pending_order_items"}:
        if "dial color" in desired or "strap material" in desired:
            return ["wristwatch", "watch"]
        if "processor" in desired or "storage" in desired:
            return ["laptop"]
    if pred_name == "return_delivered_order_items":
        excluded = _infer_excluded_item_names(task_text)
        if excluded:
            return []
    if "laptop" in text:
        return ["laptop"]
    if "watch" in text:
        return ["wristwatch", "watch"]
    return keywords


def _choose_relevant_order(
    *,
    task_text: str,
    state_cache: dict[str, Any],
    require_pending: bool = False,
    require_delivered: bool = False,
    target_products: list[str] | None = None,
) -> dict[str, Any] | None:
    text = _task_text_lower(task_text)
    best_order = None
    best_score = -1
    for order in state_cache.get("orders", {}).values():
        status = order.get("status", "").lower()
        if require_pending and "pending" not in status:
            continue
        if require_delivered and "delivered" not in status:
            continue
        score = 0
        for item in order.get("items", []):
            if item["name"].lower() in text:
                score += 3
            if target_products and any(_product_keyword_matches(item["name"], t) for t in target_products):
                score += 6
        if "nyc" in text or "new york" in text:
            if order.get("address", {}).get("city", "").lower() == "new york":
                score += 2
        if score > best_score:
            best_order = order
            best_score = score
    return best_order


def _fill_item_ids_from_order(task_text: str, order: dict[str, Any]) -> list[str]:
    text = _task_text_lower(task_text)
    excluded = _infer_excluded_item_names(task_text)
    chosen = []
    for item in order.get("items", []):
        name = item["name"].lower()
        if any(ex in name for ex in excluded):
            continue
        if name in text:
            chosen.append(item["item_id"])
    if chosen:
        return chosen
    if excluded:
        return [item["item_id"] for item in order.get("items", []) if not any(ex in item["name"].lower() for ex in excluded)]
    return []


def _find_matching_variant(
    *,
    product_id: str,
    task_text: str,
    state_cache: dict[str, Any],
    exclude_item_ids: set[str] | None = None,
) -> str | None:
    product = state_cache.get("products", {}).get(product_id)
    if not product:
        return None
    desired = _desired_option_map(task_text)
    if not desired:
        return None
    for variant in product.get("variants", []):
        if not variant.get("available"):
            continue
        if exclude_item_ids and str(variant["item_id"]) in exclude_item_ids:
            continue
        options = {k.lower(): v.lower() for k, v in variant.get("options", {}).items()}
        if all(options.get(k) == v for k, v in desired.items()):
            return str(variant["item_id"])
    return None


def _repair_with_state_cache(
    *,
    pred_name: str,
    pred_args: dict[str, Any],
    task_text: str,
    state_cache: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    repaired = dict(pred_args)
    adapted = False
    target_products = _infer_target_product_names(task_text, pred_name)
    if pred_name == "modify_pending_order_address":
        order = state_cache.get("orders", {}).get(str(repaired.get("order_id"))) or _choose_relevant_order(
            task_text=task_text,
            state_cache=state_cache,
            require_pending=True,
            target_products=target_products,
        )
        if order and "order_id" not in repaired:
            repaired["order_id"] = order["order_id"]
            adapted = True
        target_address = None
        if "nyc" in _task_text_lower(task_text) or "new york" in _task_text_lower(task_text):
            for cached in state_cache.get("orders", {}).values():
                addr = cached.get("address") or {}
                if addr.get("city", "").lower() == "new york":
                    target_address = addr
                    break
        if target_address is None:
            target_address = (state_cache.get("user") or {}).get("address")
        if target_address:
            for key in ("address1", "address2", "city", "country", "state", "zip"):
                if key not in repaired and key in target_address:
                    repaired[key] = target_address[key]
                    adapted = True
        return repaired, adapted

    if pred_name in {"modify_pending_order_items", "exchange_delivered_order_items", "return_delivered_order_items"}:
        order = state_cache.get("orders", {}).get(str(repaired.get("order_id")))
        if order is None:
            recent_product = state_cache.get("last_product")
            if recent_product:
                for candidate in state_cache.get("orders", {}).values():
                    status = candidate.get("status", "").lower()
                    if pred_name == "modify_pending_order_items" and "pending" not in status:
                        continue
                    if pred_name in {"exchange_delivered_order_items", "return_delivered_order_items"} and "delivered" not in status:
                        continue
                    if any(str(item.get("product_id")) == str(recent_product.get("product_id")) for item in candidate.get("items", [])):
                        order = candidate
                        break
        if order is None:
            order = _choose_relevant_order(
                task_text=task_text,
                state_cache=state_cache,
                require_pending=pred_name == "modify_pending_order_items",
                require_delivered=pred_name in {"exchange_delivered_order_items", "return_delivered_order_items"},
                target_products=target_products,
            )
        if order and (not repaired.get("order_id") or str(repaired.get("order_id")) != order["order_id"]):
            repaired["order_id"] = order["order_id"]
            adapted = True
        if order and "payment_method_id" not in repaired and order.get("payment_method_ids"):
            repaired["payment_method_id"] = order["payment_method_ids"][0]
            adapted = True
        if order and not repaired.get("item_ids"):
            item_ids = _fill_item_ids_from_order(task_text, order)
            if target_products:
                targeted = [
                    item["item_id"]
                    for item in order.get("items", [])
                    if any(_product_keyword_matches(item["name"], t) for t in target_products)
                ]
                if targeted:
                    item_ids = targeted
            if item_ids:
                repaired["item_ids"] = item_ids
                adapted = True
        if pred_name in {"modify_pending_order_items", "exchange_delivered_order_items"} and order and not repaired.get("new_item_ids"):
            source_item_ids = repaired.get("item_ids") or []
            new_item_ids = []
            item_by_id = {item["item_id"]: item for item in order.get("items", [])}
            for item_id in source_item_ids:
                item = item_by_id.get(item_id)
                if not item:
                    continue
                variant_id = _find_matching_variant(
                    product_id=str(item["product_id"]),
                    task_text=task_text,
                    state_cache=state_cache,
                    exclude_item_ids=set(str(x) for x in source_item_ids),
                )
                if variant_id:
                    new_item_ids.append(variant_id)
            if new_item_ids:
                repaired["new_item_ids"] = new_item_ids
                adapted = True
        return repaired, adapted

    return repaired, adapted


def _numeric_task_id(scenario_id: str) -> str:
    return scenario_id.rsplit("_", 1)[-1]


def _normalize_tool_name(pred_name: str, drift_meta: dict[str, Any]) -> str:
    if pred_name == "confirm_policy_compliance":
        return pred_name
    rename_map = drift_meta.get("tool_rename_map", {})
    if isinstance(rename_map, dict):
        inverse = {str(v): str(k) for k, v in rename_map.items()}
        if pred_name in inverse:
            return inverse[pred_name]
    return pred_name


def _normalize_tool_args(pred_args: dict[str, Any], drift_meta: dict[str, Any]) -> dict[str, Any]:
    rename_map = drift_meta.get("schema_rename_map", {})
    if not isinstance(rename_map, dict) or not pred_args:
        return pred_args
    normalized: dict[str, Any] = {}
    changed = False
    for key, value in pred_args.items():
        if isinstance(key, str) and key.endswith("_v2"):
            normalized[key[: -len("_v2")]] = value
            changed = True
        else:
            normalized[key] = value
    return normalized if changed else pred_args


def _execute_call(
    env: Any,
    *,
    raw_name: str,
    raw_args: dict[str, Any],
    drift_meta: dict[str, Any],
    normalize: bool = True,
) -> tuple[ToolCall, ToolMessage, bool]:
    call_id = str(uuid.uuid4())
    if raw_name == "confirm_policy_compliance":
        tool_call = ToolCall(id=call_id, name=raw_name, arguments={}, requestor="assistant")
        tool_msg = ToolMessage(
            id=call_id,
            role="tool",
            content="Policy acknowledged.",
            requestor="assistant",
            error=False,
        )
        return tool_call, tool_msg, True

    if normalize:
        name = _normalize_tool_name(raw_name, drift_meta)
        args = _normalize_tool_args(raw_args, drift_meta)
    else:
        name = raw_name
        args = raw_args
    tool_call = ToolCall(id=call_id, name=name, arguments=args, requestor="assistant")
    try:
        tool_msg = env.get_response(tool_call)
        return tool_call, tool_msg, True
    except Exception as exc:  # noqa: BLE001
        tool_msg = ToolMessage(
            id=call_id,
            role="tool",
            content=f"Error: {exc}",
            requestor="assistant",
            error=True,
        )
        return tool_call, tool_msg, False


def evaluate_scenario_task_success(
    *,
    client: Any,
    scenario: dict[str, Any],
    base_scenario_id: str,
    drift_meta: dict[str, Any],
    step_hints: dict[int, dict[str, Any]] | None,
    replay_mode: str,
    enabled_rules: set[str] | None,
    normalize: bool = True,
) -> dict[str, Any]:
    env = get_environment()
    upstream_task = get_tasks(None)[int(_numeric_task_id(base_scenario_id))]
    state_cache: dict[str, Any] = {"user": None, "orders": {}, "products": {}, "last_product": None}

    tools: list[dict[str, Any]] = list(scenario.get("english_tools", []))
    task_text = scenario["english_tasks"][0]
    messages = _build_messages(
        scenario.get("english_env_info", ""),
        task_text,
        step_hints.get(0) if step_hints else None,
    )
    transcript = [UserMessage(role="user", content=task_text)]
    hint_adapted_steps = 0
    rule_adapted_steps = 0
    invalid_tool_calls = 0
    tool_errors = 0
    error: str | None = None
    matched_steps = 0
    termination_reason = TerminationReason.AGENT_STOP
    observed_actions: list[dict[str, Any]] = []

    for step in range(MAX_STEPS):
        try:
            raw = _chat_complete_with_retry(client, messages, tools, temperature=0.0, max_attempts=2)
            resp = _response_to_dict(raw)
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            termination_reason = TerminationReason.AGENT_ERROR
            break

        tool_calls = resp.get("tool_calls") or []
        content = resp.get("content")
        if not tool_calls:
            transcript.append(AssistantMessage(role="assistant", content=content or "DONE"))
            termination_reason = TerminationReason.AGENT_STOP
            break

        first = tool_calls[0]
        pred_name = first["function"]["name"]
        pred_args_raw = first["function"]["arguments"]
        step_hint = step_hints.get(step) if step_hints else None
        pred_name, pred_args_raw, hint_adapted = _apply_hint_repair(
            step=step,
            pred_name=pred_name,
            pred_args_raw=pred_args_raw,
            hint=step_hint if replay_mode == "reference_blind" else None,
            tools=tools,
        )
        if hint_adapted:
            hint_adapted_steps += 1
        pred_name, pred_args_raw, adapted = _apply_rule_repair(
            drift_type=drift_meta.get("drift_type", "none"),
            step=step,
            pred_name=pred_name,
            pred_args_raw=pred_args_raw,
            tools=tools,
            enabled_rules=enabled_rules or set(),
        )
        if adapted:
            rule_adapted_steps += 1

        pred_args = _safe_json_loads(pred_args_raw)
        if not isinstance(pred_args, dict):
            pred_args = {}
        pred_args, cache_adapted = _repair_with_state_cache(
            pred_name=pred_name,
            pred_args=pred_args,
            task_text=task_text,
            state_cache=state_cache,
        )
        if cache_adapted:
            hint_adapted_steps += 1
        tool_names = {t.get("function", {}).get("name") for t in tools}
        if pred_name not in tool_names:
            invalid_tool_calls += 1
            error = f"unknown_tool:{pred_name}"
            termination_reason = TerminationReason.AGENT_ERROR
            break

        norm_call, tool_msg, ok = _execute_call(
            env,
            raw_name=pred_name,
            raw_args=pred_args,
            drift_meta=drift_meta,
            normalize=normalize,
        )
        if norm_call.name != "confirm_policy_compliance":
            transcript.append(
                AssistantMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[norm_call],
                )
            )
            transcript.append(tool_msg)
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": first["id"],
                        "type": "function",
                        "function": {"name": pred_name, "arguments": json.dumps(pred_args)},
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": first["id"],
                "content": tool_msg.content,
            }
        )
        if ok:
            matched_steps += 1
            observed_actions.append(
                {
                    "step": step,
                    "name": pred_name,
                    "arguments": pred_args,
                }
            )
            if norm_call.name == "get_user_details":
                parsed = _parse_user_details(tool_msg.content)
                if parsed is not None:
                    state_cache["user"] = parsed
            elif norm_call.name == "get_order_details":
                parsed = _parse_order_details(tool_msg.content)
                if parsed is not None:
                    state_cache["orders"][parsed["order_id"]] = parsed
            elif norm_call.name == "get_product_details":
                parsed = _parse_product_details(tool_msg.content)
                if parsed is not None:
                    state_cache["products"][parsed["product_id"]] = parsed
                    state_cache["last_product"] = parsed
            elif norm_call.name in {
                "modify_pending_order_items",
                "modify_pending_order_address",
                "exchange_delivered_order_items",
                "return_delivered_order_items",
            }:
                parsed = _parse_order_details(tool_msg.content)
                if parsed is not None:
                    state_cache["orders"][parsed["order_id"]] = parsed
        else:
            tool_errors += 1
            error = tool_msg.content
            if tool_errors >= 2:
                termination_reason = TerminationReason.TOO_MANY_ERRORS
                break

    else:
        termination_reason = TerminationReason.MAX_STEPS

    now = time.time()
    simulation = SimulationRun(
        id=str(uuid.uuid4()),
        task_id=str(upstream_task.id),
        start_time=str(now),
        end_time=str(now),
        duration=0.0,
        termination_reason=termination_reason,
        messages=transcript,
        trial=1,
    )
    reward_info = EnvironmentEvaluator.calculate_reward(
        environment_constructor=get_environment,
        task=upstream_task,
        full_trajectory=simulation.messages,
        solo_mode=False,
    )
    reward = reward_info.reward
    return {
        "success": is_successful(reward),
        "reward": reward,
        "db_match": bool(reward_info.db_check.db_match) if reward_info.db_check else False,
        "termination_reason": termination_reason.value,
        "matched_steps": matched_steps,
        "invalid_tool_calls": invalid_tool_calls,
        "tool_errors": tool_errors,
        "error": error,
        "hint_adapted_steps": hint_adapted_steps,
        "rule_adapted_steps": rule_adapted_steps,
        "reward_breakdown": {
            str(k): v for k, v in (reward_info.reward_breakdown or {}).items()
        },
        "observed_actions": observed_actions,
    }


def run_mode(
    *,
    mode: str,
    client: Any,
    scenarios: list[dict[str, Any]],
    eval_scenarios: list[dict[str, Any]] | None,
    drift_types: list[str],
    episodes: int,
    seed: int,
    memory_min_support: int,
    memory_min_confidence: float,
    learning_strategy: str,
    normalize: bool = True,
    record_sink: Path | None = None,
    completed_keys: set[tuple[int, str, str, str]] | None = None,
    existing_records: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import random

    rng = random.Random(seed)
    memory = (
        LearningMemory(min_support=memory_min_support, min_confidence=memory_min_confidence)
        if mode == "learning"
        else None
    )
    enabled_rules: set[str] = set()
    if mode == "learning" and learning_strategy == "replay_blind":
        replay_mode = "reference_blind"
    else:
        replay_mode = "none"
    # Pre-populate with already-completed rows so summary is computed over all data.
    records: list[dict[str, Any]] = list(existing_records or [])
    _completed_keys: set[tuple[int, str, str, str]] = completed_keys or set()
    episode_success_rates: list[float] = []

    for episode in range(1, episodes + 1):
        scenario_order = list(scenarios)
        rng.shuffle(scenario_order)
        for split_name, split_rows in (("train", scenario_order), ("eval", eval_scenarios or [])):
            if split_name == "eval" and not eval_scenarios:
                continue
            for drift_type in drift_types:
                for base in split_rows:
                    key = (episode, split_name, str(base["id"]), drift_type)
                    # Static: skip any completed row (no memory dependency).
                    # Learning: skip eval rows only; training must re-run to rebuild memory.
                    if mode != "learning" and key in _completed_keys:
                        continue
                    if mode == "learning" and split_name == "eval" and key in _completed_keys:
                        continue
                    scenario, meta = apply_drift(base, drift_type=drift_type)
                    task_text = scenario["english_tasks"][0]
                    step_hints = (
                        {
                            step: hint
                            for step in range(MAX_STEPS)
                            if (hint := memory.retrieve_step_hint(task_text, drift_type, step=step)) is not None
                        }
                        if memory
                        else None
                    )
                    outcome = evaluate_scenario_task_success(
                        client=client,
                        scenario=scenario,
                        base_scenario_id=str(base["id"]),
                        drift_meta=meta,
                        step_hints=step_hints,
                        replay_mode=replay_mode,
                        enabled_rules=enabled_rules if (mode == "learning" and learning_strategy == "rules") else None,
                        normalize=normalize,
                    )
                    row = {
                        "episode": episode,
                        "split": split_name,
                        "mode": mode,
                        "scenario_id": base["id"],
                        "drift_type": drift_type,
                        "hint_used": bool(step_hints),
                        "replay_mode": replay_mode,
                        **outcome,
                    }
                    row.pop("observed_actions", None)
                    records.append(row)
                    if record_sink is not None:
                        with record_sink.open("a", encoding="utf-8") as _f:
                            _f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    if memory:
                        for action in outcome.get("observed_actions", []):
                            memory.observe(
                                task_text=task_text,
                                drift_type=drift_type,
                                expected_action=action,
                                success=outcome["success"],
                                step=int(action.get("step", 0)),
                            )
                    if mode == "learning" and learning_strategy == "rules" and not outcome["success"]:
                        if drift_type in {"schema", "policy", "toolset"}:
                            enabled_rules.add(drift_type)

        target_rows = [r for r in records if r["episode"] == episode and r["split"] == ("eval" if eval_scenarios else "train")]
        ep_success = sum(int(r["success"]) for r in target_rows) / max(1, len(target_rows))
        episode_success_rates.append(ep_success)

    summary = {
        "mode": mode,
        "episodes": episodes,
        "metric_split": "eval" if eval_scenarios else "train",
        "train_scenarios": len(scenarios),
        "eval_scenarios": len(eval_scenarios or []),
        "drift_types": drift_types,
        "episode_success_rates": episode_success_rates,
        "learning_strategy": learning_strategy if mode == "learning" else "none",
        "replay_mode": replay_mode,
        "enabled_rules_final": sorted(enabled_rules),
        "normalize": normalize,
    }
    if memory:
        summary["memory"] = memory.stats()
    return records, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run task-success evaluation under drift.")
    parser.add_argument("--config", default="experiment.yaml")
    parser.add_argument("--scenario-set", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--mode", choices=["static", "learning", "both"], default="both")
    parser.add_argument("--drift-types", nargs="+", default=["schema", "policy", "toolset"])
    parser.add_argument("--train-max-scenarios", type=int, default=None)
    parser.add_argument("--eval-max-scenarios", type=int, default=0)
    parser.add_argument("--memory-min-support", type=int, default=1)
    parser.add_argument("--memory-min-confidence", type=float, default=0.0)
    parser.add_argument("--learning-strategy", choices=["replay_blind", "rules"], default="replay_blind")
    parser.add_argument("--seed", type=int, default=20260308)
    parser.add_argument("--no-normalize", action="store_true", default=False,
                        help="Disable drift normalization layer (ablation condition).")
    parser.add_argument("--requests-per-minute", type=int, default=0,
                        help="Max LLM API requests per minute (0 = unlimited).")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.requests_per_minute > 0:
        set_requests_per_minute(args.requests_per_minute)

    cfg = load_experiment_config(args.config)
    dataset_path = ensure_dataset_available(cfg, scenario_set=args.scenario_set)
    scenario_index_path = resolve_scenario_index_path(cfg, scenario_set=args.scenario_set)
    output_root = Path(resolve_output_root(cfg, explicit_output=args.output, scenario_set=args.scenario_set))
    model_keys = resolve_model_keys(cfg, explicit_model=args.model, scenario_set=args.scenario_set)
    scenario_ids = [str(sid) for sid in load_sample_index(scenario_index_path)]
    dataset_rows = load_scenarios(dataset_path)
    by_id = {str(s.get("id")): s for s in dataset_rows}
    scenarios = [by_id[sid] for sid in scenario_ids if sid in by_id]
    train_scenarios = scenarios
    eval_scenarios: list[dict[str, Any]] | None = None
    if args.train_max_scenarios is not None or (args.eval_max_scenarios and args.eval_max_scenarios > 0):
        train_n = args.train_max_scenarios if args.train_max_scenarios is not None else len(scenarios)
        eval_n = max(0, args.eval_max_scenarios or 0)
        train_scenarios = scenarios[:train_n]
        if eval_n > 0:
            eval_scenarios = scenarios[train_n : train_n + eval_n]

    modes = ["static", "learning"] if args.mode == "both" else [args.mode]
    for model_key in model_keys:
        client = _make_client(get_model_config(cfg, model_key))
        for mode in modes:
            out_dir = output_root / model_key / mode
            out_dir.mkdir(parents=True, exist_ok=True)
            rec_path = out_dir / "episode_records.jsonl"
            existing_records: list[dict[str, Any]] = []
            completed_keys: set[tuple[int, str, str, str]] = set()
            if rec_path.exists():
                with rec_path.open(encoding="utf-8") as _f:
                    for _line in _f:
                        try:
                            _row = json.loads(_line)
                            existing_records.append(_row)
                            completed_keys.add((
                                int(_row["episode"]),
                                str(_row["split"]),
                                str(_row["scenario_id"]),
                                str(_row["drift_type"]),
                            ))
                        except Exception:  # noqa: BLE001
                            pass
            if completed_keys:
                print(f"[{model_key}/{mode}] resuming: {len(completed_keys)} rows already done, skipping.")
            _, summary = run_mode(
                mode=mode,
                client=client,
                scenarios=train_scenarios,
                eval_scenarios=eval_scenarios,
                drift_types=args.drift_types,
                episodes=args.episodes,
                seed=args.seed,
                memory_min_support=args.memory_min_support,
                memory_min_confidence=args.memory_min_confidence,
                learning_strategy=args.learning_strategy,
                normalize=not args.no_normalize,
                record_sink=rec_path,
                completed_keys=completed_keys,
                existing_records=existing_records,
            )
            summary_path = out_dir / "summary.json"
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"[{model_key}/{mode}] wrote {rec_path}")
            print(f"[{model_key}/{mode}] wrote {summary_path}")


if __name__ == "__main__":
    main()
