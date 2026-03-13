"""Shared experiment configuration loader.

This module centralizes how config paths and model metadata are read so the
runner, progress tracker, and analysis tools use one source of truth.
"""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve


def all_conditions() -> list[str]:
    return ["".join(bits) for bits in product("01", repeat=4)]


_VALID_FACTORS = {"F1", "F2", "F3", "F4"}
_VALID_ORDER_VARIANT_META_FIELDS = {"label", "rationale", "hypothesis"}


def _resolve_path(config_dir: Path, value: str | None) -> str | None:
    if not value:
        return value
    p = Path(value)
    if p.is_absolute():
        return str(p)
    return str((config_dir / p).resolve())


def _resolve_scenario_paths(config_dir: Path, scenario_value: dict[str, Any]) -> None:
    if "scenario_index_path" in scenario_value:
        scenario_value["scenario_index_path"] = _resolve_path(
            config_dir, scenario_value.get("scenario_index_path")
        )
    if "scenarios_path" in scenario_value:
        # Backward compatibility: normalize legacy key to scenario_index_path.
        resolved = _resolve_path(config_dir, scenario_value.get("scenarios_path"))
        if "scenario_index_path" not in scenario_value:
            scenario_value["scenario_index_path"] = resolved
        scenario_value.pop("scenarios_path", None)
    if "output_root" in scenario_value:
        scenario_value["output_root"] = _resolve_path(
            config_dir, scenario_value.get("output_root")
        )
    analysis_cfg = scenario_value.get("analysis")
    if isinstance(analysis_cfg, dict):
        for key in ("artifacts_root", "exploration_output_dir"):
            if key in analysis_cfg:
                analysis_cfg[key] = _resolve_path(config_dir, analysis_cfg.get(key))


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(current, value)
            continue
        merged[key] = value
    return merged


def _validate_factor_schema(cfg: dict[str, Any]) -> None:
    factors = cfg.get("factors")
    if factors is not None:
        if not isinstance(factors, dict) or not factors:
            raise ValueError("config.factors must be a non-empty mapping when provided.")
        factor_keys = set(factors.keys())
        if factor_keys != _VALID_FACTORS:
            raise ValueError(
                "config.factors keys must exactly match [F1,F2,F3,F4]. "
                f"Got: {sorted(factor_keys)}"
            )
        for key, value in factors.items():
            if not isinstance(value, dict):
                raise ValueError(f"config.factors.{key} must be a mapping.")
            factor_id = value.get("id")
            if factor_id is not None and str(factor_id) != key:
                raise ValueError(f"config.factors.{key}.id must equal '{key}'.")

    encoding = cfg.get("factor_encoding")
    if encoding is None:
        return
    if not isinstance(encoding, dict):
        raise ValueError("config.factor_encoding must be a mapping when provided.")
    bit_positions = encoding.get("bit_positions")
    if not isinstance(bit_positions, dict):
        raise ValueError(
            "config.factor_encoding.bit_positions is required and must be a mapping."
        )

    normalized: dict[str, str] = {}
    for raw_key, raw_factor in bit_positions.items():
        bit = str(raw_key)
        factor = str(raw_factor)
        if bit not in {"0", "1", "2", "3"}:
            raise ValueError(
                f"Invalid config.factor_encoding.bit_positions key '{bit}'. "
                "Allowed keys are 0,1,2,3."
            )
        if factor not in _VALID_FACTORS:
            raise ValueError(
                f"Invalid factor '{factor}' in config.factor_encoding.bit_positions[{bit}]. "
                "Allowed factors are F1,F2,F3,F4."
            )
        normalized[bit] = factor

    if set(normalized.keys()) != {"0", "1", "2", "3"}:
        raise ValueError("config.factor_encoding.bit_positions must define all keys 0,1,2,3.")
    if set(normalized.values()) != _VALID_FACTORS:
        raise ValueError(
            "config.factor_encoding.bit_positions must map to each factor exactly once."
        )
    if factors is not None and set(normalized.values()) != set(factors.keys()):
        raise ValueError(
            "config.factor_encoding.bit_positions values must match config.factors keys."
        )


def _validate_order_variant_schema(cfg: dict[str, Any]) -> None:
    variants = cfg.get("order_variants")
    if variants is None:
        return
    if not isinstance(variants, dict):
        raise ValueError("config.order_variants must be a mapping when provided.")
    for variant_id, variant_cfg in variants.items():
        if not isinstance(variant_cfg, dict):
            raise ValueError(f"config.order_variants.{variant_id} must be a mapping.")
        for field in _VALID_ORDER_VARIANT_META_FIELDS:
            if field in variant_cfg and not isinstance(variant_cfg[field], str):
                raise ValueError(
                    f"config.order_variants.{variant_id}.{field} must be a string when provided."
                )


def load_experiment_config(path: str) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as f:
        raw = f.read()

    suffix = config_path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        # Prefer PyYAML when available; fallback to JSON-compatible YAML.
        try:
            import yaml  # type: ignore
            loaded = yaml.safe_load(raw)
        except ModuleNotFoundError:
            loaded = json.loads(raw)
        cfg = loaded if isinstance(loaded, dict) else {}
    else:
        cfg = json.loads(raw)

    cfg["_config_path"] = str(config_path)
    cfg["_config_dir"] = str(config_path.parent)
    config_dir = config_path.parent

    # Resolve common path fields relative to config location for portability.
    for key in ("dataset_path", "scenario_index_path", "output_root"):
        if key in cfg:
            cfg[key] = _resolve_path(config_dir, cfg.get(key))
    if "scenarios_path" in cfg:
        # Backward compatibility: normalize legacy top-level key.
        cfg["scenario_index_path"] = _resolve_path(config_dir, cfg.get("scenarios_path"))
        cfg.pop("scenarios_path", None)
    scenarios = cfg.get("scenarios")
    if isinstance(scenarios, dict):
        defaults = scenarios.get("defaults")
        if isinstance(defaults, dict):
            _resolve_scenario_paths(config_dir, defaults)
        available = scenarios.get("available")
        if isinstance(available, dict):
            for name, scenario_value in list(available.items()):
                if isinstance(scenario_value, str):
                    available[name] = _resolve_path(config_dir, scenario_value)
                    continue
                if isinstance(scenario_value, dict):
                    _resolve_scenario_paths(config_dir, scenario_value)
    dataset = cfg.get("dataset")
    if isinstance(dataset, dict) and "local_path" in dataset:
        dataset["local_path"] = _resolve_path(config_dir, dataset.get("local_path"))
    datasets = cfg.get("datasets")
    if isinstance(datasets, dict):
        for _, ds in datasets.items():
            if isinstance(ds, dict) and "local_path" in ds:
                ds["local_path"] = _resolve_path(config_dir, ds.get("local_path"))

    analysis = cfg.get("analysis")
    if isinstance(analysis, dict):
        for key in ("artifacts_root", "exploration_output_dir"):
            if key in analysis:
                analysis[key] = _resolve_path(config_dir, analysis.get(key))

    _validate_factor_schema(cfg)
    _validate_order_variant_schema(cfg)
    return cfg


def get_model_config(cfg: dict[str, Any], model_key: str) -> dict[str, Any]:
    models = cfg.get("models", {})
    if model_key not in models:
        raise ValueError(
            f"Model key '{model_key}' not found in config models: {list(models.keys())}"
        )
    model_cfg = models[model_key]
    if not isinstance(model_cfg, dict):
        raise ValueError(f"Invalid model config for key '{model_key}'")
    return model_cfg


def resolve_scenario_bundle(
    cfg: dict[str, Any],
    scenario_set: str | None = None,
) -> dict[str, Any]:
    """Resolve selected scenario bundle from config.scenarios.available."""
    scenarios_cfg = cfg.get("scenarios")
    if not isinstance(scenarios_cfg, dict):
        raise ValueError("config.scenarios is required and must be a mapping.")
    available = scenarios_cfg.get("available")
    if not isinstance(available, dict) or not available:
        raise ValueError("config.scenarios.available is required and must be non-empty.")
    defaults_raw = scenarios_cfg.get("defaults", {})
    if defaults_raw is None:
        defaults_raw = {}
    if not isinstance(defaults_raw, dict):
        raise ValueError("config.scenarios.defaults must be a mapping when provided.")

    selected = scenario_set or scenarios_cfg.get("active")
    if not selected:
        raise ValueError(
            "config.scenarios.active is required when using config.scenarios.available."
        )
    entry = available.get(selected)
    if entry is None:
        raise ValueError(
            f"Scenario set '{selected}' not found in config.scenarios.available: "
            f"{list(available.keys())}"
        )

    if isinstance(entry, str):
        bundle = {"scenario_index_path": entry}
    elif isinstance(entry, dict):
        bundle = dict(entry)
    else:
        raise ValueError(f"Invalid scenario bundle for '{selected}': expected string or object.")

    merged = _deep_merge(dict(defaults_raw), bundle)
    merged["_scenario_key"] = selected
    return merged


def resolve_scenario_index_path(
    cfg: dict[str, Any],
    explicit_path: str | None = None,
    scenario_set: str | None = None,
) -> str:
    """Resolve scenario index path from CLI override or selected scenario bundle."""
    if explicit_path:
        return str(explicit_path)

    bundle = resolve_scenario_bundle(cfg, scenario_set=scenario_set)
    if bundle.get("scenario_index_path"):
        return str(bundle["scenario_index_path"])
    if bundle.get("scenarios_path"):
        # Backward compatibility for old configs not normalized by loader.
        return str(bundle["scenarios_path"])

    raise ValueError(
        "Scenario path not configured. Use --scenarios, or set "
        "config.scenarios.available.<scenario>.scenario_index_path."
    )


def resolve_scenarios_path(
    cfg: dict[str, Any],
    explicit_path: str | None = None,
    scenario_set: str | None = None,
) -> str:
    """Backward-compatible alias for resolve_scenario_index_path."""
    return resolve_scenario_index_path(
        cfg,
        explicit_path=explicit_path,
        scenario_set=scenario_set,
    )


def resolve_model_keys(
    cfg: dict[str, Any],
    explicit_model: str | None = None,
    scenario_set: str | None = None,
) -> list[str]:
    """Resolve which model keys should run for this invocation."""
    models = cfg.get("models", {})
    if not isinstance(models, dict) or not models:
        raise ValueError("No models found in config.")

    if explicit_model:
        _ = get_model_config(cfg, explicit_model)
        return [explicit_model]

    bundle = resolve_scenario_bundle(cfg, scenario_set=scenario_set)
    bundle_models = bundle.get("models")
    if isinstance(bundle_models, list) and bundle_models:
        for key in bundle_models:
            _ = get_model_config(cfg, str(key))
        return [str(key) for key in bundle_models]

    scenario_key = bundle.get("_scenario_key", "<unknown>")
    raise ValueError(
        f"Scenario bundle '{scenario_key}' must define non-empty list field 'models'."
    )


def resolve_conditions(
    cfg: dict[str, Any],
    explicit_conditions: list[str] | None = None,
    scenario_set: str | None = None,
) -> list[str]:
    def _resolve_condition_ref(value: Any) -> list[str] | None:
        if isinstance(value, list) and value:
            return [str(c) for c in value]
        if isinstance(value, str):
            sets = cfg.get("condition_sets")
            if not isinstance(sets, dict):
                raise ValueError(
                    f"Condition set '{value}' was referenced but config.condition_sets is missing."
                )
            resolved = sets.get(value)
            if not isinstance(resolved, list) or not resolved:
                raise ValueError(
                    f"Condition set '{value}' not found in config.condition_sets."
                )
            return [str(c) for c in resolved]
        return None

    if explicit_conditions:
        return explicit_conditions
    bundle = resolve_scenario_bundle(cfg, scenario_set=scenario_set)
    bundle_conditions = _resolve_condition_ref(bundle.get("conditions"))
    if bundle_conditions:
        return bundle_conditions
    scenario_key = bundle.get("_scenario_key", "<unknown>")
    raise ValueError(
        f"Scenario bundle '{scenario_key}' must define 'conditions' "
        "(list or condition_set ID)."
    )


def resolve_order_variants(
    cfg: dict[str, Any],
    explicit_variant: str | None = None,
    include_all: bool = False,
    scenario_set: str | None = None,
) -> list[dict[str, Any]]:
    """Resolve intervention-order variants for this run.

    Returns a list of dicts:
      [{"id": "canonical", "order": ["F1","F3","F4","F2"]}, ...]

    If not configured, falls back to a single canonical variant.
    """
    default = [{"id": "canonical", "order": ["F1", "F3", "F4", "F2"]}]
    bundle = resolve_scenario_bundle(cfg, scenario_set=scenario_set)
    bundle_os = bundle.get("order_sensitivity")
    cfg_os = bundle_os
    variants = default
    if isinstance(cfg_os, dict):
        catalog = cfg.get("order_variants")
        sets = cfg.get("order_variant_sets")

        def _resolve_catalog_variant(variant_id: str) -> dict[str, Any]:
            if not isinstance(catalog, dict):
                raise ValueError(
                    f"Order variant '{variant_id}' referenced but config.order_variants is missing."
                )
            entry = catalog.get(variant_id)
            if not isinstance(entry, dict):
                raise ValueError(
                    f"Order variant '{variant_id}' not found in config.order_variants."
                )
            order = entry.get("order")
            if not isinstance(order, list):
                raise ValueError(
                    f"Order variant '{variant_id}' must define list field 'order'."
                )
            return {"id": variant_id, "order": [str(x) for x in order]}

        def _validate(variant: dict[str, Any]) -> dict[str, Any]:
            order_norm = [str(x) for x in variant["order"]]
            if len(order_norm) != 4 or set(order_norm) != _VALID_FACTORS:
                raise ValueError(
                    f"Invalid order_sensitivity variant '{variant['id']}': expected a "
                    "permutation of [F1,F2,F3,F4]."
                )
            return {"id": str(variant["id"]), "order": order_norm}

        parsed: list[dict[str, Any]] = []
        set_ref = cfg_os.get("variant_set")
        if isinstance(set_ref, str):
            if not isinstance(sets, dict):
                raise ValueError(
                    f"Order variant set '{set_ref}' referenced but config.order_variant_sets is missing."
                )
            ids = sets.get(set_ref)
            if not isinstance(ids, list) or not ids:
                raise ValueError(
                    f"Order variant set '{set_ref}' not found in config.order_variant_sets."
                )
            parsed = [_validate(_resolve_catalog_variant(str(v))) for v in ids]
        else:
            raw_variants = cfg_os.get("variants")
            if isinstance(raw_variants, list) and raw_variants:
                for item in raw_variants:
                    if isinstance(item, str):
                        parsed.append(_validate(_resolve_catalog_variant(item)))
                        continue
                    if isinstance(item, dict):
                        variant_id = str(item.get("id", "")).strip()
                        order = item.get("order")
                        if variant_id and isinstance(order, list):
                            parsed.append(_validate({"id": variant_id, "order": order}))
                # ignore invalid rows for backward compatibility
        if parsed:
            variants = parsed

    if explicit_variant:
        selected = [v for v in variants if v["id"] == explicit_variant]
        if not selected:
            valid = [v["id"] for v in variants]
            raise ValueError(
                f"Order variant '{explicit_variant}' not found. Available variants: {valid}"
            )
        return selected

    if include_all:
        return variants
    return [variants[0]]


def resolve_output_root(
    cfg: dict[str, Any],
    explicit_output: str | None = None,
    scenario_set: str | None = None,
) -> str:
    if explicit_output:
        return explicit_output
    bundle = resolve_scenario_bundle(cfg, scenario_set=scenario_set)
    if bundle.get("output_root"):
        return str(bundle["output_root"])
    scenario_key = bundle.get("_scenario_key", "<unknown>")
    raise ValueError(
        f"Scenario bundle '{scenario_key}' must define 'output_root' "
        "(or pass --output)."
    )


def _resolve_dataset_cfg(
    cfg: dict[str, Any],
    scenario_set: str | None = None,
) -> dict[str, Any] | None:
    bundle = resolve_scenario_bundle(cfg, scenario_set=scenario_set)
    dataset_ref = bundle.get("dataset")
    if isinstance(dataset_ref, dict):
        return dataset_ref
    if isinstance(dataset_ref, str):
        datasets = cfg.get("datasets")
        if not isinstance(datasets, dict):
            raise ValueError(
                "Scenario bundle references dataset key but config.datasets is missing."
            )
        dataset_cfg = datasets.get(dataset_ref)
        if not isinstance(dataset_cfg, dict):
            raise ValueError(
                f"Dataset key '{dataset_ref}' not found in config.datasets."
            )
        return dataset_cfg

    scenario_key = bundle.get("_scenario_key", "<unknown>")
    raise ValueError(
        f"Scenario bundle '{scenario_key}' must define 'dataset' "
        "(dataset config object or dataset catalog key)."
    )


def ensure_dataset_available(
    cfg: dict[str, Any],
    explicit_path: str | None = None,
    scenario_set: str | None = None,
) -> str:
    """Return an existing dataset path, downloading if configured and needed."""
    if explicit_path:
        p = Path(explicit_path)
        if not p.exists():
            raise FileNotFoundError(f"Dataset not found at explicit --data path: {p}")
        return str(p)

    dataset_cfg = _resolve_dataset_cfg(cfg, scenario_set=scenario_set)
    if isinstance(dataset_cfg, dict):
        local = dataset_cfg.get("local_path")
        if not local:
            raise ValueError("dataset.local_path is required for selected scenario bundle")
        p = Path(str(local))
        if p.exists():
            return str(p)
        auto_download = bool(dataset_cfg.get("auto_download", False))
        url = dataset_cfg.get("url")
        if auto_download and url:
            p.parent.mkdir(parents=True, exist_ok=True)
            urlretrieve(str(url), str(p))
            return str(p)
        raise FileNotFoundError(
            f"Dataset missing at {p}. Set dataset.auto_download=true with dataset.url, "
            "or pass --data with an existing file."
        )

    raise ValueError(
        "Dataset path not configured for selected scenario bundle. "
        "Use bundle dataset.local_path or pass --data."
    )


def resolve_analysis_paths(
    cfg: dict[str, Any],
    scenario_set: str | None = None,
) -> dict[str, str]:
    """Resolve analysis paths from selected scenario bundle analysis block."""
    bundle = resolve_scenario_bundle(cfg, scenario_set=scenario_set)
    analysis = bundle.get("analysis")
    if not isinstance(analysis, dict):
        return {}
    out: dict[str, str] = {}
    if analysis.get("artifacts_root"):
        out["artifacts_root"] = str(analysis["artifacts_root"])
    if analysis.get("exploration_output_dir"):
        out["exploration_output_dir"] = str(analysis["exploration_output_dir"])
    return out
