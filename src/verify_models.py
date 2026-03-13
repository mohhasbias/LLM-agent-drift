"""Model endpoint verifier for configured experiment models.

This script performs a minimal chat completion call for each selected model key
and reports whether the model endpoint appears usable with current credentials.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Allow `python src/verify_models.py ...` without pre-setting PYTHONPATH.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.experiment_config import get_model_config, load_experiment_config, resolve_model_keys
from src.models import GroqClient, NIMClient, OpenRouterClient


@dataclass
class VerificationResult:
    model_key: str
    provider: str
    model_name: str
    category: str | None
    base_url: str | None
    usable: bool
    status: str
    latency_ms: int | None
    input_tokens: int | None
    output_tokens: int | None
    error: str | None


def _build_client(model_key: str, model_cfg: dict[str, Any]):
    provider = str(model_cfg.get("provider", model_key))
    model_name = model_cfg.get("model_name")
    rpm = model_cfg.get("rpm")
    base_url = model_cfg.get("base_url")

    if provider in ("groq",):
        return GroqClient(
            model_name=model_name,
            rpm=int(rpm) if rpm is not None else None,
            base_url=str(base_url) if base_url else None,
        )
    if provider in ("nvidia-nim", "nim"):
        return NIMClient(
            model_name=model_name,
            rpm=int(rpm) if rpm is not None else None,
            base_url=str(base_url) if base_url else None,
        )
    if provider in ("openrouter",):
        return OpenRouterClient(
            model_name=model_name,
            rpm=int(rpm) if rpm is not None else None,
            base_url=str(base_url) if base_url else None,
        )
    raise ValueError(
        f"Unknown provider '{provider}' for model '{model_key}'. "
        "Supported providers: groq, nvidia-nim, openrouter."
    )


def verify_model(
    model_key: str,
    model_cfg: dict[str, Any],
    prompt: str,
) -> VerificationResult:
    provider = str(model_cfg.get("provider", model_key))
    model_name = str(model_cfg.get("model_name", ""))
    category = model_cfg.get("category")

    try:
        client = _build_client(model_key, model_cfg)
    except Exception as e:
        return VerificationResult(
            model_key=model_key,
            provider=provider,
            model_name=model_name,
            category=str(category) if category is not None else None,
            base_url=str(model_cfg.get("base_url")) if model_cfg.get("base_url") else None,
            usable=False,
            status="client_error",
            latency_ms=None,
            input_tokens=None,
            output_tokens=None,
            error=str(e),
        )

    started = time.perf_counter()
    try:
        response = client.chat_complete(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            tools=None,
            temperature=0.0,
        )
        latency_ms = int((time.perf_counter() - started) * 1000)

        input_tokens = getattr(getattr(response, "usage", None), "prompt_tokens", None)
        output_tokens = getattr(getattr(response, "usage", None), "completion_tokens", None)

        return VerificationResult(
            model_key=model_key,
            provider=provider,
            model_name=client.model,
            category=str(category) if category is not None else None,
            base_url=client.base_url,
            usable=True,
            status="ok",
            latency_ms=latency_ms,
            input_tokens=int(input_tokens) if input_tokens is not None else None,
            output_tokens=int(output_tokens) if output_tokens is not None else None,
            error=None,
        )
    except Exception as e:
        latency_ms = int((time.perf_counter() - started) * 1000)
        return VerificationResult(
            model_key=model_key,
            provider=provider,
            model_name=model_name,
            category=str(category) if category is not None else None,
            base_url=getattr(client, "base_url", None),
            usable=False,
            status="request_error",
            latency_ms=latency_ms,
            input_tokens=None,
            output_tokens=None,
            error=str(e),
        )


def verify_models(
    cfg: dict[str, Any],
    model_keys: list[str],
    prompt: str,
) -> list[VerificationResult]:
    results: list[VerificationResult] = []
    for key in model_keys:
        model_cfg = get_model_config(cfg, key)
        results.append(verify_model(key, model_cfg, prompt))
    return results


def _default_report_path() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    return str(Path("logs") / "metadata" / f"model-endpoint-verification-{ts}.json")


def _print_summary(results: list[VerificationResult]) -> None:
    print("\nModel Endpoint Verification")
    print("-" * 90)
    print(f"{'model_key':<34} {'provider':<12} {'usable':<7} {'status':<14} {'latency_ms':<10}")
    print("-" * 90)
    for r in results:
        print(
            f"{r.model_key:<34} {r.provider:<12} {str(r.usable):<7} "
            f"{r.status:<14} {str(r.latency_ms):<10}"
        )
    ok = sum(1 for r in results if r.usable)
    print("-" * 90)
    print(f"usable: {ok}/{len(results)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify configured model endpoints are usable")
    parser.add_argument(
        "--config",
        default="experiment.yaml",
        help="Path to experiment config file (.yaml/.yml/.json).",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help=(
            "Model key from config. Repeat to verify multiple keys. "
            "If omitted, verifies all model keys from selected scenario-set."
        ),
    )
    parser.add_argument(
        "--scenario-set",
        default=None,
        help="Scenario set key from config.scenarios.available.",
    )
    parser.add_argument(
        "--prompt",
        default="Reply with exactly: OK",
        help="Prompt used for the minimal verification request.",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Path to JSON report output. Default: logs/metadata/model-endpoint-verification-<timestamp>.json",
    )
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Exit 0 even if some models fail verification.",
    )
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)

    if args.model:
        model_keys = list(args.model)
        for key in model_keys:
            _ = get_model_config(cfg, key)
    else:
        model_keys = resolve_model_keys(cfg, explicit_model=None, scenario_set=args.scenario_set)

    results = verify_models(cfg, model_keys, args.prompt)
    _print_summary(results)

    report_path = args.report or _default_report_path()
    report_abs = Path(report_path).resolve()
    report_abs.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "config": str(Path(args.config).resolve()),
        "scenario_set": args.scenario_set,
        "results": [asdict(r) for r in results],
    }
    report_abs.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"report: {report_abs}")

    failed = [r for r in results if not r.usable]
    if failed and not args.allow_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
