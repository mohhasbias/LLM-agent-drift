#!/usr/bin/env python3
"""Recompute all manuscript metrics from the stored episode_records.jsonl files.

Intentionally stdlib-only — runs without installing any packages.

Covers the expanded evaluation (Tables 1 and 2):
  - Table 1: reference-free task-success rates (5 models × 3 conditions, eval split)
  - Table 2: normalization ablation (5 models, static with vs. without normalization)
  - Per-drift breakdown (schema / policy / toolset, static condition)
  - Wilson 95% confidence intervals

Usage (run from the package root):
    python3 scripts/recompute_metrics.py
    python3 scripts/recompute_metrics.py --output metrics_recomputed.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# ---------------------------------------------------------------------------
# Package directory layout
# ---------------------------------------------------------------------------

MODELS = [
    "or_gpt4o_mini",
    "or_gpt4o",
    "or_qwen3_32b",
    "or_kimi_k2",
    "or_deepseek_v3",
]

MODEL_LABELS = {
    "or_gpt4o_mini":  "GPT-4o mini",
    "or_gpt4o":       "GPT-4o",
    "or_qwen3_32b":   "Qwen3-32B",
    "or_kimi_k2":     "Kimi K2",
    "or_deepseek_v3": "DeepSeek V3",
}

# (results sub-dir, mode sub-dir)
CONDITION_PATH = {
    "static": ("results/expanded/static_run", "static"),
    "replay": ("results/expanded/replay_run", "learning"),
    "rules":  ("results/expanded/rules_run",  "learning"),
}

NONORM_PATH_TMPL = "results/normalization_ablation/static_nonorm_run/{model}/static"

DRIFT_TYPES = ["schema", "policy", "toolset"]

# Manuscript claims for exact-match verification
TABLE1_CLAIMS = {
    ("or_gpt4o_mini", "static"):  (107, 270, 0.396),
    ("or_gpt4o",      "static"):  ( 77, 270, 0.285),
    ("or_qwen3_32b",  "static"):  ( 45, 270, 0.167),
    ("or_kimi_k2",    "static"):  ( 35, 270, 0.130),
    ("or_deepseek_v3","static"):  ( 29, 270, 0.107),
    ("or_gpt4o_mini", "replay"):  ( 61, 270, 0.226),
    ("or_gpt4o",      "replay"):  ( 27, 270, 0.100),
    ("or_qwen3_32b",  "replay"):  ( 27, 270, 0.100),
    ("or_kimi_k2",    "replay"):  ( 27, 270, 0.100),
    ("or_deepseek_v3","replay"):  ( 27, 270, 0.100),
    ("or_gpt4o_mini", "rules"):   ( 27, 270, 0.100),
    ("or_gpt4o",      "rules"):   ( 27, 270, 0.100),
    ("or_qwen3_32b",  "rules"):   ( 27, 270, 0.100),
    ("or_kimi_k2",    "rules"):   ( 27, 270, 0.100),
    ("or_deepseek_v3","rules"):   ( 27, 270, 0.100),
}

TABLE2_CLAIMS = {
    "or_gpt4o_mini":  (270, 0.396, 236, 0.322),
    "or_gpt4o":       (270, 0.285, 270, 0.204),
    "or_qwen3_32b":   (270, 0.167, 270, 0.211),
    "or_kimi_k2":     (270, 0.130, 270, 0.130),
    "or_deepseek_v3": (270, 0.107, 270, 0.104),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def eval_only(records: list[dict]) -> list[dict]:
    return [r for r in records if r.get("split") == "eval"]


def agg(records: list[dict]) -> tuple[int, int]:
    ev = eval_only(records)
    k = sum(1 for r in ev if r.get("success"))
    return k, len(ev)


def agg_by_drift(records: list[dict], drift_type: str) -> tuple[int, int]:
    ev = [r for r in eval_only(records) if r.get("drift_type") == drift_type]
    k = sum(1 for r in ev if r.get("success"))
    return k, len(ev)


def rate(k: int, n: int) -> float:
    return round(k / n, 4) if n > 0 else float("nan")


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return round(max(0.0, center - margin), 4), round(min(1.0, center + margin), 4)


def check(label: str, actual_k: int, actual_n: int,
          claimed_k: int | None, claimed_n: int, claimed_rate: float,
          tol: float = 0.001) -> bool:
    r = rate(actual_k, actual_n)
    ok = (actual_n == claimed_n
          and (claimed_k is None or actual_k == claimed_k)
          and abs(r - claimed_rate) <= tol)
    status = "PASS" if ok else "FAIL"
    flag = "" if ok else "  <---"
    claimed_k_str = str(claimed_k) if claimed_k is not None else "?"
    print(
        f"  {status}  {label:<44}  "
        f"actual={actual_k}/{actual_n}={r:.4f}  "
        f"claimed={claimed_k_str}/{claimed_n}={claimed_rate:.4f}{flag}"
    )
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recompute manuscript metrics from Zenodo episode records."
    )
    parser.add_argument(
        "--root", type=Path, default=Path("."),
        help="Root of the extracted package (default: current directory).",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Write full metrics JSON to this path (optional).",
    )
    args = parser.parse_args()
    root = args.root.resolve()

    total_checks = 0
    total_pass = 0
    has_missing = False
    out: dict = {}

    # -----------------------------------------------------------------------
    # Table 1 — 5 models × 3 conditions
    # -----------------------------------------------------------------------
    print("\n=== TABLE 1: Reference-free task-success rates (eval split) ===")
    print(f"  {'Model':<16} {'Cond':<8} {'k':>5} {'n':>5} {'rate':>6}  95% Wilson CI")
    table1_rows: list[dict] = []

    for model in MODELS:
        for condition in ["static", "replay", "rules"]:
            run_dir, mode = CONDITION_PATH[condition]
            ep_path = root / run_dir / model / mode / "episode_records.jsonl"
            if not ep_path.exists():
                print(f"  MISS  {model}/{condition}: {ep_path}")
                has_missing = True
                total_checks += 1
                continue
            recs = load_jsonl(ep_path)
            k, n = agg(recs)
            r = rate(k, n)
            lo, hi = wilson_ci(k, n)
            print(f"  {MODEL_LABELS[model]:<16} {condition:<8} {k:>5} {n:>5} {r:>6.4f}  [{lo:.4f}, {hi:.4f}]")
            drift_breakdown = {}
            for dt in DRIFT_TYPES:
                dk, dn = agg_by_drift(recs, dt)
                drift_breakdown[dt] = {"k": dk, "n": dn, "rate": rate(dk, dn)}
            table1_rows.append({
                "model": model, "condition": condition,
                "k": k, "n": n, "rate": r,
                "wilson_95_low": lo, "wilson_95_high": hi,
                "drift_breakdown": drift_breakdown,
            })
    out["table1"] = table1_rows

    # -----------------------------------------------------------------------
    # Table 1 — exact-match verification
    # -----------------------------------------------------------------------
    print("\n=== TABLE 1: Exact-match verification vs. manuscript ===")
    for model in MODELS:
        for condition in ["static", "replay", "rules"]:
            key = (model, condition)
            if key not in TABLE1_CLAIMS:
                continue
            claimed_k, claimed_n, claimed_rate = TABLE1_CLAIMS[key]
            run_dir, mode = CONDITION_PATH[condition]
            ep_path = root / run_dir / model / mode / "episode_records.jsonl"
            if not ep_path.exists():
                total_checks += 1
                continue
            recs = load_jsonl(ep_path)
            actual_k, actual_n = agg(recs)
            ok = check(
                f"{model}/{condition}",
                actual_k, actual_n, claimed_k, claimed_n, claimed_rate,
            )
            total_checks += 1
            total_pass += ok

    # -----------------------------------------------------------------------
    # Table 2 — normalization ablation
    # -----------------------------------------------------------------------
    print("\n=== TABLE 2: Normalization ablation (static condition) ===")
    print(f"  {'Model':<16}  {'With norm':>14}  {'Without norm':>14}")
    table2_rows: list[dict] = []
    for model in MODELS:
        ep_with = root / CONDITION_PATH["static"][0] / model / CONDITION_PATH["static"][1] / "episode_records.jsonl"
        ep_without = root / NONORM_PATH_TMPL.format(model=model) / "episode_records.jsonl"
        row: dict = {"model": model}

        if ep_with.exists():
            k, n = agg(load_jsonl(ep_with))
            row.update(norm_k=k, norm_n=n, norm_rate=rate(k, n))
            row["norm_wilson_low"], row["norm_wilson_high"] = wilson_ci(k, n)
        else:
            print(f"  MISS  {model} (with norm): {ep_with}")
            has_missing = True

        if ep_without.exists():
            k, n = agg(load_jsonl(ep_without))
            row.update(nonorm_k=k, nonorm_n=n, nonorm_rate=rate(k, n))
            row["nonorm_wilson_low"], row["nonorm_wilson_high"] = wilson_ci(k, n)
        else:
            print(f"  MISS  {model} (no norm): {ep_without}")
            has_missing = True

        if "norm_rate" in row and "nonorm_rate" in row:
            print(
                f"  {MODEL_LABELS[model]:<16}  "
                f"{row['norm_k']}/{row['norm_n']}={row['norm_rate']:.4f}  "
                f"    {row['nonorm_k']}/{row['nonorm_n']}={row['nonorm_rate']:.4f}"
            )
        table2_rows.append(row)
    out["table2"] = table2_rows

    # Table 2 exact-match
    print("\n=== TABLE 2: Exact-match verification vs. manuscript ===")
    for model in MODELS:
        if model not in TABLE2_CLAIMS:
            continue
        cn_norm, cr_norm, cn_nonorm, cr_nonorm = TABLE2_CLAIMS[model]
        ep_with = root / CONDITION_PATH["static"][0] / model / CONDITION_PATH["static"][1] / "episode_records.jsonl"
        ep_without = root / NONORM_PATH_TMPL.format(model=model) / "episode_records.jsonl"

        if ep_with.exists():
            k, n = agg(load_jsonl(ep_with))
            ok = check(f"{model}/static (norm)", k, n, None, cn_norm, cr_norm)
            total_checks += 1; total_pass += ok

        if ep_without.exists():
            k, n = agg(load_jsonl(ep_without))
            ok = check(f"{model}/static (no-norm)", k, n, None, cn_nonorm, cr_nonorm)
            total_checks += 1; total_pass += ok

    # -----------------------------------------------------------------------
    # Per-drift breakdown (static condition)
    # -----------------------------------------------------------------------
    print("\n=== PER-DRIFT BREAKDOWN (static condition, eval split) ===")
    print(f"  {'Model':<16}  {'schema':>14}  {'policy':>14}  {'toolset':>14}")
    for row in table1_rows:
        if row["condition"] != "static":
            continue
        bd = row.get("drift_breakdown", {})
        cells = []
        for dt in DRIFT_TYPES:
            if dt in bd:
                cells.append(f"{bd[dt]['k']}/{bd[dt]['n']}={bd[dt]['rate']:.4f}")
            else:
                cells.append("MISS")
        print(f"  {MODEL_LABELS[row['model']]:<16}  {'  '.join(cells)}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Verification: {total_pass}/{total_checks} exact-match checks passed")
    if has_missing:
        print("WARNING: some episode_records.jsonl files were missing.")
    elif total_pass == total_checks:
        print("ALL CHECKS PASSED — stored results match manuscript claims.")
    else:
        print(f"WARNING: {total_checks - total_pass} check(s) FAILED.")

    if args.output:
        out_path = args.output if args.output.is_absolute() else root / args.output
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nMetrics written to: {out_path}")

    return 0 if (not has_missing and total_pass == total_checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
