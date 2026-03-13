# Benchmarking Reference-Free LLM Agent Robustness Under Schema, Policy, and Toolset Drift

[![DOI](https://zenodo.org/badge/1180487370.svg)](https://doi.org/10.5281/zenodo.18995497)

**Data and Code Artifact**

> **Status**: The associated manuscript is currently under peer review. This artifact is
> released to enable independent verification of the reported results prior to publication.

This repository provides all episode-level evaluation records, source code, and analysis
scripts needed to reproduce the numerical results of the **expanded evaluation** reported in
the manuscript: 5 models × 3 conditions × 40 scenarios × 3 drift families.

---

## Package layout

```
data/
  tau2bench/converted/
    tau2bench-retail-test-adapted-observed.jsonl   ← 111-scenario input dataset
  expanded_n40_index.json   ← 40 scenario IDs (10 train / 30 eval)
  SOURCE_AND_LICENSE.md     ← provenance and license for upstream data

results/
  expanded/                 ← TABLE 1: 5 models × 3 conditions
    static_run/{model}/static/
      episode_records.jsonl   ← one JSON object per episode
      summary.json
    replay_run/{model}/learning/
      episode_records.jsonl
      summary.json
    rules_run/{model}/learning/
      episode_records.jsonl
      summary.json
  normalization_ablation/   ← TABLE 2: static without drift-inversion layer
    static_nonorm_run/{model}/static/
      episode_records.jsonl
      summary.json

src/                        ← full Python evaluation pipeline
  run_task_success_evaluation.py   ← main entry point
  drift_injector.py
  learning_memory.py
  experiment_config.py
  data_loader.py
  output_formatter.py
  progress_tracker.py
  structures.py
  verify_models.py
  models/
    openrouter_client.py
    base_client.py
    rate_limiter.py

scripts/
  recompute_metrics.py        ← stdlib-only offline verification (start here)
  run_expanded_evaluation.sh  ← re-run the full experiment from scratch

tests/
  test_drift_injector.py
  test_learning_memory.py
  test_models.py
  test_run_task_success_evaluation.py

experiment.yaml    ← experiment configuration (models, scenario sets, seed)
pyproject.toml     ← Python dependency spec
uv.lock            ← locked dependency versions
SHA256SUMS.txt     ← file integrity checksums
CITATION.cff
zenodo-metadata.json
licenses/tau2-bench-LICENSE-MIT.txt
```

---

## Models evaluated

| Key | Provider model |
|---|---|
| `or_gpt4o_mini` | `openai/gpt-4o-mini` via OpenRouter |
| `or_gpt4o` | `openai/gpt-4o` via OpenRouter |
| `or_qwen3_32b` | `qwen/qwen3-32b` via OpenRouter |
| `or_kimi_k2` | `moonshotai/kimi-k2` via OpenRouter |
| `or_deepseek_v3` | `deepseek/deepseek-chat` via OpenRouter |

---

## Experimental design

**Drift families** (injected at inference time; a normalization layer inverts each before
executing against the upstream tau2-bench retail environment):

| Family | Mechanism |
|---|---|
| Schema | Argument key renamed: `order_id` → `order_id_v2` |
| Policy | Policy-confirmation tool call prepended as a required first step |
| Toolset | Tool name remapped: `cancel_order` → `cancel_order__new` |

**Evaluation conditions:**

| Condition | Controller stack |
|---|---|
| `static` | LLM call only (Table 1 primary baseline) |
| `replay` | LLM + first-step memory cue (blind replay) |
| `rules` | LLM + deterministic drift-specific repair rules |

**Trial counts (eval split only):**
- 30 eval scenarios × 3 episodes × 3 drift families = **270 trials per model-condition cell**
- 5 models × 3 conditions = 15 cells → **4,050 total episodes** (Table 1)
- 5 models × 1 condition (no-norm static) → **1,350 episodes** (Table 2)

**Success criterion:** binary — exact database-state match against ground truth (no partial credit)

---

## Reproducing results without an API key

### Step 1 — Verify file integrity

```bash
sha256sum -c SHA256SUMS.txt
```

### Step 2 — Recompute all manuscript metrics from stored episode records

```bash
python3 scripts/recompute_metrics.py --output recomputed_metrics.json
```

Requires Python 3.8+ and **no external packages**. Prints Tables 1 and 2 with Wilson 95%
CIs and per-drift breakdowns. Compare with manuscript-reported values.

### Step 3 — Run the unit tests (optional — requires `uv`)

```bash
uv sync --group dev
uv run pytest tests/ -v
```

---

## Re-running the experiment from scratch (requires API access)

### Prerequisites

- Python 3.12+
- `uv` package manager — https://github.com/astral-sh/uv
- OpenRouter API key with access to the 5 model endpoints listed above

### Install dependencies

```bash
uv sync
```

### Provide API credentials

```bash
export OPENROUTER_API_KEY=your_key_here
```

### Run the full expanded evaluation

```bash
bash scripts/run_expanded_evaluation.sh
```

Runs all 5 models across 3 conditions (static, blind-replay, rules) plus the
normalization ablation. Results land in `results/raw_task_success_expanded_2026-03-10/`
and `results/raw_task_success_nonorm_2026-03-10/` by default.
Expected wall-clock time: 8–12 hours.

### Run a single model/condition (quick check)

```bash
uv run python src/run_task_success_evaluation.py \
  --config experiment.yaml \
  --scenario-set expanded_n40 \
  --model or_gpt4o_mini \
  --mode static \
  --episodes 3 \
  --train-max-scenarios 10 \
  --eval-max-scenarios 30 \
  --seed 20260310 \
  --output results/rerun/static_run
```

---

## Reported results (Table 1 — static condition, eval split)

| Model | k | n | rate | 95% Wilson CI |
|---|---|---|---|---|
| GPT-4o mini | 107 | 270 | 0.396 | [0.339, 0.456] |
| GPT-4o | 77 | 270 | 0.285 | [0.232, 0.345] |
| Qwen3-32B | 45 | 270 | 0.167 | [0.125, 0.218] |
| Kimi K2 | 35 | 270 | 0.130 | [0.093, 0.177] |
| DeepSeek V3 | 29 | 270 | 0.107 | [0.074, 0.151] |

---

## Citation

If you use this artifact, please cite it:

```bibtex
@misc{assidiqi2026benchmarking,
  title  = {Benchmarking Reference-Free {LLM} Agent Robustness Under Schema, Policy, and Toolset Drift},
  author = {Assidiqi, Mohammad Hasbi and Alghazzawi, Daniyal and Alarifi, Suaad and Cheng, Li},
  year   = {2026},
  note   = {Manuscript under peer review. Dataset DOI: 10.5281/zenodo.18995498}
}
```

---

## License

- **Code and evaluation results** (this artifact): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Upstream tau2-bench dataset**: MIT License — see `licenses/tau2-bench-LICENSE-MIT.txt`
