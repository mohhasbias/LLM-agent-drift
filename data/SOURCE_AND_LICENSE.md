# Source and License — Learning-Agent Online Adaptation Dataset

- Primary upstream dataset: tau2-bench (retail domain, test split)
  - Upstream repository: https://github.com/sierra-research/tau2-bench
  - Upstream license: MIT
  - Archived local snapshot used for conversion:
    - `experiments/archive/01-tool-use-brittleness-ablation/data/tau2bench/raw/tau2-bench/data/tau2/domains/retail/tasks.json`
    - `experiments/archive/01-tool-use-brittleness-ablation/data/tau2bench/raw/tau2-bench/data/tau2/domains/retail/split_tasks.json`
    - `experiments/archive/01-tool-use-brittleness-ablation/data/tau2bench/raw/tau2-bench/LICENSE`
    - `experiments/archive/01-tool-use-brittleness-ablation/data/tau2bench/raw/tau2-bench/SOURCE_AND_LICENSE.md`

- Active converted replay-format file used by this experiment:
  - `experiments/active/learning-agent-online-adaptation/data/tau2bench/converted/tau2bench-retail-test-adapted.jsonl`
  - Conversion script lineage: copied/adapted from archived experiment conversion utilities
  - Provenance marker: each JSONL row includes `english_env_info` containing upstream repo and license text.

- Scenario index used for pilot/holdout gates:
  - `experiments/active/learning-agent-online-adaptation/data/pilot_n12_index.json`

- Data availability statement basis:
  - This experiment distributes only adapted task records and indexes needed for replication in this repository.
  - Full upstream repository history is not redistributed here; users should consult the official upstream repository for complete source distribution, updates, and attribution requirements.
