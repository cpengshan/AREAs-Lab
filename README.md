# AREA — AI-Assisted Requirement Elicitation Agent

AREA is a research framework for benchmarking how LLM agents transform underspecified user requests into structured task requirement specifications. Agents interact with a simulated user and/or real dataset samples across four strategies, and are evaluated with an atomic-unit LLM-based scoring pipeline.

---

## Repository Structure

```
AREA/
├── agent/                              # Agent layer (main entry point)
│   ├── run_agent.py                    # CLI entry point for all strategies
│   ├── zero_shot_agent.py              # Zero-shot strategy
│   ├── user_interaction_agent.py       # User-interaction strategy
│   ├── zero_shot_variants_agent.py     # Data-interaction strategy (DataInteractionAgent)
│   ├── hybrid_agent.py                 # Hybrid strategy
│   ├── agent_state.py                  # LangGraph AgentState TypedDict
│   ├── prompts/                        # Jinja2 prompt templates
│   │   ├── user/                       # user_interaction prompts
│   │   └── data/                       # data_interaction prompts
│   ├── scripts/                        # SLURM batch submission scripts
│   │   ├── run_zero_shot_agent.sh
│   │   ├── run_user_interaction_agent.sh
│   │   ├── run_data_interaction_agent.sh
│   │   └── run_hybrid_agent.sh
│   ├── logs/                           # Per-strategy run logs (gitignored)
│   └── results/                        # Per-dataset experiment outputs (gitignored)
│       └── <dataset_name>/
│           └── <strategy>/Experiment<N>/
│
├── AREAEnv/                            # Benchmark environment package
│   ├── area_env/
│   │   ├── config.py                   # AREAEnvConfig dataclass
│   │   ├── configs/
│   │   │   └── default.yaml            # Default model & env config (edit here)
│   │   ├── env/
│   │   │   ├── area_env.py             # Core gym-style environment
│   │   │   ├── actions.py              # Action constructors & validation
│   │   │   └── state.py                # EpisodeState dataclass
│   │   ├── dataset/
│   │   │   ├── schema.py               # PersonaInfo, TaskInstance dataclasses
│   │   │   └── loader.py               # load_dataset() from synthesized JSON
│   │   ├── users/
│   │   │   ├── mimic_user.py        # Simulated user with communication habits
│   │   │   └── prompts/                # User simulator Jinja2 templates
│   │   ├── evaluator/
│   │   │   ├── atomic_evaluator.py     # 3-stage LLM evaluation pipeline
│   │   │   ├── metrics.py              # F1/precision/recall computation
│   │   │   ├── cache/                  # Cached gold decompositions (gitignored)
│   │   │   └── prompt/                 # LLM judge Jinja2 templates
│   │   └── utils/
│   │       ├── llm.py                  # Unified LLM call interface (retry + cost)
│   │       ├── jinja_utils.py          # render_template() helper
│   │       └── json_utils.py           # parse_json_output() helper
│   ├── examples/
│   │   ├── minimal_example.py          # End-to-end env loop (no API key needed)
│   │   └── policies.py                 # Reference policies (ZeroShot/UserOnly/DataOnly)
│   ├── scripts/
│   │   ├── run_experiment.py           # Experiment runner CLI
│   │   ├── run_evaluation.py           # Standalone re-evaluator
│   │   └── analyze_requirement_categories.py  # Category-level score analysis
│   └── data/                           # AREAEnv local data (gitignored)
│       ├── data_synthesized/           # synthesized_output.json per dataset
│       └── data_sampled/               # Sampled data per dataset
│
├── visualization-ui/                   # React/TypeScript/Vite results viewer
├── result_visualization.ipynb          # Notebook for results analysis
├── requirements.txt                    # Python dependencies
├── pyproject.toml                      # Python project / dependency config
└── .env                                # API keys (gitignored)
```

---

## Architecture

### Environment (`AREAEnv`)

`AREAEnv` is a gym-style benchmark environment decoupled from any specific agent. It exposes a standard `reset() → step() → get_trajectory_log()` interface.

**Action space:**

| Action | Description |
|--------|-------------|
| `ask_user(question)` | Pose a clarification question to the MIMIC user |
| `inspect_data(n_samples, split)` | Sample rows from the task dataset |
| `finish(final_requirement)` | Submit the final requirement and trigger evaluation |

**Episode flow:**
1. `env.reset(task, user)` — initialize with a `TaskInstance` and `MimicUser`
2. Agent submits actions via `env.step(action)` → `(observation, reward, done, info)`
3. On `finish()`, `AtomicEvaluator` runs automatically
4. `env.get_trajectory_log()` returns the full episode log including eval scores

### Agent Strategies

All agents are LangGraph `StateGraph` policies over `AgentState`, sharing the same interface: `agent.run(env, task, [user]) → log`.

#### Zero-shot (`zero_shot_agent.py`)
Single LLM call on the elevator pitch. No interaction.
```
[area_agent] → END
```

#### User Interaction (`user_interaction_agent.py`)
Iterative clarification with the MIMIC user, then final synthesis.
```
Phase 0: zero-shot draft (in memory)
Phase 1: ask_user loop (≤ max_turns) ← MIMIC user feedback
Phase 2: synthesize final → finish(requirement)
```

#### Data Interaction (`zero_shot_variants_agent.py` — `DataInteractionAgent`)
Zero-shot draft augmented with real data samples and structured reasoning about modifications.
```
Phase 0: inspect_data (non_defining_instances split)
Phase 1: one-shot rewrite with samples + modification trace → finish(requirement)
```

#### Hybrid (`hybrid_agent.py`)
Combines user clarification and data inspection with a router that decides which action to take at each turn.
```
Phase 0: zero-shot draft (or seed from prior experiment)
Phase 1: router loop (≤ max_iterations):
           → ask_user  OR  inspect_data  OR  finish
```

### Evaluation (`AtomicEvaluator`)

Four-stage LLM pipeline comparing the agent's final requirement against the gold requirement:

1. **Decompose gold** → list of atomic requirement units (cached per task)
2. **Decompose predicted** → list of atomic requirement units
3. **Classify** → assign each unit to a category (`overall`, `user_specified`, `data_specified`)
4. **Compare** → matched pairs, missing units, hallucinated units

Scores from TP/FP/FN counts: **precision**, **recall**, **F1** — reported overall and per category.

### MIMIC User

`MimicUser` simulates user feedback using Jinja2 templates with a configurable `communication_habit` (`passive` | `neutral` | `active`).

---

## Data Setup

Place the data folder inside `AREAEnv/data/`. Two subdirectories are required:

```
AREAEnv/data/
├── data_synthesized/          # Synthesized personas and gold task requirements
│   └── <hf_owner>/
│       └── <dataset_name>/
│           ├── synthesized_output.json   # Personas + gold requirements (loaded by env)
│           ├── data_analysis.json
│           └── ground_truth_decompose.json  # Cached gold decompositions (optional)
└── data_sampled/              # Sampled dataset instances for inspect_data actions
    └── <hf_owner>/
        └── <dataset_name>/
            └── data_sampled.json
```

The directory structure mirrors the HuggingFace dataset ID. For example, dataset `alexfabbri/multi_news` maps to:

```
AREAEnv/data/data_synthesized/alexfabbri/multi_news/synthesized_output.json
AREAEnv/data/data_sampled/alexfabbri/multi_news/data_sampled.json
```

Pass the dataset ID to `--dataset` and the env resolves the paths automatically. To use a different file name (e.g. a versioned snapshot), set `synthesized_output_file` or `data_sampled_file` in your YAML config.

---

## Installation

**Requirements:** Python 3.10+

```bash
pip install -r requirements.txt
# or via uv:
uv sync
```

**API keys** — create a `.env` file in the repo root:
```
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
ANTHROPIC_API_KEY=sk-ant-...
```

---

## Configuration

All model and environment settings are read from a YAML file passed via `--config`. The default is `AREAEnv/area_env/configs/default.yaml`. To use a different setup, create a new YAML file in `AREAEnv/area_env/configs/` and point `--config` at it.

**Available fields** (any subset can be set; unset fields fall back to `AREAEnvConfig` defaults):

```yaml
# AREAEnv/area_env/configs/my_experiment.yaml

# LLM models
agent_model: gpt-4.1                  # model for the AREA agent
agent_model_temperature: 0.0
user_model: gemini-3.1-pro-preview    # model for the MimicUser
user_model_temperature: 0.7
evaluator_model: gpt-4.1              # model for AtomicEvaluator
evaluator_model_temperature: 0.0

# Episode limits
max_steps: 15                         # hard step cap per episode

# Data file names (pin to a specific version)
synthesized_output_file: synthesized_output.json
data_sampled_file: data_sampled.json

# Misc
max_tokens: 12288
verbose: false
```

Then pass it to any run command:

```bash
python agent/run_agent.py \
  --config AREAEnv/area_env/configs/my_experiment.yaml \
  --strategy user_interaction \
  --dataset alexfabbri/multi_news \
  --persona 1 2 3 \
  --communication_habit neutral
```

CLI flags (`--agent_model`, `--max_turns`, etc.) override the YAML values when both are provided.

---

## Running Agents

All agents are launched via `agent/run_agent.py` from the repo root. Models and environment defaults are read from a YAML config; CLI flags override individual fields.

```bash
# Zero-shot — single LLM call, no interaction
python agent/run_agent.py \
  --config AREAEnv/area_env/configs/default.yaml \
  --strategy zero_shot \
  --agent_model gpt-4.1 \
  --dataset alexfabbri/multi_news \
  --persona 1 2 3

# User interaction — clarify via MIMIC user, then synthesize
python agent/run_agent.py \
  --config AREAEnv/area_env/configs/default.yaml \
  --strategy user_interaction \
  --agent_model claude-sonnet-4-6 \
  --dataset alexfabbri/multi_news \
  --persona 1 2 3 \
  --max_turns 15 \

  --communication_habit neutral

# Data interaction — inspect dataset samples, then rewrite with reasoning
python agent/run_agent.py \
  --config AREAEnv/area_env/configs/default.yaml \
  --strategy data_interaction \
  --agent_model gemini-3.1-pro-preview \
  --dataset alexfabbri/multi_news \
  --persona 1 2 3 \
  --split non_defining_instances

# Hybrid — route between user and data actions
python agent/run_agent.py \
  --config AREAEnv/area_env/configs/default.yaml \
  --strategy hybrid \
  --agent_model gemini-3.1-pro-preview \
  --dataset alexfabbri/multi_news \
  --persona 1 2 3 \
  --max_iterations 20 \

  --communication_habit active
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | none | Path to `AREAEnvConfig` YAML |
| `--strategy` | `zero_shot` | `zero_shot` \| `user_interaction` \| `data_interaction` \| `hybrid` |
| `--agent_model` | from config | LLM for the AREA agent |
| `--mimic_model` | from config | LLM for the MIMIC user |
| `--evaluator_model` | from config | LLM for the atomic evaluator |
| `--dataset` | from config | HuggingFace dataset id — must have a matching folder under `AREAEnv/data/data_synthesized/` |
| `--persona` | **required** | Space-separated persona IDs (e.g. `--persona 1 2 3`) |
| `--input_type` | `elevator_pitch_summary` | `elevator_pitch_summary` \| `deep_dive_summary` |
| `--max_turns` | from config | Max clarification rounds (`user_interaction`) |
| `--max_iterations` | from config | Max routing iterations (`hybrid`) |
| `--max_steps` | from config | Max env steps before forced termination |
| `--split` | `all` | Data split for `data_interaction`: `defining_instances` \| `non_defining_instances` \| `all` |
| `--communication_habit` | **required** for `user_interaction`/`hybrid` | `passive` \| `neutral` \| `active` |
| `--output_dir` | auto under `agent/results/` | Override results output directory |
| `--exp_id` | auto-increment | Resume a specific `Experiment<N>` |
| `--log_file_path` | auto | Path for the run log file |
| `--no_eval` | false | Skip evaluation; save `output.json` only |

### Supported Models

Model provider is inferred automatically from the model name prefix.

| Prefix | Provider | Example models |
|--------|----------|----------------|
| `gpt-`, `o1`, `o3`, `o4` | OpenAI | `gpt-4.1`, `gpt-4.1-mini` |
| `gemini-` | Google | `gemini-2.0-flash`, `gemini-3.1-pro-preview` |
| `claude-` | Anthropic | `claude-sonnet-4-6`, `claude-opus-4-7` |

To add a new model, insert an entry in `PRICING_DATA` in [AREAEnv/area_env/utils/llm.py](AREAEnv/area_env/utils/llm.py).

### SLURM Batch Jobs

```bash
bash agent/scripts/run_zero_shot_agent.sh
bash agent/scripts/run_user_interaction_agent.sh
bash agent/scripts/run_data_interaction_agent.sh
bash agent/scripts/run_hybrid_agent.sh
```

Edit the variables at the top of each script before submitting:

```bash
REPO_ROOT="/path/to/AREA"
AGENT_MODEL="gemini-3.1-pro-preview"   # or gpt-4.1, claude-sonnet-4-6, etc.
MAX_STEPS=5                             # zero_shot / data_interaction
MAX_TURNS=15                            # user_interaction
MAX_ITERATIONS=20                       # hybrid
OUTPUT_DIR="${REPO_ROOT}/results/..."
DATASETS=(...)                          # "data_home_dir:dataset_name" pairs
```

---

## Output Format

Results are saved to `agent/results/<dataset_name>/<strategy>/Experiment<N>/`.

**`output.json`** — per-persona, per-task results:
```json
{
  "<persona_id>": {
    "task_<n>": {
      "messages": [...],
      "task_requirement_final": "...",
      "strategy": "user_interaction",
      "model": "claude-sonnet-4-6",
      "ground_truth": "...",
      "cost": 0.012
    }
  }
}
```

**`eval_results.json`** — evaluation scores per task:
```json
{
  "<persona_id>": {
    "task_<n>": {
      "predicted_units": [...],
      "ground_truth_units": [...],
      "counts": {"tp": 8, "fp": 2, "fn": 3},
      "scores": {"precision": 0.8, "recall": 0.727, "f1": 0.762},
      "subcategory_scores": {
        "user_specified": {"precision": 0.9, "recall": 0.75, "f1": 0.818},
        "data_specified": {"precision": 0.7, "recall": 0.7, "f1": 0.7}
      }
    }
  }
}
```

---

## Data

Synthesized personas and gold task requirements are stored under `AREAEnv/data/data_synthesized/<dataset_name>/synthesized_output_2.6.json`. Each entry contains:

- `user_<N>` — persona background, expertise level, goals
- Per-persona task entries with `elevator_pitch`, `deep_dive_summary`, and `task_requirement` (gold)

Data samples used by `inspect_data` actions are stored under `AREAEnv/data/data_sampled/<dataset_name>/data_sampled_2.1.json`, split into `defining_instances` and `non_defining_instances`.
