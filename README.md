# AUNU — AI-Assisted User Needs Understanding

AUNU is a research framework for benchmarking how LLM agents transform underspecified user requests into structured task requirement specifications. Agents interact with a simulated user and/or real dataset samples across three strategies, and are evaluated with an atomic-unit LLM-based scoring pipeline.

---

## Repository Structure

```
AUNU/
├── agent/                              # Agent layer (main entry point)
│   ├── run_agent.py                    # CLI entry point for all strategies
│   ├── zero_shot_agent.py              # Zero-shot strategy implementation
│   ├── user_interaction_agent.py       # User-interaction strategy implementation
│   ├── data_interaction_agent.py       # Data-interaction strategy implementation
│   ├── agent_state.py                  # LangGraph AgentState TypedDict
│   ├── prompts/                        # Jinja2 prompt templates
│   │   ├── agents/
│   │   │   ├── aunu_agent/
│   │   │   │   ├── zero_shot.jinja
│   │   │   │   ├── user/               # user_interaction prompts
│   │   │   │   ├── data/               # data_interaction prompts
│   │   │   │   └── hybrid/
│   │   │   └── mimic_user/             # MIMIC user simulation prompts
│   │   └── Evaluation/                 # LLM judge prompts
│   │       ├── LLM_judge_decompose.jinja
│   │       └── LLM_judge_compare.jinja
│   ├── scripts/                        # SLURM batch submission scripts
│   │   ├── run_zero_shot_agent.sh
│   │   ├── run_user_interaction_agent.sh
│   │   └── run_data_interaction_agent.sh
│   ├── logs/                           # Per-strategy run logs
│   │   ├── zero_shot/
│   │   ├── user_interaction/
│   │   └── data_interaction/
│   └── results/                        # Per-dataset experiment outputs
│       └── <dataset_name>/
│           ├── zero_shot/Experiment<N>/
│           ├── user_interaction/Experiment<N>/
│           └── data_interaction/Experiment<N>/
│
├── AUNUEnv/                            # Benchmark environment package
│   ├── aunu_env/
│   │   ├── config.py                   # AUNUEnvConfig dataclass
│   │   ├── configs/
│   │   │   └── default.yaml            # Default model & env config (edit here)
│   │   ├── env/
│   │   │   ├── aunu_env.py             # Core gym-style environment
│   │   │   ├── actions.py              # Action constructors & validation
│   │   │   └── state.py                # EpisodeState dataclass
│   │   ├── dataset/
│   │   │   ├── schema.py               # PersonaInfo, TaskInstance dataclasses
│   │   │   └── loader.py               # load_dataset() from synthesized JSON
│   │   ├── users/
│   │   │   ├── mimic_user.py           # Simulated user (passive & persona modes)
│   │   │   └── prompts/                # User simulator Jinja2 templates
│   │   ├── evaluator/
│   │   │   ├── atomic_evaluator.py     # 3-stage LLM evaluation pipeline
│   │   │   ├── metrics.py              # F1/precision/recall computation
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
│   │   └── run_evaluation.py           # Standalone re-evaluator
│   └── data/                           # AUNUEnv local data (gitignored)
│       ├── data_raw/                   # Sampled CSVs per dataset
│       └── data_synthesized/           # synthesized_output.json per dataset
│
├── results/                            # Legacy / cross-dataset results
│   └── <dataset_name>/
│       ├── zero_shot/
│       ├── user/
│       ├── data/
│       ├── persona/
│       └── ground_truth_units.json
│
├── agent-ui/                           # React + Vite visualization frontend
│   └── src/App.tsx                     # Main UI component
│
├── result_visualization.ipynb          # Notebook for results analysis
├── pyproject.toml                      # Python project / dependency config
└── .env                                # API keys (gitignored)
```

---

## Architecture

### Environment (`AUNUEnv`)

`AUNUEnv` is a gym-style benchmark environment decoupled from any specific agent. It exposes a standard `reset() → step() → get_trajectory_log()` interface.

**Action space** (4 action types):

| Action | Description |
|--------|-------------|
| `ask_user(question)` | Pose a clarification question to the MIMIC user |
| `inspect_data(n_samples, query)` | Sample rows from the task dataset CSV |
| `finish(final_requirement)` | Submit the final requirement and trigger evaluation |

**Episode flow:**
1. `env.reset(task, user)` — initialize state with a `TaskInstance` and `MimicUser`
2. Agent submits actions via `env.step(action)` → receives `(observation, reward, done, info)`
3. On `finish()`, the `AtomicEvaluator` runs automatically
4. `env.get_trajectory_log()` returns the full episode log including eval scores

### Agents

All three agents are LangGraph `StateGraph` policies operating over `AgentState`. They share the same interface: `agent.run(env, task, [user]) → log`.

#### Zero-shot (`zero_shot_agent.py`)
Single LLM call on the elevator pitch. No interaction.
```
[zero_shot node] → END
```

#### User Interaction (`user_interaction_agent.py`)
Iterative clarification with the MIMIC user, then final synthesis.
```
Phase 0: zero-shot draft → propose_requirement_update
Phase 1: ask_user loop (≤ max_turns) → env.step(ask_user(question))
Phase 2: synthesize final → env.step(finish(requirement))
```

#### Data Interaction (`data_interaction_agent.py`)
Iterative refinement by stress-testing the current requirement against real data.
```
Phase 0: zero-shot draft → propose_requirement_update
Phase 1: per turn:
           inspect_data → execute on samples → reflect → rewrite
           → propose_requirement_update
         (≤ max_turns, falls through to finish on last turn)
Phase 2: env.step(finish(refined_requirement))
```

### Evaluation (`AtomicEvaluator`)

Three-stage LLM pipeline comparing the agent's final requirement against the gold requirement:

1. **Decompose gold** → list of atomic requirement units (cached per task)
2. **Decompose predicted** → list of atomic requirement units
3. **Compare** → matched pairs, missing units, hallucinated units

Scores computed from TP/FP/FN counts: **precision**, **recall**, **F1**

### MIMIC User (`MimicUser`)

Simulates user feedback using Jinja2 templates. Two modes:
- **passive** — generic confirm/deny responses
- **persona** — responses conditioned on the persona's background and domain expertise

---

## Installation

**Requirements:** Python 3.10+

```bash
# Install from requirements.txt
pip install -r requirements.txt

# Or with uv
uv sync
```

**API keys** — create a `.env` file in the repo root:
```
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
ANTHROPIC_API_KEY=sk-ant-...
```

---

## Running Agents

All agents are launched via `agent/run_agent.py` from the repo root. Models and environment defaults are read from a YAML config file; CLI flags override individual fields.

```bash
# Zero-shot — single LLM call, no interaction
python agent/run_agent.py \
  --config AUNUEnv/aunu_env/configs/default.yaml \
  --strategy zero_shot \
  --agent_model gpt-4.1 \
  --dataset alexfabbri/multi_news \
  --persona 1 2 3

# User interaction — clarify via MIMIC user, then synthesize
python agent/run_agent.py \
  --config AUNUEnv/aunu_env/configs/default.yaml \
  --strategy user_interaction \
  --agent_model gpt-4.1 \
  --mimic_model gpt-4.1-mini \
  --evaluator_model gpt-4.1 \
  --dataset alexfabbri/multi_news \
  --persona 1 2 3 4 5 \
  --max_turns 3 \
  --user_mode passive

# Data interaction — inspect dataset samples, reflect, rewrite
python agent/run_agent.py \
  --config AUNUEnv/aunu_env/configs/default.yaml \
  --strategy data_interaction \
  --agent_model gpt-4.1 \
  --evaluator_model gpt-4.1 \
  --dataset alexfabbri/multi_news \
  --persona 1 2 3 4 5 \
  --max_turns 3
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | none | Path to `AUNUEnvConfig` YAML. Models, temperatures, `max_steps`, and `user_mode` are read from here; CLI flags override individual fields. |
| `--strategy` | `zero_shot` | `zero_shot` \| `user_interaction` \| `data_interaction` |
| `--agent_model` | from config | LLM for the AUNU agent (overrides config) |
| `--mimic_model` | from config `user_model` | LLM for the MIMIC user (overrides config) |
| `--evaluator_model` | from config | LLM for the atomic evaluator (overrides config) |
| `--dataset` | from config | HuggingFace dataset id — must have a matching folder under `AUNUEnv/data/data_synthesized/` |
| `--persona` | **required** | Space-separated persona IDs to run (e.g. `--persona 1 2 3`) |
| `--input_type` | `elevator_pitch_summary` | `elevator_pitch_summary` \| `deep_dive_summary` |
| `--max_turns` | `5` | Max clarification rounds (`user_interaction`) or data-inspection cycles (`data_interaction`) |
| `--max_steps` | from config (`15`) | Max env steps before forced termination (overrides config) |
| `--user_mode` | from config (`passive`) | `passive` \| `persona` (overrides config) |
| `--output_dir` | auto under `agent/results/` | Override the results output directory |
| `--log_file_path` | auto under `agent/logs/` | Path for the run log file |
| `--verbose` | false | Enable DEBUG-level logging |

### Supported Models

Model provider is inferred automatically from the model name prefix. Models not listed in the pricing table in `AUNUEnv/aunu_env/utils/llm.py` will still run but cost will be reported as `0.0`.

| Prefix | Provider | Example models |
|--------|----------|----------------|
| `gpt-`, `o1`, `o3`, `o4` | OpenAI | `gpt-4.1`, `gpt-4.1-mini`, `o3` |
| `gemini` | Google | `gemini-2.0-flash`, `gemini-2.5-pro-preview-03-25` |
| `claude` | Anthropic | `claude-opus-4-7`, `claude-haiku-4-5-20251001` |

To add a new model, insert an entry in `PRICING_DATA` in [AUNUEnv/aunu_env/utils/llm.py](AUNUEnv/aunu_env/utils/llm.py).

### SLURM Batch Jobs

```bash
sbatch agent/scripts/run_zero_shot_agent.sh
sbatch agent/scripts/run_user_interaction_agent.sh
sbatch agent/scripts/run_data_interaction_agent.sh
```

Edit the variables at the top of each script before submitting:

```bash
CONFIG="..."           # path to default.yaml
AGENT_MODEL="gpt-4.1"
MAX_TURNS=3            # user_interaction / data_interaction only
MAX_STEPS=15
OUTPUT_DIR="..."
DATASETS=(...)         # list of "data_dir:dataset_name" pairs to run
```

---

## Output Format

Results are saved to `agent/results/<dataset_name>/<strategy>/Experiment<N>/`:

**`output.json`** — per-persona, per-task results:
```json
{
  "<persona_id>": {
    "task_<n>": {
      "messages": [...],
      "is_complete": true,
      "task_requirement_final": "...",
      "strategy": "user_interaction",
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
      "scores": {"precision": 0.8, "recall": 0.727, "f1": 0.762, "alignment": 0.9, "constraint_preservation": 0.75}
    }
  }
}
```

---

## Visualization UI

A React + Vite frontend is available for exploring agent trajectories and evaluation results.

```bash
# Serve result files (from repo root)
python -m http.server 3000 --bind 0.0.0.0

# Start the frontend (from agent-ui/)
cd agent-ui 
npm install 
npm run dev -- --host 0.0.0.0 --port 5173
```

If running on a remote server, forward both ports locally:
```bash
ssh -N -L 5173:localhost:5173 -L 3000:localhost:3000 <your-server>
```

Then open `http://localhost:5173` in your browser.

---

## Data

Synthesized personas and gold task requirements are stored under `data/data_synthesized/<dataset_name>/synthesized_output.json`. Each entry contains:
- `user_<N>` — persona background, expertise level, goals
- Per-persona task entries with `elevator_pitch`, `deep_dive_summary`, and `task_requirement` (gold)

Raw dataset samples (CSV) used by `inspect_data` actions are stored under `data/data_raw/` (excluded from version control).
