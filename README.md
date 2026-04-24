# AUNU — AI-Assisted User Needs Understanding

AUNU is a research framework for benchmarking how LLM agents transform underspecified user requests into structured task requirement specifications. Agents interact with a simulated user and/or real dataset samples across three strategies, and are evaluated with an atomic-unit LLM-based scoring pipeline.

---

## Repository Structure

```
AUNU/
├── agent/                          # Agent layer (main entry point)
│   ├── run_agent.py                # CLI entry point for all strategies
│   ├── zero_shot_agent.py          # Zero-shot strategy
│   ├── user_interaction_agent.py   # User-interaction strategy
│   ├── data_interaction_agent.py   # Data-interaction strategy
│   ├── agent_state.py              # LangGraph AgentState TypedDict
│   ├── prompts/                    # All Jinja2 prompt templates
│   │   ├── agents/
│   │   │   ├── aunu_agent/
│   │   │   │   ├── zero_shot.jinja
│   │   │   │   ├── user/           # user_interaction prompts
│   │   │   │   ├── data/           # data_interaction prompts
│   │   │   │   └── hybrid/
│   │   │   └── mimic_user/         # MIMIC user simulation prompts
│   │   └── Evaluation/             # LLM judge prompts (legacy location)
│   └── scripts/                    # SLURM batch submission scripts
│
├── AUNUEnv/                        # Benchmark environment package
│   └── aunu_env/
│       ├── env/
│       │   ├── aunu_env.py         # Core gym-style environment
│       │   ├── actions.py          # Action constructors & validation
│       │   └── state.py            # EpisodeState dataclass
│       ├── dataset/
│       │   ├── schema.py           # PersonaInfo, TaskInstance dataclasses
│       │   └── loader.py           # load_dataset() from synthesized JSON
│       ├── users/
│       │   └── mimic_user.py       # Simulated user (passive & persona modes)
│       ├── evaluator/
│       │   ├── atomic_evaluator.py # 3-stage LLM evaluation pipeline
│       │   ├── metrics.py          # F1/precision/recall computation
│       │   └── prompt/             # LLM judge Jinja2 templates
│       ├── utils/
│       │   ├── llm.py              # Unified LLM call interface (retry + cost)
│       │   ├── jinja_utils.py      # render_template() helper
│       │   └── json_utils.py       # parse_json_output() helper
│       └── config.py               # AUNUEnvConfig dataclass
│
├── data/
│   └── data_synthesized/
│       └── <dataset_name>/
│           └── synthesized_output.json   # Personas + gold task specs per dataset
│
├── scripts/                        # Shared utility scripts
│   └── model/
│       └── model_base.py           # Alternative unified LLM interface
│
└── agent-ui/                       # React + Vite visualization frontend
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
| `propose_requirement_update(requirement)` | Update the current draft requirement |
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

Scores computed from TP/FP/FN counts: **precision**, **recall**, **F1**, **alignment**, **constraint_preservation**.

### MIMIC User (`MimicUser`)

Simulates user feedback using Jinja2 templates. Two modes:
- **passive** — generic confirm/deny responses
- **persona** — responses conditioned on the persona's background and domain expertise

---

## Installation

**Requirements:** Python 3.10+

```bash
# Install dependencies
pip install anthropic datasets jinja2 langgraph pandas python-dotenv

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

All agents are launched via `agent/run_agent.py` from the repo root.

```bash
# Zero-shot
python agent/run_agent.py \
  --strategy zero_shot \
  --agent_model gpt-4.1 \
  --dataset alexfabbri/multi_news \
  --persona 1 2 3

# User interaction
python agent/run_agent.py \
  --strategy user_interaction \
  --agent_model claude-haiku-4-5-20251001 \
  --mimic_model gemini-3.1-flash-lite-preview \
  --evaluator_model gpt-5.4 \
  --dataset alexfabbri/multi_news \
  --persona 1 2 3 4 5 \
  --max_turns 4 \
  --user_mode passive

# Data interaction
python agent/run_agent.py \
  --strategy data_interaction \
  --agent_model claude-haiku-4-5-20251001 \
  --evaluator_model gpt-5.4 \
  --dataset alexfabbri/multi_news \
  --persona 1 2 3 4 5 \
  --max_turns 3
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--strategy` | `zero_shot` | `zero_shot` \| `user_interaction` \| `data_interaction` |
| `--agent_model` | required | LLM for the AUNU agent |
| `--mimic_model` | agent_model | LLM for the MIMIC user (`user_interaction` only) |
| `--evaluator_model` | agent_model | LLM for the atomic evaluator |
| `--dataset` | `alexfabbri/multi_news` | Dataset name (must exist under `data/data_synthesized/`) |
| `--persona` | required | Space-separated persona IDs to run |
| `--input_type` | `elevator_pitch_summary` | `elevator_pitch_summary` \| `deep_dive_summary` |
| `--max_turns` | `5` | Max clarification/refinement turns |
| `--max_steps` | `15` | Max env steps before forced termination |
| `--user_mode` | `passive` | `passive` \| `persona` |
| `--log_file_path` | auto | Path for run log (auto-increments experiment number) |

### Supported Models

Any model name starting with the following prefixes is routed automatically to the correct provider. Models not in the pricing table will run with `cost=0.0`.

| Prefix | Provider |
|--------|----------|
| `gpt-`, `o1`, `o3`, `o4` | OpenAI |
| `gemini` | Google |
| `claude` | Anthropic |

### SLURM Batch Jobs

```bash
sbatch agent/scripts/run_zero_shot_agent.sh
sbatch agent/scripts/run_user_interaction_agent.sh
sbatch agent/scripts/run_data_interaction_agent.sh
```

Edit the model, dataset, and turn variables at the top of each script before submitting.

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
cd agent-ui && npm install && npm run dev -- --host 0.0.0.0 --port 5173
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
