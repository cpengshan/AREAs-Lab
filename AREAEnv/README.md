# AREAEnv

**Interactive Benchmark Environment for AI-Assisted User Needs Understanding (AREA)**

AREAEnv evaluates how an agent transforms an underspecified user request into a complete task requirement specification through interaction with a simulated user and/or inspection of task data.

---

## How AREAEnv Differs from ReqElicitBench

| Dimension | ReqElicitBench | AREAEnv |
|---|---|---|
| **Domain** | Software requirements elicitation | AI task requirement understanding |
| **Gold object** | User stories / interview coverage | Complete task requirement specification |
| **Action space** | Single string (natural language question) | Typed dict: `ask_user` / `inspect_data` / `finish` |
| **Information sources** | User only | User + dataset samples |
| **User simulator** | One mode (quality levels: high/med/low) | Two modes: passive confirmation or persona-conditioned |
| **Evaluation** | TKQR, ORA, elicitation ratio | Atomic unit precision / recall / F1 (3-stage LLM pipeline) |
| **Policies** | Fixed interviewer prompt baked into env | External — env exposes an API, policies live outside it |
| **Gym dependency** | `gymnasium` required | No external gym dependency |

---

## Directory Structure

```
AREAEnv/
  area_env/                        # Environment package (env primitives only)
    config.py                      # AREAEnvConfig dataclass
    configs/
      default.yaml                 # Model & env config — edit this to change models
    dataset/
      schema.py                    # PersonaInfo, TaskInstance dataclasses
      loader.py                    # load_dataset() — reads synthesized_output.json
    env/
      actions.py                   # Action constructors + validation
      state.py                     # EpisodeState dataclass
      area_env.py                  # AREAEnv: reset / step / get_trajectory_log
    users/
      mimic_user.py             # MimicUser: communication habit-conditioned
      prompts/
        feedback_mimic_user_v3.jinja  # Feedback prompt (current)
        responser_habit.json          # passive/neutral/active habit configs
    evaluator/
      atomic_evaluator.py          # 3-stage LLM evaluation pipeline
      metrics.py                   # compute_scores(), aggregate_results()
      prompt/
        LLM_judge_decompose.jinja  # Decompose requirement → atomic units
        LLM_judge_compare.jinja    # Align predicted vs gold units
        LLM_judge.jinja
    utils/
      llm.py                       # call_llm() — add new models/providers here
      jinja_utils.py               # render_template()
      json_utils.py                # parse_json_output()

  examples/
    minimal_example.py             # End-to-end env loop with stub LLM (no API key needed)
    policies.py                    # Reference policy implementations (ZeroShot, UserOnly, DataOnly)

  scripts/
    run_experiment.py              # Experiment runner + CLI
    run_evaluation.py              # Standalone re-evaluator

  data/                            # Local data (gitignored)
    data_raw/                      # Sampled CSVs per dataset
    data_synthesized/              # synthesized_output.json per dataset
```

---

## Core Concepts

### The `area_env` Package

The package exposes only environment primitives. It has no opinions about policies:

```
area_env
  ├── AREAEnvConfig   — experiment configuration
  ├── load_dataset    — load TaskInstance objects from synthesized_output.json
  ├── AREAEnv         — the environment (reset / step / get_trajectory_log)
  ├── MimicUser     — simulated user with configurable communication habit
  └── AtomicEvaluator — LLM-based requirement evaluator
```

### Action Space

Each `env.step()` call receives an action dict:

| Action | Required keys | Effect |
|---|---|---|
| `ask_user` | `question` | MIMIC user generates a response |
| `inspect_data` | `n_samples`, `query` | Returns sampled CSV rows |
| `finish` | `final_requirement` | Evaluates and ends the episode |

### MIMIC User

`MimicUser(model_name, habit)` — simulates user feedback conditioned on a pre-loaded communication habit dict (one of `passive`, `neutral`, or `active` from `responser_habit.json`). The habit controls how proactively and verbosely the user volunteers information.

### Evaluation Pipeline

Reuses the atomic unit pipeline from `scripts/evaluation/user_interaction_judge.py`:

1. Decompose gold requirement → atomic units (`LLM_judge_decompose.jinja`)
2. Decompose predicted requirement → atomic units
3. LLM alignment comparison (`LLM_judge_compare.jinja`)
4. Python scoring: TP / FP / FN → precision, recall, F1

Gold unit decompositions are cached per `task_id` to avoid redundant LLM calls.

### Policies and Runner

Policies and the experiment runner are **external** to `area_env`. Reference implementations live in `examples/policies.py` and the runner lives in `scripts/run_experiment.py`.

Three reference policies are provided:

| Policy | Strategy |
|---|---|
| `ZeroShotPolicy` | Single LLM call from elevator pitch → `finish` |
| `UserOnlyPolicy` | Zero-shot draft → `ask_user` loop → synthesize → `finish` |
| `DataOnlyPolicy` | Zero-shot draft → `inspect_data` → execute/reflect/rewrite loop → `finish` |

Any object with a `run(env, task) -> dict` method is a valid policy.

---

## Quick Start

### 1. Install dependencies

```bash
pip install jinja2 pandas pyyaml openai python-dotenv
export OPENAI_API_KEY=sk-...
```

### 2. Run the minimal example (no LLM calls)

```bash
cd /local/scratch/zzh2365/AREA/AREAEnv
python3 examples/minimal_example.py
```

### 3. Run a built-in policy on one task

```bash
python3 scripts/run_experiment.py \
    --config area_env/configs/zero_shot.yaml \
    --policy zero_shot \
    --task_id user_1_task_0
```

### 4. Run user-only over the full dataset

```bash
python3 scripts/run_experiment.py \
    --config area_env/configs/user_only.yaml \
    --policy user_only \
    --max_turns 5
```

### 5. Run data-only

```bash
python3 scripts/run_experiment.py \
    --config area_env/configs/data_only.yaml \
    --policy data_only \
    --max_data_iterations 3 \
    --n_data_samples 3
```

### 6. Plug in a custom policy

```bash
python3 scripts/run_experiment.py \
    --config area_env/configs/user_only.yaml \
    --policy_module my_package.my_policy_factory
```

The factory must have the signature `factory(user: MimicUser) -> policy`.

### 7. Use the env directly in your own script

```python
from area_env import AREAEnv, MimicUser, AtomicEvaluator, load_dataset, AREAEnvConfig
from area_env.env.actions import ask_user, finish
import json

config = AREAEnvConfig.from_yaml("area_env/configs/zero_shot.yaml")
tasks   = load_dataset(config.dataset_path, config.data_csv_path)
habits  = json.load(open("area_env/users/prompts/responser_habit.json"))
user    = MimicUser(model_name=config.user_model, habit=habits["neutral"])
env     = AREAEnv(evaluator=AtomicEvaluator(config.evaluator_model), max_steps=config.max_steps)

for task in tasks:
    obs, info = env.reset(task, user)
    # ... your policy logic ...
    obs, reward, done, info = env.step(finish("my final requirement"))
    log = env.get_trajectory_log()
```

---

## Output Format

Each experiment produces a timestamped JSON file in `output_dir`:

```json
{
  "config": { "agent_model": "gpt-4.1", "max_steps": 15, ... },
  "timestamp": "2025-04-23T12:00:00",
  "n_tasks": 10,
  "n_successful": 10,
  "aggregate_scores": {
    "precision": 0.72, "recall": 0.68, "f1": 0.70, "n_tasks": 10
  },
  "task_results": [
    {
      "task_id": "user_1_task_0",
      "n_steps": 7,
      "total_cost": 0.0042,
      "trajectory": [
        { "step": 1, "action": { "type": "ask_user", "question": "..." }, "response": "..." },
        { "step": 2, "action": { "type": "inspect_data", "..." }, "response": "..." },
        "..."
      ],
      "final_requirement": "### 1. Strategic Intent\n...",
      "eval_result": {
        "scores": { "precision": 0.75, "recall": 0.70, "f1": 0.72, "..." },
        "counts": { "tp": 7, "fp": 2, "fn": 3, "..." },
        "gold_units": ["...", "..."],
        "predicted_units": ["...", "..."]
      }
    }
  ]
}
```

---

## Assumptions

1. `scripts/model/model_base.py` (the project's `LLM` class) is importable — `area_env/utils/llm.py` resolves it dynamically from the project root.
2. Jinja prompt templates from `prompts/agents/area_agent/` and `prompts/Evaluation/` are used in place; no copies are made inside `AREAEnv/`.
3. An OpenAI-compatible API key is set via the `OPENAI_API_KEY` environment variable.
4. `sampled_data.csv` path is provided in the config or via CLI.
5. `gymnasium` is **not** required — AREAEnv uses a gym-style interface without inheriting `gym.Env`.
