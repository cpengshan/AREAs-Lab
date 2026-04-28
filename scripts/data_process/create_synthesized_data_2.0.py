import os
import json
import argparse
import pandas as pd
from jinja2 import Environment, FileSystemLoader
from json_repair import repair_json
from collections import defaultdict
import sys
import re
import concurrent.futures

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)
from scripts.model.model_base import LLM

def parse_args():
    parser = argparse.ArgumentParser(description="Synthesize user and task data using an LLM.")
    
    # Required arguments
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--num_users", type=int, required=True, help="Number of synthesized users to create")
    
    # Optional arguments with defaults
    parser.add_argument("--dir_prompt", type=str, default="constants/prompts/data_process", help="Prompt directory")
    parser.add_argument("--dir_data_sampled", type=str, default="data/data_sampled", help="Sampled data directory")
    parser.add_argument("--dir_data_synthesized", type=str, default="data/data_synthesized", help="Synthesized data directory")
    parser.add_argument("--fname_prompt_user", type=str, default="step 2: user_creation.jinja", help="User prompt template")
    parser.add_argument("--fname_prompt_task", type=str, default="step 3: task_creation.jinja", help="Task prompt template")
    parser.add_argument("--fname_prompt_eval", type=str, default="step 4: evaluation.jinja", help="Evaluation prompt template")
    parser.add_argument("--fname_output", type=str, default="synthesized_output.json", help="The output file name")
    parser.add_argument("--llm_creator", type=str, default="gemini-3.1-pro-preview", help="The creator LLM model name")
    parser.add_argument("--llm_evaluators", type=str, nargs='+', default=["gpt-5", "deepseek-reasoner-v4"], help="The evaluator LLM model names")
    parser.add_argument("--max_workers", type=int, default=10, help="Maximum number of concurrent threads")
    return parser.parse_args()


def extract_json(text_in):
    pattern = r"```json\s*(.*?)\s*```"
    matches = re.findall(pattern, text_in, re.DOTALL)
    if matches:
        text_in_ = matches[0]
    else:
        text_in_ = text_in
    return json.loads(repair_json(text_in_))


def evaluate_single_task(id_task, task, model_name, prompt_eval, eval_key):
    """Worker function for running a single evaluation via an LLM."""
    llm_evaluator = LLM(model_name=model_name)
    output_eval = llm_evaluator.generate(prompt_eval)
    
    eval_result = None
    try:
        eval_result = extract_json(output_eval["output"])
    except Exception as e:
        print(f"Evaluation Error for task {id_task} using {model_name}: {e}")
        
    cost = output_eval.get("cost", 0)
    print(f"Evaluation completed for task {id_task} by {model_name}. Cost ${cost}")
    
    return {
        "id_task": id_task,
        "model_name": model_name,
        "eval_result": eval_result,
        "cost_key": f"{eval_key}_{model_name}",
        "cost": cost
    }


def process_single_user(id_user, user, args, dataset_info, defining_instances, prompt_temp_task, prompt_temp_eval):
    """Worker function for generating tasks and orchestrating evaluations for a single user."""
    print(f"Synthesizing tasks for user {id_user} ...")
    
    # Initialize separate LLM instance per thread to ensure thread-safety
    llm_creator = LLM(model_name=args.llm_creator)
    
    user_info = {
        "competency_matrix": user.get("competency_matrix"),
        "business_motivation": user.get("business_motivation"),
        "workflow_friction": user.get("workflow_friction")
    }        
    
    prompt_task = prompt_temp_task.render(
        dataset_name=args.dataset_name,
        user_info=user_info,
        dataset_info=dataset_info,
        sampled_data=defining_instances
    )
    
    output_task = llm_creator.generate(prompt_task)
    
    local_results = {
        "user_key": f"user_{id_user}",
        "data": {
            "user_info": user,
            "tasks_info": None,
            "validation_results": {}
        },
        "prompts": {
            f"synthesizing_tasks_for_user_{id_user}": prompt_task
        },
        "costs": defaultdict(float)
    }
    
    local_results["costs"][f"synthesizing_tasks_for_user_{id_user}"] = output_task.get("cost", 0)

    try:
        tasks = extract_json(output_task["output"])
        assert "error" not in tasks
        local_results["data"]["tasks_info"] = tasks
    except Exception as e:
        print(f"Output_task Error for user {id_user}: {e} \n{output_task}")
        return local_results # Return early on failure
        
    print(f"Task synthesis completed for user {id_user}. Cost ${output_task.get('cost', 0)}")

    # Evaluate the generated tasks in parallel
    eval_key = f"evaluating_tasks_for_user_{id_user}"
    eval_futures = []
    
    # We spawn a nested ThreadPool for the evaluations to maximize throughput
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.llm_evaluators)) as eval_executor:
        for id_task, task in enumerate(tasks, start=1):
            local_results["data"]["validation_results"][f"task_{id_task}"] = {}
            
            prompt_eval = prompt_temp_eval.render(
                dataset_info=json.dumps(dataset_info),
                user_info=json.dumps(user_info),
                task_info=json.dumps(task)
            )
            local_results["prompts"][eval_key] = prompt_eval
            
            for model_name in args.llm_evaluators:
                print(f"Scheduling evaluation: user {id_user} - task {id_task} - {task.get('task_name', 'Unnamed')} with {model_name}...")
                eval_futures.append(
                    eval_executor.submit(
                        evaluate_single_task, id_task, task, model_name, prompt_eval, eval_key
                    )
                )
        
        for future in concurrent.futures.as_completed(eval_futures):
            res = future.result()
            task_key = f"task_{res['id_task']}"
            local_results["data"]["validation_results"][task_key][res['model_name']] = res['eval_result']
            local_results["costs"][res['cost_key']] += res['cost']

    return local_results


def synthesize_data(args):
    # 1. Setup Environment and Load Templates
    env = Environment(loader=FileSystemLoader(args.dir_prompt))
    prompt_temp_user = env.get_template(args.fname_prompt_user)
    prompt_temp_task = env.get_template(args.fname_prompt_task)
    prompt_temp_eval = env.get_template(args.fname_prompt_eval)
    
    # 2. Load Data
    with open(os.path.join(args.dir_data_sampled, args.dataset_name, "data_sampled.json")) as f:
        sampled_data = json.load(f)
        defining_instances = sampled_data["defining_instances"]
    print(f"Start synthesizing data for {os.path.join(args.dir_data_sampled, args.dataset_name)}")

    # 3. Initialize Creator LLM (for user generation)
    llm_creator = LLM(model_name=args.llm_creator)
    print(f"Creator loaded: {args.llm_creator}. Evaluators: {', '.join(args.llm_evaluators)}")

    # 4. Load Dataset Information
    analysis_path = os.path.join(args.dir_data_synthesized, args.dataset_name, 'data_analysis.json')
    with open(analysis_path, 'r') as f:
        dataset_info = json.load(f)

    outputs = defaultdict(dict)
    outputs["args"] = vars(args)
    outputs["cost"] = defaultdict(float) 
    outputs["prompt"] = {}

    # --- Step 1: Create synthesized users (Sequential) ---
    print(f"Synthesizing users ...")
    prompt_user = prompt_temp_user.render(
        dataset_name=args.dataset_name,
        dataset_info=dataset_info,
        num_users=args.num_users
    )
    output_user = llm_creator.generate(prompt_user)
    try:
        users = extract_json(output_user["output"])
        assert "error" not in users
    except Exception as e:
        print(f"Output_user Error: {e} \n{output_user}")
        return
    
    print(f"User synthesis completed. Cost ${output_user.get('cost', 0)}")
    outputs["cost"]["sythesizing_users"] = output_user.get("cost", 0)
    outputs["prompt"]["sythesizing_users"] = prompt_user
    
    # --- Step 2 & 3: Create and evaluate tasks for each user (Parallel) ---
    print(f"\nStarting parallel task generation and evaluation using up to {args.max_workers} workers...\n")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = []
        for id_user, user in enumerate(users, start=1):
            futures.append(
                executor.submit(
                    process_single_user, 
                    id_user, user, args, dataset_info, defining_instances, 
                    prompt_temp_task, prompt_temp_eval
                )
            )
            
        # Collect results as they complete
        for future in concurrent.futures.as_completed(futures):
            user_res = future.result()
            user_key = user_res["user_key"]
            
            # Merge returned localized data into the main tracking dictionary
            outputs[user_key] = user_res["data"]
            outputs["prompt"].update(user_res["prompts"])
            
            for cost_key, cost_value in user_res["costs"].items():
                outputs["cost"][cost_key] += cost_value

    # --- Save the outputs ---
    output_dir = os.path.join(args.dir_data_synthesized, args.dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    save_path = os.path.join(output_dir, args.fname_output)
    with open(save_path, 'w') as f_out:
        json.dump(outputs, f_out, indent=4)
    
    print(f"Data synthesis complete! Results saved to: {save_path}\n\n" + "*" * 50 + "\n\n")

if __name__ == "__main__":
    args = parse_args()
    synthesize_data(args)