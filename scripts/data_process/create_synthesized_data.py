import os
import json
import argparse
import pandas as pd
from jinja2 import Environment, FileSystemLoader
from json_repair import repair_json
from collections import defaultdict
import sys
import re

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
    parser.add_argument("--dir_data_processed", type=str, default="data/data_processed", help="Processed data directory")
    parser.add_argument("--dir_data_synthesized", type=str, default="data/data_synthesized", help="Synthesized data directory")
    parser.add_argument("--fname_prompt_user", type=str, default="step 2: user_creation.jinja", help="User prompt template")
    parser.add_argument("--fname_prompt_task", type=str, default="step 3: task_creation.jinja", help="Task prompt template")
    parser.add_argument("--fname_prompt_eval", type=str, default="step 4: evaluation.jinja", help="Evaluation prompt template")
    parser.add_argument("--data_sample_size", type=int, default=1, help="Number of sampled data records per task")
    parser.add_argument("--llm_creator", type=str, default="gemini-3.1-pro-preview", help="The creator LLM model name")
    parser.add_argument("--llm_evaluator", type=str, default="gpt-5", help="The evaluator LLM model name")
    return parser.parse_args()


def extract_json(text_in):
    pattern = r"```json\s*(.*?)\s*```"
    matches = re.findall(pattern, text_in, re.DOTALL)
    if matches:
        text_in_ = matches[0]
    else:
        text_in_ = text_in
    return json.loads(repair_json(text_in_))


def synthesize_data(args):
    # 1. Setup Environment and Load Templates
    env = Environment(loader=FileSystemLoader(args.dir_prompt))
    prompt_temp_user = env.get_template(args.fname_prompt_user)
    prompt_temp_task = env.get_template(args.fname_prompt_task)
    prompt_temp_eval = env.get_template(args.fname_prompt_eval)
    
    # 2. Load Data
    df = pd.read_csv(os.path.join(args.dir_data_processed, args.dataset_name, "sampled_data.csv"))
    print("Data loaded ...")

    # 3. Initialize LLM
    llm_creator = LLM(model_name=args.llm_creator)
    llm_evaluator = LLM(model_name=args.llm_evaluator)
    print("LLM loaded ...")

    # 4. Load Dataset Information
    analysis_path = os.path.join(args.dir_data_synthesized, args.dataset_name, 'data_analysis.json')
    with open(analysis_path, 'r') as f:
        dataset_info = json.load(f)

    outputs = defaultdict(dict)
    outputs["args"] = vars(args)
    outputs["cost"] = defaultdict(float) # Ensure cost is a float-ready dict

    # --- Step 1: Create synthesized users ---
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
        return # Stop if no users are generated
    print(f"User synthesis completed. Cost ${output_user["cost"]}")
    outputs["cost"]["sythesizing_users"] = output_user["cost"]
    outputs["prompt"]["sythesizing_users"] = prompt_user
    
    # --- Step 2: Create tasks for each user ---
    for id_user, user in enumerate(users, start=1):
        print(f"Synthesizing tasks for user {id_user} ...")
        user_info = {
            "competency_matrix": user.get("competency_matrix"),
            "business_motivation": user.get("business_motivation"),
            "workflow_friction": user.get("workflow_friction")
        }        
        
        sampled_data = df.sample(n=args.data_sample_size, random_state=id_user).to_json(orient='records', indent=2)
        
        prompt_task = prompt_temp_task.render(
            dataset_name=args.dataset_name,
            user_info=user_info,
            dataset_info=dataset_info,
            sampled_data=sampled_data
        )
        
        output_task = llm_creator.generate(prompt_task)
        # Handle potential list/dict output from tasks
        try:
            tasks = extract_json(output_task["output"])
            assert "error" not in tasks
        except:
            print(f"Output_user Error: {e} \n{output_task}")
            return # Stop if no tasks are generated
        user_key = f"user_{id_user}"
        outputs[user_key]["user_info"] = user
        outputs[user_key]["tasks_info"] = tasks
        outputs["prompt"][f"synthesizing_tasks_for_user_{id_user}"] = prompt_task
        outputs["cost"][f"synthesizing_tasks_for_user_{id_user}"] = output_task["cost"]
        print(f"Task synthesis completed for user {id_user}. Cost ${output_task["cost"]}")

        # --- Step 3: Evaluate the quality of generated tasks ---
        outputs[user_key]["validation_results"] = {} 
        eval_key = f"evaluating_tasks_for_user_{id_user}"
        
        for id_task, task in enumerate(tasks, start=1):
            print(f"Evaluating user {id_user} - task {id_task} - {task.get('task_name', 'Unnamed')} ...")
            prompt_eval = prompt_temp_eval.render(
                dataset_info=json.dumps(dataset_info),
                user_info=json.dumps(user_info),
                task_info=json.dumps(task)
            )
            output_eval = llm_evaluator.generate(prompt_eval)
            
            try:
                eval_result = extract_json(output_eval["output"])
                outputs[user_key]["validation_results"][f"task_{id_task}"] = eval_result
            except Exception as e:
                print(f"Evaluation Error for {task.get('task_name')}: {e}")
            
            # Increment cost rather than overwriting
            outputs["prompt"][eval_key] = prompt_eval
            outputs["cost"][eval_key] += output_eval.get("cost", 0)
            print(f"Evaluation completed for user {id_user}. Cost ${output_eval["cost"]}")
           
         

    # --- Save the outputs ---
    output_dir = os.path.join(args.dir_data_synthesized, args.dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    save_path = os.path.join(output_dir, "synthesized_output.json")
    with open(save_path, 'w') as f_out:
        json.dump(outputs, f_out, indent=4)
    
    print(f"Data synthesis complete! Results saved to: {save_path}")


if __name__ == "__main__":
    args = parse_args()
    synthesize_data(args)
