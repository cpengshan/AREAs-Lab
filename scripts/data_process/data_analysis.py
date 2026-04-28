import json
from pathlib import Path
import pandas as pd

path_data = "data/data_synthesized"
base_dir = Path(path_data)

# List to store the loaded JSON content
outputs = []
users = []
tasks = []

# glob("*/*/") ensures we are looking exactly two folders deep 
# under path_data for the specific file.
for file_path in base_dir.glob("*/*/synthesized_output.json"):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            output = json.load(file)
            outputs.append(output)
            print(f"Successfully loaded: {file_path}")
            for k in output:
                if k.startswith('user'):
                    users.append(output[k]['user_info'])
                    assert len(output[k]['tasks_info']) == 2
                    task_info = output[k]['tasks_info']
                    validation = output[k]['validation_results']
                    task1 = task_info[0]
                    task2 = task_info[1]
                    task1.update(validation['task_1'])
                    task2.update(validation['task_2'])
                    tasks.append(task1)
                    tasks.append(task2)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {file_path}")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

df_u = pd.DataFrame(users)
df_u.to_csv("data/za/synthesized_users.csv", index=False)
df_t = pd.DataFrame(tasks)
df_t.to_csv("data/za/synthesized_tasks.csv", index=False)