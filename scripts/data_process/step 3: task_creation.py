import os
import json
import pandas as pd
from jinja2 import Environment, FileSystemLoader

dir_data_analysis = "results/synthesized_data"
dir_prompt_template = "prompts/data_process/"
dir_data = "data/data_processed"
fname_prompt_template = "step 3: task_creation.jinja"
NUM_USERS = 8
DATA_SAMPLE_SIZE = 2
difficulty = "Medium"

env = Environment(loader=FileSystemLoader(dir_prompt_template))
prompt_template = env.get_template(fname_prompt_template)


dataset_names = [
    'ccdv/patent-classification',
    'alexfabbri/multi_news',
    'kritsadaK/EDGAR-CORPUS-Financial-Summarization',
]
dataset_name = 'kritsadaK/EDGAR-CORPUS-Financial-Summarization'
with open(os.path.join(dir_data_analysis, dataset_name, 'data_analysis.json')) as f:
    dataset_info = json.load(f)
with open(os.path.join(dir_data_analysis, dataset_name, 'synthesized_users.json')) as f:
    user_infos = json.load(f)
user_info = user_infos[0]
df = pd.read_csv(os.path.join(dir_data, dataset_name, "sampled_data.csv"))
sampled_data = json.dumps([row.to_dict() for i, row in df.sample(DATA_SAMPLE_SIZE).iterrows()])


prompt = prompt_template.render(
    dataset_name = dataset_name,
    user_info = user_info,
    dataset_info = dataset_info,
    num_users = NUM_USERS
)
print(prompt)

