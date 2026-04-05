import os
import json
from jinja2 import Environment, FileSystemLoader

dir_data_analysis = "results/data_analysis"
dir_prompt_template = "prompts/data_process/"
fname_prompt_template = "step 2: user_creation.jinja"
NUM_USERS = 8

env = Environment(loader=FileSystemLoader(dir_prompt_template))
prompt_template = env.get_template(fname_prompt_template)


dataset_names = [
    'ccdv/patent-classification',
    'alexfabbri/multi_news',
    'kritsadaK/EDGAR-CORPUS-Financial-Summarization',
]

dataset_name = 'ccdv/patent-classification'
with open(os.path.join(dir_data_analysis, dataset_name, 'analysis.json')) as f:
    dataset_info = json.load(f)
prompt = prompt_template.render(
    dataset_name = dataset_name,
    dataset_info = dataset_info,
    num_users = NUM_USERS
)


