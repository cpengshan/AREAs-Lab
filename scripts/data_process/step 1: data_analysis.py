from jinja2 import Environment, FileSystemLoader
from pathlib import Path
import os
from json_repair import repair_json
import json

dataset_names = [
    'HuggingFaceFW/fineweb-edu',
    'alexfabbri/multi_news',
    'ccdv/arxiv-summarization',
    'ccdv/govreport-summarization',
    'ccdv/mediasum',
    'ccdv/patent-classification',
    'ccdv/pubmed-summarization',
    'kritsadaK/EDGAR-CORPUS-Financial-Summarization',
    'santoshtyss/uk_legislation',
    'thu-coai/esconv',
    'rohitsaxena/MovieSum',
    'starmpcc/Asclepius-Synthetic-Clinical-Notes'
]

dir_prompt_template = "prompts/data_process/"
fname_prompt_template = "step 1: data_analysis.jinja"

env = Environment(loader=FileSystemLoader(dir_prompt_template))
prompt_template = env.get_template(fname_prompt_template)
dir_base = "data/data_synthesized"  


dataset_name = 'ccdv/pubmed-summarization'
path = Path(os.path.join(dir_base, dataset_name))
path.mkdir(parents=True, exist_ok=True)



prompt = prompt_template.render(dataset_name=dataset_name)
print(prompt)

output = """""".replace("true", "True").replace("false", "False").replace("null", 'None')
print(output)
with open(os.path.join(dir_base, dataset_name, 'data_analysis.json'), 'w') as f:
    json.dump(output, f)


