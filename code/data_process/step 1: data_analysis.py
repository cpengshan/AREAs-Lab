from jinja2 import Environment, FileSystemLoader


dataset_names = [
    'HuggingFaceFW/fineweb-edu',
    'alexfabbri/multi_news',
    'ccdv/arxiv-summarization',
    'ccdv/govreport-summarization',
    'ccdv/mediasumrohitsaxena/MovieSum',
    'ccdv/patent-classification',
    'ccdv/pubmed-summarization',
    'kritsadaK/EDGAR-CORPUS-Financial-Summarization',
    'santoshtyss/uk_legislation',
    'thu-coai/esconv',
    'rohitsaxena/MovieSum',
    'starmpcc/Asclepius-Synthetic-Clinical-Notes'
]

dir_prompt_template = "prompts/"
fname_prompt_template = "step 1: data_analysis.jinja"

env = Environment(loader=FileSystemLoader(dir_prompt_template))
prompt_template = env.get_template(fname_prompt_template)

dataset_name = 'alexfabbri/multi_news'
prompt = prompt_template.render(dataset_name=dataset_name)
print(prompt)