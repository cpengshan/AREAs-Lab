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

dir_prompt = "prompts/"
fname_prompt_template = "step 1: data_analysis.jinja"

env = Environment(loader=FileSystemLoader(dir_prompt))
prompt_template = env.get_template(fname_prompt_template)

