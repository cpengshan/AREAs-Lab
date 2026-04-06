import os
from datasets import load_dataset
from pathlib import Path
import pandas as pd

data_dir = "data/data_raw"
dataset_names = [
    # "ccdv/govreport-summarization", 
    # "ccdv/pubmed-summarization", 
    # "ccdv/arxiv-summarization", 
    # "ccdv/patent-classification", 
    # "rohitsaxena/MovieSum", 
    # "thu-coai/esconv", 
    # "alexfabbri/multi_news", 
    # "kritsadaK/EDGAR-CORPUS-Financial-Summarization", 
    # "santoshtyss/uk_legislation", 
]

def download_dataset(dataset_name, split="test"):
    dir_out = os.path.join(data_dir, dataset_name)
    path = Path(dir_out)
    path.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)
    path_out = os.path.join(dir_out, "sampled_data.csv")
    dataset.to_csv(path_out)
    print(f"Dataset {dataset_name} ({split} set) downloaded to {path_out}")


for dataset_name in dataset_names:
    try:
        download_dataset(dataset_name, 'test')
    except:
        download_dataset(dataset_name, 'train')



# Three individual datasets that can not be directly downloaded
# We process them from downloaded file
# ccdv/mediasum
dataset_name = "ccdv/mediasum"
path_in = "data/data_raw/ccdv/mediasum/test_data.txt"
path_out = "data/data_processed/ccdv/mediasum/sampled_data.csv"
with open(path_in) as f:
    lines = f.readlines()
outputs = [eval(line) for line in lines]
df_out = pd.DataFrame(outputs)
df_out.to_csv(path_out, index=False)


# alexfabbri/multi_news




# HuggingFaceFW/fineweb-edu