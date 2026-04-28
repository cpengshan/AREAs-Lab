import pandas as pd
import json
from pathlib import Path

# Define base directories
dir_base = Path("data/data_processed")
out_base = Path("data/data_sampled")

RANDOM_SEED = 42 

# Variables to track global statistics
global_total_words = 0
global_total_records = 0
SAMPLE_SIZE = 20


def truncate_df_columns(df: pd.DataFrame, max_len: int) -> pd.DataFrame:
    # Create a copy to avoid modifying the original DataFrame in place
    truncated_df = df.copy()
    # Identify columns that contain string or object data types
    text_cols = truncated_df.select_dtypes(include=['object', 'string']).columns
    for col in text_cols:
        # Use the Pandas string accessor (.str) to safely slice the strings
        # This automatically handles NaN values without throwing errors
        truncated_df[col] = truncated_df[col].str[:max_len]
    return truncated_df



for file_path in dir_base.glob("*/*/data_processed.csv"):
    df = pd.read_csv(file_path)
    
    if len(df) < 100:
        print(f"⚠️ Skipping {file_path}: Only {len(df)} rows.")
        continue
        
    sampled_df = df.sample(n=100, random_state=RANDOM_SEED)
    defining_df = sampled_df.iloc[:SAMPLE_SIZE]
    non_defining_df = sampled_df.iloc[SAMPLE_SIZE:]
    
    # --- WORD COUNT LOGIC ---
    text_cols = sampled_df.select_dtypes(include=['object', 'string'])
    
    file_avg_words = 0  # Default to 0 if there are no text columns
    
    if not text_cols.empty:
        # Count words per row
        word_counts = text_cols.fillna('').astype(str).apply(lambda x: ' '.join(x), axis=1).str.split().str.len()
        
        # 1. Calculate the average for THIS specific file
        file_avg_words = word_counts.mean()
        
        # 2. Add to our global tracking variables
        global_total_words += word_counts.sum()
        global_total_records += len(sampled_df)
    # ------------------------

    output_dict = {
        "defining_instances": defining_df.to_dict(orient="records"),
        "non_defining_instances": non_defining_df.to_dict(orient="records")
    }
    
    rel_path = file_path.relative_to(dir_base)
    output_path = out_base / rel_path
    output_json_path = output_path.with_suffix('.json')
    
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_json_path.open("w", encoding="utf-8") as f:
        json.dump(output_dict, f, ensure_ascii=False, indent=4)
        
    # Print the success message ALONG with the specific file's average length
    print(f"✅ Saved: {output_json_path} | Avg length: {file_avg_words:.1f} words/record")
