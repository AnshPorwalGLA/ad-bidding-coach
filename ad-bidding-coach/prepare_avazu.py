# prepare_avazu.py
import os
import pandas as pd
import numpy as np

INPUT_CANDIDATES = ["train.csv", "train.gz", "train.csv.gz"]

def find_input_file():
    for name in INPUT_CANDIDATES:
        if os.path.exists(name):
            return name
    # try any file starting with "train"
    for f in os.listdir("."):
        if f.startswith("train"):
            return f
    return None

def prepare_avazu(input_file=None, out_file="ad_data.csv", nrows=300000):
    if input_file is None:
        input_file = find_input_file()
    if input_file is None:
        raise FileNotFoundError("No Avazu train file found. Place train.csv or train.gz in project root.")

    print(f"Loading {input_file} (may take a while)...")
    # pandas will detect gzip by extension automatically if compression=None
    df = pd.read_csv(input_file, nrows=nrows, low_memory=False)
    print("Raw data shape:", df.shape)

    # Ensure `click` column exists
    if "click" not in df.columns:
        raise ValueError("Expected column `click` in Avazu dataset.")

    # Basic RL-ready columns:
    df = df.reset_index(drop=True)
    df["impressions"] = 1
    df["ctr"] = df["click"]  # click is 0/1; for per-row CTR this is enough

    # Simulate CVR (since Avazu lacks conversions). Use higher CVR for clicked rows.
    df["cvr"] = df["click"].apply(lambda x: 0.20 if x == 1 else 0.02)

    # Simulate cost per impression (CPM/1000 style but simple)
    rng = np.random.default_rng(42)
    df["cost"] = rng.uniform(0.5, 2.0, size=len(df))  # cost in currency units per impression

    # Simulate revenue per conversion if click==1 else 0
    # Choose random revenue between 10 and 40 for clicked rows
    revenues = rng.uniform(10, 40, size=len(df))
    df["revenue"] = df["click"] * revenues

    # Normalize CTR/CVR to 0-1 (already are)
    out = df[["ctr", "cvr", "impressions", "cost", "revenue"]].copy()
    out.to_csv(out_file, index=False)
    print(f"Saved RL-ready dataset to {out_file}. Shape: {out.shape}")
    return out

if __name__ == "__main__":
    prepare_avazu(nrows=200000)  # adjust nrows to size you can handle
