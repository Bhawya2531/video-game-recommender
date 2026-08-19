import pandas as pd
import gzip
import json
from datetime import datetime

DATA_PATH = "/mnt/user-data/uploads/Video_Games_5_json.gz"

def parse(path):
    with gzip.open(path, 'rb') as g:
        for l in g:
            yield json.loads(l)

def get_df(path):
    rows = []
    for d in parse(path):
        rows.append({
            "user_id": d.get("reviewerID"),
            "item_id": d.get("asin"),
            "rating": d.get("overall"),
            "unix_time": d.get("unixReviewTime"),
            "verified": d.get("verified"),
        })
    return pd.DataFrame(rows)

print("Loading data...")
df = get_df(DATA_PATH)
print(f"Loaded {len(df):,} rows\n")

print("=== Basic info ===")
print(df.info())
print()

print("=== Nulls ===")
print(df.isnull().sum())
print()

print("=== Duplicates (full row) ===")
print(df.duplicated().sum())
print()

print("=== Duplicate (user_id, item_id) pairs ===")
dup_ui = df.duplicated(subset=["user_id", "item_id"]).sum()
print(f"{dup_ui:,} duplicate (user,item) pairs out of {len(df):,}")
print()

print("=== Unique users / items ===")
n_users = df["user_id"].nunique()
n_items = df["item_id"].nunique()
print(f"Users: {n_users:,}")
print(f"Items: {n_items:,}")
print(f"Density: {len(df) / (n_users * n_items):.6f}")
print()

print("=== Rating distribution ===")
print(df["rating"].value_counts().sort_index())
print()

print("=== Interactions per user ===")
upu = df.groupby("user_id").size()
print(upu.describe())
print(f"Min interactions/user: {upu.min()}")
print()

print("=== Interactions per item ===")
ipi = df.groupby("item_id").size()
print(ipi.describe())
print(f"Min interactions/item: {ipi.min()}")
print()

print("=== Time range ===")
df["review_date"] = pd.to_datetime(df["unix_time"], unit="s")
print(f"Earliest: {df['review_date'].min()}")
print(f"Latest: {df['review_date'].max()}")
print()

print("=== Verified purchases ===")
print(df["verified"].value_counts(normalize=True))

df.to_parquet("/home/claude/video_games_raw.parquet", index=False)
print("\nSaved to video_games_raw.parquet")
