import pandas as pd
import gzip
import json

DATA_PATH = "/mnt/user-data/uploads/Video_Games_5_json.gz"
MIN_INTERACTIONS = 5

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
        })
    return pd.DataFrame(rows)

print("Loading raw data...")
df = get_df(DATA_PATH)
print(f"Raw rows: {len(df):,}")

# 1. Dedup: keep most recent interaction per (user, item)
df = df.sort_values("unix_time").drop_duplicates(subset=["user_id", "item_id"], keep="last")
print(f"After dedup: {len(df):,}")

# 2. Iterative k-core filtering to guarantee every user/item has >= MIN_INTERACTIONS
round_num = 0
while True:
    round_num += 1
    n_before = len(df)

    user_counts = df["user_id"].value_counts()
    valid_users = user_counts[user_counts >= MIN_INTERACTIONS].index
    df = df[df["user_id"].isin(valid_users)]

    item_counts = df["item_id"].value_counts()
    valid_items = item_counts[item_counts >= MIN_INTERACTIONS].index
    df = df[df["item_id"].isin(valid_items)]

    n_after = len(df)
    print(f"  k-core round {round_num}: {n_before:,} -> {n_after:,} rows")
    if n_after == n_before:
        break

print(f"\nFinal filtered rows: {len(df):,}")
print(f"Final unique users: {df['user_id'].nunique():,}")
print(f"Final unique items: {df['item_id'].nunique():,}")

# Sanity check
final_user_min = df.groupby("user_id").size().min()
final_item_min = df.groupby("item_id").size().min()
print(f"Verified min interactions/user: {final_user_min}")
print(f"Verified min interactions/item: {final_item_min}")
assert final_user_min >= MIN_INTERACTIONS
assert final_item_min >= MIN_INTERACTIONS
print("5-core property verified.\n")

# 3. Temporal split: sort each user's interactions by time,
#    last interaction -> test, second-to-last -> val, rest -> train
#    (standard "leave-last-out" protocol for top-K recommendation eval)
df = df.sort_values(["user_id", "unix_time"])
df["rank_desc"] = df.groupby("user_id").cumcount(ascending=False)  # 0 = most recent

test = df[df["rank_desc"] == 0].copy()
val = df[df["rank_desc"] == 1].copy()
train = df[df["rank_desc"] >= 2].copy()

print(f"Train: {len(train):,} rows")
print(f"Val:   {len(val):,} rows")
print(f"Test:  {len(test):,} rows")

# Guard against leakage: val/test users/items must exist in train
# (users with exactly 5 interactions: train has 3, val 1, test 1 -- all present, fine)
train_users = set(train["user_id"])
train_items = set(train["item_id"])

val_leak_users = set(val["user_id"]) - train_users
test_leak_users = set(test["user_id"]) - train_users
print(f"\nVal users not in train: {len(val_leak_users)}")
print(f"Test users not in train: {len(test_leak_users)}")

val_cold_items = set(val["item_id"]) - train_items
test_cold_items = set(test["item_id"]) - train_items
print(f"Val items not in train (cold-start items): {len(val_cold_items)}")
print(f"Test items not in train (cold-start items): {len(test_cold_items)}")

# Save
train.drop(columns=["rank_desc"]).to_parquet("/home/claude/train.parquet", index=False)
val.drop(columns=["rank_desc"]).to_parquet("/home/claude/val.parquet", index=False)
test.drop(columns=["rank_desc"]).to_parquet("/home/claude/test.parquet", index=False)
df.drop(columns=["rank_desc"]).to_parquet("/home/claude/full_filtered.parquet", index=False)
print("\nSaved train.parquet, val.parquet, test.parquet, full_filtered.parquet")
