import pandas as pd
import numpy as np

train = pd.read_parquet("/home/claude/train.parquet")
val = pd.read_parquet("/home/claude/val.parquet")
test = pd.read_parquet("/home/claude/test.parquet")

K_VALUES = [5, 10, 20]

# Build per-user "seen in train" set (to exclude already-seen items from recs)
train_seen = train.groupby("user_id")["item_id"].apply(set).to_dict()

# Popularity ranking: item interaction count in train
item_popularity = train["item_id"].value_counts()
popular_items_ranked = item_popularity.index.tolist()  # most popular first

def recommend_popularity(user_id, k):
    """Top-k most popular items the user hasn't seen in train."""
    seen = train_seen.get(user_id, set())
    recs = []
    for item in popular_items_ranked:
        if item not in seen:
            recs.append(item)
            if len(recs) == k:
                break
    return recs

def recall_at_k(recs, relevant_item):
    return 1.0 if relevant_item in recs else 0.0

def ndcg_at_k(recs, relevant_item):
    if relevant_item in recs:
        rank = recs.index(relevant_item) + 1  # 1-indexed
        return 1.0 / np.log2(rank + 1)
    return 0.0

def evaluate(eval_df, k_values):
    results = {k: {"recall": [], "ndcg": []} for k in k_values}
    max_k = max(k_values)

    for _, row in eval_df.iterrows():
        user_id = row["user_id"]
        true_item = row["item_id"]
        recs_max = recommend_popularity(user_id, max_k)

        for k in k_values:
            recs_k = recs_max[:k]
            results[k]["recall"].append(recall_at_k(recs_k, true_item))
            results[k]["ndcg"].append(ndcg_at_k(recs_k, true_item))

    summary = {}
    for k in k_values:
        summary[k] = {
            "Recall@K": np.mean(results[k]["recall"]),
            "NDCG@K": np.mean(results[k]["ndcg"]),
        }
    return summary

print("Evaluating popularity baseline on validation set...")
val_results = evaluate(val, K_VALUES)
for k, metrics in val_results.items():
    print(f"  K={k}: Recall@{k}={metrics['Recall@K']:.4f}, NDCG@{k}={metrics['NDCG@K']:.4f}")

print("\nEvaluating popularity baseline on test set...")
test_results = evaluate(test, K_VALUES)
for k, metrics in test_results.items():
    print(f"  K={k}: Recall@{k}={metrics['Recall@K']:.4f}, NDCG@{k}={metrics['NDCG@K']:.4f}")

# Save results for later comparison across models
results_df = pd.DataFrame([
    {"model": "Popularity", "split": "val", "k": k, **m}
    for k, m in val_results.items()
] + [
    {"model": "Popularity", "split": "test", "k": k, **m}
    for k, m in test_results.items()
])
results_df.to_csv("/home/claude/results_popularity.csv", index=False)
print("\nSaved results_popularity.csv")
