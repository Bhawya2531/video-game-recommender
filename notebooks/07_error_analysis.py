import pandas as pd
import numpy as np
import pickle
import time
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder

t0 = time.time()
train = pd.read_parquet("/home/claude/train.parquet")
test = pd.read_parquet("/home/claude/test.parquet")

with open("/home/claude/models/svd_k200.pkl", "rb") as f:
    saved = pickle.load(f)
U, Vt = saved["U"], saved["Vt"]

user_enc = LabelEncoder().fit(train["user_id"])
item_enc = LabelEncoder().fit(train["item_id"])
n_users, n_items = len(user_enc.classes_), len(item_enc.classes_)

train_u = user_enc.transform(train["user_id"])
train_i = item_enc.transform(train["item_id"])
R = csr_matrix((np.ones(len(train), dtype=np.float32), (train_u, train_i)), shape=(n_users, n_items))

user_train_counts = train["user_id"].value_counts()
item_train_counts = train["item_id"].value_counts()
median_pop = item_train_counts.median()

class_to_idx = {c: i for i, c in enumerate(item_enc.classes_)}
K = 10

test_eval = test[test["user_id"].isin(user_enc.classes_)].reset_index(drop=True).copy()
test_eval["train_count"] = test_eval["user_id"].map(user_train_counts)
test_eval["is_cold_item"] = ~test_eval["item_id"].isin(item_enc.classes_)
test_eval["item_pop"] = test_eval["item_id"].map(item_train_counts).fillna(0)
test_eval["is_longtail"] = test_eval["item_pop"] <= median_pop

u_idx_all = user_enc.transform(test_eval["user_id"].values)
items_all = test_eval["item_id"].values
item_known_mask = ~test_eval["is_cold_item"].values

hits = np.zeros(len(test_eval), dtype=np.int8)
chunk_size = 4000
for start in range(0, len(test_eval), chunk_size):
    end = min(start + chunk_size, len(test_eval))
    u_idx_chunk = u_idx_all[start:end]
    U_chunk = U[u_idx_chunk]
    scores = U_chunk @ Vt
    R_chunk = R[u_idx_chunk].toarray()
    scores[R_chunk > 0] = -np.inf

    top_idx = np.argpartition(-scores, K - 1, axis=1)[:, :K]

    for row_i in range(end - start):
        gi = start + row_i
        if not item_known_mask[gi]:
            continue
        true_idx = class_to_idx[items_all[gi]]
        hits[gi] = 1 if true_idx in top_idx[row_i] else 0
    print(f"  chunk {start}-{end}/{len(test_eval)}  ({time.time()-t0:.1f}s)")

test_eval["hit@10"] = hits
print(f"\nOverall Recall@10 (test, k=200): {test_eval['hit@10'].mean():.4f}")

cold = test_eval[test_eval["is_cold_item"]]
warm = test_eval[~test_eval["is_cold_item"]]
print("\n=== Cold-start items ===")
print(f"Cold-start: {len(cold)} ({len(cold)/len(test_eval)*100:.2f}%) -- Recall@10 = {cold['hit@10'].mean() if len(cold) else float('nan'):.4f}")
print(f"Warm: {len(warm)} -- Recall@10 = {warm['hit@10'].mean():.4f}")

sparse_users = test_eval[test_eval["train_count"] <= 3]
active_users = test_eval[test_eval["train_count"] > 3]
print("\n=== User sparsity ===")
print(f"Sparse users (<=3 train interactions): {len(sparse_users)} -- Recall@10 = {sparse_users['hit@10'].mean():.4f}")
print(f"Active users (>3 train interactions):  {len(active_users)} -- Recall@10 = {active_users['hit@10'].mean():.4f}")

popular = test_eval[~test_eval["is_longtail"]]
longtail = test_eval[test_eval["is_longtail"]]
print("\n=== Popular vs long-tail items ===")
print(f"Popular items: {len(popular)} -- Recall@10 = {popular['hit@10'].mean():.4f}")
print(f"Long-tail items: {len(longtail)} -- Recall@10 = {longtail['hit@10'].mean():.4f}")

summary = pd.DataFrame([
    {"segment": "Overall", "n": len(test_eval), "recall@10": test_eval["hit@10"].mean()},
    {"segment": "Cold-start items", "n": len(cold), "recall@10": cold["hit@10"].mean() if len(cold) else np.nan},
    {"segment": "Warm items", "n": len(warm), "recall@10": warm["hit@10"].mean()},
    {"segment": "Sparse users (<=3)", "n": len(sparse_users), "recall@10": sparse_users["hit@10"].mean()},
    {"segment": "Active users (>3)", "n": len(active_users), "recall@10": active_users["hit@10"].mean()},
    {"segment": "Popular items", "n": len(popular), "recall@10": popular["hit@10"].mean()},
    {"segment": "Long-tail items", "n": len(longtail), "recall@10": longtail["hit@10"].mean()},
])
summary.to_csv("/home/claude/results_error_analysis.csv", index=False)
print(f"\nSaved results_error_analysis.csv  ({time.time()-t0:.1f}s)")
print(summary.to_string(index=False))
