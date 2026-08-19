import pandas as pd
import numpy as np
import time
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder, normalize

t0 = time.time()

train = pd.read_parquet("/home/claude/train.parquet")
val = pd.read_parquet("/home/claude/val.parquet")
test = pd.read_parquet("/home/claude/test.parquet")

K_VALUES = [5, 10, 20]
MAX_K = max(K_VALUES)

user_enc = LabelEncoder().fit(train["user_id"])
item_enc = LabelEncoder().fit(train["item_id"])
n_users, n_items = len(user_enc.classes_), len(item_enc.classes_)
print(f"Train users: {n_users}, Train items: {n_items}  ({time.time()-t0:.1f}s)")

train_u = user_enc.transform(train["user_id"])
train_i = item_enc.transform(train["item_id"])
data = np.ones(len(train), dtype=np.float32)
R = csr_matrix((data, (train_u, train_i)), shape=(n_users, n_items))

R_norm = normalize(R, axis=0)
item_sim = (R_norm.T @ R_norm).tocsr()
print(f"item_sim built: {item_sim.shape}, nnz={item_sim.nnz:,}  ({time.time()-t0:.1f}s)")

def eval_chunked(eval_df, chunk_size=2000):
    eval_df = eval_df[eval_df["user_id"].isin(user_enc.classes_)].reset_index(drop=True)
    eval_users = eval_df["user_id"].values
    eval_items = eval_df["item_id"].values
    u_idx_all = user_enc.transform(eval_users)
    item_known_mask = np.isin(eval_items, item_enc.classes_)

    n_eval = len(eval_df)
    recalls = {k: np.zeros(n_eval) for k in K_VALUES}
    ndcgs = {k: np.zeros(n_eval) for k in K_VALUES}

    item_classes = item_enc.classes_
    class_to_idx = {c: i for i, c in enumerate(item_classes)}

    for start in range(0, n_eval, chunk_size):
        end = min(start + chunk_size, n_eval)
        u_idx_chunk = u_idx_all[start:end]

        R_chunk = R[u_idx_chunk]
        scores = (R_chunk @ item_sim).toarray()
        seen_mask = R_chunk.toarray() > 0
        scores[seen_mask] = -np.inf

        top_idx = np.argpartition(-scores, MAX_K - 1, axis=1)[:, :MAX_K]
        row_scores = np.take_along_axis(scores, top_idx, axis=1)
        order = np.argsort(-row_scores, axis=1)
        top_idx_sorted = np.take_along_axis(top_idx, order, axis=1)

        for row_i in range(end - start):
            global_i = start + row_i
            if not item_known_mask[global_i]:
                continue
            true_item_idx = class_to_idx[eval_items[global_i]]
            ranked = top_idx_sorted[row_i]
            pos = np.where(ranked == true_item_idx)[0]
            for k in K_VALUES:
                in_topk = len(pos) > 0 and pos[0] < k
                recalls[k][global_i] = 1.0 if in_topk else 0.0
                if in_topk:
                    ndcgs[k][global_i] = 1.0 / np.log2(pos[0] + 2)

        print(f"    chunk {start}-{end}/{n_eval}  ({time.time()-t0:.1f}s)")

    summary = {}
    for k in K_VALUES:
        summary[k] = {"Recall@K": recalls[k].mean(), "NDCG@K": ndcgs[k].mean()}
    return summary

print("\nEvaluating item-CF on validation set...")
val_results = eval_chunked(val)
for k, m in val_results.items():
    print(f"  K={k}: Recall@{k}={m['Recall@K']:.4f}, NDCG@{k}={m['NDCG@K']:.4f}")

print("\nEvaluating item-CF on test set...")
test_results = eval_chunked(test)
for k, m in test_results.items():
    print(f"  K={k}: Recall@{k}={m['Recall@K']:.4f}, NDCG@{k}={m['NDCG@K']:.4f}")

results_df = pd.DataFrame(
    [{"model": "ItemCF", "split": "val", "k": k, **m} for k, m in val_results.items()]
    + [{"model": "ItemCF", "split": "test", "k": k, **m} for k, m in test_results.items()]
)
results_df.to_csv("/home/claude/results_itemcf.csv", index=False)
print(f"\nSaved results_itemcf.csv  (total {time.time()-t0:.1f}s)")
