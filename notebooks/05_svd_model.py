import pandas as pd
import numpy as np
import time
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD

t0 = time.time()

train = pd.read_parquet("/home/claude/train.parquet")
val = pd.read_parquet("/home/claude/val.parquet")
test = pd.read_parquet("/home/claude/test.parquet")

K_VALUES = [5, 10, 20]
MAX_K = max(K_VALUES)

user_enc = LabelEncoder().fit(train["user_id"])
item_enc = LabelEncoder().fit(train["item_id"])
n_users, n_items = len(user_enc.classes_), len(item_enc.classes_)

train_u = user_enc.transform(train["user_id"])
train_i = item_enc.transform(train["item_id"])
data = np.ones(len(train), dtype=np.float32)
R = csr_matrix((data, (train_u, train_i)), shape=(n_users, n_items))
print(f"R: {R.shape}, nnz={R.nnz:,}  ({time.time()-t0:.1f}s)")


def train_svd(n_components):
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    U = svd.fit_transform(R)          # (n_users, k) -- already scaled by singular values
    Vt = svd.components_              # (k, n_items)
    return svd, U, Vt


def eval_svd(U, Vt, eval_df, chunk_size=4000):
    eval_df = eval_df[eval_df["user_id"].isin(user_enc.classes_)].reset_index(drop=True)
    eval_users = eval_df["user_id"].values
    eval_items = eval_df["item_id"].values
    u_idx_all = user_enc.transform(eval_users)
    item_known_mask = np.isin(eval_items, item_enc.classes_)
    item_classes = item_enc.classes_
    class_to_idx = {c: i for i, c in enumerate(item_classes)}

    n_eval = len(eval_df)
    recalls = {k: np.zeros(n_eval) for k in K_VALUES}
    ndcgs = {k: np.zeros(n_eval) for k in K_VALUES}

    for start in range(0, n_eval, chunk_size):
        end = min(start + chunk_size, n_eval)
        u_idx_chunk = u_idx_all[start:end]

        U_chunk = U[u_idx_chunk]               # (chunk, k)
        scores = U_chunk @ Vt                  # (chunk, n_items)

        R_chunk = R[u_idx_chunk].toarray()
        scores[R_chunk > 0] = -np.inf          # exclude seen train items

        top_idx = np.argpartition(-scores, MAX_K - 1, axis=1)[:, :MAX_K]
        row_scores = np.take_along_axis(scores, top_idx, axis=1)
        order = np.argsort(-row_scores, axis=1)
        top_idx_sorted = np.take_along_axis(top_idx, order, axis=1)

        for row_i in range(end - start):
            gi = start + row_i
            if not item_known_mask[gi]:
                continue
            true_idx = class_to_idx[eval_items[gi]]
            pos = np.where(top_idx_sorted[row_i] == true_idx)[0]
            for k in K_VALUES:
                in_topk = len(pos) > 0 and pos[0] < k
                recalls[k][gi] = 1.0 if in_topk else 0.0
                if in_topk:
                    ndcgs[k][gi] = 1.0 / np.log2(pos[0] + 2)

    return {k: {"Recall@K": recalls[k].mean(), "NDCG@K": ndcgs[k].mean()} for k in K_VALUES}


# Single quick run at k=50 to sanity-check the pipeline before the full ablation (Phase 4)
print("\nSanity-check SVD run at k=50...")
svd, U, Vt = train_svd(50)
print(f"SVD trained  ({time.time()-t0:.1f}s)")

val_results = eval_svd(U, Vt, val)
print("Validation results (k=50):")
for k, m in val_results.items():
    print(f"  K={k}: Recall@{k}={m['Recall@K']:.4f}, NDCG@{k}={m['NDCG@K']:.4f}")

test_results = eval_svd(U, Vt, test)
print("Test results (k=50):")
for k, m in test_results.items():
    print(f"  K={k}: Recall@{k}={m['Recall@K']:.4f}, NDCG@{k}={m['NDCG@K']:.4f}")

print(f"\nDone  ({time.time()-t0:.1f}s)")
