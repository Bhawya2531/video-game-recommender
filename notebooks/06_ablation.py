import pandas as pd
import numpy as np
import time
import pickle
import os
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD

t0 = time.time()

train = pd.read_parquet("/home/claude/train.parquet")
val = pd.read_parquet("/home/claude/val.parquet")
test = pd.read_parquet("/home/claude/test.parquet")

K_VALUES = [5, 10, 20]
MAX_K = max(K_VALUES)
LATENT_DIMS = [20, 50, 100, 200]

user_enc = LabelEncoder().fit(train["user_id"])
item_enc = LabelEncoder().fit(train["item_id"])
n_users, n_items = len(user_enc.classes_), len(item_enc.classes_)

train_u = user_enc.transform(train["user_id"])
train_i = item_enc.transform(train["item_id"])
data = np.ones(len(train), dtype=np.float32)
R = csr_matrix((data, (train_u, train_i)), shape=(n_users, n_items))


def eval_svd(U, Vt, eval_df, chunk_size=4000):
    eval_df = eval_df[eval_df["user_id"].isin(user_enc.classes_)].reset_index(drop=True)
    eval_users = eval_df["user_id"].values
    eval_items = eval_df["item_id"].values
    u_idx_all = user_enc.transform(eval_users)
    item_known_mask = np.isin(eval_items, item_enc.classes_)
    class_to_idx = {c: i for i, c in enumerate(item_enc.classes_)}

    n_eval = len(eval_df)
    recalls = {k: np.zeros(n_eval) for k in K_VALUES}
    ndcgs = {k: np.zeros(n_eval) for k in K_VALUES}

    for start in range(0, n_eval, chunk_size):
        end = min(start + chunk_size, n_eval)
        u_idx_chunk = u_idx_all[start:end]
        U_chunk = U[u_idx_chunk]
        scores = U_chunk @ Vt
        R_chunk = R[u_idx_chunk].toarray()
        scores[R_chunk > 0] = -np.inf

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


os.makedirs("/home/claude/models", exist_ok=True)
ablation_rows = []

for k_dim in LATENT_DIMS:
    t_start = time.time()
    svd = TruncatedSVD(n_components=k_dim, random_state=42)
    U = svd.fit_transform(R)
    Vt = svd.components_
    train_time = time.time() - t_start

    t_inf_start = time.time()
    val_results = eval_svd(U, Vt, val)
    test_results = eval_svd(U, Vt, test)
    inf_time = time.time() - t_inf_start
    n_val = val[val["user_id"].isin(user_enc.classes_)].shape[0]
    latency_per_user_ms = (inf_time / max(n_val, 1)) * 1000

    # model size on disk
    model_path = f"/home/claude/models/svd_k{k_dim}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"svd": svd, "U": U, "Vt": Vt}, f)
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)

    row = {
        "k": k_dim,
        "train_time_s": round(train_time, 2),
        "eval_latency_ms_per_user": round(latency_per_user_ms, 4),
        "model_size_mb": round(model_size_mb, 2),
        "val_recall@10": round(val_results[10]["Recall@K"], 4),
        "val_ndcg@10": round(val_results[10]["NDCG@K"], 4),
        "test_recall@5": round(test_results[5]["Recall@K"], 4),
        "test_ndcg@5": round(test_results[5]["NDCG@K"], 4),
        "test_recall@10": round(test_results[10]["Recall@K"], 4),
        "test_ndcg@10": round(test_results[10]["NDCG@K"], 4),
        "test_recall@20": round(test_results[20]["Recall@K"], 4),
        "test_ndcg@20": round(test_results[20]["NDCG@K"], 4),
    }
    ablation_rows.append(row)
    print(f"k={k_dim}: train={train_time:.1f}s | test Recall@10={row['test_recall@10']:.4f} "
          f"NDCG@10={row['test_ndcg@10']:.4f} | model={model_size_mb:.1f}MB | "
          f"latency/user={latency_per_user_ms:.3f}ms  ({time.time()-t0:.1f}s elapsed)")

ablation_df = pd.DataFrame(ablation_rows)
ablation_df.to_csv("/home/claude/results_svd_ablation.csv", index=False)
print("\nSaved results_svd_ablation.csv")
print(ablation_df.to_string(index=False))
