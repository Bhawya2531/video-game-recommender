"""
Serializes the selected SVD model (k=200, chosen from the Phase 4 ablation)
into a single lightweight artifact used by the FastAPI serving layer.
"""
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix

train = pd.read_parquet("../data/train.parquet")

user_enc = LabelEncoder().fit(train["user_id"])
item_enc = LabelEncoder().fit(train["item_id"])
n_users, n_items = len(user_enc.classes_), len(item_enc.classes_)

train_u = user_enc.transform(train["user_id"])
train_i = item_enc.transform(train["item_id"])
R = csr_matrix((np.ones(len(train), dtype=np.float32), (train_u, train_i)), shape=(n_users, n_items))

with open("../../home/claude/models/svd_k200.pkl", "rb") as f:
    saved = pickle.load(f)

serving_artifact = {
    "U": saved["U"].astype(np.float32),          # user latent vectors (n_users, 200)
    "Vt": saved["Vt"].astype(np.float32),         # item latent vectors (200, n_items)
    "user_classes": user_enc.classes_,            # index -> original user_id
    "item_classes": item_enc.classes_,            # index -> original item_id (asin)
    "R_indptr": R.indptr,                         # sparse train matrix, for filtering seen items
    "R_indices": R.indices,
    "R_shape": R.shape,
    "k": 200,
}

with open("../models/serving_artifact.pkl", "wb") as f:
    pickle.dump(serving_artifact, f)

print("Saved serving_artifact.pkl")
print(f"Users: {n_users}, Items: {n_items}, k=200")
