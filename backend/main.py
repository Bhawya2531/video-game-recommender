"""
FastAPI recommendation service.

Loads the serialized SVD model (k=200) once at startup and serves
top-K recommendations for a given user via GET /recommend/{user_id}.

Run locally:
    uvicorn main:app --reload --port 8000

Then visit http://localhost:8000/docs for interactive API docs.
"""
import pickle
import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from scipy.sparse import csr_matrix

MODEL_PATH = Path(__file__).parent.parent / "models" / "serving_artifact.pkl"

app = FastAPI(
    title="Video Game Recommender API",
    description="Latent-factor (Truncated SVD) top-K recommendation service.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_artifact = None
_user_to_idx = None
_item_classes = None
_R = None


@app.on_event("startup")
def load_model():
    global _artifact, _user_to_idx, _item_classes, _R
    with open(MODEL_PATH, "rb") as f:
        _artifact = pickle.load(f)
    _user_to_idx = {u: i for i, u in enumerate(_artifact["user_classes"])}
    _item_classes = _artifact["item_classes"]
    _R = csr_matrix(
        (np.ones(len(_artifact["R_indices"]), dtype=np.float32),
         _artifact["R_indices"], _artifact["R_indptr"]),
        shape=_artifact["R_shape"],
    )
    print(f"Model loaded: k={_artifact['k']}, users={_R.shape[0]}, items={_R.shape[1]}")


class RecommendationItem(BaseModel):
    item_id: str
    score: float


class RecommendationResponse(BaseModel):
    user_id: str
    model: str
    k: int
    recommendations: list[RecommendationItem]


@app.get("/")
def root():
    return {"status": "ok", "message": "Video Game Recommender API. See /docs."}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _artifact is not None}


@app.get("/recommend/{user_id}", response_model=RecommendationResponse)
def recommend(user_id: str, top_k: int = 10):
    if _artifact is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    if user_id not in _user_to_idx:
        raise HTTPException(status_code=404, detail=f"Unknown user_id: {user_id}")
    if not (1 <= top_k <= 50):
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 50.")

    u_idx = _user_to_idx[user_id]
    scores = _artifact["U"][u_idx] @ _artifact["Vt"]  # (n_items,)

    seen = _R.getrow(u_idx).toarray().flatten() > 0
    scores = scores.copy()
    scores[seen] = -np.inf

    top_idx = np.argpartition(-scores, top_k - 1)[:top_k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]

    recs = [
        RecommendationItem(item_id=str(_item_classes[i]), score=float(scores[i]))
        for i in top_idx
    ]
    return RecommendationResponse(
        user_id=user_id, model="TruncatedSVD", k=_artifact["k"], recommendations=recs
    )


@app.get("/users/sample")
def sample_users(n: int = 10):
    """Returns a sample of valid user_ids, useful for the demo frontend."""
    if _artifact is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    users = _artifact["user_classes"][:n].tolist()
    return {"users": users}
