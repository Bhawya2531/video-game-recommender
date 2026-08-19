"""
Lightweight Streamlit demo for the video game recommender.
Talks to the FastAPI backend (must be running, default http://localhost:8000).

Run:
    streamlit run app.py
"""
import streamlit as st
import requests
import pandas as pd
st.set_page_config(page_title="Video Game Recommender", page_icon="🎮", layout="centered")

API_URL = st.sidebar.text_input("API base URL", value="http://localhost:8000")

st.title("🎮 Video Game Recommender")
st.caption("Latent-factor collaborative filtering (Truncated SVD, k=200) — Amazon Reviews 2018, Video Games")

st.sidebar.header("Model info")
try:
    health = requests.get(f"{API_URL}/health", timeout=3).json()
    st.sidebar.success("Backend connected" if health.get("model_loaded") else "Backend up, model not loaded")
except Exception:
    st.sidebar.error("Cannot reach backend. Start it with:\n`uvicorn main:app --port 8000` in /backend")

st.subheader("Pick a demo user")

if "sample_users" not in st.session_state:
    try:
        resp = requests.get(f"{API_URL}/users/sample", params={"n": 20}, timeout=5)
        st.session_state["sample_users"] = resp.json().get("users", [])
    except Exception:
        st.session_state["sample_users"] = []

col1, col2 = st.columns([3, 1])
with col1:
    selected_user = st.selectbox(
        "Select a user_id",
        options=st.session_state["sample_users"] or ["(backend unreachable)"],
    )
with col2:
    top_k = st.number_input("Top-K", min_value=1, max_value=20, value=10)

if st.button("Get recommendations", type="primary"):
    try:
        resp = requests.get(f"{API_URL}/recommend/{selected_user}", params={"top_k": top_k}, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            st.success(f"Model: {data['model']} (k={data['k']})")
            df = pd.DataFrame(data["recommendations"])
            df.index = df.index + 1
            st.dataframe(df, use_container_width=True)
        else:
            st.error(f"Error {resp.status_code}: {resp.json().get('detail')}")
    except Exception as e:
        st.error(f"Request failed: {e}")

st.divider()
st.subheader("Model / evaluation summary")
st.markdown(
    """
    - **Method:** Truncated SVD matrix factorization on a sparse user-item interaction matrix
    - **Selected latent dimension:** k = 200 (chosen from ablation across k = 20/50/100/200)
    - **Test Recall@10:** 5.30%  |  **Test NDCG@10:** 3.05%
    - **Known limitation:** near-zero recall on long-tail / cold-start items — see the
      error-analysis section of the project README for details and future-work mitigations.
    """
)
