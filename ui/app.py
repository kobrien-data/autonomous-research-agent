import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("Autonomous Research Agent")

with st.sidebar:
    st.header("Status")
    try:
        resp = requests.get(f"{API_URL}/health", timeout=2)
        if resp.ok:
            st.success("API connected")
        else:
            st.error("API unhealthy")
    except requests.exceptions.ConnectionError:
        st.warning("API unreachable")

query = st.text_input("Research query", placeholder="Enter a topic to research...")

if st.button("Run") and query:
    with st.spinner("Running..."):
        try:
            resp = requests.post(f"{API_URL}/research", params={"query": query}, timeout=10)
            st.json(resp.json())
        except requests.exceptions.ConnectionError:
            st.error("Could not reach API.")
