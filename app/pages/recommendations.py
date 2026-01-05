import streamlit as st
from utils.engine import load_local_data

st.title("🎯 Picked For You")
movies, sim_map = load_local_data()

if movies is not None:
    choice = st.selectbox("Based on which movie?", movies['title'].values)
    
    if st.button("Get AI Recommendations"):
        st.info(f"Analyzing patterns for {choice}...")
        # Add your similarity logic here using sim_map
        st.write("Results will appear here based on your 32M ratings engine.")