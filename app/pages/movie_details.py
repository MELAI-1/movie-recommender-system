import streamlit as st

col1, col2 = st.columns([1, 2])

with col1:
    st.image("https://image.tmdb.org/t/p/w500/8Gxv8S7Yhp1uP69TtmEqX7s3ZpX.jpg")

with col2:
    st.title("Oppenheimer")
    st.write("2023 | 3h 0m | Biography, Drama, History")
    st.write("⭐ 8.4/10")
    st.markdown("### Summary")
    st.write("The story of American scientist J. Robert Oppenheimer and his role in the development of the atomic bomb.")
    
    st.markdown("### Cast")
    st.write("Cillian Murphy, Emily Blunt, Matt Damon")
    
    st.button("▶️ Play Now")
    st.button("➕ Add to My List")