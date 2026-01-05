import streamlit as st

st.title("⭐ My Ratings")
st.write("Rate movies to improve your AI recommendations.")

col1, col2 = st.columns([2, 1])

with col1:
    movie_to_rate = st.selectbox("Select a movie you've watched", ["Toy Story", "The Avengers", "Shrek"])
    rating = st.feedback("stars") # New Streamlit star rating feature
    comment = st.text_area("What did you think?")
    st.button("Submit Rating")

with col2:
    st.markdown("### Your History")
    st.caption("✅ Inception - 5 Stars")
    st.caption("✅ Titanic - 4 Stars")