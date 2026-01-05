import streamlit as st
from utils.engine import load_local_data, get_poster_url

st.title("🔍 The Cinema Hall")
movies, _ = load_local_data()

if movies is not None:
    search = st.text_input("Search movie title...", placeholder="Enter a name...")
    
    # Simple Filter
    display_df = movies[movies['title'].str.contains(search, case=False)] if search else movies.head(20)
    
    cols = st.columns(4)
    for i, (idx, row) in enumerate(display_df.head(12).iterrows()):
        with cols[i % 4]:
            # In a real scenario, use TMDB search to find IDs, here we use placeholders for UI
            st.image("https://via.placeholder.com/500x750/111/fff?text=Poster", use_container_width=True)
            st.write(row['title'])