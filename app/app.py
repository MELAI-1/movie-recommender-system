import streamlit as st
from utils.engine import get_trending_movies, get_poster_url

st.set_page_config(page_title="REELVIBE | Home", layout="wide")

with open('assets/style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="sidebar-logo">REELVIBE</div>', unsafe_allow_html=True)
    st.write("---")
    st.markdown(f"👋 Welcome, **Melvin**")
    st.markdown(f"""
        <div style="background-color: #111; padding: 15px; border-radius: 10px; border-left: 4px solid #E50914;">
            <p style="margin:0; font-size: 11px; color: #777;">Developed by</p>
            <p style="margin:0; font-weight: bold; color: #fff;">{st.secrets["DEVELOPER_NAME"]}</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("""
    <div class="hero-box">
        <h1 style="font-size: 55px; color: white !important;">Stories that move you.</h1>
        <p style="font-size: 18px; color: #ccc; max-width: 600px;">
            Stop scrolling and start watching. We've handpicked the best cinematic experiences just for you.
        </p>
    </div>
""", unsafe_allow_html=True)

st.subheader("🔥 Trending This Week")
trending = get_trending_movies()

if trending:
    cols = st.columns(5)
    for i, movie in enumerate(trending):
        with cols[i]:
            st.image(get_poster_url(movie.get('poster_path')), use_container_width=True)
            st.markdown(f"<p style='text-align:center; font-weight:bold;'>{movie.get('title')}</p>", unsafe_allow_html=True)
else:
    st.warning("Please check your API Key in secrets.toml")