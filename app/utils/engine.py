import streamlit as st
import requests
import pickle

# Access secrets
TMDB_API_KEY = st.secrets["TMDB_API_KEY"]

@st.cache_data
def get_trending_movies():
    url = f"https://api.themoviedb.org/3/trending/movie/week?api_key={TMDB_API_KEY}"
    try:
        res = requests.get(url)
        return res.json().get('results', [])[:5]
    except:
        return []

@st.cache_resource
def load_local_data():
    try:
        movies = pickle.load(open('data/movies_list.pkl', 'rb'))
        sim_map = pickle.load(open('data/similarity_map.pkl', 'rb'))
        return movies, sim_map
    except:
        return None, None

def get_poster_url(path):
    if path:
        return f"https://image.tmdb.org/t/p/w500{path}"
    return "https://via.placeholder.com/500x750?text=No+Poster"