import streamlit as st
import requests
import pickle
import pandas as pd

API_KEY = "TA_CLE_TMDB_ICI" # Remplace par ta clé gratuite sur themoviedb.org

@st.cache_data
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"
    try:
        data = requests.get(url).json()
        poster_path = data['poster_path']
        return "https://image.tmdb.org/t/p/w500/" + poster_path
    except:
        return "https://via.placeholder.com/500x750?text=No+Image"

@st.cache_resource
def load_data():
    # On charge les fichiers préparés
    movies = pickle.load(open('data/movies_list.pkl', 'rb'))
    similarity = pickle.load(open('data/similarity.pkl', 'rb'))
    return movies, similarity

def get_recommendations(movie_title, movies, similarity):
    idx = movies[movies['title'] == movie_title].index[0]
    distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])
    
    recoms = []
    for i in distances[1:7]: # Top 6
        movie_id = movies.iloc[i[0]].movie_id
        title = movies.iloc[i[0]].title
        recoms.append((title, fetch_poster(movie_id)))
    return recoms