import streamlit as st

st.set_page_config(page_title="Netflix AI Recommender", layout="wide")

# Chargement du style CSS
with open('assets/style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Barre latérale (Sidebar) - Identité visuelle
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg", width=150)
    st.markdown("---")
    st.markdown("### 👤 Developer")
    st.info("**Astrid Melvin Fokam Ninyim**")
    st.caption("Expert en IA & Systèmes de Recommandation")
    st.markdown("---")

# Contenu de la Home Page
st.title("🎬 Netflix Movie Recommender")
st.markdown("### Bienvenue dans votre futur système de divertissement intelligent.")
st.image("https://images.alphacoders.com/134/1344262.png", use_container_width=True)

st.success("Utilisez le menu à gauche pour naviguer dans l'application.")