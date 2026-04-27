import streamlit as st
import pickle
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors

# Load movie data
movies = pickle.load(open('movies.pkl', 'rb'))

# Create vectors
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()

# Train KNN model
model = NearestNeighbors(n_neighbors=6, metric='cosine')
model.fit(vectors)

# TMDB API Key
API_KEY = "YOUR_API_KEY_HERE"

# Fetch poster
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}"
    data = requests.get(url).json()

    if data.get("poster_path"):
        return "https://image.tmdb.org/t/p/w500" + data["poster_path"]
    else:
        return "https://via.placeholder.com/300x450?text=No+Image"

# UI
st.set_page_config(page_title="Movie Recommendation System", layout="wide")

st.title("🎬 Movie Recommendation System")

selected_movie = st.selectbox("Select Movie", movies['title'].values)

if st.button("Recommend"):

    index = movies[movies['title'] == selected_movie].index[0]

    distances, indices = model.kneighbors([vectors[index]])

    cols = st.columns(5)

    for count, i in enumerate(indices[0][1:]):
        movie_id = movies.iloc[i].movie_id
        poster = fetch_poster(movie_id)
        name = movies.iloc[i].title

        with cols[count]:
            st.image(poster)
            st.write(name)
