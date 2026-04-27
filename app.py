import streamlit as st
import pickle
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(page_title="Movie Recommendation System", layout="wide")

# ---------------- LOAD DATA ----------------
movies = pickle.load(open('movies.pkl', 'rb'))

# ---------------- CREATE VECTORS ----------------
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()

# ---------------- TRAIN MODEL ----------------
model = NearestNeighbors(n_neighbors=6, metric='cosine')
model.fit(vectors)

# ---------------- TMDB API KEY ----------------
API_KEY = "YOUR_TMDB_API_KEY"

# ---------------- FETCH POSTER ----------------
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}"
    
    try:
        response = requests.get(url)
        data = response.json()

        if response.status_code == 200 and data.get("poster_path"):
            return "https://image.tmdb.org/t/p/w500" + data["poster_path"]
        else:
            return "https://via.placeholder.com/300x450.png?text=No+Poster"

    except:
        return "https://via.placeholder.com/300x450.png?text=No+Poster"

# ---------------- RECOMMEND FUNCTION ----------------
def recommend(movie):
    index = movies[movies['title'] == movie].index[0]

    distances, indices = model.kneighbors([vectors[index]])

    recommended_names = []
    recommended_posters = []

    for i in indices[0][1:]:
        movie_id = movies.iloc[i].movie_id
        recommended_names.append(movies.iloc[i].title)
        recommended_posters.append(fetch_poster(movie_id))

    return recommended_names, recommended_posters

# ---------------- UI ----------------
st.title("🎬 Movie Recommendation System")

selected_movie = st.selectbox(
    "Select Movie",
    movies['title'].values
)

if st.button("Recommend"):
    names, posters = recommend(selected_movie)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.image(posters[0])
        st.write(names[0])

    with col2:
        st.image(posters[1])
        st.write(names[1])

    with col3:
        st.image(posters[2])
        st.write(names[2])

    with col4:
        st.image(posters[3])
        st.write(names[3])

    with col5:
        st.image(posters[4])
        st.write(names[4])
