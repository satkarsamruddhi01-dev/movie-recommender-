import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors

movies = pickle.load(open('movies.pkl','rb'))

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()

model = NearestNeighbors(n_neighbors=6, metric='cosine')
model.fit(vectors)

st.title("🎬 Movie Recommendation System")

movie = st.selectbox("Select Movie", movies['title'].values)

if st.button("Recommend"):
    index = movies[movies['title']==movie].index[0]
    distances, indices = model.kneighbors([vectors[index]])

    for i in indices[0][1:]:
        st.write(movies.iloc[i].title)
