import ast
import os

import pandas as pd
import requests
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from nltk.stem.porter import PorterStemmer
    _stemmer = PorterStemmer()
except ImportError:
    _stemmer = None


# -----------------------------------
# 0. Config
# -----------------------------------

# Optional: set your TMDB API key as an environment variable to enable posters.
# Get a free key at https://www.themoviedb.org/settings/api
TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "")
TMDB_POSTER_BASE = "https://image.tmdb.org/t/p/w342"


# -----------------------------------
# 1-9. Load, merge, clean, build tags
#      (cached so this only runs once per session, not on every click)
# -----------------------------------

@st.cache_data
def load_and_prepare_data():

    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")

    movies = movies.merge(credits, left_on="id", right_on="movie_id")

    movies = movies[
        ['movie_id', 'title_x', 'genres', 'keywords', 'overview', 'cast', 'crew']
    ]

    movies.rename(
        columns={'movie_id': 'id', 'title_x': 'title', 'crew': 'director'},
        inplace=True
    )

    movies['overview'] = movies['overview'].fillna('')

    def convert(obj):
        return [i['name'] for i in ast.literal_eval(obj)]

    def convert3(obj):
        return [i['name'] for i in ast.literal_eval(obj)][:3]

    def fetch_director(obj):
        return [i['name'] for i in ast.literal_eval(obj) if i['job'] == 'Director']

    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(convert3)
    movies['director'] = movies['director'].apply(fetch_director)

    # Squash spaces inside multi-word names so "Sam Worthington" becomes one
    # token "SamWorthington" instead of polluting the vocabulary with two
    # separate common first/last names.
    def collapse(items):
        return [item.replace(" ", "") for item in items]

    movies['genres'] = movies['genres'].apply(collapse)
    movies['keywords'] = movies['keywords'].apply(collapse)
    movies['cast'] = movies['cast'].apply(collapse)
    movies['director'] = movies['director'].apply(collapse)

    # Overview is now included so plot/theme similarity contributes too,
    # not just metadata overlap.
    movies['tags'] = (
        movies['genres'].apply(lambda x: ' '.join(x)) + ' ' +
        movies['keywords'].apply(lambda x: ' '.join(x)) + ' ' +
        movies['cast'].apply(lambda x: ' '.join(x)) + ' ' +
        movies['director'].apply(lambda x: ' '.join(x)) + ' ' +
        movies['overview']
    )

    movies['tags'] = movies['tags'].str.lower()

    if _stemmer is not None:
        def stem(text):
            return ' '.join(_stemmer.stem(word) for word in text.split())
        movies['tags'] = movies['tags'].apply(stem)

    return movies


# -----------------------------------
# 10-11. TF-IDF + cosine similarity
#         (cached separately so re-vectorizing doesn't happen on every click)
# -----------------------------------

@st.cache_resource
def build_similarity_matrix(movies: pd.DataFrame):

    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['tags'])

    similarity = cosine_similarity(tfidf_matrix)

    return similarity


# -----------------------------------
# 12. Movie index lookup
# -----------------------------------

def build_movie_index(movies: pd.DataFrame) -> pd.Series:
    return pd.Series(movies.index, index=movies['title']).drop_duplicates()


# -----------------------------------
# 13. Recommendation function (with error handling + similarity scores)
# -----------------------------------

def recommend(movie: str, movies: pd.DataFrame, similarity, movie_index: pd.Series):

    if movie not in movie_index:
        return []

    index = movie_index[movie]
    distances = similarity[index]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommendations = []

    for i, score in movies_list:
        row = movies.iloc[i]
        recommendations.append({
            "title": row['title'],
            "id": row['id'],
            "score": round(float(score) * 100, 1),
        })

    return recommendations


# -----------------------------------
# Poster fetching (optional, needs TMDB_API_KEY)
# -----------------------------------

@st.cache_data(show_spinner=False)
def fetch_poster(movie_id: int):

    if not TMDB_API_KEY:
        return None

    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}"
        response = requests.get(
            url,
            params={"api_key": TMDB_API_KEY},
            timeout=5,
        )
        response.raise_for_status()
        data = response.json()
        poster_path = data.get("poster_path")

        if poster_path:
            return f"{TMDB_POSTER_BASE}{poster_path}"

    except requests.RequestException:
        pass

    return None


# -----------------------------------
# 14. Streamlit UI
# -----------------------------------

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #0f0f0f, #1c1c1c);
    color: white;
}

.main {
    padding-top: 30px;
}

.title {
    text-align: center;
    font-size: 45px;
    font-weight: bold;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #bbbbbb;
    margin-bottom: 40px;
}

.movie-card {
    background: #252525;
    padding: 18px 22px;
    margin: 12px 0;
    border-radius: 12px;
    border: 1px solid #3a3a3a;
    font-size: 18px;
    transition: 0.3s;
}

.movie-card:hover {
    transform: translateY(-3px);
    border-color: #777777;
}

.rank {
    font-size: 22px;
    font-weight: bold;
    margin-right: 12px;
}

.score {
    float: right;
    color: #8fd694;
    font-size: 15px;
}

.footer {
    text-align: center;
    color: #777777;
    margin-top: 50px;
    font-size: 14px;
}

div[data-testid="stButton"] > button,
.stButton > button {
    background: linear-gradient(135deg, #7b2ff7, #f107a3) !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 0 !important;
    transition: 0.2s !important;
}

div[data-testid="stButton"] > button *,
.stButton > button * {
    color: white !important;
    font-weight: bold !important;
    font-size: 16px !important;
}

div[data-testid="stButton"] > button:hover,
.stButton > button:hover {
    transform: translateY(-2px);
    opacity: 0.9;
}

</style>
""", unsafe_allow_html=True)


# Title
st.markdown(
    '<div class="title">🎬 Movie Recommendation System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Discover movies similar to the ones you love ✨'
    '</div>',
    unsafe_allow_html=True
)

# Load data + model (cached, so this is instant after the first run)
with st.spinner("Loading movie database..."):
    movies = load_and_prepare_data()
    similarity = build_similarity_matrix(movies)
    movie_index = build_movie_index(movies)

if not TMDB_API_KEY:
    st.info(
        "Posters are disabled. Set the TMDB_API_KEY environment variable "
        "to show movie posters (free key at themoviedb.org).",
        icon="ℹ️"
    )

# Movie selection
st.markdown("### 🍿 Choose a movie")

selected_movie = st.selectbox(
    "Select a movie from the list:",
    movies['title'].values
)


# Recommendation button
if st.button("✨ Get Recommendations", use_container_width=True):

    recommendations = recommend(selected_movie, movies, similarity, movie_index)

    if not recommendations:
        st.error("Sorry, that movie isn't in the database. Please pick another.")
    else:
        st.markdown("## 🎯 Recommended For You")

        cols = st.columns(len(recommendations)) if TMDB_API_KEY else None

        for idx, rec in enumerate(recommendations, start=1):

            if TMDB_API_KEY:
                poster_url = fetch_poster(rec["id"])
                with cols[idx - 1]:
                    if poster_url:
                        st.image(poster_url, use_container_width=True)
                    st.markdown(f"**#{idx}. {rec['title']}**")
                    st.caption(f"{rec['score']}% match")
            else:
                st.markdown(
                    f"""
                    <div class="movie-card">
                        <span class="score">{rec['score']}% match</span>
                        <span class="rank">#{idx}</span>
                        🎬 {rec['title']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )


# Footer
st.markdown(
    '<div class="footer">'
    'Powered by TF-IDF + Cosine Similarity 🤖'
    '</div>',
    unsafe_allow_html=True
)
