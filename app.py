import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from collections import defaultdict

# ── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="🎬 Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

st.markdown("""
<style>
    .main-title { font-size: 2.5rem; font-weight: 800; color: #e50914; text-align: center; }
    .sub-title  { font-size: 1.1rem; color: #888; text-align: center; margin-bottom: 2rem; }
    .metric-card { background: #1e1e1e; border-radius: 10px; padding: 1rem;
                   border-left: 4px solid #e50914; margin: 0.5rem 0; }
    .movie-card  { background: #1a1a2e; border-radius: 8px; padding: 0.8rem;
                   margin: 0.3rem 0; border: 1px solid #333; }
    .stButton > button { background-color: #e50914; color: white;
                         border-radius: 8px; border: none;
                         padding: 0.5rem 2rem; font-weight: bold; }
    .stButton > button:hover { background-color: #b20710; }
</style>
""", unsafe_allow_html=True)

# ── Load Data ──────────────────────────────────────────────
@st.cache_data
def load_data():
    ratings = pd.read_csv('u.data', sep='\t',
                          names=['user_id','movie_id','rating','timestamp'])
    movies  = pd.read_csv('u.item', sep='|', encoding='latin-1', header=None,
                          names=['movie_id','title','release_date','video_date','imdb_url',
                                 'unknown','Action','Adventure','Animation','Childrens','Comedy',
                                 'Crime','Documentary','Drama','Fantasy','FilmNoir','Horror',
                                 'Musical','Mystery','Romance','SciFi','Thriller','War','Western'])
    genre_cols = ['Action','Adventure','Animation','Childrens','Comedy','Crime',
                  'Documentary','Drama','Fantasy','FilmNoir','Horror','Musical',
                  'Mystery','Romance','SciFi','Thriller','War','Western']
    movies['genres'] = movies.apply(
        lambda r: ' '.join([g for g in genre_cols if r[g]==1]) or 'Unknown', axis=1)
    return ratings, movies

@st.cache_resource
def build_models(ratings, movies):
    genre_cols = ['Action','Adventure','Animation','Childrens','Comedy','Crime',
                  'Documentary','Drama','Fantasy','FilmNoir','Horror','Musical',
                  'Mystery','Romance','SciFi','Thriller','War','Western']

    # Content-Based
    tfidf      = TfidfVectorizer()
    tfidf_mat  = tfidf.fit_transform(movies['genres'])
    cosine_sim = cosine_similarity(tfidf_mat, tfidf_mat)
    indices    = pd.Series(movies.index, index=movies['title']).drop_duplicates()

    # SVD
    reader   = Reader(rating_scale=(1, 5))
    data     = Dataset.load_from_df(ratings[['user_id','movie_id','rating']], reader)
    trainset, _ = train_test_split(data, test_size=0.2, random_state=42)
    svd = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
    svd.fit(trainset)

    return cosine_sim, indices, svd

ratings, movies = load_data()
cosine_sim, indices, svd = build_models(ratings, movies)

# ── Recommendation Functions ───────────────────────────────
def content_recommend(title, n=10):
    if title not in indices:
        return pd.DataFrame()
    idx        = indices[title]
    sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)[1:n+1]
    movie_idx  = [i[0] for i in sim_scores]
    scores     = [round(i[1], 3) for i in sim_scores]
    result     = movies['title'].iloc[movie_idx].reset_index(drop=True)
    genre_list = movies['genres'].iloc[movie_idx].reset_index(drop=True)
    return pd.DataFrame({'🎬 Movie': result, '🎭 Genres': genre_list, '📊 Similarity': scores})

def svd_recommend(user_id, n=10):
    try:
        rated   = ratings[ratings['user_id']==user_id]['movie_id'].tolist()
        unrated = [m for m in movies['movie_id'].tolist() if m not in rated]
        preds   = sorted([(m, svd.predict(user_id, m).est) for m in unrated],
                         key=lambda x: x[1], reverse=True)[:n]
        result  = pd.DataFrame(preds, columns=['movie_id','predicted_rating'])
        result  = result.merge(movies[['movie_id','title','genres']], on='movie_id')
        result['predicted_rating'] = result['predicted_rating'].round(2)
        return result[['title','genres','predicted_rating']].rename(
            columns={'title':'🎬 Movie','genres':'🎭 Genres','predicted_rating':'⭐ Predicted Rating'})
    except:
        return pd.DataFrame()

# ── UI ─────────────────────────────────────────────────────
st.markdown('<div class="main-title">🎬 Movie Recommendation System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">MovieLens 100K · Content-Based · Collaborative · SVD Matrix Factorization</div>',
            unsafe_allow_html=True)

# Stats Row
col1, col2, col3, col4 = st.columns(4)
col1.metric("🎬 Total Movies",  "1,682")
col2.metric("👤 Total Users",   "943")
col3.metric("⭐ Total Ratings", "100,000")
col4.metric("📉 Best RMSE",     "0.9352")

st.markdown("---")

# Tabs
tab1, tab2, tab3 = st.tabs(["🎭 Content-Based Filtering", "🤝 SVD Recommendations", "📊 Model Comparison"])

# ── Tab 1: Content-Based ───────────────────────────────────
with tab1:
    st.subheader("🎭 Find Similar Movies")
    st.write("Select a movie you like and get similar movies based on genre.")

    movie_list   = sorted(movies['title'].tolist())
    selected     = st.selectbox("🔍 Choose a Movie", movie_list, index=movie_list.index('Toy Story (1995)'))
    n_recs       = st.slider("Number of Recommendations", 5, 20, 10)

    if st.button("🎬 Get Similar Movies", key='content'):
        with st.spinner("Finding similar movies..."):
            recs = content_recommend(selected, n_recs)
            if not recs.empty:
                # Show selected movie info
                movie_info = movies[movies['title']==selected].iloc[0]
                st.success(f"**Selected:** {selected} | **Genres:** {movie_info['genres']}")
                st.markdown("### 🎯 Recommended Movies")
                st.dataframe(recs, width="stretch", hide_index=True)
            else:
                st.error("Movie not found.")

# ── Tab 2: SVD ────────────────────────────────────────────
with tab2:
    st.subheader("🤝 Personalized Recommendations via SVD")
    st.write("Enter your User ID to get personalized movie recommendations.")

    col1, col2 = st.columns([1, 2])
    with col1:
        user_id = st.number_input("👤 Enter User ID (1–943)", min_value=1, max_value=943, value=1)
        n_svd   = st.slider("Number of Recommendations", 5, 20, 10, key='svd_n')

    with col2:
        user_ratings = ratings[ratings['user_id']==user_id]
        st.info(f"User {user_id} has rated **{len(user_ratings)} movies** | "
                f"Average Rating: **{user_ratings['rating'].mean():.2f}**")

        st.write("**Movies this user has rated (sample):**")
        sample = user_ratings.merge(movies[['movie_id','title']], on='movie_id')\
                              .sort_values('rating', ascending=False).head(5)
        st.dataframe(sample[['title','rating']].rename(
            columns={'title':'Movie','rating':'Rating'}),
            width="stretch", hide_index=True)

    if st.button("🚀 Get Personalized Recommendations", key='svd'):
        with st.spinner("Running SVD predictions..."):
            recs = svd_recommend(user_id, n_svd)
            if not recs.empty:
                st.markdown("### 🎯 Your Personalized Recommendations")
                st.dataframe(recs, width="stretch", hide_index=True)
            else:
                st.error("Could not generate recommendations.")

# ── Tab 3: Model Comparison ────────────────────────────────
with tab3:
    st.subheader("📊 Model Performance Comparison")

    metrics_data = {
        'Model':     ['SVD', 'User-Based CF', 'Item-Based CF'],
        'RMSE':      [0.9352, 0.9802, 0.9726],
        'MAE':       [0.7375, 0.7727, 0.7681],
        'Precision@5':  [0.6409, 0.6555, 0.6043],
        'Precision@10': [0.5074, 0.5218, 0.4928],
        'Recall@10':    [0.5428, 0.5753, 0.5184],
        'F1@10':        [0.5245, 0.5472, 0.5053],
    }
    df_metrics = pd.DataFrame(metrics_data)

    st.dataframe(df_metrics.set_index('Model'), width="stretch")

    st.markdown("### 🏆 Key Insights")
    col1, col2, col3 = st.columns(3)
    col1.success("✅ **SVD** achieves the best RMSE (0.9352) — lowest prediction error")
    col2.info("📌 **User-Based CF** has slightly better Precision@K for top-N lists")
    col3.warning("⚡ **SVD** is most scalable — ideal for large datasets like Amazon")

    st.markdown("### 📖 About This Project")
    st.markdown("""
    | Component | Details |
    |---|---|
    | Dataset | MovieLens 100K — 100,000 ratings, 943 users, 1,682 movies |
    | Approach 1 | Content-Based Filtering — TF-IDF + Cosine Similarity |
    | Approach 2 | Collaborative Filtering — User-Based & Item-Based (KNN) |
    | Approach 3 | Matrix Factorization — SVD (Surprise library) |
    | Evaluation | RMSE, MAE, Precision@K, Recall@K, F1@K |
    | Deployment | Streamlit web application |
    """)

st.markdown("---")
st.markdown("<center>Built with ❤️ by Naveen T</center>",
            unsafe_allow_html=True)
