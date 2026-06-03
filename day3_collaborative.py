import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

# ── Load Data ──────────────────────────────────────────────
ratings = pd.read_csv('u.data', sep='\t', names=['user_id','movie_id','rating','timestamp'])
movies  = pd.read_csv('u.item', sep='|', encoding='latin-1', header=None,
                      names=['movie_id','title','release_date','video_date','imdb_url',
                             'unknown','Action','Adventure','Animation','Childrens','Comedy',
                             'Crime','Documentary','Drama','Fantasy','FilmNoir','Horror',
                             'Musical','Mystery','Romance','SciFi','Thriller','War','Western'])

# ── Build User-Movie Matrix ────────────────────────────────
user_movie_matrix = ratings.pivot_table(index='user_id', columns='movie_id', values='rating')
matrix_filled     = user_movie_matrix.fillna(0)
print(f"User-Movie Matrix shape: {user_movie_matrix.shape}")
print(f"Sparsity: {user_movie_matrix.isna().sum().sum() / (user_movie_matrix.shape[0]*user_movie_matrix.shape[1])*100:.1f}%")

# ── User-Based Collaborative Filtering ────────────────────
user_sim    = cosine_similarity(matrix_filled)
user_sim_df = pd.DataFrame(user_sim, index=user_movie_matrix.index, columns=user_movie_matrix.index)

def user_based_recommend(user_id, n_recommendations=10, n_similar_users=10):
    similar_users = user_sim_df[user_id].sort_values(ascending=False)[1:n_similar_users+1]
    rated_movies  = user_movie_matrix.loc[user_id].dropna().index.tolist()

    weighted_ratings = {}
    similarity_sums  = {}

    for sim_user, sim_score in similar_users.items():
        sim_user_ratings = user_movie_matrix.loc[sim_user].dropna()
        for movie_id, rating in sim_user_ratings.items():
            if movie_id not in rated_movies:
                weighted_ratings[movie_id] = weighted_ratings.get(movie_id, 0) + sim_score * rating
                similarity_sums[movie_id]  = similarity_sums.get(movie_id,  0) + sim_score

    predicted    = {m: weighted_ratings[m]/similarity_sums[m] for m in weighted_ratings}
    predicted_df = pd.Series(predicted).sort_values(ascending=False).head(n_recommendations)
    result       = pd.DataFrame({'movie_id': predicted_df.index, 'predicted_rating': predicted_df.values.round(2)})
    result       = result.merge(movies[['movie_id','title']], on='movie_id')
    return result[['title','predicted_rating']]

# ── Item-Based Collaborative Filtering ────────────────────
item_sim    = cosine_similarity(matrix_filled.T)
item_sim_df = pd.DataFrame(item_sim, index=user_movie_matrix.columns, columns=user_movie_matrix.columns)

def item_based_recommend(user_id, n_recommendations=10):
    user_ratings = user_movie_matrix.loc[user_id].dropna()
    scores = {}
    for rated_movie, rating in user_ratings.items():
        similar_items = item_sim_df[rated_movie].drop(index=user_ratings.index, errors='ignore')
        for movie_id, sim in similar_items.items():
            scores[movie_id] = scores.get(movie_id, 0) + sim * rating

    recommended = pd.Series(scores).sort_values(ascending=False).head(n_recommendations)
    result      = pd.DataFrame({'movie_id': recommended.index, 'score': recommended.values.round(2)})
    result      = result.merge(movies[['movie_id','title']], on='movie_id')
    return result[['title','score']]

# ── Test ───────────────────────────────────────────────────
print("\n=== User-Based Recommendations for User 1 ===")
print(user_based_recommend(1))

print("\n=== Item-Based Recommendations for User 1 ===")
print(item_based_recommend(1))

# ── Visualizations ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Day 3 — Collaborative Filtering', fontsize=14, fontweight='bold')

# User-User Similarity Heatmap
ax = axes[0]
sample_users = [1, 5, 10, 20, 30, 50, 100, 150, 200, 250]
subset = user_sim_df.loc[sample_users, sample_users]
sns.heatmap(subset, annot=True, fmt='.2f', cmap='Blues', ax=ax, linewidths=0.5)
ax.set_title('User-User Similarity Heatmap\n(Sample 10 Users)', fontweight='bold')
ax.set_xlabel('User ID')
ax.set_ylabel('User ID')

# Item-Item Similarity Heatmap
ax = axes[1]
top10_movies  = ratings.groupby('movie_id').size().nlargest(10).index.tolist()
item_subset   = item_sim_df.loc[top10_movies, top10_movies]
top10_titles  = [movies[movies['movie_id']==m]['title'].values[0][:20] for m in top10_movies]
item_subset.index   = top10_titles
item_subset.columns = top10_titles
sns.heatmap(item_subset, annot=True, fmt='.2f', cmap='Oranges', ax=ax, linewidths=0.5)
ax.set_title('Item-Item Similarity Heatmap\n(Top 10 Most Rated Movies)', fontweight='bold')
ax.tick_params(axis='x', rotation=45)
ax.tick_params(axis='y', rotation=0)

plt.tight_layout()
plt.savefig('day3_collaborative.png', dpi=150, bbox_inches='tight')
plt.show()

# Save matrices for Day 4
np.save('user_sim.npy', user_sim)
np.save('item_sim.npy', item_sim)
print("\nDay 3 Collaborative Filtering complete!")
