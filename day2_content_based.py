import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Load Data ──────────────────────────────────────────────
movies = pd.read_csv('u.item', sep='|', encoding='latin-1', header=None,
                     names=['movie_id','title','release_date','video_date','imdb_url',
                            'unknown','Action','Adventure','Animation','Childrens','Comedy',
                            'Crime','Documentary','Drama','Fantasy','FilmNoir','Horror',
                            'Musical','Mystery','Romance','SciFi','Thriller','War','Western'])

genre_cols = ['Action','Adventure','Animation','Childrens','Comedy','Crime',
              'Documentary','Drama','Fantasy','FilmNoir','Horror','Musical',
              'Mystery','Romance','SciFi','Thriller','War','Western']

# ── Build Genre String ─────────────────────────────────────
movies['genres'] = movies.apply(
    lambda row: ' '.join([g for g in genre_cols if row[g] == 1]) or 'Unknown', axis=1
)

# ── TF-IDF Vectorization ───────────────────────────────────
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(movies['genres'])
print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
print(f"Vocabulary: {list(tfidf.vocabulary_.keys())}")

# ── Cosine Similarity Matrix ───────────────────────────────
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
print(f"Cosine similarity matrix shape: {cosine_sim.shape}")

# ── Index Mapping ──────────────────────────────────────────
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# ── Recommendation Function ────────────────────────────────
def content_based_recommend(title, n=10):
    if title not in indices:
        return f"Movie '{title}' not found."
    idx = indices[title]
    sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)[1:n+1]
    movie_indices = [i[0] for i in sim_scores]
    scores = [round(i[1], 3) for i in sim_scores]
    result = movies['title'].iloc[movie_indices].reset_index(drop=True)
    return pd.DataFrame({'Movie': result, 'Similarity Score': scores})

# ── Test ───────────────────────────────────────────────────
print("\n=== Recommendations for: Toy Story (1995) ===")
print(content_based_recommend('Toy Story (1995)'))

print("\n=== Recommendations for: GoldenEye (1995) ===")
print(content_based_recommend('GoldenEye (1995)'))

print("\n=== Recommendations for: Twelve Monkeys (1995) ===")
print(content_based_recommend('Twelve Monkeys (1995)'))

# ── Visualizations ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Day 2 — Content-Based Filtering (TF-IDF + Cosine Similarity)', fontsize=14, fontweight='bold')

# Genre Distribution
ax = axes[0]
genre_counts = movies[genre_cols].sum().sort_values(ascending=True)
colors = plt.cm.tab20(np.linspace(0, 1, len(genre_counts)))
bars = ax.barh(genre_counts.index, genre_counts.values, color=colors, edgecolor='black')
ax.set_title('Movie Count by Genre', fontweight='bold')
ax.set_xlabel('Number of Movies')
for bar, val in zip(bars, genre_counts.values):
    ax.text(bar.get_width()+5, bar.get_y()+bar.get_height()/2, str(val), va='center', fontsize=8)

# Similarity Heatmap
ax = axes[1]
sample_titles = ['Toy Story (1995)', 'GoldenEye (1995)', 'Twelve Monkeys (1995)',
                 'Fargo (1996)', 'Aladdin (1992)', 'Silence of the Lambs, The (1991)',
                 'Star Wars (1977)', 'Babe (1995)']
sample_idx = [indices[t] for t in sample_titles]
sim_subset = cosine_sim[np.ix_(sample_idx, sample_idx)]
short_titles = [t[:20] for t in sample_titles]
sns.heatmap(sim_subset, annot=True, fmt='.2f', xticklabels=short_titles,
            yticklabels=short_titles, cmap='YlOrRd', ax=ax, linewidths=0.5)
ax.set_title('Cosine Similarity Heatmap\n(Sample Movies)', fontweight='bold')
ax.tick_params(axis='x', rotation=45)
ax.tick_params(axis='y', rotation=0)

plt.tight_layout()
plt.savefig('day2_content_based.png', dpi=150, bbox_inches='tight')
plt.show()

# Save for later days
movies.to_csv('movies_processed.csv', index=False)
np.save('cosine_sim.npy', cosine_sim)
print("\nDay 2 Content-Based Filtering complete!")
