import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ── Load Data ──────────────────────────────────────────────
ratings = pd.read_csv('u.data', sep='\t', names=['user_id','movie_id','rating','timestamp'])

movies = pd.read_csv('u.item', sep='|', encoding='latin-1', header=None,
                     names=['movie_id','title','release_date','video_date','imdb_url',
                            'unknown','Action','Adventure','Animation','Childrens','Comedy',
                            'Crime','Documentary','Drama','Fantasy','FilmNoir','Horror',
                            'Musical','Mystery','Romance','SciFi','Thriller','War','Western'])

users = pd.read_csv('u.user', sep='|', names=['user_id','age','gender','occupation','zip'])

# ── Basic Stats ────────────────────────────────────────────
print("=== RATINGS ===")
print(ratings.shape)
print(ratings.describe())

print("\n=== MOVIES ===")
print(movies[['movie_id','title']].head())

print("\n=== USERS ===")
print(users.head())

print("\nRating distribution:\n", ratings['rating'].value_counts().sort_index())
print(f"\nMatrix Sparsity: {(1 - len(ratings)/(943*1682))*100:.1f}%")

# ── Visualizations ─────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('MovieLens 100K — Exploratory Data Analysis', fontsize=16, fontweight='bold')

# 1. Rating Distribution
ax = axes[0, 0]
rating_counts = ratings['rating'].value_counts().sort_index()
bars = ax.bar(rating_counts.index, rating_counts.values,
              color=['#e74c3c','#e67e22','#f1c40f','#2ecc71','#3498db'],
              edgecolor='black', width=0.6)
ax.set_title('Rating Distribution', fontweight='bold')
ax.set_xlabel('Rating')
ax.set_ylabel('Count')
for bar, val in zip(bars, rating_counts.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+200, str(val), ha='center', fontsize=9)

# 2. Ratings per User
ax = axes[0, 1]
user_counts = ratings.groupby('user_id').size()
ax.hist(user_counts, bins=40, color='#3498db', edgecolor='black', alpha=0.8)
ax.set_title('Ratings per User', fontweight='bold')
ax.set_xlabel('Number of Ratings')
ax.set_ylabel('Number of Users')
ax.axvline(user_counts.mean(), color='red', linestyle='--', label=f'Mean: {user_counts.mean():.0f}')
ax.legend()

# 3. Ratings per Movie
ax = axes[0, 2]
movie_counts = ratings.groupby('movie_id').size()
ax.hist(movie_counts, bins=40, color='#e67e22', edgecolor='black', alpha=0.8)
ax.set_title('Ratings per Movie', fontweight='bold')
ax.set_xlabel('Number of Ratings')
ax.set_ylabel('Number of Movies')
ax.axvline(movie_counts.mean(), color='red', linestyle='--', label=f'Mean: {movie_counts.mean():.0f}')
ax.legend()

# 4. Top 10 Most Rated Movies
ax = axes[1, 0]
top_movies = ratings.groupby('movie_id').size().reset_index(name='count')
top_movies = top_movies.merge(movies[['movie_id','title']], on='movie_id')
top_movies = top_movies.nlargest(10, 'count')
top_movies['short_title'] = top_movies['title'].str[:25]
ax.barh(top_movies['short_title'], top_movies['count'], color='#9b59b6', edgecolor='black')
ax.set_title('Top 10 Most Rated Movies', fontweight='bold')
ax.set_xlabel('Number of Ratings')
ax.invert_yaxis()

# 5. Top 10 Highest Rated Movies (min 50 ratings)
ax = axes[1, 1]
avg_ratings = ratings.groupby('movie_id')['rating'].mean().reset_index()
avg_ratings = avg_ratings.merge(movies[['movie_id','title']], on='movie_id')
popular = ratings.groupby('movie_id').size().reset_index(name='count')
avg_ratings = avg_ratings.merge(popular, on='movie_id')
avg_ratings = avg_ratings[avg_ratings['count'] >= 50].nlargest(10, 'rating')
avg_ratings['short_title'] = avg_ratings['title'].str[:25]
ax.barh(avg_ratings['short_title'], avg_ratings['rating'], color='#1abc9c', edgecolor='black')
ax.set_title('Top 10 Highest Rated Movies\n(min 50 ratings)', fontweight='bold')
ax.set_xlabel('Average Rating')
ax.set_xlim(0, 5)
ax.invert_yaxis()

# 6. User Age Distribution
ax = axes[1, 2]
ax.hist(users['age'], bins=20, color='#e74c3c', edgecolor='black', alpha=0.8)
ax.set_title('User Age Distribution', fontweight='bold')
ax.set_xlabel('Age')
ax.set_ylabel('Number of Users')
ax.axvline(users['age'].mean(), color='blue', linestyle='--', label=f'Mean: {users["age"].mean():.0f}')
ax.legend()

plt.tight_layout()
plt.savefig('day1_eda.png', dpi=150, bbox_inches='tight')
plt.show()
print("Day 1 EDA complete!")
