import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split, cross_validate

# ── Load Data ──────────────────────────────────────────────
ratings = pd.read_csv('u.data', sep='\t', names=['user_id','movie_id','rating','timestamp'])
movies  = pd.read_csv('u.item', sep='|', encoding='latin-1', header=None,
                      names=['movie_id','title','release_date','video_date','imdb_url',
                             'unknown','Action','Adventure','Animation','Childrens','Comedy',
                             'Crime','Documentary','Drama','Fantasy','FilmNoir','Horror',
                             'Musical','Mystery','Romance','SciFi','Thriller','War','Western'])

# ── Load into Surprise format ──────────────────────────────
reader   = Reader(rating_scale=(1, 5))
data     = Dataset.load_from_df(ratings[['user_id','movie_id','rating']], reader)

# ── Train/Test Split (80/20) ───────────────────────────────
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# ── Train SVD Model ────────────────────────────────────────
svd = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
svd.fit(trainset)

# ── Evaluate ──────────────────────────────────────────────
predictions = svd.test(testset)
rmse = accuracy.rmse(predictions)
mae  = accuracy.mae(predictions)
print(f"RMSE: {rmse:.4f}")
print(f"MAE : {mae:.4f}")

# ── 5-Fold Cross Validation ────────────────────────────────
print("\nRunning 5-Fold Cross Validation...")
cv_results = cross_validate(SVD(random_state=42), data, measures=['RMSE','MAE'], cv=5, verbose=True)
print(f"\nMean RMSE: {cv_results['test_rmse'].mean():.4f}")
print(f"Mean MAE : {cv_results['test_mae'].mean():.4f}")

# ── SVD Recommendation Function ───────────────────────────
def svd_recommend(user_id, n=10):
    rated   = ratings[ratings['user_id']==user_id]['movie_id'].tolist()
    unrated = [m for m in movies['movie_id'].tolist() if m not in rated]
    preds   = sorted([(m, svd.predict(user_id, m).est) for m in unrated],
                     key=lambda x: x[1], reverse=True)[:n]
    result  = pd.DataFrame(preds, columns=['movie_id','predicted_rating'])
    result  = result.merge(movies[['movie_id','title']], on='movie_id')
    result['predicted_rating'] = result['predicted_rating'].round(2)
    return result[['title','predicted_rating']]

print("\n=== SVD Recommendations for User 1 ===")
print(svd_recommend(1))

print("\n=== SVD Recommendations for User 50 ===")
print(svd_recommend(50))

# ── Visualizations ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Day 4 — SVD Matrix Factorization', fontsize=14, fontweight='bold')

# 1. Actual vs Predicted
ax = axes[0]
actual    = [p.r_ui for p in predictions[:500]]
predicted = [p.est  for p in predictions[:500]]
ax.scatter(actual, predicted, alpha=0.3, color='#3498db', edgecolors='none', s=20)
ax.plot([1,5],[1,5], 'r--', linewidth=2, label='Perfect prediction')
ax.set_xlabel('Actual Rating'); ax.set_ylabel('Predicted Rating')
ax.set_title('Actual vs Predicted Ratings\n(500 samples)', fontweight='bold')
ax.legend(); ax.set_xlim(0.5, 5.5); ax.set_ylim(0.5, 5.5)

# 2. Error Distribution
ax = axes[1]
errors = [p.est - p.r_ui for p in predictions]
ax.hist(errors, bins=40, color='#e74c3c', edgecolor='black', alpha=0.8)
ax.axvline(0, color='black', linestyle='--', linewidth=2)
ax.axvline(np.mean(errors), color='blue', linestyle='--', label=f'Mean: {np.mean(errors):.3f}')
ax.set_xlabel('Prediction Error (Predicted - Actual)')
ax.set_ylabel('Count')
ax.set_title('Prediction Error Distribution', fontweight='bold')
ax.legend()

# 3. RMSE vs Number of Latent Factors
ax = axes[2]
factors_list = [10, 20, 50, 100, 150]
rmse_list    = []
for f in factors_list:
    cv = cross_validate(SVD(n_factors=f, random_state=42), data, measures=['RMSE'], cv=3, verbose=False)
    rmse_list.append(cv['test_rmse'].mean())
ax.plot(factors_list, rmse_list, 'o-', color='#2ecc71', linewidth=2, markersize=8)
ax.set_xlabel('Number of Latent Factors'); ax.set_ylabel('RMSE')
ax.set_title('RMSE vs Number of Latent Factors', fontweight='bold')
ax.grid(True, alpha=0.3)
best_f = factors_list[np.argmin(rmse_list)]
ax.axvline(best_f, color='red', linestyle='--', label=f'Best: {best_f} factors')
ax.legend()

plt.tight_layout()
plt.savefig('day4_svd.png', dpi=150, bbox_inches='tight')
plt.show()

# Save model for Day 5 & 6
with open('svd_model.pkl', 'wb') as f:
    pickle.dump(svd, f)
print("\nDay 4 SVD complete! Model saved as svd_model.pkl")
