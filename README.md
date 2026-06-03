# 🎬 Movie Recommendation System

A complete Movie Recommendation System built on the **MovieLens 100K dataset**, implementing three approaches — Content-Based Filtering, Collaborative Filtering, and SVD Matrix Factorization — with an interactive Streamlit web app.

---

## 📌 Project Overview

| Property | Details |
|---|---|
| Dataset | MovieLens 100K |
| Total Ratings | 100,000 |
| Total Users | 943 |
| Total Movies | 1,682 |
| Matrix Sparsity | 93.7% |
| Best RMSE | 0.9352 (SVD) |

---

## 🏗️ Project Structure

```
MovieRecommendation/
│
├── u.data              # 100K ratings (user, movie, rating, timestamp)
├── u.item              # Movie titles + genres
├── u.user              # User demographics
│
├── day1_eda.py         # Exploratory Data Analysis
├── day2_content_based.py   # Content-Based Filtering (TF-IDF + Cosine Similarity)
├── day3_collaborative.py   # Collaborative Filtering (User-Based + Item-Based)
├── day4_svd.py         # SVD Matrix Factorization
├── day5_evaluation.py  # Model Evaluation (RMSE, MAE, Precision@K, Recall@K)
├── app.py              # Streamlit Web Application
│
├── requirements.txt    # Dependencies
└── README.md
```

---

## 🔧 Approaches Implemented

### 1. Content-Based Filtering
- Extracted genre features from all 1,682 movies
- Applied **TF-IDF Vectorization** on genre strings
- Computed **Cosine Similarity** between all movie pairs
- Recommends movies with highest genre similarity to a selected movie

### 2. Collaborative Filtering
- Built a **943 × 1,682 User-Movie Rating Matrix**
- **User-Based CF**: finds similar users → recommends what they liked
- **Item-Based CF**: finds similar items → recommends based on item patterns
- Similarity computed using Cosine Similarity

### 3. SVD Matrix Factorization
- Used the **Surprise** library's SVD implementation
- Decomposed the sparse rating matrix into latent factors
- Hyperparameters: `n_factors=100`, `n_epochs=20`, `lr=0.005`, `reg=0.02`
- Trained on 80% data, evaluated on 20% held-out test set

---

## 📊 Evaluation Results

| Model | RMSE | MAE | Precision@5 | Precision@10 | Recall@10 | F1@10 |
|---|---|---|---|---|---|---|
| **SVD** | **0.9352** | **0.7375** | 0.6409 | 0.5074 | 0.5428 | 0.5245 |
| User-Based CF | 0.9802 | 0.7727 | 0.6555 | 0.5218 | 0.5753 | 0.5472 |
| Item-Based CF | 0.9726 | 0.7681 | 0.6043 | 0.4928 | 0.5184 | 0.5053 |

**SVD achieves the best RMSE and MAE**, making it the most accurate model for predicting user ratings.

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/movie-recommendation-system.git
cd movie-recommendation-system
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download **MovieLens 100K** from [grouplens.org](https://grouplens.org/datasets/movielens/100k/)
and place `u.data`, `u.item`, `u.user` in the project folder.

### 4. Run individual scripts
```bash
python day1_eda.py           # EDA + visualizations
python day2_content_based.py # Content-Based Filtering
python day3_collaborative.py # Collaborative Filtering
python day4_svd.py           # SVD Matrix Factorization
python day5_evaluation.py    # Model Evaluation
```

### 5. Launch the Streamlit App
```bash
streamlit run app.py
```
Opens at **http://localhost:8501**

---

## 🖥️ Streamlit App Features

- **Tab 1 — Content-Based**: Select any movie → get top-N similar movies by genre
- **Tab 2 — SVD Recommendations**: Enter User ID (1–943) → get personalized recommendations
- **Tab 3 — Model Comparison**: Full metrics table + project summary

---

## 📈 Key Insights from EDA

- Average rating: **3.53 / 5** (users tend to rate positively)
- Most rated movie: **Star Wars (1977)** with 583 ratings
- Most active user rated **737 movies**
- Matrix sparsity of **93.7%** highlights the cold-start challenge in recommendation systems

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3 |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, Scikit-surprise |
| Visualization | Matplotlib, Seaborn |
| Web App | Streamlit |
| Version Control | Git, GitHub |

---

## 👤 Author

**Naveen T**
Pre-Final Year B.E. Computer Science Engineering
- 📧 naveent1905@gmail.com
- 💻 [LeetCode](https://leetcode.com) | [LinkedIn](https://linkedin.com)

---

## 📄 Dataset Reference

F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context.
ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 29:1–29:19.
