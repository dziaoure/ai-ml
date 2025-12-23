# Product Recommendation System (MovieLens)

An end-to-end **movie recommendation system** built using the MovieLens dataset.
This project demonstrates how real-world recommender systems evolve from
simple baselines to personalized, explainable models.

The system includes:
- Popularity-based recommendations (baseline)
- Item-based collaborative filtering (cosine similarity)
- Automatic cold-start fallback
- An interactive Streamlit demo with lightweight explanations

---

## 🎯 Problem Statement

Given a user’s historical interactions (movie ratings),  
**recommend the top-K movies** they are most likely to enjoy next.

This mirrors common personalization problems in:
- e-commerce
- media streaming
- content discovery platforms

---

## 📦 Dataset

- **MovieLens (ml-latest-small)**
- Source: GroupLens Research
- Files used:
  - `ratings.csv`
  - `movies.csv`

Each rating includes:
- `userId`
- `movieId`
- `rating`
- `timestamp`

---

## 🧠 Approach

### 1. Data Splitting
- **Time-based split per user**
- Prevents data leakage
- Simulates real recommendation scenarios

### 2. Baseline Model — Popularity
- Movies ranked by:
  - number of ratings
  - average rating
- Strong, non-personalized baseline
- Used as a **fallback for cold-start users**

### 3. Personalized Model — Item-Based Collaborative Filtering
- Builds a sparse **user–item interaction matrix**
- Computes **cosine similarity** between items
- Recommends items similar to those the user has interacted with
- Excludes previously seen items

### 4. Production Logic
- If collaborative filtering cannot produce recommendations:
  → automatically falls back to popularity
- Ensures a robust, always-on experience

---

## 📊 Evaluation

Models are evaluated using:
- Precision@K
- Recall@K
- MAP@K

Evaluation is performed on the **future (test) interactions** only,
making the metrics realistic and leakage-free.

---

## 🖥️ Interactive Demo (Streamlit)

The project includes a Streamlit app that allows you to:
- Select a user
- Choose Top-K recommendations
- Switch between models
- View recent user history
- See **“Because you watched…”** explanations for filtering-based recommendations

### Run the app:
```bash
streamlit run app/app.py
