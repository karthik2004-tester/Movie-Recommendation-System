# 🎬 Hybrid Movie Recommendation System with Explainable AI

A Hybrid Movie Recommendation System that combines **Collaborative Filtering (SVD)**, **Content-Based Filtering (TF-IDF + Cosine Similarity)**, and **Explainable AI (XAI)** to generate personalized movie recommendations with transparent explanations.

The system is built using **Python**, **FastAPI**, and **Streamlit**, providing an interactive web interface and REST API for recommendations and explainability.

---

## 📌 Features

- 🎯 Personalized movie recommendations
- 🤝 Collaborative Filtering using SVD
- 🎭 Content-Based Filtering using TF-IDF
- 📊 Hybrid recommendation engine
- 🧠 Explainable AI for recommendation transparency
- 🎬 User genre preference profiling
- 📈 Model evaluation dashboard
- 🌐 FastAPI backend
- 💻 Streamlit frontend
- 📉 Confusion Matrix and ROC Curve visualization

---

# 🏗️ System Architecture

```
                  MovieLens Dataset
                          │
                          ▼
                 Data Preprocessing
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
   SVD Collaborative Model       TF-IDF Content Model
          │                               │
          └───────────────┬───────────────┘
                          ▼
                 Hybrid Recommendation
                          │
                          ▼
               Explainable AI Module
                          │
                          ▼
               FastAPI REST Backend
                          │
                          ▼
                 Streamlit Dashboard
```

---

# 🛠️ Tech Stack

### Programming Language

- Python

### Machine Learning

- Scikit-learn
- Surprise (SVD)
- TF-IDF Vectorizer
- Cosine Similarity
- Random Forest Classifier

### Backend

- FastAPI
- Uvicorn

### Frontend

- Streamlit

### Data Processing

- Pandas
- NumPy

### Visualization

- Matplotlib
- Seaborn

### Model Storage

- Joblib

---

# 📂 Project Structure

```
Movie-Recommendation-System/
│
├── app.py
├── train.py
├── recommender.py
├── explain.py
├── streamlit_app.py
├── requirements.txt
├── .gitignore
│
├── models/
│
├── outputs/
│
└── README.md
```

---

# 🚀 Recommendation Pipeline

### Step 1

Load MovieLens Dataset

↓

### Step 2

Train SVD Collaborative Filtering Model

↓

### Step 3

Generate TF-IDF Genre Features

↓

### Step 4

Calculate Cosine Similarity

↓

### Step 5

Compute Movie Popularity

↓

### Step 6

Hybrid Recommendation Score

```
Final Score =
0.60 × SVD Score
+ 0.25 × Genre Similarity
+ 0.15 × Popularity Score
```

↓

### Step 7

Generate Top-N Recommendations

↓

### Step 8

Explain Recommendation

---

# 🧠 Explainable AI

The recommendation explanation is generated using:

- User's highly rated movies
- Genre overlap
- User genre profile
- Similar previously liked movies

Example:

> "The Matrix (1999) was recommended because it matches your preferred genres (Adventure, Fantasy, Action) and is similar to movies you rated highly, such as Blade Runner (1982)."

---

# 📊 Model Evaluation

## Recommendation Model (SVD)

| Metric | Value |
|---------|--------|
| RMSE | **0.9441** |
| MAE | **0.7330** |

---

## Classification Model

| Metric | Value |
|---------|--------|
| Accuracy | **80.23%** |
| Precision | **81.50%** |
| Recall | **78.27%** |
| F1 Score | **79.86%** |
| ROC-AUC | **88.73%** |

---

# 📈 Visualizations

The application provides:

- Confusion Matrix
- ROC Curve
- User Genre Distribution
- Recommendation Confidence
- AI Quality Index

---

# 🌐 API Endpoints

| Endpoint | Description |
|-----------|-------------|
| `/` | Home |
| `/recommend/{user_id}` | Movie recommendations |
| `/explain/{user_id}/{movie_id}` | Explain recommendation |
| `/genre-profile/{user_id}` | User genre profile |
| `/movie/{movie_id}` | Movie details |
| `/train` | Model metrics |
| `/health` | API health |

---

# 💻 Streamlit Dashboard

The dashboard includes:

- Recommendation Generation
- Explainable AI
- User Genre Profile
- Model Metrics
- ROC Curve
- Confusion Matrix
- AI Quality Index

---

# ▶️ Installation

Clone the repository

```bash
git clone https://github.com/karthik2004-tester/Movie-Recommendation-System.git

cd Movie-Recommendation-System
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Train the Models

```bash
python train.py
```

---

# ▶️ Start FastAPI

```bash
python -m uvicorn app:app --reload
```

---

# ▶️ Launch Streamlit

```bash
streamlit run streamlit_app.py
```

---

# 📊 Dataset

This project uses the **MovieLens Latest Dataset**.

Due to GitHub's file size limitations, the dataset is **not included** in this repository.

Download the dataset from:

https://grouplens.org/datasets/movielens/latest/

Place the required CSV files in the project root before training.

Required files:

- rating.csv
- movie.csv
- genome_scores.csv
- genome_tags.csv
- tag.csv
- link.csv

---

# 🔮 Future Improvements

- Deep Learning Recommender Models
- Neural Collaborative Filtering (NCF)
- BERT-based Movie Embeddings
- Transformer-based Recommendation
- Real-time Recommendation Updates
- User Authentication
- Docker Deployment
- Cloud Deployment (Render/AWS)

---



GitHub:
https://github.com/karthik2004-tester

---

# ⭐ If you found this project useful, consider giving it a star!
