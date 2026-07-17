from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib

from recommender import recommend_movies
from explain import (
    explain_recommendation,
    get_user_genre_profile
)

# =====================================
# APP
# =====================================

app = FastAPI(
    title="Movie Recommendation XAI API",
    description="Hybrid Movie Recommendation System using SVD + TF-IDF Content Filtering + Explainable AI",
    version="1.0"
)

# =====================================
# LOAD MODEL + DATA
# =====================================

try:
    model = joblib.load("models/svd_recommender.pkl")
except FileNotFoundError:
    model = None
    print("WARNING: models/svd_recommender.pkl not found")

try:
    metrics = joblib.load("models/metrics.pkl")
except FileNotFoundError:
    metrics = {
        "rmse": 0,
        "mae": 0,
        "accuracy": 0,
        "precision": 0,
        "recall": 0,
        "f1_score": 0,
        "roc_auc": 0
    }

ratings_df = pd.read_csv(
    "rating.csv",
    usecols=["userId", "movieId", "rating"]
)

movies_df = pd.read_csv("movie.csv")

# =====================================
# HOME
# =====================================

@app.get("/")
def home():

    return {
        "message": "Movie Recommendation API",
        "status": "Running"
    }

# =====================================
# TRAIN METRICS
# =====================================

@app.get("/train")
def train_metrics():

    return {
        "rmse": metrics.get("rmse", 0),
        "mae": metrics.get("mae", 0),
        "accuracy": metrics.get("accuracy", 0),
        "precision": metrics.get("precision", 0),
        "recall": metrics.get("recall", 0),
        "f1_score": metrics.get("f1_score", 0),
        "roc_auc": metrics.get("roc_auc", 0)
    }

# =====================================
# RECOMMEND MOVIES
# =====================================

@app.get("/recommend/{user_id}")
def recommend(user_id: int, top_n: int = 10):

    try:

        recommendations = recommend_movies(
            user_id,
            top_n
        )

        return {
            "user_id": user_id,
            "total_recommendations": len(
                recommendations
            ),
            "recommendations": recommendations
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# =====================================
# EXPLAIN RECOMMENDATION
# =====================================

@app.get("/explain/{user_id}/{movie_id}")
def explain(user_id: int, movie_id: int):

    try:

        result = explain_recommendation(
            user_id,
            movie_id
        )

        return result

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# =====================================
# USER GENRE PROFILE
# =====================================

@app.get("/genre-profile/{user_id}")
def genre_profile(user_id: int):

    try:

        return get_user_genre_profile(
            user_id
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# =====================================
# MOVIE DETAILS
# =====================================

@app.get("/movie/{movie_id}")
def movie_details(movie_id: int):

    movie = movies_df[
        movies_df["movieId"] == movie_id
    ]

    if movie.empty:

        raise HTTPException(
            status_code=404,
            detail="Movie not found"
        )

    movie = movie.iloc[0]

    return {

        "movie_id":
        int(movie["movieId"]),

        "title":
        movie["title"],

        "genres":
        movie["genres"]
    }

# =====================================
# HEALTH CHECK
# =====================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "api": "running",
        "svd_model_loaded": model is not None,
        "recommendation_engine": "Hybrid SVD + TF-IDF Content Filtering",
        "xai_enabled": True,
        "genre_profile_enabled": True
    }

# =====================================
# RUN DIRECTLY
# =====================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )