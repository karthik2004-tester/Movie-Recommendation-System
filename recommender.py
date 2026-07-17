import pandas as pd
import joblib
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# LOAD DATA
# ==========================================

ratings = pd.read_csv(
    "rating.csv",
    usecols=["userId", "movieId", "rating"]
)

movies = pd.read_csv(
    "movie.csv"
)

svd_model = joblib.load(
    "models/svd_recommender.pkl"
)

# ==========================================
# MOVIE POPULARITY
# ==========================================

movie_popularity = ratings.groupby(
    "movieId"
)["rating"].count()

max_popularity = movie_popularity.max()

# ==========================================
# CONTENT FEATURES
# ==========================================

movies["genres"] = movies["genres"].fillna("")

tfidf = TfidfVectorizer()

genre_matrix = tfidf.fit_transform(
    movies["genres"]
)

genre_similarity = cosine_similarity(
    genre_matrix
)

# ==========================================
# MOVIE INDEX MAPPING
# ==========================================

movie_id_to_index = {
    movie_id: idx
    for idx, movie_id in enumerate(
        movies["movieId"].values
    )
}

# ==========================================
# RECOMMENDATION FUNCTION
# ==========================================

def recommend_movies(
    user_id,
    top_n=10
):

    user_ratings = ratings[
        ratings["userId"] == user_id
    ]

    if user_ratings.empty:

        return {
            "error":
            f"User {user_id} not found"
        }

    watched_movies = set(
        user_ratings["movieId"]
    )

    liked_movies = user_ratings[
        user_ratings["rating"] >= 4
    ]["movieId"].tolist()

    recommendations = []

    candidate_movies = movies[
        ~movies["movieId"].isin(
            watched_movies
        )
    ]

    for movie_id in candidate_movies["movieId"]:

        # ==================================
        # SVD SCORE
        # ==================================

        svd_score = svd_model.predict(
            user_id,
            movie_id
        ).est

        # ==================================
        # TF-IDF SIMILARITY
        # ==================================

        similarity_score = 0

        if liked_movies:

            total_similarity = 0
            valid_movies = 0

            for liked_movie in liked_movies:

                if (
                    liked_movie in movie_id_to_index
                    and
                    movie_id in movie_id_to_index
                ):

                    i = movie_id_to_index[
                        liked_movie
                    ]

                    j = movie_id_to_index[
                        movie_id
                    ]

                    total_similarity += (
                        genre_similarity[i][j]
                    )

                    valid_movies += 1

            if valid_movies > 0:

                similarity_score = (
                    total_similarity /
                    valid_movies
                )

        # ==================================
        # POPULARITY SCORE
        # ==================================

        popularity_score = (
            movie_popularity.get(
                movie_id,
                0
            )
            /
            max_popularity
        )

        # ==================================
        # FINAL HYBRID SCORE
        # ==================================

        final_score = (
            0.60 * svd_score
            +
            0.25 * similarity_score
            +
            0.15 * popularity_score
        )

        recommendations.append(
            (
                movie_id,
                final_score
            )
        )

    # ==================================
    # SORT
    # ==================================

    recommendations.sort(
        key=lambda x: x[1],
        reverse=True
    )

    # ==================================
    # FORMAT RESULTS
    # ==================================

    results = []

    for movie_id, score in recommendations[:top_n]:

        movie_row = movies[
            movies["movieId"] == movie_id
        ]

        if movie_row.empty:
            continue

        movie = movie_row.iloc[0]

        results.append({

            "movie_id":
            int(movie_id),

            "title":
            movie["title"],

            "genres":
            movie["genres"],

            "recommendation_score":
            round(float(score), 3)

        })

    return results

# ==========================================
# TEST
# ==========================================

if __name__ == "__main__":

    recs = recommend_movies(
        user_id=1,
        top_n=10
    )

    for movie in recs:
        print(movie)