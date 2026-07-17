import pandas as pd
from collections import Counter

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

movie_map = movies.set_index(
    "movieId"
).to_dict("index")

# ==========================================
# USER GENRE PROFILE
# ==========================================

def get_user_genre_profile(user_id):

    user_data = ratings[
        (ratings["userId"] == user_id)
        &
        (ratings["rating"] >= 4)
    ]

    if user_data.empty:

        return {
            "genres": {},
            "message": "No sufficient user history found."
        }

    genre_counter = Counter()

    for movie_id in user_data["movieId"]:

        if movie_id not in movie_map:
            continue

        genres = movie_map[movie_id]["genres"]

        for genre in genres.split("|"):

            if genre != "(no genres listed)":
                genre_counter[genre] += 1

    total = sum(
        genre_counter.values()
    )

    genre_scores = {}

    for genre, count in genre_counter.items():

        genre_scores[genre] = round(
            count / total,
            3
        )

    genre_scores = dict(
        sorted(
            genre_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
    )

    return {
        "genres": genre_scores
    }

# ==========================================
# EXPLAIN RECOMMENDATION
# ==========================================

def explain_recommendation(
    user_id,
    movie_id
):

    if movie_id not in movie_map:

        return {
            "error": "Movie not found"
        }

    target_movie = movie_map[movie_id]

    target_title = target_movie["title"]

    target_genres = set(
        target_movie["genres"].split("|")
    )

    user_data = ratings[
        ratings["userId"] == user_id
    ]

    liked_movies = user_data[
        user_data["rating"] >= 4
    ]["movieId"].tolist()

    supporting_movies = []

    for liked_movie_id in liked_movies[:30]:

        if liked_movie_id not in movie_map:
            continue

        liked_movie = movie_map[
            liked_movie_id
        ]

        liked_genres = set(
            liked_movie["genres"].split("|")
        )

        shared_genres = list(
            target_genres.intersection(
                liked_genres
            )
        )

        if len(shared_genres) > 0:

            supporting_movies.append({

                "liked_movie":
                liked_movie["title"],

                "shared_genres":
                shared_genres,

                "overlap_score":
                len(shared_genres)

            })

    supporting_movies.sort(
        key=lambda x: x["overlap_score"],
        reverse=True
    )

    top_supporting = supporting_movies[:5]

    profile = get_user_genre_profile(
        user_id
    )

    top_genres = list(
        profile["genres"].keys()
    )[:3]

    if top_supporting:

        explanation_text = (
            f"'{target_title}' was recommended because "
            f"it matches your preferred genres "
            f"({', '.join(top_genres)}). "
            f"It is also similar to movies you previously rated highly, "
            f"such as '{top_supporting[0]['liked_movie']}'."
        )

    else:

        explanation_text = (
            f"'{target_title}' was recommended because "
            f"it matches your historical preferences "
            f"for genres such as "
            f"{', '.join(top_genres)}."
        )

    return {

        "recommended_movie":
        target_title,

        "genres":
        target_movie["genres"],

        "user_top_genres":
        top_genres,

        "supporting_movies":
        top_supporting,

        "explanation":
        explanation_text

    }

# ==========================================
# TEST
# ==========================================

if __name__ == "__main__":

    from pprint import pprint

    print("\nEXPLANATION TEST\n")

    pprint(
        explain_recommendation(
            user_id=1,
            movie_id=2571
        )
    )

    print("\nGENRE PROFILE TEST\n")

    pprint(
        get_user_genre_profile(1)
    )