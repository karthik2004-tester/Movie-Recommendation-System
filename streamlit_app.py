import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import os

# =====================================================
# CONFIG
# =====================================================

API_URL = "http://127.0.0.1:8000"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="Movie Recommendation System",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.metric-container {
    background-color:#1e1e1e;
    padding:15px;
    border-radius:10px;
}

div[data-testid="stMetric"] {
    background-color:#1f2937;
    padding:12px;
    border-radius:10px;
    border:1px solid #333;
}

h1,h2,h3 {
    color:white;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================

st.markdown("""
<h1 style='text-align:center;color:#E50914;'>
🎬 Movie Recommendation System with Explainable AI
</h1>

<h4 style='text-align:center;color:gray;'>
Hybrid SVD + Content-Based Filtering + Explainable AI
</h4>

<hr>
""", unsafe_allow_html=True)

# =====================================================
# LOAD METRICS
# =====================================================

try:

    metrics = requests.get(
        f"{API_URL}/train",
        timeout=10
    ).json()

except Exception as e:

    st.error(
        "Backend not running. Start FastAPI first."
    )

    st.stop()

# =====================================================
# USER INPUT
# =====================================================

col1, col2 = st.columns(2)

with col1:

    user_id = st.number_input(
        "👤 User ID",
        min_value=1,
        value=1
    )

with col2:

    top_n = st.slider(
        "🎯 Number of Recommendations",
        5,
        20,
        10
    )

st.markdown("---")

# =====================================================
# TABS
# =====================================================

tab1, tab2, tab3 = st.tabs([
    "🎬 Recommendations",
    "🧠 Explainability",
    "📊 Dashboard"
])

# =====================================================
# TAB 1 - RECOMMENDATIONS
# =====================================================

with tab1:

    st.subheader("🔥 Personalized Movie Recommendations")

    if st.button("Generate Recommendations"):

        try:

            response = requests.get(
                f"{API_URL}/recommend/{user_id}",
                params={"top_n": top_n},
                timeout=20
            )

            data = response.json()

            if "recommendations" not in data:

                st.error(
                    data.get(
                        "error",
                        "Unable to fetch recommendations"
                    )
                )

            else:

                recs = data["recommendations"]

                if len(recs) == 0:

                    st.warning(
                        "No recommendations found."
                    )

                else:

                    df = pd.DataFrame(recs)

                    left, right = st.columns(
                        [2, 1]
                    )

                    with left:

                        st.markdown(
                            "### 🎬 Recommended Movies"
                        )

                        st.dataframe(
                            df[
                                [
                                    "title",
                                    "genres",
                                    "recommendation_score"
                                ]
                            ],
                            use_container_width=True,
                            hide_index=True
                        )

                    with right:

                        st.markdown(
                            "### 📈 Recommendation Scores"
                        )

                        chart_df = df[
                            [
                                "title",
                                "recommendation_score"
                            ]
                        ]

                        st.bar_chart(
                            chart_df.set_index(
                                "title"
                            )
                        )

                    st.markdown("---")

                    st.subheader(
                        "🎯 Recommendation Confidence"
                    )

                    for movie in recs[:5]:

                        confidence = min(
                            100,
                            max(
                                50,
                                int(
                                    movie[
                                        "recommendation_score"
                                    ] * 20
                                )
                            )
                        )

                        st.write(
                            f"**{movie['title']}**"
                        )

                        st.progress(
                            confidence / 100
                        )

                        st.caption(
                            f"Confidence: {confidence}%"
                        )

        except Exception as e:

            st.error(str(e))

# =====================================================
# TAB 2 - EXPLAINABILITY
# =====================================================

with tab2:

    st.subheader(
        "🧠 Explain Recommendation"
    )

    movie_id = st.number_input(
        "🎞 Movie ID",
        min_value=1,
        value=2571
    )

    if st.button(
        "Explain Recommendation"
    ):

        try:

            response = requests.get(
                f"{API_URL}/explain/{user_id}/{movie_id}",
                timeout=20
            )

            result = response.json()

            if "error" in result:

                st.error(result["error"])

            else:

                st.markdown(
                    f"## 🎬 {result['recommended_movie']}"
                )

                st.info(
                    result["explanation"]
                )

                st.markdown(
                    "### 📌 Supporting Movies"
                )

                if len(
                    result[
                        "supporting_movies"
                    ]
                ) == 0:

                    st.warning(
                        "No supporting movies found."
                    )

                else:

                    support_df = pd.DataFrame(
                        result[
                            "supporting_movies"
                        ]
                    )

                    st.dataframe(
                        support_df,
                        use_container_width=True,
                        hide_index=True
                    )

        except Exception as e:

            st.error(str(e))

# =====================================================
# TAB 3 - DASHBOARD
# =====================================================

with tab3:

    st.subheader(
        "📊 System Dashboard"
    )

    # =================================
    # KPI METRICS
    # =================================

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "RMSE",
        round(
            metrics.get("rmse", 0),
            4
        )
    )

    c2.metric(
        "MAE",
        round(
            metrics.get("mae", 0),
            4
        )
    )

    c3.metric(
        "Accuracy",
        round(
            metrics.get(
                "accuracy",
                0
            ),
            4
        )
    )

    c4.metric(
        "F1 Score",
        round(
            metrics.get(
                "f1_score",
                0
            ),
            4
        )
    )

    c5.metric(
        "ROC AUC",
        round(
            metrics.get(
                "roc_auc",
                0
            ),
            4
        )
    )

    st.markdown("---")

    # =================================
    # GENRE PROFILE
    # =================================

    st.subheader(
        "🎭 User Genre Preference Profile"
    )

    try:

        profile = requests.get(
            f"{API_URL}/genre-profile/{user_id}",
            timeout=10
        ).json()

        if (
            "genres" in profile
            and len(
                profile["genres"]
            ) > 0
        ):

            genre_df = pd.DataFrame(
                list(
                    profile[
                        "genres"
                    ].items()
                ),
                columns=[
                    "Genre",
                    "Score"
                ]
            )

            fig, ax = plt.subplots(
                figsize=(8, 4)
            )

            ax.bar(
                genre_df["Genre"],
                genre_df["Score"]
            )

            plt.xticks(
                rotation=45
            )

            plt.tight_layout()

            st.pyplot(fig)

        else:

            st.info(
                "No genre profile available."
            )

    except:

        st.info(
            "Genre profile unavailable."
        )

    st.markdown("---")

    # =================================
    # CONFUSION MATRIX + ROC
    # =================================

    st.subheader(
        "📈 Model Evaluation Visualizations"
    )

    cm_path = os.path.join(
        BASE_DIR,
        "outputs",
        "confusion_matrix.png"
    )

    roc_path = os.path.join(
        BASE_DIR,
        "outputs",
        "roc_curve.png"
    )

    left, right = st.columns(2)

    with left:

        if os.path.exists(cm_path):

            st.image(
                cm_path,
                caption="Confusion Matrix",
                use_container_width=True
            )

        else:

            st.warning(
                "Confusion Matrix not found."
            )

    with right:

        if os.path.exists(roc_path):

            st.image(
                roc_path,
                caption="ROC Curve",
                use_container_width=True
            )

        else:

            st.warning(
                "ROC Curve not found."
            )

    st.markdown("---")

    # =================================
    # AI QUALITY INDEX
    # =================================

    quality = (
        metrics.get(
            "f1_score",
            0
        ) * 0.4
        +
        metrics.get(
            "roc_auc",
            0
        ) * 0.4
        +
        (
            1 -
            min(
                metrics.get(
                    "rmse",
                    5
                ),
                5
            ) / 5
        ) * 0.2
    ) * 100

    st.subheader(
        "🧠 AI Quality Index"
    )

    st.progress(
        quality / 100
    )

    st.metric(
        "System Intelligence Score",
        f"{quality:.2f}/100"
    )

    st.markdown("---")

    # =================================
    # SYSTEM STATUS
    # =================================

    st.subheader(
        "⚙️ System Components"
    )

    col_a, col_b = st.columns(2)

    with col_a:

        st.success(
            "✔ Collaborative Filtering (SVD)"
        )

        st.success(
            "✔ Content-Based Filtering"
        )

        st.success(
            "✔ Hybrid Recommendation Engine"
        )

    with col_b:

        st.success(
            "✔ Explainable AI"
        )

        st.success(
            "✔ FastAPI Backend"
        )

        st.success(
            "✔ Streamlit Frontend"
        )

