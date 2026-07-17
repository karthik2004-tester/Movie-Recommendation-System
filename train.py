import os
import time
import joblib
import numpy as np
import pandas as pd

from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise.model_selection import train_test_split as surprise_split

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    mean_absolute_error,
    mean_squared_error
)

import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# CREATE FOLDERS
# ==========================================

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ==========================================
# LOAD DATASET
# ==========================================

print("Loading Dataset...")

ratings = pd.read_csv(
    "rating.csv",
    usecols=["userId", "movieId", "rating"]
)

# Adjust according to RAM
ratings = ratings.sample(
    n=200000,
    random_state=42
)

print("Dataset Loaded")
print("Shape:", ratings.shape)

# ==========================================
# SVD RECOMMENDER TRAINING
# ==========================================

print("\nTraining SVD Recommender...")

reader = Reader(
    rating_scale=(0.5, 5)
)

data = Dataset.load_from_df(
    ratings[["userId", "movieId", "rating"]],
    reader
)

svd_trainset, svd_testset = surprise_split(
    data,
    test_size=0.2,
    random_state=42
)

svd_model = SVD(
    n_factors=100,
    n_epochs=30,
    lr_all=0.005,
    reg_all=0.02,
    random_state=42
)

start = time.time()

svd_model.fit(svd_trainset)

end = time.time()

print(
    f"SVD Training Time: {end-start:.2f} sec"
)

joblib.dump(
    svd_model,
    "models/svd_recommender.pkl"
)

# ==========================================
# SVD EVALUATION
# ==========================================

svd_predictions = svd_model.test(
    svd_testset
)

y_true_svd = np.array(
    [p.r_ui for p in svd_predictions]
)

y_pred_svd = np.array(
    [p.est for p in svd_predictions]
)

rmse = np.sqrt(
    mean_squared_error(
        y_true_svd,
        y_pred_svd
    )
)

mae = mean_absolute_error(
    y_true_svd,
    y_pred_svd
)

print(
    f"SVD RMSE : {rmse:.4f}"
)

print(
    f"SVD MAE  : {mae:.4f}"
)

# ==========================================
# RANDOM FOREST DATA
# ==========================================

ratings["liked"] = (
    ratings["rating"] >= 4
).astype(int)

print("\nCreating Features...")

movie_avg = ratings.groupby(
    "movieId"
)["rating"].mean()

user_avg = ratings.groupby(
    "userId"
)["rating"].mean()

movie_count = ratings.groupby(
    "movieId"
)["rating"].count()

ratings["movie_avg_rating"] = ratings[
    "movieId"
].map(movie_avg)

ratings["user_avg_rating"] = ratings[
    "userId"
].map(user_avg)

ratings["movie_popularity"] = ratings[
    "movieId"
].map(movie_count)

# ==========================================
# FEATURES
# ==========================================

X = ratings[
    [
        "userId",
        "movieId",
        "movie_avg_rating",
        "user_avg_rating",
        "movie_popularity"
    ]
]

y = ratings["liked"]

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================
# RANDOM FOREST
# ==========================================

print("\nTraining Random Forest...")

rf_model = RandomForestClassifier(
    n_estimators=150,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

start = time.time()

rf_model.fit(
    X_train,
    y_train
)

end = time.time()

print(
    f"RF Training Time: {end-start:.2f} sec"
)

joblib.dump(
    rf_model,
    "models/classifier.pkl"
)

# ==========================================
# RF EVALUATION
# ==========================================

print("Evaluating Model...")

y_pred = rf_model.predict(
    X_test
)

y_prob = rf_model.predict_proba(
    X_test
)[:, 1]

acc = accuracy_score(
    y_test,
    y_pred
)

prec = precision_score(
    y_test,
    y_pred
)

rec = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

auc = roc_auc_score(
    y_test,
    y_prob
)

cm = confusion_matrix(
    y_test,
    y_pred
)

# ==========================================
# RESULTS
# ==========================================

print("\n==========================")
print("MODEL EVALUATION")
print("==========================")

print(f"RMSE      : {rmse:.4f}")
print(f"MAE       : {mae:.4f}")

print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC AUC   : {auc:.4f}")

# ==========================================
# SAVE METRICS
# ==========================================

metrics = {
    "rmse": float(rmse),
    "mae": float(mae),
    "accuracy": float(acc),
    "precision": float(prec),
    "recall": float(rec),
    "f1_score": float(f1),
    "roc_auc": float(auc)
}

joblib.dump(
    metrics,
    "models/metrics.pkl"
)

# ==========================================
# CONFUSION MATRIX
# ==========================================

plt.figure(figsize=(6,4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    "outputs/confusion_matrix.png"
)

plt.close()

# ==========================================
# ROC CURVE
# ==========================================

fpr, tpr, _ = roc_curve(
    y_test,
    y_prob
)

plt.figure(figsize=(6,4))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {auc:.3f}"
)

plt.plot(
    [0,1],
    [0,1],
    "--"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "outputs/roc_curve.png"
)

plt.close()

print("\n==========================")
print("TRAINING COMPLETED")
print("==========================")

print("Saved:")
print("models/svd_recommender.pkl")
print("models/classifier.pkl")
print("models/metrics.pkl")
print("outputs/confusion_matrix.png")
print("outputs/roc_curve.png")