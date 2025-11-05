# ========== CBF (TF-IDF + Cosine Similarity) ==========

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load train/test data
train_df = pd.read_csv("train_data.csv")
test_df = pd.read_csv("test_data.csv")
recipes = pd.read_csv("preprocessed_data.csv")[["recipe_id", "text"]]

# Reduce item space (For improve execution speed)
top_items = (
    train_df["recipe_id"].value_counts()
    .nlargest(5000)  # 3000~10000
    .index
)
recipes = recipes[recipes["recipe_id"].isin(top_items)]

# Vectorize recipe text
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(train_df["text"].fillna(""))

# recipe id <-> index mapping
recipe_idx = {rid: i for i, rid in enumerate(train_df["recipe_id"])}

# seen recipe (train)
user_train_items = train_df.groupby("user_id")["recipe_id"].apply(set)

cbf_results = []

for uid in train_df["user_id"].unique():
    # user liked (rating >= 4) recipe
    liked = train_df[(train_df["user_id"] == uid) & (train_df["rating"] >= 4)]
    liked_idx = [recipe_idx[rid] for rid in liked["recipe_id"] if rid in recipe_idx]

    if not liked_idx:
        continue

    # Vector (user profile)
    profile_vec = np.asarray(tfidf_matrix[liked_idx].mean(axis=0))

    # Calculate cosine similarity (all recipes)
    sims = cosine_similarity(profile_vec, tfidf_matrix).ravel()

    # Save Top-N (Improve speed)
    top_n = 100
    top_idx = sims.argsort()[-top_n:][::-1]

    for idx in top_idx:
        cbf_results.append((uid, recipes.iloc[idx]["recipe_id"], sims[idx]))

cbf_df = pd.DataFrame(cbf_results, columns=["user_id", "recipe_id", "cbf_score"])
cbf_df["cbf_score"] = cbf_df["cbf_score"].clip(0, 1)

# Normalizing CBF score
scaler = MinMaxScaler()
cbf_df["cbf_normalized_score"] = scaler.fit_transform(cbf_df[["cbf_score"]])

# Save CBF scores data
cbf_df.to_csv("cbf_scores.csv", index=False)
print("\nSaved: cbf_scores.csv")
