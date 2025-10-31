# =====================================================
# Hybrid Recommendation System (CBF + CF)
# Features:
# - Merge CBF & CF scores to compute hybrid
# - Hit@K evaluation
# - Top-N recommendations per user
# - CLI input for user_id
# - Visualization of hybrid scores for input user
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
#  Load and Normalize Data
# -------------------------------
cbf_df = pd.read_csv("cbf_scores.csv")  # user_id, recipe_id, cbf_score
cf_df = pd.read_csv("cf_recommendations.csv")  # user_id, recipe_id, pred_rating
ratings_df = pd.read_csv("preprocessed_data.csv")[["user_id", "recipe_id", "rating"]]

# Load recipe names for display purposes
recipes_full_df = pd.read_csv("preprocessed_data.csv")[["recipe_id", "name"]].drop_duplicates()

# Ensure all key columns are treated as strings for consistent merging
cbf_df["user_id"] = cbf_df["user_id"].astype(str)
cf_df["user_id"] = cf_df["user_id"].astype(str)
ratings_df["user_id"] = ratings_df["user_id"].astype(str)

cbf_df["recipe_id"] = cbf_df["recipe_id"].astype(str)
cf_df["recipe_id"] = cf_df["recipe_id"].astype(str)
ratings_df["recipe_id"] = ratings_df["recipe_id"].astype(str)
recipes_full_df["recipe_id"] = recipes_full_df["recipe_id"].astype(str)

# -------------------------------
#  Merge & compute hybrid score
# -------------------------------
alpha = 0.6  # weight for CBF
df = cbf_df.merge(cf_df, on=["user_id", "recipe_id"], how="inner")
df = df.merge(ratings_df, on=["user_id", "recipe_id"], how="inner")
df = df.merge(recipes_full_df, on="recipe_id", how="left")

# 💡NORMALIZE CF SCORE (pred_rating) to [0, 1]
MIN_RATING = 1
MAX_RATING = 5
RANGE = MAX_RATING - MIN_RATING
# Min-Max Normalization Formula: (X - Min) / (Max - Min)
df["cf_normalized_score"] = (df["pred_rating"] - MIN_RATING) / RANGE

df["hybrid_score"] = alpha * df["cbf_score"] + (1 - alpha) * df["cf_normalized_score"]


# -------------------------------
# ⃣ Train/Test Split
# -------------------------------
test_size = 0.2
df = df.sample(frac=1, random_state=42)
train_len = int(len(df) * (1 - test_size))
train_df = df.iloc[:train_len].copy()
test_df = df.iloc[train_len:].copy()

# -------------------------------
#  Evaluation ( Hit@K)

def hit_rate_at_k(df, k=5):
    hits, evaluated = 0, 0
    for uid, group in df.groupby("user_id"):
        top_k = group.sort_values("hybrid_score", ascending=False).head(k)["recipe_id"].tolist()
        liked = group[group["rating"] >= 4]["recipe_id"].tolist()
        if liked:
            evaluated += 1
            if any(r in top_k for r in liked):
                hits += 1
    return hits / evaluated if evaluated > 0 else 0
hit5 = hit_rate_at_k(test_df, k=5)
hit10 = hit_rate_at_k(test_df, k=10)


# -------------------------------
# Generate Top-N Recommendations for All Users
# -------------------------------
top_n = 10
cols_to_keep = ["user_id", "recipe_id", "name", "hybrid_score"]
top_df = df.groupby("user_id").apply(lambda x: x.nlargest(top_n, "hybrid_score")).reset_index(drop=True)
top_df = top_df[cols_to_keep]  # Lấy thêm cột name
top_df.to_csv("hybrid.csv", index=False)


# -------------------------------
#  User Input and Visualization
# -------------------------------
while True:
    user_id = input("Enter user_id to view recommendations (or 'exit' to quit): ").strip()
    if user_id.lower() == "exit":
        print("👋 Exiting hybrid recommender. Goodbye!")
        break

    user_rec = top_df[top_df["user_id"] == user_id].sort_values("hybrid_score", ascending=False)

    if user_rec.empty:
        print(f"⚠️ No recommendations found for user {user_id}. Try another ID.\n")
        print("💡 Example user_ids:", top_df["user_id"].drop_duplicates().sample(5, random_state=42).to_list())
    else:
        print(f"\n🍽️ Top-{top_n} Hybrid Recommendations for User {user_id}:")
        print(user_rec[["recipe_id", "name", "hybrid_score"]].to_string(index=False))

        plt.figure(figsize=(10, 5))
        plt.bar(user_rec["recipe_id"].astype(str), user_rec["hybrid_score"], color="skyblue")
        plt.xlabel("Recipe ID")
        plt.ylabel("Hybrid Score")
        plt.title(f"Top-{top_n} Hybrid Recommendations for User {user_id}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(2)  # 2s
        plt.close()
        print()
