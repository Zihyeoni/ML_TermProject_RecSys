# Content-Based Filtering with Explainability + Train/Test Split Evaluation
# -------------------------------------------------------------
# - Uses preprocessed_data.csv (columns: user_id, recipe_id, rating, name, text)
# - Deduplicates recipes before TF-IDF
# - Provides explainable output showing top overlapping terms
# - Supports both existing and new users
# - Includes user-level train/test split for fair evaluation
# -------------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import argparse
import random

# 1️ Load & Prepare Data
DATA_PATH = "preprocessed_data.csv"

df = pd.read_csv(DATA_PATH)
df["user_id"] = df["user_id"].astype(str)

# Aggregate to unique recipe level
recipes_rating = df.groupby(["recipe_id", "name"])["rating"].mean().reset_index()
recipes_text = df.groupby("recipe_id")["text"].first().reset_index()
recipes = recipes_rating.merge(recipes_text, on="recipe_id")

# 2️ TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(recipes["text"])

# 3️ Core Functions
def _to_profile_from_indices(idx_array):
    """Average TF-IDF vectors for selected recipes → user profile vector"""
    if len(idx_array) == 0:
        return None
    return np.asarray(tfidf_matrix[idx_array].mean(axis=0))

def get_user_profile(user_id: str, like_threshold: float = 4.0):
    """Build user preference profile based on liked recipes"""
    user_rows = df[df["user_id"] == user_id]
    liked = user_rows[user_rows["rating"] >= like_threshold]
    if liked.empty:
        return None
    liked_idx = recipes[recipes["recipe_id"].isin(liked["recipe_id"])].index.values
    return _to_profile_from_indices(liked_idx)

def recommend_from_vector(user_profile, top_n=5, exclude_recipe_ids=None):
    """Compute cosine similarity and return top-N recommendations"""
    if user_profile is None:
        return pd.DataFrame(columns=["recipe_id", "name"])
    sim = cosine_similarity(user_profile, tfidf_matrix).flatten()
    order = sim.argsort()[::-1]
    if exclude_recipe_ids is not None and len(exclude_recipe_ids) > 0:
        mask = ~recipes["recipe_id"].isin(set(exclude_recipe_ids))
        order = [i for i in order if mask[i]]
    top_idx = order[:top_n]
    return recipes.iloc[top_idx][["recipe_id", "name"]].reset_index(drop=True)

def recommend_for_user(user_id: str, top_n=5):
    """Personalized recommendations for existing user"""
    profile = get_user_profile(user_id)
    if profile is None:
        return None
    already = df[(df["user_id"] == user_id) & (df["rating"] >= 4)]["recipe_id"].unique()
    return recommend_from_vector(profile, top_n=top_n, exclude_recipe_ids=already)

def recommend_popular(top_n=5):
    """Fallback: Top-rated popular recipes"""
    top_ids = recipes.sort_values("rating", ascending=False).head(top_n)["recipe_id"]
    return recipes[recipes["recipe_id"].isin(top_ids)][["recipe_id", "name"]].reset_index(drop=True)

def recommend_by_selection(favorite_ids, top_n=5):
    """Cold-start for new user: build profile from selected favorite IDs"""
    liked_idx = recipes[recipes["recipe_id"].isin(favorite_ids)].index.values
    profile = _to_profile_from_indices(liked_idx)
    return recommend_from_vector(profile, top_n=top_n, exclude_recipe_ids=favorite_ids)

# 4️ Explainability
def explain_recommendation(user_profile, rec_idx, top_terms=5):
    """Find overlapping important terms between user profile and recipe"""
    feature_names = tfidf.get_feature_names_out()

    user_vec = np.array(user_profile).flatten()
    user_top_idx = user_vec.argsort()[::-1][:50]

    recipe_vec = tfidf_matrix[rec_idx].toarray().flatten()
    recipe_top_idx = recipe_vec.argsort()[::-1][:50]

    common_idx = list(set(user_top_idx) & set(recipe_top_idx))
    if not common_idx:
        return []

    sorted_common = sorted(common_idx, key=lambda i: user_vec[i] + recipe_vec[i], reverse=True)
    top_common = [feature_names[i] for i in sorted_common[:top_terms]]
    return top_common

# 5️ Visualization
def visualize_recommendations(rec_df):
    """Display WordCloud and TF-IDF top terms"""
    if rec_df is None or rec_df.empty:
        print("⚠️ No recommendations to visualize.")
        return

    text = " ".join(recipes[recipes["recipe_id"].isin(rec_df["recipe_id"])]["text"].astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Keywords in Recommended Recipes", fontsize=18)
    plt.show()

    first_id = rec_df.iloc[0]["recipe_id"]
    ridx = recipes[recipes["recipe_id"] == first_id].index[0]
    feature_names = tfidf.get_feature_names_out()
    tfidf_vector = tfidf_matrix[ridx].toarray().flatten()
    top_indices = tfidf_vector.argsort()[::-1][:10]

    plt.figure(figsize=(8, 5))
    plt.barh(range(10), tfidf_vector[top_indices][::-1])
    plt.yticks(range(10), [feature_names[i] for i in top_indices][::-1])
    plt.xlabel("TF-IDF Weight")
    plt.title(f"Top Terms of Recipe {first_id}")
    plt.tight_layout()
    plt.show()

# 6️ Train/Test Split Evaluation
def split_train_test(df, test_size=0.2, seed=42):
    """Split each user's ratings into train/test sets"""
    train_rows, test_rows = [], []
    for uid, group in df.groupby("user_id"):
        if len(group) < 2:
            train_rows.append(group)
            continue
        train, test = train_test_split(group, test_size=test_size, random_state=seed)
        train_rows.append(train)
        test_rows.append(test)
    return pd.concat(train_rows), pd.concat(test_rows)

def evaluate_hit_rate_at_k_split(k=5, test_size=0.2, seed=42):
    """Evaluate Hit@K using per-user train/test split"""
    train_df, test_df = split_train_test(df, test_size=test_size, seed=seed)
    users = test_df["user_id"].unique()

    hits, evaluated = 0, 0

    for uid in users:
        train_user = train_df[train_df["user_id"] == uid]
        test_user = test_df[test_df["user_id"] == uid]

        liked_train = train_user[train_user["rating"] >= 4]["recipe_id"].tolist()
        liked_test = test_user[test_user["rating"] >= 4]["recipe_id"].tolist()
        if len(liked_train) == 0 or len(liked_test) == 0:
            continue

        idx = recipes[recipes["recipe_id"].isin(liked_train)].index.values
        profile = _to_profile_from_indices(idx)

        rec_df = recommend_from_vector(profile, top_n=k, exclude_recipe_ids=liked_train)
        rec_list = rec_df["recipe_id"].tolist()

        if any(r in rec_list for r in liked_test):
            hits += 1
        evaluated += 1

    hit_rate = hits / evaluated if evaluated > 0 else 0
    print(f"[Split Eval] Hit@{k}: {hit_rate:.3f} (users evaluated: {evaluated})")
    return hit_rate

# 7️ Interactive CLI
def main():
    parser = argparse.ArgumentParser(description="Explainable CBF with Train/Test Split")
    parser.add_argument("--mode", type=str, default="run", help="run | eval")
    parser.add_argument("--k", type=int, default=5, help="Top-K recommendations")
    args = parser.parse_args()

    if args.mode == "eval":
        evaluate_hit_rate_at_k_split(k=args.k)
        return

    print("✨ Recipe Recommendation System (Preprocessed, Explainable + Split) ✨")
    mode = input("Type 'user' for existing user, or 'new' for a new user: ").strip().lower()

    if mode == "user":
        user_id = input("Enter user_id: ").strip()
        result = recommend_for_user(user_id)
        if result is None:
            print("⚠️ No preference data found. Showing popular recipes.\n")
            pop_df = recommend_popular()
            print(pop_df)
            visualize_recommendations(pop_df)
        else:
            print("\n🍽️ Personalized Recommendations (with reasons):")
            profile = get_user_profile(user_id)
            for _, row in result.iterrows():
                rec_idx = recipes[recipes["recipe_id"] == row["recipe_id"]].index[0]
                reasons = explain_recommendation(profile, rec_idx)
                reason_text = ", ".join(reasons) if reasons else "general similarity"
                print(f"{row['name']}  ← similar terms: {reason_text}")
            visualize_recommendations(result)

    elif mode == "new":
        print("\n🔰 New user mode!")
        print("Enter 2–3 favorite recipe IDs (e.g., 12345,67890):")
        favorite_ids = input("Input: ").split(",")
        favorite_ids = [int(x.strip()) for x in favorite_ids if x.strip().isdigit()]

        if len(favorite_ids) == 0:
            print("⚠️ No input detected. Showing popular recipes instead.")
            pop_df = recommend_popular()
            print(pop_df)
            visualize_recommendations(pop_df)
        else:
            print("\n🍳 Recommendations similar to your selections:")
            sel_df = recommend_by_selection(favorite_ids)
            print(sel_df)
            visualize_recommendations(sel_df)
    else:
        print("⚠️ Invalid input. Showing popular recipes.")
        pop_df = recommend_popular()
        print(pop_df)
        visualize_recommendations(pop_df)

if __name__ == "__main__":
    main()
