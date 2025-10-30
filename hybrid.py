import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import argparse


# Load & Prepare Data
df = pd.read_csv('preprocessed_data.csv')
df["user_id"] = df["user_id"].astype(str)

# Aggregate recipe-level info
recipes_rating = df.groupby(["recipe_id", "name"])["rating"].mean().reset_index()
recipes_text = df.groupby("recipe_id")["text"].first().reset_index()
recipes = recipes_rating.merge(recipes_text, on="recipe_id")


#TF-IDF Vectorization (CBF)
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(recipes["text"])

def _to_profile_from_indices(idx_array):
    if len(idx_array) == 0:
        return None
    return np.asarray(tfidf_matrix[idx_array].mean(axis=0))

def get_user_profile(user_id: str, like_threshold: float = 4.0):
    user_rows = df[df["user_id"] == user_id]
    liked = user_rows[user_rows["rating"] >= like_threshold]
    if liked.empty:
        return None
    liked_idx = recipes[recipes["recipe_id"].isin(liked["recipe_id"])].index.values
    return _to_profile_from_indices(liked_idx)

def get_cbf_scores(user_id):
    profile = get_user_profile(user_id)
    if profile is None:
        return pd.Series(0, index=recipes["recipe_id"])
    sim = cosine_similarity(profile, tfidf_matrix).flatten()
    return pd.Series(sim, index=recipes["recipe_id"])


#Collaborative Filtering (CF)

# Dummy CF function: Replace with your group's CF function
rating_matrix = df.pivot_table(index="user_id", columns="recipe_id", values="rating").fillna(0)
user_sim = cosine_similarity(rating_matrix)
user_sim_df = pd.DataFrame(user_sim, index=rating_matrix.index, columns=rating_matrix.index)

def get_cf_scores(user_id):
    if user_id not in rating_matrix.index:
        return pd.Series(0, index=recipes["recipe_id"])
    sim_scores = user_sim_df[user_id]
    weighted_ratings = rating_matrix.T.dot(sim_scores) / sim_scores.sum()
    weighted_ratings = (weighted_ratings - weighted_ratings.min()) / (weighted_ratings.max() - weighted_ratings.min())
    return pd.Series(weighted_ratings, index=recipes["recipe_id"])


#Hybrid Recommendation (Weighted)

def recommend_hybrid(user_id, top_n=10, alpha=0.6):
    cbf_scores = get_cbf_scores(user_id)
    cf_scores = get_cf_scores(user_id)

    hybrid_scores = alpha * cbf_scores + (1 - alpha) * cf_scores

    already_rated = df[(df["user_id"] == user_id) & (df["rating"] >= 4)]["recipe_id"]
    hybrid_scores = hybrid_scores.drop(already_rated, errors="ignore")

    top_scores = hybrid_scores.sort_values(ascending=False).head(top_n)
    top_scores_df = top_scores.reset_index()
    top_scores_df.columns = ['recipe_id', 'score']
    top_recipes_df = recipes[recipes["recipe_id"].isin(top_scores.index)][["recipe_id", "name"]].reset_index(drop=True)

    # Merge với DataFrame recipe info
    rec_with_scores = top_recipes_df.merge(top_scores_df, on='recipe_id')
    # Sort theo score giảm dần
    rec_with_scores = rec_with_scores.sort_values('score', ascending=False).reset_index(drop=True)
    return rec_with_scores

#Explainability

def explain_recommendation(user_profile, rec_idx, top_terms=5):
    feature_names = tfidf.get_feature_names_out()
    user_vec = np.array(user_profile).flatten() if user_profile is not None else np.zeros(tfidf_matrix.shape[1])
    user_top_idx = user_vec.argsort()[::-1][:50]
    recipe_vec = tfidf_matrix[rec_idx].toarray().flatten()
    recipe_top_idx = recipe_vec.argsort()[::-1][:50]
    common_idx = list(set(user_top_idx) & set(recipe_top_idx))
    if not common_idx:
        return []
    sorted_common = sorted(common_idx, key=lambda i: user_vec[i] + recipe_vec[i], reverse=True)
    return [feature_names[i] for i in sorted_common[:top_terms]]

#Visualization (Alternative 2: Recommendation Score)
# -------------------------------
def visualize_recommendations(rec_df_with_scores):
    if rec_df_with_scores is None or rec_df_with_scores.empty:
        print("⚠️ No recommendations to visualize.")
        return

    plt.figure(figsize=(10, 5))
    rec_df_with_scores = rec_df_with_scores.sort_values("score", ascending=False)

    plt.bar(rec_df_with_scores["name"], rec_df_with_scores["score"], color='lightgreen')
    plt.xlabel("Recipe Name")
    plt.ylabel("Hybrid Recommendation Score")
    plt.title("Confidence Score for Recommendations", fontsize=16)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

# Train/Test Split Evaluation (Hit@K)

def split_train_test(df, test_size=0.2, seed=42):
    train_rows, test_rows = [], []
    for uid, group in df.groupby("user_id"):
        if len(group) < 2:
            train_rows.append(group)
            continue
        train, test = train_test_split(group, test_size=test_size, random_state=seed)
        train_rows.append(train)
        test_rows.append(test)
    return pd.concat(train_rows), pd.concat(test_rows)

def evaluate_hit_rate_at_k_hybrid(k=5, test_size=0.2, seed=42, alpha=0.6):
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

        profile = _to_profile_from_indices(recipes[recipes["recipe_id"].isin(liked_train)].index.values)
        cbf_scores = get_cbf_scores(uid)
        cf_scores = get_cf_scores(uid)
        hybrid_scores = alpha * cbf_scores + (1 - alpha) * cf_scores
        hybrid_scores = hybrid_scores.drop(liked_train, errors="ignore")
        rec_list = hybrid_scores.sort_values(ascending=False).head(k).index.tolist()

        if any(r in rec_list for r in liked_test):
            hits += 1
        evaluated += 1

    hit_rate = hits / evaluated if evaluated > 0 else 0
    print(f"[Hybrid Hit@{k}] {hit_rate:.3f} (users evaluated: {evaluated})")
    return hit_rate

#  CLI Interactive

def main():
    parser = argparse.ArgumentParser(description="Hybrid CBF + CF Recommendation System")
    parser.add_argument("--mode", type=str, default="run", help="run | eval")
    parser.add_argument("--top", type=int, default=10, help="Top-K recommendations")
    parser.add_argument("--alpha", type=float, default=0.6, help="Weight for CBF (0-1)")
    args = parser.parse_args()

    if args.mode == "eval":
        evaluate_hit_rate_at_k_hybrid(k=args.top, alpha=args.alpha)
        return

    print("✨ Hybrid Recipe Recommendation System (CBF + CF) ✨")
    user_id = input("Enter user_id: ").strip()
    rec_df = recommend_hybrid(user_id, top_n=args.top, alpha=args.alpha)  # rec_df bây giờ đã có 'score'
    print("\n🍽️ Hybrid Recommendations:")
    print(rec_df[['recipe_id','name', 'score']])  # In cả score

    profile = get_user_profile(user_id)
    if profile is not None:
        for _, row in rec_df.iterrows():  # 'row' bây giờ có 'recipe_id', 'name', 'score'
            rec_idx = recipes[recipes["recipe_id"] == row["recipe_id"]].index[0]
            reasons = explain_recommendation(profile, rec_idx)
            reason_text = ", ".join(reasons) if reasons else "general similarity"
            print(f"{row['name']}  ← similar terms: {reason_text}")


    visualize_recommendations(rec_df)

if __name__ == "__main__":
    main()
