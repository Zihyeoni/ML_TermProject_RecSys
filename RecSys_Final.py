# ===== Save train/test split (For sharing) =====

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load preprocessed data
df = pd.read_csv("preprocessed_data.csv")
df["user_id"] = df["user_id"].astype(str)
df["recipe_id"] = df["recipe_id"].astype(str)

# Remove cold-start user
user_counts = df["user_id"].value_counts()
df = df[df["user_id"].isin(user_counts[user_counts > 1].index)]

# train/test split (stratify: "user_id")
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["user_id"])

# Save train/test data (Using for CF, CBF, Hybrid)
train_df.to_csv("train_data.csv", index=False)
test_df.to_csv("test_data.csv", index=False)

print("Saved: train_data.csv, test_data.csv")

# ========== CF (SVD) ==========

from surprise import SVD, Dataset, Reader
from sklearn.preprocessing import MinMaxScaler

# Load train/test data
train_df = pd.read_csv("train_data.csv")
test_df = pd.read_csv("test_data.csv")

#  Convert Surprise format
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(train_df[["user_id", "recipe_id", "rating"]], reader)
trainset = data.build_full_trainset()

# Train SVD
model = SVD(n_factors=100, reg_all=0.02)
model.fit(trainset)

# Unseen items prediction
user_train_items = train_df.groupby("user_id")["recipe_id"].apply(set)
all_items = set(train_df["recipe_id"]).union(set(test_df["recipe_id"]))

preds = []
for uid in train_df["user_id"].unique():
    seen = user_train_items.get(uid, set())
    unseen = all_items - seen
    for rid in unseen:
        preds.append(model.predict(uid, rid))

cf_df = pd.DataFrame([{
    "user_id": p.uid,
    "recipe_id": p.iid,
    "cf_pred": p.est
} for p in preds])

# Normalizing CF score
scaler = MinMaxScaler()
cf_df["cf_normalized_score"] = scaler.fit_transform(cf_df[["cf_pred"]])

# Save CF scores data
cf_df.to_csv("cf_scores.csv", index=False)
print("Saved: cf_scores.csv")

# ========== CBF (TF-IDF + Cosine Similarity) ==========

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

# Save CBF scores data
cbf_df.to_csv("cbf_scores.csv", index=False)
print("Saved: cbf_scores.csv")

# ========== CF + CBF (Hybrid) ==========

# Load CBF & CF data
cbf_df = pd.read_csv("cbf_scores.csv")
cf_df = pd.read_csv("cf_scores.csv")
train_df = pd.read_csv("train_data.csv")[["user_id", "recipe_id", "rating"]]
test_df = pd.read_csv("test_data.csv")[["user_id", "recipe_id", "rating", "name"]]

# Change type to string (For consistent merging)
for df_ in [cbf_df, cf_df, train_df, test_df]:
    df_["user_id"] = df_["user_id"].astype(str)
    df_["recipe_id"] = df_["recipe_id"].astype(str)

# Merge
merged_df = cbf_df.merge(cf_df, on=["user_id", "recipe_id"], how="outer")
merged_df["cbf_score"] = merged_df["cbf_score"].fillna(0)
merged_df["cf_normalized_score"] = merged_df["cf_normalized_score"].fillna(0)

# Calculate hybrid score (alpha = 0.6)
alpha = 0.6
merged_df["hybrid_score"] = alpha * merged_df["cbf_score"] + (1 - alpha) * merged_df["cf_normalized_score"]

# Top-N Recommendations for Users
top_n = 10
merged_df = merged_df.sort_values(["user_id", "hybrid_score"], ascending=[True, False])
top_df = merged_df.groupby("user_id").head(top_n).reset_index(drop=True)

# recipe name join
top_df = top_df.merge(test_df[["recipe_id", "name"]].drop_duplicates(), on="recipe_id", how="left")

# Save Top-N Hybrid scores data
top_df.to_csv("hybrid_scores.csv", index=False)
print(f"Saved: hybrid_scores.csv (Top-{top_n}, alpha = {alpha})\n")

#  User Input and Visualization
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

        # Visualization
        plt.figure(figsize=(10, 5))
        plt.bar(user_rec["recipe_id"].astype(str), user_rec["hybrid_score"], color="skyblue")
        plt.xlabel("Recipe ID")
        plt.ylabel("Hybrid Score")
        plt.title(f"Top-{top_n} Hybrid Recommendations for User {user_id}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(3)
        plt.close()
        print()

# ========== Evaluate Hybrid model ==========

# Load Hybrid data
pred_df = pd.read_csv("hybrid_scores.csv")
test_df = pd.read_csv("test_data.csv")[["user_id", "recipe_id", "rating", "name"]]

# Change type to string (For consistent merging)
for d in [pred_df, test_df]:
    d["user_id"] = d["user_id"].astype(str)
    d["recipe_id"] = d["recipe_id"].astype(str)

# Rename column name
pred_df = pred_df.rename(columns={"hybrid_score": "rating_pred"})
test_df = test_df.rename(columns={"rating": "rating_true"})

# == Ranking-based Metrics (Precision, Recall, nDCG, Hit) ==
# Precision@K
def precision_at_k(pred_df, test_df, k):
    precisions = []
    for uid in test_df["user_id"].unique():
        liked = test_df[(test_df["user_id"] == uid) & (test_df["rating_true"] >= 4)]["recipe_id"].tolist()
        if not liked:
            continue
        pred_items = pred_df[pred_df["user_id"] == uid].nlargest(k, "rating_pred")["recipe_id"].tolist()
        hits = len(set(pred_items) & set(liked))
        precisions.append(hits / k)
    return np.mean(precisions) if precisions else np.nan

# Recall@K
def recall_at_k(pred_df, test_df, k):
    recalls = []
    for uid in test_df["user_id"].unique():
        liked = test_df[(test_df["user_id"] == uid) & (test_df["rating_true"] >= 4)]["recipe_id"].tolist()
        if not liked:
            continue
        pred_items = pred_df[pred_df["user_id"] == uid].nlargest(k, "rating_pred")["recipe_id"].tolist()
        hits = len(set(pred_items) & set(liked))
        recalls.append(hits / len(liked))
    return np.mean(recalls) if recalls else np.nan

# nDCG@K
def ndcg_at_k(pred_df, test_df, k):
    ndcgs = []
    for uid in test_df["user_id"].unique():
        user_pred = pred_df[pred_df["user_id"] == uid].nlargest(k, "rating_pred")
        user_true = test_df[(test_df["user_id"] == uid) & (test_df["rating_true"] >= 4)]["recipe_id"].tolist()
        if not len(user_true):
            continue
        gains = [1 / np.log2(i + 2) if row["recipe_id"] in user_true else 0 for i, row in user_pred.iterrows()]
        dcg = np.sum(gains)
        idcg = np.sum([1 / np.log2(i + 2) for i in range(min(k, len(user_true)))])
        ndcgs.append(dcg / idcg if idcg > 0 else 0)
    return np.mean(ndcgs) if ndcgs else np.nan

# HitRate@K
def hit_rate_at_k(pred_df, test_df, k):
    hits, evaluated = 0, 0
    for uid in test_df["user_id"].unique():
        liked = test_df[(test_df["user_id"] == uid) & (test_df["rating_true"] >= 4)]["recipe_id"].tolist()
        if not liked:
            continue
        pred_items = pred_df[pred_df["user_id"] == uid].nlargest(k, "rating_pred")["recipe_id"].tolist()
        evaluated += 1
        if any(item in pred_items for item in liked):
            hits += 1
    return hits / evaluated if evaluated > 0 else 0

# Evaluate across different K values
metrics = []
for k in [5, 10]:
    prec = precision_at_k(pred_df, test_df, k)
    rec = recall_at_k(pred_df, test_df, k)
    ndcg = ndcg_at_k(pred_df, test_df, k)
    hit = hit_rate_at_k(pred_df, test_df, k)
    metrics.append((k, prec, rec, ndcg, hit))

eval_df = pd.DataFrame(metrics, columns=["K", "Precision", "Recall", "nDCG", "HitRate"])
print("\nRanking-based Evaluation Results: \n", eval_df.round(4))

# Save results
eval_df.to_csv("evaluation_ranking.csv", index=False)
print("\nSaved: evaluation_ranking.csv (Precision/Recall/nDCG/HitRate)\n")

# Visualization
plt.figure(figsize=(8,5))
for col in ["Precision", "Recall", "nDCG", "HitRate"]:
    plt.scatter(eval_df["K"], eval_df[col], label=f"{col}@K")
    plt.plot(eval_df["K"], eval_df[col], linestyle="--", alpha=0.5)

plt.xlabel("K (Top-N)")
plt.ylabel("Score")
plt.title("Hybrid Model Performance (Ranking-based)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# == Tuning alpha ==
alphas = [0.2, 0.4, 0.6, 0.8]
results = [] 

for alpha in alphas: 
  pred_df["rating_pred"] = alpha * pred_df["cbf_score"] + (1-alpha) * pred_df["cf_normalized_score"]
  hit5 = hit_rate_at_k(pred_df, test_df, k=5)
  hit10 = hit_rate_at_k(pred_df, test_df, k=10)
  results.append((alpha, hit5, hit10))

alpha_hit = pd.DataFrame(results, columns=["alpha", "Hit@5", "Hit@10"])
print("\nHit Rate by alpha: \n", alpha_hit)

# Visualization
plt.figure(figsize=(8, 5))
plt.plot(alpha_hit["alpha"], alpha_hit["Hit@5"], marker="o", label="Hit@5")
plt.plot(alpha_hit["alpha"], alpha_hit["Hit@10"], marker="o", label="Hit@10")
plt.xlabel("Hybrid Weight (α)")
plt.ylabel("Hit Rate")
plt.title("Hybrid Recommendation Performance by α")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()