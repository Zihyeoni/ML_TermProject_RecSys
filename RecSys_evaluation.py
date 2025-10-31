# ========== Evaluate Hybrid model ==========

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
print("Ranking-based Evaluation Results: \n", eval_df.round(4))

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