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

# Same user_id
common_users = set(pred_df["user_id"]).intersection(set(test_df["user_id"]))

pred_df = pred_df[pred_df["user_id"].isin(common_users)]
test_df = test_df[test_df["user_id"].isin(common_users)]
print("\nCommon users retained: ", len(common_users))

# === Vectorized Evaluation (Precision, Recall, nDCG, Hit) ===
def get_topk(pred_df, k):
    """Return top-k prediction results per user"""
    return (
        pred_df.sort_values(["user_id", "rating_pred"], ascending=[True, False])
        .groupby("user_id")
        .head(k)[["user_id", "recipe_id"]]
    )

def ranking_metrics(pred_df, test_df, k=10):
    liked_df = test_df[test_df["rating_true"] >= 4][["user_id", "recipe_id"]]
    liked_dict = liked_df.groupby("user_id")["recipe_id"].apply(set).to_dict()
    topk_df = get_topk(pred_df, k)

    hits, precisions, recalls, ndcgs = [], [], [], []
    for uid, group in topk_df.groupby("user_id"):
        liked_items = liked_dict.get(uid, set())
        if not liked_items:
            continue
        rec_items = list(group["recipe_id"])
        inter = set(rec_items) & liked_items
        n_hit = len(inter)
        hits.append(1 if n_hit > 0 else 0)
        precisions.append(n_hit / k)
        recalls.append(n_hit / len(liked_items))

        # nDCG
        gains = [1 / np.log2(i + 2) if r in liked_items else 0 for i, r in enumerate(rec_items)]
        dcg = np.sum(gains)
        idcg = np.sum([1 / np.log2(i + 2) for i in range(min(k, len(liked_items)))])
        ndcgs.append(dcg / idcg if idcg > 0 else 0)

    return {
        "HitRate": np.mean(hits) if hits else 0,
        "Precision": np.mean(precisions) if precisions else 0,
        "Recall": np.mean(recalls) if recalls else 0,
        "nDCG": np.mean(ndcgs) if ndcgs else 0,
    }

# Evaluate across different K values
metrics = []
for k in [5, 10, 20, 50]:
    scores = ranking_metrics(pred_df, test_df, k)
    metrics.append((k, scores["Precision"], scores["Recall"], scores["nDCG"], scores["HitRate"]))

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