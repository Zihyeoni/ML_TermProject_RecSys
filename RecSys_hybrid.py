# =====================================================
# Hybrid Recommendation System (CBF + CF)
# Features:
# - Merge CBF & CF scores to compute hybrid
# - Top-N recommendations per user
# - Visualization of hybrid scores for input user
# =====================================================

import pandas as pd
import matplotlib.pyplot as plt


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
        print("💡 Example user_ids:", top_df["user_id"].drop_duplicates().sample(5).to_list())
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
