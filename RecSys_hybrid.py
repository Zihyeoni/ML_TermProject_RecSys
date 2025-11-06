import pandas as pd
import matplotlib.pyplot as plt


# Load CBF & CF data
cbf_df = pd.read_csv("ML_TermProject_RecSys/cbf_scores.csv")
cf_df = pd.read_csv("ML_TermProject_RecSys/cf_scores.csv")
train_df = pd.read_csv("ML_TermProject_RecSys/train_data.csv")[["user_id", "recipe_id", "rating"]]
test_df = pd.read_csv("ML_TermProject_RecSys/test_data.csv")[["user_id", "recipe_id", "rating", "name"]]


# Change type to string (For consistent merging)
for df_ in [cbf_df, cf_df, train_df, test_df]:
    df_["user_id"] = df_["user_id"].astype(str)
    df_["recipe_id"] = df_["recipe_id"].astype(str)


# Merge
merged_df = cbf_df.merge(cf_df, on=["user_id", "recipe_id"], how="outer")
merged_df["cbf_normalized_score"] = merged_df["cbf_normalized_score"].fillna(0)
merged_df["cf_normalized_score"] = merged_df["cf_normalized_score"].fillna(0)


# HitRate@K (Evaluation)
def hit_rate_at_k(m, t, k):
    hits, evaluated = 0, 0
    for uid in t["user_id"].unique():
        liked = t[(t["user_id"] == uid) & (t["rating"] >= 4)]["recipe_id"].tolist()
        if not liked:
            continue
        pred_items = m[m["user_id"] == uid].nlargest(k, "hybrid_score")["recipe_id"].tolist()
        evaluated += 1
        if any(item in pred_items for item in liked):
            hits += 1
    return hits / evaluated if evaluated > 0 else 0


# Alpha Tuning
alphas = [0.2, 0.4, 0.6, 0.8]
results = []


print("Running alpha tuning...")
for a in alphas:
    temp_df = merged_df.copy()
    temp_df["hybrid_score"] = a * temp_df["cbf_normalized_score"] + (1 - a) * temp_df["cf_normalized_score"]


    # Sort and pick Top-N
    top_n = 10
    temp_df = temp_df.sort_values(["user_id", "hybrid_score"], ascending=[True, False])
    top_df = temp_df.groupby("user_id").head(top_n).reset_index(drop=True)


    # Evaluate Hit@5 and Hit@10
    hit5 = hit_rate_at_k(top_df, test_df, k=5)
    hit10 = hit_rate_at_k(top_df, test_df, k=10)
    results.append((a, hit5, hit10))
    print(f"alpha={a:.1f} → Hit@5={hit5:.4f}, Hit@10={hit10:.4f}")


# Save results
alpha_results = pd.DataFrame(results, columns=["alpha", "Hit@5", "Hit@10"])
alpha_results.to_csv("alpha_tuning_results.csv", index=False)


print("\nAlpha tuning completed. Saved: alpha_tuning_results.csv")


# Find Best Alpha
best_row = alpha_results.loc[alpha_results["Hit@10"].idxmax()]
best_alpha = best_row["alpha"]
print(f"\nBest alpha: {best_alpha:.2f} (Hit@10 = {best_row['Hit@10']:.4f})")


# Save final hybrid results
merged_df["hybrid_score"] = best_alpha * merged_df["cbf_normalized_score"] + (1 - best_alpha) * merged_df["cf_normalized_score"]
merged_df = merged_df.sort_values(["user_id", "hybrid_score"], ascending=[True, False])


merged_df.to_csv("hybrid_scores.csv", index=False)
print(f"Saved hybrid_scores.csv (alpha = {best_alpha})")


# Visualization
plt.figure(figsize=(8, 5))
plt.plot(alpha_results["alpha"], alpha_results["Hit@5"], marker="o", label="Hit@5")
plt.plot(alpha_results["alpha"], alpha_results["Hit@10"], marker="o", label="Hit@10")
plt.title("Hybrid Model Performance by Alpha")
plt.xlabel("Alpha (CBF weight)")
plt.ylabel("Hit Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show(block=False)
plt.pause(3)
plt.close()

#  ======= Recommendation : User Input and Visualization =======
top_n = 10
top_df = merged_df.groupby("user_id").head(top_n).reset_index(drop=True)
top_df = top_df.merge(test_df[["recipe_id", "name"]].drop_duplicates(), on="recipe_id", how="left")


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
        plt.p