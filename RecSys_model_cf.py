# Machine Learning Term Project - Model-based Collaborative Filtering (Matrix Factorization)
# Topic: Personalized Recipe Recommendation System
# -------------------------------------------------------------
# This code uses SVD-based Matrix Factorization to learn user-recipe rating data
# and generate personalized recommendations.
# Includes RMSE/MAE evaluation, cross-validation, result saving, and visualization.
# -------------------------------------------------------------

# 1. Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split, cross_validate

# 2. Load dataset
# The preprocessed data must contain the columns: user_id, recipe_id, rating
df = pd.read_csv("preprocessed_data.csv")

print("Dataset loaded successfully.")
print(df.head())
print(f"Total records: {len(df)}\n")

# 3. Prepare data for Surprise
reader = Reader(rating_scale=(0, 5))
data = Dataset.load_from_df(df[['user_id', 'recipe_id', 'rating']], reader)

# 4. Train-test split
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# 5. Train model (SVD: Matrix Factorization)
model = SVD(random_state=42)
model.fit(trainset)
print("Model training completed.\n")

# 6. Prediction and evaluation (RMSE, MAE)
predictions = model.test(testset)
rmse = accuracy.rmse(predictions)
mae = accuracy.mae(predictions)

# 7. Cross-validation (optional)
cv_results = cross_validate(model, data, measures=['RMSE', 'MAE'], cv=5, verbose=False)
rmse_mean = cv_results['test_rmse'].mean()
mae_mean = cv_results['test_mae'].mean()

print(f"\nAverage RMSE from cross-validation: {rmse_mean:.4f}")
print(f"Average MAE from cross-validation: {mae_mean:.4f}\n")

# 8. Generate Top-N recommendations for each user
def get_top_n(predictions, n=5):
    """
    Return top-N recommended items for each user.
    :param predictions: List of prediction results from Surprise
    :param n: Number of recommendations per user (default=5)
    """
    from collections import defaultdict
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n

top_n = get_top_n(predictions, n=5)

# 9. Print sample recommendation results
print("Top-5 recommendations for sample users:")
for uid, recs in list(top_n.items())[:5]:
    print(f"User {uid} → {[iid for (iid, _) in recs]}")

# 10. Save recommendation results
output = []
for uid, recs in top_n.items():
    for iid, est in recs:
        output.append({'user_id': uid, 'recipe_id': iid, 'pred_rating': est})

pd.DataFrame(output).to_csv("cf_recommendations.csv", index=False)
print("\nRecommendation results saved to cf_recommendations.csv\n")

# 11. Visualization: RMSE and MAE comparison
plt.figure(figsize=(6, 4))
plt.bar(["RMSE", "MAE"], [rmse_mean, mae_mean], color=["#4C72B0", "#55A868"])
plt.title("Model-based CF (SVD) Performance Comparison", fontsize=13)
plt.ylabel("Error")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("cf_performance.png")
plt.show()

print("Performance visualization completed (cf_performance.png saved).")