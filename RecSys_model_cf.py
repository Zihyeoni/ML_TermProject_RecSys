# ============= 3. CF (SVD) =============

import pandas as pd
from surprise import SVD, Dataset, Reader
from sklearn.preprocessing import MinMaxScaler
from surprise.model_selection import GridSearchCV

# Load train/test data
train_df = pd.read_csv("train_data.csv")
test_df = pd.read_csv("test_data.csv")

# Convert to Surprise format
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(train_df[["user_id", "recipe_id", "rating"]], reader)
trainset = data.build_full_trainset()

# Select best parameters using Grid Search
param_grid = {
    'n_factors': [50, 100, 150],
    'reg_all': [0.01, 0.02, 0.05],
    'lr_all': [0.003, 0.005, 0.007]
}

gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
gs.fit(data)

print(gs.best_params['rmse'])

# Train final SVD model with best parameters
model = SVD(n_factors=50, reg_all=0.05, lr_all=0.003, random_state=42)
model.fit(trainset)

# Predict ratings for unseen items
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

# Normalize CF predicted scores (0~1 scale)
scaler = MinMaxScaler()
cf_df["cf_normalized_score"] = scaler.fit_transform(cf_df[["cf_pred"]])

# Save CF prediction results
cf_df.to_csv("cf_scores.csv", index=False)
print("\nSaved: cf_scores.csv")