# ===== Save train/test split (For sharing) =====

import pandas as pd
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