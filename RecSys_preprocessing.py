import ast
import numpy as np
import pandas as pd

# Select only necessary columns
r_cols = ["id", "name", "ingredients", "tags"]
i_cols = ["user_id", "recipe_id", "rating"]

recipes = pd.read_csv("RAW_recipes.csv", usecols=r_cols)
interactions = pd.read_csv("RAW_interactions.csv", usecols=i_cols)

# Handle missing values
interactions = interactions.dropna().copy()
interactions["rating"] = interactions["rating"].clip(1 ,5).astype(float)

# Convert stringified lists(string) to lists
def safe_list(s):
    if pd.isna(s): return []
    try:
        v = ast.literal_eval(str(s))
        return v if isinstance(v, list) else []
    except Exception:
        return []

# list -> Text (For CBF; TF-IDF)
def join_tokens(lst):
    return " ".join([str(x).replace(" ", "_") for x in lst])

recipes["ingredients_list"] = recipes["ingredients"].apply(safe_list)
recipes["tags_list"] = recipes["tags"].apply(safe_list)
recipes["text"] = (recipes["ingredients_list"].apply(join_tokens) + " " +
                   recipes["tags_list"].apply(join_tokens)).str.strip()

items = recipes.rename(columns={"id": "recipe_id"})[["recipe_id", "name", "text"]]

# Reduce dataset size (actual interactions remain)
top_users = interactions["user_id"].value_counts().sort_values(ascending=False).head(2000).sort_index().index  # Users 2,000
top_items = interactions["recipe_id"].value_counts().sort_values(ascending=False).head(2000).sort_index().index  # Items 2,000
inter_small = interactions[interactions["user_id"].isin(top_users) & interactions["recipe_id"].isin(top_items)].copy()

# Merge items & interactions
df = inter_small.merge(items, on="recipe_id", how="inner")

# Filter users & items with enough interactions
min_user_ratings = 5
min_item_ratings = 5

user_counts = df["user_id"].value_counts()
item_counts = df["recipe_id"].value_counts()

df = df[
    df["user_id"].isin(user_counts[user_counts >= min_user_ratings].index)
    & df["recipe_id"].isin(item_counts[item_counts >= min_item_ratings].index)
].copy()

# Sort for stability
df = df.sort_values(["user_id", "recipe_id"]).reset_index(drop=True)

df.to_csv("preprocessed_data.csv", index=False)
print("Saved: preprocessed_data.csv\n")
print(df)