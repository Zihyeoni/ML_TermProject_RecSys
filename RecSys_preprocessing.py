import ast
import numpy as np
import pandas as pd

# 필요한 columns만 사용
r_cols = ["id", "name", "ingredients", "tags"]
i_cols = ["user_id", "recipe_id", "rating"]

recipes = pd.read_csv("RAW_recipes.csv", usecols=r_cols)
interactions = pd.read_csv("RAW_interactions.csv", usecols=i_cols)

# 결측값 처리
interactions = interactions.dropna().copy()
interactions["rating"] = interactions["rating"].clip(1 ,5).astype(float)

# 문자열 list로 변환 (결측/이상값인 경우, [] return)
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

# 상위 일부만 추출 (상호작용 많은 user/item)
top_users = interactions["user_id"].value_counts().nlargest(2000).index  # 사용자 2000명
top_items = interactions["recipe_id"].value_counts().nlargest(2000).index  # 아이템 2000개
inter_small = interactions[interactions["user_id"].isin(top_users) & interactions["recipe_id"].isin(top_items)].copy()

# Merge items & interactions
df = inter_small.merge(items, on="recipe_id", how="inner")
df.to_csv("preprocessed_data.csv", index=False)
print(df)