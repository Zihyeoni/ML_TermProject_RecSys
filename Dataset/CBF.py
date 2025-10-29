import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# =========================
# 1) Load data
# =========================
recipes = pd.read_csv('RAW_recipes.csv')
interactions = pd.read_csv('RAW_interactions.csv')
interactions['user_id'] = interactions['user_id'].astype(str)

# Keep only necessary columns
recipes = recipes[['id', 'name', 'tags', 'ingredients', 'nutrition']].fillna('')

# Build content string
recipes['content'] = (
    recipes['name'] + ' ' +
    recipes['tags'] + ' ' +
    recipes['ingredients'] + ' ' +
    recipes['nutrition']
)

# TF-IDF vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(recipes['content'])

# =========================
# 2) User profile functions
# =========================
def get_user_profile(user_id: str):
    """Create a user preference profile from items the user liked (rating >= 4)."""
    user_ratings = interactions[interactions['user_id'] == user_id]
    liked = user_ratings[user_ratings['rating'] >= 4]
    if liked.empty:
        return None
    liked_idx = recipes[recipes['id'].isin(liked['recipe_id'])].index
    # Ensure ndarray (avoid np.matrix issues)
    user_profile = np.asarray(tfidf_matrix[liked_idx].mean(axis=0))
    return user_profile

def recommend_from_vector(user_profile, top_n=5):
    """Recommend items by cosine similarity against the user/item vector."""
    sim_scores = cosine_similarity(user_profile, tfidf_matrix).flatten()
    top_idx = sim_scores.argsort()[::-1][:top_n]
    return recipes.iloc[top_idx][['id', 'name']]

def recommend_for_user(user_id: str, top_n=5):
    """Personalized recommendation for an existing user_id."""
    user_profile = get_user_profile(user_id)
    if user_profile is None:
        return None
    return recommend_from_vector(user_profile, top_n)

def recommend_popular(top_n=5):
    """Fallback for cold-start users: recommend globally popular recipes."""
    popular_recipes = (
        interactions.groupby('recipe_id')['rating']
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )
    return recipes[recipes['id'].isin(popular_recipes)][['id', 'name']]

def recommend_by_selection(favorite_ids, top_n=5):
    """Cold-start onboarding: user selects a few favorite recipe IDs."""
    liked_idx = recipes[recipes['id'].isin(favorite_ids)].index
    if len(liked_idx) == 0:
        return pd.DataFrame(columns=['id', 'name'])
    user_profile = np.asarray(tfidf_matrix[liked_idx].mean(axis=0))
    return recommend_from_vector(user_profile, top_n)

# =========================
# 3) Visualization
# =========================
def visualize_recommendations(recommended_recipes: pd.DataFrame):
    """Show a word cloud of keywords and a TF-IDF bar chart for the top recommendation."""
    if recommended_recipes is None or recommended_recipes.empty:
        print("⚠️ No recommendations to visualize.")
        return

    # WordCloud text from recommended items
    text = " ".join(recipes[recipes['id'].isin(recommended_recipes['id'])]['content'])

    # WordCloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Key Terms in Recommended Recipes", fontsize=18)
    plt.show()

    # TF-IDF top terms for the first recommended recipe
    first_id = recommended_recipes.iloc[0]['id']
    recipe_idx = recipes[recipes['id'] == first_id].index[0]
    feature_names = tfidf.get_feature_names_out()
    tfidf_vector = tfidf_matrix[recipe_idx].toarray().flatten()
    top_indices = tfidf_vector.argsort()[::-1][:10]

    plt.figure(figsize=(8, 5))
    plt.barh(range(10), tfidf_vector[top_indices][::-1])
    plt.yticks(range(10), [feature_names[i] for i in top_indices][::-1])
    plt.xlabel("TF-IDF Weight")
    plt.title(f"Top Terms of Recipe {first_id}")
    plt.tight_layout()
    plt.show()

# =========================
# 4) Main flow
# =========================
print("✨ Recipe Recommendation System ✨")
mode = input("Type 'user' for existing user, or 'new' for a new user: ").strip().lower()

if mode == 'user':
    user_id = input("Please enter user_id: ").strip()
    result = recommend_for_user(user_id)
    if result is None:
        print("⚠️ This user has no preference data. Showing popular recipes instead.\n")
        pop_df = recommend_popular()
        print(pop_df)
        visualize_recommendations(pop_df)
    else:
        print("\n🍽️ Personalized Recommendations:")
        print(result)
        visualize_recommendations(result)

elif mode == 'new':
    print("\n🔰 New user mode!")
    print("Enter 2–3 favorite recipe IDs separated by commas (e.g., 12345,67890):")
    favorite_ids = input("Input: ").split(',')
    favorite_ids = [int(x.strip()) for x in favorite_ids if x.strip().isdigit()]

    if len(favorite_ids) == 0:
        print("⚠️ No input detected. Showing popular recipes instead.")
        pop_df = recommend_popular()
        print(pop_df)
        visualize_recommendations(pop_df)
    else:
        print("\n🍳 Recommendations similar to your selections:")
        sel_df = recommend_by_selection(favorite_ids)
        print(sel_df)
        visualize_recommendations(sel_df)

else:
    print("⚠️ Invalid input. Showing popular recipes.\n")
    pop_df = recommend_popular()
    print(pop_df)
    visualize_recommendations(pop_df)
