# Content-Based Game Recommender using Steam Dataset

import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np

# Load dataset
df = pd.read_csv('C:/Users/xingj/Desktop/TARUMT/RSTY2S3/AI/Assignment/steam.csv')

# Display basic info
print("Total games:", df.shape[0])
print(df[['name', 'genres', 'steamspy_tags', 'developer']].head())

# Step 1: Preprocessing
def clean_text(x):
    if pd.isna(x):
        return ''
    return str(x).lower().replace(',', ' ').replace(';', ' ')

# Create a new column combining genres, tags, and developer
df['combined_features'] = (
    df['genres'].fillna('') + ' ' +
    df['steamspy_tags'].fillna('') + ' ' +
    df['developer'].fillna('')
).apply(clean_text)

# Step 2: Vectorize the text
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['combined_features'])

# Step 3: Compute cosine similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Step 4: Map game titles to index
df = df.reset_index()
indices = pd.Series(df.index, index=df['name']).drop_duplicates()

# Step 5: Recommender function
def recommend(game_title, num_recommendations=5):
    if game_title not in indices:
        return f"Game '{game_title}' not found in dataset."

    idx = indices[game_title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:num_recommendations+1]  # skip the game itself
    game_indices = [i[0] for i in sim_scores]

    recommended_games = df[['name', 'genres', 'developer']].iloc[game_indices]
    return recommended_games.reset_index(drop=True)

# Step 6: Example usage
game_to_search = "Counter-Strike: Global Offensive"
print(f"\nTop Recommendations for: {game_to_search}")
print(recommend(game_to_search, 5))