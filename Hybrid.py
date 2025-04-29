import pandas as pd
from tkinter import *
from tkinter import ttk, messagebox
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from surprise import Dataset, Reader, SVD

# ---------- Load and Prepare Data ----------

cf_df = pd.read_csv("user_steam.csv", header=None)
cf_df.columns = ["user_id", "game", "behavior", "value", "timestamp"]

print(cf_df.head())

cf_df = cf_df[cf_df["behavior"] == "play"]
cf_df = cf_df.groupby(["user_id", "game"])["value"].sum().reset_index()
cf_df.rename(columns={"value": "playtime_hours"}, inplace=True)

content_df = pd.read_csv("steam.csv")
content_df = content_df[["appid", "name", "genres", "developer", "publisher", "categories"]].dropna()
content_df.drop_duplicates(subset="name", inplace=True)

# Normalize for joining
content_df["name_lower"] = content_df["name"].str.lower()
cf_df["game_lower"] = cf_df["game"].str.lower()
merged_df = pd.merge(cf_df, content_df, left_on="game_lower", right_on="name_lower")

# Content-based filtering setup
content_df["combined"] = content_df["genres"] + " " + content_df["developer"] + " " + content_df["categories"]
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(content_df["combined"])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(content_df.index, index=content_df["name"]).drop_duplicates()

# Collaborative filtering setup
reader = Reader(rating_scale=(0, cf_df["playtime_hours"].max()))
data = Dataset.load_from_df(merged_df[["user_id", "name", "playtime_hours"]], reader)
trainset = data.build_full_trainset()
algo = SVD()
algo.fit(trainset)

# ---------- Recommender Functions ----------

def content_recommendations(title, top_n=10):
    idx = indices.get(title)
    if idx is None:
        return []
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    game_indices = [i[0] for i in sim_scores]
    return content_df.iloc[game_indices]["name"].tolist()

def collaborative_recommendations(user_id, top_n=10):
    unique_games = merged_df["name"].unique()
    predictions = [algo.predict(user_id, game) for game in unique_games]
    predictions.sort(key=lambda x: x.est, reverse=True)
    return [pred.iid for pred in predictions[:top_n]]

def hybrid_recommendation(user_id, liked_game, top_n=5):
    liked_game_lower = liked_game.lower()

    content_recs = content_recommendations(liked_game, top_n=10)
    collab_recs = collaborative_recommendations(user_id, top_n=10)

    # Remove the liked game from both recommendation lists
    content_recs = [g for g in content_recs if g.lower() != liked_game_lower]
    collab_recs = [g for g in collab_recs if g.lower() != liked_game_lower]

    hybrid_scores = {}
    for game in content_recs:
        hybrid_scores[game] = hybrid_scores.get(game, 0) + 1
    for game in collab_recs:
        hybrid_scores[game] = hybrid_scores.get(game, 0) + 1.5

    sorted_games = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
    return [g[0] for g in sorted_games[:top_n]]

# ---------- GUI Setup with Tkinter ----------
def get_recommendations():
    try:
        user_id = int(user_entry.get().strip())
    except ValueError:
        messagebox.showerror("Input Error", "User ID must be a number.")
        return

    game_title = game_entry.get().strip()
    top_n = rec_slider.get()

    if not game_title:
        messagebox.showwarning("Input Error", "Please enter a Game Title.")
        return

    # Validate user ID
    if user_id not in cf_df["user_id"].unique():
        messagebox.showerror("Invalid User ID", f"User ID '{user_id}' not found in the dataset.")
        return

    # Validate game title
    if game_title.lower() not in content_df["name"].str.lower().values:
        messagebox.showerror("Invalid Game Title", f"Game '{game_title}' not found in the dataset.")
        return

    try:
        recs = hybrid_recommendation(user_id, game_title, top_n=top_n)
        liked_game_row = content_df[content_df["name"].str.lower() == game_title.lower()].iloc[0]
        liked_features = set(liked_game_row["combined"].lower().split())

        output.delete("1.0", END)

        # Show how many users played the liked game
        played_liked = merged_df[merged_df["name"].str.lower() == game_title.lower()]["user_id"].nunique()
        output.insert(END, f"Liked Game: {liked_game_row['name']} (Played by {played_liked} users)\n\n")

        for i, rec_game in enumerate(recs, 1):
            game_row = content_df[content_df["name"] == rec_game].iloc[0]
            rec_features = set(game_row["combined"].lower().split())
            shared = liked_features.intersection(rec_features)

            user_count = merged_df[merged_df["name"] == rec_game]["user_id"].nunique()
            output.insert(END, f"{i}. {rec_game}\n")
            output.insert(END, f"   ➤ Shared Features: {', '.join(shared) if shared else 'None'}\n")
            output.insert(END, f"   ➤ Played by {user_count} users\n\n")

        if not recs:
            output.insert(END, "No recommendations available.")

    except Exception as e:
        messagebox.showerror("Error", str(e))



root = Tk()
root.title("Steam Game Recommender")
root.geometry("700x700")

# Title
Label(root, text="🎮 Hybrid Recommender System", font=("Roboto", 18)).pack(pady=10)

Label(root, text="Enter User ID:").pack(pady=5)
user_entry = Entry(root, width=50)
user_entry.pack()

Label(root, text="Enter a Game You Like:").pack(pady=5)
game_entry = Entry(root, width=50)
game_entry.pack()

Label(root, text="Number of Recommendations:").pack(pady=5)
rec_slider = Scale(root, from_=1, to=10, orient=HORIZONTAL)
rec_slider.set(5)  
rec_slider.pack()

Button(root, text="Get Recommendations", command=get_recommendations).pack(pady=10)

Label(root, text="Recommendations:").pack()
output = Text(root, height=20, width=80)
output.pack()

root.mainloop()
