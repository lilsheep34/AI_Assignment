import pandas as pd
from tkinter import *
from tkinter import ttk, messagebox
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from surprise import Dataset, Reader, SVD

# ---------- Load and Prepare Data ----------

cf_df = pd.read_csv("user_steam.csv", header=None, names=["user_id", "game", "behavior", "value", "timestamp"])
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
    content_recs = content_recommendations(liked_game, top_n=10)
    collab_recs = collaborative_recommendations(user_id, top_n=10)

    hybrid_scores = {}
    for game in content_recs:
        hybrid_scores[game] = hybrid_scores.get(game, 0) + 1
    for game in collab_recs:
        hybrid_scores[game] = hybrid_scores.get(game, 0) + 1.5

    sorted_games = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
    return [g[0] for g in sorted_games[:top_n]]

# ---------- GUI Setup with Tkinter ----------

def get_recommendations():
    user_id = user_entry.get()
    game_title = game_entry.get()

    if not user_id or not game_title:
        messagebox.showwarning("Input Error", "Please enter both User ID and Game Title.")
        return

    try:
        recs = hybrid_recommendation(user_id, game_title)
        output.delete("1.0", END)
        output.insert(END, "\n".join(recs))
    except Exception as e:
        messagebox.showerror("Error", str(e))

root = Tk()
root.title("Steam Game Recommender")
root.geometry("500x400")

Label(root, text="Enter User ID:").pack(pady=5)
user_entry = Entry(root, width=50)
user_entry.pack()

Label(root, text="Enter a Game You Like:").pack(pady=5)
game_entry = Entry(root, width=50)
game_entry.pack()

Button(root, text="Get Recommendations", command=get_recommendations).pack(pady=10)

Label(root, text="Recommendations:").pack()
output = Text(root, height=10, width=60)
output.pack()

root.mainloop()
