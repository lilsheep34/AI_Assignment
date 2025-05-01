# Content-Based Game Recommender with Tkinter GUI (with Auto Suggest and Explanation)
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# ------------------------- Preprocessing function --------------------------------
def clean_text(x):
    if pd.isna(x): # if the input is empty or missing (like nothing id written)
        return '' # return an empty string so computer doesn't get confused
    
    # take the text make all letters small, replace any , and ; with space so it looks clean
    return str(x).lower().replace(',', ' ').replace(';', ' ')

# ------------------------------- Load and prepare data -------------------------------------
# open file and load it into DataFrame (table of data)
df = pd.read_csv('steam.csv')
#print(df.head()) # show first 5 rows of the table

# take some columns, fill in missing ones with an empty string, glue them tgt with space between, then clean text using clean_text helper
df['genres'] = df['genres'].str.replace(';', ', ', regex=False)
df['combined_features'] = df[['genres', 'steamspy_tags', 'developer']].fillna('').agg(' '.join, axis=1).apply(clean_text)

# ---------------------------------- Create List of Game Names -------------------------------
# make a list of game names - remove any that are missing, make sure each name is only once (unique) and sort them alphabetically
game_list = sorted(df['name'].dropna().unique())

# ------------------------------ Turn words into Numbers (Vectorization) -------------------------
# tool to turn words into numbers, and ignore english words like 'the' , 'is', etc
vectorizer = TfidfVectorizer(stop_words='english')

# tell the tool to learn all important words from games' features and turn them into big table of numbers called tfidf_matrix
tfidf_matrix = vectorizer.fit_transform(df['combined_features'])

# measure similarity between games - compare every game with every other game using cosine similarity to tell us how similar they are
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# ---------------------------------- Index mapping --------------------------------------------
# reset table's index so each row has a nice number froom 0, 1, 2,...
df = df.reset_index()

# make special dictionary: if you give it a game name, it tells you row number, which help us find games quickly
indices = pd.Series(df.index, index=df['name']).drop_duplicates()

# -------------------- Function to find shared features ---------------------------
# another helper function, takes two games (using their row numbers) to find what they have in common
def get_shared_features(game1_idx, game2_idx):
    # we take combined features for each game, split them into words and make them into sets (like unique lists)
    features1 = set(df.loc[game1_idx, 'combined_features'].split())
    features2 = set(df.loc[game2_idx, 'combined_features'].split())

    # find words both games share (like if both multiplayer) and join them into one string to show
    return ', '.join(features1.intersection(features2))

# ----------------------------- Recommender function with explainability --------------------------------
# function that gives game suggestion by give a game name and how many games you want
def recommend(game_title, num_recommendations=5):
    # if game isn't in our list, game not found and return an empty list
    if game_title not in indices:
        return [], f"Game '{game_title}' not found."

    # otherwise, get row number for that game so we can compare it to others
    idx = indices[game_title]

    # get similarity score of our game compared to all other games and pair each score with game's index
    sim_scores = list(enumerate(cosine_sim[idx]))

    # sort list so most simlar games come first (from highest to lowest score)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # skip the first one (which is game itself!) and keep only number of recommendation the user asked for
    sim_scores = sim_scores[1:num_recommendations+1]
    

    # make an empty list to store recommended games
    recommendations = []

    # for each similar games, we grab index and similarity score
    for i, score in sim_scores:
        # find what features the current game and this recommended game have in common
        shared = get_shared_features(idx, i)

        # we get all the details of the recommended game from the table
        row = df.loc[i]

        # add game's name, its genre, who made it and shared features in our list
        recommendations.append((row['name'], row['genres'], row['developer'], shared, score))

    # when done, we return list of recommendations and None because there's no error
    return recommendations, None
    
#get the similarity score and do a table with the game name 
def get_sim_scores_table(sim_scores):
    data_sim_scores = []
    for i,score in sim_scores:
        name = df.loc[i,'name']
        data_sim_scores.append({'Game Name':name,'Similarity Score':score})

    return pd.DataFrame(data_sim_scores)

def graph_display(parent_window, sim_scores):
    scores = get_sim_scores_table(sim_scores).head(10)
    fig = Figure(figsize=(9,5),dpi=100)
    ax = fig.add_subplot(111)
    scores.plot(kind='bar',x='Game Name',y='Similarity Score',ax=ax)
    ax.set_title('Top 10 of similarity score')
    ax.set_ylabel('Similarity score')
    ax.set_xlabel('Name of Steam Game')
    ax.tick_params(axis='x',rotation=90)

    fig.tight_layout()

    #embed plot into tkinter window
    canvas = FigureCanvasTkAgg(fig,master=parent_window)
    canvas.draw()
    canvas.get_tk_widget().pack(pady=10)
    
# --------------------------- GUI setup -------------------------------------
app = tk.Tk()
app.title("Steam Game Recommender")
app.geometry("700x900")

# Title
tk.Label(app, text="🎮 Content-Based Filtering Recommender System", font=("Roboto", 18)).pack(pady=10)

last_sim_scores = []

# Frame to hold search input and suggestions
search_frame = tk.Frame(app)
search_frame.pack(pady=10)

tk.Label(search_frame, text="Search for a game you like:").pack(anchor='w')
search_var = tk.StringVar()
search_entry = tk.Entry(search_frame, textvariable=search_var, width=50)
search_entry.pack()

suggest_listbox = tk.Listbox(search_frame, height=5, width=50)
suggest_listbox.pack(pady=(5, 10))
suggest_listbox.pack_forget()

# Function to update suggestions
def update_suggestions(event):
    typed = search_var.get().lower()
    matches = [name for name in game_list if typed in name.lower()]
    suggest_listbox.delete(0, tk.END)

    if matches:
        for match in matches[:10]:
            suggest_listbox.insert(tk.END, match)
        suggest_listbox.pack()
    else:
        suggest_listbox.pack_forget()

# Function to autofill on click
def fill_from_suggest(event):
    if suggest_listbox.curselection():
        selected = suggest_listbox.get(tk.ACTIVE)
        search_var.set(selected)
        suggest_listbox.place_forget()

search_entry.bind('<KeyRelease>', update_suggestions)
suggest_listbox.bind('<<ListboxSelect>>', fill_from_suggest)

# Number of recommendations
tk.Label(app, text="Number of recommendations:").pack()
num_slider = tk.Scale(app, from_=1, to=10, orient=tk.HORIZONTAL)
num_slider.set(1)
num_slider.pack()

# Results display
result_box = tk.Text(app, height=15, width=80)
result_box.pack(pady=10)

# Recommend button callback
def show_recommendations():
    global last_sim_scores  # make accessible to other functions
    
    result_box.delete("1.0", tk.END)
    game = search_var.get()
    num = num_slider.get()
    recommendations, error = recommend(game, num)

    if error:
        messagebox.showerror("Error", error)
        return

    idx = indices[game]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num+1]
    last_sim_scores = sim_scores  # Save for graph
    
    # display recommendations in the text box
    for idx, (name, genre, developer, shared, score) in enumerate(recommendations, start=1):
        result_box.insert(tk.END, f"{idx}. 🎯 {name}\nGenres   : {genre}\nDeveloper: {developer}\nShared features with {game}: {shared}\nSimilarity score: {score:.4f}\n\n")


# Recommend button
tk.Button(app, text="🔍 Recommend", command=show_recommendations).pack(pady=5)

# Cold-start preference search
pref_label = tk.Label(app, text="Or enter your preferred genre/tag/developer:")
pref_label.pack(pady=(10, 0))
pref_entry = tk.Entry(app, width=50)
pref_entry.pack()

# Cold-start callback
def cold_start_recommend():
    # clear old stuff
    result_box.delete("1.0", tk.END)

    # remove messy stuff like extra symbols or spaces for user input
    user_pref = clean_text(pref_entry.get())
    num = num_slider.get()  # use slider value

    # if left box empty, show pop up message
    if not user_pref:
        messagebox.showwarning("Input needed", "Please enter a genre, tag, or developer preference.")
        return

    # Vectorize user input (turn words like action/adventure into numbers called vectors that computer can understand)
    user_vec = vectorizer.transform([user_pref])

    # compare robot words to all games we know (gives us a score for each game, show how similar they are to your fav stuff)
    # higher score = better match
    sim_scores = linear_kernel(user_vec, tfidf_matrix).flatten()

    # gets the best scores (the most similar ones)
    top_indices = sim_scores.argsort()[-num:][::-1]  # use top N from slider

    for idx, i in enumerate(top_indices, start=1):
        row = df.loc[i]
        shared = ', '.join(set(user_pref.split()).intersection(row['combined_features'].split()))
        score = sim_scores[i]
        result_box.insert(tk.END, f"{idx}. 🎯 {row['name']}\nGenres   : {row['genres']}\nDeveloper: {row['developer']}\nShared features: {shared}\nSimilarity score: {score:.4f}\n\n")

# Cold-start button
tk.Button(app, text="✨ Recommend by Preference", command=cold_start_recommend).pack(pady=5)

tk.Label(app, text="\n\n----- Summary Bar Chart -----").pack()

def view_graph():
    if not last_sim_scores:
        messagebox.showwarning("No data", "Please run a recommendation first.")
        return
    graph_win = tk.Toplevel(app)
    graph_win.geometry("900x500")
    graph_win.title("Similarity score based on the")
    graph_display(graph_win, last_sim_scores)

#view graph
tk.Button(app, text="🔍 View Graph", command=view_graph).pack(pady=5)

# Start app
app.mainloop()