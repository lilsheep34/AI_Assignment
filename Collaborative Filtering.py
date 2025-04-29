import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox

# ------------------------------- Load and prepare data -------------------------------------
#create column name
cols_name = ['User ID', 'Name of steam game', 'Behavior name', 'Hours of playing', 'Extra column']

# open file and load it into DataFrame (table of data)
data = pd.read_csv('user_steam.csv', names=cols_name, header=None)
data = data.drop(columns=['Extra column'])

#create a game name list
unique_game_names = sorted(data['Name of steam game'].dropna().unique())

# ------------------------------- Filter the data basedon the behavior -------------------------------------
#get the data that only ply by user
play_data = data[data['Behavior name']=='play']

# ------------------------------- Calculation of behaviour 'play' -------------------------------------
#calculate the mean hours of playing of the game 
mean_play = play_data.groupby('Name of steam game')['Hours of playing'].mean() #.sort_values(ascending=False)
count_play = play_data.groupby('Name of steam game')['Hours of playing'].count() #.sort_values(ascending=False)

#create a new dataframe(table) for mean and count based on the game name
playing = pd.DataFrame(mean_play)
playing['Number of playing'] = pd.DataFrame(count_play)

# ------------------------------- Collaborative recommeder function -------------------------------------
def recommend_collborative(game_name, num_recommendations):
    # if game isn't in our list, game not found and return an empty list
    if game_name not in unique_game_names:
        return [], f"Game '{game_name}' not found."

    #Create pivot table based on the behavior
    moviemat = play_data.pivot_table(index='User ID',columns='Name of steam game',values='Hours of playing')

    #selecting a specific game's play data
    user_playing = moviemat[game_name]
    
    #calculate the correlation
    similar_to_user_playing = moviemat.corrwith(user_playing) #how much other games are played by the same people who play 'game_name'

    #wrapping the result in a dataframe(table)
    corr_user_playing = pd.DataFrame(similar_to_user_playing, columns =['Correlation'])
    corr_user_playing.dropna(inplace = True) #remove the empty or bad data

    #joint the correlation table and the playing(contain mean and count) table
    corr_user_playing = corr_user_playing.sort_values('Correlation', ascending = False)
    corr_user_playing = corr_user_playing.join(playing[['Number of playing','Hours of playing']]) 
    
    #Filter the game that played by more than 100 people nd very similar to 'game_name'
    filtered_corr = corr_user_playing[corr_user_playing['Number of playing']>100].sort_values('Correlation', ascending = False)

    #skip the first one (which is usually the game itself), and get top N recommendations with name and number of playing
    recommendations_df = filtered_corr.iloc[1:num_recommendations+1][['Number of playing','Hours of playing']]
    recommendations_df = recommendations_df.reset_index().rename(columns={'index': 'Name of steam game'})

    return recommendations_df,None

# --------------------------- GUI setup -------------------------------------
app = tk.Tk()
app.title("Steam Game Recommender")
app.geometry("700x700")

# Title
tk.Label(app, text="🎮 Collaborative Filtering Recommender System", font=("Roboto", 18)).pack(pady=10)

# Frame to hold search input and suggestions
search_frame = tk.Frame(app)
search_frame.pack(pady=10)

tk.Label(search_frame, text="Enter a game you like:").pack(anchor='w')
search_var = tk.StringVar()
search_entry = tk.Entry(search_frame, textvariable=search_var, width=50)
search_entry.pack()

tk.Label(app, text="Recommended game that you’ll enjoy:").pack()
suggest_listbox = tk.Listbox(search_frame, height=5, width=50)
suggest_listbox.pack(pady=(5, 10))
suggest_listbox.pack_forget()

# Function to update suggestions
def update_suggestions(event):
    typed = search_var.get().lower()
    matches = [name for name in unique_game_names if typed in name.lower()]
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
num_slider = tk.Scale(app, from_=1, to=10, orient=tk.HORIZONTAL)
num_slider.set(1)
num_slider.pack()

# Results display
tk.Label(app, text="\nNumber of recommendations:").pack()
result_box = tk.Text(app, height=15, width=80)
result_box.pack(pady=10)

# Recommend button callback
def show_recommendations():
    result_box.delete("1.0", tk.END)
    game = search_var.get()
    num = num_slider.get()
    recommendations, error = recommend_collborative(game, num)

    if error:
        messagebox.showerror("Error", error)
        return

    for i,row in recommendations.iterrows():
        result_box.insert(tk.END, f"{i+1}. 🎯 {row['Name of steam game']}\nNumber of people are playing : {int(row['Number of playing'])}\nAverage hour played : {row['Hours of playing']:.2f}hours\n\n")

# Recommend button
tk.Button(app, text="🔍 Recommend", command=show_recommendations).pack(pady=5)

# Start app
app.mainloop()