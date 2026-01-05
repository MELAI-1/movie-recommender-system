import pandas as pd
import pickle
import numpy as np
import os

# Create data folder if it doesn't exist
if not os.path.exists('data'):
    os.makedirs('data')

# Create 10 fake movies for testing the UI
data = {
    'movieId': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'title': [
        'Toy Story (1995)', 'Jumanji (1995)', 'Heat (1995)', 
        'Sabrina (1995)', 'GoldenEye (1995)', 'The American President (1995)',
        'Casino (1995)', 'Sense and Sensibility (1995)', 'Ace Ventura (1995)', 'Get Shorty (1995)'
    ],
    'genres': ['Animation|Children', 'Adventure|Children', 'Action|Crime', 'Comedy|Romance', 'Action|Thriller', 'Comedy|Drama', 'Crime|Drama', 'Drama|Romance', 'Comedy', 'Comedy|Crime']
}

df = pd.DataFrame(data)

# Create a fake similarity map (each movie is "similar" to the next one)
# Structure: { index: [list of similar indices] }
sim_map = {i: np.array([(i+1)%10, (i+2)%10, (i+3)%10]) for i in range(10)}

# Save the dummy files
pickle.dump(df, open('data/movies_list.pkl', 'wb'))
pickle.dump(sim_map, open('data/similarity_map.pkl', 'wb'))

print("✅ Dummy data created! You can now test the UI.")