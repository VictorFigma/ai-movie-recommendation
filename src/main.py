import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder
import json

data = pd.read_csv('data/movies_dataset.csv')

# Data Preprocessing
encoder = OneHotEncoder()
encoded_users = encoder.fit_transform(data[['user']])
encoded_themes = encoder.fit_transform(data[['theme']])
encoded_users_df = pd.DataFrame(encoded_users.toarray(), columns=[f"user_{i}" for i in range(encoded_users.shape[1])])
encoded_themes_df = pd.DataFrame(encoded_themes.toarray(), columns=[f"theme_{i}" for i in range(encoded_themes.shape[1])])
data_encoded = pd.concat([data.drop(columns=['user', 'theme']), encoded_users_df, encoded_themes_df], axis=1)

users = data['user']

sorted_users = sorted(users.unique(), key=lambda x: int(x.replace('user', '')))

# Splitting Data
train_data, test_data = train_test_split(data_encoded, test_size=0.2, random_state=42)

# Model Training
model = NearestNeighbors(n_neighbors=5)
model.fit(train_data.drop(columns=['name']))

# Making Predictions
recommendations = {}
for user in sorted_users:
    user_data = test_data[users == user].drop(columns=['name'])
    if len(user_data) > 0:
        neighbors = model.kneighbors(user_data)
        top_neighbors = neighbors[1][0][:3]
        recommended_movies = train_data.iloc[top_neighbors]['name'].tolist()
        recommendations[user] = recommended_movies

# Save the dictionary to a JSON file
recommendations_dict = {}

for user, movies in recommendations.items():
    recommendations_dict[user] = movies

final_dict = {"target": recommendations_dict}

with open('predictions/predictions.json', 'w') as f:
    json.dump(final_dict, f, indent=4)