# RECOMMENDATION-SYSTEM Create a simple recommendation system that suggests items to users based on their preferences. You can use techniques like collaborative filtering or content-based filtering to recommend  movies, books, or products to users.

import pandas as pd
from sklearn.metrics.pairwise import linear_kernel 
from sklearn.feature_extraction.text import TfidfVectorizer 

# Sample movie data
data = {
    'movie_id': [1, 2, 3, 4, 5],
    'title': ['Movie A', 'Movie B', 'Movie C', 'Movie D', 'Movie E'],
    'genres': ['Action|Adventure', 'Comedy', 'Action|Sci-Fi', 'Comedy|Romance', 'Sci-Fi|Thriller']
}

# Create a DataFrame from the data
movies_df = pd.DataFrame(data)

# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the TF-IDF matrix
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['genres'])

# Calculate the cosine similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to get recommendations for a movie title
def get_recommendations(movie_title):
    idx = movies_df.index[movies_df['title'] == movie_title].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Exclude the input movie itself
    movie_indices = [i[0] for i in sim_scores]
    return movies_df['title'].iloc[movie_indices] # type: ignore

# Get recommendations for a movie
input_movie = 'Movie A'
recommendations = get_recommendations(input_movie)

print(f"Recommended movies for '{input_movie}':")
for movie in recommendations:
    print(movie)
