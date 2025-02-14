import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast


movies = pd.read_csv('C:/Users/alokr/OneDrive/Desktop/tmdb_5000_movies.csv')
credits = pd.read_csv('C:/Users/alokr/OneDrive/Desktop/tmdb_5000_credits.csv')


movies = movies.merge(credits, on='title')


def process_data():
    
    movies_clean = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']].copy()
    
    
    for feature in ['overview', 'genres', 'keywords', 'cast', 'crew']:
        movies_clean.loc[:, feature] = movies_clean[feature].fillna('')
    
    return movies_clean

movies_clean = process_data()


def convert_json_to_string(json_data):
    
    try:
        data = ast.literal_eval(json_data)
        return ' '.join(item['name'] for item in data)
    except (ValueError, SyntaxError):
        return ''

movies_clean['genres'] = movies_clean['genres'].apply(convert_json_to_string)
movies_clean['keywords'] = movies_clean['keywords'].apply(convert_json_to_string)


def get_top_actors(cast):
    try:
        data = ast.literal_eval(cast)
        return ' '.join(actor['name'] for actor in data[:3])  # Top 3 actors
    except (ValueError, SyntaxError):
        return ''

movies_clean['cast'] = movies_clean['cast'].apply(get_top_actors)


def get_director(crew):
    try:
        data = ast.literal_eval(crew)
        for member in data:
            if member['job'] == 'Director':
                return member['name']
        return ''
    except (ValueError, SyntaxError):
        return ''

movies_clean['crew'] = movies_clean['crew'].apply(get_director)


movies_clean['combined_features'] = (
    movies_clean['genres'] + ' ' +
    movies_clean['keywords'] + ' ' +
    movies_clean['cast'] + ' ' +
    movies_clean['crew'] + ' ' +
    movies_clean['overview']
)


vectorizer = CountVectorizer(stop_words='english')
feature_matrix = vectorizer.fit_transform(movies_clean['combined_features'])


cosine_sim = cosine_similarity(feature_matrix)


def recommend_movies(movie_title, num_recommendations=5):
    
    
    try:
        movie_index = movies_clean[movies_clean['title'].str.lower() == movie_title.lower()].index[0]
    except IndexError:
        return "Movie not found in the dataset!"
    
   
    similarity_scores = list(enumerate(cosine_sim[movie_index]))
    
   
    sorted_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:]
    
   
    recommendations = []
    for i in range(num_recommendations):
        movie_idx = sorted_movies[i][0]
        recommendations.append(movies_clean.iloc[movie_idx]['title'])
    
   
    return list(dict.fromkeys(recommendations))


movie_to_search = input("enter the movie name: ")
print(f"Movies similar to '{movie_to_search}':")
print(recommend_movies(movie_to_search, num_recommendations=5))
