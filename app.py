import pandas as pd
import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate
import streamlit as st

# Download NLTK VADER for Sentiment Analysis
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Load MovieLens dataset
movies = pd.read_csv("movies.csv")  # movieId, title, genres
ratings = pd.read_csv("ratings.csv")  # userId, movieId, rating
tags = pd.read_csv("tags.csv")  # userId, movieId, tag

# Preprocess genres for Content-Based Filtering
movies["genres"] = movies["genres"].str.replace("|", " ").str.lower()

# TF-IDF Vectorization for Content-Based Filtering
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["genres"])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend_movies_content(title):
    if title not in movies["title"].values:
        return ["Movie not found!"]
    
    idx = movies[movies["title"] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    movie_indices = [i[0] for i in sim_scores]
    
    return movies["title"].iloc[movie_indices].tolist()

# Collaborative Filtering using SVD
reader = Reader(rating_scale=(0.5, 5))
data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)
svd = SVD()
cross_validate(svd, data, cv=5, verbose=True)
trainset = data.build_full_trainset()
svd.fit(trainset)

def predict_rating(userId, movieId):
    return svd.predict(userId, movieId).est

def recommend_movies_collaborative(userId, num_recommendations=5):
    user_movies = ratings[ratings["userId"] == userId]["movieId"].unique()
    all_movies = ratings["movieId"].unique()
    unseen_movies = [m for m in all_movies if m not in user_movies]

    predictions = [(m, predict_rating(userId, m)) for m in unseen_movies]
    predictions.sort(key=lambda x: x[1], reverse=True)

    top_movies = [movies[movies["movieId"] == m[0]]["title"].values[0] for m in predictions[:num_recommendations]]
    return top_movies

# Sentiment Analysis on User Reviews
def analyze_sentiment(movieId):
    movie_reviews = tags[tags["movieId"] == movieId]["tag"]
    sentiments = [sia.polarity_scores(str(tag))["compound"] for tag in movie_reviews]
    return np.mean(sentiments) if sentiments else 0

def hybrid_recommendation(userId, movie_title):
    content_recs = recommend_movies_content(movie_title)
    collab_recs = recommend_movies_collaborative(userId)
    combined_recs = list(set(content_recs + collab_recs))
    
    sentiment_scores = {movie: analyze_sentiment(movies[movies["title"] == movie]["movieId"].values[0]) for movie in combined_recs}
    sorted_recs = sorted(sentiment_scores.items(), key=lambda x: x[1], reverse=True)
    
    return [movie[0] for movie in sorted_recs]

# Streamlit Web App
st.title("🎬 Hybrid Movie Recommendation System")
user_id = st.number_input("Enter User ID:", min_value=1, step=1)
movie_name = st.text_input("Enter a movie title:")

if st.button("Get Recommendations"):
    recommendations = hybrid_recommendation(user_id, movie_name)
    st.write("Recommended Movies:", recommendations)
