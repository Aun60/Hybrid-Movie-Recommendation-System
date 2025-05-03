import pandas as pd
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

# Load data
try:
    ratings = pd.read_csv("Dataset/rating.csv")
    movies = pd.read_csv("Dataset/movie.csv")
except Exception as e:
    ratings = pd.DataFrame()
    movies = pd.DataFrame()

ratings = ratings.head(300000)

movies['genres'] = movies['genres'].fillna('')

movies['year'] = movies['title'].str.extract(r'\((\d{4})\)$')
movies['clean_title'] = movies['title'].str.replace(r'\s*\(\d{4}\)\s*$', '', regex=True)


# This combines genres with the movie title for better matching for the content-based filtering
movies['combined_features'] = movies['clean_title'] + ' ' + movies['genres'].str.replace('|', ' ')


# mapping from userId and movieId to sequential index
user_id_map = {id_: idx for idx, id_ in enumerate(ratings['userId'].unique())}
movie_id_map = {id_: idx for idx, id_ in enumerate(ratings['movieId'].unique())}
user_index_map = {v: k for k, v in user_id_map.items()}
movie_index_map = {v: k for k, v in movie_id_map.items()}

# Pre Processing to lower titles and no duplication of titles

title_to_id = {}
for idx, row in movies.iterrows():
    clean_title_lower = row['clean_title'].lower()
    if clean_title_lower not in title_to_id:
        title_to_id[clean_title_lower] = row['movieId']

ratings['userIndex'] = ratings['userId'].map(user_id_map)
ratings['movieIndex'] = ratings['movieId'].map(movie_id_map)

# Build sparse user-item matrix
num_users = len(user_id_map)
num_movies = len(movie_id_map)
sparse_matrix = csr_matrix((ratings['rating'], (ratings['userIndex'], ratings['movieIndex'])),
                           shape=(num_users, num_movies))


user_similarity = cosine_similarity(sparse_matrix, dense_output=False)

# Content-based similarity on combined features (title + genres)
tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=2)
tfidf_matrix = tfidf.fit_transform(movies['combined_features'])
content_similarity = cosine_similarity(tfidf_matrix, dense_output=False)

# Matrix Factorization : : Reduces the dimensionality of the user-item matrix to discover latent patterns.
Number_of_latent_factors = 50
svd = TruncatedSVD(n_components=Number_of_latent_factors, random_state=42)
latent_matrix = svd.fit_transform(sparse_matrix)

def find_similar_movies(movie_id, n=10):

    if movie_id not in movie_id_map:
        return pd.DataFrame()

    # Getting movie details
    movie_idx = movie_id_map[movie_id]

    movie_similarities = content_similarity[movie_idx].toarray().flatten()

    # Get top similar movies
    similar_indices = np.argsort(movie_similarities)[::-1][1:n*2]  # Get more indices to handle missing ones

    # Filter out indices that aren't in the map
    valid_indices = [idx for idx in similar_indices if idx in movie_index_map]
    valid_indices = valid_indices[:n]  # Limit to requested number

    if not valid_indices:
        return pd.DataFrame(columns=['title', 'genres'])

    similar_ids = [movie_index_map[idx] for idx in valid_indices]
    similar_movies = movies[movies['movieId'].isin(similar_ids)][['title', 'genres']]

    return similar_movies.reset_index(drop=True)

def collaborative_filtering(user_id, n=10):
    if user_id not in user_id_map:
        return []

    user_idx = user_id_map[user_id]
    sim_scores = user_similarity[user_idx].toarray().flatten()
    top_users = np.argsort(sim_scores)[::-1][1:11]  # Skip self

    # Get ratings from top similar users
    top_ratings = sparse_matrix[top_users].toarray()
    sim_weights = sim_scores[top_users]

    weighted_scores = np.dot(sim_weights, top_ratings)
    sim_sum = np.sum(sim_weights)
    if sim_sum == 0:
        return []

    final_scores = weighted_scores / sim_sum

    # Remove movies already rated ChatGpt(recommended when found errors)
    rated_movie_indices = sparse_matrix[user_idx].nonzero()[1]
    final_scores[rated_movie_indices] = -1

    top_movie_indices = np.argsort(final_scores)[::-1][:n]
    top_movie_ids = [movie_index_map[i] for i in top_movie_indices]

    recommended = movies[movies['movieId'].isin(top_movie_ids)][['title', 'genres']]

    return recommended.reset_index(drop=True)

def matrix_factorization_recommendations(user_id, n=10):
    if user_id not in user_id_map:
        return []

    user_idx = user_id_map[user_id]
    user_vector = latent_matrix[user_idx]

    # Calculate predicted ratings for all movies Chatgpt did this part as we didnt knew the methods of Matrix Factorization
    movie_vectors = svd.components_.T
    predicted_ratings = np.dot(user_vector, movie_vectors.T)

    # Remove movies already rated
    rated_movie_indices = sparse_matrix[user_idx].nonzero()[1]
    predicted_ratings[rated_movie_indices] = -1

    # Get top N movies
    top_movie_indices = np.argsort(predicted_ratings)[::-1][:n]
    top_movie_ids = [movie_index_map[i] for i in top_movie_indices]

    recommended = movies[movies['movieId'].isin(top_movie_ids)][['title', 'genres']]

    return recommended.reset_index(drop=True)


def get_recommendations(user_id, movie_title=None, top_n=20):   #CF + CBF

    try:
        top_n = int(top_n)
    except ValueError:
        top_n = 20  # Increased default from 10 to 20

    if user_id not in user_id_map:
        return "No recommendations available. User ID not found."

    # Initialize recommendations list
    cf_recs = collaborative_filtering(user_id, n=top_n)
    mf_recs = matrix_factorization_recommendations(user_id, n=top_n)

    cb_recs = pd.DataFrame()

    if movie_title and isinstance(movie_title, str) and movie_title.strip():
        movie_title_lower = movie_title.lower().strip()

        # Try exact match first
        if movie_title_lower in title_to_id:
            movie_id = title_to_id[movie_title_lower]
            cb_recs = find_similar_movies(movie_id, n=top_n * 2)
        else:
            safe_title = re.escape(movie_title_lower)
            containing_matches = movies[movies['clean_title'].str.lower().str.contains(safe_title)]

            if not containing_matches.empty:
                best_match = containing_matches.iloc[containing_matches['clean_title'].str.len().argsort()[0]]
                movie_id = best_match['movieId']
                cb_recs = find_similar_movies(movie_id, n=top_n * 2)

            else:  #Recommended By ChatGpt
                # Try word-by-word matching as a last resort
                words = movie_title_lower.split()
                if len(words) > 1:
                    # Try with the first two words
                    partial_title = ' '.join(words[:2])
                    safe_partial = re.escape(partial_title)
                    partial_matches = movies[movies['clean_title'].str.lower().str.contains(safe_partial)]

                    if not partial_matches.empty:
                        best_match = partial_matches.iloc[0]
                        movie_id = best_match['movieId']
                        cb_recs = find_similar_movies(movie_id, n=top_n * 2)
                    else:
                        pass  # Movie title not found
                else:
                    pass 

    final_recs = pd.DataFrame()
    #Combining
    #if no content-based recommendations
    if not cb_recs.empty:
        # 60% content-based and 40% collaborative/matrix factorization
        n_cb = int(top_n * 0.6)
        n_cf_mf = top_n - n_cb
        cb_part = cb_recs.head(n_cb)

        #collaborative and matrix factorization recommendations
        cf_part = cf_recs.head(n_cf_mf // 2)
        mf_part = mf_recs.head(n_cf_mf - n_cf_mf // 2)

        # Combine all parts
        combined_parts = []
        if not cb_part.empty:
            combined_parts.append(cb_part)
        if not cf_part.empty:
            combined_parts.append(cf_part)
        if not mf_part.empty:
            combined_parts.append(mf_part)

        if combined_parts:
            final_recs = pd.concat(combined_parts)
        
            final_recs = final_recs.drop_duplicates(subset=['title'])
    else:   #ChatGpt Recommended
        # Without content-based, 50% collaborative filtering and 50% matrix factorization
        cf_part = cf_recs.head(top_n)
        mf_part = mf_recs.head(top_n)

        combined_parts = []
        if not cf_part.empty:
            combined_parts.append(cf_part)
        if not mf_part.empty:
            combined_parts.append(mf_part)

        if combined_parts:
            final_recs = pd.concat(combined_parts)
            # Remove duplicates but keep more recommendations
            final_recs = final_recs.drop_duplicates(subset=['title'])

    if final_recs.empty:
        return "No recommendations available."

    return final_recs.reset_index(drop=True)
