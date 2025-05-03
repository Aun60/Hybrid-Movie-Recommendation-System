import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import time

# Import the recommender engine
from recommender.recommender_engine import get_recommendations, ratings, movies

def calculate_precision_at_k(recommended_items, relevant_items, k=10):
    """
    Calculate precision@k for a single user

    Args:
        recommended_items: List of recommended item IDs
        relevant_items: List of relevant (ground truth) item IDs
        k: Number of recommendations to consider

    Returns:
        Precision@k value
    """
    if len(recommended_items) == 0:
        return 0.0

    # Consider only top-k recommendations
    recommended_items = recommended_items[:k]

    # Count relevant items in the recommendations
    relevant_and_recommended = len(set(relevant_items) & set(recommended_items))

    return relevant_and_recommended / min(k, len(recommended_items))

def calculate_recall_at_k(recommended_items, relevant_items, k=10):
    """
    Calculate recall@k for a single user

    Args:
        recommended_items: List of recommended item IDs
        relevant_items: List of relevant (ground truth) item IDs
        k: Number of recommendations to consider

    Returns:
        Recall@k value
    """
    if len(relevant_items) == 0:
        return 0.0

    # top-k recommendations
    recommended_items = recommended_items[:k]

    # Count relevant items in the recommendations
    relevant_and_recommended = len(set(relevant_items) & set(recommended_items))

    return relevant_and_recommended / len(relevant_items)

def evaluate_recommender(k=10, num_users=50, threshold=3.0):
    """
    Evaluate the recommender system using precision and recall metrics

    Args:
        k: Number of recommendations to consider
        num_users: Number of users to evaluate
        threshold: Rating threshold to consider an item relevant (e.g., 3.0 out of 5)

    Returns:
        Dictionary with evaluation metrics
    """
    print(f"Evaluating recommender system with k={k} for {num_users} users...")

    # Create a debug flag to print detailed information
    debug = True

    # Create a title to ID mapping for faster lookups
    title_to_id_map = {}
    for _, row in movies.iterrows():
        title_to_id_map[row['title']] = row['movieId']

    # Get users with sufficient ratings
    user_rating_counts = ratings['userId'].value_counts()
    users_with_enough_ratings = user_rating_counts[user_rating_counts >= 20].index.tolist()

    # Limit to specified number of users for faster evaluation
    if num_users > 0 and num_users < len(users_with_enough_ratings):
        selected_users = np.random.choice(users_with_enough_ratings, num_users, replace=False)
    else:
        selected_users = users_with_enough_ratings[:num_users]

    print(f"Selected {len(selected_users)} users for evaluation")

    # For each user, evaluate recommendations
    precision_values = []
    recall_values = []
    hit_rate_values = []

    for i, user_id in enumerate(selected_users):
        if i % 10 == 0:
            print(f"Processing user {i+1}/{len(selected_users)}")

        # Get all ratings for this user
        user_ratings = ratings[ratings['userId'] == user_id]

        # Use a different approach: hold out 5 highly-rated movies as the test set
        highly_rated = user_ratings[user_ratings['rating'] >= threshold]

        # Skip users with too few highly rated items
        if len(highly_rated) < 5:
            continue

        # Hold out 5 random highly-rated items as the test set
        test_items = highly_rated.sample(min(5, len(highly_rated)))['movieId'].tolist()

        # The rest of the highly-rated items are the training set
        train_items = highly_rated[~highly_rated['movieId'].isin(test_items)]['movieId'].tolist()

        # Skip if no test items
        if not test_items:
            continue

        # Get recommendations for user
        recs = get_recommendations(user_id, top_n=k)

        if isinstance(recs, str):  # Error message returned
            continue

        # Extract movie IDs from recommendations
        recommended_items = []
        for _, row in recs.iterrows():
            title = row['title']
            # Use our mapping for faster lookup
            if title in title_to_id_map:
                recommended_items.append(title_to_id_map[title])
            else:
                # Fallback to dataframe lookup
                movie_match = movies[movies['title'] == title]
                if not movie_match.empty:
                    recommended_items.append(movie_match.iloc[0]['movieId'])

        # Debug information
        if debug and i < 5:  # Print for first 5 users
            print(f"\nDebug for user {user_id}:")
            print(f"  Total ratings: {len(user_ratings)}")
            print(f"  Highly rated items: {len(highly_rated)}")
            print(f"  Test items (relevant): {test_items}")
            print(f"  Recommended items: {recommended_items}")
            print(f"  Intersection: {set(recommended_items) & set(test_items)}")

        # Calculate metrics
        if recommended_items:
            # Precision: fraction of recommended items that are relevant
            precision = calculate_precision_at_k(recommended_items, test_items, k)
            precision_values.append(precision)

            # Recall: fraction of relevant items that are recommended
            recall = calculate_recall_at_k(recommended_items, test_items, k)
            recall_values.append(recall)

            # Hit rate: 1 if at least one recommended item is relevant, 0 otherwise
            hit = 1 if len(set(recommended_items) & set(test_items)) > 0 else 0
            hit_rate_values.append(hit)

    # Calculate average metrics
    avg_precision = np.mean(precision_values) if precision_values else 0
    avg_recall = np.mean(recall_values) if recall_values else 0
    hit_rate = np.mean(hit_rate_values) if hit_rate_values else 0

    # Calculate F1 score
    f1_score = 0
    if avg_precision > 0 and avg_recall > 0:
        f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)

    # Return metrics
    metrics = {
        f'Precision@{k}': avg_precision,
        f'Recall@{k}': avg_recall,
        'F1 Score': f1_score,
        'Hit Rate': hit_rate,
        'Number of Users Evaluated': len(precision_values)
    }

    return metrics

def evaluate_content_based(movie_title, k=10, threshold=3.0):
    """
    Evaluate content-based recommendations for a specific movie

    Args:
        movie_title: Title of the movie to evaluate
        k: Number of recommendations to consider
        threshold: Rating threshold to consider an item relevant (e.g., 3.0 out of 5)

    Returns:
        Dictionary with evaluation metrics
    """
    print(f"Evaluating content-based recommendations for '{movie_title}' with k={k}...")

    # Create a debug flag to print detailed information
    debug = True

    # Create a title to ID mapping for faster lookups
    title_to_id_map = {}
    for _, row in movies.iterrows():
        title_to_id_map[row['title']] = row['movieId']

    # Find movie ID from title
    movie_match = movies[movies['title'] == movie_title]
    if movie_match.empty:
        # Try partial match
        movie_match = movies[movies['title'].str.contains(movie_title, case=False)]
        if movie_match.empty:
            print(f"Movie '{movie_title}' not found in the dataset.")
            return None
        print(f"Using closest match: '{movie_match.iloc[0]['title']}'")
        movie_title = movie_match.iloc[0]['title']  # Update to exact title

    movie_id = movie_match.iloc[0]['movieId']
    print(f"Movie ID: {movie_id}")

    # Get users who rated this movie highly
    high_ratings = ratings[(ratings['movieId'] == movie_id) & (ratings['rating'] >= threshold)]

    if high_ratings.empty:
        print(f"No users found who rated '{movie_title}' highly.")
        return None

    # Get users with sufficient ratings
    user_counts = ratings['userId'].value_counts()
    users_with_enough = user_counts[user_counts >= 20].index

    # Filter high_ratings to only include users with enough ratings
    high_ratings_filtered = high_ratings[high_ratings['userId'].isin(users_with_enough)]

    if high_ratings_filtered.empty:
        print(f"No users with sufficient ratings found who rated '{movie_title}' highly.")
        return None

    # Get a sample of users (up to 20)
    sample_size = min(20, len(high_ratings_filtered))
    sample_users = high_ratings_filtered['userId'].sample(sample_size).tolist()
    print(f"Evaluating with {len(sample_users)} users who rated this movie highly")

    # Get recommendations for these users
    precision_values = []
    recall_values = []
    hit_rate_values = []

    for i, user_id in enumerate(sample_users):
        # Get all ratings for this user
        user_ratings = ratings[ratings['userId'] == user_id]

        # Get highly rated movies for this user
        highly_rated = user_ratings[user_ratings['rating'] >= threshold]

        # Skip users with too few highly rated items
        if len(highly_rated) < 5:
            continue

        # Sample 5 highly rated movies as the test set (excluding the input movie)
        test_candidates = highly_rated[highly_rated['movieId'] != movie_id]
        if len(test_candidates) < 5:
            continue

        test_items = test_candidates.sample(min(5, len(test_candidates)))['movieId'].tolist()

        # Skip if no test items
        if not test_items:
            continue

        # Get recommendations for user with the specified movie
        recs = get_recommendations(user_id, movie_title=movie_title, top_n=k)

        if isinstance(recs, str):  # Error message returned
            continue

        # Extract movie IDs from recommendations
        recommended_items = []
        for _, row in recs.iterrows():
            title = row['title']
            # Use our mapping for faster lookup
            if title in title_to_id_map:
                recommended_items.append(title_to_id_map[title])
            else:
                # Fallback to dataframe lookup
                rec_match = movies[movies['title'] == title]
                if not rec_match.empty:
                    recommended_items.append(rec_match.iloc[0]['movieId'])

        # Debug information
        if debug and i < 5:  # Print for first 5 users
            print(f"\nDebug for user {user_id} with movie '{movie_title}':")
            print(f"  Total ratings: {len(user_ratings)}")
            print(f"  Highly rated items: {len(highly_rated)}")
            print(f"  Test items (relevant): {test_items}")
            print(f"  Recommended items: {recommended_items}")
            print(f"  Intersection: {set(recommended_items) & set(test_items)}")

        # Calculate metrics
        if recommended_items:
            # Precision: fraction of recommended items that are relevant
            precision = calculate_precision_at_k(recommended_items, test_items, k)
            precision_values.append(precision)

            # Recall: fraction of relevant items that are recommended
            recall = calculate_recall_at_k(recommended_items, test_items, k)
            recall_values.append(recall)

            # Hit rate: 1 if at least one recommended item is relevant, 0 otherwise
            hit = 1 if len(set(recommended_items) & set(test_items)) > 0 else 0
            hit_rate_values.append(hit)

    # Calculate average metrics
    avg_precision = np.mean(precision_values) if precision_values else 0
    avg_recall = np.mean(recall_values) if recall_values else 0
    hit_rate = np.mean(hit_rate_values) if hit_rate_values else 0

    # Calculate F1 score
    f1_score = 0
    if avg_precision > 0 and avg_recall > 0:
        f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)

    # Return metrics
    metrics = {
        f'Content-Based Precision@{k}': avg_precision,
        f'Content-Based Recall@{k}': avg_recall,
        'Content-Based F1 Score': f1_score,
        'Content-Based Hit Rate': hit_rate,
        'Number of Users Evaluated': len(precision_values)
    }

    return metrics

if __name__ == "__main__":
    print("Movie Recommender System Evaluation")
    print("===================================")

    # Ask user what to evaluate
    print("\nWhat would you like to evaluate?")
    print("1. Overall recommender system")
    print("2. Content-based recommendations for a specific movie")
    print("3. Both")

    choice = input("\nEnter your choice (1-3): ")

    # Set parameters
    k_value = 10
    try:
        k_input = input(f"\nEnter number of recommendations to evaluate (default: {k_value}): ")
        if k_input.strip():
            k_value = int(k_input)
    except ValueError:
        print(f"Invalid input. Using default value: {k_value}")

    # Evaluate based on choice
    if choice in ['1', '3']:
        # Evaluate overall recommender system
        num_users = 50
        try:
            users_input = input(f"\nEnter number of users to evaluate (default: {num_users}): ")
            if users_input.strip():
                num_users = int(users_input)
        except ValueError:
            print(f"Invalid input. Using default value: {num_users}")

        print("\nEvaluating overall recommender system...")
        print("This may take a few minutes...")
        metrics = evaluate_recommender(k=k_value, num_users=num_users)

        print("\nRecommender System Evaluation Results:")
        print("--------------------------------------")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}" if isinstance(value, float) else f"{metric}: {value}")

    if choice in ['2', '3']:
        # Evaluate content-based recommendations
        movie_title = input("\nEnter movie title to evaluate content-based recommendations: ")

        print("\nEvaluating content-based recommendations...")
        cb_metrics = evaluate_content_based(movie_title, k=k_value)

        if cb_metrics:
            print("\nContent-Based Evaluation Results:")
            print("--------------------------------")
            for metric, value in cb_metrics.items():
                print(f"{metric}: {value:.4f}" if isinstance(value, float) else f"{metric}: {value}")

    print("\nEvaluation complete!")
