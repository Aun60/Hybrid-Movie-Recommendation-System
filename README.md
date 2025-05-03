# Hybrid Movie Recommendation System

A movie recommendation system that combines collaborative filtering, content-based filtering, and matrix factorization techniques to provide personalized movie recommendations.

## Overview

This project implements a hybrid recommendation system that leverages multiple recommendation techniques to suggest movies to users based on their preferences and movie content features. The system combines:

1. **Collaborative Filtering**: Recommends movies based on similar users' preferences
2. **Content-Based Filtering**: Recommends movies similar to those the user has enjoyed
3. **Matrix Factorization**: Uses SVD to discover latent factors in user-movie interactions

By combining these approaches, the system provides more robust and diverse recommendations than any single approach could offer.

## Dataset

The project uses the MovieLens dataset, which includes:
- User ratings for movies (on a scale of 1-5)
- Movie information (title, genres, etc.)

The dataset needs to be downloaded separately and placed in the `Dataset/` directory with the following files:
- `movie.csv`: Contains movie information (movieId, title, genres)
- `rating.csv`: Contains user ratings (userId, movieId, rating, timestamp)

## Features

- **Hybrid Recommendation Engine**: Combines multiple recommendation techniques
- **Content-Based Recommendations**: Find movies similar to a specific movie
- **User-Based Recommendations**: Get personalized recommendations for a specific user
- **Evaluation Metrics**: Measure the performance of the recommendation system
- **Web Interface**: Simple web interface to interact with the recommendation system

## How It Works

### Collaborative Filtering
The system identifies similar users based on their rating patterns and recommends movies that similar users have rated highly. This is implemented using cosine similarity between user vectors.

### Content-Based Filtering
The system recommends movies similar to those the user has previously enjoyed, based on movie features like genres and titles. This is implemented using TF-IDF vectorization and cosine similarity.

### Matrix Factorization
The system uses Singular Value Decomposition (SVD) to discover latent factors that explain the rating patterns and predict user preferences. This helps capture underlying patterns in the data.

### Hybrid Integration
The final recommendations are a weighted combination of the above approaches:
- When a specific movie is provided: 60% content-based, 20% collaborative, 20% matrix factorization
- When no specific movie is provided: 50% collaborative, 50% matrix factorization

## Installation

1. Clone the repository:
```
git clone https://github.com/Aun60/Hybrid-Movie-Recommendation-System.git
cd Hybrid-Movie-Recommendation-System
```

2. Install the required packages:
```
pip install -r Requirements.txt
```

3. Download the MovieLens dataset and place the files in the `Dataset/` directory

4. Run the evaluation script:
```
python evaluate_metrics.py
```

5. To run the web interface:
```
python manage.py runserver
```

## Project Structure

- `recommender/`: Main package containing the recommendation engine
  - `recommender_engine.py`: Core recommendation algorithms
  - `models.py`: Data models
  - `urls.py`: URL routing
  - `views.py`: View functions
  - `templates/`: HTML templates
  - `static/`: CSS and JavaScript files
- `evaluate_metrics.py`: Script to evaluate the recommendation system
- `Dataset/`: Directory for the MovieLens dataset files
- `rs_web/`: Django project settings

## Evaluation

The recommendation system is evaluated using several metrics:
- **Precision@k**: Measures the proportion of recommended items that are relevant
- **Recall@k**: Measures the proportion of relevant items that were recommended
- **Hit Rate**: Proportion of users who received at least one relevant recommendation
- **F1 Score**: Harmonic mean of precision and recall

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MovieLens dataset provided by GroupLens Research
