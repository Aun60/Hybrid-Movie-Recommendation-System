from django.shortcuts import render
from .recommender_engine import get_recommendations
import pandas as pd

def index(request):
    recommendations = []
    error = ""
    user_id = ""
    movie_title = ""

    if request.method == 'POST':
        user_id = request.POST.get('user_id', '')
        movie_title = request.POST.get('movie_title', '')

        try:
            user_id = int(user_id)
            result = get_recommendations(user_id, movie_title, top_n=20)

            # is result a string (error message)
            if isinstance(result, str):
                error = result
                recommendations = []
            
            elif isinstance(result, pd.DataFrame):
                if result.empty:
                    error = "No recommendations found."
                    recommendations = []
                else:
                    # Converting DataFrame to list of dictionaries
                    recommendations = result.to_dict('records')

            else:
                error = "Unexpected result type from recommendation engine."
                recommendations = []

        except ValueError:
            error = "Invalid User ID. Please enter a numeric value."
         
        except Exception as e:
            error = f"An error occurred: {str(e)}"
        
    return render(request, 'recommender/index.html', {
        'recommendations': recommendations,
        'error': error,
        'user_id': user_id,
        'movie_title': movie_title
    })
