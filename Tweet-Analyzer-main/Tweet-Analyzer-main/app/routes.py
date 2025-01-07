from flask import render_template, request
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
import io
import base64

# Load the BERT sentiment analysis model
sentiment_analyzer = pipeline("sentiment-analysis")

# Route to render the index page with the form
def index():
    return render_template('index.html')

# Route to handle form submission and display results
def results():
    keyword = request.form.get('keyword')  # Get the keyword from the form
    
    # Load the dataset (now it only has 'Text' column)
    df = pd.read_csv('dataset/twitter_dataset.csv')

    # Filter tweets that contain the keyword (case-insensitive)
    filtered_df = df[df['Text'].str.contains(keyword, case=False, na=False)]

    # Initialize sentiment counters (positive, negative, neutral)
    sentiment_labels = ['Positive', 'Negative']
    sentiment_values = [0, 0]
    
    # Create separate lists for positive, negative, and neutral tweets
    positive_tweets = []
    negative_tweets = []
    
    
    if not filtered_df.empty:
        # Perform sentiment analysis on the filtered tweets
        for tweet in filtered_df['Text']:
            sentiment = sentiment_analyzer(tweet)[0]['label'].lower()

            if sentiment == 'positive' :
                sentiment_values[0] += 1
                positive_tweets.append(tweet)
            elif sentiment == 'negative' :
                sentiment_values[1] += 1
                negative_tweets.append(tweet)
            
            
        positive_tweets = positive_tweets or []
        negative_tweets = negative_tweets or []
  

        # Pass data to the results.html template
        return render_template('results.html', 
                               results=filtered_df.to_dict(orient='records'), 
                               keyword=keyword, 
                               sentiment_labels=sentiment_labels, 
                               sentiment_values=sentiment_values,
                               positive_tweets=positive_tweets,
                               negative_tweets=negative_tweets,
                               )
    else:
        # If no tweets match the keyword, just display the message
        return render_template('results.html', keyword=keyword, results=None)

