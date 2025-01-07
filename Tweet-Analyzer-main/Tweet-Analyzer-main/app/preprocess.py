import pandas as pd

def filter_tweets(keyword):
    # Load the dataset
    df = pd.read_csv("dataset/twitter_dataset.csv")

    # Filter tweets containing the keyword (case-insensitive)
    filtered_df = df[df["Text"].str.contains(keyword, case=False, na=False)]

    # Return the tweet text for analysis
    return filtered_df["Text"].tolist()