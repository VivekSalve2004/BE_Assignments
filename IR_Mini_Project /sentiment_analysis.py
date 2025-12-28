# sentiment_analysis.py
import pandas as pd
import nltk
import re
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import os

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Sample tweets (replace with real data or CSV input)
sample_tweets = [
    "I love this new phone! Battery life is amazing 😍 #tech",
    "Terrible customer service, waited 2 hours and no help.",
    "It's okay, nothing special. Works fine I guess.",
    "OMG!! This movie was INCREDIBLE!!! Best ever! 🎉",
    "Lost my wallet today... feeling so sad and stressed 😔",
    "Just another Monday... coffee please ☕"
]

# -------------------------------
# 1. Text Cleaning Function
# -------------------------------
def clean_tweet(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#\w+', '', text)  # Remove hashtags (keep text)
    text = re.sub(r'[^A-Za-z\s]', '', text)  # Remove special chars & numbers
    text = text.lower().strip()
    return text

# -------------------------------
# 2. Sentiment Analysis Function
# -------------------------------
def analyze_sentiment(tweet):
    scores = analyzer.polarity_scores(tweet)
    compound = scores['compound']
    
    if compound >= 0.05:
        sentiment = 'Positive'
    elif compound <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    
    return sentiment, scores

# -------------------------------
# 3. Main Processing
# -------------------------------
def main():
    # Option 1: Use sample data
    tweets = [clean_tweet(tweet) for tweet in sample_tweets]
    
    # Option 2: Or load from CSV (uncomment below)
    # df = pd.read_csv('tweets.csv')
    # tweets = df['text'].apply(clean_tweet).tolist()

    results = []
    for tweet in tweets:
        sentiment, scores = analyze_sentiment(tweet)
        results.append({
            'tweet': tweet,
            'sentiment': sentiment,
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu'],
            'compound': scores['compound']
        })

    # Convert to DataFrame
    df_results = pd.DataFrame(results)

    # -------------------------------
    # 4. Save Results
    # -------------------------------
    os.makedirs('output', exist_ok=True)
    df_results.to_csv('output/results.csv', index=False)
    print("Results saved to output/results.csv")

    # -------------------------------
    # 5. Visualization
    # -------------------------------
    # Sentiment Distribution
    sentiment_counts = df_results['sentiment'].value_counts()
    plt.figure(figsize=(8, 6))
    sentiment_counts.plot(kind='bar', color=['green', 'gray', 'red'])
    plt.title('Tweet Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Tweets')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('output/sentiment_distribution.png')
    plt.show()

    # Word Cloud (all cleaned tweets)
    all_text = ' '.join(tweets)
    if all_text.strip():  # Check if not empty
        wordcloud = WordCloud(width=800, height=400, background_color='white',
                              stopwords=set(stopwords.words('english'))).generate(all_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Tweet Word Cloud')
        plt.tight_layout()
        plt.savefig('output/wordcloud.png')
        plt.show()

    # Print summary
    print("\nSentiment Summary:")
    print(sentiment_counts)

if __name__ == "__main__":
    main()