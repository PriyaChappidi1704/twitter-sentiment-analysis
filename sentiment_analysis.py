import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Download NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

print("Loading dataset...")
df = pd.read_csv('Tweets.csv')

# Force column names to lowercase and strip spaces to prevent formatting errors
df.columns = df.columns.str.strip().str.lower()

# Extract 'text' (Column K) and 'airline_sentiment' (Column B)
df = df[['text', 'airline_sentiment']].dropna()

# Text cleaning function
def clean_tweet(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE) # remove URLs
    text = re.sub(r'\@\w+|\#','', text) # remove @mentions and hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text) # remove numbers and special characters
    text = text.lower() # lowercase
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words] # remove stop words
    return " ".join(tokens)

print("Cleaning tweet text...")
df['clean_text'] = df['text'].apply(clean_tweet)

# Define Features (X) and Target (y)
X = df['clean_text']
y = df['airline_sentiment']

# Split data into 80% training and 20% testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text into numerical vectors using TF-IDF
print("Vectorizing text using TF-IDF...")
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Logistic Regression classification model
print("Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Make predictions and evaluate performance
y_pred = model.predict(X_test_tfidf)
print("\n--- Sentiment Analysis Model Results ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print(classification_report(y_test, y_pred))