import pandas as pd
from dateutil.relativedelta import relativedelta
from datetime import datetime
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
    
df=pd.read_excel('E-commerce customer analysis.xlsx')
#print(df.head())

def parse_date(date_str):
    if pd.isna(date_str):
        return pd.NaT

    date_str = str(date_str).strip().lower()

    # Handle relative date like "8 months ago"
    if 'ago' in date_str:
        try:
            number = int(date_str.split()[0])
            unit = date_str.split()[1]

            today = datetime.today()
            if 'month' in unit:
                return today - relativedelta(months=number)
            elif 'day' in unit:
                return today - relativedelta(days=number)
            elif 'year' in unit:
                return today - relativedelta(years=number)
            else:
                return pd.NaT
        except:
            return pd.NaT

    # Handle absolute date like "Feb, 2020"
    try:
        return pd.to_datetime(date_str, format='%b, %Y')
    except:
        return pd.NaT
df['clean_date'] = df['date'].apply(parse_date)
#print(df[['date', 'clean_date']].head(20))

df.to_csv('E-commerce_cleaned_reviews_.csv', index=False)
#print(df.columns)
#print(df.info())

#EDA

#Distribution of Ratings
'''plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='rating', palette='viridis')
plt.title('Distribution of Product Ratings')
plt.xlabel('Rating')
plt.ylabel('Number of Reviews')
plt.show()

#Most Reviewed Products
top_products = df['product_title'].value_counts().head(10)

plt.figure(figsize=(10, 6))
sns.barplot(y=top_products.index, x=top_products.values, palette='Blues_r')
plt.title('Top 10 Most Reviewed Products')
plt.xlabel('Number of Reviews')
plt.ylabel('Product')
plt.show()

# Review Length Distribution
df['review_length'] = df['review'].astype(str).apply(lambda x: len(x.split()))

plt.figure(figsize=(10, 5))
sns.histplot(df['review_length'], bins=50, kde=True, color='orchid')
plt.title('Review Length Distribution (in Words)')
plt.xlabel('Words in Review')
plt.ylabel('Frequency')
plt.show()

#Ratings Over Time
df['clean_date'] = pd.to_datetime(df['clean_date'])

monthly_avg = df.set_index('clean_date').resample('M').agg({'rating': 'mean'}).reset_index()

plt.figure(figsize=(12, 6))
sns.lineplot(data=monthly_avg, x='clean_date', y='rating', marker='o')
plt.title(' Monthly Average Rating Over Time')
plt.xlabel('Month')
plt.ylabel('Average Rating')
plt.grid(True)
plt.show()'''

df['helpfulness'] = df['upvotes'] - df['downvotes']
top_helpful = df.sort_values(by='helpfulness', ascending=False).head(10)

for i, row in top_helpful.iterrows():
    '''print(f"\n Review #{i}")
    print(f"Product: {row['product_title']}")
    print(f"Rating: {row['rating']}")
    print(f"Upvotes: {row['upvotes']} | Downvotes: {row['downvotes']} | Helpfulness: {row['helpfulness']}")
    print(f"Summary: {row['summary']}")
    print(f"Review: {row['review'][:300]}...")'''
 
 # Replace True/False and NaN with empty string
df['review'] = df['review'].apply(lambda x: str(x) if isinstance(x, str) else '')
   
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    if not text.strip():
        return 'neutral'
    score = analyzer.polarity_scores(text)
    compound = score['compound']
    if compound >= 0.05:
        return 'positive'
    elif compound <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# Apply safely
df['sentiment'] = df['review'].apply(get_sentiment)

# Check results
'''print(df['sentiment'].value_counts())

print(df['review'].unique()[:10])  # See sample values
print(df['review'].map(type).value_counts())  # Make sure all are str'''



#Train a Linear Regression Model to predict reviews.

df_ml = df.dropna(subset=['review', 'sentiment'])
df_ml = df_ml[df_ml['review'].astype(bool)]  # remove empty strings

X = df_ml['review']
y = df_ml['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Predictions
y_pred = model.predict(X_test_tfidf)

# Evaluate
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

df.to_csv('final E-commerce_reviews_.csv', index=False)
