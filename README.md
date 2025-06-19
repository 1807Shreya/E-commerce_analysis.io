# E-commerce_analysis.io
# E-commerce Customer Sentiment Analysis

## Project Overview
This project analyzes customer reviews from an e-commerce platform to uncover key insights about product ratings, common complaints, brand perception, and customer sentiment. The goal is to predict customer satisfaction using machine learning and visualize the findings with an interactive dashboard.

## Data
- Dataset: Customer reviews including product ID, title, rating, summary, review text, location, upvotes/downvotes, review date, helpfulness score, sentiment label, and review length.
- Source: Excel file (`E-commerce_reviews_tb.xlsx`) containing real customer reviews.

## Features
- Exploratory Data Analysis (EDA) with visualizations of ratings distribution, review length, sentiment analysis, and product popularity.
- Sentiment analysis using VADER to label reviews as positive, negative, or neutral.
- Machine learning model (Logistic Regression) trained on TF-IDF features to predict sentiment from reviews.
- Dashboard built with Power BI (or Streamlit) showing insights and allowing interaction.
- SQL scripts to store and query data from a MySQL database.

## How to Run
1. Clone the repo.
2. Install required Python packages:  
   `pip install -r requirements.txt`
3. Run the Python scripts to train the model and generate predictions.
4. Open the Power BI dashboard (`e-commerce analysis_2.pbix`) to explore the visualizations.
5. (Optional) Deploy the Streamlit app using:  
   `streamlit run app.py`

## Technologies Used
- Python (pandas, scikit-learn, VADER, matplotlib, seaborn)
- Power BI
- MySQL
- Git & GitHub

## Future Improvements
- Integrate real-time data streaming for up-to-date sentiment.
- Add deeper NLP models (BERT, transformers).
- Deploy dashboard as a web app with live database connections.




