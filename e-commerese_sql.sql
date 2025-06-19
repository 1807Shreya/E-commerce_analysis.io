CREATE DATABASE ecommerce_reviews;

USE ecommerce_reviews;

CREATE TABLE reviews (
    id INT AUTO_INCREMENT PRIMARY KEY,
    product_id VARCHAR(255),
    product_title TEXT,
    rating INT,
    summary TEXT,
    review TEXT,
    location VARCHAR(255),
    upvotes INT,
    downvotes INT,
    clean_time time(6),
    helpfulness INT,
    sentiment ENUM('positive', 'negative', 'neutral'),
    review_length INT
);

select * from reviews;

#Show total number of reviews
SELECT COUNT(*) AS total_reviews FROM reviews;

#Count of reviews by sentiment
SELECT sentiment, COUNT(*) AS count
FROM reviews
GROUP BY sentiment;

#Average rating per product
SELECT product_title, ROUND(AVG(rating), 2) AS avg_rating, COUNT(*) AS total_reviews
FROM reviews
GROUP BY product_title
ORDER BY avg_rating DESC
LIMIT 10;

#Top 5 products with most negative reviews

SELECT product_title, COUNT(*) AS negative_review_count
FROM reviews
WHERE sentiment = 'negative'
GROUP BY product_title
ORDER BY negative_review_count DESC
LIMIT 5;

#Top 5 locations with most negative reviews

SELECT location, COUNT(*) AS negative_review_count
FROM reviews
WHERE sentiment = 'negative'
GROUP BY location
ORDER BY negative_review_count DESC
LIMIT 5;

#Most helpful reviews (helpfulness highest)

SELECT product_title, rating, summary, review, helpfulness
FROM reviews
ORDER BY helpfulness DESC
LIMIT 5;

#Distribution of ratings

SELECT rating, COUNT(*) AS count
FROM reviews
GROUP BY rating
ORDER BY rating;

#Average review length by sentiment

SELECT sentiment, ROUND(AVG(review_length), 2) AS avg_review_length
FROM reviews
GROUP BY sentiment;

#to find common words in negative reviews 

SELECT review
FROM reviews
WHERE sentiment = 'negative';












