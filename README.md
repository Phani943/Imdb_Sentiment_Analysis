# IMDB Sentiment Analysis

This repository contains a Jupyter Notebook `imdb-review-classifier.ipynb` for performing sentiment analysis on IMDb movie reviews. The goal is to classify reviews as either positive or negative based on their content.

## Dataset

The dataset used is the IMDb movie reviews dataset. It contains 50,000 reviews labeled as positive or negative. You can find the dataset in the `imdb-reviews-data` directory. The dataset is divided evenly with 25,000 positive reviews and 25,000 negative reviews.


## Repository Structure

```
Imdb_sentiment_analysis/
├── imdb-review-classifier.ipynb
├── imdb-reviews-data/
│   └── IMDB Dataset.csv
├── README.md
└── LICENSE
```

## Overview

The notebook provides a step-by-step process to:
1. Load and preprocess the IMDb dataset.
2. Transform the text data into numerical features.
3. Train a machine learning model to classify sentiments.
4. Evaluate the model's performance.
5. Visualize the results.

### Load and Preprocess the IMDb Dataset

In this step, we load the IMDb dataset and preprocess the text data. Preprocessing involves:
- Removing HTML tags
- Removing punctuation
- Removing numbers
- Converting text to lowercase
- Removing stop words (common words that don't carry much meaning, like 'and', 'the', etc.)

### Transform the Text Data into Numerical Features

We use the TF-IDF (Term Frequency-Inverse Document Frequency) Vectorizer to convert the text data into numerical features. TF-IDF helps in understanding the importance of a word in a document relative to a collection of documents.

### Train a Machine Learning Model

We use a RandomForest Classifier to train the model. This classifier is an ensemble learning method that operates by constructing a multitude of decision trees during training and outputting the mode of the classes for classification tasks.

### Evaluate the Model's Performance

The performance of the model is evaluated using metrics like accuracy, precision, recall, and F1-score. We also visualize the results using a confusion matrix to understand the number of correct and incorrect predictions.

### Visualize the Results

Visualizations help in better understanding the performance of the model. We use Seaborn and Matplotlib libraries to create plots for visualizing the confusion matrix and other metrics.

## Dependencies

The following libraries are used in this project:

```
re
nltk
pandas
seaborn
matplotlib
sklearn
```
