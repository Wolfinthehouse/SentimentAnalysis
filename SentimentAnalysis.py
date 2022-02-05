# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 14:32:42 2021

@author: acer
"""

import pandas as pd
import numpy as np
import nltk
nltk.download('wordnet')
import re
from bs4 import BeautifulSoup
import csv
import contractions
pd.set_option('display.max_colwidth', -1)

df = pd.read_csv("data.tsv", sep='\t', error_bad_lines=False, quoting=csv.QUOTE_NONE)

df = df[['star_rating','review_body']]
df = df.dropna(subset=['star_rating'])
df['star_rating'] = df['star_rating'].astype(int)

ratings = {5:0, 1:0, 2:0, 3:0, 4:0}
sentiment = {'Postive':0, 'Negative':0, 'Neutral':0}
def count_rating(x):
    ratings[int(x)]+=1
    if int(x)>3:
        sentiment['Postive']+=1
    elif int(x)<3:
        sentiment['Negative']+=1
    else:
        sentiment['Neutral']+=1
result = [count_rating(x) for x in df['star_rating']]
print("Postive: %d , Negative: %d , Neutral: %d" % (sentiment['Postive'],sentiment['Negative'], sentiment['Neutral']))

df = df[df['star_rating'] != 3]
df['Sentiment'] = [1 if x > 3 else 0 for x in df['star_rating']]

df_postive = df[(df.Sentiment == 1)].head(100000)
df_negative = df[(df.Sentiment == 0)].head(100000)
frames = [df_postive, df_negative]
df = pd.concat(frames)

length_before_cleaning = str(df['review_body'].str.len().mean())

df['review_body'] = df['review_body'].str.lower()

def remove_HTML(review):
    cleanText = BeautifulSoup(review, "html.parser").text
    return cleanText

df['review_body'] = df['review_body'].apply(lambda review : remove_HTML(str(review)))

regex = re.compile('[^a-zA-Z \']')
df['review_body'] = df['review_body'].apply(lambda review : regex.sub('', review))

regex = re.compile('[ +]')
df['review_body'] = df['review_body'].apply(lambda review : regex.sub(' ', review))

def contractionfunction(s):
    return contractions.fix(s)
df['review_body'] = df['review_body'].apply(lambda review :contractionfunction(review))

length_after_cleaning = str(df['review_body'].str.len().mean())
print("Length before cleaning %s , Length_after_cleaning %s" % (length_before_cleaning,length_after_cleaning))

length_before_processing = str(df['review_body'].str.len().mean())

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))

def remove_stopWords(review):
    tokens = word_tokenize(review)
    filtered_words = [word for word in tokens if word not in stop_words]
    return " ".join(filtered_words)

df['review_body'] = df['review_body'].apply(lambda review : remove_stopWords(review))

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WhitespaceTokenizer

torkenizer = WhitespaceTokenizer()
lemmatizer = WordNetLemmatizer()

def lemmatize_text(review):
    lemmatized_words = [lemmatizer.lemmatize(w) for w in torkenizer.tokenize(review)]
    return " ".join(lemmatized_words)

df['review_body'] = df['review_body'].apply(lambda review : lemmatize_text(review))

length_after_processing = str(df['review_body'].str.len().mean())
print("Length before processing %s , Length after processing %s" % (length_before_processing,length_after_processing))


from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

X_train, X_test, y_train, y_test = train_test_split(df['review_body'],df['Sentiment'], test_size=0.2, random_state=0)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

from sklearn.linear_model import Perceptron

clf = Perceptron(random_state=0)

clf.fit(X_train, y_train)

y_pred_train = clf.predict(X_train)
accuracy_train = accuracy_score(y_train, y_pred_train)
precision_score_train = precision_score(y_train, y_pred_train)
recall_score_train = recall_score(y_train, y_pred_train)
f1_score_train = f1_score(y_train, y_pred_train)

print("Accuracy %2.4f Precision %2.4f Recall %2.4f and f1-score %2.4f for Perceptron on training data" % (accuracy_train, precision_score_train, recall_score_train, f1_score_train))

y_pred_test= clf.predict(X_test)
accuracy_test = accuracy_score(y_test, y_pred_test)
precision_score_test = precision_score(y_test, y_pred_test)
recall_score_test = recall_score(y_test, y_pred_test)
f1_score_test = f1_score(y_test, y_pred_test)

print("Accuracy %2.4f Precision %2.4f Recall %2.4f and f1-score %2.4f for Perceptron on test data" % (accuracy_test, precision_score_test, recall_score_test, f1_score_test))


from sklearn.svm import LinearSVC

clf = LinearSVC()

clf.fit(X_train, y_train)

y_pred_train = clf.predict(X_train)
accuracy_train = accuracy_score(y_train, y_pred_train)
precision_score_train = precision_score(y_train, y_pred_train)
recall_score_train = recall_score(y_train, y_pred_train)
f1_score_train = f1_score(y_train, y_pred_train)

print("Accuracy %2.4f Precision %2.4f Recall %2.4f and f1-score %2.4f for SVM on training data" % (accuracy_train, precision_score_train, recall_score_train, f1_score_train))

y_pred_test= clf.predict(X_test)
accuracy_test = accuracy_score(y_test, y_pred_test)
precision_score_test = precision_score(y_test, y_pred_test)
recall_score_test = recall_score(y_test, y_pred_test)
f1_score_test = f1_score(y_test, y_pred_test)

print("Accuracy %2.4f Precision %2.4f Recall %2.4f and f1-score %2.4f for SVM on test data" % (accuracy_test, precision_score_test, recall_score_test, f1_score_test))


from sklearn.linear_model import LogisticRegression


clf = LogisticRegression(random_state=0)

clf.fit(X_train, y_train)

y_pred_train = clf.predict(X_train)
accuracy_train = accuracy_score(y_train, y_pred_train)
precision_score_train = precision_score(y_train, y_pred_train)
recall_score_train = recall_score(y_train, y_pred_train)
f1_score_train = f1_score(y_train, y_pred_train)

print("Accuracy %2.4f Precision %2.4f Recall %2.4f and f1-score %2.4f for Logistic Regression on training data" % (accuracy_train, precision_score_train, recall_score_train, f1_score_train))

y_pred_test= clf.predict(X_test)
accuracy_test = accuracy_score(y_test, y_pred_test)
precision_score_test = precision_score(y_test, y_pred_test)
recall_score_test = recall_score(y_test, y_pred_test)
f1_score_test = f1_score(y_test, y_pred_test)

print("Accuracy %2.4f Precision %2.4f Recall %2.4f and f1-score %2.4f for Logistic Regression on test data" % (accuracy_test, precision_score_test, recall_score_test, f1_score_test))

from sklearn.naive_bayes import MultinomialNB


clf =  MultinomialNB()

clf.fit(X_train, y_train)

y_pred_train = clf.predict(X_train)
accuracy_train = accuracy_score(y_train, y_pred_train)
precision_score_train = precision_score(y_train, y_pred_train)
recall_score_train = recall_score(y_train, y_pred_train)
f1_score_train = f1_score(y_train, y_pred_train)

print("Accuracy %2.4f Precision %2.4f Recall %2.4f and f1-score %2.4f for Naive Bayes on training data" % (accuracy_train, precision_score_train, recall_score_train, f1_score_train))

y_pred_test= clf.predict(X_test)
accuracy_test = accuracy_score(y_test, y_pred_test)
precision_score_test = precision_score(y_test, y_pred_test)
recall_score_test = recall_score(y_test, y_pred_test)
f1_score_test = f1_score(y_test, y_pred_test)

print("Accuracy %2.4f Precision %2.4f Recall %2.4f and f1-score %2.4f for Naive Bayes on test data" % (accuracy_test, precision_score_test, recall_score_test, f1_score_test))