import os
import shutil
import re
import pickle

import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix

from deep_learning.load_data import preprocess, load_data

if os.path.exists("./data/load_data.pkl"):
    X_train_data, y_train_data, X_test_data, y_test_data = pickle.load(open("./data/load_data.pkl", "rb"))
else:
    X_train_data, y_train_data, X_test_data, y_test_data = load_data()
    pickle.dump((X_train_data, y_train_data, X_test_data, y_test_data), open("./data/load_data.pkl", "wb"))

wordcloud = WordCloud(
    scale=4,
    font_path="chinese.simhei.ttf",
    background_color="white",
    max_words=100,
    max_font_size=60,
    random_state=20).generate(X_train_data[1000])
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

stop_words = open("./dict/stop_words.txt", "r", encoding="utf-8").read().split()

#Pipeline
text_clf = Pipeline([
    ('vect', TfidfVectorizer(stop_words=stop_words)),
    ('clf', MultinomialNB()),
])
text_clf.fit(X_train_data, y_train_data)
predicted = text_clf.predict(X_test_data)
print("Naive Bayes")
print(classification_report(predicted, y_test_data))

#Logistic Regression
text_clf = Pipeline([
    ('vect', TfidfVectorizer(stop_words=stop_words)),
    ('clf', LogisticRegression()),
])
text_clf.fit(X_train_data, y_train_data)
predicted_lr = text_clf.predict(X_test_data)
print("\n")
print("Logistic Regression")
print(classification_report(predicted_lr, y_test_data))

# SVM
text_clf_svm = Pipeline([
    ('vect', TfidfVectorizer(stop_words=stop_words)),
    ('clf', SGDClassifier(loss='hinge', penalty='l2')),
])
text_clf_svm.fit(X_train_data, y_train_data)
predicted_svm = text_clf_svm.predict(X_test_data)
print("\n")
print("SVM")
print(classification_report(predicted_svm, y_test_data))