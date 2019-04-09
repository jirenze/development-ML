# -*- coding: utf-8 -*-
"""
Created on Mon. April.08 2019

@author: jirenze
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
import sklearn
import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text  import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import LatentDirichletAllocation



reviews_train = load_files("E:/development/aclImdb/train/")
# load_files returns a bunch, containing training texts and training labels
text_train, y_train = reviews_train.data, reviews_train.target
#print("type of text_train: {}".format(type(text_train)))
#print("length of text_train: {}".format(len(text_train)))
#print("text_train[1]: \n{}".format(text_train[1]))
text_train = [doc.replace(b"<br />", b" ") for doc in text_train]
#print("samples per class(training): {}".format(np.bincount(y_train)))

reviews_test = load_files("E:/development/aclImdb/test/")
text_test, y_test = reviews_test.data, reviews_test.target
#print("number of documents in test data: {}".format(len(text_test)))
#print("sample per class(test): {}".format(np.bincount(y_test)))
text_test = [doc.replace(b"<br />", b" ") for doc in text_test]

#bards_words = ["the fool doth think he is wise,", "but the wise man knows himself to be a fool"]
#vect = CountVectorizer()
#vect.fit(bards_words)
#print("vocabulary size: {}".format(len(vect.vocabulary_)))
#print("vocabulary content: \n {}".format(vect.vocabulary_))
#bag_of_words = vect.transform(bards_words)
#print("bag of words: {}".format(repr(bag_of_words)))
#print("Dense representation of bag_of_words: \n {}".format(bag_of_words.toarray()))

vect = CountVectorizer().fit(text_train)
X_train = vect.transform(text_train)
#print("X_train: \n{}".format(repr(X_train)))
feature_names = vect.get_feature_names()
#print("Number of feature: {}".format(len(feature_names)))
#print("first 20 feature: \n{}".format(feature_names[: 20]))
#print("feature 20010 to 20030 : \n{}".format(feature_names[20010: 20030]))
#print("every 2000th feature: \n{}".format(feature_names[::2000]))
scores = cross_val_score(LogisticRegression(), X_train, y_train, cv=5)
print("mean cross-validation accuracy: {:.2f}".format(np.mean(scores)))

param_grid = {"C":[0.001, 0.01, 0.1, 1, 10]}
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(X_train, y_train)
print("best cross-validation score: {:.2f}".format(grid.best_score_))
print("best parameters ", grid.best_params_)

X_test = vect.transform(text_test)
print("{:.2f}".format(grid.score(X_test, y_test)))

vectC = CountVectorizer(min_df = 5).fit(text_train)
X_trainC = vectC.transform(text_train)
print("\n X_train with min_df: {}".format(repr(X_train)))

feature_namesC = vectC.get_feature_names()
print("First 50 features: \n{}".format(feature_names[:50]))
print("Features 20010 to 20030: \n{}".format(feature_names[20010:20030]))
print("Every 700th feature: \n{}".format(feature_names[::700]))

gridC = GridSearchCV(LogisticRegression(), param_grid = param_grid, cv=5)
gridC.fit(X_trainC, y_train)
print("best cross-validation score: {:.2f}".format(grid.best_score_))

print("Number of stop words: {}".format(len(ENGLISH_STOP_WORDS)))
print("Every 10th stopword: \n{}".format(list(ENGLISH_STOPWORDS)[:: 10]))

# specifying stop_words = "english" uses the built-in list.
# we could also augment it and pass our own
vect = CountVectorizer(min_df = 5, stop_words="english").fit(text_train)
X_train = vect.transform(text_train)
print("X_train with stop words: \n{}".format(repr(X_train)))

gird = GridSearchCV(LogisticRegression(), param_grid, cv=5)
gird.fit(X_train, y_trian)
print("best cross-validation socre: {:.2f}".format(grid.best_score_))

pipe = make_pipeline(TfidfTransformer(min_df = 5, norm=None), LogisticRegression())
param_grid = {"logisticregression__C":[0.001, 0.01, 0.1, 1, 10]}
grid = GridSearchCV(pipe, param_grid, cv = 5)
grid.fit(text_train, y_train)
print("best cross-validation score: {:.2f}".format(grid.best_score_))

vectorizer = grid.best_estimator_.named_steps["tfidfvectorizer"]
# transform the training dataset
X_trian = vectorizer.transform(text_train)
# find maximum value for each of the features over the dataset
max_value = X_train.max(axis = 0).toarray().reval()
sorted_by_tfidf = max_value.argsort()
# get feature names
feature_names = np.array(vectorize.get_feature_names())
print("Features with lowest tfidf: \n{}".format(feature_names[sorted_by_tfidf[:20]]))
print("Features with highest tfidf: \n{}".format(features_names[sorted_by_tfidf[:-20]]))

sorted_by_idf = np.argsort(vectorizer.idf_)
print("feature with lowest idf: \n{}".format(feature_names[sorted_by_idf[: 100]]))

print("bards_words:\n{}".format(bards_words))
cv = CountVectorizer(ngram_range = (1, 1)).fit(bards_words)
print("Vocabulary size:{}".format(len(cv.vocabulary_)))
print("Vocabulary:\n {}".format(cv.get_feature_names()))

cv = CountVectorizer(ngram_range = (2, 2)).fit(bards_words)
print("Vocabulary size:{}".format(len(cv.vocabulary_)))
print("Vocabulary:\n {}".format(cv.get_feature_names()))

cv = CountVectorizer(ngram_range = (1, 3)).fit(bards_words)
print("Vocabulary size:{}".format(len(cv.vocabulary_)))
print("Vocabulary:\n {}".format(cv.get_feature_names()))

pipe = make_pipeline(TfidfTransformer(min_df = 5), LogisticRegression())
# running the grid search takes a long time because of the relatively large grid and the inclusion of trigrams
param_grid = {"logisticregression__C": [0.001, 0.01, 0.1, 1, 10, 100],
			  "tfidfvectorizer__ngram_range":[(1, 1), (1, 2), (1, 3)]}
grid = GridSearchCV(pipe, param_grid, cv = 5)
grid.fit(text_train, y_train)
print("best cross-validation score: {:.2f}".format(grid.best_score_))
print("best parameters: \n{}".format(grid.best_params_))

# extract scores from grid_search
scores = grid.cv_results_["mean_test_score"].reshape(-1, 3).T
# visualize heat map
heatmap = mglearn.tools.heatmap(
	scores, xlabel="C", ylabel="ngram_range", cmap="viridis", fmt="%.3f",
	xticklabels = param_grid["logisticregression__C"], yticklabels=param_grid["tfidfvectorizer__ngram_range"])
plt.colorbar(heatmap)

vect = CountVectorizer(max_features = 10000, max_df = .15)
X= vect.fit_transform(text_train)
lda = LatentDirichletAllocation(n_topics = 10, learning_method="batch", max_iter=25, random_state=0)
# we build the model and transform the data in one step
# computing transform takes some time, and we can save time by doing both at once
document_topics = lda.fit_transform(X)
lda.components_.shape

# For each topic(a row in the components_), sort the features (ascending)
# Invert rows with [:, ::-1] to make sorting descending
sorting = np.argsort(lda.components_, axis=1)[:, ::-1]
# Get the feature names from the vectorizer
feature_names = np.array(vectorize.get_feature_names())
#print out the 10 topics:
mglearn.tools.print_topics(topics = range(10), feature_names = feature_names, sorting =sorting, topics_per_chunk = 5, n_words = 10)

lda100 = LatentDirichletAllocation(n_topics = 100, learning_method="batch", max_iter=25, random_state=0)
document_topics100 = lda100.fit_transform(X)
topics = np.array([7, 16, 24, 25, 28, 36, 37, 45, 51, 53, 54, 63, 89, 97])
sorting = np.argsort(lda100.components_, axis=1)[:, ::-1]
feature_names = np.array(vect.get_feature_names())
mglearn.tools.print_topics(topics = topics, feature_names = feature_names, sorting = sorting, topics_per_chunk = 7, n_words =20)