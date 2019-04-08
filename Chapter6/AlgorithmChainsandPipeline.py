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

from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_boston
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier



# load and split the data
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state = 0)

## compute minimum and maximum on the training data
#X_train_scaler = MinMaxScaler().fit(X_train)

## rescale the training data
#X_train_scaled = X_train_scaler.transform(X_train)
#svm = SVC()
## learn an SVM on the scaled training data
#svm.fit(X_train_scaled, y_train)
## scale the test data and score the scaled data
#X_test_scaled = X_train_scaler.transform(X_test)
#print("test score: {:.2f}".format(svm.score(X_test_scaled, y_test)))

## for illustration purposes only, don't use this code
#para_grid = {"C": [0.001, 0.01, 0.1, 1, 10 ,100],
#			 "gamma": [0.001, 0.01, 0.1, 1, 10 ,100]}
#grid = GridSearchCV(SVC(), param_grid = para_grid, cv=5)
#grid.fit(X_train_scaled, y_train)
#print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))
#print("best set score: {:.2f}".format(grid.score(X_test_scaled, y_test)))
#print("best parameters: ", grid.best_params_)


#pipe = Pipeline([("scaler", MinMaxScaler()), ("svm", SVC())])
#pipe.fit(X_train, y_train)
#print("Test score: {:.2f}".format(pipe.score(X_test, y_test)))

#param_grid = {"svm__C":[0.001, 0.01, 0.1, 1, 10 ,100],
#			 "svm__gamma": [0.001, 0.01, 0.1, 1, 10 ,100]}
#grid = GridSearchCV(pipe, param_grid = param_grid, cv=5)
#grid.fit(X_train, y_train)
#print("best cross-validation accuracy: {:.2f}".format(grid.best_score_))
#print("test set score: {:.2f}".format(grid.score(X_test, y_test)))
#print("best parameters: {}".format(grid.best_params_))


#def pipeline_fit(self, x, y):
#	X_transformed = X
#	for name, estimator in self.steps[: -1]:
#		# iterate over all but the final step
#		# fit and transform the data
#		X_transformed = estimator.fit_transform(X_transformed, y)
#	# fit the last step
#	self.steps[-1][1].fit(X_transformed, y)
#	return self
#def predict(self, X):
#	X_transformed = X
#	for step in self.steps[: -1]:
#		# iterate over all but the final step
#		# transform the data
#		X_transformed = step[1].transform(X_transformed)
#	# fit the last step
#	return self.steps[-1][1].predict(X_transformed)


## standard syntax
#pipe_long = Pipeline([("scaler", MinMaxScaler()), ("svm", SVC(C = 100))])
## abbreviated syntax
#pipe_short = make_pipeline(MinMaxScaler(), SVC(C = 100))
#print("Pipeline steps: \n {}".format(pipe_short.steps))


#pipe = make_pipeline(StandardScaler(), PCA(n_components = 2), StandardScaler())

## fit the pipeline defined before to the cancer dataset
#pipe.fit(cancer.data)
## extract the first two principal components from the "pca" step
#components = pipe.named_steps["pca"].components_
#print("\n components.shape {}".format(components.shape))

#pipe = make_pipeline(StandardScaler(), LogisticRegression())
#param_grid = {"logisticregression__C": [0.01, 0.1, 1, 10, 100]}
#X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state = 4)
#grid = GridSearchCV(pipe, param_grid, cv = 5)
#grid.fit(X_train, y_train)
#print("best estimator: \n{}".format(grid.best_estimator_))
#print("logistic regression step: \n{}".format(grid.best_estimator_.named_steps["logisticregression"]))
#print("logistic regression coefficients: \n{}".format(grid.best_estimator_.named_steps["logisticregression"].coef_))

#boston = load_boston()
#X_train, X_test, y_train , y_test = train_test_split(boston.data, boston.target, random_state = 0)
#pipe = make_pipeline(StandardScaler(), PolynomialFeatures(), Ridge())

#param_grid = {"polynomialfeatures__degree": [1, 2, 3],
#			  "ridge__alpha": [0.001, 0.01, 0.1, 1, 10, 100]}
#grid = GridSearchCV(pipe, param_grid = param_grid, cv=5, n_jobs= -1)
#grid.fit(X_train, y_train)
#print("best parameters: {}".format(grid.best_params_))
#plt.matshow(grid.cv_results_["mean_test_score"].reshape(3, -1), vmin = 0, cmap = "viridis")
#plt.xlabel("ridge__alpha")
#plt.ylabel("polynomialfeatures__degree")
#plt.xticks(range(len(param_grid["ridge__alpha"])), param_grid["ridge__alpha"])
#plt.yticks(range(len(param_grid["polynomialfeatures__degree"])), param_grid["polynomialfeatures__degree"])
#plt.colorbar()
#plt.show()
#print("test-set score: {:.2f}".format(grid.score(X_test, y_test)))

#param_grid = {"ridge__alpha": [0.001, 0.01, 0.1, 1, 10, 100]}
#pipe = make_pipeline(StandardScaler(), Ridge())
#grid = GridSearchCV(pipe, param_grid, cv = 5)
#grid.fit(X_train, y_train)
#print("score without poly features: {:.2f}".format(grid.score(X_test, y_test)))

pipe = Pipeline([("preprocessing", StandardScaler()), ("classifier", SVC())])
param_grid = [
	{"classifier": [SVC()], "preprocessing": [StandardScaler(), None],
	 "classifier__gamma": [0.001, 0.01, 0.1, 1, 10, 100],
	 "classifier__C": [0.001, 0.01, 0.1, 1, 10, 100]},
    {"classifier": [RandomForestClassifier(n_estimators = 100)],
	 "preprocessing": [None], "classifier__max_features": [1, 2, 3]}]
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state = 0)
grid = GridSearchCV(pipe, param_grid= param_grid, cv = 5)
grid.fit(X_train, y_train)
print("best params : \n{}\n".format(grid.best_params_))
print("best cross-validation score: {:.2f}".format(grid.best_score_))
print("test set score: {:.2f}".format(grid.score(X_test, y_test)))