# -*- coding: utf-8 -*-
"""
Created on Wed April.20 2019

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

# native grid search implementation
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
iris = load_iris()
from IPython import display
#X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state = 0)
#best_score = 0
#for in_gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
#	for in_C in [0.001, 0.01, 0.1, 1, 10, 100]:
#		# for each combination of parameters, train an SVC
#		svm = SVC(gamma = in_gamma, C = in_C)
#		svm.fit(X_train ,y_train)
#		# evaluate the SVC on the test set
#		score = svm.score(X_test, y_test)
#		# if we got a better score, store the score and parameters
#		if score >best_score:
#			best_score = score
#			best_parameters = {"c":in_C, "gamma": in_gamma}
#print("best score: {:.2f}".format(best_score))
#print("best parameters: {}".format(best_parameters))





# split data into train+validation set and test set
#X_trainval, X_test, y_trainval, y_test = train_test_split(iris.data, iris.target, random_state = 0)
# split train+validation set into training and validation sets
#X_train, X_valid, y_train, y_valid = train_test_split(X_trainval, y_trainval, random_state = 1)
#print("size of training set: {} size of validation set: {}  szie of test set:" "{}\n".format(X_train.shape[0], X_valid.shape[0], X_test.shape[0]))
#best_score = 0
#for in_gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
#	for in_C in [0.001, 0.01, 0.1, 1, 10, 100]:
#		# for each combination of parameters, train an SVC
#		svm = SVC(gamma = in_gamma, C = in_C)
#		svm.fit(X_train, y_train)
#		# evaluate the SVC on the test set
#		score = svm.score(X_valid, y_valid)
#		# if we got a better score, store the score and parameters
#		if score > best_score:
#			best_score = score
#			best_parameters_c =in_C
#			best_parameters_gamma = in_gamma
## rebuild a model on the combined training and validation set,
## and evaluate it on the test set
#svm = SVC(gamma = best_parameters_gamma,C = best_parameters_c)
#svm.fit(X_trainval, y_trainval)
#test_score =  svm.score(X_test, y_test)
#print("best score on validation set: {:.2f}".format(best_score))
#print("best parameters c: ", best_parameters_c)
#print("best parameters gamma: ", best_parameters_gamma)
#print("test set score with beset parameters : {:.2f}".format(test_score))


#for in_gamma_s in [0.001, 0.01, 0.1, 1, 10, 100]:
#	for in_C_s in [0.001, 0.01, 0.1, 1, 10, 100]:
#		# for each combination of parameters,
#		# train an SVC
#		svm = SVC(gamma = in_gamma_s, C = in_C_s)
#		# perform cross validation
#		scores = cross_val_score(svm ,X_trainval, y_trainval, cv = 5)
#		# compute mean cross-validation accuracy
#		score =np.mean(scores)
#		# if we got a better score, store the score and parameters
#		if score > best_score:
#			best_score = score
#			best_parameters_c = in_C_s
#			best_parameters_gamma = in_gamma_s
## rebuild a model on the combined training and validation set
#svm = SVC(gamma = in_gamma_s, C = in_C_s)
#svm.fit(X_trainval, y_trainval)



#param_grid = {"C": [0.001, 0.01, 0.1, 1, 10, 100],"gamma": [0.001, 0.01, 0.1, 1, 10, 100]}

#grid_search = GridSearchCV(SVC(), param_grid, cv = 5)
#X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state = 0)
#grid_search.fit(X_train, y_train)
#print("Test set score: {:.2f}".format(grid_search.score(X_test, y_test)))
#print("best parameters: {}".format(grid_search.best_params_))
#print("best cross_validation score:{:.2f}".format(grid_search.best_score_))
#print("best estimator: \n{}".format(grid_search.best_estimator_))

# convert to DataFrame
#results = pd.DataFrame(grid_search.cv_results_)
# show the first 5 rows
#scores = np.array(results.mean_test_score).reshape(6, 6)

# plot the mean cross-validation scores
#mglearn.tools.heatmap(scores, xlabel = "gamma", xticklabels = param_grid["gamma"], ylabel = "C", yticklabels = param_grid["C"], cmap = "viridis")


#fig, axes = plt.subplots(1, 3, figsize = (13, 5))
#param_grid_linear = {"C": np.linspace(1, 2, 6), "gamma": np.linspace(1, 2, 6)}
#param_grid_one_log = {"C": np.linspace(1, 2, 6), "gamma": np.logspace(-3, 2, 6)}
#param_grid_range = {"C": np.logspace(-3, 2, 6), "gamma": np.logspace(-7, -2, 6)}

#for param_grid, ax in zip([param_grid_linear, param_grid_one_log, param_grid_range], axes):
#	grid_search = GridSearchCV(SVC(), param_grid, cv = 5)
#	grid_search.fit(X_train, y_train)
#	scores = grid_search.cv_results_["mean_test_score"].reshape(6, 6)

#	# plot the mean cross-validation scores
#	scores_image = mglearn.tools.heatmap(
#		scores, xlabel="gamma", ylabel="C", xticklabels= param_grid["gamma"], yticklabels= param_grid["C"], cmap= "viridis", ax = ax)
#plt.show()

#X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state = 0)
#param_grid = [{"kernel":["rbf"],
#			  "C":[0.001, 0.01, 0.1, 1, 10, 100],
#			  "gamma": [0.001, 0.01, 0.1, 1, 10, 100]},
#			  {"kernel":["linear"],
#				"C":[0.001, 0.01, 0.1, 1, 10, 100]}]
##print("list of grids: \n{}".format(param_grid))
#grid_search = GridSearchCV(SVC(), param_grid, cv=5)
#grid_search.fit(X_train, y_train)
#print("best parameters:{}".format(grid_search.best_params_))
#print("best cross-validation score: {:.2f}".format(grid_search.best_score_))

#results = pd.DataFrame(grid_search.cv_results_)
#display.display(results.T)

param_grid = {"C": [0.001, 0.01, 0.1, 1, 10, 100],"gamma": [0.001, 0.01, 0.1, 1, 10, 100]}
#scores = cross_val_score(GridSearchCV(SVC(), param_grid, cv = 5), iris.data, iris.target, cv=5)
#print("cross-validation scores: ", scores)
#print("mean cross-validation score", scores.mean())

def nested_cv(X, y, inner_cv, outer_cv, Classifier, parameter_grid):
	outer_scores = []
	# for each split of the data in the outer cross-validataion
	# (split method returns indices)
	for training_samples, test_samples in outer_cv.split(X, y):
		# find best parameter using inner cross-validation
		best_parms = {}
		best_score = -np.inf
		# iterate over parameters
		for parameters in parameter_grid:
			# accumulate score over inner splits
			cv_scores = []
			# iterate over inner cross- validation
			for inner_train, inner_test in inner_cv.split(X[training_samples], y[training_samples]):
				# bulid classifier given parameters and training data
				clf = Classifier(**parameters)
				clf.fit(X[inner_train], y[inner_train])
				# evaluate on inner test set
				score = clf.score(X[inner_test], y[inner_test])
				cv_scores.append(score)
			# compute mean score over inner folds
			mean_score = np.mean(cv_scores)
			if mean_score > best_score:
				# if better than so far, remember parameters
				best_score = mean_score
				best_parms = parameters
		# build classifier on best parameters using outer training set
		clf = Classifier(**best_parms)
		clf.fit(X[training_samples], y[training_samples])
		# evaluate
		outer_scores.append(clf.score(X[test_samples], y[test_samples]))
	return np.array(outer_scores)

from sklearn.model_selection import ParameterGrid, StratifiedKFold

print("\n")
scores = nested_cv(iris.data, iris.target, StratifiedKFold(5), StratifiedKFold(5), SVC, ParameterGrid(param_grid))

print("cross-validation scores: {}".format(scores))