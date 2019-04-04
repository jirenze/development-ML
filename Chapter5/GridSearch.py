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
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
iris = load_iris()
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
X_trainval, X_test, y_trainval, y_test = train_test_split(iris.data, iris.target, random_state = 0)
# split train+validation set into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X_trainval, y_trainval, random_state = 1)
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


for in_gamma_s in [0.001, 0.01, 0.1, 1, 10, 100]:
	for in_C_s in [0.001, 0.01, 0.1, 1, 10, 100]:
		# for each combination of parameters,
		# train an SVC
		svm = SVC(gamma = in_gamma_s, C = in_C_s)
		# perform cross validation
		scores = cross_val_score(svm ,X_trainval, y_trainValueError, cv = 5)
		# compute mean cross-validation accuracy
		score =np.mean(scores)
		# if we got a better score, store the score and parameters
		if score > best_score:
			best_score = score
			best_parameters_c = in_C_s
			best_parameters_gamma = in_gamma_s
# rebuild a model on the combined training and validation set
svm = SVC(gamma = in_gamma_s, C = in_C_s)
svm.fit(X_trainval, y_trainval)