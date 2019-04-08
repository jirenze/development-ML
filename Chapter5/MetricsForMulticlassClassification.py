# -*- coding: utf-8 -*-
"""
Created on Sat. April.06 2019

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

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, random_state = 0)
lr = LogisticRegression()
lr.fit(X_train, y_train)
pred = lr.predict(X_test)
#print("Accuracy: {:.3f}".format(accuracy_score(y_test, pred)))
#print("confusion matrix: \n{}".format(confusion_matrix(y_test, pred)))

#scores_image = mglearn.tools.heatmap(confusion_matrix(y_test, pred), xlabel="predicted label",
#									 ylabel = "True label", xticklabels= digits.target_names,
#									 yticklabels = digits.target_names, cmap=plt.cm.gray_r, fmt="%d")
#plt.title("confusion matrix")
#plt.gca().invert_yaxis()
#plt.show()
#print(classification_report(y_test, pred))
#print("micro average f1 score: {:.3f}".format(f1_score(y_test, pred, average = "micro")))
#print("macro average f1 score: {:.3f}".format(f1_score(y_test, pred, average = "macro")))

# default scoring for classification is accuracy
print("default scoring: {}".format(cross_val_score(SVC(), digits.data, digits.target == 9)))

# providing scoring = "accuracy" doesn't change the results
explicit_accuracy = cross_val_score(SVC(), digits.data, digits.target == 9, scoring="accuracy")
print("explicit accuracy scoring: {}".format(explicit_accuracy))

roc_auc = cross_val_score(SVC(), digits.data, digits.target == 9, scoring="roc_auc")
print("auc scoring:{}".format(roc_auc))

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target == 9, random_state = 0)
# we provide a somewhat bad grid to illustrate the point
param_grid = {"gamma": [0.0001, 0.01, 0.1, 1, 10]}
# using the default scoring of accuracy:
grid = GridSearchCV(SVC(), param_grid = param_grid)
grid.fit(X_train, y_train)
print("Grid search with accuracy:")
print("best parameters:", grid.best_params_)
print("best cross-validation score (accuracy): {:.3f}".format(grid.best_score_))
print("test set Auc: {:.3f}".format(roc_auc_score(y_test, grid.decision_function(X_test))))
print("test set accuracy: {:.3f}".format(grid.score(X_test, y_test)))
