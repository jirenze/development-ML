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

#digits = load_digits()
#y = digits.target == 9

## most_frequent
#X_train, X_test, y_train, y_test = train_test_split(digits.data, y, random_state = 0)
#dummy_majority = DummyClassifier(strategy = "most_frequent").fit(X_train, y_train)
#pred_most_frequent = dummy_majority.predict(X_test)
##print("unique predicted labels: {}".format(np.unique(pred_most_frequent)))
#print("test score: {:.2f}".format(dummy_majority.score(X_test, y_test)))

## dummy
#from sklearn.linear_model import LogisticRegression
#dummy = DummyClassifier().fit(X_train, y_train)
#pred_dummy = dummy.predict(X_test)
##print("dummy predict: {}".format(pred_dummy))
#print("dummy score: {:.2f}".format(dummy.score(X_test, y_test)))

## decision tree
#from sklearn.tree import DecisionTreeClassifier
#tree = DecisionTreeClassifier(max_depth = 2).fit(X_train, y_train)
#pred_tree = tree.predict(X_test)
##print("tree predict: {}".format(pred_tree))
#print("pred_tree score: {:.2f}".format(tree.score(X_test, y_test)))

##logistic_regression
#logreg = LogisticRegression(C = 0.1).fit(X_train, y_train)
#pred_logreg = logreg.predict(X_test)
#print("logreg score: {:.2f}".format(logreg.score(X_test, y_test)))

#from sklearn.metrics import confusion_matrix
#confusion = confusion_matrix(y_test, pred_logreg)
##print("confusion matrix: \n{}".format(confusion))
##mglearn.plots.plot_confusion_matrix_illustration()
##mglearn.plots.plot_binary_confusion_matrix()

#print("Most frequent class:")
#print(confusion_matrix(y_test, pred_most_frequent))
#print("\n dummy model: ")
#print(confusion_matrix(y_test, pred_dummy))
#print("\n decision tree model: ")
#print(confusion_matrix(y_test, pred_tree))
#print("\n logistic regression: ")
#print(confusion_matrix(y_test, pred_logreg))

#from sklearn.metrics import f1_score
#print("f1 score most frequent: {:.2f}".format(f1_score(y_test, pred_most_frequent)))
#print("f1 socre dummy: {:.2f}".format(f1_score(y_test, pred_dummy)))
#print("f1 socre decision tree: {:.2f}".format(f1_score(y_test, pred_tree)))
#print("f1 score logistic regression: {:.2f}".format(f1_score(y_test, pred_logreg)))

#from sklearn.metrics import classification_report
#print(classification_report(y_test, pred_most_frequent, target_names = ["not nine", "nine"]))

#from mglearn.datasets import make_blobs
#X, y = make_blobs(n_samples = (400, 50), centers = 2, cluster_std = [7.0, 2], random_state = 22)
#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
#svc = SVC(gamma = 0.05).fit(X_train, y_train)

from mglearn.datasets import make_blobs
from sklearn.metrics import precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
# use more data points for a smoother curve
X, y = make_blobs(n_samples = (4000, 500), centers = 2, cluster_std = [7.0, 2], random_state = 22)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
svc = SVC(gamma = 0.5).fit(X_train, y_train)
rf = RandomForestClassifier(n_estimators = 100, random_state=0, max_features=2)
rf.fit(X_train, y_train)

#precision, recall, thresholds = precision_recall_curve(y_test, svc.decision_function(X_test))
## RandomForestClassifier has predict_proba, but not decision_function
#precision_rf, recall_rf, thresholds_rf = precision_recall_curve(y_test, rf.predict_proba(X_test)[:, 1])

## find threshold closest to zero
#plt.plot(precision, recall, label = "svc")
#close_zero = np.argmin(np.abs(thresholds))
#plt.plot(precision[close_zero], recall[close_zero], "o", markersize = 10, label = "threshold zero svc", fillstyle = "none", c = "k", mew = 2)

#plt.plot(precision_rf, recall_rf, label = "rf")
#close_default_rf = np.argmin(np.abs(thresholds_rf - 0.5))
#plt.plot(precision_rf[close_default_rf], recall_rf[close_default_rf], "^", c = "k", markersize = 10, label = "threshold 0.5 rf", fillstyle = "none", mew = 2)
#plt.xlabel("precsion")
#plt.ylabel("recall")
#plt.legend(loc = "best")

#print("f1_score of random forest: {:.3f}".format(f1_score(y_test, rf.predict(X_test))))
#print("f1_score of svc: {:.3f}".format(f1_score(y_test, svc.predict(X_test))))

#from sklearn.metrics import average_precision_score
#ap_rf = average_precision_score(y_test, rf.predict_proba(X_test)[:, 1])
#ap_svc = average_precision_score(y_test, svc.decision_function(X_test))
#print("average precision of random forest: {:.2f}".format(ap_rf))
#print("average precision of svc: {:.2f}".format(ap_svc))


from sklearn.metrics import roc_curve
fpr, tpr, roc_thresholds = roc_curve(y_test, svc.decision_function(X_test))
plt.plot(fpr, tpr, label = "Roc Curve")
plt.xlabel("FPR")
plt.ylabel("TPR(recall)")

# find thresholds closest to zero
roc_close_zero = np.argmin(np.abs(roc_thresholds))
plt.plot(fpr[roc_close_zero], tpr[roc_close_zero], "o", markersize = 10, label = "thresholds zero", fillstyle = "none", c = "k",mew = 2)
plt.legend(loc = 4)
plt.show()