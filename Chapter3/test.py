# -*- coding: utf-8 -*-
"""
Created on Thur Mar.20 2019

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
# =============================================================================
# print("numpy version: {}".format(np.__version__))
# print("mglearn version: {}".format(mglearn.__version__))
# print("pandas version: {}".format(pd.__version__))
# print("sp version: {}".format(sp.__version__))
# print("sklearn version: {}".format(sklearn.__version__))
# =============================================================================
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
# =============================================================================
# X, y = mglearn.datasets.make_forge()
# fig, axes = plt.subplots(1, 2, figsize = (10, 3))
# 
# for model ,ax in zip([LinearSVC(), LogisticRegression()],axes):
#     clf = model.fit(X, y)
#     mglearn.plots.plot_2d_separator(clf, X, fill = False, eps = 0.5, ax = ax, alpha = .7)
#     mglearn.discrete_scatter(X[:, 0], X[:, 1],y ,ax = ax)
#     ax.set_title("{}".format(clf.__class__.__name__))
#     ax.set_xlabel("Feature 0")
#     ax.set_ylabel("Festure 1")
# axes[0].legend()
# =============================================================================
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify = cancer.target, random_state = 42)
# C = 1
# =============================================================================
# logreg = LogisticRegression()
# logreg.fit(X_train, y_train)
# print("Training set score: {:.3f}".format(logreg.score(X_train, y_train)))
# print("Testing set score: {:.3f}".format(logreg.score(X_test, y_test)))
# print("\n")
# 
# # C = 100
# logreg100 = LogisticRegression(C = 100)
# logreg100.fit(X_train, y_train)
# print("Training set score: {:.3f}".format(logreg100.score(X_train, y_train)))
# print("Testing set score: {:.3f}".format(logreg100.score(X_test, y_test)))
# print("\n")
# 
# # C = 0.01
# logreg001 = LogisticRegression(C = 0.01)
# logreg001.fit(X_train, y_train)
# print("Training set score: {:.3f}".format(logreg001.score(X_train, y_train)))
# print("Testing set score: {:.3f}".format(logreg001.score(X_test, y_test)))
# print("\n")
# 
# plt.plot(logreg.coef_.T, "o", label = "C = 1")
# plt.plot(logreg100.coef_.T, "^", label = "C = 100")
# plt.plot(logreg001.coef_.T, "v", label = "C = 0.01")
# plt.xticks(range(cancer.data.shape[1]),cancer.feature_names, rotation = 90)
# plt.hlines(0, 0, cancer.data.shape[1])
# plt.ylim(-5, 5)
# plt.xlabel("Coefficient index")
# plt.ylabel("Coefficient magnitude")
# plt.legend()
# 
# =============================================================================
print("\n")
#L1
for in_c, in_marker in zip([0.001, 1, 100], ["o", "^", "v"]):
   lr_l1 = LogisticRegression(C = in_c, penalty = "l1") 
   lr_l1.fit(X_train, y_train)
   
   print("Training accuracy of l1 logreg with c = {:.3f}: {:.2f}".format(
           in_c, lr_l1.score(X_train, y_train)))
   print("Testing accuracy of l1 logreg with c = {:.3f}: {:.2f}".format(
           in_c, lr_l1.score(X_test, y_test)))
   plt.plot(lr_l1.coef_.T, in_marker,label = "C = {:.3f}".format(in_c))
   print("\n")
   
plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation = 90)
plt.hlines(0, 0, cancer.data.shape[1])
plt.xlabel("Coefficient index")
plt.ylabel("Coefficietn magnitude")

plt.ylim(-5, 5)
plt.legend(loc =3)



















































































































