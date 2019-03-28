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

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_blobs

#cancer = load_breast_cancer()
#X_train, X_test, y_train, y_test = train_test_split(cancer.data,
#cancer.target, random_state = 1)
#scaler.fit(X_train)

##transform data
#X_train_scaled = scaler.transform(X_train)
##print dataset properties before and after scaling
##print("transformed shape: {}".format(X_train_scaled.shape))
##print("per-feature minimum before scaling: \n {}".format(X_train.min(axis =
##0)))
##print("per-feature maximum before scaling: \n {}".format(X_train.max(axis =
##0)))
##print("per-feature minimum after scaling: \n
##{}".format(X_train_scaled.min(axis = 0)))
##print("per-feature maximum after sacling: \n
##{}".format(X_train_scaled.max(axis = 0)))

##transform test data
#X_test_scaled = scaler.transform(X_test)
##print test data properties after scaling
#print("per-feature minimum after scaling: \n {}".format(X_test_scaled.min(axis
#= 0)))
#print("per-feature maximum after sacling: \n {}".format(X_test_scaled.max(axis
#= 0)))

# make synthetic data
X, _ = make_blobs(n_samples = 50, centers = 5, random_state= 4, cluster_std= 2)
# split it into training and test sets
X_train, X_test = train_test_split(X, random_state = 5, test_size = .1)

# plot the training and test sets
fig, axes = plt.subplots(1, 3, figsize = (13, 4))

axes[0].scatter(X_train[:, 0], X_train[:, 1], c = "b", label = "Training set", s = 60)
axes[0].scatter(X_test[:, 0], X_test[:, 1], marker = "^", c = "r", label = "Test set", s = 60)
axes[0].legend(loc= "upper left")
axes[0].set_title("original data")

# scale the data using MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# visualize the properly scaled data
axes[1].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c = "b", label = "Training set", s = 60)
axes[1].scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], marker = "^", c = "r", label = "Testing set", s = 60)
axes[1].set_title("Scaled data")

# rescale the test set separately
# so test set min is 0 and test set max is 1
# DO NOT DO THIS! for illustration purposes only
test_scaler = MinMaxScaler()
test_scaler.fit(X_test)
X_test_scaled_badly = test_scaler.transform(X_test)

# visualize wrongly scaled data
axes[2].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c = "b", label = "training set", s = 60)
axes[2].scatter(X_test_scaled_badly[:, 0], X_test_scaled_badly[:, 1], marker = "^", c = "r", label = "testing set", s= 60)
axes[2].set_title("Improperly Scaled Data")

for ax in axes:
	ax.set_xlabel = "feature 0"
	ax.set_ylabel = "feature 1"

plt.show()