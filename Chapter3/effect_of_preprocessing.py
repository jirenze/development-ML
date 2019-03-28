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

from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state = 0)
svm = SVC(C = 100)
svm.fit(X_train, y_train)
print("test set accuracy: {:.2f}".format(svm.score(X_test, y_test)))

# preprocessing using 0-1 scaling
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# learning an SVM on the scaled training data
svm.fit(X_train_scaled, y_train)

# scoring on the scaled test set
print("Scaled test set accuracy : {:.2f}".format(svm.score(X_test_scaled, y_test)))

# preprocessing using zero mean and unit variance scaling
from sklearn.preprocessing import StandardScaler
scaler_std = StandardScaler()
scaler_std.fit(X_train)
X_train_scaled_std = scaler_std.transform(X_train)
X_test_scaled_std = scaler_std.transform(X_test)

# learning an SVM on the scaled training set
svm_std = SVC()
svm_std.fit(X_train_scaled_std, y_train)

# scoring on the scaled test set
print("SVM test accuracy : {:.2f}".format(svm_std.score(X_test_scaled_std, y_test)))
