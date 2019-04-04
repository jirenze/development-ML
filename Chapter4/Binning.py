# -*- coding: utf-8 -*-
"""
Created on Thes April.02 2019

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

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

X, y = mglearn.datasets.make_wave(n_samples = 100)
line = np.linspace(-3, 3, 1000, endpoint= False).reshape(-1, 1)

#reg = DecisionTreeRegressor(min_samples_split = 3).fit(X, y)
#plt.plot(line, reg.predict(line), label = "decision tree")
#lreg = LinearRegression().fit(X, y)
#plt.plot(line, lreg.predict(line), label = "linear regressoion")

bins = np.linspace(-3, 3, 11)
which_bin = np.digitize(X, bins = bins)

from sklearn.preprocessing import OneHotEncoder
# transform using the OneHotEncoder
encoder = OneHotEncoder(sparse = False)
# encoder.fit finds the unique values that appear in which_bin
encoder.fit(which_bin)
# transform creates the one-hot encoding
X_binned = encoder.transform(which_bin)
print(X_binned[:5])

line_binned = encoder.transform(np.digitize(line, bins = bins))

reg = LinearRegression().fit(X_binned, y)
plt.plot(line, reg.predict(line_binned), label = "linear regression binned")

reg = DecisionTreeRegressor(min_samples_split = 3).fit(X_binned, y)
plt.plot(line, reg.predict(line_binned), label = "decision tree binned")
plt.plot(X[:, 0], y, "o", c = "k")                                                                                                                                                                                                                                                      
plt.vlines(bins, -3, 3, linewidth = 1, alpha = 0.2)
plt.legend(loc = "best")
plt.xlabel("Input feature")
plt.ylabel("regression output")
plt.show()