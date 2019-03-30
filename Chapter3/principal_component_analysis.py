# -*- coding: utf-8 -*-
"""
Created on Thur Mar.28 2019

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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

#fig, axes = plt.subplots(15, 2 , figsize = (8, 16))
#malignant = cancer.data[cancer.target == 0]
#benign = cancer.data[cancer.target == 1]

#ax = axes.ravel()

#for i in range(30):
#	_, out_bins = np.histogram(cancer.data[:, i], bins = 50)
#	ax[i].hist(malignant[:, i], bins = out_bins, color = "r", alpha = 0.5)
#	ax[i].hist(benign[:, i],bins = out_bins, color = "g", alpha = 0.5)
#	ax[i].set_title(cancer.feature_names[i])
#	ax[i].set_yticks(())
#ax[0].set_xlabel("feature magnitude")
#ax[0].set_ylabel("frequency")
#ax[0].legend(["malignant", "bengin"], loc = "best")
#fig.tight_layout()


scaler_std = StandardScaler()
X_scaled_std = scaler_std.fit_transform(cancer.data)

# keep the first two principal components of the data
pca = PCA(n_components = 2)
# fit PAC model to breast cancer data
# transform data onto the first two principal components
X_pca = pca.fit_transform(X_scaled_std)
print("original shape: {}".format(str(X_scaled_std.shape)))
print("reduced shape: {}".format(str(X_pca.shape)))

# plot first vs. second prinicipal component, colored by class
#plt.figure(figsize = (8, 8))
#mglearn.discrete_scatter(X_pca[:, 0], X_pca[:, 1], cancer.target)
#plt.legend(cancer.target_names, loc = "best")
#plt.gca().set_aspect("equal")
#plt.xlabel("first principal component")
#plt.ylabel("second principal component")


print("PCA component shape: {}".format(pca.components_.shape))
print("PCA component {}".format(pca.components_))
plt.matshow(pca.components_, cmap = "viridis")
plt.yticks([0, 1], ["first component", "second component"])
plt.colorbar()
plt.xticks(range(len(cancer.feature_names)), cancer.feature_names, rotation = 60, ha = "left")
plt.xlabel("feature")
plt.ylabel("principal components")
plt.show()