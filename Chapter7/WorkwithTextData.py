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

from sklearn.datasets import load_files

reviews_train = load_files("E:/development/aclImdb/train/")
# load_files returns a bunch, containing training texts and training labels
text_train, y_train = reviews_train.data, reviews_train.target
print("type of text_train: {}".format(type(text_train)))
print("length of text_train: {}".format(len(text_train)))
print("text_train[1]: \n{}".format(text_train[1]))
text_train = [doc.replace(b"<br />", b" ") for doc in text_train]
print("samples per class(training): {}".format(np.bincount(y_train)))

reviews_test = load_files("E:/development/aclImdb/test/")
text_test, y_test = reviews_test.data, reviews_test.target
print("number of documents in test data: {}".format(len(text_test)))
print("sample per class(test): {}".format(np.bincount(y_test)))
text_test = [doc.replace(b"<br />", b" ") for doc in text_test]