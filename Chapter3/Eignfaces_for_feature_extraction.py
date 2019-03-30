# -*- coding: utf-8 -*-
"""
Created on Fri Mar.29 2019

@author: jirenze
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
import sklearn

from sklearn.datasets import fetch_lfw_people
people = fetch_lfw_people(min_faces_per_person = 20, resize = 0.7)
image_shape = people.images[0].shape

fig, axes = plt.subplots(2, 5, figsize = (15, 8),
						subplot_kw = {"xticks": (), "yticks": ()})
for target, image, ax in zip(people.target, people.images, axes.ravel()):
	ax.imshow(image)
	ax.set_title(people.target_names[target])

print("people.image.shape: {}".format(people.images.shape))
print("Number of classes: {}".format(len(people.target_names)))

# count how often each target appears
counts = np.bincount(people.target) 
# print counts next to target names
for i, (count, name) in enumerate(zip(counts, people.target_names)):
	print("{0:25} {1:3}".format(name, count), end = "  ")
	if (i + 1) % 2 == 0:
		print()


#mask = np.zeros(people.target.shape, dtype = np.bool)
#for target in np.unique(people.target):
#	mask[np.where(people.target == target)[0][:50]] = 1

#X_people = people.data[mask]
#X_people = people.target[mask]

## scale the grayscale values to be between 0 and 1
## instead of 0 and 255 for better numeric stability
#X_people = X_people / 255.

plt.show()