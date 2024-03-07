import pandas as pd
import numpy as np
import time
from itertools import product
from vincenty import vincenty
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, median_absolute_error
from sklearn.metrics.pairwise import haversine_distances
from math import radians
from sklearn.preprocessing import LabelEncoder


dataset = pd.read_csv('sigfox_dataset_rural (1).csv')

X = dataset.iloc[:, :137]
y = dataset[["'Latitude'", "'Longitude'"]]
X = np.array(X)
y = np.array(y)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


k = 1
errors = []

for i in range(len(X_test)):
    all_distances = np.sqrt(np.sum((X_train - X_test[i]) ** 2, axis=1))

    k_indexes = np.argsort(all_distances)[0:k]
    centroids = np.mean(y_train[k_indexes, :], axis=0)
    error = vincenty(y_test[i], centroids)
    errors.append(error)

print(np.mean(errors)*1000)
print(np.median(errors)*1000)

