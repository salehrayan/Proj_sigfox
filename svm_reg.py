import numpy as np
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import haversine_distances
import time
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, median_absolute_error
from sklearn.metrics.pairwise import haversine_distances
from math import radians
from sklearn.preprocessing import LabelEncoder

X_train_distance = np.load('X_train_distance.npz')
y_train_distance = np.load('y_train_distance.npz')
X_test_distance = np.load('X_test_distance.npz')
y_test_distance = np.load('y_test_distance.npz')

X_train_distance = X_train_distance['data']
y_train_distance = y_train_distance['data']
X_test_distance = X_test_distance['data']
y_test_distance = y_test_distance['data']

k = 18

X_train_distance = X_train_distance[:, 0:k, :]
X_test_distance = X_test_distance[:, 0:k, :]

X_train_distance = np.reshape(X_train_distance, (17945, k*2))
X_test_distance = np.reshape(X_test_distance, (7692, k*2))

degree = 4 # You can adjust the degree of the polynomial
poly_features = PolynomialFeatures(degree=degree)
X_train_poly = poly_features.fit_transform(X_train_distance)
X_test_poly = poly_features.transform(X_test_distance)

# Initialize separate regressors for each target dimension
poly_regressor_0 = LinearRegression()
poly_regressor_1 = LinearRegression()

poly_regressor_0.fit(X_train_poly, y_train_distance[:, 0])
poly_regressor_1.fit(X_train_poly, y_train_distance[:, 1])

y_pred_0 = poly_regressor_0.predict(X_test_poly).reshape(-1, 1)
y_pred_1 = poly_regressor_1.predict(X_test_poly).reshape(-1, 1)
y_pred = np.concatenate((y_pred_0, y_pred_1), axis=1)

errors = []
for i in range(len(y_pred)):
    centroids = y_pred[i]
    error = haversine_distances(np.reshape(np.radians(y_test_distance[i]), (1, -1)), np.reshape(np.radians(centroids), (1, -1))) * 6371000
    errors.append(error)

# Print mean and median errors
print(f"Mean Error: {np.mean(errors)} meters")
print(f"Median Error: {np.median(errors)} meters")