import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import haversine_distances

X_train_distance = np.load('X_train_distance.npz')
y_train_distance = np.load('y_train_distance.npz')
X_test_distance = np.load('X_test_distance.npz')
y_test_distance = np.load('y_test_distance.npz')

X_train_distance = X_train_distance['data']
y_train_distance = y_train_distance['data']
X_test_distance = X_test_distance['data']
y_test_distance = y_test_distance['data']

y2 = np.load('y_pred.npz')
y1 = np.load('y_pred_svm.npz')
y1 = y1['data']
y2 = y2['data']

y_pred = (y1 + y2) / 2
# y_pred = y1
# y_pred = y2


errors = []
for i in range(len(y_pred)):
    centroids = y_pred[i]
    error = haversine_distances(np.reshape(np.radians(y_test_distance[i]), (1, -1)), np.reshape(np.radians(centroids), (1, -1))) * 6371000
    errors.append(error)

# Print mean and median errors
print(f"Mean Error: {np.mean(errors)} meters")
print(f"Median Error: {np.median(errors)} meters")
