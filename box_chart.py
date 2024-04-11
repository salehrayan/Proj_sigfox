import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from tqdm import tqdm
from vincenty import vincenty
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

dataset = pd.read_csv('sigfox_dataset_rural (1).csv')
X = dataset.iloc[:, :137]
y = dataset[["'Latitude'", "'Longitude'"]]
X = np.array(X)
y = np.array(y)

X_norm = (X - np.min(X))/np.min(X)*-1
X_exp = np.exp((X - np.min(X))/24)/np.exp(np.min(X)*-1/24)
X_pow = X_norm ** np.e

pca = PCA(n_components=0.95)
X_norm_PCA = pca.fit_transform(X_norm)
# X_exp = pca.fit_transform(X_exp)
# X_pow = pca.fit_transform(X_pow)


X_train_norm, X_test_norm, y_train, y_test = train_test_split(X_norm, y, test_size=0.30, random_state=42)
X_train_norm_PCA, X_test_norm_PCA, _, _ = train_test_split(X_norm_PCA, y, test_size=0.30, random_state=42)
X_train_exp, X_test_exp, _, _ = train_test_split(X_exp, y, test_size=0.30, random_state=42)
X_train_pow, X_test_pow, _, _ = train_test_split(X_pow, y, test_size=0.30, random_state=42)
X_train, X_test, _, _ = train_test_split(X, y, test_size=0.30, random_state=42)

X_train_norm, X_test_norm, y_train, y_test = train_test_split(X_norm, y, test_size=0.30, random_state=42)

# Initialize regressors
poly = PolynomialFeatures(degree=2)
knn_regressor = KNeighborsRegressor(n_neighbors=1)
ols_regressor = LinearRegression()
ridge_regressor = Ridge(alpha=0.17012542798525893)
lasso_regressor = Lasso(alpha=0.0005484416576121015)
elastic_regressor = ElasticNet(alpha=0.0005214008287999684, l1_ratio=1.0)
poly_regressor = LinearRegression()
rf_regressor = RandomForestRegressor(n_estimators=100)

# Fit the regressors
knn_regressor.fit(X_train_norm, y_train)
ols_regressor.fit(X_train_norm, y_train)
ridge_regressor.fit(X_train_norm, y_train)
lasso_regressor.fit(X_train_norm, y_train)
elastic_regressor.fit(X_train_norm, y_train)
poly_regressor.fit(poly.fit_transform(X_train_norm_PCA), y_train)
rf_regressor.fit(X_train_norm, y_train)

# Predictions
knn_pred = knn_regressor.predict(X_test_norm)
ols_pred = ols_regressor.predict(X_test_norm)
ridge_pred = ridge_regressor.predict(X_test_norm)
lasso_pred = lasso_regressor.predict(X_test_norm)
elastic_pred = elastic_regressor.predict(X_test_norm)
poly_pred = poly_regressor.predict(poly.fit_transform(X_test_norm_PCA))
rf_pred = rf_regressor.predict(X_test_norm)

# Calculate errors
error_accu = []
errors = np.ones((len(knn_pred), 0))
for i in range(len(knn_pred)):
    centroids = knn_pred[i]
    error = vincenty(centroids, y_test[i])
    error_accu.append(error)

errors = np.concatenate((errors, np.reshape(error_accu, (-1, 1))), axis=1)
error_accu = []
for i in range(len(knn_pred)):
    centroids = ols_pred[i]
    error = vincenty(centroids, y_test[i])
    error_accu.append(error)
errors = np.concatenate((errors, np.reshape(error_accu, (-1, 1))), axis=1)
error_accu = []
for i in range(len(knn_pred)):
    centroids = ridge_pred[i]
    error = vincenty(centroids, y_test[i])
    error_accu.append(error)
errors = np.concatenate((errors, np.reshape(error_accu, (-1, 1))), axis=1)
error_accu = []
for i in range(len(knn_pred)):
    centroids = lasso_pred[i]
    error = vincenty(centroids, y_test[i])
    error_accu.append(error)
errors = np.concatenate((errors, np.reshape(error_accu, (-1, 1))), axis=1)
error_accu = []
for i in range(len(knn_pred)):
    centroids = elastic_pred[i]
    error = vincenty(centroids, y_test[i])
    error_accu.append(error)

errors = np.concatenate((errors, np.reshape(error_accu, (-1, 1))), axis=1)
error_accu = []
for i in range(len(knn_pred)):
    centroids = poly_pred[i]
    error = vincenty(centroids, y_test[i])
    error_accu.append(error)

errors = np.concatenate((errors, np.reshape(error_accu, (-1, 1))), axis=1)
error_accu = []
for i in range(len(knn_pred)):
    centroids = rf_pred[i]
    error = vincenty(centroids, y_test[i])
    error_accu.append(error)

errors = np.concatenate((errors, np.reshape(error_accu, (-1, 1))), axis=1)
error_accu = []


# Create a box plot of errors
print(f'{np.mean(errors, axis=0)}')
plt.figure(figsize=(10, 6))
# plt.subplots_adjust(bottom=0.2)
ax = plt.gca()
plt.boxplot(errors * 1000, whis=(0, 100))
plt.title("Box Plot of Errors")
plt.xlabel("Machine Learning Regressor")
plt.ylabel("Error (Vincenty distance)")
plt.yscale('log')
x_ticks = [1, 2, 3, 4, 5, 6, 7]
x_labels = ["KNN", "OLS", "Ridge", "Lasso", "Elastic", "Polynomial", "Random Forest"]
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_labels)
# plt.tight_layout()
plt.show()