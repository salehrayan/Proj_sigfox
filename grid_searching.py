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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

dataset = pd.read_csv('sigfox_dataset_rural (1).csv')
X = dataset.iloc[:, :137]
y = dataset[["'Latitude'", "'Longitude'"]]
X = np.array(X)
y = np.array(y)

X_norm = (X - np.min(X))/np.min(X)*-1
X_exp = np.exp((X - np.min(X))/24)/np.exp(np.min(X)*-1/24)
X_pow = X_norm ** np.e

# pca = PCA(n_components=0.95)
# X_norm = pca.fit_transform(X_norm)
# X_exp = pca.fit_transform(X_exp)
# X_pow = pca.fit_transform(X_pow)


X_train_norm, X_test_norm, y_train, y_test = train_test_split(X_norm, y, test_size=0.30, random_state=42)
X_train_exp, X_test_exp, _, _ = train_test_split(X_exp, y, test_size=0.30, random_state=42)
X_train_pow, X_test_pow, _, _ = train_test_split(X_pow, y, test_size=0.30, random_state=42)
X_train, X_test, _, _ = train_test_split(X, y, test_size=0.30, random_state=42)


# regressor = MSVR(C=1000, epsilon=0.001)
# regressor = Ridge(alpha=0.1)
#
# regressor.fit(X_train_norm, y_train)
#
# pred = regressor.predict(X_test_norm)
# #
# errors = []
# for i in range(len(pred)):
#     centroids = pred[i]
#     error = vincenty(centroids, y_test[i])
#     errors.append(error)

# k = 3
# for i in tqdm(range(len(X_test_norm))):
#     all_distances = np.sqrt(np.sum(np.abs(X_train_norm - X_test_norm[i]) ** 2, axis=1))
#
#     k_indexes = np.argsort(all_distances)[0:k]
#     centroids = np.mean(y_train[k_indexes, :], axis=0)
#     error = vincenty(centroids, y_test[i])
#     errors.append(error)

# print(f"Mean Error: {np.mean(errors)*1000} meters")
# print(f"Median Error: {np.median(errors)*1000} meters")
# print(f"R2 Score: {r2_score(y_test, pred)}")

regressor = ElasticNet()

# Set up the parameter grid for alpha values
# param_grid = {'alpha': np.logspace(-5, 0, num=100), 'l1_ratio': np.logspace(-3, 0, 10)}
param_grid = {'alpha': np.logspace(-5, 0, 100), 'l1_ratio': np.logspace(-2, 0, 5)}


# Perform grid search with cross-validation
grid_search = GridSearchCV(regressor, param_grid, cv=5, scoring='neg_mean_absolute_error')
grid_search.fit(X_train_pow, y_train)

# Get the best alpha value
best_alpha = grid_search.best_params_['alpha']
best_l1_ratio = grid_search.best_params_['l1_ratio']

# Fit the model with the best alpha
best_regressor = ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio)
best_regressor.fit(X_train_pow, y_train)

# Make predictions
pred = best_regressor.predict(X_test_pow)

# Calculate errors
errors = []
for i in range(len(pred)):
    centroids = pred[i]
    error = vincenty(centroids, y_test[i])
    errors.append(error)

# Print results
print(f"Best alpha: {best_alpha}")
print(f"Best L1_ratio: {best_l1_ratio}")
print(f"Mean Error: {np.mean(errors)*1000:.2f} meters")
print(f"Median Error: {np.median(errors)*1000:.2f} meters")
print(f"R2 Score: {r2_score(y_test, pred):.4f}")

