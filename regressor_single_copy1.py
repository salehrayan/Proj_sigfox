import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.multioutput import MultiOutputRegressor
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

dataset_train = pd.read_csv('ahmadi_data_masir_asli_darovorde/train_points_with_all_columns.csv')
dataset_test = pd.read_csv('ahmadi_data_masir_asli_darovorde/test_points_with_all_columns.csv')
X_train = np.array(dataset_train.iloc[:, :137])
X_test = np.array(dataset_test.iloc[:, :137])
y_train = np.array(dataset_train[["Latitude", "Longitude"]])
y_test = np.array(dataset_test[["Latitude", "Longitude"]])


X_train_norm = (X_train - np.min(X_train))/np.min(X_train)*-1
X_test_norm = (X_test - np.min(X_test))/np.min(X_test)*-1
X_train_exp = np.exp((X_train - np.min(X_train))/24)/np.exp(np.min(X_train)*-1/24)
X_test_exp = np.exp((X_test - np.min(X_test))/24)/np.exp(np.min(X_test)*-1/24)
X_train_pow = X_train_norm ** np.e
X_test_pow = X_test_norm ** np.e

# pca = PCA(n_components=0.95)
# X_norm = pca.fit_transform(X_norm)
# X_exp = pca.fit_transform(X_exp)
# X_pow = pca.fit_transform(X_pow)


# X_train_norm, X_test_norm, y_train, y_test = train_test_split(X_norm, y, test_size=0.30, random_state=42)
# X_train_exp, X_test_exp, _, _ = train_test_split(X_exp, y, test_size=0.30, random_state=42)
# X_train_pow, X_test_pow, _, _ = train_test_split(X_pow, y, test_size=0.30, random_state=42)
# X_train, X_test, _, _ = train_test_split(X, y, test_size=0.30, random_state=42)

# regressor = MSVR(C=1000, epsilon=0.001)
# poly = PolynomialFeatures(degree=2)
regressor = KNeighborsRegressor(n_neighbors=1)

regressor.fit(X_train, y_train)

pred = regressor.predict(X_test)

errors = []
for i in range(len(pred)):
    centroids = pred[i]
    error = vincenty(centroids, y_test[i])
    errors.append(error)

# k = 3
# for i in tqdm(range(len(X_test_norm))):
#     all_distances = np.sqrt(np.sum(np.abs(X_train_norm - X_test_norm[i]) ** 2, axis=1))
#
#     k_indexes = np.argsort(all_distances)[0:k]
#     centroids = np.mean(y_train[k_indexes, :], axis=0)
#     error = vincenty(centroids, y_test[i])
#     errors.append(error)

print(f"Mean Error: {np.mean(errors)*1000} meters")
print(f"Median Error: {np.median(errors)*1000} meters")
print(f"R2 Score: {r2_score(y_test, pred)}")
#
# regressor = ElasticNet()
#
# # Set up the parameter grid for alpha values
# param_grid = {'alpha': np.logspace(-5, 9, num=2000), 'l1_ratio': np.logspace(-2, 2, 50)}
#
# # Perform grid search with cross-validation
# grid_search = GridSearchCV(regressor, param_grid, cv=5, scoring='neg_mean_absolute_error')
# grid_search.fit(X_train_norm, y_train)
#
# # Get the best alpha value
# best_alpha = grid_search.best_params_['alpha']
#
# # Fit the model with the best alpha
# best_regressor = ElasticNet(alpha=best_alpha)
# best_regressor.fit(X_train_norm, y_train)
#
# # Make predictions
# pred = best_regressor.predict(X_test_norm)
#
# # Calculate errors
# errors = []
# for i in range(len(pred)):
#     centroids = pred[i]
#     error = vincenty(centroids, y_test[i])
#     errors.append(error)
#
# # Print results
# print(f"Best Alpha: {best_alpha}")
# print(f"Mean Error: {np.mean(errors)*1000:.2f} meters")
# print(f"Median Error: {np.median(errors)*1000:.2f} meters")
# print(f"R2 Score: {r2_score(y_test, pred):.4f}")

