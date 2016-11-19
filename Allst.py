import pandas as pd
import numpy as np
import math
from sklearn import neighbors
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn import linear_model
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV

##############################
##DECLARACION E VARIABLES ####
##############################

train = pd.read_csv("\Users\if689229\Documents\\train_woe.csv", index_col=0)
train = pd.read_csv("\Users\if689229\Documents\\test_woe.csv", index_col=0) #test
my_median = math.floor(train["loss"].median())  # for categorical data

# data x
train_x = train.ix[:, 0:130]
del train_x[train_x.columns[91]]
train_x_scale = preprocessing.scale(train_x)


# Y lineal
train_y_lin = train["loss"]
test_y_lin = np.zeros(len(train))

# Y Categorica (abajo o arriba de la mediana)
train_y = train["loss"]
train_y[train_y < my_median] = 1
train_y[train_y >= my_median] = 0

# PCA
pca = PCA(n_components=40)
pca.fit(train_x_scale)
print(pca.explained_variance_ratio_)
a = pca.explained_variance_ratio_
plt.plot(a)  # el ratio de varianza no baja mucho despues de los 40 componentes
train_x_scale_reduced = pca.fit_transform(train_x_scale)
train_x_scale = np.concatenate((train_x_scale, train_x_scale_reduced), axis=1)

# Cluster
k_means = cluster.KMeans(n_clusters=4)
k_means.fit(train_x_scale)
b = k_means.fit_predict(train_x_scale)
b = b.astype("float64")
train_x_scale = np.concatenate((train_x_scale, b.reshape(len(train_x_scale), 1)), axis=1)
cluster_x = k_means.fit_transform(train_x_scale)

##############################
#####MODELOS CATEGORICOS######
##############################

# regresion logistica
regr_log = LogisticRegression(C=10, penalty='l2')
regr_log.fit(train_x_scale, train_y)
acc_rl = cross_val_score(regr_log, train_x_scale, train_y, cv=10)
print acc_rl.mean()
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]###### Parametro para grid search
}
GSCV = GridSearchCV(estimator=regr_log, param_grid=param_grid, cv=5)
GSCV.fit(train_x_scale, train_y)
print '\n', GSCV.best_params_

# K-Nearest Neighbor
knn = neighbors.KNeighborsClassifier(n_neighbors=5, leaf_size=10)
knn.fit(train_x_scale, train_y)
acc_KNN = cross_val_score(knn, train_x_scale, train_y, cv=10)
print acc_KNN.mean()
imp = knn.feature_importances_
imp = imp > .0005
train_x_scale = np.where(imp, train_x_scale, None)
train_x_scale = pd.DataFrame(train_x_scale)
train_x_scale = train_x_scale.dropna(axis=1)
print acc_KNN.mean()

param_grid = {'leaf_size': [10, 30, 50, 70, 100],
              'n_neighbors': [5, 10, 15, 20], }

GSCV = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5)
GSCV.fit(train_x_scale, train_y)
print '\n', GSCV.best_params_

# Random-Forest
rfc = RandomForestClassifier(min_samples_split=5, n_estimators=6, min_samples_leaf=2)
rfc.fit(train_x_scale, train_y)
acc_rfc = cross_val_score(rfc, train_x_scale, train_y, cv=10)
print acc_rfc.mean()
imp = rfc.feature_importances_
imp = imp > .0005
train_x_scale = np.where(imp, train_x_scale, None)
train_x_scale = pd.DataFrame(train_x_scale)
train_x_scale = train_x_scale.dropna(axis=1)
param_grid = {'min_samples_leaf': [1, 2, 3],
              'min_samples_split': [2, 3, 4, 5],
              'n_estimators': [1, 2, 3, 4, 5, 6], }

GSCV = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
GSCV.fit(train_x_scale, train_y)
print '\n', GSCV.best_params_

# Boosted Trees
bt = AdaBoostClassifier(n_estimators=3, learning_rate=.5)
bt.fit(train_x_scale, train_y)
acc_bt = cross_val_score(bt, train_x_scale, train_y, cv=10)
print acc_bt.mean()
imp = bt.feature_importances_
imp = imp > .0005
train_x_scale = np.where(imp, train_x_scale, None)
train_x_scale = pd.DataFrame(train_x_scale)
train_x_scale = train_x_scale.dropna(axis=1)
param_grid = {'learning_rate': [.5, 1.0, 1.5, 2],
              'n_estimators': [1, 2, 3, 4, 5, 6],
              }

GSCV = GridSearchCV(estimator=bt, param_grid=param_grid, cv=5)
GSCV.fit(train_x_scale, train_y)
print '\n', GSCV.best_params_

##############################
#####MODELOS DE REGRESION#####
##############################

# Regresion lineal
regr_lin = linear_model.LinearRegression()
regr_lin.fit(train_x_scale, train_y_lin)
pred = regr_lin.predict(train_x_scale)
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
}

GSCV = GridSearchCV(estimator=regr_lin, param_grid=param_grid, cv=5)
GSCV.fit(train_x_scale, train_y_lin)
print '\n', GSCV.best_params_

# Random-Forest
rfc = RandomForestRegressor(min_samples_split=5, n_estimators=6, min_samples_leaf=2)
rfc.fit(train_x_scale, train_y_lin)
pred = rfc.predict(train_x_scale)
imp = rfc.feature_importances_
imp = imp > .0005
train_x_scale = np.where(imp, train_x_scale, None)
train_x_scale = pd.DataFrame(train_x_scale)
train_x_scale = train_x_scale.dropna(axis=1)
param_grid = {'min_samples_leaf': [1, 2, 3],
              'min_samples_split': [2, 3, 4, 5],
              'n_estimators': [1, 2, 3, 4, 5, 6], }
np.savetxt("Sample.csv", pred, delimiter=",")
######{'min_samples_split': 2, 'n_estimators': 6, 'min_samples_leaf': 3}
GSCV = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5, scoring="neg_mean_absolute_error")
GSCV.fit(train_x_scale, train_y_lin)
print '\n', GSCV.best_params_

# Boosted-Trees
bt = AdaBoostRegressor(n_estimators=10, learning_rate=.001)
bt.fit(train_x_scale, train_y_lin)
pred = bt.predict(train_x_scale)
imp = bt.feature_importances_
imp = imp > .0005
train_x_scale = np.where(imp, train_x_scale, None)
train_x_scale = pd.DataFrame(train_x_scale)
train_x_scale = train_x_scale.dropna(axis=1)
param_grid = {'learning_rate': [.5, 1.0, 1.5, 2],
              'n_estimators': [1, 2, 3, 4, 5, 6],
              }

GSCV = cross_val_score(estimator=bt, fit_params=param_grid, cv=5)
GSCV.fit(train_x_scale, train_y_lin)
print '\n', GSCV.best_params_

# Mean Absolute Error
mae = sum(abs(train_y_lin - pred)) / len(train_y_lin)
print mae
