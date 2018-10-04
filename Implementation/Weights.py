import warnings
import nnensemble
import numpy as np
from copy import deepcopy
from sklearn.datasets import load_boston
from sklearn.exceptions import DataConversionWarning
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from Implementation.master import master
from pandas import read_csv, read_table
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

estimators = [
    LinearRegression(),

    KNeighborsRegressor(algorithm='kd_tree', weights='distance', n_jobs=-1)
]

data = read_csv('C:/Users/Victor/PycharmProjects/ML/Implementation/Datasets/winequality-white.csv', sep=';')
X = data.iloc[:, range(data.shape[1] - 1)].values
y = data.iloc[:, data.shape[1] - 1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

nne = nnensemble.NNE(
    verbose=2,
    nn_weight_decay=0.0,
    es=True,
    es_give_up_after_nepochs=10,
    hls_multiplier=10,
    nhlayers=10,
    estimators=deepcopy(estimators),
    gpu=False,
    ensemble_method="f_to_m",
    ensemble_addition=False
).fit(X_train, y_train.reshape(len(y_train), 1))

linear = nnensemble.LinearStack(estimators=estimators, verbose=1).fit(X_train, y_train)

print(linear.score(X_test, y_test))

print(- nne.score(X_test, y_test.reshape(len(y_test), 1)))

w_nne = nne.get_weights(X_test)

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.plot(np.array(w_nne[0][:, 0]), 'o')
ax1.hlines(linear.parameters[0], xmin=0, xmax=len(y_test), colors='red')
ax2.plot(np.array(w_nne[0][:, 1]), 'o')
ax2.hlines(linear.parameters[1], xmin=0, xmax=len(y_test), colors='red')
ax1.set_title('Linear regression weights')
ax1.set_title('KNN weights')



