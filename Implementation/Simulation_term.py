import warnings
import numpy as np
from copy import deepcopy
from sklearn.datasets import load_boston
from sklearn.exceptions import DataConversionWarning
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from Implementation.master import master
from pandas import read_csv, read_table
import matplotlib.pyplot as plt

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

estimators = [
    ElasticNetCV(random_state=0),

    KNeighborsRegressor(algorithm='kd_tree', weights='distance', n_jobs=-1)
]

if __name__ == '__main__':
    """
    # Boston
    data = load_boston(return_X_y=True)
    x = data[0]
    y = data[1]
    master(x, y, deepcopy(estimators), verbose=True, eval_file='Boston.txt', replicate=10)
    """
    # Wine
    data = read_csv('Datasets/winequality-white.csv', sep=';')
    x = data.iloc[:, range(data.shape[1]-1)].values
    y = data.iloc[:, data.shape[1]-1].values
    master(x, y, deepcopy(estimators), verbose=True, eval_file='Wine.txt', replicate=10)
    """
    # Year
    data = read_table('Datasets/YearPredictionMSD.txt', sep=',')
    x = data.iloc[:, range(1, data.shape[1])].values
    y = data.iloc[:, 0].values
    master(x, y, deepcopy(estimators), verbose=True, eval_file='Year.txt')

    size = 1_000

    x1 = np.random.uniform(0, 2, size=size)

    y = []

    for i in range(size):

        if x1[i] < 1:

            y.append(float(np.random.normal(loc=x1[i], scale=0.2, size=1)))

        else:

            y.append(np.exp(x1[i]) + float(np.random.normal(scale=0.2)))

    y = np.array(y)

    plt.plot(x1, y, 'o')

    plt.show()

    x1 = x1.reshape(size, 1)

    master(x1, y, deepcopy(estimators), verbose=True, eval_file='Simulation.txt')
    """
