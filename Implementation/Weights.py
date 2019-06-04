import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pickle
import os
from sklearn.model_selection import train_test_split


if __name__ == '__main__':

    # Superconductivity
    os.chdir('C:/Users/Victor/PycharmProjects/ML/NNS/Results/Superconductivity')
    data = pd.read_csv('Superconductivity.csv')
    x = data.iloc[:, range(0, data.shape[1] - 1)].values
    y = data.iloc[:, -1].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 / 4, random_state=0)

    with open('nnsm3', 'rb') as f:
        nne = pickle.load(f)

    w = pd.DataFrame(nne.get_weights(x_test)[0].numpy())
    w.columns = ['LSR', 'Lasso', 'Ridge', 'Bagging', 'RF', 'GBR']

    f, ax = plt.subplots()
    ax.set_ylabel('Weights', fontsize=18)
    w.boxplot(fontsize='large')
    f.savefig('Weights.pdf', bbox_inches='tight')

    # Blog
    os.chdir('C:/Users/Victor/PycharmProjects/ML/NNS/Results/Blog')
    data = pd.read_csv('Blog.csv')
    x = data.iloc[:, range(0, data.shape[1] - 1)].values
    y = data.iloc[:, -1].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/4, random_state=0)

    with open('nnswa10', 'rb') as f:
        nne = pickle.load(f)

    w = pd.DataFrame(nne.get_weights(x_test)[0].numpy())
    w.columns = ['LSR', 'Lasso', 'Ridge', 'Bagging', 'RF', 'GBR']

    f, ax = plt.subplots()
    ax.set_ylabel('Weights', fontsize=18)
    w.boxplot(fontsize='large')
    f.savefig('Weights.pdf', bbox_inches='tight')

    # Year
    os.chdir('C:/Users/Victor/PycharmProjects/ML/NNS/Results/Year')
    data = pd.read_table('Year.txt', sep=',')
    x = data.iloc[:, range(1, data.shape[1])].values
    y = data.iloc[:, 0].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/4, random_state=0)

    with open('nnsm3', 'rb') as f:
        nne = pickle.load(f)

    w = pd.DataFrame(nne.get_weights(x_test)[0].numpy())
    w.columns = ['LSR', 'Lasso', 'Ridge', 'Bagging', 'RF', 'GBR']

    f, ax = plt.subplots()
    ax.set_ylabel('Weights', fontsize=18)
    w.boxplot(fontsize='large')
    f.savefig('Weights.pdf', bbox_inches='tight')

    # GPU
    os.chdir('C:/Users/Victor/PycharmProjects/ML/NNS/Results/GPU')
    data = pd.read_csv('GPU.csv')

    x = data.iloc[:, :13].values
    y = data.iloc[:, 14:].apply(np.mean, axis=1).values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 / 4, random_state=0)

    with open('nnsw3', 'rb') as f:
        nne = pickle.load(f)

    w = pd.DataFrame(nne.get_weights(x_test)[0].numpy())
    w.columns = ['LSR', 'Lasso', 'Ridge', 'Bagging', 'RF', 'GBR']

    f, ax = plt.subplots()
    ax.set_ylabel('Weights', fontsize=18)
    w.boxplot(fontsize='large')
    f.savefig('Weights.pdf', bbox_inches='tight')
