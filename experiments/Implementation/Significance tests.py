import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pickle
import os
from nne import NNE, LinearStack
from sklearn.model_selection import train_test_split
from scipy.stats import ttest_rel
from tabulate import tabulate
from keras.models import Sequential
from keras.layers import Dense, Dropout

def significance_table(x, y, path, nns_size, nnmeta_size, nn_size):
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/4, random_state=0)
    preds = np.empty((len(y_test), 7))
    
    #nnswa
    with open(path+'/Fitted/nnswa'+nns_size[0], 'rb') as f:
        model = pickle.load(f)
    preds[:, 0] = y_test - model.predict(x_test).flatten()
    
    #nnsma
    with open(path+'/Fitted/nnsma'+nns_size[1], 'rb') as f:
        model = pickle.load(f)
    preds[:, 1] = y_test - model.predict(x_test).flatten()
    
    #nnsw
    with open(path+'/Fitted/nnsw'+nns_size[2], 'rb') as f:
        model = pickle.load(f)
    preds[:, 2] = y_test - model.predict(x_test).flatten()
    
    #nnsm
    with open(path+'/Fitted/nnsm'+nns_size[3], 'rb') as f:
        model = pickle.load(f)
    preds[:, 3] = y_test - model.predict(x_test).flatten()
    
    estimators = model.estimators
    #breiman
    if x_train.shape[0] > 50000:
        x_train2 = x_train[range(50000), :]
        y_train2 = y_train[range(50000)]
        g = model.predictions[range(50000), :].reshape(len(x_train2), 3)
        model = LinearStack(estimators=estimators, verbose=0).fit(x_train2, y_train2, g)
    else:
        model = LinearStack(estimators=estimators, verbose=0).fit(x_train, y_train, model.predictions.reshape(len(x_train), 3))
    preds[:, 4] = y_test - model.predict(x_test).flatten()
    
    #NNMETA
    model = Sequential()
    for j in range(nnmeta_size):
        model.add(Dense(100, input_shape=(3,), activation='elu'))
        model.add(Dropout(0.5))
    model.add(Dense(1, activation='linear'))
    model.compile(loss="mse", optimizer='adam')
    model.load_weights(path+'/Fitted/NNMETA_Weights')
    g_test = np.zeros(len(y_test)).reshape(len(y_test), 1)
    for est in estimators:
        g_test = np.hstack((g_test, est.predict(x_test).reshape(len(y_test), 1)))
    g_test = np.delete(g_test, 0, 1)
    preds[:, 5] = y_test - model.predict(g_test).flatten()
    
    #NN
    model = Sequential()
    for j in range(nn_size):
        model.add(Dense(100, input_shape=(x.shape[1],), activation='elu'))
        model.add(Dropout(0.5))
    model.add(Dense(1, input_shape=(x.shape[1],), activation='linear'))
    model.compile(loss="mse", optimizer='adam')
    model.load_weights(path+'/Fitted/NN_Weights')
    preds[:, 6] = y_test - model.predict(x_test).flatten()
    
    out = np.empty((7, 7))
    for i in range(7):
        for j in range(7):
            out[i, j] = ttest_rel(preds[:, i], preds[:, j]).pvalue
    
    with open(path+'/Significance.txt', 'w') as f:
        print(tabulate(out, tablefmt="latex", floatfmt=".2f"), file=f)
        
    return 'Success!'
    
    
if __name__ == '__main__':

    # Superconductivity
    data = pd.read_csv('Datasets/Superconductivity.csv')
    x = data.iloc[:, range(0, data.shape[1] - 1)].values
    y = data.iloc[:, -1].values
    significance_table(x, y, 'Results/Superconductivity', ['10', '1', '10', '10'], 2, 2)
    
    # Blog
    data = pd.read_csv('Datasets/Blog.csv')
    x = data.iloc[:, range(0, data.shape[1] - 1)].values
    y = data.iloc[:, -1].values
    significance_table(x, y, 'Results/Blog', ['10', '1', '10', '10'], 2, 0)
    '''
    # Year
    data = pd.read_table('Datasets/Year.txt', sep=',')
    x = data.iloc[:, range(1, data.shape[1])].values
    y = data.iloc[:, 0].values
    significance_table(x, y, 'Results/Year', ['10', '3', '10', '3'], 1, 1)

    # GPU
    data = pd.read_csv('GPU.csv')
    x = data.iloc[:, :13].values
    y = data.iloc[:, 14:].apply(np.mean, axis=1).values
    significance_table(x, y, 'Results/GPU', ['3', '3', '3', '3'], 10, 10)
    '''
   