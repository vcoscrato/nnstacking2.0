#----------------------------------------------------------------------
# Copyright 2018 Victor Coscrato <vcoscrato@gmail.com>;
# Copyright 2018 Marco Inacio <pythonpackages@marcoinacio.com>
#
#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, version 3 of the License.

#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    See the
#GNU General Public License for more details.

#You should have received a copy of the GNU General Public License
#along with this program. If not, see <http://www.gnu.org/licenses/>.
#----------------------------------------------------------------------

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_predict
from scipy.optimize import nnls

class LinearStack(BaseEstimator):

    '''
    Breiman Stacking
    '''

    def __init__(self, estimators=None, fitted=False, verbose=1):

        for prop in dir():

            if prop != "self":

                setattr(self, prop, locals()[prop])

    def fit(self, x_train, y_train):

        z = np.zeros(len(y_train)).reshape(len(y_train),1)

        for i in range(len(self.estimators)):

            z = np.hstack((z, cross_val_predict(self.estimators[i], x_train, y_train.reshape(-1)).reshape(len(y_train),1)))

            if self.verbose > 1:

                print('Cross-validated: ', self.estimators[i])

            if not self.fitted:

                self.estimators[i].fit(x_train, y_train.reshape(-1))

                if self.verbose > 1:

                    print('Estimated: ', self.estimators[i])

        adj = nnls(np.delete(z, 0, 1), y_train.reshape(-1))

        self.parameters = adj[0]

        self.train_mse = adj[1]/len(y_train)

        if self.verbose > 0:

            print('Ensemble fitted MSE (train): ', self.train_mse/len(y_train))

        return self

    def prediction_matrix(self, x):

        z = np.zeros(x.shape[0]).reshape(x.shape[0], 1)

        for i in range(len(self.estimators)):

            z = np.hstack((z, self.estimators[i].predict(x).reshape(x.shape[0], 1)))

        return np.delete(z, 0, 1)

    def predict(self, x):

        z = self.prediction_matrix(x)

        return np.dot(z, self.parameters)

    def score(self, x, y):

        return np.mean(np.square(self.predict(x) - np.array(y)))

    def polish(self, x):

        z = self.prediction_matrix(x)

        exp_y_x = np.dot(z, self.parameters)

        out = []

        for i in range(len(x)):

            gx = z[i, :]

            gx = np.array(gx, ndmin=2)

            N_inv = np.linalg.inv(np.matmul(gx.T, gx))

            theta = exp_y_x * np.matmul(N_inv, gx.T)

            out.append(np.matmul(theta.T, gx.T))

        return out

    def get_weights(self, x):

        z = self.prediction_matrix(x)

        exp_y_x = np.dot(z, self.parameters)

        out = np.zeros(len(self.estimators)).reshape(1, len(self.estimators))

        for i in range(len(x)):

            gx = z[i, :]

            gx = np.array(gx, ndmin=2)

            N_inv = np.linalg.inv(np.matmul(gx.T, gx))

            theta = exp_y_x * np.matmul(N_inv, gx.T)

            out = np.vstack((out, theta))

        return np.delete(out, 0, 0)

