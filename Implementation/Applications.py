import numpy as np
import os
import warnings
import pandas as pd
from sklearn.exceptions import DataConversionWarning
from NNS.Implementation.Applications_beckend import apply, apply2
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

if __name__ == '__main__':

    # Superconductivity
    os.chdir('C:/Users/Victor/PycharmProjects/ML/NNS/Results/Superconductivity')
    data = pd.read_csv('Superconductivity.csv')
    x = data.iloc[:, range(0, data.shape[1] - 1)].values
    y = data.iloc[:, -1].values

    apply(x=x,
          y=y,
          base_est=[
              LinearRegression(),
              LassoCV(),
              RidgeCV(),
              GridSearchCV(BaggingRegressor(), {'n_estimators': (5, 10, 20, 50)}),
              RandomForestRegressor(),
              GridSearchCV(GradientBoostingRegressor(), {'learning_rate': (0.01, 0.1, 0.2),
                                                         'n_estimators': (50, 100, 200)})
          ],
          layer_grid=[1, 3, 10],
          hidden=100
          )

    apply2(x=x,
           y=y,
           base_est=[
              GridSearchCV(BaggingRegressor(), {'n_estimators': (5, 10, 20, 50)}),
              RandomForestRegressor(),
              GridSearchCV(GradientBoostingRegressor(), {'learning_rate': (0.01, 0.1, 0.2),
                                                         'n_estimators': (50, 100, 200)})
           ],
           layer_grid=3,
           hidden=100,
           ensemble_method='f_to_m',
           ensemble_addition=False)

    # Blog
    os.chdir('C:/Users/Victor/PycharmProjects/ML/NNS/Results/Blog')
    data = pd.read_csv('Blog.csv')
    x = data.iloc[:, range(0, data.shape[1] - 1)].values
    y = data.iloc[:, -1].values

    apply(x=x,
          y=y,
          base_est=[
              LinearRegression(),
              LassoCV(),
              RidgeCV(),
              GridSearchCV(BaggingRegressor(), {'n_estimators': (5, 10, 20, 50)}),
              RandomForestRegressor(),
              GridSearchCV(GradientBoostingRegressor(), {'learning_rate': (0.01, 0.1, 0.2),
                                                         'n_estimators': (50, 100, 200)})
          ],
          layer_grid=[3],
          hidden=100
          )

    apply2(x=x,
           y=y,
           base_est=[
               GridSearchCV(BaggingRegressor(), {'n_estimators': (5, 10, 20, 50)}),
               RandomForestRegressor(),
               GridSearchCV(GradientBoostingRegressor(), {'learning_rate': (0.01, 0.1, 0.2),
                                                          'n_estimators': (50, 100, 200)})
           ],
           layer_grid=3,
           hidden=100,
           ensemble_method='f_to_m',
           ensemble_addition=True)

    # Year
    os.chdir('C:/Users/Victor/PycharmProjects/ML/NNS/Results/Year')
    data = pd.read_table('Year.txt', sep=',')
    x = data.iloc[:, range(1, data.shape[1])].values
    y = data.iloc[:, 0].values

    apply(x=x,
          y=y,
          base_est=[
              LinearRegression(),
              LassoCV(),
              RidgeCV(),
              GridSearchCV(BaggingRegressor(), {'n_estimators': (5, 10, 20, 50)}),
              RandomForestRegressor(),
              GridSearchCV(GradientBoostingRegressor(), {'learning_rate': (0.01, 0.1, 0.2),
                                                         'n_estimators': (50, 100, 200)})
          ],
          layer_grid=[1, 3, 10],
          hidden=100
          )

    apply2(x=x,
           y=y,
           base_est=[
              GridSearchCV(BaggingRegressor(), {'n_estimators': (5, 10, 20, 50)}),
              RandomForestRegressor(),
              GridSearchCV(GradientBoostingRegressor(), {'learning_rate': (0.01, 0.1, 0.2),
                                                         'n_estimators': (50, 100, 200)})
           ],
           layer_grid=3,
           hidden=100,
           ensemble_method='f_to_m',
           ensemble_addition=False)

    # GPU

    os.chdir('C:/Users/Victor/PycharmProjects/ML/NNS/Results/GPU')
    data = pd.read_csv('GPU.csv')

    x = data.iloc[:, :13].values
    y = data.iloc[:, 14:].apply(np.mean, axis=1).values

    apply(x=x,
          y=y,
          base_est=[
              LinearRegression(),
              LassoCV(),
              RidgeCV(),
              GridSearchCV(BaggingRegressor(), {'n_estimators': (5, 10, 20, 50)}),
              RandomForestRegressor(),
              GridSearchCV(GradientBoostingRegressor(), {'learning_rate': (0.01, 0.1, 0.2),
                                                         'n_estimators': (50, 100, 200)})
          ],
          layer_grid=[3],
          hidden=100
          )

    apply2(x=x,
           y=y,
           base_est=[
               GridSearchCV(BaggingRegressor(), {'n_estimators': (5, 10, 20, 50)}),
               RandomForestRegressor(),
               GridSearchCV(GradientBoostingRegressor(), {'learning_rate': (0.01, 0.1, 0.2),
                                                          'n_estimators': (50, 100, 200)})
           ],
           layer_grid=3,
           hidden=100,
           ensemble_method='f_to_w',
           ensemble_addition=False)
