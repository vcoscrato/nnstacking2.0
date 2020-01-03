import numpy as np
import os
import warnings
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from nns import NNS
from sklearn.exceptions import DataConversionWarning, ConvergenceWarning
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=DeprecationWarning)

os.chdir('/home/vcoscrato/Documents/NNStacking')
data = pd.read_csv('Datasets/train.csv')
x = data.iloc[:, range(0, data.shape[1] - 1)].values
y = data.iloc[:, -1].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, shuffle=True, random_state=0)

models = []
f, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(11, 8))
txt = open('Results/CV/results.txt', 'w')
for idx, (train, test) in enumerate(KFold(n_splits=4, random_state=0, shuffle=True).split(x_train, y_train)):    
    print(idx)
    
    base_est=[
        LinearRegression(),
        LassoCV(random_state=0),
        RidgeCV(),
        GridSearchCV(BaggingRegressor(random_state=0), {'n_estimators': (5, 10, 20, 50)}),
        RandomForestRegressor(random_state=0),
        GridSearchCV(GradientBoostingRegressor(random_state=0), {'learning_rate': (0.01, 0.1, 0.2),'n_estimators': (50, 100, 200)})
    ]

    model = NNS(
            verbose=0,
            nn_weight_decay=0.0,
            es=True,
            es_give_up_after_nepochs=10,
            num_layers=3,
            hidden_size=100,
            estimators=base_est,
            gpu=False,
            ensemble_method="CNNS",
            ensemble_addition=False,
            es_splitter_random_state=0
        ).fit(x[train], y[train].reshape(len(train), 1))
    
    models.append(model)
    
    preds = model.predict(x[test]).flatten()
    print(('MSE:', mean_squared_error(y[test], preds), '; MAE:',
        mean_absolute_error(y[test], preds), '; MSE STD:', 
        np.std((preds - y[test]) ** 2) / (len(test) ** (1 / 2)), '; MAE STD:',
        np.std(abs(preds - y[test])) / (len(test) ** (1 / 2))), file=txt)

    w = pd.DataFrame(model.get_weights(x[test])[0])
    w.columns = ['LSR', 'Lasso', 'Ridge', 'Bagging', 'RF', 'GBR']
    
    plt.subplot(2, 2, idx+1)
    w.boxplot(fontsize='large')
    plt.ylabel('Weights', fontsize=15)
    plt.title('Fold %i' %(idx+1))

f.savefig('Results/CV/Weights.pdf', bbox_inches='tight')
txt.close()
with open('Results/CV/models.pkl', 'wb') as f:
    pickle.dump(models, f, pickle.HIGHEST_PROTOCOL)

weights = np.empty((len(models), len(y_test), len(base_est)))
for idx, model in enumerate(models):
    weights[idx] = model.get_weights(x_test)[0]
sd = pd.DataFrame(weights.std(axis=(0)))
sd.columns = ['LSR', 'Lasso', 'Ridge', 'Bagging', 'RF', 'GBR']
f, ax = plt.subplots()
sd.boxplot(fontsize='large')
plt.ylabel('Deviations', fontsize=15)
f.savefig('Results/CV/Deviation.pdf', bbox_inches='tight')
