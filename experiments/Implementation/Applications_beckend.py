import numpy as np
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import mean_squared_error, mean_absolute_error
from nne import NNE, LinearStack
import pickle
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from time import time
import sys
np.set_printoptions(threshold=sys.maxsize)


def apply(x, y, base_est, layer_grid, hidden):

    t0 = time()

    f = open('Full_fit.txt', 'w')

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/4, random_state=0)

    print('Cross-Validating / Fitting ...')

    g = np.zeros(len(y_train)).reshape(len(y_train), 1)

    preds = np.empty((len(y_test), len(base_est)))

    for i, est in enumerate(base_est):

        g = np.hstack((g, cross_val_predict(est, x_train, y_train, cv=2).reshape(len(y_train), 1)))

        print('Cross-validated: ', i+1, '/', len(base_est))

        est.fit(x_train, y_train)

        print('Fitted: ', i+1, '/', len(base_est))

        preds[:, i] = est.predict(x_test)

        print(('Single_perf: ', mean_squared_error(y_test, preds[:, i]), ';',
               mean_absolute_error(y_test, preds[:, i]), ';',
               np.std((preds[:, i] - y_test) ** 2) / (len(y_test) ** (1 / 2)), ';',
               np.std(abs(preds[:, i] - y_test)) / (len(y_test) ** (1 / 2))), file=f)

        preds[:, i] -= y_test

    print(np.corrcoef(preds, rowvar=False))

    print(np.corrcoef(preds, rowvar=False), file=f)

    g = np.delete(g, 0, 1)

    if x_train.shape[0] > 50000:

        x_train2 = x_train[range(50000), :]

        y_train2 = y_train[range(50000)]

        linear = LinearStack(estimators=base_est, fitted=True, verbose=0).fit(x_train2, y_train2)

    else:

        linear = LinearStack(estimators=base_est, fitted=True, verbose=0).fit(x_train, y_train)

    pred = linear.predict(x_test)

    print(('Breiman\'s stacking: ', mean_squared_error(y_test, pred), ';',
           mean_absolute_error(y_test, pred), ';',
           np.std((pred - y_test)**2)/(len(y_test)**(1/2)), ';',
           np.std(abs(pred - y_test))/(len(y_test)**(1/2))), file=f)

    print(('Breimans exec time = ', time() - t0), file=f)

    t0 = time()

    for i, layer in enumerate(layer_grid):

        print('Fitting NNS MA ...')

        nne = NNE(
            verbose=0,
            nn_weight_decay=0.0,
            es=True,
            es_give_up_after_nepochs=10,
            num_layers=layer,
            hidden_size=hidden,
            estimators=base_est,
            gpu=False,
            ensemble_method="f_to_m",
            ensemble_addition=True,
            es_splitter_random_state=0
        ).fit(x_train, y_train.reshape(len(y_train), 1), g.reshape(g.shape[0], 1, g.shape[1]))

        with open('Fitted/nnsma'+str(layer), 'wb') as file:
            pickle.dump(nne, file)

        pred = nne.predict(x_test)
        mse = mean_squared_error(y_test, pred)
        mae = mean_absolute_error(y_test, pred)
        std = np.std((pred.flatten()-y_test)**2) / (len(y_test)**(1/2))
        std2 = np.std(abs(pred.flatten()-y_test)) / (len(y_test)**(1/2))
        print(('NNSMA'+str(layer)+': ', mse, ';', mae, ';',
               std, std2), file=f)

        print('Fitting NNS WA ...')

        nne = NNE(
            verbose=0,
            nn_weight_decay=0.0,
            es=True,
            es_give_up_after_nepochs=10,
            num_layers=layer,
            hidden_size=hidden,
            estimators=base_est,
            gpu=False,
            ensemble_method="f_to_w",
            ensemble_addition=True,
            es_splitter_random_state=0,
        ).fit(x_train, y_train.reshape(len(y_train), 1), g.reshape(g.shape[0], 1, g.shape[1]))

        with open('Fitted/nnswa'+str(layer), 'wb') as file:
            pickle.dump(nne, file)

        pred = nne.predict(x_test)
        mse = mean_squared_error(y_test, pred)
        mae = mean_absolute_error(y_test, pred)
        std = np.std((pred.flatten()-y_test)**2) / (len(y_test)**(1/2))
        std2 = np.std(abs(pred.flatten()-y_test)) / (len(y_test)**(1/2))
        print(('NNSWA'+str(layer)+': ', mse, ';', mae, ';',
               std, std2), file=f)

        print('Fitting NNS M ...')

        nne = NNE(
            verbose=0,
            nn_weight_decay=0.0,
            es=True,
            es_give_up_after_nepochs=10,
            num_layers=layer,
            hidden_size=hidden,
            estimators=base_est,
            gpu=False,
            ensemble_method="f_to_m",
            ensemble_addition=False,
            es_splitter_random_state=0,
        ).fit(x_train, y_train.reshape(len(y_train), 1), g.reshape(g.shape[0], 1, g.shape[1]))

        with open('Fitted/nnsm'+str(layer), 'wb') as file:
            pickle.dump(nne, file)

        pred = nne.predict(x_test)
        mse = mean_squared_error(y_test, pred)
        mae = mean_absolute_error(y_test, pred)
        std = np.std((pred.flatten()-y_test)**2) / (len(y_test)**(1/2))
        std2 = np.std(abs(pred.flatten()-y_test)) / (len(y_test)**(1/2))
        print(('NNSM'+str(layer)+': ', mse, ';', mae, ';',
               std, std2), file=f)

        print('Fitting NNS W ...')

        nne = NNE(
            verbose=0,
            nn_weight_decay=0.0,
            es=True,
            es_give_up_after_nepochs=10,
            num_layers=layer,
            hidden_size=hidden,
            estimators=base_est,
            gpu=False,
            ensemble_method="f_to_w",
            ensemble_addition=False,
            es_splitter_random_state=0,
        ).fit(x_train, y_train.reshape(len(y_train), 1), g.reshape(g.shape[0], 1, g.shape[1]))

        with open('Fitted/nnsw'+str(layer), 'wb') as file:
            pickle.dump(nne, file)

        pred = nne.predict(x_test)
        mse = mean_squared_error(y_test, pred)
        mae = mean_absolute_error(y_test, pred)
        std = np.std((pred.flatten()-y_test)**2) / (len(y_test)**(1/2))
        std2 = np.std(abs(pred.flatten()-y_test)) / (len(y_test)**(1/2))
        print(('NNSW'+str(layer)+': ', mse, ';', mae, ';',
               std, std2), file=f)

        print(('NNS exec time = ', time() - t0), file=f)

        t0 = time()

        print('Fitting NN ...')

        x_train1, x_val, y_train1, y_val = train_test_split(x_train, y_train, test_size=1 / 10, random_state=0)

        model = Sequential()
        for j in range(i):
            model.add(Dense(hidden, input_shape=(x.shape[1],), activation='elu'))
            model.add(Dropout(0.5))
        model.add(Dense(1, activation='linear'))
        model.compile(loss="mse", optimizer='adam')
        callbacks = [EarlyStopping(monitor='val_loss', patience=10),
                     ModelCheckpoint(filepath='Fitted/NN_Weights', monitor='val_loss', save_best_only=True)]
        model.fit(x_train1, y_train1, epochs=1000, batch_size=128, verbose=0, callbacks=callbacks, validation_data=(x_val, y_val))
        model.load_weights('Fitted/NN_Weights')

        pred = model.predict(x_test)
        mse = mean_squared_error(y_test, pred)
        mae = mean_absolute_error(y_test, pred)
        std = np.std((pred.flatten()-y_test)**2) / (len(y_test)**(1/2))
        std2 = np.std(abs(pred.flatten()-y_test)) / (len(y_test)**(1/2))

        print(('NN'+str(layer)+': ', mse, ';', mae, ';',
               std, std2), file=f)

        print(('NN exec time = ', time() - t0), file=f)

        t0 = time()

        print('Fitting NN Meta ...')

        g_train1, g_val, y_train1, y_val = train_test_split(g, y_train, test_size=1/10, random_state=0)

        model = Sequential()
        for j in range(i):
            model.add(Dense(hidden, input_shape=(g.shape[1],), activation='elu'))
            model.add(Dropout(0.5))
        model.add(Dense(1, activation='linear'))
        model.compile(loss="mse", optimizer='adam')
        callbacks = [EarlyStopping(monitor='val_loss', patience=10),
                     ModelCheckpoint(filepath='Fitted/NNMETA_Weights', monitor='val_loss', save_best_only=True)]
        model.fit(g_train1, y_train1, epochs=1000, batch_size=128, verbose=0, callbacks=callbacks, validation_data=(g_val, y_val))
        model.load_weights('Fitted/NNMETA_Weights')

        g_test = np.zeros(len(y_test)).reshape(len(y_test), 1)
        for est in base_est:
            g_test = np.hstack((g_test, est.predict(x_test).reshape(len(y_test), 1)))
        g_test = np.delete(g_test, 0, 1)

        pred = model.predict(g_test)
        mse = mean_squared_error(y_test, pred)
        mae = mean_absolute_error(y_test, pred)
        std = np.std((pred.flatten()-y_test)**2) / (len(y_test)**(1/2))
        std2 = np.std(abs(pred.flatten()-y_test)) / (len(y_test)**(1/2))

        print(('NNMETA'+str(layer)+': ',  mse, ';', mae, ';',
               std, std2), file=f)

        print(('NNMETA exec time = ', time() - t0), file=f)

    f.close()

    return 'Success!'


def apply2(x, y, base_est, layer_grid, hidden, ensemble_method, ensemble_addition):

    f = open('Remove_linear.txt', 'w')

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1 / 4, random_state=0)

    print('Cross-Validating / Fitting ...')

    g = np.zeros(len(y_train)).reshape(len(y_train), 1)

    preds = np.empty((len(y_test), len(base_est)))

    for i, est in enumerate(base_est):

        g = np.hstack((g, cross_val_predict(est, x_train, y_train, cv=2).reshape(len(y_train), 1)))

        print('Cross-validated: ', i+1, '/', len(base_est))

        est.fit(x_train, y_train)

        print('Fitted: ', i+1, '/', len(base_est))

        preds[:, i] = est.predict(x_test)

    g = np.delete(g, 0, 1)

    if x_train.shape[0] > 50000:

        x_train2 = x_train[range(50000), :]

        y_train2 = y_train[range(50000)]

        linear = LinearStack(estimators=base_est, fitted=True, verbose=0).fit(x_train2, y_train2)

    else:

        linear = LinearStack(estimators=base_est, fitted=True, verbose=0).fit(x_train, y_train)

    pred = linear.predict(x_test)

    print(('Breiman\'s stacking: ', mean_squared_error(y_test, pred), ';',
           mean_absolute_error(y_test, pred), ';',
           np.std((pred - y_test)**2)/(len(y_test)**(1/2)), ';',
           np.std(abs(pred - y_test))/(len(y_test)**(1/2))), file=f)

    print('Fitting NNS ...')

    nne = NNE(
        verbose=0,
        nn_weight_decay=0.0,
        es=True,
        es_give_up_after_nepochs=10,
        num_layers=layer_grid,
        hidden_size=hidden,
        estimators=base_est,
        gpu=False,
        ensemble_method=ensemble_method,
        ensemble_addition=ensemble_addition,
        es_splitter_random_state=0,
    ).fit(x_train, y_train.reshape(len(y_train), 1), g.reshape(g.shape[0], 1, g.shape[1]))

    pred = nne.predict(x_test)
    mse = mean_squared_error(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    std = np.std((pred.flatten() - y_test) ** 2) / (len(y_test) ** (1 / 2))
    std2 = np.std(abs(pred.flatten() - y_test)) / (len(y_test) ** (1 / 2))
    print(('NNS' + str(layer_grid) + ': ', mse, ';', mae, ';',
           std, std2), file=f)

    print('Fitting NN Meta ...')

    g_train1, g_val, y_train1, y_val = train_test_split(g, y_train, test_size=1 / 10, random_state=0)

    model = Sequential()
    for j in range(i):
        model.add(Dense(hidden, input_shape=(g.shape[1],), activation='elu'))
        model.add(Dropout(0.5))
    model.add(Dense(1, activation='linear'))
    model.compile(loss="mse", optimizer='adam')
    callbacks = [EarlyStopping(monitor='val_loss', patience=10),
                 ModelCheckpoint(filepath='NNMETA_Weights', monitor='val_loss', save_best_only=True)]
    model.fit(g_train1, y_train1, epochs=1000, batch_size=128, verbose=0, callbacks=callbacks,
              validation_data=(g_val, y_val))
    model.load_weights('NNMETA_Weights')

    g_test = np.zeros(len(y_test)).reshape(len(y_test), 1)
    for est in base_est:
        g_test = np.hstack((g_test, est.predict(x_test).reshape(len(y_test), 1)))
    g_test = np.delete(g_test, 0, 1)

    pred = model.predict(g_test)
    mse = mean_squared_error(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    std = np.std((pred.flatten() - y_test) ** 2) / (len(y_test) ** (1 / 2))
    std2 = np.std(abs(pred.flatten() - y_test)) / (len(y_test) ** (1 / 2))

    print(('NNMETA' + str(layer_grid) + ': ', mse, ';', mae, ';',
           std, std2), file=f)

    f.close()
    return 'Success!'
