from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from NNE import Package
from copy import deepcopy


def master(X, y, estimators, verbose, eval_file=None, replicate=None):

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
        ensemble_addition=True
    )

    if replicate is not None:

        nne_1 = []
        nne_2 = []
        nne_3 = []
        nne_4 = []
        linear_ = []
        mean = []
        est_1 = []
        est_2 = []

        i = 0

        for train_index, test_index in KFold(n_splits=replicate).split(X):

            x_train, x_test = X[train_index], X[test_index]

            y_train, y_test = y[train_index], y[test_index]

            if verbose:

                i = i + 1

                print('Fold: ', i)

            nne_ = deepcopy(nne)

            nne_.fit(x_train, y_train.reshape(len(y_train), 1))

            nne_1.append(- nne_.score(x_test, y_test.reshape(len(y_test), 1)))

            nne_ = deepcopy(nne)

            nne_.ensemble_method = 'f_to_w'

            nne_.fit(x_train, y_train.reshape(len(y_train), 1))

            nne_2.append(- nne_.score(x_test, y_test.reshape(len(y_test), 1)))

            nne_ = deepcopy(nne)

            nne_.ensemble_addition = False

            nne_.fit(x_train, y_train.reshape(len(y_train), 1))

            nne_3.append(- nne_.score(x_test, y_test.reshape(len(y_test), 1)))

            nne_ = deepcopy(nne)

            nne_.ensemble_addition = False

            nne_.fit(x_train, y_train.reshape(len(y_train), 1))

            nne_4.append(- nne_.score(x_test, y_test.reshape(len(y_test), 1)))

            linear = nnensemble.LinearStack(estimators=estimators, verbose=1).fit(x_train, y_train)

            linear_.append(linear.score(x_test, y_test))

            linear.parameters = [1 / len(linear.parameters)] * len(linear.parameters)

            mean.append(linear.score(x_test, y_test))

            est_1.append(mean_squared_error(y_test, estimators[0].predict(x_test)))

            est_2.append(mean_squared_error(y_test, estimators[1].predict(x_test)))

            if eval_file is not None:

                f = open(eval_file, 'w')

                print(("NNE(m + phi)\'s MSE:", sum(nne_1) / replicate), file=f)

                print(nne_1)

                print(("NNE(w + phi)\'s MSE:", sum(nne_2) / replicate), file=f)

                print(("NNE(m)\'s MSE:", sum(nne_3) / replicate), file=f)

                print(("NNE(w)\'s MSE:", sum(nne_4) / replicate), file=f)

                print(("Breiman\'s MSE:", sum(linear_) / replicate), file=f)

                print(linear_)

                print(("Mean\'s MSE:", sum(mean) / replicate), file=f)

                print(mean)

                print(("i-th estimator MSE:", sum(est_1) / replicate), file=f)

                print(est_1)

                print(("i-th estimator MSE:", sum(est_2) / replicate), file=f)

                print(est_2)

                f.close()

    else:

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

        if verbose:

            print('Fitting NNE (m + phi)')

        nne_ = deepcopy(nne)

        nne_.fit(x_train, y_train.reshape(len(y_train), 1))

        if eval_file is not None:

            f = open(eval_file, 'w')

            print(("NNE(m + phi)\'s MSE:", - nne_.score(x_test, y_test.reshape(len(y_test), 1))), file=f)

        if verbose:

            print("NNE(m + phi)\'s MSE:", - nne_.score(x_test, y_test.reshape(len(y_test), 1)))

            print('Fitting NNE (w + phi)')

        nne_ = deepcopy(nne)

        nne_.ensemble_method = 'f_to_w'

        nne_.fit(x_train, y_train.reshape(len(y_train), 1))

        if eval_file is not None:

            print(("NNE(w + phi)\'s MSE:", - nne_.score(x_test, y_test.reshape(len(y_test), 1))), file=f)

        if verbose:

            print("NNE(w + phi)\'s MSE:", - nne_.score(x_test, y_test.reshape(len(y_test), 1)))

            print('Fitting NNE (m)')

        nne_ = deepcopy(nne)

        nne_.ensemble_addition = False

        nne_.fit(x_train, y_train.reshape(len(y_train), 1))

        if eval_file is not None:

            print(("NNE(m) \'s MSE:", - nne_.score(x_test, y_test.reshape(len(y_test), 1))), file=f)

        if verbose:

            print("NNE(m)\'s MSE:", - nne_.score(x_test, y_test.reshape(len(y_test), 1)))

            print('Fitting NNE (w)')

        nne_ = deepcopy(nne)

        nne_.ensemble_addition = False

        nne_.fit(x_train, y_train.reshape(len(y_train), 1))

        if eval_file is not None:

            print(("NNE(w)\'s MSE:", - nne_.score(x_test, y_test.reshape(len(y_test), 1))), file=f)

        if verbose:

            print("NNE(w)\'s MSE:", - nne_.score(x_test, y_test.reshape(len(y_test), 1)))

            print('Fitting linear stack')

        linear = nnensemble.LinearStack(estimators=estimators, verbose=1).fit(x_train, y_train)

        if eval_file is not None:

            print(('Breiman\'s MSE: ', linear.score(x_test, y_test)), file=f)

        if verbose:

            print('Means\'s MSE: ', linear.score(x_test, y_test))

            print('Calculating MSE for mean ensembler')

        print(linear.parameters)

        linear.parameters = [1 / len(linear.parameters)] * len(linear.parameters)

        if eval_file is not None:

            print(('Means\'s MSE: ', linear.score(x_test, y_test)), file=f)

        if verbose:

            print('Means\'s MSE: ', linear.score(x_test, y_test))

            print('Calculate MSE for single regressors')

        if eval_file is not None:

            for i in estimators:

                score = mean_squared_error(y_test, i.predict(x_test))

                print(('i-th estimator MSE: ', score), file=f)

            f.close()

    return 'Success!'
