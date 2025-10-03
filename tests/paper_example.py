import numpy as np
from matplotlib import pyplot as plt
import warnings
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.exceptions import DataConversionWarning
from nnstacking import NNS, LinearStack

warnings.filterwarnings(action="ignore", category=DataConversionWarning)


class QuadReg(BaseEstimator):
    def fit(self, x, y):
        x2 = np.square(x)
        self.model = LinearRegression().fit(x2.reshape(-1, 1), y)
        return self

    def predict(self, x):
        x2 = np.square(x)
        return self.model.predict(x2)


if __name__ == "__main__":
    lim = [-10, 10]
    x = np.arange(lim[0], lim[1], 0.001)
    e = np.random.normal(0, 3, len(x))
    y = np.zeros(len(x))
    y[x < 0] = x[x < 0] + e[x < 0]
    y[x >= 0] = np.square(x[x >= 0]) + e[x >= 0]

    # Single models
    linear = LinearRegression().fit(x.reshape(-1, 1), y)
    quad = QuadReg().fit(x.reshape(-1, 1), y)

    f, ax = plt.subplots()
    ax.plot(x, y, "ko", markersize=1, label="True regression")
    ax.plot(
        lim,
        linear.intercept_ + linear.coef_ * lim,
        "r--",
        linewidth=2,
        label="Linear fit",
    )
    ax.plot(
        x,
        quad.model.intercept_ + quad.model.coef_ * np.square(x),
        "g-.",
        linewidth=2,
        label="Quadratic fit",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("Y")
    ax.legend(loc=2)
    f.savefig("tests/output/Singles.pdf", bbox_inches="tight")

    # Multiple models
    stack = LinearStack(estimators=[LinearRegression(), QuadReg()], verbose=2).fit(
        x.reshape(-1, 1), y
    )
    pred_stack = stack.predict(x.reshape(-1, 1))

    nne = NNS(
        verbose=2,
        nn_weight_decay=0.0,
        es=True,
        es_give_up_after_nepochs=10,
        num_layers=1,
        hidden_size=100,
        estimators=[LinearRegression(), QuadReg()],
        ensemble_method="UNNS",
        ensemble_addition=True,
    ).fit(x.reshape(-1, 1), y.reshape(len(y), 1))
    nne_pred = nne.predict(x.reshape(-1, 1))

    f, ax = plt.subplots()
    ax.plot(x, y, "ko", markersize=1, label="True regression")
    ax.plot(x, pred_stack, "r--", linewidth=2, label="Breiman's linear stacking")
    ax.plot(x, nne_pred, "g-.", linewidth=2, label="NNS")
    ax.set_xlabel("x")
    ax.set_ylabel("Y")
    ax.legend(loc=2)
    f.savefig("tests/output/Stackings.pdf", bbox_inches="tight")
