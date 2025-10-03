# ----------------------------------------------------------------------
# Copyright 2018 Marco Inacio <pythonpackages@marcoinacio.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data

import numpy as np
import time
import itertools
from sklearn.base import BaseEstimator
from sklearn.model_selection import ShuffleSplit
from copy import deepcopy
from torch.optim import Adamax as optimm
import attrs



from typing import Optional

@attrs.define
class NNPredict(BaseEstimator):
    """
    Regression estimation using neural networks

    Parameters
    ----------
    nn_weight_decay : object
        Mulplier for penalizaing the size of neural network weights. This penalization occurs for training only (does not affect score method nor validation of early stopping).
    num_layers : integer
        Number of hidden layers for the neural network. If set to 0, then it degenerates to linear regression.
    hidden_size : integer
        Number of nodes (neurons) of each hidden layer.
    criterion : object
        Loss criterion for the neural network, defaults to torch.nn.MSELoss().

    es : bool
        If true, then will split the training set into training and validation and calculate the validation internally on each epoch and check if the validation loss increases or not.
    es_validation_set : float
        Size of the validation set if es == True.
    es_give_up_after_nepochs : float
        Amount of epochs to try to decrease the validation loss before giving up and stoping training.
    es_splitter_random_state : float
        Random state to split the dataset into training and validation.

    nepoch : integer
        Number of epochs to run. Ignored if es == True.

    batch_initial : integer
        Initial batch size.
    batch_step_multiplier : float
        See batch_inital.
    batch_step_epoch_expon : float
        See batch_inital.
    batch_max_size : float
        See batch_inital.

    dataloader_workers : int
        Number of parallel workers for the Pytorch dataloader.

    batch_test_size : integer
        Size of the batch for validation and score methods.
        Does not affect training efficiency, usefull when there's
        little GPU memory.
    device : str
        Device to use for computation, e.g. "cpu" or "cuda". If None, will auto-detect.
    verbose : integer
        Level verbosity. Set to 0 for silent mode.
    """

    nn_weight_decay: float = attrs.field(default=0.0, validator=attrs.validators.ge(0))
    num_layers: int = attrs.field(default=10, validator=attrs.validators.ge(0))
    hidden_size: int = attrs.field(default=100, validator=attrs.validators.gt(0))
    convolutional: bool = False
    es: bool = True
    es_validation_set_size: Optional[int] = attrs.field(
        default=None, validator=attrs.validators.optional(attrs.validators.gt(0))
    )
    es_give_up_after_nepochs: int = attrs.field(
        default=50, validator=attrs.validators.gt(0)
    )
    es_splitter_random_state: int = 0
    nepoch: int = attrs.field(default=200, validator=attrs.validators.ge(0))
    batch_initial: int = attrs.field(default=300, validator=attrs.validators.ge(1))
    batch_step_multiplier: float = attrs.field(
        default=1.4, validator=attrs.validators.gt(0)
    )
    batch_step_epoch_expon: float = attrs.field(
        default=2.0, validator=attrs.validators.gt(0)
    )
    batch_max_size: int = attrs.field(default=1000, validator=attrs.validators.ge(1))
    dataloader_workers: int = attrs.field(default=1, validator=attrs.validators.ge(1))
    optim_lr: float = attrs.field(default=1e-3, validator=attrs.validators.gt(0))
    batch_test_size: int = attrs.field(default=2000, validator=attrs.validators.ge(1))
    device: str = attrs.field(
        default="cuda" if torch.cuda.is_available() else "cpu",
        validator=attrs.validators.in_(["cpu", "cuda"]),
    )
    verbose: int = 1

    # Non-init fields
    x_dim: Optional[int] = attrs.field(init=False, default=None)
    y_dim: Optional[int] = attrs.field(init=False, default=None)
    neural_net: Optional[nn.Module] = attrs.field(init=False, default=None)
    epoch_count: int = attrs.field(init=False, default=0)
    index_train: Optional[np.ndarray] = attrs.field(init=False, default=None)
    index_val: Optional[np.ndarray] = attrs.field(init=False, default=None)
    best_loss_val: float = attrs.field(init=False, default=np.inf)
    loss_history_validation: list = attrs.field(init=False, factory=list)
    loss_history_train: list = attrs.field(init=False, factory=list)
    actual_optim_lr: Optional[float] = attrs.field(init=False, default=None)

    @staticmethod
    def _numpy_to_tensor(arr: np.ndarray) -> torch.Tensor:
        arr = np.array(arr, dtype="f4")
        arr = torch.from_numpy(arr)
        return arr

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> "NNPredict":

        self.x_dim = x_train.shape[1]
        if len(y_train.shape) == 1:
            y_train = y_train[:, None]
        self.y_dim = y_train.shape[1]

        self._construct_neural_net()
        self.neural_net.to(self.device)
        self.epoch_count = 0

        return self.improve_fit(x_train, y_train, self.nepoch)

    def improve_fit(self, x_train: np.ndarray, y_train: np.ndarray, nepoch: int = 1) -> "NNPredict":
        if len(y_train.shape) == 1:
            y_train = y_train[:, None]
        criterion = nn.MSELoss()

        inputv_train = np.array(x_train, dtype="f4")
        target_train = np.array(y_train, dtype="f4")

        range_epoch = range(nepoch)
        if self.es:
            es_validation_set_size = self.es_validation_set_size
            if es_validation_set_size is None:
                es_validation_set_size = round(min(x_train.shape[0] * 0.10, 5000))
            splitter = ShuffleSplit(
                n_splits=1,
                test_size=es_validation_set_size,
                random_state=self.es_splitter_random_state,
            )
            index_train, index_val = next(iter(splitter.split(x_train, y_train)))
            self.index_train = index_train
            self.index_val = index_val

            inputv_val = inputv_train[index_val]
            target_val = target_train[index_val]

            inputv_train = inputv_train[index_train]
            target_train = target_train[index_train]

            es_tries = 0
            range_epoch = itertools.count()  # infty iterator

        batch_max_size = min(self.batch_max_size, inputv_train.shape[0])

        start_time = time.time()

        self.actual_optim_lr = self.optim_lr
        optimizer = optimm(
            self.neural_net.parameters(),
            lr=self.actual_optim_lr,
            weight_decay=self.nn_weight_decay,
        )

        train_dataset = data.TensorDataset(
            self._numpy_to_tensor(inputv_train), self._numpy_to_tensor(target_train)
        )

        for _ in range_epoch:
            batch_size = int(
                min(
                    batch_max_size,
                    self.batch_initial
                    + self.batch_step_multiplier
                    * self.epoch_count**self.batch_step_epoch_expon,
                )
            )

            train_loader = data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=self.dataloader_workers,
            )

            try:
                self.neural_net.train()
                self._one_epoch(True, train_loader, optimizer, criterion)

                if self.es:
                    self.neural_net.eval()
                    val_dataset = data.TensorDataset(
                        self._numpy_to_tensor(inputv_val),
                        self._numpy_to_tensor(target_val),
                    )
                    val_loader = data.DataLoader(
                        val_dataset,
                        batch_size=self.batch_test_size,
                        shuffle=False,
                        num_workers=self.dataloader_workers,
                    )
                    avloss = self._one_epoch(False, val_loader, optimizer, criterion)
                    self.loss_history_validation.append(avloss)
                    if avloss <= self.best_loss_val:
                        self.best_loss_val = avloss
                        best_state_dict = self.neural_net.state_dict()
                        best_state_dict = deepcopy(best_state_dict)
                        es_tries = 0
                        if self.verbose >= 2:
                            print("This is the lowest validation loss", "so far.")
                    else:
                        es_tries += 1

                    if (
                        es_tries == self.es_give_up_after_nepochs // 3
                        or es_tries == self.es_give_up_after_nepochs // 3 * 2
                    ):
                        if self.verbose >= 2:
                            print("No improvement for", es_tries, "tries")
                            print("Decreasing learning rate by half")
                            print("Restarting from best route.")
                        optimizer.param_groups[0]["lr"] *= 0.5
                        self.neural_net.load_state_dict(best_state_dict)
                    elif es_tries >= self.es_give_up_after_nepochs:
                        self.neural_net.load_state_dict(best_state_dict)
                        if self.verbose >= 1:
                            print(
                                "Validation loss did not improve after",
                                self.es_give_up_after_nepochs,
                                "tries. Stopping",
                            )
                        break

                self.epoch_count += 1
            except RuntimeError:
                if self.verbose >= 1:
                    print(
                        "Runtime error problem probably due to", "high learning rate."
                    )
                    print("Decreasing learning rate by half.")

                self._construct_neural_net()
                self.neural_net.to(self.device)
                self.actual_optim_lr /= 2
                optimizer = optimm(
                    self.neural_net.parameters(),
                    lr=self.actual_optim_lr,
                    weight_decay=self.nn_weight_decay,
                )
                self.epoch_count = 0

                continue
            except KeyboardInterrupt:
                if self.epoch_count > 0 and self.es:
                    print(
                        "Keyboard interrupt detected.",
                        "Switching weights to lowest validation loss",
                        "and exiting",
                    )
                    self.neural_net.load_state_dict(best_state_dict)
                break

        elapsed_time = time.time() - start_time
        if self.verbose >= 1:
            print("Elapsed time:", elapsed_time, flush=True)

        return self

    def _one_epoch(self, is_train: bool, data_loader: data.DataLoader, optimizer: optimm, criterion: nn.Module) -> float:
        with torch.set_grad_enabled(is_train):
            loss_vals = []
            batch_sizes = []

            for inputv_this, target_this in data_loader:
                inputv_this = inputv_this.to(self.device)
                target_this = target_this.to(self.device)

                batch_actual_size = inputv_this.shape[0]
                optimizer.zero_grad()
                output = self.neural_net(inputv_this)
                loss = criterion(output, target_this)

                np_loss = loss.data.item()
                if np.isnan(np_loss):
                    raise RuntimeError("Loss is NaN")

                loss_vals.append(np_loss)
                batch_sizes.append(batch_actual_size)

                if is_train:
                    loss.backward()
                    optimizer.step()

            avgloss = np.average(loss_vals, weights=batch_sizes)
            if self.verbose >= 2:
                print(
                    "Finished epoch",
                    self.epoch_count,
                    "with batch size",
                    data_loader.batch_size,
                    "and",
                    ("train" if is_train else "validation"),
                    "loss",
                    avgloss,
                    flush=True,
                )

            return avgloss

    def score(self, x_test: np.ndarray, y_test: np.ndarray) -> float:
        if len(y_test.shape) == 1:
            y_test = y_test[:, None]

        with torch.no_grad():
            self.neural_net.eval()
            inputv = self._numpy_to_tensor(np.ascontiguousarray(x_test))
            target = self._numpy_to_tensor(y_test)

            dataset = data.TensorDataset(inputv, target)
            data_loader = data.DataLoader(
                dataset,
                batch_size=self.batch_test_size,
                shuffle=False,
                num_workers=self.dataloader_workers,
            )

            loss_vals = []
            batch_sizes = []

            for inputv_this, target_this in data_loader:
                inputv_this = inputv_this.to(self.device)
                target_this = target_this.to(self.device)

                batch_actual_size = inputv_this.shape[0]
                output = self.neural_net(inputv_this)
                criterion = nn.MSELoss()
                loss = criterion(output, target_this)

                loss_vals.append(loss.data.item())
                batch_sizes.append(batch_actual_size)

            return -1 * np.average(loss_vals, weights=batch_sizes)

    def predict(self, x_pred: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            self.neural_net.eval()
            inputv = self._numpy_to_tensor(x_pred)
            inputv = inputv.to(self.device)

            output_pred = self.neural_net(inputv)

            return output_pred.data.cpu().numpy()

    def _construct_neural_net(self) -> None:
        class NeuralNet(nn.Module):
            def __init__(self, x_dim, y_dim, num_layers, hidden_size, convolutional):
                super(NeuralNet, self).__init__()

                output_hl_size = int(hidden_size)
                self.dropl = nn.Dropout(p=0.5)
                self.convolutional = convolutional
                next_input_l_size = x_dim

                if self.convolutional:
                    next_input_l_size = 1
                    self.nclayers = 4
                    clayers = []
                    polayers = []
                    normclayers = []
                    for i in range(self.nclayers):
                        if next_input_l_size == 1:
                            output_hl_size = 16
                        else:
                            output_hl_size = 32
                        clayers.append(
                            nn.Conv1d(
                                next_input_l_size,
                                output_hl_size,
                                kernel_size=5,
                                stride=1,
                                padding=2,
                            )
                        )
                        polayers.append(
                            nn.MaxPool1d(stride=1, kernel_size=5, padding=2)
                        )
                        normclayers.append(nn.BatchNorm1d(output_hl_size))
                        next_input_l_size = output_hl_size
                        self._initialize_layer(clayers[i])
                    self.clayers = nn.ModuleList(clayers)
                    self.polayers = nn.ModuleList(polayers)
                    self.normclayers = nn.ModuleList(normclayers)

                    faked = torch.randn(2, 1, x_dim)
                    for i in range(self.nclayers):
                        faked = polayers[i](clayers[i](faked))
                    faked = faked.view(faked.size(0), -1)
                    next_input_l_size = faked.size(1)
                    del faked

                llayers = []
                normllayers = []
                for i in range(num_layers):
                    llayers.append(nn.Linear(next_input_l_size, output_hl_size))
                    normllayers.append(nn.BatchNorm1d(output_hl_size))
                    next_input_l_size = output_hl_size
                    self._initialize_layer(llayers[i])

                self.llayers = nn.ModuleList(llayers)
                self.normllayers = nn.ModuleList(normllayers)

                self.fc_last = nn.Linear(next_input_l_size, y_dim)
                self._initialize_layer(self.fc_last)
                self.num_layers = num_layers

            def forward(self, x):
                if self.convolutional:
                    x = x[:, None]
                    for i in range(self.nclayers):
                        fc = self.clayers[i]
                        fpo = self.polayers[i]
                        fcn = self.normclayers[i]
                        x = fcn(F.elu(fc(x)))
                        x = fpo(x)
                    x = x.view(x.size(0), -1)

                for i in range(self.num_layers):
                    fc = self.llayers[i]
                    fcn = self.normllayers[i]
                    x = fcn(F.elu(fc(x)))
                    x = self.dropl(x)
                x = self.fc_last(x)

                return x

            def _initialize_layer(self, layer):
                nn.init.constant_(layer.bias, 0)
                gain = nn.init.calculate_gain("relu")
                nn.init.xavier_normal_(layer.weight, gain=gain)

        self.neural_net = NeuralNet(
            self.x_dim,
            self.y_dim,
            self.num_layers,
            self.hidden_size,
            self.convolutional,
        )

    def __getstate__(self):
        d = self.__dict__.copy()
        if "neural_net" in d:
            d["neural_net_params"] = d["neural_net"].state_dict()
            del d["neural_net"]
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        if "neural_net_params" in d:
            self._construct_neural_net()
            self.neural_net.load_state_dict(d["neural_net_params"])
            del d["neural_net_params"]
            if self.device:
                self.neural_net.to(self.device)
