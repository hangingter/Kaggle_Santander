import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from torch.optim.optimizer import Optimizer

import os
print(os.listdir("data"))

# Load data
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')
print(train_df.shape, test_df.shape)

train_features = train_df.drop(['target', 'ID_code'], axis=1)
test_features = test_df.drop(['ID_code'], axis=1)
train_target = train_df['target']

print(train_features.shape, test_features.shape, train_target.shape)


#### Scaling feature #####
sc = StandardScaler()
train_features = sc.fit_transform(train_features)
test_features = sc.transform(test_features)

# Implement K-fold validation to improve results
n_splits = 4  # Number of K-fold Splits

splits = list(StratifiedKFold(n_splits=n_splits, shuffle=True).split(
    train_features, train_target))
print(splits[:3])


class CyclicLR(object):

    def __init__(self, optimizer, base_lr=1e-3, max_lr=6e-3,
                 step_size=2000, mode='triangular', gamma=1.,
                 scale_fn=None, scale_mode='cycle', last_batch_iteration=-1):

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(base_lr, list) or isinstance(base_lr, tuple):
            if len(base_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} base_lr, got {}".format(
                    len(optimizer.param_groups), len(base_lr)))
            self.base_lrs = list(base_lr)
        else:
            self.base_lrs = [base_lr] * len(optimizer.param_groups)

        if isinstance(max_lr, list) or isinstance(max_lr, tuple):
            if len(max_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} max_lr, got {}".format(
                    len(optimizer.param_groups), len(max_lr)))
            self.max_lrs = list(max_lr)
        else:
            self.max_lrs = [max_lr] * len(optimizer.param_groups)

        self.step_size = step_size

        if mode not in ['triangular', 'triangular2', 'exp_range'] \
                and scale_fn is None:
            raise ValueError('mode is invalid and scale_fn is None')

        self.mode = mode
        self.gamma = gamma

        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = self._triangular_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = self._triangular2_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = self._exp_range_scale_fn
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

        self.batch_step(last_batch_iteration + 1)
        self.last_batch_iteration = last_batch_iteration

    def batch_step(self, batch_iteration=None):
        if batch_iteration is None:
            batch_iteration = self.last_batch_iteration + 1
        self.last_batch_iteration = batch_iteration
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def _triangular_scale_fn(self, x):
        return 1.

    def _triangular2_scale_fn(self, x):
        return 1 / (2. ** (x - 1))

    def _exp_range_scale_fn(self, x):
        return self.gamma**(x)

    def get_lr(self):
        step_size = float(self.step_size)
        cycle = np.floor(1 + self.last_batch_iteration / (2 * step_size))
        x = np.abs(self.last_batch_iteration / step_size - 2 * cycle + 1)

        lrs = []
        param_lrs = zip(self.optimizer.param_groups,
                        self.base_lrs, self.max_lrs)
        for param_group, base_lr, max_lr in param_lrs:
            base_height = (max_lr - base_lr) * np.maximum(0, (1 - x))
            if self.scale_mode == 'cycle':
                lr = base_lr + base_height * self.scale_fn(cycle)
            else:
                lr = base_lr + base_height * \
                    self.scale_fn(self.last_batch_iteration)
            lrs.append(lr)
        return lrs


class Simple_NN(nn.Module):

    def __init__(self, input_dim, hidden_dim, dropout=0.5):
        super(Simple_NN, self).__init__()

        self.inpt_dim = input_dim
        self.hidden_dim = hidden_dim
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(int(hidden_dim * input_dim), int(hidden_dim * input_dim))
        self.fc3 = nn.Linear(int(hidden_dim * input_dim), int(hidden_dim * input_dim))
        self.fc4 = nn.Linear(int(hidden_dim * input_dim), 1)
        #self.fc5 = nn.Linear(int(hidden_dim/8*input_dim), 1)
        #self.bn1 = nn.BatchNorm1d(hidden_dim)
        #self.bn2 = nn.BatchNorm1d(int(hidden_dim/2))
        #self.bn3 = nn.BatchNorm1d(int(hidden_dim/4))
        #self.bn4 = nn.BatchNorm1d(int(hidden_dim/8))

    def forward(self, x):
        b_size = x.size(0)
        x = x.view(-1, 1)
        y = self.fc1(x)
        y = self.relu(y)
        y = y.view(b_size, -1)
        out = self.fc2(y)
        y = self.relu(out)
        y = y.view(b_size, -1)
        out = self.fc3(y)
        y = self.relu(out)
        y = y.view(b_size, -1)

        out = self.fc4(y)

        return out


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Hyperparameter
n_epochs = 40
batch_size = 256

# Build tensor data for torch
train_preds = np.zeros((len(train_features)))
test_preds = np.zeros((len(test_features)))

x_test = np.array(test_features)
x_test_cuda = torch.tensor(x_test, dtype=torch.float)
test = torch.utils.data.TensorDataset(x_test_cuda)
test_loader = torch.utils.data.DataLoader(
    test, batch_size=batch_size, shuffle=False)

avg_losses_f = []
avg_val_losses_f = []

# Start K-fold validation
for i, (train_idx, valid_idx) in enumerate(splits):
    x_train = np.array(train_features)
    y_train = np.array(train_target)

    x_train_fold = torch.tensor(
        x_train[train_idx.astype(int)], dtype=torch.float)
    y_train_fold = torch.tensor(y_train[train_idx.astype(
        int), np.newaxis], dtype=torch.float32)
    x_val_fold = torch.tensor(
        x_train[valid_idx.astype(int)], dtype=torch.float)
    y_val_fold = torch.tensor(y_train[valid_idx.astype(
        int), np.newaxis], dtype=torch.float32)

    # Loss function
    # loss_fn = FocalLoss(2)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Build model, initial weight and optimizer
    model = Simple_NN(200, 16)
    # model.cuda()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, weight_decay=1e-5)  # Using Adam optimizer

    ######################Cycling learning rate########################

    step_size = 2000
    base_lr, max_lr = 0.001, 0.005
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=max_lr)

    scheduler = CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr,
                         step_size=step_size, mode='exp_range',
                         gamma=0.99994)

    ###################################################################

    train = torch.utils.data.TensorDataset(x_train_fold, y_train_fold)
    valid = torch.utils.data.TensorDataset(x_val_fold, y_val_fold)

    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(
        valid, batch_size=batch_size, shuffle=False)

    print('Fold:',i+1)
    for epoch in range(n_epochs):
        start_time = time.time()
        model.train()
        avg_loss = 0.
        #avg_auc = 0.
        for i, (x_batch, y_batch) in enumerate(train_loader):
            y_pred = model(x_batch)
            ###################tuning learning rate###############
            if scheduler:
                # print('cycle_LR')
                scheduler.batch_step()

            ######################################################
            loss = loss_fn(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            avg_loss += loss.item() / len(train_loader)
            #avg_auc += round(roc_auc_score(y_batch.cpu(),y_pred.detach().cpu()),4) / len(train_loader)
        model.eval()

        valid_preds_fold = np.zeros((x_val_fold.size(0)))
        test_preds_fold = np.zeros((len(test_features)))

        avg_val_loss = 0.
        #avg_val_auc = 0.
        for i, (x_batch, y_batch) in enumerate(valid_loader):
            y_pred = model(x_batch).detach()

            #avg_val_auc += round(roc_auc_score(y_batch.cpu(),sigmoid(y_pred.cpu().numpy())[:, 0]),4) / len(valid_loader)
            avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
            valid_preds_fold[
                i * batch_size:(i + 1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]

        elapsed_time = time.time() - start_time
        print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(
            epoch + 1, n_epochs, avg_loss, avg_val_loss, elapsed_time))

    avg_losses_f.append(avg_loss)
    avg_val_losses_f.append(avg_val_loss)

    for i, (x_batch,) in enumerate(test_loader):
        y_pred = model(x_batch).detach()

        test_preds_fold[
            i * batch_size:(i + 1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]

    train_preds[valid_idx] = valid_preds_fold
    test_preds += test_preds_fold / len(splits)

auc = round(roc_auc_score(train_target, train_preds), 4)
print('All \t loss={:.4f} \t val_loss={:.4f} \t auc={:.4f}'.format(
    np.average(avg_losses_f), np.average(avg_val_losses_f), auc))


submission = pd.read_csv('data/sample_submission.csv')
submission['target'] = test_preds
filename = "{:%Y-%m-%d_%H_%M}_NN_sub.csv".format(datetime.now())
submission.to_csv(filename, index=False)
