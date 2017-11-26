# iceberg or ship
# band_1, band_2 - the flattened image data.
# Each band has 75x75 pixel values in the list,
# so the list has 5625 elements. Note that these values are
# not the normal non-negative integers in image files since they
# have physical meanings - these are float numbers with unit being dB.
# Band 1 and Band 2 are signals characterized by radar backscatter produced
# from different polarizations at a particular incidence angle.
# The polarizations correspond to HH (transmit/receive horizontally)
# and HV (transmit horizontally and receive vertically

# inc_angle - the incidence angle of which the image was taken.
# Note that this field has missing data marked as "na",
# and those images with "na" incidence angles are all in the training data
# to prevent leakage.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.data as Data


train = pd.read_json('~/workFiles/pyCharm/iceberg_ship/train.json')
train.shape
test = pd.read_json('~/workFiles/pyCharm/iceberg_ship/test.json')
test.columns

test_id = test.id.values

# split to train and val set

train_X, val, train_target, val_target = train_test_split(
    train.drop(['is_iceberg'], axis=1), train['is_iceberg'], test_size=.2)
# with 1283 training and 321 as validation

# reshape the 75*75 data

train_band_1 = np.array([np.array(b_1).reshape(75, 75) for b_1 in train_X['band_1']])
# (1283, 75, 75)
train_band_2 = np.array([np.array(b_2).reshape(75, 75) for b_2 in train_X['band_2']])

val_band_1 = np.array([np.array(b_1).reshape(75, 75) for b_1 in val['band_1']])
val_band_2 = np.array([np.array(b_2).reshape(75, 75) for b_2 in val['band_2']])

# stack together
# input is (batch_size, color_len, width, height)

train_stack = np.stack((train_band_1, train_band_2), axis=1)  # (1283, 2, 75, 75)
val_stack = np.stack((val_band_1, val_band_2), axis=1)

# train loader

torch_train = Data.TensorDataset(data_tensor=torch.from_numpy(train_stack).float(),
                                 target_tensor=torch.from_numpy(train_target.values))

train_loader = Data.DataLoader(dataset=torch_train,
                               batch_size=5,
                               shuffle=True,
                               num_workers=2)

torch_val = Data.TensorDataset(data_tensor=torch.from_numpy(val_stack).float(),
                               target_tensor=torch.from_numpy(val_target.values))

# build net work


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.batch = nn.BatchNorm2d(2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 16, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Linear(64*7*7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.batch(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)

        # need to return a prob, so function softmax should be applied
        return F.softmax(output)


cnn = CNN()
print cnn

optimizer = torch.optim.Adam(cnn.parameters(), lr=.001)
loss_func = nn.CrossEntropyLoss()

# load and train

val_data, val_target = Variable(torch_val.data_tensor), \
                       Variable(torch_val.target_tensor)

epoch = 2

for i in range(epoch):
    for step, (x, y) in enumerate(train_loader):
        b_x = Variable(x)
        b_y = Variable(y)

        output = cnn(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print out the loss every 50 steps
        if step % 50 == 0:
            pred = cnn(val_data)
            pred_y = torch.max(pred, 1)[1].squeeze()
            acc = np.sum(val_target.data.numpy()==pred_y.data.numpy())/float(val_target.size(0))
            print 'step', step, '|accuracy:', acc, '|loss:', loss.data[0]


# test data

test_band_1 = np.array([np.array(b_1).reshape(75, 75) for b_1 in test['band_1']])
test_band_2 = np.array([np.array(b_2).reshape(75, 75) for b_2 in test['band_2']])

test_stack = np.stack((test_band_1, test_band_2), axis=1)  # (8424, 2, 75, 75)

test_tensor = torch.from_numpy(test_stack).float()

test_loader = Data.DataLoader(test_tensor, batch_size=10, shuffle=False)


# predict
test_prob = []
for i, t_x in enumerate(test_loader):
    test_pred = cnn(Variable(t_x))
    test_pred_prob = test_pred.data.numpy()[:, 1]
    test_prob.append(test_pred_prob)

result = []
for j in range(len(test_prob)):
    for value in test_prob[j]:
        result.append(value)


naive_sub = pd.DataFrame({'id': test_id, 'is_iceberg': result})
naive_sub.to_csv('naive_sub.csv', index=False)

# np.set_printoptions(precision=2)
# np.savetxt('pred.txt', np.round(pred.data.numpy(), decimals=3))


