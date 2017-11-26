import pandas as pd
import torch
import torch.utils.data as Data
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision


batch_size = 64
time_step = 28  # height of image
input_size = 28  # width of image

train_data = torchvision.datasets.MNIST()
train_loader = torch.utils.data.DataLoader(dataset=train_data)


class RNN(nn.Module):

    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(
            input_size=input_size,  # width of image
            hidden_size=64,  # number of hidden layer
            num_layers=1,
            batch_first=True  # whether put batch in the first dim
        )
        self.out = nn.Linear(64, 10)  # output is 10 digits

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)
        output = self.out(r_out[:, -1, :])  # select the last output from
        # r_out -> (batch, time_step, input_step)
        return output


rnn = RNN()
print rnn

optimizer = torch.optim.Adam()
loss_func = nn.CrossEntropyLoss()

for step, (x, y) in enumerate(train_loader):
    train_x = Variable(x.view(-1, 28, 28))  # reshape to (batch, time_step, input_step)
    train_y = Variable(y)
    output = rnn(x)

    loss = loss_func(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


class RNN2(nn.Module):

    def __init__(self):
        super(RNN2, self).__init__()
        self.rnn = nn.RNN(
            input_size=1,
            hidden_size=32,
            num_layers=1,
            batch_first=True
        )
        self.out = nn.Linear(32, 1)

    def forward(self, x, h_state):
        r_out, h_state = self.rnn(x, h_state)
        out = []
        for time_step in range(r_out.size(1)):
            out.append(self.out(r_out[:, time_step, :]))

        return torch.stack(out, dim=1), h_state






