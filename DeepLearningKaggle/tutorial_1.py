import torch
import torch.utils.data as Data
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd


# Input data -> data reshape -> data loader ->
# build CNN -> define optimizer&loss -> test data

train = pd.read_csv('digits.csv')
test = pd.read_csv('test.csv')

train_x = train.drop(['label'], axis=1, inplace=False)
train_y = train['label']

plt.imshow(train_x.ix[1, :].reshape(28, -1), cmap='gray')

train_x = train_x/255.0
test = test/255.0

# transform into tensor

train_tensor = torch.from_numpy(train_x.values.reshape(-1, 28, 28)).float()
test_tensor = torch.from_numpy(test.values.reshape(-1, 28, 28)).float()

# data loader

torch_train = Data.TensorDataset(data_tensor=train_tensor,
                                 target_tensor=torch.from_numpy(train_y.values))

loader = Data.DataLoader(dataset=torch_train,
                         batch_size=5,
                         shuffle=True,
                         num_workers=2)

# build net work


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # 28*28*1
            nn.Conv2d(in_channels=1,
                      out_channels=16,
                      kernel_size=3,
                      stride=1,
                      padding=1),  # (kernel - stride)/2
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 14*14*16
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1),  # (kernel - stride)/2
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 7*7*32
        )
        self.fc1 = nn.Linear(32*7*7, 120)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # stop conv
        x = x.view(x.size(0), -1)
        x = F.sigmoid(self.fc1(x))
        output = self.fc2(x)

        return output


cnn = CNN()

# define optimizer and loss function

optimizer = torch.optim.Adam(cnn.parameters(), lr=.001)
loss_func = nn.CrossEntropyLoss()

# training

for epoch in range(1):
    for step, (x, y) in enumerate(loader):
        b_x = Variable(x)
        b_y = Variable(y)

        pred = cnn(b_x.unsqueeze(dim=1))  # batch*channel*width*height
        loss = loss_func(pred, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 2000 == 0:
            print 'loss', loss.data[0]
    print 'training finished!'

# apply on test data

test_pred = cnn(Variable(test_tensor.unsqueeze(1)))

labels = torch.max(test_pred, 1)[1].data.numpy()

res = pd.DataFrame({'ImageId': range(1, 28001), 'Label': labels})

res.to_csv('test_sub.csv', index=False)

