import pandas as pd
import torch
import torch.utils.data as Data
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
# import torchvision


train = pd.read_csv('digits.csv')
test = pd.read_csv('test.csv')

plt.imshow(test.ix[1, :].reshape(28, -1), cmap='gray')

train_x = train.drop(['label'], axis=1, inplace=False)
train_y = train['label']

train_x = train_x/255.0
test = test/255.0

train_x_tensor = train_x.values.reshape(train_x.shape[0], 28, 28)
test_x_tensor = test.values.reshape(test.shape[0], 28, 28)

print train_x_tensor.shape
print test_x_tensor.shape
print torch.from_numpy(train_y.values).size()
print type(torch.from_numpy(train_y.values))

# transfer to tensor and load in data loader

torch_train = Data.TensorDataset(data_tensor=torch.from_numpy(train_x_tensor).float(),
                                 target_tensor=torch.from_numpy(train_y.values))

loader = Data.DataLoader(dataset=torch_train,
                         batch_size=5,
                         shuffle=True,
                         num_workers=2)

# build cnn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(32*7*7, 120)
        self.fc2 = nn.Linear(120, 80)
        self.fc3 = nn.Linear(80, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)  # compress the width and height
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # print x.size()
        x = x.view(x.size(0), -1)
        # print x.size()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)

        return output


cnn = CNN()
print cnn

optimizer = torch.optim.Adam(cnn.parameters(), lr=.001)
loss_func = nn.CrossEntropyLoss()


# cnn2

class CNN2(nn.Module):

    def __init__(self):
        super(CNN2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1),
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
        self.fc1 = nn.Linear(64*1*1, 30)
        self.fc2 = nn.Linear(30, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        output = self.fc2(x)

        return output


cnn = CNN2()
cnn.cuda()  # put in gpu

optimizer = torch.optim.Adam(cnn.parameters(), lr=.001)
loss_func = nn.CrossEntropyLoss()




# training...

for epoch in range(1):
    for step, (batch_x, batch_y) in enumerate(loader):
        # print batch_x.size()
        # print batch_y.size()
        batch_x = Variable(batch_x).cuda()  # batch_x (batch, 1, 28, 28)
        batch_y = Variable(batch_y).cuda()

        pred = cnn(batch_x.unsqueeze(dim=1))
        loss = loss_func(pred, batch_y)  # pred is float while y is long
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if step > 8000:
        #     continue

        if step % 2000 == 0:
            print 'step:{}'.format(step)
            print 'loss:{:.3f}'.format(loss.data[0])


# apply on test data

test_x_tensor = Variable(torch.from_numpy(test_x_tensor).float()).unsqueeze(1)
test_x_tensor.size()
type(test_x_tensor)

# do prediction

prediction = cnn(test_x_tensor)
prediction.size()  # [28000, 10]
type(prediction)  # variable

pred_y = torch.max(prediction, 1)[1].squeeze()

result = pd.DataFrame({'ImageId': range(1, 28001), 'Label': pred_y.data.numpy()})
result.head()

result.to_csv('digit_submit2.csv', index=False)  # 97.5% ~ 97.7%

