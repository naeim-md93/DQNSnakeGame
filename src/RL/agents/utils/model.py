import os
import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, num_classes, save_path):
        super(Model, self).__init__()

        self.save_path = save_path

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=16, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1)

        self.fc1 = nn.Linear(in_features=1024, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):

        x = torch.relu(input=self.conv1(x))
        x = torch.relu(input=self.conv2(x))
        x = torch.relu(input=self.conv3(x))
        x = torch.relu(input=self.conv4(x))

        x = x.flatten(start_dim=1)

        x = torch.relu(input=self.fc1(x))
        x = self.fc2(x)

        return x

    def save(self, file_name='Model.pth'):
        file_path = os.path.join(self.save_path, file_name)
        torch.save(obj=self.state_dict(), f=file_path)