import os
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F


class LBRD(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super().__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features, bias=True)
        # self.bn = nn.BatchNorm1d(num_features=out_features)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):
        # ndim = 2: (b, in_features)

        # (b, out_features) <- (b, in_features)
        x = self.fc(x)

        # (b, out_features) <- (b, in_features)
        # x = self.bn(x)

        # (b, out_features) <- (b, out_features)
        x = self.relu(x)

        # (b, out_features) <- (b, out_features)
        x = self.drop(x)

        # ndim = 2: (b, out_features)

        return x


class CBRD(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dropout):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True
        )
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):  # (b, in_channels, h1, w1)

        # (b, out_channels, h2, w2) <- (b, in_channels, h1, w1)
        x = self.conv(x)
        # (b, out_channels, h2, w2) <- (b, out_channels, h2, w2)
        x = self.bn(x)
        # (b, out_channels, h2, w2) <- (b, out_channels, h2, w2)
        x = self.relu(x)
        # (b, out_channels, h2, w2) <- (b, out_channels, h2, w2)
        x = self.drop(x)

        return x  # (b, out_channels, h2, w2)


class QModel(nn.Module):
    def __init__(self, num_classes, save_path):
        super(QModel, self).__init__()
        self.step = 0
        self.save_path = save_path

        self.conv3x3_1 = CBRD(in_channels=5, out_channels=16, kernel_size=3, stride=1, padding=0, dropout=0)
        self.conv3x3_2 = CBRD(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0, dropout=0)
        self.conv3x3_3 = CBRD(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0, dropout=0)

        self.conv5x5_1 = CBRD(in_channels=5, out_channels=28, kernel_size=5, stride=1, padding=1, dropout=0)
        self.conv5x5_2 = CBRD(in_channels=28, out_channels=64, kernel_size=5, stride=1, padding=0, dropout=0)

        self.conv7x7_1 = CBRD(in_channels=5, out_channels=64, kernel_size=7, stride=1, padding=0, dropout=0)

        self.fc1 = LBRD(in_features=4096, out_features=2048, dropout=0.0)
        self.fc2 = LBRD(in_features=2048, out_features=256, dropout=0.0)
        self.fc3 = nn.Linear(in_features=256, out_features=num_classes)

    def forward(self, x):  # (b, 5, 14, 14)

        if self.training:
            self.step += 1

        x1_3x3 = self.conv3x3_1(x)  # (b, 16, 12, 12)
        x2_3x3 = self.conv3x3_2(x1_3x3)  # (b, 32, 10, 10)
        x3_3x3 = self.conv3x3_3(x2_3x3)  # (b, 64, 8, 8)

        x1_5x5 = self.conv5x5_1(x)  # (b, 28, 10, 10)
        x2_5x5 = self.conv5x5_2(x1_5x5)  # (b, 64, 8, 8)

        x1_7x7 = self.conv7x7_1(x)  # (b, 64, 8, 8)

        x = x3_3x3 + x2_5x5 + x1_7x7  # (b, 64, 8, 8)

        x = x.flatten(start_dim=1)  # (b, 4096)

        x = self.fc1(x)  # (b, 2048)
        x = self.fc2(x)  # (b, 256)
        x = self.fc3(x)  # (b, 4)

        # if (self.step + 1) % 10000 == 0:
        #     for i in range(x1_3x3.size(1)):
        #         plt.imshow(x1_3x3[0, i].detach().cpu())
        #         plt.title(f'x1_3x3 {i}')
        #         plt.colorbar()
        #         plt.show()
        #
        #     for i in range(x2_3x3.size(1)):
        #         plt.imshow(x2_3x3[0, i].detach().cpu())
        #         plt.title(f'x2_3x3 {i}')
        #         plt.colorbar()
        #         plt.show()
        #
        #     for i in range(x3_3x3.size(1)):
        #         plt.imshow(x3_3x3[0, i].detach().cpu())
        #         plt.title(f'x3_3x3 {i}')
        #         plt.colorbar()
        #         plt.show()
        #
        #     for i in range(x1_5x5.size(1)):
        #         plt.imshow(x1_5x5[0, i].detach().cpu())
        #         plt.title(f'x1_5x5 {i}')
        #         plt.colorbar()
        #         plt.show()
        #
        #     for i in range(x2_5x5.size(1)):
        #         plt.imshow(x2_5x5[0, i].detach().cpu())
        #         plt.title(f'x2_5x5 {i}')
        #         plt.colorbar()
        #         plt.show()
        #
        #     for i in range(x1_7x7.size(1)):
        #         plt.imshow(x1_7x7[0, i].detach().cpu())
        #         plt.title(f'x1_7x7 {i}')
        #         plt.colorbar()
        #         plt.show()

        return x

    def save(self, file_name: str = 'Model.pth'):
        torch.save(obj=self.state_dict(), f=os.path.join(self.save_path, file_name))
