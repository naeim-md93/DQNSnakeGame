import os
import torch
from torch import nn


class LBRD(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout: float = 0):
        super().__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features, bias=True)
        self.bn = nn.LayerNorm(normalized_shape=out_features)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):
        # ndim = 2: (b, in_features)

        # (b, out_features) <- (b, in_features)
        x = self.fc(x)

        # (b, out_features) <- (b, in_features)
        x = self.bn(x)

        # (b, out_features) <- (b, out_features)
        x = self.relu(x)

        # (b, out_features) <- (b, out_features)
        x = self.drop(x)

        # ndim = 2: (b, out_features)

        return x


class CBRD(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dropout: float = 0):
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
        self.save_path = save_path

        self.conv1 = CBRD(in_channels=5, out_channels=32, kernel_size=5, dropout=0.3)
        self.conv2 = CBRD(in_channels=32, out_channels=64, kernel_size=5, dropout=0.3)
        self.conv3 = CBRD(in_channels=64, out_channels=128, kernel_size=5, dropout=0.3)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.conv_lstm = nn.LSTM(input_size=128, hidden_size=256, batch_first=False, bias=True)

        self.fc1 = LBRD(in_features=12, out_features=32, dropout=0.3)
        self.fc2 = LBRD(in_features=32, out_features=64, dropout=0.3)
        self.fc3 = LBRD(in_features=64, out_features=128, dropout=0.3)
        self.fc_lstm = nn.LSTM(input_size=128, hidden_size=256, batch_first=False, bias=True)

        self.pred1 = LBRD(in_features=512, out_features=64, dropout=0.3)
        self.pred2 = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, state2d, state1d):  # (b, t, 5, 14, 14), (b, t, 12)

        conv_feats = []  # [(b, 128), ...]
        for t in range(state2d.size(1)):
            c = self.conv1(state2d[:, t, :, :, :])  # (b, 32, 12, 12)
            c = self.conv2(c)  # (b, 64, 10, 10)
            c = self.conv3(c)  # (b, 128, 6, 6)
            c = self.avgpool(c)  # (b, 128, 1, 1)
            c = c.view(c.size(0), -1)  # (b, 128)
            conv_feats.append(c)
        conv_feats = torch.stack(tensors=conv_feats, dim=0)  # (t, b, 128)
        _, (conv_feats, _) = self.conv_lstm(input=conv_feats, hx=None)  # (1, b, 256)
        conv_feats = conv_feats.squeeze(0)  # (b, 256)

        fc_feats = []  # [(b, 128), ...]
        for t in range(state1d.size(1)):
            f = self.fc1(state1d[:, t, :])  # (b, 32)
            f = self.fc2(f)  # (b, 64)
            f = self.fc3(f)  # (b, 128)
            fc_feats.append(f)
        fc_feats = torch.stack(tensors=fc_feats, dim=0)  # (t, b, 128)
        _, (fc_feats, _) = self.fc_lstm(input=fc_feats, hx=None)  # (1, b, 256)
        fc_feats = fc_feats.squeeze(0)  # (b, 256)

        feats = torch.cat(tensors=(conv_feats, fc_feats), dim=-1)  # (b, 512)

        x = self.pred1(feats)  # (b, 64)

        x = self.pred2(x)  # (b, 4)

        return x

    def save(self, file_name: str = 'Model.pth'):
        torch.save(obj=self.state_dict(), f=os.path.join(self.save_path, file_name))


