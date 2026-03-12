import torch
from torch import nn

import lib


class ByteMaster90(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear = nn.Sequential(
            nn.Linear(lib.CHUNK_SIZE * 2, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, lib.CHUNK_SIZE * 8),
            nn.Sigmoid()
        )

    def forward(self, x, _h, _c):
        x = self.linear(x)

        return x, _h, _c

    @staticmethod
    def init_state():
        return torch.rand(1), torch.rand(1)


class ResBlock(nn.Module):
    def __init__(self, in_out_channels: int, bottleneck: int):
        super().__init__()

        self.linear = nn.Sequential(
            nn.Linear(in_out_channels, bottleneck),
            nn.LeakyReLU(),
            nn.Linear(bottleneck, in_out_channels),
        )
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = x + self.linear(x)
        x = self.relu(x)
        return x


class LongMaster(nn.Module):
    def __init__(self):
        super().__init__()

        self.chunk_size = lib.CHUNK_SIZE
        self.input_size = lib.CHUNK_SIZE * 2  # input also contains indexes
        self.output_size = lib.CHUNK_SIZE * 8  # output is in Bits

        self.hidden_size = 8
        self.num_layers = 1

        self.res_width = 4
        self.res_bottleneck = 1

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=False,
        )

        self.res_net = nn.Sequential(
            # Hidden Size -> ResNet Width
            nn.Linear(self.hidden_size, self.res_width),
            nn.LeakyReLU(),

            # ResNet
            *[ResBlock(self.res_width, self.res_bottleneck) for _ in range(80)],

            # ResNet Width -> Bottleneck
            nn.Linear(self.res_width, self.res_bottleneck),
            nn.LeakyReLU(),

            # Bottleneck -> Output Size
            nn.Linear(self.res_bottleneck, self.output_size),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, h, c):
        x, (h, c) = self.lstm(x, (h, c))

        x = self.res_net(x)
        x = self.sigmoid(x)

        return x, h, c

    def init_state(self):
        return torch.zeros(self.num_layers, self.hidden_size), torch.zeros(self.num_layers, self.hidden_size)
