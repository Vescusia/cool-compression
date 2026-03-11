import torch
from torch import nn


class ByteMaster90(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear = nn.Sequential(
            nn.Linear(1024, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 8192),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.linear(x)
        return x


class LongMaster(nn.Module):
    def __init__(self):
        super().__init__()

        self.chunk_size = 128
        self.input_size = self.chunk_size * 2
        self.output_size = self.chunk_size * 8
        self.hidden_size = 64
        self.num_layers = 1

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers)

        self.linear = nn.Linear(self.hidden_size, self.output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, h, c):
        x, (h, c) = self.lstm(x, (h, c))
        x = self.linear(x)
        x = self.sigmoid(x)

        return x, h, c

    def init_state(self):
        return torch.zeros(self.num_layers, self.hidden_size), torch.zeros(self.num_layers, self.hidden_size)
