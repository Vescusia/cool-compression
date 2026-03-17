import torch
from torch import nn

import lib


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

        self.hidden_size = 64
        self.num_layers = 1

        self.res_width = 64
        self.res_bottleneck = 4

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
        )

        self.res_net = nn.Sequential(
            # Hidden Size -> ResNet Width
            nn.Linear(self.hidden_size, self.res_width),
            nn.LeakyReLU(),

            # ResNet
            *[ResBlock(self.res_width, self.res_bottleneck) for _ in range(10)],

            # # ResNet Width -> Bottleneck
            # nn.Linear(self.res_width, self.res_bottleneck),
            # nn.LeakyReLU(),
        )

        # ResNet -> Output Size
        self.fc_to_output = nn.Linear(self.res_width, self.output_size)

        # -> [0, 1]
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hx, cx):
        x, (hx, cx) = self.lstm(x, (hx, cx))

        x = self.res_net(x)
        x = self.fc_to_output(x)
        x = self.sigmoid(x)

        return x, hx, cx

    def init_state(self):
        return torch.zeros(1, self.hidden_size).to(lib.DEVICE), torch.zeros(1, self.hidden_size).to(lib.DEVICE)

    @staticmethod
    def init_weights(module: nn.Module):
        # initialize ResBlocks such that they start out as identifiers
        if isinstance(module, ResBlock):
            nn.init.constant_(module.linear[-1].weight, 0)
            nn.init.constant_(module.linear[-1].bias, 0)

        elif isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.01)

        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.kaiming_uniform_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)
