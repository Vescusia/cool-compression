from dataclasses import dataclass

import torch
from torch import nn

import lib


@dataclass
class LSTMState:
    main_hx: torch.Tensor
    main_cx: torch.Tensor

    @staticmethod
    def init(main_hidden_size: int) -> LSTMState:
        main_hx = torch.zeros(1, main_hidden_size).to(lib.DEVICE)
        main_cx = torch.zeros(1, main_hidden_size).to(lib.DEVICE)
            
        return LSTMState(main_hx, main_cx)

    def detach(self) -> LSTMState:
        # detach main state
        main_hx = self.main_hx.detach()
        main_cx = self.main_cx.detach()

        return LSTMState(main_hx, main_cx)


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
        self.input_size = lib.CHUNK_SIZE  # # input also contains indexes
        self.output_size = lib.CHUNK_SIZE * 8  # output is in Bits

        # LSTM sizes
        self.use_lstm = False
        self.hidden_size = 32
        self.num_layers = 1

        # ResNet sizes
        self.res_width = 32
        self.res_bottleneck = 4
        self.res_depth = 10

        if self.use_lstm:
            self.lstm = nn.LSTM(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
            )
        else:
            self.lstm = nn.Identity()

        if self.use_lstm:
            # Hidden Size -> ResNet
            self.hidden_to_res = nn.Sequential(
                nn.Linear(self.hidden_size, self.res_width),
                nn.LeakyReLU(),
            )
        else:
            # Input Size -> ResNet
            self.hidden_to_res = nn.Sequential(
                nn.Linear(self.input_size, self.res_width),
                nn.LeakyReLU(),
            )

        self.res_net = nn.Sequential(
            *[ResBlock(self.res_width, self.res_bottleneck) for _ in range(self.res_depth)],
        )

        # ResNet -> Output Size
        self.fc_to_output = nn.Linear(self.res_width, self.output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, state: LSTMState) -> tuple[torch.Tensor, LSTMState]:
        # LSTM
        if self.use_lstm:
            x, (hx, cx) = self.lstm(x, (state.main_hx, state.main_cx))
            state.main_hx, state.main_cx = hx, cx

        x = self.hidden_to_res(x)

        # ResNet
        x = self.res_net(x)
        # x, state = self.res_lstm(x, state)

        # Output
        x = self.fc_to_output(x)
        x = self.sigmoid(x)

        return x, state

    def init_state(self) -> LSTMState:
        return LSTMState.init(self.hidden_size)

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
