from dataclasses import dataclass

import torch
from torch import nn

import lib


@dataclass
class LSTMState:
    main_hx: torch.Tensor
    main_cx: torch.Tensor

    res_hxs: list[torch.Tensor]
    res_cxs: list[torch.Tensor]

    @staticmethod
    def init(main_hidden_size: int, res_hidden_size: int, res_len_depth: int) -> LSTMState:
        main_hx = torch.zeros(1, main_hidden_size).to(lib.DEVICE)
        main_cx = torch.zeros(1, main_hidden_size).to(lib.DEVICE)

        res_hxs: list[torch.Tensor] = []
        res_cxs: list[torch.Tensor] = []

        for _ in range(res_len_depth):
            res_hxs.append(torch.zeros(1, res_hidden_size).to(lib.DEVICE))
            res_cxs.append(torch.zeros(1, res_hidden_size).to(lib.DEVICE))
            
        return LSTMState(main_hx, main_cx, res_hxs, res_cxs)

    def detach(self) -> LSTMState:
        # detach main state
        main_hx = self.main_hx.detach()
        main_cx = self.main_cx.detach()

        # detach ResLen state
        res_hxs = []
        res_cxs = []
        for hx, cx in zip(self.res_hxs, self.res_cxs):
            res_hxs.append(hx.detach())
            res_cxs.append(cx.detach())

        return LSTMState(main_hx, main_cx, res_hxs, res_cxs)


class ResLenBlock(nn.Module):
    def __init__(self, in_out_channels: int, hidden_size: int):
        super().__init__()

        self.lstm = nn.LSTM(in_out_channels, hidden_size)
        self.fc_from_hidden = nn.Linear(hidden_size, in_out_channels)

        self.relu = nn.LeakyReLU()

    def forward(self, x, hidden: tuple[torch.Tensor, torch.Tensor]):
        res_x, (hx, cx) = self.lstm(x, hidden)

        res_x = self.fc_from_hidden(res_x)

        x = x + res_x
        x = self.relu(x)
        return x, (hx, cx)


class ResLenNet(nn.Module):
    def __init__(self, in_out_channels: int, hidden_size: int, depth: int):
        super().__init__()

        self.blocks = nn.ModuleList(ResLenBlock(in_out_channels, hidden_size) for _ in range(depth))

    def forward(self, x, state: LSTMState) -> tuple[torch.Tensor, LSTMState]:
        for i, block in enumerate(self.blocks):
            x, (hx, cx) = block(x, (state.res_hxs[i], state.res_cxs[i]))
            state.res_hxs[i] = hx
            state.res_cxs[i] = cx

        return x, state


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

        self.hidden_size = 32
        self.num_layers = 1

        self.res_width = 4
        self.res_bottleneck = 24
        self.res_depth = 8

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
        )

        self.hidden_to_res = nn.Sequential(
            nn.Linear(self.hidden_size, self.res_width),
            nn.LeakyReLU(),
        )
        self.res_lstm = ResLenNet(self.res_width, self.res_bottleneck, self.res_depth)

        # self.res_net = nn.Sequential(
        #     # Hidden Size -> ResNet Width
        #     nn.Linear(self.hidden_size, self.res_width),
        #     nn.LeakyReLU(),
        #
        #     # ResNet
        #     *[ResBlock(self.res_width, self.res_bottleneck) for _ in range(self.res_depth)],
        #
        #     # # ResNet Width -> Bottleneck
        #     # nn.Linear(self.res_width, self.res_bottleneck),
        #     # nn.LeakyReLU(),
        # )

        # ResNet -> Output Size
        self.fc_to_output = nn.Linear(self.res_width, self.output_size)

        # -> [0, 1]
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, state: LSTMState) -> tuple[torch.Tensor, LSTMState]:
        x, (hx, cx) = self.lstm(x, (state.main_hx, state.main_cx))
        state.main_hx, state.main_cx = hx, cx

        # x = self.res_net(x)
        x = self.hidden_to_res(x)
        x, state = self.res_lstm(x, state)

        x = self.fc_to_output(x)
        x = self.sigmoid(x)

        return x, state

    def init_state(self) -> LSTMState:
        return LSTMState.init(self.hidden_size, self.res_bottleneck, self.res_depth)

    @staticmethod
    def init_weights(module: nn.Module):
        # initialize ResBlocks such that they start out as identifiers
        if isinstance(module, ResBlock):
            nn.init.constant_(module.linear[-1].weight, 0)
            nn.init.constant_(module.linear[-1].bias, 0)

        elif isinstance(module, ResLenBlock):
            nn.init.constant_(module.fc_from_hidden.weight, 0)
            nn.init.constant_(module.fc_from_hidden.bias, 0)

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
