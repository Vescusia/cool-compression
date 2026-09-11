from dataclasses import dataclass

import torch
from sklearn.inspection import permutation_importance
from torch import nn
import torch.nn.functional as F

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

        self.input_size = lib.INPUT_CHUNK_SIZE
        self.output_size = lib.TARGET_CHUNK_SIZE

        # LSTM sizes
        self.use_lstm = True
        self.hidden_size = 16
        self.num_layers = 1

        # ResNet sizes
        self.res_width = 8
        self.res_bottleneck = 1
        self.res_depth = 4

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


class Attanton63(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_size = lib.INPUT_CHUNK_SIZE
        self.target_size = lib.TARGET_CHUNK_SIZE

        self.heads = 4

        # embedding
        self.embed_dim = self.heads * 2
        self.embedding = nn.Linear(self.input_size, self.input_size * self.embed_dim, bias=True)

        # MHA blocks
        self.encoder = nn.MultiheadAttention(self.embed_dim, self.heads, batch_first=True)
        self.decoder = nn.MultiheadAttention(self.embed_dim, self.heads, batch_first=True)

        # FC layers
        self.fc_encoder = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LeakyReLU(),
        )
        self.fc_decoder = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LeakyReLU(),
        )
        self.fc_to_output = nn.Linear(self.embed_dim * self.input_size, self.target_size)

        # norm layers
        self.encoder_norm = nn.LayerNorm(self.embed_dim)
        self.decoder_norm = nn.LayerNorm(self.embed_dim)

        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def init_state():
        return LSTMState.init(1)

    def forward(self, x_in: torch.Tensor, state: LSTMState):
        # embed
        x_embed = self.embedding(x_in)
        x_embed = x_embed.reshape(x_in.shape[0], self.input_size, self.embed_dim)

        # encode
        x = self.encoder(x_embed, x_embed, x_embed, need_weights=False)[0]  # returns single element tuple
        x = self.encoder_norm(x + x_embed)  # add & norm
        x_encoded = self.fc_encoder(x)

        # decode
        x = self.decoder(x_encoded, x_encoded, x_embed, need_weights=False)[0]  # same here
        x = self.decoder_norm(x + x_encoded)  # add & norm
        x = self.fc_decoder(x)

        # to target
        x = x.reshape(x_in.shape[0], -1)  # reshape to [batch, input_size * embed_dim]
        x = self.fc_to_output(x)  # [batch, target_size]
        x = self.sigmoid(x)

        return x, state
