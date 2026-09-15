from dataclasses import dataclass
from typing import Any

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


class AttantonBlock(nn.Module):
    def __init__(self, embed_dim: int, heads: int, batch_first: bool = True, initial_alpha: float = 0.001):
        """
        ResNet-like Block with a Multi-Head-Attention core.
        Residually adds the MHA output to the inputs and then has an FFN at the end.
        :param initial_alpha: The initial residual factor of the addition. For Deep ResNets, keep small so it remains stable.
        """
        super().__init__()

        self.mha = nn.MultiheadAttention(embed_dim, heads, batch_first=batch_first)
        self.alpha = nn.Parameter(torch.Tensor([initial_alpha]))  # weight of the attention output
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Linear(embed_dim, embed_dim)
        self.relu = nn.LeakyReLU()

    def forward(self, res: torch.Tensor, q: torch.Tensor | None = None, k: torch.Tensor | None = None, v: torch.Tensor | None = None) -> torch.Tensor:
        """
        If ``q, k, v`` are ``None``, ``res`` is used for self-attention and main residual part.
        :param res: Residually gets passed through after adding the output of MHA(Q, K, V)
        :type v: torch.Tensor
        :type k: torch.Tensor
        :type q: torch.Tensor
        """
        if q is None or k is None or v is None:
            assert q is None and k is None and v is None, "If any of the MHA arguments is None, all have to be none as it then is self-attention."
            q, k, v = res, res, res  # set to self-attention

        # compute Attention
        x_att = self.mha(q, k, v, need_weights=False)[0]

        # add & norm
        res = res + x_att * self.alpha
        res = self.norm(res)

        # FFN
        res = self.ffn(res)
        res = self.relu(res)
        return res


"""
Attanton Architecture:

     [SoftMax]
         |         __
      [Linear]      |
         |          | stepwise transition to output dimension
      [Linear]     _|
         |         
     [Decoder]    (self MHA)
      |--|--|
         |          __
    [MHA Block]      |
      |--|  |---|    |
         |      |    | repeated multiple times
    [MHA Block] |    |
      |--|  |----   _|
         |      |
     [Encoder]  | (self MHA)
      |--|--|   |
         |      |
    [Embedding]-| 
         |
      [input]
"""


class Attanton63(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_size = lib.INPUT_CHUNK_SIZE
        self.chunk_size = lib.CHUNK_SIZE
        self.target_size = lib.TARGET_CHUNK_SIZE

        self.heads = 16

        # embedding
        self.embed_dim = self.heads
        self.embedding = nn.Sequential(
            nn.Linear(1, self.embed_dim),
            nn.LeakyReLU()
        )

        self.resnet = nn.Sequential(
            *[ResBlock(self.embed_dim, 4) for _ in range(4)]
        )

        # MHA ResNet
        # self.mha_resnet = nn.ModuleList([AttantonBlock(self.embed_dim, self.heads, batch_first=True) for _ in range(1)])

        # MHA blocks
        # self.encoder = AttantonBlock(self.embed_dim, self.heads, initial_alpha=1., batch_first=True)
        # self.decoder = AttantonBlock(self.embed_dim, self.heads, initial_alpha=1., batch_first=True)

        # FC to output
        self.fc_to_output = nn.Sequential(
            nn.Linear(self.embed_dim, 1),  # scale down back to single bytes
            nn.LeakyReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(self.input_size, self.target_size),  # all bytes down to 8 bits
        )

        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def init_state():
        return LSTMState.init(1)

    def forward(self, x_in: torch.Tensor, state: LSTMState) -> tuple[torch.Tensor, LSTMState]:
        # embed
        x = torch.unsqueeze(x_in, 2)
        x = self.embedding(x)  # embed

        # encode
        # x = self.encoder(x)

        # MHA ResNet
        # for block in self.mha_resnet:
        #     x = block(res=x, q=x, k=x, v=x_embed)

        # decode
        # x = self.decoder(res=x, q=x, k=x, v=x_embed)
        x = self.resnet(x)

        # to target
        x = self.fc_to_output(x)
        x = self.sigmoid(x)

        return x, state

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
        else:
            return
