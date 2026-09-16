import torch
from torch import nn

import lib


class ParamBuilder:
    _embed_dim: int | None = None
    _heads: int | None = None
    _encoder: bool = False
    _decoder: bool = False
    _resnet: bool = False
    _resnet_bottleneck: int | None = None
    _resnet_depth: int | None = None
    _input_chunk_size: int = lib.INPUT_CHUNK_SIZE
    _chunk_size: int = lib.CHUNK_SIZE
    _target_chunk_size: int = lib.TARGET_CHUNK_SIZE
    _chunk_shift: int = lib.CHUNK_SHIFT

    def __init__(self):
        pass

    def with_embed_dim(self, embed_dim: int):
        self._embed_dim = embed_dim
        return self

    def with_heads(self, heads: int):
        self._heads = heads
        return self

    def with_input_chunk_size(self, input_chunk_size: int):
        self._input_chunk_size = input_chunk_size
        return self

    def with_chunk_size(self, chunk_size: int):
        self._chunk_size = chunk_size
        return self

    def with_target_chunk_size(self, target_chunk_size: int):
        self._target_chunk_size = target_chunk_size
        return self

    def with_chunk_shift(self, chunk_shift: int):
        self._chunk_shift = chunk_shift
        return self

    def use_encoder(self, use: bool):
        self._encoder = use
        return self

    def use_decoder(self, use: bool):
        self._decoder = use
        return self

    def use_resnet(self, use: bool):
        self._resnet = use
        return self

    def with_resnet_bottleneck(self, bottleneck: int):
        self._resnet_bottleneck = bottleneck
        return self

    def with_resnet_depth(self, depth: int):
        self._resnet_depth = depth
        return self

    def build(self) -> dict:
        if self._encoder is not None and self._decoder is not None:
            assert self._heads is not None, "If you want to use an encoder and decoder, you have to specify the number of heads."

        if self._resnet is not None:
            assert self._resnet_bottleneck is not None and self._resnet_depth is not None, "If you want to use a ResNet, you have to specify the bottleneck and depth."

        return {
            'INPUT_CHUNK_SIZE': self._input_chunk_size,
            'CHUNK_SIZE': self._chunk_size,
            'TARGET_CHUNK_SIZE': self._target_chunk_size,
            'CHUNK_SHIFT': self._chunk_shift,
            'EMBED_DIM': self._embed_dim,
            'HEADS': self._heads,
            'ENCODER': self._encoder,
            'DECODER': self._decoder,
            'RESNET': self._resnet,
            'RESNET_BOTTLENECK': self._resnet_bottleneck,
            'RESNET_DEPTH': self._resnet_depth,
        }


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


class   AttantonBlock(nn.Module):
    def __init__(self, embed_dim: int, heads: int, batch_first: bool = True):
        """
        ResNet-like Block with a Multi-Head-Attention core.
        Residually adds the MHA output to the inputs and then has an FFN at the end.
        :param initial_alpha: The initial residual factor of the addition. For Deep ResNets, keep small so it remains stable.
        """
        super().__init__()

        self.mha = nn.MultiheadAttention(embed_dim, heads, batch_first=batch_first)
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
        res = res + x_att
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
    def __init__(self, params: ParamBuilder | dict):
        super().__init__()

        # make params
        if type(params) is ParamBuilder:
            self.params = params.build()
        else:
            self.params = params

        self.input_size = self.params['INPUT_CHUNK_SIZE']
        self.chunk_size = self.params['CHUNK_SIZE']
        self.target_size = self.params['TARGET_CHUNK_SIZE']

        self.heads = self.params['HEADS']

        # embedding
        self.embed_dim = self.params['EMBED_DIM']
        self.embedding = nn.Sequential(
            nn.Linear(1, self.embed_dim),
            nn.LeakyReLU()
        )

        # ResNet
        self.use_resnet = self.params['RESNET']
        if self.use_resnet:
            self.resnet = nn.Sequential(
                *[ResBlock(self.embed_dim, self.params['RESNET_BOTTLENECK']) for _ in range(self.params['RESNET_DEPTH'])]
            )

        # MHA blocks
        self.use_encoder = self.params['ENCODER']
        if self.use_encoder:
            self.encoder = AttantonBlock(self.embed_dim, self.heads, batch_first=True)
        self.use_decoder = self.params['DECODER']
        if self.use_decoder:
            self.decoder = AttantonBlock(self.embed_dim, self.heads, batch_first=True)

        # FC to output
        self.fc_to_output = nn.Sequential(
            nn.Linear(self.embed_dim, 1),  # scale down back to single bytes
            nn.LeakyReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(self.input_size, self.target_size),  # all bytes down to 8 bits
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        # embed
        x = torch.unsqueeze(x_in, 2)
        x = self.embedding(x)  # embed

        # encode
        if self.use_encoder:
            x = self.encoder(x)

        # decode
        if self.use_decoder:
            x = self.decoder(x)
            
        # resnet
        if self.use_resnet:
            x = self.resnet(x)

        # to target
        x = self.fc_to_output(x)
        x = self.sigmoid(x)

        return x

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
