from dataclasses import dataclass

import torch
from torch import nn

import lib


@dataclass
class ModelParams:
    """
    Configure Model parameters as well as architecture

    :ivar architecture: String that defines the model architecture.
        'A' -> Attention layer;
        'F' -> FFN layer;
        'R' -> ResNet layer;
        'M' -> Mamba layer -
        Example: the string 'arraf' will result in a model with an Attention "encoder",
        two layers of ResNet, then a "decoder" and final FFN.
    """
    embed_dim: int
    num_heads: int
    resnet_bottleneck: int
    architecture: str
    chunk_size = lib.CHUNK_SIZE
    input_chunk_size = lib.INPUT_CHUNK_SIZE
    target_chunk_size = lib.TARGET_CHUNK_SIZE
    chunk_shift = lib.CHUNK_SHIFT

    def build(self) -> nn.ModuleDict:
        modules = {}

        for i, char in enumerate(self.architecture):
            identifier = f"{char}-{i}"

            match char:
                case 'A':
                    modules[identifier] = AttantonBlock(self.embed_dim, self.num_heads, batch_first=True)
                case 'F':
                    modules[identifier] = nn.Sequential(
                        nn.Linear(self.embed_dim, self.embed_dim),
                        nn.LeakyReLU(),
                    )
                case 'R':
                    modules[identifier] = ResBlock(self.embed_dim, self.resnet_bottleneck)
                case 'M':
                    exit(-1)
                case c:
                    print(f"'{c}' is not recognized as model architecture block")
                    exit(-1)

        return nn.ModuleDict(modules)


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


class AttantonBlock(nn.Module):
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


class Attanton63(nn.Module):
    def __init__(self, params: ModelParams):
        super().__init__()

        # make params
        self.params = params

        self.input_size = self.params.input_chunk_size
        self.chunk_size = self.params.chunk_size
        self.target_size = self.params.target_chunk_size

        self.heads = self.params.num_heads

        # embedding
        self.embed_dim = self.params.embed_dim
        self.embedding = nn.Sequential(
            nn.Linear(1, self.embed_dim),
            nn.LeakyReLU()
        )

        # architecture
        self.architecture = self.params.architecture
        self.layers: nn.ModuleDict = self.params.build()

        # FC to output
        self.fc_to_output = nn.Sequential(
            nn.Linear(self.embed_dim, 1),  # scale down back to single bytes
            nn.LeakyReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(self.input_size, self.target_size),  # all bytes down to 8 bits
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # embed
        x = torch.unsqueeze(x, 2)
        x = self.embedding(x)  # embed

        # run model architecture
        for layer in self.layers.values():
            x = layer(x)

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
