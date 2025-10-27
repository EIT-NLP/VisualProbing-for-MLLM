import re
import math
import torch
from torch import nn
from functools import partial
import torch
import torch.nn as nn
import re
# from honeybee.projectors import CAbstractor


from functools import partial

import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import LayerNorm, LayerNorm2d
from timm.models.regnet import RegStage

# from .configuration_honeybee import HoneybeeVisualProjectorConfig


def build_pos_embeds(
    pos_emb,num_input_tokens: int, vision_hidden_size: int
):
    # pos emb
    if pos_emb:
        pos_emb1 = torch.nn.Parameter(torch.zeros(1, num_input_tokens, vision_hidden_size))
        nn.init.trunc_normal_(pos_emb1, mean=0.0, std=0.02)
    else:
        pos_emb1 = None

    return pos_emb1


def build_eos_tokens(num_eos_tokens,initializer_range,output_hidden_size: int):
    # think tokens

    if num_eos_tokens:
        eos_tokens = torch.nn.Parameter(torch.randn(1, num_eos_tokens, output_hidden_size))
        nn.init.trunc_normal_(eos_tokens, mean=0.0, std=initializer_range)
    else:
        eos_tokens = None

    return eos_tokens


def build_prenorm(prenorm,encoder_hidden_size):
    if prenorm:
        prenorm1 = LayerNorm(encoder_hidden_size)
    else:
        prenorm1 = None
    return prenorm1


def build_mlp(depth, hidden_size, output_hidden_size):
    layers = [nn.Linear(hidden_size, output_hidden_size)]
    for _ in range(1, depth):
        layers.append(nn.SiLU())
        layers.append(nn.Linear(output_hidden_size, output_hidden_size))
    return nn.Sequential(*layers)


class CAbstractor(nn.Module):
    """Base projector class"""

    def __init__(
        self,
       
        num_input_tokens: int,
        mm_hidden_size: int,
        output_hidden_size: int,
        n_queries: int,
    ):
        super().__init__()
        self.n_queries = n_queries
        self.num_input_tokens = num_input_tokens
        self.output_hidden_size = output_hidden_size
        self.mm_hidden_size = mm_hidden_size

        # think tokens
        self.eos_tokens = build_eos_tokens(num_eos_tokens=0,initializer_range=0.02,output_hidden_size=output_hidden_size)

        # pos emb
        self.pos_emb = build_pos_embeds(True, num_input_tokens, self.mm_hidden_size)
        # self.pos_emb = torch.nn.Parameter(torch.zeros(1, num_input_tokens, self.mm_hidden_size))
        # nn.init.trunc_normal_(self.pos_emb, mean=0.0, std=0.02)
        # self.pos_emb = torch.nn.Parameter(torch.zeros(1, 576, 1024))

        # print("pos_emb----------------------------------")
        # print(self.pos_emb.shape)
        self.prenorm = build_prenorm(False,self.mm_hidden_size)

        self.build_net()

    def build_net(self):
        encoder_hidden_size = self.mm_hidden_size
        hidden_size = self.mm_hidden_size
        output_hidden_size = self.output_hidden_size
        depth = 3
        mlp_depth = 2

        n_queries = self.n_queries
        assert (n_queries ** 0.5).is_integer(), "n_queries must be square number"
        hw = int(n_queries ** 0.5)

        # RegBlock = ResBlock + SE
        RegBlock = partial(
            RegStage,
            stride=1,
            dilation=1,
            act_layer=nn.SiLU,
            norm_layer=LayerNorm2d,
        )

        s1 = RegBlock(
            depth,
            encoder_hidden_size,
            hidden_size,
        )
        sampler = nn.AdaptiveAvgPool2d((hw, hw))
        s2 = RegBlock(
            depth,
            hidden_size,
            hidden_size,
        )

        self.net = nn.Sequential(s1, sampler, s2)
        self.readout = build_mlp(mlp_depth, hidden_size, output_hidden_size)
        print("------------------99999999",self.pos_emb.shape)

    def _forward(self, x):
        # x: [B, L, dim]
        # x = x[:, 1:]  # drop cls token and 2d forward
        hw = int(x.size(1) ** 0.5)
        x = rearrange(x, "b (h w) d -> b d h w", h=hw, w=hw)
        x = self.net(x)
        x = rearrange(x, "b d h w -> b (h w) d")
        x = self.readout(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, encoder_hidden_size) tensor from the visual backbone (CLIP visual encoder), including cls token.
        """
        if self.prenorm is not None:
            x = self.prenorm(x)

        if self.pos_emb is not None:
            x = x + self.pos_emb

        x = self._forward(x)  # (B, L, output_hidden_size)

        B = x.size(0)
        if self.eos_tokens is not None:
            x = torch.cat([x, self.eos_tokens.expand(B, -1, -1)], dim=1)
        return x

def build_vision_projector(config, delay_load=False, **kwargs):

    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)
    elif projector_type.startswith('mlp'):
        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config.hidden_size, config.hidden_size))
            return nn.Sequential(*modules)
    elif projector_type == 'c-abstractor':
        if '224' in config.mm_vision_tower:
            num_patches = 256  # 256 
            n_queries = 144
        elif '336' in config.mm_vision_tower:
            num_patches = 576  # 256 
            n_queries = 144
        else:
            raise NotImplementedError("This function has not been implemented yet.")
        return CAbstractor(num_input_tokens=num_patches, mm_hidden_size = config.mm_hidden_size, n_queries= n_queries, output_hidden_size=config.hidden_size)

    raise ValueError(f'Unknown projector type: {projector_type}')