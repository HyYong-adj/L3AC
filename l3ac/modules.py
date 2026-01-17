import torch
from torch import nn
import torch.nn.functional as F
from l3ac.layers import ChannelNorm, Conv1d, Linear, GRN, CausalGRNEMA, GRN_3d, CausalGRNRMS, Snake1d
from xtract.nn.layers import Residual

from .tconv import FirstBlock, EnhanceBlock

def _left_pad_1d(x: torch.Tensor, pad: int, value: float = 0.0):
    if pad <= 0:
        return x
    return F.pad(x, (pad, 0), mode="constant", value=value)

class CausalConv1d(nn.Module):
    """
    Strict causal Conv1d wrapper:
    y[t] depends only on x[:t]
    """
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.stride = int(stride)
        self.conv = Conv1d(
            in_ch, out_ch,
            kernel_size=self.kernel_size,
            stride=self.stride,
            dilation=self.dilation,
            padding=0,
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        left = (self.kernel_size - 1) * self.dilation
        x = _left_pad_1d(x, left)
        return self.conv(x)
    

class ConvUnit(nn.Module):
    def __init__(self, dim, snake_act=True, norm=False, dilation=1, kernel_size=7, causal: bool = False):
        super().__init__()
        """
        Args:
            dim (int): Number of input channels.
        """
        if causal: #CausalGRNRMS
            self.dw_conv = CausalConv1d(dim, dim, kernel_size=kernel_size, dilation=dilation, groups=dim)
        else: #GRN_3d
            total_pad = (kernel_size - 1) * dilation
            self.dw_conv = Conv1d(dim, dim, kernel_size=kernel_size, dilation=dilation, padding=total_pad // 2,
                                  groups=dim)  # depth-wise conv

        self.norm = ChannelNorm(dim, data_format="channels_last") if norm else nn.Identity()
        self.pw_conv1 = Linear(dim, 4 * dim)  # point-wise/1x1 conv, implemented with linear layer

        if snake_act:
            self.act = Snake1d(4 * dim, data_format="channels_last")
        else:
            self.act = nn.GELU()
        if causal:
            self.grn = CausalGRNEMA(
                4 * dim,
                alpha=0.99,
                data_format="channels_last",
            )
        else:
            self.grn = GRN(4 * dim, data_format="channels_last")

        self.pw_conv2 = Linear(4 * dim, dim)

    def forward(self, x):
        x = self.dw_conv(x)
        x = x.permute(0, 2, 1)  # (N, C, T) -> (N, T, C)
        x = self.norm(x)
        x = self.pw_conv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pw_conv2(x)
        x = x.permute(0, 2, 1)  # (N, T, C) -> (N, C, T)
        return x


ResidualUnit = lambda *args, drop_rate=0., **kwargs: Residual(ConvUnit(*args, **kwargs), drop_prob=drop_rate)


class LegacyUnit(nn.Module):
    def __init__(self, dim, snake_act=True, norm=False, dilation=1, kernel_size=7, causal: bool = False):
        super().__init__()
        assert snake_act, "LegacyUnit only supports snake_act=True"
        assert norm == False, "LegacyUnit only supports norm=False"
        
        conv = CausalConv1d(dim, dim, kernel_size=kernel_size, dilation=dilation) if causal else \
               Conv1d(dim, dim, kernel_size=kernel_size, dilation=dilation, padding=((kernel_size - 1) * dilation) // 2)

        self.block = nn.Sequential(
            Snake1d(dim),
            conv,
            Snake1d(dim),
            Conv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        return self.block(x)


ResidualLegacyUnit = lambda *args, **kwargs: Residual(LegacyUnit(*args, **kwargs), drop_prob=0.)

BaseUnit = ResidualUnit


# BaseUnit = ResidualSuperConvUnit

class Encoder(nn.Module):
    def __init__(
            self,
            feature_dim: int = 512,
            strides: tuple = (2, 2, 2, 2),
            depths: tuple = (1, 1, 1, 1, 1),
            dims: tuple = (32, 64, 128, 256, 512),
            drop_path_rate: float = 0.0,
            use_norm=False,
            use_snake_act=True,
            causal: bool = False,
    ):
        super().__init__()
        # Create first convolution
        blocks = [
            # Conv1d(1, dims[0], kernel_size=7, padding=3),
            FirstBlock(dims[0], causal=causal), #causal setting
        ]

        drop_path_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i_d, o_d, stride, depth in zip(dims[:-1], dims[1:], strides, depths):
            stage = nn.Sequential(
                *[BaseUnit(dim=i_d, drop_rate=drop_path_rates[cur + j], snake_act=use_snake_act, norm=use_norm)
                  for j in range(depth)]
            )
            down_layer = nn.Sequential(
                Conv1d(i_d, o_d, kernel_size=stride, stride=stride),
                ChannelNorm(o_d, data_format="channels_first") if use_norm else nn.Identity()
            )
            blocks += [stage, down_layer]
            cur += depth

        # Create last convolution
        last_conv = CausalConv1d(dims[-1], feature_dim, kernel_size=3) if causal else \
                    Conv1d(dims[-1], feature_dim, kernel_size=3, padding=1)
        blocks += [
            nn.Sequential(
                *[BaseUnit(dim=dims[-1], 
                           drop_rate=drop_path_rates[cur + j], 
                           snake_act=use_snake_act, 
                           norm=use_norm)
                  for j in range(depths[-1])]
            ),
            last_conv,
            # Snake1d(dims[-1]),

        ]

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class LastBlock(nn.Module):
    def __init__(self, block, high_precision=True):
        super().__init__()
        self.block = block
        self.high_precision = high_precision
        if high_precision:
            self.to(dtype=torch.float64)

    def forward(self, x):
        x_dtype = x.dtype
        if self.high_precision:
            x = x.to(dtype=torch.float64)
        y = self.block(x).to(dtype=x_dtype)
        return y


class Decoder(nn.Module):
    def __init__(
            self,
            feature_dim=512,
            strides: tuple = (2, 4, 2, 4),
            depths: tuple = (1, 2, 6, 2, 2),
            dims: tuple = (512, 256, 128, 64, 32),
            drop_path_rate: float = 0.0,
            use_snake_act=True,
            use_norm=False,
            decoder_last_layer="legacy",
            causal: bool = False,
    ):
        super().__init__()
        # Create first convolution
        if causal:
            blocks = [
                CausalConv1d(feature_dim, dims[0], kernel_size=3),
            ]
        else:
            blocks = [
                Conv1d(feature_dim, dims[0], kernel_size=3, padding=1),
            ]
        

        drop_path_rates = [x.item() for x in torch.linspace(drop_path_rate, 0, sum(depths))]
        cur = 0
        for i_d, o_d, stride, depth in zip(dims[:-1], dims[1:], strides, depths):
            stage = nn.Sequential(
                *[BaseUnit(dim=i_d, drop_rate=drop_path_rates[cur + j], snake_act=use_snake_act, norm=use_norm)
                  for j in range(depth)]
            )
            upsample = nn.Upsample(scale_factor=stride, mode='nearest') if causal else \
                       nn.Upsample(scale_factor=stride, mode='linear')
            up_layer = nn.Sequential(
                Conv1d(i_d, o_d, kernel_size=1, stride=1),
                upsample,
                CausalConv1d(o_d, o_d, kernel_size=3) if causal else \
                nn.Identity(),
                ChannelNorm(o_d, data_format="channels_first") if use_norm else nn.Identity()
            )
            blocks += [stage, EnhanceBlock(i_d,causal=causal), up_layer]  # !
            cur += depth

        last_block = []

        if decoder_last_layer is None:
            last_block += [nn.Sequential(
                *[BaseUnit(dim=dims[-1], drop_rate=0., snake_act=use_snake_act, norm=use_norm)
                  for _ in range(2)])]
        elif decoder_last_layer == 'legacy':
            last_block += [nn.Sequential(
                ResidualLegacyUnit(dims[-1], dilation=1, snake_act=True, norm=False),
                ResidualLegacyUnit(dims[-1], dilation=3, snake_act=True, norm=False),
                ResidualLegacyUnit(dims[-1], dilation=9, snake_act=True, norm=False),
            )]
        elif decoder_last_layer == 'dilation':
            last_block += [nn.Sequential(
                ResidualUnit(dims[-1], dilation=1, drop_rate=0., snake_act=use_snake_act, norm=use_norm),
                ResidualUnit(dims[-1], dilation=3, drop_rate=0., snake_act=use_snake_act, norm=use_norm),
                ResidualUnit(dims[-1], dilation=9, drop_rate=0., snake_act=use_snake_act, norm=use_norm),
            )]
        else:
            raise NotImplementedError(decoder_last_layer)

        # Add final conv layer
        last_conv = CausalConv1d(dims[-1], 1, kernel_size=7) if causal else \
                    Conv1d(dims[-1], 1, kernel_size=7, padding=3)
        last_block += [
            # EnhanceBlock(dims[-1]),  # !
            Snake1d(dims[-1]),
            last_conv,
            nn.Tanh(),
        ]
        last_block = LastBlock(nn.Sequential(*last_block), high_precision=False)

        self.blocks = nn.Sequential(*blocks, last_block)

    def forward(self, x):
        return self.blocks(x)


if __name__ == '__main__':
    xi = torch.randn(1, 1, 16000)
    encode = Encoder()
    z = encode(xi)
    print(z.shape)
    decode = Decoder()
    xo = decode(z)
    print(xo.shape)
