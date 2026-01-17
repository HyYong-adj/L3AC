import torch
from torch import nn
import torch.nn.functional as F

from ..layers import ChannelNorm, Conv1d, Linear, GRN, Snake1d

def _left_pad_1d(x: torch.Tensor, pad: int, mode: str = "constant", value: float = 0.0):
    """
    Left-pad only for (B, C, T) tensors.
    """
    if pad <= 0:
        return x
    return F.pad(x, (pad, 0), mode=mode, value=value)

def trend_pool(x, kernel_size):
    if kernel_size > 1:
        pool_args = dict(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        return F.avg_pool1d(F.max_pool1d(x.abs(), **pool_args), **pool_args)
        # return F.avg_pool1d(F.max_pool1d(x, **pool_args), **pool_args)  # woabs
    else:
        return x

def trend_pool_causal(x, kernel_size):
    """
    Strict causal version:
    output at time t depends only on x[:, :, <= t]
    by left padding (k-1) and using padding=0 in pooling.
    """
    if kernel_size <= 1:
        return x

    k = int(kernel_size)
    # use abs envelope like original
    xa = x.abs()

    # left pad so that pooling window ending at t has enough history
    xa = _left_pad_1d(xa, k - 1, mode="constant", value=0.0)

    # stride=1, padding=0 keeps causal (window is [t-k+1, t])
    y = F.max_pool1d(xa, kernel_size=k, stride=1, padding=0)
    
    y = _left_pad_1d(y, k - 1, mode="constant", value=0.0)
    y = F.avg_pool1d(y, kernel_size=k, stride=1, padding=0)

    # length becomes T again because we padded by (k-1)
    return y

class TrendPool(nn.Module):
    def __init__(self, kernel_size=5, causal: bool = False):
        super().__init__()
        self.kernel_size = kernel_size
        self.causal = causal

    def forward(self, x):
        if self.causal:
            return trend_pool_causal(x, self.kernel_size)
        return trend_pool(x, self.kernel_size)

class CausalConv1d(nn.Module):
    """
    Wrap Conv1d(1, out, k, dilation=d, padding=0) with left padding only.
    """
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = Conv1d(in_ch, out_ch, kernel_size=kernel_size, dilation=dilation, padding=0)

    def forward(self, x):
        # receptive field size in samples: (k-1)*d
        left = (self.kernel_size - 1) * self.dilation
        x = _left_pad_1d(x, left, mode="constant", value=0.0)
        return self.conv(x)

class BaseBlock(nn.Module):
    def __init__(self, target_dim, conv_kernels=(7, 7, 7, 7), pool_kernels=(1, 3, 5, 9), dilation_rate=2,causal: bool = False):
        super().__init__()
        assert target_dim % len(pool_kernels) == 0
        each_dim = target_dim // len(pool_kernels)
        blocks = []
        for conv_kernel, pool_kernel in zip(conv_kernels, pool_kernels):
            conv_dilation = pool_kernel // dilation_rate + 1

            if causal:
                conv_layer = CausalConv1d(1, each_dim, kernel_size=conv_kernel, dilation=conv_dilation)
                tp = TrendPool(pool_kernel, causal=True)
            else:
                conv_padding = (conv_kernel - 1) * conv_dilation // 2
                conv_layer = Conv1d(1, each_dim, kernel_size=conv_kernel, dilation=conv_dilation, padding=conv_padding)
                tp = TrendPool(pool_kernel, causal=False)

            blocks.append(nn.Sequential(tp, conv_layer))
            
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        return torch.cat([block(x) for block in self.blocks], dim=1)

