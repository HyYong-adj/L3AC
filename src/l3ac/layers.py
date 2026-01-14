import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

import xtract.nn as xnn


# ! use weight_norm or not
# ! use weight_init or not
def nn_wrapper(nn_class, norm_weight=True, init_weight=True):
    def nn_builder(*args, **kwargs):
        nn_instance = nn_class(*args, **kwargs)
        if init_weight:
            nn.init.trunc_normal_(nn_instance.weight, std=.02)
            nn.init.constant_(nn_instance.bias, 0)
        if norm_weight:
            nn_instance = weight_norm(nn_instance)
        return nn_instance

    return nn_builder


Conv1d = nn_wrapper(nn.Conv1d, norm_weight=True, init_weight=True)
Linear = nn_wrapper(nn.Linear, norm_weight=True, init_weight=True)


# Scripting this brings model speed up 1.4x
@torch.jit.script
def snake(x, alpha):
    # torch.clamp_(alpha, 0.05, 50.)
    x = x + (alpha + xnn.EPS).reciprocal() * torch.sin(alpha * x).pow(2)
    return x


class Snake1d(nn.Module):
    def __init__(self, channels, data_format="channels_first"):
        super().__init__()
        if data_format == "channels_first":
            self.alpha = nn.Parameter(torch.ones(1, channels, 1))
        elif data_format == "channels_last":
            self.alpha = nn.Parameter(torch.ones(1, 1, channels))
        else:
            raise NotImplementedError

    def forward(self, x):
        return snake(x, self.alpha)


@torch.jit.script
def channel_norm(x, weight, bias, eps):
    u = x.mean(1, keepdim=True)
    s = (x - u).pow(2).mean(1, keepdim=True)
    x = (x - u) / torch.sqrt(s + eps)
    x = weight * x + bias
    return x


class ChannelNorm(nn.Module):
    """ ChannelNorm that supports two data formats: channels_last (default) or channels_first.
    Channels_last corresponds to inputs with shape (batch_size, ..., channels)
    while channels_first corresponds to inputs with shape (batch_size, channels, ...).
    """

    def __init__(self, n_channels, eps=xnn.EPS, data_format="channels_last"):
        super().__init__()
        self.n_channels = n_channels
        self.data_format = data_format
        self.weight = nn.Parameter(torch.ones(n_channels))
        self.bias = nn.Parameter(torch.zeros(n_channels))
        self.eps = torch.tensor(eps)

    def forward(self, x):

        if self.data_format == "channels_first":
            extend_dims = ((1,) * len(x.shape[2:]))
            return channel_norm(x, self.weight.view(-1, *extend_dims), self.bias.view(-1, *extend_dims), self.eps)

        elif self.data_format == "channels_last":
            return F.layer_norm(x, (self.n_channels,), self.weight, self.bias, self.eps.item())

        else:
            raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}(n_channels={self.n_channels}, {self.data_format})"

#현재 입력 x 에서 더 중요한 채널(l2 norm 기준)을 강조하는 역할
class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    Which supports two data formats: channels_last (default) or channels_first.
    Channels_last corresponds to inputs with shape (batch_size, Sequence, channels)
    while channels_first corresponds to inputs with shape (batch_size, channels, Sequence).
    """

    def __init__(self, n_channels, eps=xnn.EPS, data_format="channels_last"):
        super().__init__()
        self.n_channels = n_channels
        self.data_format = data_format
        if data_format == "channels_last":
            self.gamma = nn.Parameter(torch.zeros(1, n_channels))
            self.beta = nn.Parameter(torch.zeros(1, n_channels))
            self.channel_dim = -1
        elif data_format == "channels_first":
            self.gamma = nn.Parameter(torch.zeros(n_channels, 1))
            self.beta = nn.Parameter(torch.zeros(n_channels, 1))
            self.channel_dim = 1
        else:
            raise ValueError(f"Unsupported data_format: {data_format}")
        self.eps = torch.tensor(eps)

    def forward(self, x):
        g_x = torch.norm(x, p=2, dim=[1, 2], keepdim=True) #(b,1,1) #(b,1,1,c)
        n_x = g_x / (g_x.mean(dim=self.channel_dim, keepdim=True) + self.eps) #(b,1,1) #(b,1,1,c)
        return self.gamma * (x * n_x) + self.beta + x

    def __repr__(self):
        return f"{self.__class__.__name__}(n_channels={self.n_channels}, {self.data_format})"

class GRN_3d(nn.Module):
    def __init__(self, n_channels, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.n_channels = n_channels
        self.data_format = data_format
        self.eps = eps
        if data_format == "channels_last":
            self.gamma = nn.Parameter(torch.zeros(1, 1, n_channels))
            self.beta  = nn.Parameter(torch.zeros(1, 1, n_channels))
        elif data_format == "channels_first":
            self.gamma = nn.Parameter(torch.zeros(1, n_channels, 1))
            self.beta  = nn.Parameter(torch.zeros(1, n_channels, 1))
        else:
            raise ValueError

    def forward(self, x):
        if self.data_format == "channels_last":   # x: (B,T,C)
            gx = torch.norm(x, p=2, dim=1, keepdim=True)               # (B,1,C)
            nx = gx / (gx.mean(dim=-1, keepdim=True) + self.eps)       # (B,1,C)
        else:                                       # x: (B,C,T)
            gx = torch.norm(x, p=2, dim=2, keepdim=True)               # (B,C,1)
            nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)        # (B,C,1)
        return self.gamma * (x * nx) + self.beta + x
    
class CausalGRNRMS(nn.Module):
    def __init__(self, n_channels, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.n_channels = n_channels
        self.data_format = data_format
        self.eps = eps
        if data_format == "channels_last":
            self.gamma = nn.Parameter(torch.zeros(1, 1, n_channels))
            self.beta  = nn.Parameter(torch.zeros(1, 1, n_channels))
        else:
            self.gamma = nn.Parameter(torch.zeros(1, n_channels, 1))
            self.beta  = nn.Parameter(torch.zeros(1, n_channels, 1))

    def forward(self, x):
        if self.data_format == "channels_last":   # (B,T,C)
            s = torch.cumsum(x * x, dim=1)                           # (B,T,C) prefix sum
            t = torch.arange(1, x.size(1) + 1, device=x.device, dtype=x.dtype).view(1, -1, 1)    # (B,T,C)
            g = torch.sqrt(s / t + self.eps)  # prefix RMS  (B,T,C)
            n = g / (g.mean(dim=-1, keepdim=True) + self.eps)
        else:                                     # (B,C,T)
            s = torch.cumsum(x * x, dim=2)                           # (B,C,T)
            t = torch.arange(1, x.size(2) + 1, device=x.device, dtype=x.dtype).view(1, 1, -1)    # (B,C,T)
            g = torch.sqrt(s / t + self.eps)  # prefix RMS  (B,C,T)
            n = g / (g.mean(dim=1, keepdim=True) + self.eps)          # (B,C,T)
        return self.gamma * (x * n) + self.beta + x
    

class CausalGRNEMA(nn.Module):
    """
    Causal Global Response Normalization using EMA.
    - Supports channels_last (B, T, C) and channels_first (B, C, T)
    - Padding (zeros) is INCLUDED in EMA (no mask)
    - Intended for causal CNN outputs
    """

    def __init__(
        self,
        n_channels,
        alpha=0.99,
        eps=1e-6,
        ema_init=1e-4,
        data_format="channels_last",
        bias_correction=True,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.alpha = float(alpha)
        self.eps = eps
        self.ema_init = ema_init
        self.bias_correction = bias_correction
        self.data_format = data_format

        if data_format == "channels_last":
            # (B,T,C)
            self.gamma = nn.Parameter(torch.zeros(1, n_channels))
            self.beta  = nn.Parameter(torch.zeros(1, n_channels))
            self.channel_dim = -1
        elif data_format == "channels_first":
            # (B,C,T)
            self.gamma = nn.Parameter(torch.zeros(n_channels, 1))
            self.beta  = nn.Parameter(torch.zeros(n_channels, 1))
            self.channel_dim = 1
        else:
            raise ValueError(f"Unsupported data_format: {data_format}")

    def forward(self, x):
        """
        x:
          - channels_last : (B,T,C)
          - channels_first: (B,C,T)
        """
        if self.data_format == "channels_last":
            B, T, C = x.shape
            assert C == self.n_channels
            # ema: (B,1,C)
            ema = torch.full(
                (B, 1, C),
                self.ema_init,
                device=x.device,
                dtype=x.dtype,
            )
            k = torch.zeros(B, 1, 1, device=x.device, dtype=x.dtype)
            outs = []
            alpha_t = torch.tensor(self.alpha, device=x.device, dtype=x.dtype)

            for t in range(T):
                xt = x[:, t:t+1, :]            # (B,1,C)
                ema = self.alpha * ema + (1 - self.alpha) * (xt * xt)

                if self.bias_correction:
                    k = k + 1
                    denom = 1.0 - torch.pow(alpha_t, k)
                    ema_hat = ema / (denom + self.eps)
                else:
                    ema_hat = ema

                g = torch.sqrt(ema_hat + self.eps)                 # (B,1,C)
                n = g / (g.mean(dim=-1, keepdim=True) + self.eps)  # (B,1,C)

                yt = self.gamma * (xt * n) + self.beta + xt
                outs.append(yt)

            return torch.cat(outs, dim=1)  # (B,T,C)

        else:
            # channels_first: (B,C,T)
            B, C, T = x.shape
            assert C == self.n_channels
            # ema: (B,C,1)
            ema = torch.full(
                (B, C, 1),
                self.ema_init,
                device=x.device,
                dtype=x.dtype,
            )
            k = torch.zeros(B, 1, 1, device=x.device, dtype=x.dtype)
            outs = []
            alpha_t = torch.tensor(self.alpha, device=x.device, dtype=x.dtype)

            for t in range(T):
                xt = x[:, :, t:t+1]            # (B,C,1)
                ema = self.alpha * ema + (1 - self.alpha) * (xt * xt)

                if self.bias_correction:
                    k = k + 1
                    denom = 1.0 - torch.pow(alpha_t, k)
                    ema_hat = ema / (denom + self.eps)
                else:
                    ema_hat = ema

                g = torch.sqrt(ema_hat + self.eps)                 # (B,C,1)
                n = g / (g.mean(dim=1, keepdim=True) + self.eps)   # (B,C,1)

                yt = self.gamma * (xt * n) + self.beta + xt
                outs.append(yt)

            return torch.cat(outs, dim=2)  # (B,C,T)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"(n_channels={self.n_channels}, "
            f"alpha={self.alpha}, "
            f"data_format={self.data_format})"
        )
